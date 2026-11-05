"""Development-set pure SAE fitting, frozen-actor fidelity, and fresh state probes."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from temp_scripts.activation_baseline import (
    TARGETS, build_representations, collect_windows, coordinate_ranking, dump,
    fit_probe, metrics, predict_model, sample_weights, sha256, summarize_prediction, write_csv)
from temp_scripts.actor_model import load_actor, parameter_fingerprint
from temp_scripts.data_access import DEV_ROLES
from temp_scripts.project_paths import OUTPUT_ROOT, PROJECT_ROOT, output_path, source_path
from temp_scripts.sae_model import (
    ActivationStandardizer, SAEConfig, TopKSAE, load_checkpoint, save_checkpoint)

SAE_HOME = OUTPUT_ROOT / "pure_sae"


def infer_sae(model, normalized, batch_size=256):
    device = next(model.parameters()).device
    zs, reconstructions = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            x = torch.as_tensor(normalized[start:start + batch_size], device=device, dtype=torch.float32)
            z, reconstructed = model(x)
            zs.append(z.cpu().numpy())
            reconstructions.append(reconstructed.cpu().numpy())
    z, reconstruction = np.concatenate(zs), np.concatenate(reconstructions)
    if not np.isfinite(z).all() or not np.isfinite(reconstruction).all():
        raise ValueError("Nonfinite SAE inference")
    return z, reconstruction


def weighted_mse(prediction, reference, weights):
    return float(np.average(np.mean((np.asarray(prediction, float) - reference) ** 2, axis=1), weights=weights))


def fit_sae(activations, splits, groups, config, seed, folder, *, weighting="trajectory", device="cpu", progress=print):
    """Only activations, split identities and trajectory weights enter SAE optimization."""
    config.validate()
    train, validation = splits == "train", splits == "validation"
    if not train.any() or not validation.any() or not set(splits).issubset(DEV_ROLES):
        raise ValueError("SAE fitting needs train and validation, with no test rows")
    weights = sample_weights(groups[train], weighting)
    validation_weights = sample_weights(groups[validation], "trajectory")
    standardizer = ActivationStandardizer.fit(activations[train], weights, config.scale_floor, config.preprocessing)
    normalized = standardizer.transform(activations)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TopKSAE(activations.shape[1], config.width, config.top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    training_values = torch.as_tensor(normalized[train], device=device)
    training_weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    history, best_score, best_epoch, stale, updates = [], float("inf"), 0, 0, 0
    started = perf_counter()
    checkpoint = folder / "best_model.pt"
    for epoch in range(config.max_epochs + 1):
        if epoch:
            model.train()
            for start in range(0, len(training_values), config.batch_size):
                # Build the permutation once per epoch, and keep the final partial batch.
                if start == 0:
                    order = rng.permutation(len(training_values))
                indices = torch.as_tensor(order[start:start + config.batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                _, predicted = model(training_values[indices])
                row_loss = (predicted - training_values[indices]).square().mean(dim=1)
                loss = (row_loss * training_weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                model.project_decoder_gradient()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Nonfinite SAE gradient")
                optimizer.step()
                model.normalize_decoder()
                updates += 1
        _, reconstructed = infer_sae(model, normalized, config.batch_size)
        train_score = weighted_mse(reconstructed[train], normalized[train], weights)
        score = weighted_mse(reconstructed[validation], normalized[validation], validation_weights)
        improved = score < best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            save_checkpoint(checkpoint, model, standardizer, config, seed, epoch, score)
        elif epoch:
            stale += 1
        history.append({"epoch": epoch, "optimizer_steps": updates,
                        "train_mse_standardized": train_score, "validation_mse_standardized": score,
                        "best_validation_mse": best_score, "is_new_best": improved})
        if epoch == 0 or epoch % 10 == 0 or stale >= config.patience or epoch == config.max_epochs:
            write_csv(folder / "training_history.csv", history)
            progress(f"  epoch {epoch}/{config.max_epochs}: train={train_score:.6g}, validation={score:.6g}; best={best_epoch}")
        if stale >= config.patience:
            break
    write_csv(folder / "training_history.csv", history)
    model, standardizer, _ = load_checkpoint(checkpoint, device)
    z, normalized_reconstruction = infer_sae(model, normalized, config.batch_size)
    reconstructed = standardizer.inverse_transform(normalized_reconstruction)
    result = {"seed": seed, "best_epoch": best_epoch, "epochs_completed": epoch,
              "optimizer_steps": updates, "early_stopped": stale >= config.patience,
              "best_validation_mse_standardized": best_score,
              "best_is_initial_model": best_epoch == 0, "fitting_seconds": perf_counter() - started,
              "small_variance_coordinates": np.flatnonzero(standardizer.small_variance).tolist(),
              "parameters": sum(p.numel() for p in model.parameters()),
              "selection": "minimum validation trajectory-weighted standardized activation MSE; epoch 0 included"}
    return model, standardizer, normalized, z, reconstructed, result


def actor_outputs(actor, activations, layer, batch_size=256):
    forward = actor.from_h1 if layer == "h1" else actor.from_h2
    values = []
    with torch.no_grad():
        for start in range(0, len(activations), batch_size):
            x = torch.as_tensor(activations[start:start + batch_size], dtype=torch.float32,
                                device=next(actor.parameters()).device)
            values.append(forward(x).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite frozen actor output")
    return result


def selected_action_references(repo, data):
    mu, action = np.empty((len(data.rows), 22)), np.empty((len(data.rows), 22))
    for tid in dict.fromkeys(data.groups):
        mask = data.groups == tid
        indices = [int(data.rows[i]["state_index"]) for i in np.flatnonzero(mask)]
        trajectory = repo.load(tid)
        mu[mask] = trajectory.arrays["action_mean_raw"][indices]
        action[mask] = trajectory.arrays["actions"][indices]
    return mu, action


def check_original_actor(predicted, saved_mu, saved_action):
    error = np.abs(predicted.astype(float) - saved_mu)
    if (error > 1e-5 + 1e-4 * np.abs(saved_mu)).any():
        raise ValueError("Frozen actor does not reproduce saved means; stop before SAE fitting")
    if not np.allclose(np.clip(saved_mu, -1, 1), saved_action, atol=1e-6, rtol=0):
        raise ValueError("Saved action clipping mismatch")
    if not np.allclose(np.clip(predicted, -1, 1), saved_action, atol=1e-5, rtol=1e-4):
        raise ValueError("Frozen actor does not reproduce saved clipped actions")
    return {"passed": True, "mean_max_abs_error": float(error.max()),
            "clipped_max_abs_error": float(np.abs(np.clip(predicted, -1, 1) - saved_action).max()),
            "atol": 1e-5, "rtol": 1e-4}


def evaluation_weights(data, mask, weighting):
    weights = sample_weights(data.groups[mask], "frame" if weighting == "frame" else "trajectory")
    if weighting == "object":
        objects = np.array([r["object_id"] for r in data.rows])[mask]
        for obj in set(objects):
            local = objects == obj
            weights[local] /= weights[local].sum()
    return weights


def reconstruction_metrics(data, raw, reconstructed, normalized, standardizer, z, original_mu, new_mu):
    train = data.splits == "train"
    summaries, by_trajectory, by_object, dimensions, usage = [], [], [], [], []
    train_mean = standardizer.mean
    objects = np.array([r["object_id"] for r in data.rows])
    components = [("activation_raw", raw, reconstructed),
                  ("activation_standardized", normalized, standardizer.transform(reconstructed)),
                  ("action_mean", original_mu, new_mu),
                  ("action_clipped", np.clip(original_mu, -1, 1), np.clip(new_mu, -1, 1))]

    def compare(mask, weights):
        rows = []
        for name, reference, prediction in components:
            error = prediction[mask].astype(float) - reference[mask]
            rows.append({"quantity": name, "mae": float(np.average(np.abs(error).mean(axis=1), weights=weights)),
                         "mse": float(np.average((error ** 2).mean(axis=1), weights=weights)),
                         "max_abs": float(np.abs(error).max()),
                         "p95_row_rmse_frame_weighted": float(np.quantile(np.sqrt((error ** 2).mean(axis=1)), .95))})
        return rows

    for split in DEV_ROLES:
        mask = data.splits == split
        for weighting in ("trajectory", "frame", "object"):
            weights = evaluation_weights(data, mask, weighting)
            for result in compare(mask, weights):
                result.update(split=split, weighting=weighting, steps=int(mask.sum()),
                              trajectories=len(set(data.groups[mask])))
                if result["quantity"] == "activation_raw":
                    denominator = weighted_mse(raw[mask], np.broadcast_to(train_mean, raw[mask].shape), weights)
                    result["nmse_vs_train_mean"] = result["mse"] / denominator if denominator > 1e-15 else None
                summaries.append(result)
        weights = evaluation_weights(data, mask, "trajectory")
        for quantity, ref, pred in components:
            for dimension in range(ref.shape[1]):
                delta = pred[mask, dimension].astype(float) - ref[mask, dimension]
                dimensions.append({"split": split, "quantity": quantity, "dimension": dimension,
                                   "mae": float(np.average(np.abs(delta), weights=weights)),
                                   "rmse": float(np.sqrt(np.average(delta ** 2, weights=weights))),
                                   "max_abs": float(np.abs(delta).max())})
        for feature in range(z.shape[1]):
            active = z[mask, feature] > 0
            usage.append({"split": split, "feature": feature, "active_steps": int(active.sum()),
                          "active_trajectories": len(set(data.groups[mask][active])),
                          "trajectory_weighted_frequency": float(np.average(active, weights=weights)),
                          "frame_frequency": float(active.mean()),
                          "mean_activation": float(np.average(z[mask, feature], weights=weights)),
                          "max_activation": float(z[mask, feature].max())})
        for ids, name, destination in ((data.groups, "trajectory_id", by_trajectory),
                                       (objects, "object_id", by_object)):
            for identifier in sorted(set(ids[mask])):
                local = mask & (ids == identifier)
                local_weights = sample_weights(data.groups[local])
                for result in compare(local, local_weights):
                    destination.append({"split": split, name: str(identifier), "steps": int(local.sum()), **result})
    health = {split: {"mean_l0": float(np.average((z[data.splits == split] > 0).sum(axis=1),
                          weights=evaluation_weights(data, data.splits == split, "trajectory"))),
                      "max_l0": int((z[data.splits == split] > 0).sum(axis=1).max()),
                      "never_active_features": int((z[data.splits == split] == 0).all(axis=0).sum()),
                      "negative_reconstructed_coordinate_fraction": float((reconstructed[data.splits == split] < 0).mean())}
              for split in DEV_ROLES}
    health["train"]["all_zero_code_steps"] = int((z[train] == 0).all(axis=1).sum())
    return summaries, by_trajectory, by_object, dimensions, usage, health


def export_state_probes(folder, data, z, layer, weighting, quick=False):
    """Fresh identical-capacity linear probes; state labels never train the SAE."""
    train, validation = data.splits == "train", data.splits == "validation"
    weights = sample_weights(data.groups[train], weighting)
    validation_weights = sample_weights(data.groups[validation])
    representations, transforms = build_representations(data, weights)
    representations = {k: v for k, v in representations.items()
                       if k in ("constant", "time_spline") or k.startswith(layer + "_")}
    z_scale = StandardScaler().fit(z[train], sample_weight=weights)
    representations["z_linear"] = z_scale.transform(z)
    representations["z_time_linear"] = np.column_stack([representations["time_spline"], representations["z_linear"]])
    artifacts = {"transforms": transforms, "z_scaler": z_scale, "models": {}}
    all_metrics, trajectories, objects, ranking, selection, predictions = [], [], [], [], [], {}
    for column, target in enumerate(TARGETS):
        y, classification = data.y[:, column], target == "any_contact"
        task = dict(representations)
        coordinates = {}
        for prefix in (layer, "z"):
            rank, score = coordinate_ranking(representations[prefix + "_linear"][train], y[train], weights)
            ranking.extend({"target": target, "representation": prefix, "rank": i + 1,
                            "coordinate": int(c), "absolute_train_correlation": float(score[c])} for i, c in enumerate(rank))
            for k in (1, 2, 4):
                if k <= len(rank):
                    name = f"{prefix}_top{k}"
                    task[name], coordinates[name] = representations[prefix + "_linear"][:, rank[:k]], rank[:k].tolist()
        for name, values in task.items():
            candidates, target_scaler, single_class = fit_probe(values, y, train, validation, weights,
                      "constant" if name == "constant" else "linear", classification, quick=quick)
            scored = []
            for parameter, model, messages in candidates:
                p = predict_model(model, values[validation], classification)
                if target_scaler is not None:
                    p = target_scaler.inverse_transform(p[:, None]).ravel()
                score = metrics(y[validation], p, classification, validation_weights)["log_loss" if classification else "mae"]
                selection.append({"target": target, "model": name, "parameter": float(parameter),
                                  "validation_score": score, "warnings": " | ".join(messages)})
                scored.append((score, parameter, model, messages))
            _, parameter, model, messages = min(scored, key=lambda item: item[0])
            p = predict_model(model, values, classification)
            if target_scaler is not None:
                p = target_scaler.inverse_transform(p[:, None]).ravel()
            if not np.isfinite(p).all():
                raise ValueError("Nonfinite state probe prediction")
            key = f"{target}__{name}__seed0"
            predictions[key] = p
            artifacts["models"][key] = {"estimator": model, "target_scaler": target_scaler,
                    "coordinates": coordinates.get(name), "single_class_train": single_class,
                    "parameter": parameter, "warnings": messages}
            s, t, o = summarize_prediction(data, target, name, 0, p)
            all_metrics.extend(s)
            trajectories.extend(t)
            objects.extend(o)
    write_union(folder / "probe_metrics.csv", all_metrics)
    write_union(folder / "probe_by_trajectory.csv", trajectories)
    write_union(folder / "probe_by_object.csv", objects)
    write_csv(folder / "coordinate_ranking.csv", ranking)
    write_csv(folder / "probe_model_selection.csv", selection)
    np.savez_compressed(folder / "probe_predictions.npz", **predictions)
    joblib.dump(artifacts, folder / "fitted_probes.joblib")
    return {"models": len(artifacts["models"]),
            "models_with_warnings": sum(bool(m["warnings"]) for m in artifacts["models"].values()),
            "single_class_models": sum(m["single_class_train"] for m in artifacts["models"].values()),
            "coordinate_selection": "train-only absolute weighted correlation; heuristic, not best subset",
            "selection_metric": "validation trajectory-weighted MAE / log loss",
            "full_z_has_more_candidate_coordinates_than_raw_h": z.shape[1] > 64}


def write_union(path, rows):
    write_csv(path, rows, list(dict.fromkeys(key for row in rows for key in row)))


def snapshot_sources(folder, repo, data):
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    names = ("5_run_pure_sae.py", "sae_model.py", "sae_training.py", "sae_plots.py", "sae_viewer.py",
             "activation_baseline.py", "actor_model.py", "data_access.py", "project_paths.py", "plots.py",
             "__init__.py", "STEP5_PURE_SAE.md", "requirements.txt")
    for name in names:
        source = source_path(name)
        target = snapshot / source.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    dump(folder / "provenance.json", {
        "source_index": str(repo.root), "source_files_sha256": {name: sha256(repo.root / name)
            for name in ("trajectory_index.csv", "events.csv", "event_rules.json", "provenance.json", "segments.csv")},
        "numeric_archives_sha256": {tid: sha256(repo.root / repo.by_id[tid]["numeric_archive"])
                                    for tid in dict.fromkeys(data.groups)},
        "code_sha256": {p.relative_to(snapshot).as_posix(): sha256(p) for p in snapshot.rglob("*") if p.is_file()},
        "test_numeric_archives_loaded": 0, "policy_updated": False,
        "event_rules_manually_approved": False, "labels_used_for_sae_optimization": False})


def run_sae(repo, checkpoint, *, configurations=None, seeds=(0,), layer="h1", event="sustained_opening",
            radius=5, weighting="trajectory", device="cpu", destination=None, prepare_only=False,
            skip_probes=False, quick=False, publish=True, progress=print):
    configurations = list(configurations or [SAEConfig()])
    if layer not in ("h1", "h2") or weighting not in ("trajectory", "frame"):
        raise ValueError("Unknown layer or training weighting")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Use unique nonnegative integer seeds")
    if len({(c.width, c.top_k) for c in configurations}) != len(configurations):
        raise ValueError("Dictionary width / TopK pairs must be unique within a run")
    for config in configurations:
        config.validate()
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or SAE_HOME / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    config_record = {"layer": layer, "input_dimension": 64, "event": event, "radius": radius,
        "occurrence": "first observed event per development trajectory",
        "window_center": "trigger action max(0,event_state_index-1)", "all_selected_frames_used": True,
        "roles": list(DEV_ROLES), "weighting": weighting, "seeds": list(seeds),
        "configurations": [asdict(c) for c in configurations], "prepare_only": prepare_only,
        "skip_probes": skip_probes, "quick_check": quick, "device": device,
        "threads": torch.get_num_threads(), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "loss": "weighted mean squared error over 64 standardized activation coordinates",
        "sparsity": "TopK followed by ReLU; at most K positive entries per sample; no L1",
        "state_loss_weight": 0., "action_loss_weight": 0., "weight_decay": 0.,
        "decoder": "linear, learnable bias, unit-norm columns; projected gradients",
        "preprocessing": "train-only weighted coordinate mean with coordinate std or one global RMS std; invert before actor; mode saved per configuration",
        "checkpoint_selection": "minimum validation trajectory-weighted standardized MSE, including initialization",
        "actor_replay_tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "test_evaluation": False, "physical_simulation": False,
        "source_index": str(repo.root)}
    dump(folder / "configuration.json", config_record)
    try:
        with threadpool_limits(limits=1):
            data = collect_windows(repo, event, radius, progress=progress)
            write_csv(folder / "window_coverage.csv", data.coverage)
            train, validation = data.splits == "train", data.splits == "validation"
            if not train.any() or not validation.any():
                raise ValueError("Both train and validation need observed event windows")
            write_csv(folder / "selected_frames.csv", [{**row, **dict(zip(TARGETS, map(float, data.y[i])))}
                                                       for i, row in enumerate(data.rows)])
            raw = data.h1 if layer == "h1" else data.h2
            np.savez_compressed(folder / "selected_arrays.npz", h=raw, y=data.y,
                  trajectory_id=data.groups, split=data.splits,
                  state_index=np.array([r["state_index"] for r in data.rows]))
            expected = {repo.source_meta[tid]["checkpoint_sha256"] for tid in set(data.groups)}
            if len(expected) != 1:
                raise ValueError("Selected trajectories have different policy checkpoints")
            actor, actor_info = load_actor(checkpoint, expected.pop(), device)
            before = parameter_fingerprint(actor)
            saved_mu, saved_action = selected_action_references(repo, data)
            original_mu = actor_outputs(actor, raw, layer)
            original_check = check_original_actor(original_mu, saved_mu, saved_action)
            dump(folder / "actor_reference_check.json", {**original_check, "model": actor_info})
            (folder / "model").mkdir()
            torch.save({k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                       folder / "model" / "frozen_actor_state.pt")
            np.savez_compressed(folder / "action_references.npz", saved_mean=saved_mu, saved_action=saved_action,
                                original_mean_replayed=original_mu)
            snapshot_sources(folder, repo, data)
            counts = {split: {"eligible_trajectories": sum(r["split"] == split for r in data.coverage),
                 "included_trajectories": len(set(data.groups[data.splits == split])),
                 "selected_steps": int((data.splits == split).sum()),
                 "contact_positive_steps": int(data.y[data.splits == split, 2].sum()),
                 "strict_contact_positive_steps": sum(bool(r["strict_contact"]) for r in data.rows if r["split"] == split)}
                 for split in DEV_ROLES}
            summary = {"status": "prepared_only" if prepare_only else "development_run_completed",
                "configuration": config_record, "counts": counts, "experiments": [],
                "test_numeric_archives_loaded": 0, "validation_used_for_model_selection": True,
                "physical_simulation": False, "policy_weights_unchanged": True,
                "scope": "event-conditioned pure SAE reconstruction, action fidelity and optional fresh state probes; no intervention or causal claim"}
            if prepare_only:
                dump(folder / "summary.json", summary)
                return folder, summary
            for config in configurations:
                for seed in seeds:
                    experiment_id = f"{layer}_w{config.width}_k{config.top_k}_seed{seed}"
                    experiment = folder / experiment_id
                    experiment.mkdir()
                    progress("Experiment: " + experiment_id)
                    model, scaler, normalized, z, reconstructed, fitted = fit_sae(
                        raw, data.splits, data.groups, config, seed, experiment,
                        weighting=weighting, device=device, progress=progress)
                    new_mu = actor_outputs(actor, reconstructed, layer)
                    fidelity, trajectories, objects, dimensions, usage, health = reconstruction_metrics(
                        data, raw, reconstructed, normalized, scaler, z, original_mu, new_mu)
                    if any(h["max_l0"] > config.top_k for h in health.values()):
                        raise ValueError("TopK sparsity invariant failed")
                    np.savez_compressed(experiment / "reconstruction.npz", z=z, reconstructed_h=reconstructed,
                          reconstructed_mean=new_mu, reconstructed_action=np.clip(new_mu, -1, 1))
                    write_union(experiment / "reconstruction_metrics.csv", fidelity)
                    write_union(experiment / "reconstruction_by_trajectory.csv", trajectories)
                    write_union(experiment / "reconstruction_by_object.csv", objects)
                    write_csv(experiment / "dimension_errors.csv", dimensions)
                    write_csv(experiment / "feature_usage.csv", usage)
                    probes = None
                    if not skip_probes:
                        progress("  Fitting fresh state probes (SAE stays frozen)")
                        probes = export_state_probes(experiment, data, z, layer, weighting, quick)
                    unchanged = parameter_fingerprint(actor) == before
                    if not unchanged or any(p.requires_grad or p.grad is not None for p in actor.parameters()):
                        raise RuntimeError("Frozen actor parameters changed or received gradients")
                    result = {"experiment_id": experiment_id, "configuration": asdict(config),
                        **fitted, "health": health, "fidelity": fidelity, "probes": probes,
                        "policy_weights_unchanged": unchanged,
                        "fidelity_acceptance_threshold": None,
                        "fidelity_acceptance_note": "Report measured errors; no automatic physical/semantic acceptance threshold is assumed"}
                    dump(experiment / "summary.json", result)
                    from temp_scripts.sae_plots import export_figures
                    export_figures(experiment, folder, result)
                    summary["experiments"].append(result)
                    dump(folder / "progress.json", summary)
            summary["policy_weights_unchanged"] = parameter_fingerprint(actor) == before
        dump(folder / "summary.json", summary)
        (folder / "README.md").write_text(
            "# 步骤5：纯 TopK SAE 开发集运行\n\n"
            "每个实验目录保存最佳模型、训练曲线、特征使用、激活/动作误差、逐轨迹/物体明细及新状态探针。\n\n"
            "- configuration.json / provenance.json：事先配置、输入和代码哈希。\n"
            "- selected_frames.csv / selected_arrays.npz：全部入选动作前快照；无终止激活或测试行。\n"
            "- actor_reference_check.json：拟合前核对原激活与原动作。\n"
            "- <实验>/best_model.pt：可独立重载的 SAE 与可逆预处理，不含原策略更新。\n"
            "- <实验>/reconstruction.npz：同一行的 z、重建激活及冻结actor动作。\n"
            "- <实验>/probe_*.csv：新线性读出；validation已用于选参，不是测试成绩。\n\n"
            "SAE仅以当前一层激活拟合MSE。状态、事件确认、未来状态和结局不输入编码器。\n"
            "任一组接触不代表抓稳；事件v1仍待语义验收。动作误差不是物理仿真误差。\n"
            "负重建坐标原样接回actor并统计，不额外裁成非负。帧间相关，不能视为独立操作。\n"
            "p95_row_rmse_frame_weighted为自然帧分位数，即使该行均值采用轨迹或物体权重。\n"
            "不得据此宣称特征语义、干预贡献或闭环效果；训练种子波动不是数据置信区间。\n",
            encoding="utf-8")
        if publish and not quick:
            dump(SAE_HOME / "latest.json", {"directory": str(folder)})
        return folder, summary
    except (Exception, KeyboardInterrupt) as exc:
        dump(folder / "failure.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                      "error": str(exc), "configuration": config_record})
        raise


def latest_result():
    pointer = SAE_HOME / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("请先运行步骤5；prepare-only、quick-check及失败运行不更新正式结果指针。")
    return output_path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
