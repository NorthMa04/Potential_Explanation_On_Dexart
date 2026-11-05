"""One-way, frozen-plan evaluation on the three original holdout splits.

Nothing here fits a scaler, PCA, coordinate ranking, probe, SAE or threshold on
holdout data.  A candidate-list manifest must exist before numeric test archives
are opened.  The reported scope remains first-confirmed-event local windows.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from temp_scripts.activation_baseline import (TARGETS, WindowData, dump, metrics,
    predict_model, sample_weights, sha256, write_csv)
from temp_scripts.actor_model import FrozenActor, load_actor, parameter_fingerprint
from temp_scripts.data_access import ALL_ROLES, DEV_ROLES, DataRepository, EVENT_LABELS
from temp_scripts.project_paths import output_path
from temp_scripts.sae_training import actor_outputs, check_original_actor, infer_sae, weighted_mse
from temp_scripts.state_assisted_model import load_assisted_checkpoint

TEST_ROLES = tuple(role for role in ALL_ROLES if role not in DEV_ROLES)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_union(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    write_csv(path, rows, fields)


def collect_holdout_windows(repo, event="sustained_opening", radius=5, progress=print):
    """Copy development-window semantics exactly, retaining missing-event audit."""
    if event not in EVENT_LABELS or not isinstance(radius, int) or radius < 0:
        raise ValueError("Unknown event or negative window radius")
    records = list(repo.filter(TEST_ROLES))
    if not records or any(row["split"] not in TEST_ROLES for row in records):
        raise ValueError("Holdout evaluation requires original test-split trajectories")
    rows, h1, h2, labels, coverage, means, actions = [], [], [], [], [], [], []
    for count, meta in enumerate(records, 1):
        traj = repo.load(meta["trajectory_id"])
        candidates = [e for e in traj.events if e["event_type"] == event]
        observed = sorted((e for e in candidates if e["status"] == "observed"), key=lambda e: e["state_index"])
        audit = {"trajectory_id": traj.tid, "object_id": meta["object_id"], "split": meta["split"],
            "trajectory_success": meta["trajectory_success"], "event_type": event,
            "status": "observed" if observed else "right_censored" if any(e["status"] == "right_censored" for e in candidates) else "not_observed",
            "observed_occurrences": len(observed), "event_id": "", "event_state_index": None,
            "confirmation_state_index": None, "center_action_index": None,
            "action_start_inclusive": None, "action_stop_exclusive": None, "selected_steps": 0,
            "left_truncated": False, "right_truncated": False}
        if observed:
            e = observed[0]
            j = e["state_index"]
            center = max(0, j - 1)
            start, stop = max(0, center - radius), min(traj.n, center + radius + 1)
            audit.update(event_id=e["event_id"], event_state_index=j,
                confirmation_state_index=e["confirmation_state_index"], center_action_index=center,
                action_start_inclusive=start, action_stop_exclusive=stop, selected_steps=stop - start,
                left_truncated=center - radius < 0, right_truncated=center + radius >= traj.n)
            for t in range(start, stop):
                rows.append({"trajectory_id": traj.tid, "object_id": int(meta["object_id"]), "split": meta["split"],
                    "state_index": t, "time_s": float(t * traj.dt), "event_state_index": j,
                    "relative_state_index": t - j, "event_id": e["event_id"],
                    "strict_contact": bool(traj.state("contact")[t]),
                    "trajectory_success": meta["trajectory_success"]})
                h1.append(traj.arrays["activations"][t])
                h2.append(traj.arrays["activations_l2"][t])
                labels.append([traj.state("palm_handle_distance")[t], traj.opening[t],
                    float(np.any(traj.state("finger_palm_contact")[t]))])
                means.append(traj.arrays["action_mean_raw"][t])
                actions.append(traj.arrays["actions"][t])
        coverage.append(audit)
        if count % 25 == 0 or count == len(records):
            progress(f"Holdout windows: {count}/{len(records)} trajectories; {len(rows)} retained steps")
    data = WindowData(rows, np.asarray(h1, dtype=float).reshape(-1, 64), np.asarray(h2, dtype=float).reshape(-1, 64),
        np.asarray(labels, dtype=float).reshape(-1, 3), coverage)
    if len({(r["trajectory_id"], r["state_index"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate holdout state rows")
    if not all(np.isfinite(a).all() for a in (data.h1, data.h2, data.y)):
        raise ValueError("Nonfinite holdout activations or targets")
    return data, np.asarray(means, dtype=float).reshape(-1, 22), np.asarray(actions, dtype=float).reshape(-1, 22)


def frozen_probe_predictions(artifacts, data, z, layer):
    """Transform and predict with serialized development-fitted objects only."""
    transform = artifacts["transforms"]
    spline = transform["time_scaler"].transform(transform["time_spline"].transform(data.time))
    raw = getattr(data, layer)
    x = transform[layer]["scaler"].transform(raw)
    standardized_z = artifacts["z_scaler"].transform(z)
    representations = {"constant": np.zeros((len(data.rows), 1)), "time_spline": spline,
        layer + "_linear": x, layer + "_time_linear": np.column_stack((spline, x)),
        "z_linear": standardized_z, "z_time_linear": np.column_stack((spline, standardized_z))}
    for k in (1, 2, 4, 8):
        components = transform[layer]["pca_components"]
        if k <= len(components):
            representations[f"{layer}_pca{k}"] = x @ components[:k].T
    predictions = {}
    for key, fitted in artifacts["models"].items():
        target, name, _ = key.split("__")
        if name in representations:
            values = representations[name]
        elif name.startswith("z_top"):
            values = standardized_z[:, fitted["coordinates"]]
        elif name.startswith(layer + "_top"):
            values = x[:, fitted["coordinates"]]
        else:
            raise ValueError("Unknown frozen probe representation: " + name)
        if fitted.get("with_time", False):
            values = np.column_stack((spline, values))
        prediction = predict_model(fitted["estimator"], values, target == "any_contact")
        if fitted["target_scaler"] is not None:
            prediction = fitted["target_scaler"].inverse_transform(prediction[:, None]).ravel()
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite frozen holdout probe prediction")
        predictions[key] = prediction
    return predictions


def summarize_holdout_prediction(data, target, name, seed, prediction):
    y = data.y[:, TARGETS.index(target)]
    classification = target == "any_contact"
    objects = np.array([str(row["object_id"]) for row in data.rows])
    summary, trajectories, by_object = [], [], []
    for split in TEST_ROLES:
        mask = data.splits == split
        if not mask.any():
            summary.append({"target": target, "model": name, "seed": seed, "split": split,
                "weighting": "trajectory", "steps": 0, "trajectories": 0, "status": "no_observed_event_windows"})
            continue
        for weighting in ("trajectory", "frame", "object"):
            w = sample_weights(data.groups[mask], "frame" if weighting == "frame" else "trajectory")
            if weighting == "object":
                for obj in set(objects[mask]):
                    local = objects[mask] == obj
                    w[local] /= w[local].sum()
            summary.append({"target": target, "model": name, "seed": seed, "split": split,
                "weighting": weighting, "steps": int(mask.sum()), "trajectories": len(set(data.groups[mask])),
                **metrics(y[mask], prediction[mask], classification, w)})
        for name_key, identifiers, destination in (("trajectory_id", data.groups, trajectories), ("object_id", objects, by_object)):
            for identifier in sorted(set(identifiers[mask])):
                local = mask & (identifiers == identifier)
                destination.append({"target": target, "model": name, "seed": seed, "split": split,
                    name_key: str(identifier), "steps": int(local.sum()),
                    **metrics(y[local], prediction[local], classification, sample_weights(data.groups[local]))})
    return summary, trajectories, by_object


def holdout_fidelity(data, raw, reconstructed, scaler, z, original_mu, new_mu):
    objects = np.array([str(row["object_id"]) for row in data.rows])
    summaries, trajectories, by_object, dimensions = [], [], [], []
    components = (("activation_raw", raw, reconstructed),
        ("activation_standardized", scaler.transform(raw), scaler.transform(reconstructed)),
        ("action_mean", original_mu, new_mu), ("action_clipped", np.clip(original_mu, -1, 1), np.clip(new_mu, -1, 1)))

    def compare(mask, w):
        rows = []
        for quantity, reference, prediction in components:
            delta = prediction[mask].astype(float) - reference[mask]
            row = {"quantity": quantity, "mae": float(np.average(np.abs(delta).mean(axis=1), weights=w)),
                "mse": float(np.average((delta ** 2).mean(axis=1), weights=w)),
                "max_abs": float(np.abs(delta).max()),
                "p95_row_rmse_frame_weighted": float(np.quantile(np.sqrt((delta ** 2).mean(axis=1)), .95))}
            if quantity == "activation_raw":
                denominator = weighted_mse(reference[mask], np.broadcast_to(scaler.mean, reference[mask].shape), w)
                row["nmse_vs_train_mean"] = row["mse"] / denominator if denominator > 1e-15 else None
            rows.append(row)
        return rows

    for split in TEST_ROLES:
        mask = data.splits == split
        if not mask.any():
            summaries.append({"split": split, "weighting": "trajectory", "steps": 0, "status": "no_observed_event_windows"})
            continue
        for weighting in ("trajectory", "frame", "object"):
            w = sample_weights(data.groups[mask], "frame" if weighting == "frame" else "trajectory")
            if weighting == "object":
                for obj in set(objects[mask]):
                    local = objects[mask] == obj
                    w[local] /= w[local].sum()
            summaries.extend({"split": split, "weighting": weighting, "steps": int(mask.sum()),
                "trajectories": len(set(data.groups[mask])), **row} for row in compare(mask, w))
        for key, identifiers, destination in (("trajectory_id", data.groups, trajectories), ("object_id", objects, by_object)):
            for identifier in sorted(set(identifiers[mask])):
                local = mask & (identifiers == identifier)
                destination.extend({"split": split, key: str(identifier), "steps": int(local.sum()), **row}
                    for row in compare(local, sample_weights(data.groups[local])))
        w = sample_weights(data.groups[mask])
        for quantity, reference, prediction in components:
            for dimension in range(reference.shape[1]):
                delta = prediction[mask, dimension].astype(float) - reference[mask, dimension]
                dimensions.append({"split": split, "quantity": quantity, "dimension": dimension,
                    "mae": float(np.average(np.abs(delta), weights=w)),
                    "rmse": float(np.sqrt(np.average(delta ** 2, weights=w))), "max_abs": float(np.abs(delta).max())})
    health = {split: {"mean_l0": float(np.average((z[data.splits == split] > 0).sum(axis=1),
        weights=sample_weights(data.groups[data.splits == split]))),
        "max_l0": int((z[data.splits == split] > 0).sum(axis=1).max()),
        "never_active_features": int((z[data.splits == split] == 0).all(axis=0).sum()),
        "negative_reconstructed_coordinate_fraction": float((reconstructed[data.splits == split] < 0).mean())}
        for split in TEST_ROLES if (data.splits == split).any()}
    return summaries, trajectories, by_object, dimensions, health


def paired_trajectory_bootstrap(y, candidate, reference, groups, classification=False, *, repetitions=2000, seed=20261102):
    """Candidate-minus-reference metric CI, resampling whole trajectories.

    Each occurrence of a sampled trajectory has total weight one.  Duplicate
    trajectory draws are represented through multiplicities, never by changing
    the relative frame weights inside that trajectory.  Seeds are not pooled.
    """
    y, candidate, reference, groups = map(np.asarray, (y, candidate, reference, groups))
    if any(a.shape != y.shape for a in (candidate, reference, groups)) or not len(y):
        raise ValueError("Bootstrap needs aligned nonempty one-dimensional rows")
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    base_weights = 1. / sizes[inverse]
    def difference(w):
        if classification:
            active = w > 0
            if len(np.unique(y[active])) < 2:
                return None
            return float(average_precision_score(y[active], candidate[active], sample_weight=w[active])
                - average_precision_score(y[active], reference[active], sample_weight=w[active]))
        return float(np.average(np.abs(candidate - y) - np.abs(reference - y), weights=w))
    point = difference(base_weights)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(repetitions):
        multiplicities = rng.multinomial(len(unique), np.full(len(unique), 1. / len(unique)))
        value = difference(base_weights * multiplicities[inverse])
        if value is not None:
            results.append(value)
    lower, upper = np.quantile(results, (.025, .975)).tolist() if results else (None, None)
    return {"candidate_minus_reference": point, "ci95_low": lower, "ci95_high": upper,
        "bootstrap_repetitions": repetitions, "defined_replicates": len(results), "trajectories": len(unique),
        "bootstrap_seed": seed, "resampling_unit": "whole trajectory, equal trajectory weight",
        "training_seed_variation_pooled": False, "interval": "percentile, descriptive, no multiplicity correction"}


def _feature_review_rows(data, z, development_arrays, development_z, artifacts, layer):
    """Examples from train-fixed coordinates and thresholds; no new candidates."""
    train = development_arrays["split"] == "train"
    y_train = development_arrays["y"][train]
    z_train = development_z[train]
    rows = []
    for target_index, target in enumerate(TARGETS):
        key = f"{target}__z_top4__seed0"
        if key not in artifacts["models"]:
            continue
        coordinates = artifacts["models"][key]["coordinates"]
        for rank, coordinate in enumerate(coordinates, 1):
            train_values = z_train[:, coordinate]
            # These display thresholds are frozen from development TRAIN rows.
            threshold = float(np.quantile(train_values, .9))
            cov = float(np.mean((train_values - train_values.mean()) * (y_train[:, target_index] - y_train[:, target_index].mean())))
            direction = 1 if cov >= 0 else -1
            target_threshold = .5 if target_index == 2 else float(np.median(y_train[:, target_index]))
            expected = data.y[:, target_index] >= target_threshold if direction > 0 else data.y[:, target_index] < target_threshold
            active = (z[:, coordinate] > 0) & (z[:, coordinate] >= threshold)
            for split in TEST_ROLES:
                split_mask = data.splits == split
                for kind, mask in (("high_expected", active & expected), ("high_unexpected", active & ~expected),
                                   ("low_expected", ~active & expected), ("low_unexpected", ~active & ~expected)):
                    indices = np.flatnonzero(split_mask & mask)
                    order = indices[np.argsort(-z[indices, coordinate] if kind.startswith("high") else z[indices, coordinate], kind="stable")]
                    selected_groups = set()
                    for index in order:
                        if data.groups[index] in selected_groups:
                            continue
                        selected_groups.add(data.groups[index])
                        rows.append({**data.rows[index], "target": target, "feature": int(coordinate),
                            "train_rank": rank, "case_type": kind, "activation": float(z[index, coordinate]),
                            "target_value": float(data.y[index, target_index]), "train_activation_p90": threshold,
                            "train_target_threshold": target_threshold, "train_covariance_sign": direction,
                            "scope": "illustrative association using train-fixed display rules; not semantic truth"})
                        if len(selected_groups) == 3:
                            break
    return rows


def evaluate_holdout(study_root, experiment_ids, *, checkpoint=None, destination=None):
    """Evaluate an exact pre-frozen candidate list; return (folder, summary)."""
    root = output_path(study_root)
    manifest_path = root / "frozen_evaluation_plan.json"
    if not manifest_path.is_file():
        raise ValueError("Freeze frozen_evaluation_plan.json before opening numeric test archives")
    manifest = _read_json(manifest_path)
    experiment_ids = list(experiment_ids)
    if (not experiment_ids or len(set(experiment_ids)) != len(experiment_ids)
            or manifest.get("experiment_ids") != experiment_ids):
        raise ValueError("Requested experiment_ids must exactly match the frozen evaluation plan")
    experiments = []
    for identifier in experiment_ids:
        path = (root / identifier).resolve()
        if path.parent != root or not (path / "summary.json").is_file():
            raise ValueError("Unknown completed candidate: " + identifier)
        for filename in ("best_model.pt", "fitted_probes.joblib", "reconstruction.npz"):
            if not (path / filename).is_file():
                raise ValueError("Candidate lacks frozen artifact: " + str(path / filename))
        result = _read_json(path / "summary.json")
        payload = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
        if (result["configuration"].get("label_mode", "true") != "true"
                or payload["configuration"].get("label_mode", "true") != "true"
                or (payload.get("label_assignment") or {}).get("holdout_eligible", True) is False):
            raise ValueError("Shuffled-label negative controls are not eligible for independent test evaluation")
        experiments.append(path)
    frozen_hashes = {p.name: {name: sha256(p / name) for name in ("best_model.pt", "fitted_probes.joblib", "summary.json")}
        for p in experiments}
    if "artifact_hashes" in manifest and manifest["artifact_hashes"] != frozen_hashes:
        raise ValueError("Frozen candidate artifacts changed after plan creation")
    study = _read_json(root / "configuration.json")
    layer, event, radius = study["layer"], study["event"], study["radius"]
    if layer not in ("h1", "h2"):
        raise ValueError("Unknown study layer")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or root / ("holdout_" + stamp))
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / "frozen_evaluation_plan.json", manifest)
    dump(folder / "evaluation_inputs.json", {"study_root": str(root), "manifest_sha256": sha256(manifest_path),
        "candidate_artifact_hashes": frozen_hashes, "test_splits": TEST_ROLES,
        "no_refitting": True, "event": event, "radius": radius, "layer": layer})
    # All checks above happen before DataRepository opens any numeric test archive.
    repo = DataRepository(study["source_index"])
    data, saved_mu, saved_actions = collect_holdout_windows(repo, event, radius)
    write_csv(folder / "window_coverage.csv", data.coverage)
    if not len(data.rows):
        summary = {"status": "no_observed_event_windows", "experiments": [], "counts": {}, "physical_simulation": False}
        dump(folder / "summary.json", summary)
        return folder, summary
    raw = getattr(data, layer)
    expected = {repo.source_meta[tid]["checkpoint_sha256"] for tid in set(data.groups)}
    if len(expected) != 1:
        raise ValueError("Test trajectories use mixed actor checkpoints")
    metadata = _read_json(root / "model" / "actor_metadata.json")
    if metadata["checkpoint_sha256"] != next(iter(expected)):
        raise ValueError("Test and development actor checkpoints differ")
    if checkpoint is not None:
        actor, _ = load_actor(checkpoint, expected.pop())
    else:
        actor = FrozenActor()
        actor.load_state_dict(torch.load(root / "model" / "frozen_actor_state.pt", map_location="cpu", weights_only=True), strict=True)
        actor.requires_grad_(False).eval()
    actor_hash = parameter_fingerprint(actor)
    original_mu = actor_outputs(actor, raw, layer)
    dump(folder / "actor_reference_check.json", check_original_actor(original_mu, saved_mu, saved_actions))
    np.savez_compressed(folder / "selected_arrays.npz", h=raw, h1=data.h1, h2=data.h2, y=data.y,
        trajectory_id=data.groups, split=data.splits, state_index=np.array([row["state_index"] for row in data.rows]))
    write_csv(folder / "selected_frames.csv", [{**r, **dict(zip(TARGETS, map(float, data.y[i])))} for i, r in enumerate(data.rows)])
    counts = {split: {"eligible_trajectories": sum(r["split"] == split for r in data.coverage),
        "included_trajectories": len(set(data.groups[data.splits == split])), "selected_steps": int((data.splits == split).sum()),
        "contact_positive_steps": int(data.y[data.splits == split, 2].sum())} for split in TEST_ROLES}
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as archive:
        development_arrays = {key: archive[key] for key in archive.files}
    experiment_summaries, all_predictions, candidate_summaries = [], {}, {}
    with threadpool_limits(limits=1):
        for experiment in experiments:
            result = _read_json(experiment / "summary.json")
            candidate_summaries[experiment.name] = result
            destination_exp = folder / experiment.name
            destination_exp.mkdir()
            model, scaler, payload = load_assisted_checkpoint(experiment / "best_model.pt")
            batch_size = payload["configuration"]["sae"]["batch_size"]
            z, normalized_reconstruction = infer_sae(model, scaler.transform(raw), batch_size)
            reconstructed = scaler.inverse_transform(normalized_reconstruction)
            new_mu = actor_outputs(actor, reconstructed, layer)
            fidelity, trajectories, objects, dimensions, health = holdout_fidelity(data, raw, reconstructed, scaler, z, original_mu, new_mu)
            if any(row["max_l0"] > model.top_k for row in health.values()):
                raise ValueError("Holdout TopK invariant failed")
            np.savez_compressed(destination_exp / "reconstruction.npz", z=z, reconstructed_h=reconstructed,
                original_mean=original_mu, reconstructed_mean=new_mu, reconstructed_action=np.clip(new_mu, -1, 1))
            for name, rows in (("reconstruction_metrics", fidelity), ("reconstruction_by_trajectory", trajectories),
                               ("reconstruction_by_object", objects), ("dimension_errors", dimensions)):
                _write_union(destination_exp / (name + ".csv"), rows)
            artifacts = joblib.load(experiment / "fitted_probes.joblib")
            predictions = frozen_probe_predictions(artifacts, data, z, layer)
            np.savez_compressed(destination_exp / "probe_predictions.npz", **predictions)
            all_predictions[experiment.name] = predictions
            state_metrics, state_trajectories, state_objects, per_state = [], [], [], []
            for key, prediction in predictions.items():
                target, name, _ = key.split("__")
                s, t, o = summarize_holdout_prediction(data, target, name, result["seed"], prediction)
                state_metrics.extend(s)
                state_trajectories.extend(t)
                state_objects.extend(o)
                per_state.extend({"trajectory_id": r["trajectory_id"], "state_index": r["state_index"],
                    "split": r["split"], "target": target, "model": name,
                    "truth": float(data.y[i, TARGETS.index(target)]), "prediction": float(prediction[i])}
                    for i, r in enumerate(data.rows))
            for name, rows in (("probe_metrics", state_metrics), ("probe_by_trajectory", state_trajectories),
                               ("probe_by_object", state_objects), ("state_predictions", per_state)):
                _write_union(destination_exp / (name + ".csv"), rows)
            with np.load(experiment / "reconstruction.npz", allow_pickle=False) as archive:
                development_z = archive["z"]
            review = _feature_review_rows(data, z, development_arrays, development_z, artifacts, layer)
            _write_union(destination_exp / "feature_examples.csv", review)
            item = {"experiment_id": experiment.name, "seed": result["seed"], "configuration": result["configuration"],
                "counts": counts, "fidelity": fidelity, "health": health, "probes": len(predictions),
                "probe_refitting": False, "candidate_selection": "frozen development plan", "physical_simulation": False}
            dump(destination_exp / "summary.json", item)
            experiment_summaries.append(item)
            print("Holdout completed:", experiment.name, flush=True)
        bootstrap_rows = []
        for identifier, predictions in all_predictions.items():
            candidate = candidate_summaries[identifier]
            pure_ids = [name for name, result in candidate_summaries.items() if result["seed"] == candidate["seed"]
                and result["configuration"]["state_weight"] == 0 and result["configuration"]["action_weight"] == 0
                and result["configuration"]["sae"] == candidate["configuration"]["sae"]]
            for target_index, target in enumerate(TARGETS):
                key = f"{target}__z_top4__seed0"
                references = [("raw_top4", predictions[f"{target}__{layer}_top4__seed0"])]
                references.extend((pure_id, all_predictions[pure_id][key]) for pure_id in pure_ids if pure_id != identifier)
                for reference_name, reference in references:
                    for split in TEST_ROLES:
                        mask = data.splits == split
                        if not mask.any():
                            continue
                        row = paired_trajectory_bootstrap(data.y[mask, target_index], predictions[key][mask],
                            reference[mask], data.groups[mask], target == "any_contact")
                        bootstrap_rows.append({"experiment_id": identifier, "seed": candidate["seed"], "reference": reference_name,
                            "target": target, "metric": "average_precision" if target == "any_contact" else "mae", "split": split, **row})
        _write_union(folder / "paired_bootstrap.csv", bootstrap_rows)
    if parameter_fingerprint(actor) != actor_hash or any(p.requires_grad or p.grad is not None for p in actor.parameters()):
        raise RuntimeError("Actor changed during holdout evaluation")
    summary = {"status": "frozen_holdout_evaluation_completed", "counts": counts,
        "experiment_ids": experiment_ids, "experiments": experiment_summaries, "candidate_artifact_hashes": frozen_hashes,
        "source_numeric_sha256": {tid: sha256(repo.root / repo.by_id[tid]["numeric_archive"]) for tid in set(data.groups)},
        "test_numeric_archives_loaded": len(data.coverage), "bootstrap_repetitions": 2000,
        "policy_weights_unchanged": True, "no_refitting": True, "physical_simulation": False,
        "scope": "first confirmed event windows; absence and truncation retained in coverage; no closed-loop claim",
        "uncertainty": "per-seed trajectory bootstrap describes trajectory sampling, not independent object-population or training-seed uncertainty"}
    dump(folder / "summary.json", summary)
    return folder, summary
