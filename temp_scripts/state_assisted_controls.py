"""Train-only supervised rotations as strong, non-SAE state-probing controls.

These controls use all three training labels to orient a reversible dense basis.
They have neither TopK sparsity nor a learned activation reconstruction bottleneck.
They test how much apparent semantic concentration can come from supervision alone.
"""
from pathlib import Path
import csv
import json
import shutil

import numpy as np
from threadpoolctl import threadpool_limits

from temp_scripts.activation_baseline import TARGETS, dump, sample_weights, sha256, write_csv
from temp_scripts.project_paths import output_path
from temp_scripts.sae_training import export_state_probes


def _orient_columns(basis):
    """Remove column sign ambiguity; degenerate subspaces remain non-unique."""
    basis = basis.copy()
    pivots = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[pivots, np.arange(basis.shape[1])])
    return basis * np.where(signs == 0, 1., signs)[None, :]


def fit_supervised_rotation(activations, targets, weights, preprocessing="coordinate", floor=1e-6):
    """Fit from training arrays only; W sums to one and covariance is X.T W Y.

    The left singular directions of the weighted activation/label cross-covariance
    form at most three supervised axes. PCA in their orthogonal complement fills
    a square orthogonal basis. Regression targets and binary contact are centered
    and standardized for this basis fit; contact is still classified in probes.
    """
    h, y, w = np.asarray(activations, float), np.asarray(targets, float), np.asarray(weights, float)
    if h.ndim != 2 or h.shape[1] < 1 or y.shape != (len(h), 3) or w.shape != (len(h),) or not len(h):
        raise ValueError("Need aligned nonempty activation, three-label and weight arrays")
    if not all(np.isfinite(v).all() for v in (h, y, w)) or (w <= 0).any():
        raise ValueError("Training activations, labels and positive weights must be finite")
    if not np.isin(y[:, 2], (0., 1.)).all():
        raise ValueError("The contact target must be binary")
    if preprocessing not in ("coordinate", "global"):
        raise ValueError("Unknown activation preprocessing")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Variance floor must be finite and positive")
    w = w / w.sum()
    h_mean = np.average(h, axis=0, weights=w)
    h_std = np.sqrt(np.average((h - h_mean) ** 2, axis=0, weights=w))
    if preprocessing == "global":
        scale = float(np.sqrt(np.mean(h_std ** 2)))
        h_scale = np.full(h.shape[1], scale if scale >= floor else 1.)
    else:
        h_scale = np.where(h_std < floor, 1., h_std)
    y_mean = np.average(y, axis=0, weights=w)
    y_std = np.sqrt(np.average((y - y_mean) ** 2, axis=0, weights=w))
    y_scale = np.where(y_std < floor, 1., y_std)
    x = (h - h_mean) / h_scale
    c = (y - y_mean) / y_scale
    cross_covariance = x.T @ (w[:, None] * c)
    left, singular_values, _ = np.linalg.svd(cross_covariance, full_matrices=True)
    tolerance = (max(cross_covariance.shape) * np.finfo(float).eps * singular_values[0]
                 if len(singular_values) and singular_values[0] > 0 else 0.)
    rank = int(np.count_nonzero(singular_values > tolerance))
    supervised, complement = left[:, :rank], left[:, rank:]
    covariance = x.T @ (w[:, None] * x)
    if complement.shape[1]:
        residual_covariance = complement.T @ covariance @ complement
        residual_covariance = (residual_covariance + residual_covariance.T) / 2
        values, axes = np.linalg.eigh(residual_covariance)
        order = np.argsort(values)[::-1]
        residual_values = values[order]
        complement = complement @ axes[:, order]
    else:
        residual_values = np.empty(0)
    rotation = _orient_columns(np.column_stack((supervised, complement)))
    if not np.allclose(rotation.T @ rotation, np.eye(h.shape[1]), atol=1e-10, rtol=0):
        raise ValueError("Supervised/complement rotation is not orthogonal")
    return {"rotation": rotation, "h_mean": h_mean, "h_scale": h_scale,
            "y_mean": y_mean, "y_scale": y_scale,
            "cross_covariance": cross_covariance, "singular_values": singular_values,
            "supervised_rank": np.array(rank), "residual_pca_variances": residual_values,
            "preprocessing": np.array(preprocessing)}


def transform_supervised_rotation(activations, projection):
    return ((np.asarray(activations, float) - projection["h_mean"]) /
            projection["h_scale"]) @ projection["rotation"]


def inverse_supervised_rotation(codes, projection):
    return ((np.asarray(codes, float) @ projection["rotation"].T) *
            projection["h_scale"] + projection["h_mean"])


def run_supervised_controls(study_root):
    """Export matched fresh probes for global and coordinate supervised rotations.

    Reads only the saved train/validation study arrays. It never opens original
    test trajectories, fits a policy, or evaluates a physical simulation.
    Returns ``(control_root, validation_rows)``.
    """
    from temp_scripts.state_assisted_study import load_study
    root = output_path(study_root)
    data, _, study = load_study(root)
    if set(data.splits) != {"train", "validation"}:
        raise ValueError("Supervised controls require only train/validation rows")
    layer = study["layer"]
    raw = data.h1 if layer == "h1" else data.h2
    train = data.splits == "train"
    weights = sample_weights(data.groups[train], "trajectory")
    home = output_path(root / "controls" / "supervised_rotation")
    home.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(root / "selected_arrays.npz")
    module_hash = sha256(Path(__file__))
    summaries, validation_rows = [], []
    with threadpool_limits(limits=1):
        for mode in ("coordinate", "global"):
            folder = home / mode
            if (folder / "summary.json").exists():
                old = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                if old["selected_arrays_sha256"] != source_hash or old["module_sha256"] != module_hash:
                    raise ValueError("Existing supervised control has different inputs/code; preserve it and use a new study")
                summaries.append(old)
            else:
                if folder.exists():
                    raise ValueError("Incomplete supervised control exists: " + str(folder))
                folder.mkdir()
                projection = fit_supervised_rotation(raw[train], data.y[train], weights, mode)
                z = transform_supervised_rotation(raw, projection)
                reconstructed = inverse_supervised_rotation(z, projection)
                error = float(np.max(np.abs(reconstructed - raw)))
                if error > 1e-9:
                    raise ValueError("Dense reversible control failed its inverse check")
                np.savez_compressed(folder / "projection.npz", **projection,
                    training_row_indices=np.flatnonzero(train), training_row_weights=weights)
                np.savez_compressed(folder / "codes.npz", z=z)
                probes = export_state_probes(folder, data, z, layer, "trajectory")
                with np.load(folder / "projection.npz", allow_pickle=False) as saved:
                    reloaded = {key: saved[key] for key in saved.files}
                reload_error = float(np.max(np.abs(z - transform_supervised_rotation(raw, reloaded))))
                summary = {"method": "weighted_supervised_crosscovariance_rotation",
                    "preprocessing": mode, "layer": layer, "latent_dimension": int(raw.shape[1]),
                    "supervised_rank": int(projection["supervised_rank"]),
                    "training_rows": int(train.sum()), "validation_rows": int((~train).sum()),
                    "training_trajectories": len(set(data.groups[train])),
                    "fitted_on": "train only, trajectory weighted",
                    "supervised_targets": list(TARGETS),
                    "target_preprocessing": "train weighted zscore for both regressions and binary contact; constant scale=1",
                    "complement": "weighted PCA in the supervised axes' orthogonal complement",
                    "probe_selection": "same fresh linear/logistic probes and train-only correlation top1/2/4 selection as SAE",
                    "maximum_inverse_abs_error": error, "maximum_reload_code_abs_error": reload_error,
                    "orthogonality_max_abs_error": float(np.max(np.abs(projection["rotation"].T @
                        projection["rotation"] - np.eye(raw.shape[1])))),
                    "capacity_note": "full-dimensional dense reversible rotation; no TopK or activation bottleneck; state labels orient up to three axes. Reconstruction identity is by construction and is not an SAE fidelity win.",
                    "test_numeric_archives_loaded": 0, "physical_simulation": False,
                    "selected_arrays_sha256": source_hash, "module_sha256": module_hash,
                    "probes": probes}
                dump(folder / "summary.json", summary)
                shutil.copyfile(Path(__file__), folder / "source_snapshot.py")
                summaries.append(summary)
            with (folder / "probe_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            validation_rows.extend({"control": "supervised_rotation", "preprocessing": mode, **row}
                for row in rows if row["split"] == "validation" and row["weighting"] == "trajectory")
    write_csv(home / "validation_metrics.csv", validation_rows)
    dump(home / "summary.json", {"controls": summaries, "test_numeric_archives_loaded": 0,
                                 "validation_used_for_probe_selection": True})
    (home / "README.md").write_text(
        "# 有监督可逆旋转：非SAE强对照\n\n"
        "仅使用train的轨迹加权状态监督：标准化激活与三个标准化状态目标的交叉协方差做SVD，"
        "前至多3个方向为监督方向，其正交补内做加权PCA，补齐64维可逆正交旋转。"
        "分别保存逐维与全局激活缩放。此方法不是PLS。\n\n"
        "编码器只读取激活；训练状态用于定向。与SAE使用相同的新线性/逻辑探针、"
        "train相关性Top1/2/4筛选、validation正则选择和时间对照。"
        "二值接触只在旋转拟合时进行中心化/缩放，实际探针仍为二分类。\n\n"
        "该表示全64维、稠密、可为负，不具有逐样本TopK约束；恒等重建是构造保证，"
        "不能作为战胜SAE的保真结果。其作用是检查少量状态读出优势是否主要来自监督定向。\n\n"
        "projection.npz保存全部变换、train行号与权重；codes.npz保存编码；"
        "probe_*.csv、coordinate_ranking.csv和fitted_probes.joblib提供同预算读出证据。"
        "validation_metrics.csv仅汇总开发集成绩；无独立测试或物理仿真结论。\n",
        encoding="utf-8")
    return home, validation_rows


def fit_ridge_rotation(activations, targets, weights, alpha=1., preprocessing="coordinate", floor=1e-6):
    """Orient a reversible basis by the train-only multi-target ridge directions.

    Objective: sum_i w_i ||X_i B - Y_i||^2 + alpha ||B||_F^2, with weights
    rescaled to mean one, matching the fresh Ridge probe's weight convention.
    Modified Gram-Schmidt in target order preserves the three coefficient
    columns' span; its orthogonal complement is filled with weighted PCA axes.
    """
    if isinstance(alpha, bool) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    base = fit_supervised_rotation(activations, targets, weights, preprocessing, floor)
    h, y, w = np.asarray(activations, float), np.asarray(targets, float), np.asarray(weights, float)
    w = w / w.mean()
    x = (h - base["h_mean"]) / base["h_scale"]
    c = (y - base["y_mean"]) / base["y_scale"]
    gram = x.T @ (w[:, None] * x)
    cross = x.T @ (w[:, None] * c)
    coefficients = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), cross)
    tolerance = max(coefficients.shape) * np.finfo(float).eps * max(
        float(np.linalg.norm(coefficients, ord=2)), np.finfo(float).tiny)
    selected = []
    for column in coefficients.T:
        vector = column.copy()
        # Two passes avoid numerical overlap when coefficient columns are close.
        for _ in range(2):
            for axis in selected:
                vector -= axis * np.dot(axis, vector)
        norm = np.linalg.norm(vector)
        if norm > tolerance:
            selected.append(vector / norm)
    rank = len(selected)
    supervised = np.column_stack(selected) if rank else np.empty((x.shape[1], 0))
    if rank:
        complement = np.linalg.qr(supervised, mode="complete")[0][:, rank:]
    else:
        complement = np.eye(x.shape[1])
    covariance = gram / w.sum()
    if complement.shape[1]:
        residual = complement.T @ covariance @ complement
        values, axes = np.linalg.eigh((residual + residual.T) / 2)
        order = np.argsort(values)[::-1]
        complement = complement @ axes[:, order]
        residual_values = values[order]
    else:
        residual_values = np.empty(0)
    rotation = _orient_columns(np.column_stack((supervised, complement)))
    if not np.allclose(rotation.T @ rotation, np.eye(x.shape[1]), rtol=0, atol=1e-10):
        raise ValueError("Ridge/complement rotation is not orthogonal")
    return {**base, "rotation": rotation, "ridge_coefficients": coefficients,
            "ridge_alpha": np.array(alpha), "ridge_training_weight_sum": np.array(w.sum()),
            "supervised_rank": np.array(rank), "residual_pca_variances": residual_values}


def run_ridge_controls(study_root, *, alphas=(.1, 1., 10.), preprocessing_modes=("coordinate", "global")):
    """Run predeclared supervised ridge rotations; select Top4 controls on val.

    Every candidate is reported. The best control is selected separately for
    each target (minimum MAE or maximum AP), before any independent evaluation.
    Returns ``(selection_path, validation_rows)``. No holdout arrays are opened.
    """
    from temp_scripts.state_assisted_study import load_study
    root = output_path(study_root)
    data, _, study = load_study(root)
    layer, train = study["layer"], data.splits == "train"
    raw = getattr(data, layer)
    weights = sample_weights(data.groups[train], "trajectory")
    home = output_path(root / "controls")
    home.mkdir(exist_ok=True)
    source_hash, module_hash = sha256(root / "selected_arrays.npz"), sha256(Path(__file__))
    rows, controls = [], []
    with threadpool_limits(limits=1):
        for alpha in alphas:
            for mode in preprocessing_modes:
                control_id = f"ridge_rotation_{str(float(alpha)).replace('.', 'p')}_{mode}"
                folder = home / control_id
                if (folder / "summary.json").exists():
                    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                    if summary["selected_arrays_sha256"] != source_hash or summary["module_sha256"] != module_hash:
                        raise ValueError("Existing ridge control has different inputs/code: " + control_id)
                else:
                    if folder.exists():
                        raise ValueError("Incomplete ridge control exists: " + str(folder))
                    folder.mkdir()
                    projection = fit_ridge_rotation(raw[train], data.y[train], weights, alpha, mode)
                    codes = transform_supervised_rotation(raw, projection)
                    reconstructed = inverse_supervised_rotation(codes, projection)
                    error = float(np.max(np.abs(raw - reconstructed)))
                    if error > 1e-9:
                        raise ValueError("Ridge rotation inverse check failed")
                    np.savez_compressed(folder / "projection.npz", **projection,
                        training_row_indices=np.flatnonzero(train), training_row_weights=weights)
                    np.savez_compressed(folder / "codes.npz", z=codes)
                    probes = export_state_probes(folder, data, codes, layer, "trajectory")
                    summary = {"method": "weighted_ridge_coefficient_rotation", "ridge_alpha": float(alpha),
                        "preprocessing": mode, "layer": layer, "latent_dimension": raw.shape[1],
                        "supervised_rank": int(projection["supervised_rank"]),
                        "fitted_on": "train only, trajectory weights rescaled to mean one",
                        "objective": "sum(w * squared standardized target residuals) + alpha * coefficient Frobenius norm squared",
                        "supervised_targets": list(TARGETS), "supervised_axis_order": list(TARGETS),
                        "complement": "weighted PCA in the ridge coefficient columns' orthogonal complement",
                        "maximum_inverse_abs_error": error,
                        "capacity_note": "64-dimensional dense reversible supervised rotation, no TopK constraint; identity reconstruction is guaranteed by construction and not an SAE fidelity win",
                        "probe_selection": "fresh identical-capacity probes; train-only correlation top1/2/4; validation regularization selection",
                        "selected_arrays_sha256": source_hash, "module_sha256": module_hash,
                        "test_numeric_archives_loaded": 0, "physical_simulation": False, "probes": probes}
                    dump(folder / "summary.json", summary)
                    shutil.copyfile(Path(__file__), folder / "source_snapshot.py")
                controls.append({"path": folder.relative_to(root).as_posix(),
                    "artifact_hashes": {name: sha256(folder / name) for name in
                        ("projection.npz", "fitted_probes.joblib", "summary.json")}})
                with (folder / "probe_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                    metrics_rows = list(csv.DictReader(stream))
                rows.extend({"control_id": control_id, "ridge_alpha": alpha, "preprocessing": mode, **r}
                    for r in metrics_rows if r["split"] == "validation" and r["weighting"] == "trajectory")
    selected = {}
    for target in TARGETS:
        candidates = [r for r in rows if r["target"] == target and r["model"] == "z_top4"]
        metric = "average_precision" if target == "any_contact" else "mae"
        if not candidates:
            raise ValueError("No validation Top4 control prediction for " + target)
        best = min(candidates, key=lambda r: ((-1 if metric == "average_precision" else 1) * float(r[metric]), r["control_id"]))
        selected[target] = {"control_id": best["control_id"], "metric": metric,
                            "validation_value": float(best[metric])}
    for control in controls:
        control["targets"] = [target for target, chosen in selected.items()
                              if control["path"].endswith("/" + chosen["control_id"])]
    selection_path = home / "ridge_rotation_selection.json"
    selection = {"selection_rule": "validation trajectory-weighted Top4: minimum MAE for regression, maximum AP for contact; lexical control_id breaks ties",
        "selected_by_target": selected, "all_development_controls": controls,
        "rotation_controls": [control for control in controls if control["targets"]],
        "test_numeric_archives_loaded": 0, "selected_arrays_sha256": source_hash,
        "capacity_note": "supervised directions; full-dimensional dense reversible rotations, not TopK SAEs"}
    if selection_path.exists():
        old = json.loads(selection_path.read_text(encoding="utf-8"))
        if old != selection:
            raise ValueError("A ridge control selection is already frozen; preserve it and use a new study")
    else:
        dump(selection_path, selection)
    write_csv(home / "ridge_rotation_validation_metrics.csv", rows)
    return selection_path, rows


def evaluate_rotation_holdout(study_root, holdout_root):
    """Apply pre-frozen control projections/probes to saved holdout arrays only.

    The original evaluation plan must contain ``rotation_controls`` entries,
    each with ``path`` relative to study root, ``targets`` and SHA256 hashes of
    projection.npz, fitted_probes.joblib and summary.json. The holdout directory
    must carry an identical copied plan, proving these controls were declared
    before that holdout evaluation opened numeric test archives. No fit occurs.
    """
    import joblib
    from temp_scripts.activation_baseline import WindowData
    from temp_scripts.state_assisted_holdout import (TEST_ROLES, frozen_probe_predictions,
        summarize_holdout_prediction)
    root, held = output_path(study_root), output_path(holdout_root)
    if not held.is_relative_to(root):
        raise ValueError("Holdout output must belong to this study")
    plan_path = root / "frozen_evaluation_plan.json"
    if not plan_path.is_file():
        raise ValueError("Freeze rotation controls in frozen_evaluation_plan.json before reading holdout arrays")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = plan.get("rotation_controls")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The frozen evaluation plan has no rotation controls")
    copied_plan = held / "frozen_evaluation_plan.json"
    if not copied_plan.is_file() or json.loads(copied_plan.read_text(encoding="utf-8")) != plan:
        raise ValueError("Holdout's copied frozen plan must exactly match the control plan")
    required = ("projection.npz", "fitted_probes.joblib", "summary.json")
    seen, verified = set(), []
    for entry in entries:
        control = (root / entry["path"]).resolve()
        if not control.is_relative_to(root / "controls") or control in seen:
            raise ValueError("Control paths must be unique descendants of study controls/")
        seen.add(control)
        targets = entry.get("targets")
        if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
                or any(target not in TARGETS for target in targets)):
            raise ValueError("Each frozen rotation control needs valid, distinct selected targets")
        hashes = {name: sha256(control / name) for name in required}
        if entry.get("artifact_hashes") != hashes:
            raise ValueError("Frozen rotation artifacts changed: " + entry["path"])
        verified.append((entry, control))
    # All manifest and development-artifact checks precede holdout numeric reads.
    study = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
    layer = study["layer"]
    if layer not in ("h1", "h2"):
        raise ValueError("Unknown study layer")
    with np.load(held / "selected_arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    with (held / "selected_frames.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("state_index", "object_id"):
            row[key] = int(row[key])
        row["time_s"] = float(row["time_s"])
    if (not rows or arrays["h"].shape != (len(rows), 64) or arrays["y"].shape != (len(rows), 3)
            or any(arrays[key].shape != (len(rows),) for key in ("split", "trajectory_id", "state_index"))):
        raise ValueError("Saved holdout rows/arrays must align")
    if not set(arrays["split"]).issubset(TEST_ROLES):
        raise ValueError("Saved holdout must contain original test roles only")
    for key in ("split", "trajectory_id", "state_index"):
        if not np.array_equal(arrays[key], np.array([row[key] for row in rows])):
            raise ValueError("Holdout CSV/NPZ identity mismatch: " + key)
    if len({(r["trajectory_id"], r["state_index"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate holdout source state")
    if not np.isfinite(arrays["y"]).all() or not np.isin(arrays["y"][:, 2], (0., 1.)).all():
        raise ValueError("Invalid saved holdout labels")
    for key in ("h", "h1", "h2"):
        if arrays[key].shape != (len(rows), 64) or not np.isfinite(arrays[key]).all():
            raise ValueError("Invalid saved holdout activations")
    if not np.array_equal(arrays["h"], arrays[layer]):
        raise ValueError("Holdout primary activation layer mismatch")
    data = WindowData(rows, arrays["h1"], arrays["h2"], arrays["y"], [])
    destination = held / "rotation_controls"
    if destination.exists():
        raise ValueError("Rotation holdout outputs already exist; preserve the completed evaluation")
    destination.mkdir()
    summaries, all_metrics = [], []
    with threadpool_limits(limits=1):
        for entry, control in verified:
            metadata = json.loads((control / "summary.json").read_text(encoding="utf-8"))
            if metadata["layer"] != layer:
                raise ValueError("Frozen control and holdout layers differ")
            with np.load(control / "projection.npz", allow_pickle=False) as saved:
                projection = {key: saved[key] for key in saved.files}
            codes = transform_supervised_rotation(arrays[layer], projection)
            artifacts = joblib.load(control / "fitted_probes.joblib")
            predictions = frozen_probe_predictions(artifacts, data, codes, layer)
            selected_predictions = {key: value for key, value in predictions.items()
                                    if key.split("__")[0] in entry["targets"]}
            folder = destination / control.name
            folder.mkdir()
            np.savez_compressed(folder / "probe_predictions.npz", **selected_predictions)
            np.savez_compressed(folder / "codes.npz", z=codes)
            metrics_rows, trajectory_rows, object_rows = [], [], []
            for key, prediction in selected_predictions.items():
                target, name, _ = key.split("__")
                s, t, o = summarize_holdout_prediction(data, target, name, 0, prediction)
                metrics_rows.extend(s)
                trajectory_rows.extend(t)
                object_rows.extend(o)
            for name, values in (("probe_metrics", metrics_rows), ("probe_by_trajectory", trajectory_rows),
                                 ("probe_by_object", object_rows)):
                fields = list(dict.fromkeys(key for value in values for key in value))
                write_csv(folder / (name + ".csv"), values, fields)
            all_metrics.extend({"control_path": entry["path"], **row} for row in metrics_rows)
            summary = {"control_path": entry["path"], "targets": entry["targets"],
                "frozen_artifact_hashes": entry["artifact_hashes"], "probe_refitting": False,
                "projection_refitting": False, "physical_simulation": False}
            dump(folder / "summary.json", summary)
            summaries.append(summary)
    fields = list(dict.fromkeys(key for row in all_metrics for key in row))
    write_csv(destination / "probe_metrics.csv", all_metrics, fields)
    summary = {"status": "frozen_rotation_holdout_evaluation_completed", "controls": summaries,
        "frozen_plan_sha256": sha256(plan_path), "holdout_arrays_sha256": sha256(held / "selected_arrays.npz"),
        "no_refitting": True, "physical_simulation": False,
        "selection": "development validation only; selected controls were present in the original copied holdout plan"}
    dump(destination / "summary.json", summary)
    return destination, summary
