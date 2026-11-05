"""Read-only summaries and paired inference for the pre-frozen holdout plan.

No model, probe, scaler, coordinate, threshold or method is fitted or selected.
Bootstrap draws are whole trajectories, shared across all seeds/comparisons.
The intervals condition on the sampled objects and do not describe an object
population or adjust for the multiple descriptive method comparisons.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from temp_scripts.project_paths import output_path

TARGETS = ("distance_m", "opening_rad", "any_contact")
SPLITS = ("test_trajectory", "test_object_policy_seen", "test_object_policy_unseen")
Z_MODELS = ("z_top1", "z_top2", "z_top4", "z_linear", "z_top4_time")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numeric(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _ap_draw_scores(y, predictions, row_weights):
    """Exact weighted AP for many bootstrap draws, with prediction ties grouped.

    Sorting each seed once avoids repeating sklearn's sort thousands of times.
    This implements the same non-interpolated weighted precision-recall area.
    Draws with only one observed class are explicitly undefined (NaN).
    """
    y = np.asarray(y, dtype=float)
    predictions = np.atleast_2d(predictions)
    weights = np.asarray(row_weights, dtype=float)
    total_positive = weights @ y
    total_negative = weights @ (1. - y)
    valid = (total_positive > 0) & (total_negative > 0)
    result = np.full((len(predictions), len(weights)), np.nan)
    for seed_index, prediction in enumerate(predictions):
        order = np.argsort(-prediction, kind="stable")
        endpoints = np.r_[np.flatnonzero(np.diff(prediction[order])), len(y) - 1]
        ordered_weights = weights[:, order]
        cumulative_positive = np.cumsum(ordered_weights * y[order][None, :], axis=1)[:, endpoints]
        cumulative_mass = np.cumsum(ordered_weights, axis=1)[:, endpoints]
        precision = np.divide(cumulative_positive, cumulative_mass, out=np.zeros_like(cumulative_positive),
                              where=cumulative_mass > 0)
        positive_increment = np.diff(cumulative_positive, axis=1, prepend=0.)
        numerator = np.sum(positive_increment * precision, axis=1)
        result[seed_index, valid] = numerator[valid] / total_positive[valid]
    return result


def seed_averaged_trajectory_bootstrap(y, candidates, references, groups, *, classification=False,
                                       repetitions=2000, rng_seed=20261102, draws=None):
    """Positive = improvement; CI of the mean paired difference across seeds.

    Regression differences are aggregated within trajectories before resampling.
    AP is recomputed separately for every seed and draw, then the paired seed
    differences are averaged. This never averages per-seed CI endpoints.
    A one-dimensional deterministic reference is shared by all candidate seeds.
    """
    y, groups = np.asarray(y, float), np.asarray(groups)
    candidates = np.atleast_2d(np.asarray(candidates, float))
    references = np.atleast_2d(np.asarray(references, float))
    if y.ndim != 1 or groups.shape != y.shape or candidates.shape[1:] != y.shape:
        raise ValueError("Targets, groups and candidate predictions must align")
    if references.shape[1:] != y.shape or references.shape[0] not in (1, candidates.shape[0]):
        raise ValueError("Use a shared reference or an equally seeded paired reference")
    references = np.broadcast_to(references, candidates.shape)
    if not all(np.isfinite(a).all() for a in (y, candidates, references)):
        raise ValueError("Bootstrap inputs must be finite")
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Need at least two whole trajectories")
    if draws is None:
        rng = np.random.default_rng(rng_seed)
        draws = rng.multinomial(len(unique), np.full(len(unique), 1. / len(unique)), size=repetitions)
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != len(unique) or np.any(draws < 0) or np.any(draws.sum(axis=1) != len(unique)):
        raise ValueError("Draws must be multiplicities of whole-trajectory samples")
    row_base_weights = 1. / sizes[inverse]
    if classification:
        if not np.isin(y, (0., 1.)).all():
            raise ValueError("AP requires binary targets")
        # Match activation_baseline.metrics, which clips serialized probabilities
        # before evaluating AP/log loss; extreme ties must remain identical.
        candidates = np.clip(candidates, 1e-12, 1 - 1e-12)
        references = np.clip(references, 1e-12, 1 - 1e-12)
        both = len(np.unique(y)) == 2
        observed = np.array([average_precision_score(y, c, sample_weight=row_base_weights)
            - average_precision_score(y, r, sample_weight=row_base_weights) for c, r in zip(candidates, references)]) if both else None
        row_weights = draws[:, inverse] * row_base_weights[None, :]
        candidate_scores = _ap_draw_scores(y, candidates, row_weights)
        reference_scores = _ap_draw_scores(y, references, row_weights)
        differences = candidate_scores - reference_scores
        valid = np.isfinite(differences).all(axis=0)
        sampled = differences[:, valid].mean(axis=0)
    else:
        row_improvement = np.abs(references - y) - np.abs(candidates - y)
        trajectory_improvement = np.array([np.bincount(inverse, weights=row, minlength=len(unique)) / sizes
                                           for row in row_improvement])
        observed = trajectory_improvement.mean(axis=1)
        sampled = draws @ trajectory_improvement.mean(axis=0) / len(unique)
    ci = np.quantile(sampled, (.025, .975)).tolist() if len(sampled) else [None, None]
    return {"improvement_mean": float(observed.mean()) if observed is not None else None,
        "improvement_training_seed_sd": float(observed.std(ddof=1)) if observed is not None and len(observed) > 1 else None,
        "ci95_low": ci[0], "ci95_high": ci[1], "defined_replicates": len(sampled), "bootstrap_repetitions": len(draws),
        "training_seeds": len(candidates), "trajectories": len(unique),
        "draw_multiplicity_sha256": hashlib.sha256(np.ascontiguousarray(draws, dtype="<i8").tobytes()).hexdigest(),
        "positive_means": "lower MAE / higher AP", "resampling_unit": "whole trajectory",
        "interval_scope": "conditional on sampled objects; descriptive percentile interval; no multiplicity correction"}


def _load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def _prediction(predictions, target, representation):
    return predictions[f"{target}__{representation}__seed0"]


def _check_frozen_artifacts(root, plan):
    for identifier, expected in plan["artifact_hashes"].items():
        for filename, digest in expected.items():
            if _hash(root / identifier / filename) != digest:
                raise ValueError("Frozen development artifact changed: " + identifier + "/" + filename)
    for entry in plan["rotation_controls"]:
        for filename, digest in entry["artifact_hashes"].items():
            if _hash(root / entry["path"] / filename) != digest:
                raise ValueError("Frozen rotation artifact changed: " + entry["path"] + "/" + filename)


def analyze_holdout(study_root, holdout_root, *, repetitions=2000):
    """Summarize the complete pre-frozen evaluation and its paired uncertainty."""
    root, held = output_path(study_root), output_path(holdout_root)
    plan = _read_json(root / "frozen_evaluation_plan.json")
    if _read_json(held / "frozen_evaluation_plan.json") != plan:
        raise ValueError("Holdout copied plan differs from the original frozen plan")
    summary = _read_json(held / "summary.json")
    if summary["status"] != "frozen_holdout_evaluation_completed" or not summary["no_refitting"]:
        raise ValueError("Wait for the complete frozen holdout evaluation")
    rotation_summary = _read_json(held / "rotation_controls" / "summary.json")
    if rotation_summary["status"] != "frozen_rotation_holdout_evaluation_completed" or not rotation_summary["no_refitting"]:
        raise ValueError("Wait for the frozen rotation controls")
    _check_frozen_artifacts(root, plan)
    output = held / "final_analysis"
    output.mkdir(exist_ok=True)
    layer = _read_json(root / "configuration.json").get("layer", "h1")
    arrays = _load_predictions(held / "selected_arrays.npz")
    if not set(arrays["split"]).issubset(SPLITS):
        raise ValueError("Saved holdout rows contain unexpected splits")
    methods, predictions, state_rows, fidelity_rows = defaultdict(dict), {}, [], []
    metric_values, fidelity_values = defaultdict(list), defaultdict(list)
    source_hashes = {}
    for identifier in plan["experiment_ids"]:
        folder = held / identifier
        result = _read_json(folder / "summary.json")
        name, seed = result["configuration"]["name"], int(result["seed"])
        methods[name][seed] = identifier
        predictions[identifier] = _load_predictions(folder / "probe_predictions.npz")
        source_hashes[identifier] = {f: _hash(folder / f) for f in ("summary.json", "probe_metrics.csv", "probe_predictions.npz")}
        for row in _read_csv(folder / "probe_metrics.csv"):
            if row.get("weighting") != "trajectory" or row.get("model") not in Z_MODELS:
                continue
            target = row["target"]
            for metric in ("average_precision", "log_loss") if target == "any_contact" else ("mae",):
                value = _numeric(row.get(metric))
                if value is not None:
                    key = (row["split"], name, "sae", target, row["model"], metric, "")
                    metric_values[key].append(value)
        for row in result["fidelity"]:
            if row.get("weighting") != "trajectory" or "quantity" not in row:
                continue
            for metric in ("mae", "mse", "max_abs", "nmse_vs_train_mean"):
                value = _numeric(row.get(metric))
                if value is not None:
                    fidelity_values[(row["split"], name, row["quantity"], metric)].append(value)
        for split, health in result.get("health", {}).items():
            for key in ("mean_l0", "never_active_features", "negative_reconstructed_coordinate_fraction"):
                if key in health:
                    fidelity_values[(split, name, "feature_health", key)].append(float(health[key]))
    primary, single = plan["primary_configuration"], plan["single_feature_configuration"]
    pure, action_only = "dense_l0p0", "action_s0.0_a1.0"
    seed_set = sorted(methods[primary])
    if seed_set != [0, 1, 2] or any(sorted(cases) != seed_set for cases in methods.values()):
        raise ValueError("Every frozen method must retain all three matched training seeds")
    if set(methods) != set(plan["configuration_roles"]):
        raise ValueError("Frozen method roles and evaluated methods differ")
    raw_source = held / methods[primary][0]
    raw_models = (layer + "_top1", layer + "_top2", layer + "_top4", layer + "_linear",
                  layer + "_top4_time", "time_spline", layer + "_time_linear")
    for row in _read_csv(raw_source / "probe_metrics.csv"):
        if row.get("weighting") != "trajectory" or row.get("model") not in raw_models:
            continue
        target, model = row["target"], row["model"]
        for metric in ("average_precision", "log_loss") if target == "any_contact" else ("mae",):
            value = _numeric(row.get(metric))
            if value is not None:
                metric_values[(row["split"], "time_only" if model == "time_spline" else "raw_neurons",
                    "time" if model == "time_spline" else "raw", target, model, metric, "")].append(value)
    # Shared Ridge is one pre-frozen configuration for all targets. The per-target
    # role is a distinct development-selected reference, even if its distance
    # control happens to be the same saved rotation as the shared control.
    rotation_predictions = {}
    shared_control = Path(plan["rotation_primary_control"]).name
    chosen_controls = {target: item["control_id"] for target, item in plan["rotation_per_target_controls"].items()}
    for entry in plan["rotation_controls"]:
        control_id = Path(entry["path"]).name
        folder = held / "rotation_controls" / control_id
        rotation_predictions[control_id] = _load_predictions(folder / "probe_predictions.npz")
        source_hashes[control_id] = {f: _hash(folder / f) for f in ("summary.json", "probe_metrics.csv", "probe_predictions.npz")}
        for row in _read_csv(folder / "probe_metrics.csv"):
            if row.get("weighting") != "trajectory" or row.get("model") not in Z_MODELS:
                continue
            target = row["target"]
            roles = []
            if control_id == shared_control:
                roles.append(("ridge_shared", "shared_ridge"))
            if control_id == chosen_controls[target]:
                roles.append(("ridge_per_target", "per_target_ridge"))
            for name, role in roles:
                for metric in ("average_precision", "log_loss") if target == "any_contact" else ("mae",):
                    value = _numeric(row.get(metric))
                    if value is not None:
                        metric_values[(row["split"], name, role, target, row["model"], metric, control_id)].append(value)
    for key, values in metric_values.items():
        split, name, role, target, representation, metric, control_id = key
        if role == "sae" and len(values) != len(seed_set):
            raise ValueError("An SAE metric is missing a training seed: " + str(key))
        state_rows.append({"split": split, "method": name, "role": role, "target": target,
            "representation": representation, "metric": metric, "mean": float(np.mean(values)),
            "training_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "training_seed_count": len(values), "source_control_id": control_id,
            "capacity": "64-dimensional dense reversible; no TopK" if "ridge" in role else "original activation" if role != "sae" else "TopK SAE"})
    for (split, name, quantity, metric), values in fidelity_values.items():
        if len(values) != len(seed_set):
            raise ValueError("A fidelity metric is missing a training seed")
        fidelity_rows.append({"split": split, "method": name, "quantity": quantity, "metric": metric,
            "mean": float(np.mean(values)), "training_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "training_seed_count": len(values)})
    _write_csv(output / "state_metrics.csv", state_rows)
    _write_csv(output / "fidelity_metrics.csv", fidelity_rows)
    comparisons = [("primary_vs_ordinary_pure", primary, "z_top4", pure, "z_top4"),
        ("primary_vs_action_only", primary, "z_top4", action_only, "z_top4"),
        ("primary_vs_raw_top4", primary, "z_top4", "raw", layer + "_top4"),
        ("primary_vs_shared_ridge", primary, "z_top4", "shared_ridge", "z_top4"),
        ("primary_time_top4_vs_raw_time_top4", primary, "z_top4_time", "raw", layer + "_top4_time"),
        ("single_top1_vs_raw_top1", single, "z_top1", "raw", layer + "_top1")]
    bootstrap_rows = []
    for split_index, split in enumerate(SPLITS):
        mask = arrays["split"] == split
        if not mask.any():
            continue
        groups = arrays["trajectory_id"][mask]
        unique = np.unique(groups)
        rng = np.random.default_rng(20261102 + split_index)
        draws = rng.multinomial(len(unique), np.full(len(unique), 1. / len(unique)), size=repetitions)
        for comparison, candidate_name, candidate_rep, reference_name, reference_rep in comparisons:
            for column, target in enumerate(TARGETS):
                candidate_arrays = np.array([_prediction(predictions[methods[candidate_name][seed]], target, candidate_rep)[mask]
                                             for seed in seed_set])
                if reference_name in methods:
                    reference_arrays = np.array([_prediction(predictions[methods[reference_name][seed]], target, reference_rep)[mask]
                                                 for seed in seed_set])
                elif reference_name == "raw":
                    reference_arrays = _prediction(predictions[methods[primary][0]], target, reference_rep)[mask]
                else:
                    reference_arrays = _prediction(rotation_predictions[shared_control], target, reference_rep)[mask]
                result = seed_averaged_trajectory_bootstrap(arrays["y"][mask, column], candidate_arrays,
                    reference_arrays, groups, classification=target == "any_contact", draws=draws)
                bootstrap_rows.append({"split": split, "comparison": comparison, "candidate": candidate_name,
                    "candidate_representation": candidate_rep, "reference": reference_name,
                    "reference_representation": reference_rep, "target": target,
                    "metric": "average_precision" if target == "any_contact" else "mae", **result})
            print("Final paired bootstrap:", split, comparison, flush=True)
    _write_csv(output / "seed_averaged_paired_bootstrap.csv", bootstrap_rows)
    _check_frozen_artifacts(root, plan)
    audit = {"status": "completed", "analysis_code_sha256": _hash(Path(__file__)),
        "frozen_plan_sha256": _hash(root / "frozen_evaluation_plan.json"), "source_artifact_hashes": source_hashes,
        "holdout_arrays_sha256": _hash(held / "selected_arrays.npz"), "methods": list(methods),
        "primary_configuration": primary, "single_feature_configuration": single,
        "shared_ridge_control": shared_control, "target_selected_ridge_controls": chosen_controls,
        "bootstrap_repetitions": repetitions, "paired_bootstrap_rows": len(bootstrap_rows),
        "bootstrap_seed_base": 20261102,
        "training_seed_aggregation": "mean of per-seed paired metric differences before percentile CI; seed sample SD separate",
        "regression_bootstrap": "per-trajectory absolute-error differences, vectorized over common trajectory multiplicities",
        "AP_bootstrap": "exact tied weighted AP computed for each seed and common draw separately, then mean paired difference",
        "frozen_artifact_hashes_unchanged": True, "probe_or_model_refitting": False,
        "uncertainty_scope": "sampled trajectories conditional on these objects; no object-population CI; descriptive multiple comparisons",
        "scope": "first-confirmed-event local windows; no physical rollout conclusion"}
    _dump(output / "analysis_provenance.json", audit)
    _write_markdown(output, state_rows, fidelity_rows, bootstrap_rows, plan)
    return output, audit


def _format(value, sd=None):
    return "—" if value is None else f"{value:.5f}" + (f" ± {sd:.5f}" if sd is not None else "")


def _write_markdown(folder, state_rows, fidelity_rows, bootstrap_rows, plan):
    lookup = {(r["split"], r["method"], r["target"], r["representation"], r["metric"]): r for r in state_rows}
    fidelity = {(r["split"], r["method"], r["quantity"], r["metric"]): r for r in fidelity_rows}
    names = list(plan["configuration_roles"])
    lines = ["# 冻结方案后的独立测试结果", "",
        "75配置×3种子开发完成后，7配置及Ridge对照先冻结再评价；没有根据测试结果重选方法、坐标、阈值或超参数。",
        "±为3个训练种子的样本标准差；表内确定性原激活/Ridge对照不制造重复种子误差线。",
        "所有结果仅覆盖首次已确认持续开启的局部窗口；三类原测试划分分别报告。", ""]
    for split in SPLITS:
        lines += [f"## {split}", "", "|表示|Top4距离MAE(m)↓|Top4开启量MAE(rad)↓|Top4接触AP↑|裁剪动作MAE↓|",
                  "|---|---:|---:|---:|---:|"]
        for name in names + ["raw_neurons", "ridge_shared", "ridge_per_target"]:
            representation = "h1_top4" if name == "raw_neurons" else "z_top4"
            cells = []
            for target in TARGETS:
                metric = "average_precision" if target == "any_contact" else "mae"
                row = lookup.get((split, name, target, representation, metric))
                cells.append(_format(row["mean"], row["training_seed_sd"]) if row else "—")
            row = fidelity.get((split, name, "action_clipped", "mae"))
            cells.append(_format(row["mean"], row["training_seed_sd"]) if row else "—")
            lines.append("|" + "|".join([name] + cells) + "|")
        lines += [""]
    lines += ["Ridge shared 是一个预冻结旋转用于全部目标；Ridge per-target 分别沿用开发集为各目标选定的旋转，不是测试上最优值。",
        "两者均为64维稠密可逆、有监督定向、无TopK，容量条件与SAE不同。", "",
        "## 配对不确定性", "",
        "seed_averaged_paired_bootstrap.csv 对预冻结primary/single比较做2000次整轨迹配对重采样。",
        "每次共同抽相同轨迹；MAE先按轨迹计算误差，AP对每个seed独立重算后再平均配对差。没有平均各seed的CI端点。",
        "所有差值统一正值表示改善；训练seed波动单独列出。区间条件于当前已采样物体，不能解释为物体总体区间；多个消融比较仅作描述。", "",
        "state_metrics.csv 包含各方法Top1/2/4/全表示、time+Top4及原激活/时间/两类Ridge对照；接触AP与log loss分别保留。",
        "fidelity_metrics.csv 保留激活与未裁剪/裁剪动作的均值、种子标准差和特征使用情况。",
        "analysis_provenance.json 记录冻结artifact前后哈希、来源和重采样方法；原模型与探针未被修改。", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
