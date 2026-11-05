"""Inspectable development-study summaries, train-only matching and reload checks.

This module reads completed outputs only. It never opens numeric test trajectories,
selects models on a test set, or treats adjacent frames as independent samples.
"""
from collections import defaultdict
from itertools import combinations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from temp_scripts.project_paths import output_path

TARGETS = ("distance_m", "opening_rad", "any_contact")
TARGET_LABELS = ("距离 MAE (cm) ↓", "开启量 MAE (rad) ↓", "任一组接触 AP ↑")
METRICS = ("mae", "mae", "average_precision")
SCALES = (100., 1., 1.)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _experiment_paths(root):
    """Completed experiments only; a running experiment has no final summary."""
    root = Path(root)
    paths = []
    for summary in sorted(root.glob("*/summary.json")):
        folder = summary.parent
        if (folder / "reconstruction.npz").is_file() and (folder / "probe_metrics.csv").is_file():
            paths.append(folder)
    return paths


def _spec(summary):
    cfg = dict(summary["configuration"])
    sae = dict(cfg.pop("sae", {}))
    # A flat configuration remains readable for compatible imported references.
    if not sae:
        sae = {k: cfg.pop(k) for k in list(cfg) if k in (
            "width", "top_k", "learning_rate", "batch_size", "max_epochs", "patience", "scale_floor", "preprocessing")}
    return {**sae, **cfg}


def _main_metric(target):
    return "average_precision" if target == "any_contact" else "mae"


def _aggregate(observations):
    groups = defaultdict(list)
    for row in observations:
        groups[row["configuration_id"]].append(row)
    aggregates = []
    for cfg_id, rows in groups.items():
        cfg = json.loads(cfg_id)
        aggregate = {"configuration_id": cfg_id, **cfg,
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
            "seed_count": len(rows)}
        candidates = set(k for r in rows for k in r if k.startswith(("validation_", "z_", "improvement_", "delta_")))
        for key in sorted(candidates):
            values = [_number(r.get(key)) for r in rows]
            complete = all(v is not None for v in values)
            aggregate[key + "_count"] = sum(v is not None for v in values)
            # Missing values stay missing rather than silently changing the seed population.
            if complete:
                aggregate.update({key + "_mean": float(np.mean(values)),
                    key + "_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    key + "_min": min(values), key + "_max": max(values)})
        aggregates.append(aggregate)
    return sorted(aggregates, key=lambda r: r.get("validation_action_clipped_mae_mean", float("inf")))


def summarize_study(root, destination=None):
    """Write long-lived tables/figures. Returns ``(analysis_folder, aggregates)``.

    Re-running updates this derived analysis, never the saved models or inputs.
    Each configuration is grouped by all serialized training choices, including
    its predeclared name, and each seed may occur only once in that group.
    """
    root = Path(root).resolve()
    folder = output_path(destination or root / "analysis")
    folder.mkdir(parents=True, exist_ok=True)
    root_cfg = _json(root / "configuration.json") if (root / "configuration.json").exists() else {}
    observations, sources, baseline_values = [], [], {}
    seen = set()
    for experiment in _experiment_paths(root):
        result = _json(experiment / "summary.json")
        spec = _spec(result)
        layer = result.get("layer", spec.get("layer", root_cfg.get("layer", "h1")))
        spec["layer"] = layer
        cfg_id = json.dumps(spec, sort_keys=True, ensure_ascii=False)
        seed = int(result["seed"])
        identity = (cfg_id, seed)
        if identity in seen:
            raise ValueError("Duplicate configuration/seed; do not count it twice: " + str(experiment))
        seen.add(identity)
        row = {"configuration_id": cfg_id, **spec, "seed": seed,
            "experiment_id": result.get("experiment_id", experiment.name), "experiment_path": str(experiment)}
        row.setdefault("name", experiment.name.rsplit("_seed", 1)[0])
        for key in ("best_epoch", "epochs_completed", "early_stopped", "optimizer_steps", "fitting_seconds"):
            row[key] = result.get(key)
        selected = {r["quantity"]: r for r in result["fidelity"]
                    if r["split"] == "validation" and r["weighting"] == "trajectory"}
        for quantity, key in (("activation_standardized", "mse_standardized"),
                              ("activation_raw", "mse_raw"), ("action_mean", "action_mean_mae"),
                              ("action_clipped", "action_clipped_mae")):
            if quantity in selected:
                row["validation_" + key] = selected[quantity]["mse" if quantity.startswith("activation") else "mae"]
        if "activation_raw" in selected:
            row["validation_nmse_raw"] = selected["activation_raw"].get("nmse_vs_train_mean")
        if "action_clipped" in selected:
            row["validation_action_clipped_max"] = selected["action_clipped"].get("max_abs")
        health = result.get("health", {})
        row["validation_mean_l0"] = health.get("validation", {}).get("mean_l0")
        row["train_unused_features"] = health.get("train", {}).get("never_active_features")
        for probe in _csv(experiment / "probe_metrics.csv"):
            if probe["split"] != "validation" or probe["weighting"] != "trajectory":
                continue
            name, target = probe["model"], probe["target"]
            metric = _main_metric(target)
            if name.startswith("z_"):
                row[f"{name}_{target}_{metric}"] = _number(probe.get(metric))
                if target == "any_contact":
                    row[f"{name}_{target}_log_loss"] = _number(probe.get("log_loss"))
            else:
                for m in (metric, "log_loss") if target == "any_contact" else (metric,):
                    value = _number(probe.get(m))
                    key = (layer, name, target, m)
                    if key in baseline_values and value is not None and baseline_values[key] is not None:
                        if abs(value - baseline_values[key]) > 1e-8:
                            raise ValueError("Raw/time/PCA baseline differs within the same study: " + str(key))
                    baseline_values[key] = value
        for target in TARGETS:
            metric = _main_metric(target)
            raw = baseline_values.get((layer, f"{layer}_top4", target, metric))
            value = row.get(f"z_top4_{target}_{metric}")
            if raw is not None and value is not None:
                row[f"improvement_vs_raw_top4_{target}"] = value - raw if target == "any_contact" else raw - value
        observations.append(row)
        sources.append({"experiment": str(experiment), "summary_sha256": _hash(experiment / "summary.json"),
                        "probe_metrics_sha256": _hash(experiment / "probe_metrics.csv")})
    if not observations:
        raise ValueError("No completed experiments with fresh state probes: " + str(root))
    # A baseline is paired on seed and identical SAE/optimization budget. Auxiliary
    # target choices do not alter which pure SAE is the valid reconstruction reference.
    structural = ("layer", "width", "top_k", "preprocessing", "learning_rate", "batch_size", "max_epochs", "patience", "seed")
    references = defaultdict(list)
    for row in observations:
        if _number(row.get("state_weight", 0)) == 0 and _number(row.get("action_weight", 0)) == 0:
            references[tuple(row.get(k) for k in structural)].append(row)
    pairs = []
    for row in observations:
        candidates = references.get(tuple(row.get(k) for k in structural), [])
        if candidates:
            reference = sorted(candidates, key=lambda r: (r.get("selection") != row.get("selection"), r["name"]))[0]
            row["paired_pure_reference"] = reference["experiment_path"]
            for key in ("validation_action_clipped_mae", "validation_action_mean_mae", "validation_nmse_raw"):
                if _number(row.get(key)) is not None and _number(reference.get(key)) is not None:
                    row["delta_vs_pure_" + key] = row[key] - reference[key]
            for target in TARGETS:
                metric = _main_metric(target)
                key = f"z_top4_{target}_{metric}"
                if _number(row.get(key)) is not None and _number(reference.get(key)) is not None:
                    row[f"improvement_vs_pure_top4_{target}"] = row[key] - reference[key] if target == "any_contact" else reference[key] - row[key]
            pairs.append({"experiment": row["experiment_path"], "pure_reference": reference["experiment_path"],
                          "seed": row["seed"], **{k: v for k, v in row.items() if k.startswith(("delta_", "improvement_vs_pure"))}})
    aggregates = _aggregate(observations)
    _write_csv(folder / "experiments.csv", observations)
    _write_csv(folder / "configurations.csv", aggregates)
    _write_csv(folder / "paired_changes.csv", pairs)
    baselines = [{"layer": k[0], "model": k[1], "target": k[2], "metric": k[3], "value": v}
                 for k, v in baseline_values.items()]
    _write_csv(folder / "raw_baselines.csv", baselines)
    control_path = root / "controls" / "supervised_rotation" / "validation_metrics.csv"
    controls = _csv(control_path) if control_path.exists() else []
    if controls:
        _write_csv(folder / "supervised_rotation_control.csv", controls)
    ridge_path = root / "controls" / "ridge_rotation_validation_metrics.csv"
    ridge_selection_path = root / "controls" / "ridge_rotation_selection.json"
    ridge_controls = _csv(ridge_path) if ridge_path.exists() else []
    ridge_selection = _json(ridge_selection_path) if ridge_selection_path.exists() else {}
    selected_controls = []
    if ridge_controls:
        _write_csv(folder / "ridge_rotation_controls.csv", ridge_controls)
        for target, selected_control in ridge_selection.get("selected_by_target", {}).items():
            selected_controls.extend({**r, "selected_for_target": target}
                for r in ridge_controls if r["control_id"] == selected_control["control_id"] and r["target"] == target)
        _write_csv(folder / "ridge_rotation_selected.csv", selected_controls)
        if ridge_selection:
            _dump(folder / "ridge_rotation_selection.json", ridge_selection)
    _dump(folder / "sources.json", {"root": str(root), "experiments": sources,
        "selected_arrays_sha256": _hash(root / "selected_arrays.npz"),
        "supervised_rotation_control_sha256": _hash(control_path) if controls else None,
        "ridge_rotation_control_sha256": _hash(ridge_path) if ridge_controls else None,
        "ridge_rotation_selection_sha256": _hash(ridge_selection_path) if ridge_selection else None,
        "analysis_code_sha256": _hash(Path(__file__)), "completed_experiments": len(observations),
        "uncertainty": "sample standard deviation across training seeds, not data confidence intervals",
        "scope": "adaptive development validation; no independent-test, causal or physical-motion conclusion"})
    _write_summary(folder, aggregates, selected_controls)
    _plot_tradeoffs(folder, aggregates, baselines)
    _plot_lambda(folder, aggregates, baselines)
    _plot_concentration(folder, aggregates, baselines, controls, selected_controls)
    return folder, aggregates


def _formatted(row, key):
    value, sd = row.get(key + "_mean"), row.get(key + "_std")
    return "—" if value is None else (f"{value:.5f} ± {sd:.5f}" if sd is not None else f"{value:.5f}")


def _write_summary(folder, rows, selected_controls=()):
    lines = ["# 状态辅助 SAE：开发集探索汇总", "",
        "每个配置的均值与样本标准差来自训练种子；不是数据置信区间。仅含已经完成且另训新探针的实验。",
        "validation 已反复用于适应性方法开发；表格按动作误差排序，不代表综合解释质量排名。",
        "同一步物理状态仅作辅助监督；编码器仅输入 h。任一组接触不表示抓稳。", "",
        "|配置|种子数|动作 MAE↓|Top4 距离 MAE(m)↓|Top4 开启量 MAE(rad)↓|Top4 接触 AP↑|",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("|" + "|".join([str(row.get("name", "unnamed")), str(row["seed_count"])] + [
            _formatted(row, k) for k in ("validation_action_clipped_mae", "z_top4_distance_m_mae",
                "z_top4_opening_rad_mae", "z_top4_any_contact_average_precision")]) + "|")
    lines += ["", "experiments.csv 保留逐种子、全部 1/2/4/全 z 读出、时间+z 及输入配置。",
        "paired_changes.csv 保留同种子、相同 SAE 结构/训练预算的纯 SAE 对照差值；没有匹配的配置不造配对。",
        "raw_baselines.csv 保留同批原始激活、时间及 PCA 探针。特征筛选仅由 train 相关性完成。",
        "原尺度 NMSE 与未裁剪/裁剪动作误差均须同时查看；裁剪可能掩盖动作均值偏差。",
        "图中误差线是训练种子标准差。单种子探索没有误差线，不能据此宣称稳健收益。", ""]
    if (folder / "supervised_rotation_control.csv").exists():
        lines += ["监督旋转控制独立见 supervised_rotation_control.csv：64维稠密可逆、无 TopK 容量约束，不计入 SAE 配置/训练种子数。",
            "其完整重建按构造恒等，不能当作与 SAE 稀疏重建相同容量下的保真胜负。", ""]
    if selected_controls:
        lines += ["## 监督定向强对照", "",
            "Ridge 系数子空间定向旋转保留64维稠密可逆表示，无 TopK；仅用 train 定向，再用相同新探针评价。",
            "共6组（α=0.1/1/10 × 全局/逐维缩放）；下表按每个目标的 validation Top4 分别选定控制，不是同一个旋转同时最优。",
            "这一对照能检验收益是否来自标签监督定向本身，不能仅超过原始Top4就把收益归因于SAE特有性质。", "",
            "|目标|选定监督旋转|Top4 validation|", "|---|---|---:|"]
        for target in TARGETS:
            chosen = [r for r in selected_controls if r["target"] == target and r["model"] == "z_top4"
                      and r["weighting"] == "trajectory" and r["split"] == "validation"]
            if chosen:
                r = chosen[0]
                value = _number(r.get(_main_metric(target)))
                lines.append(f"|{target}|{r['control_id']}|{value:.6f}|")
        lines += ["", "完整6组见 ridge_rotation_controls.csv；逐目标冻结选择及原artifact哈希见 ridge_rotation_selection.json。", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _baseline_lookup(rows, model, target, metric):
    matches = [r["value"] for r in rows if r["model"] == model and r["target"] == target and r["metric"] == metric]
    return next((v for v in matches if v is not None), None)


def _colors(rows):
    from matplotlib import colormaps
    return [colormaps["tab20"](i % 20) for i in range(len(rows))]


def _plot_tradeoffs(folder, rows, baselines):
    from matplotlib.figure import Figure
    from temp_scripts.plots import configure_style
    configure_style()
    fig = Figure(figsize=(16, 6), layout="constrained")
    axes = fig.subplots(1, 3)
    colors = _colors(rows)
    for ax, target, metric, label, scale in zip(axes, TARGETS, METRICS, TARGET_LABELS, SCALES):
        key = f"z_top4_{target}_{metric}"
        for i, row in enumerate(rows):
            x, y = row.get("validation_action_clipped_mae_mean"), row.get(key + "_mean")
            if x is None or y is None:
                continue
            ax.errorbar(x, y * scale, xerr=row.get("validation_action_clipped_mae_std") or 0.,
                yerr=(row.get(key + "_std") or 0.) * scale, fmt="o", color=colors[i], ms=5, capsize=2)
            ax.annotate(str(i + 1), (x, y * scale), xytext=(4, 4), textcoords="offset points", fontsize=7, color=colors[i])
        layer = rows[0].get("layer", "h1")
        raw = _baseline_lookup(baselines, layer + "_top4", target, metric)
        if raw is not None:
            ax.axhline(raw * scale, ls="--", lw=1.3, color="#b63c51", label="原始 Top4")
            ax.legend(loc="best")
        ax.set(xlabel="裁剪动作 MAE ↓", ylabel=label, title="固定4特征：" + label)
        ax.grid(True)
    legend = "   ".join(f"{i+1}. {r.get('name', 'unnamed')} (n={r['seed_count']})" for i, r in enumerate(rows))
    import textwrap
    fig.suptitle("状态信息集中与动作保持的取舍 · validation / 轨迹等权\n误差线：训练种子样本标准差", fontsize=12)
    # Large studies keep the full key in a text artifact, avoiding illegible legends.
    (folder / "figure_key.txt").write_text("\n".join(f"{i+1}. {r.get('name', 'unnamed')} | seeds={r['seeds']}" for i, r in enumerate(rows)) + "\n", encoding="utf-8")
    if len(rows) <= 12:
        fig.supxlabel("\n".join(textwrap.wrap(legend, width=150)), fontsize=7)
    fig.savefig(folder / "semantic_fidelity_tradeoff.png", dpi=170)


def _lambda_groups(rows):
    grouped = defaultdict(list)
    excluded = {"name", "state_weight", "configuration_id", "seeds", "seed_count"}
    for row in rows:
        cfg = json.loads(row["configuration_id"])
        key = json.dumps({k: v for k, v in cfg.items() if k not in excluded}, sort_keys=True)
        grouped[key].append(row)
    return [sorted(group, key=lambda r: float(r.get("state_weight", 0))) for group in grouped.values()
            if len({float(r.get("state_weight", 0)) for r in group}) >= 2]


def _plot_lambda(folder, rows, baselines):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(14, 9), layout="constrained")
    axes = fig.subplots(2, 2).ravel()
    panels = [("validation_action_clipped_mae", "裁剪动作 MAE ↓", 1.)] + [
        (f"z_top4_{t}_{m}", l, s) for t, m, l, s in zip(TARGETS, METRICS, TARGET_LABELS, SCALES)]
    groups = _lambda_groups(rows)
    for ax, (key, label, scale) in zip(axes, panels):
        for group in groups:
            available = [r for r in group if r.get(key + "_mean") is not None]
            if not available:
                continue
            first = available[0]
            label_name = "/".join(str(first.get(k, "")) for k in ("head_mode", "regression_loss", "classification_loss", "selection"))
            label_name += f" · K{first.get('top_k')} · {first.get('preprocessing')} · a={first.get('action_weight',0)}"
            ax.errorbar([float(r.get("state_weight", 0)) for r in available], [r[key + "_mean"] * scale for r in available],
                yerr=[(r.get(key + "_std") or 0.) * scale for r in available], fmt="o-", capsize=3, label=label_name)
        ax.set(xlabel="辅助权重 λ（symlog）", ylabel=label, title=label)
        ax.set_xscale("symlog", linthresh=.01)
        ax.grid(True)
        if groups:
            ax.legend(loc="best", fontsize=6)
    fig.suptitle("固定其余训练选择，仅改变 λ · validation\n误差线为种子标准差；不同配置组不混作同一曲线", fontsize=12)
    fig.savefig(folder / "lambda_sensitivity.png", dpi=170)


def _plot_concentration(folder, rows, baselines, controls=(), selected_controls=()):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(16, 6), layout="constrained")
    axes = fig.subplots(1, 3)
    # At most eight curves: pure reference(s) plus Pareto-useful configurations.
    pure = [r for r in rows if float(r.get("state_weight", 0)) == 0 and float(r.get("action_weight", 0)) == 0]
    useful = []
    for target, metric in zip(TARGETS, METRICS):
        key = f"z_top4_{target}_{metric}_mean"
        available = [r for r in rows if r.get(key) is not None]
        if available:
            useful.append((max if target == "any_contact" else min)(available, key=lambda r: r[key]))
    selected, keys = [], set()
    for r in pure + useful + rows:
        if r["configuration_id"] not in keys:
            keys.add(r["configuration_id"])
            selected.append(r)
        if len(selected) == 8:
            break
    for ax, target, metric, label, scale in zip(axes, TARGETS, METRICS, TARGET_LABELS, SCALES):
        for row in selected:
            prefixes = ["z_top1", "z_top2", "z_top4", "z_linear"]
            ys = [row.get(f"{p}_{target}_{metric}_mean") for p in prefixes]
            if any(y is None for y in ys):
                continue
            errors = [(row.get(f"{p}_{target}_{metric}_std") or 0.) * scale for p in prefixes]
            ax.errorbar(range(4), [y * scale for y in ys], yerr=errors, fmt="o-", capsize=2, ms=4, label=row.get("name"))
        layer = rows[0].get("layer", "h1")
        raw_values = [_baseline_lookup(baselines, f"{layer}_top{k}" if k else f"{layer}_linear", target, metric) for k in (1, 2, 4, 0)]
        if all(v is not None for v in raw_values):
            ax.plot(range(4), np.array(raw_values) * scale, "s--", color="#151b27", lw=1.7, label="原始神经元")
        for mode, color in (("global", "#7758a6"), ("coordinate", "#b95c08")):
            values = []
            for prefix in ("z_top1", "z_top2", "z_top4", "z_linear"):
                matching = [r for r in controls if r.get("preprocessing") == mode and r["model"] == prefix
                    and r["target"] == target and r["split"] == "validation" and r["weighting"] == "trajectory"]
                values.append(_number(matching[0].get(metric)) if matching else None)
            if all(v is not None for v in values):
                ax.plot(range(4), np.array(values) * scale, "d:", color=color, lw=1.4, ms=4,
                    label="监督旋转64维稠密 / " + mode)
        ridge_values, control_id = [], None
        for prefix in ("z_top1", "z_top2", "z_top4", "z_linear"):
            matching = [r for r in selected_controls if r["target"] == target and r["model"] == prefix
                        and r["weighting"] == "trajectory" and r["split"] == "validation"]
            ridge_values.append(_number(matching[0].get(metric)) if matching else None)
            if matching:
                control_id = matching[0]["control_id"]
        if all(v is not None for v in ridge_values):
            ax.plot(range(4), np.array(ridge_values) * scale, "D--", color="#9b2359", lw=2.2, ms=5,
                label="Ridge监督旋转 / 各目标独立选参")
        ax.set(xticks=range(4), xticklabels=["1", "2", "4", "全部"], xlabel="固定坐标数量（train筛选）", ylabel=label, title=label)
        ax.grid(True)
    axes[-1].legend(loc="best", fontsize=6)
    fig.suptitle("少量特征与完整表示的状态读出 · validation / 轨迹等权\n最多展示8个配置，完整结果见 configurations.csv；逐样本 TopK 与固定坐标数量不同", fontsize=12)
    fig.savefig(folder / "feature_concentration.png", dpi=170)


def _decoder(model):
    if hasattr(model, "decoder"):
        return model.decoder
    if hasattr(model, "sae"):
        return model.sae.decoder
    raise AttributeError("Checkpoint model exposes neither decoder nor sae.decoder")


def _load_checkpoint(path):
    from temp_scripts.state_assisted_model import load_assisted_checkpoint
    return load_assisted_checkpoint(path)


def verify_study(root, destination=None, atol=1e-6):
    """Audit using the training thread count, restoring the caller's setting."""
    import torch
    root = Path(root).resolve()
    cfg = _json(root / "configuration.json") if (root / "configuration.json").exists() else {}
    previous = torch.get_num_threads()
    try:
        # The numbered study entry uses one thread. Matching this matters for
        # bitwise GEMM reproducibility; we do not loosen tolerance after a failure.
        torch.set_num_threads(int(cfg.get("threads", 1)))
        return _verify_study(root, destination, atol)
    finally:
        torch.set_num_threads(previous)


def _verify_study(root, destination=None, atol=1e-6):
    """Independently reload every checkpoint and recompute saved action errors."""
    import torch
    from temp_scripts.actor_model import FrozenActor, load_actor, parameter_fingerprint
    from temp_scripts.activation_baseline import sample_weights
    from temp_scripts.sae_model import ActivationStandardizer
    root = Path(root).resolve()
    folder = output_path(destination or root / "analysis")
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as a:
        raw, splits, groups = a["h"].copy(), a["split"].copy(), a["trajectory_id"].copy()
    if not set(splits).issubset({"train", "validation"}):
        raise ValueError("Development reload audit found test rows")
    actor = FrozenActor()
    state = torch.load(root / "model" / "frozen_actor_state.pt", map_location="cpu", weights_only=True)
    actor.load_state_dict(state, strict=True)
    actor.requires_grad_(False).eval()
    fingerprint = parameter_fingerprint(actor)
    actor_matches_original = None
    metadata_path = root / "model" / "actor_metadata.json"
    if metadata_path.exists():
        metadata = _json(metadata_path)
        original_actor, _ = load_actor(metadata["source"], metadata["checkpoint_sha256"])
        actor_matches_original = parameter_fingerprint(original_actor) == fingerprint
    root_cfg = _json(root / "configuration.json") if (root / "configuration.json").exists() else {}
    audit = []
    for experiment in _experiment_paths(root):
        result = _json(experiment / "summary.json")
        cfg = _spec(result)
        model, scaler, _ = _load_checkpoint(experiment / "best_model.pt")
        model.eval()
        training = splits == "train"
        recomputed_scaler = ActivationStandardizer.fit(raw[training], sample_weights(groups[training]),
            float(cfg.get("scale_floor", 1e-6)), cfg.get("preprocessing", "global"))
        scaler_difference = max(float(np.max(np.abs(scaler.mean - recomputed_scaler.mean))),
                                float(np.max(np.abs(scaler.scale - recomputed_scaler.scale))))
        x = scaler.transform(raw)
        zs, reconstructions = [], []
        batch_size = int(cfg.get("batch_size", 128))
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                z, rx = model(torch.as_tensor(x[start:start+batch_size], dtype=torch.float32))
                zs.append(z.cpu().numpy())
                reconstructions.append(rx.cpu().numpy())
        z = np.concatenate(zs)
        reconstructed = scaler.inverse_transform(np.concatenate(reconstructions))
        layer = result.get("layer", cfg.get("layer", root_cfg.get("layer", "h1")))
        forward = actor.from_h1 if layer == "h1" else actor.from_h2
        def actor_forward(values):
            blocks = []
            with torch.no_grad():
                for start in range(0, len(values), 256):
                    blocks.append(forward(torch.tensor(values[start:start+256], dtype=torch.float32)).numpy())
            return np.concatenate(blocks)
        reference_mu, mu = actor_forward(raw), actor_forward(reconstructed)
        with np.load(experiment / "reconstruction.npz", allow_pickle=False) as saved:
            differences = {"z": float(np.max(np.abs(z.astype(float) - saved["z"]))),
                "reconstructed_h": float(np.max(np.abs(reconstructed - saved["reconstructed_h"]))),
                "reconstructed_mean": float(np.max(np.abs(mu.astype(float) - saved["reconstructed_mean"]))) }
            if "reconstructed_action" in saved:
                differences["reconstructed_action"] = float(np.max(np.abs(np.clip(mu, -1, 1).astype(float) - saved["reconstructed_action"])))
        metrics = []
        for split in ("train", "validation"):
            mask = splits == split
            weights = sample_weights(groups[mask])
            values = {"action_mean": (reference_mu, mu), "action_clipped": (np.clip(reference_mu, -1, 1), np.clip(mu, -1, 1))}
            for quantity, (ref, prediction) in values.items():
                mae = float(np.average(np.abs(prediction[mask].astype(float) - ref[mask]).mean(axis=1), weights=weights))
                saved_metric = next(r["mae"] for r in result["fidelity"] if r["split"] == split and r["weighting"] == "trajectory" and r["quantity"] == quantity)
                metrics.append({"split": split, "quantity": quantity, "recomputed_mae": mae, "saved_mae": saved_metric,
                                "absolute_difference": abs(mae - saved_metric)})
        metadata_invariants = result.get("test_numeric_archives_loaded", 0) == 0 and result.get("policy_weights_unchanged", True)
        passed = (all(v <= atol for v in differences.values()) and all(m["absolute_difference"] <= atol for m in metrics)
                  and scaler_difference <= 1e-12 and metadata_invariants)
        audit.append({"experiment": experiment.name, "passed": bool(passed), "max_array_differences": differences,
            "train_only_activation_scaler_max_difference": scaler_difference,
            "metadata_invariants_passed": bool(metadata_invariants), "metrics": metrics})
    if not audit:
        raise ValueError("No completed experiments available for independent verification")
    report = {"passed": all(r["passed"] for r in audit) and actor_matches_original is not False,
        "experiments_checked": len(audit), "tolerance": atol,
        "torch_threads": torch.get_num_threads(),
        "test_rows": 0, "actor_parameters_unchanged": fingerprint == parameter_fingerprint(actor), "experiments": audit,
        "saved_actor_matches_original_checkpoint": actor_matches_original, "frozen_actor_parameter_sha256": fingerprint,
        "scope": "checkpoint/array/action numerical reproducibility; not semantic or physical validation"}
    _dump(folder / "independent_verification.json", report)
    if not report["passed"]:
        raise ValueError("Independent checkpoint verification failed; inspect analysis/independent_verification.json")
    return report


def match_seed_features(root, destination=None):
    """Hungarian feature matching based only on train activations and directions.

    Score = (signed original-scale decoder cosine + signed train activation
    correlation) / 2. Nonnegative SAE features do not permit arbitrary sign flips.
    A forced match of unused features is reported as such, never as stable meaning.
    """
    from scipy.optimize import linear_sum_assignment
    from temp_scripts.activation_baseline import sample_weights
    root = Path(root).resolve()
    folder = output_path(destination or root / "analysis")
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as a:
        train = a["split"] == "train"
        groups = a["trajectory_id"][train]
    weights = sample_weights(groups)
    grouped = defaultdict(list)
    for experiment in _experiment_paths(root):
        summary = _json(experiment / "summary.json")
        cfg = json.dumps(_spec(summary), sort_keys=True)
        grouped[cfg].append((int(summary["seed"]), experiment))
    rows, pairs = [], []
    for cfg, cases in grouped.items():
        for (seed_a, a), (seed_b, b) in combinations(sorted(cases), 2):
            model_a, scale_a, _ = _load_checkpoint(a / "best_model.pt")
            model_b, scale_b, _ = _load_checkpoint(b / "best_model.pt")
            da = _decoder(model_a).weight.detach().cpu().numpy().astype(float) * scale_a.scale[:, None]
            db = _decoder(model_b).weight.detach().cpu().numpy().astype(float) * scale_b.scale[:, None]
            da /= np.maximum(np.linalg.norm(da, axis=0, keepdims=True), 1e-12)
            db /= np.maximum(np.linalg.norm(db, axis=0, keepdims=True), 1e-12)
            cosine = da.T @ db
            with np.load(a / "reconstruction.npz", allow_pickle=False) as archive:
                za = archive["z"][train].astype(float)
            with np.load(b / "reconstruction.npz", allow_pickle=False) as archive:
                zb = archive["z"][train].astype(float)
            ca, cb = za - np.average(za, axis=0, weights=weights), zb - np.average(zb, axis=0, weights=weights)
            covariance = ca.T @ (cb * weights[:, None]) / weights.sum()
            variance_a, variance_b = np.average(ca * ca, axis=0, weights=weights), np.average(cb * cb, axis=0, weights=weights)
            denominator = np.sqrt(variance_a[:, None] * variance_b[None, :])
            correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
            correlation = np.clip(correlation, -1, 1)
            score = (cosine + correlation) / 2
            ia, ib = linear_sum_assignment(-score)
            pair_rows = []
            for i, j in zip(ia, ib):
                row = {"name": json.loads(cfg).get("name", a.name.rsplit("_seed", 1)[0]),
                    "seed_a": seed_a, "seed_b": seed_b, "coordinate_a": int(i), "coordinate_b": int(j),
                    "decoder_cosine": float(cosine[i, j]), "train_activation_correlation": float(correlation[i, j]),
                    "matching_score": float(score[i, j]), "active_a": bool(variance_a[i] > 1e-12),
                    "active_b": bool(variance_b[j] > 1e-12), "experiment_a": str(a), "experiment_b": str(b)}
                pair_rows.append(row)
            rows.extend(pair_rows)
            active = [r for r in pair_rows if r["active_a"] and r["active_b"]]
            pairs.append({"name": json.loads(cfg).get("name", a.name.rsplit("_seed", 1)[0]),
                "seed_a": seed_a, "seed_b": seed_b, "matched_features": len(pair_rows), "active_matches": len(active),
                "mean_decoder_cosine_active": float(np.mean([r["decoder_cosine"] for r in active])) if active else None,
                "mean_train_correlation_active": float(np.mean([r["train_activation_correlation"] for r in active])) if active else None,
                "matching_rule": "Hungarian on 0.5*(signed decoder cosine + signed train correlation); train only"})
    _write_csv(folder / "cross_seed_feature_matches.csv", rows)
    _write_csv(folder / "cross_seed_match_summary.csv", pairs)
    _dump(folder / "cross_seed_matching.json", {"pairs": len(pairs), "feature_matches": len(rows), "fitted_on": "train only",
        "no_sign_flips": True, "scope": "alignment evidence; high similarity does not certify identical semantic meaning"})
    return pairs


def bootstrap_comparison(root, candidate_name, reference_name=None, *, repetitions=1000,
                         rng_seed=20261102, representations=("z_top1", "z_top2", "z_top4", "z_linear"),
                         destination=None):
    """Paired trajectory bootstrap of seed-averaged fresh-probe improvements.

    ``reference_name=None`` compares each candidate probe to its same-budget raw
    neuron probe. Otherwise both configurations must have the same seed set.
    Positive improvement means lower MAE / higher AP. This resamples trajectories,
    not frames or training seeds. Adaptive validation intervals are descriptive;
    they cannot repair validation reuse or establish independent generalization.
    """
    from sklearn.metrics import average_precision_score
    from temp_scripts.activation_baseline import sample_weights
    root = Path(root).resolve()
    folder = output_path(destination or root / "analysis")
    folder.mkdir(parents=True, exist_ok=True)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Use a positive integer bootstrap repetition count")
    configs = defaultdict(dict)
    for path in _experiment_paths(root):
        result = _json(path / "summary.json")
        name = _spec(result).get("name", path.name.rsplit("_seed", 1)[0])
        seed = int(result["seed"])
        if seed in configs[name]:
            raise ValueError("Configuration name/seed is not unique")
        configs[name][seed] = path
    if candidate_name not in configs or (reference_name is not None and reference_name not in configs):
        raise ValueError("Requested configuration has no completed experiments")
    candidates = configs[candidate_name]
    if reference_name is not None and set(candidates) != set(configs[reference_name]):
        raise ValueError("Paired bootstrap requires identical training seed sets")
    seeds = sorted(candidates)
    root_cfg = _json(root / "configuration.json")
    layer = root_cfg.get("layer", "h1")
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as source:
        if not set(source["split"]).issubset({"train", "validation"}):
            raise ValueError("Development bootstrap found nondevelopment rows")
        mask = source["split"] == "validation"
        y, groups = source["y"][mask].copy(), source["trajectory_id"][mask].copy()
    unique, inverse = np.unique(groups, return_inverse=True)
    base_weights = sample_weights(groups)
    if len(unique) < 2:
        raise ValueError("Need at least two validation trajectories")
    rng = np.random.default_rng(rng_seed)
    multiplicities = np.array([np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
                               for _ in range(repetitions)])
    prediction_pairs = defaultdict(list)
    for seed in seeds:
        with np.load(candidates[seed] / "probe_predictions.npz", allow_pickle=False) as a:
            predictions = {k: a[k][mask].copy() for k in a.files}
        if reference_name is None:
            reference_predictions = predictions
        else:
            with np.load(configs[reference_name][seed] / "probe_predictions.npz", allow_pickle=False) as a:
                reference_predictions = {k: a[k][mask].copy() for k in a.files}
        for representation in representations:
            reference_representation = (layer + representation[1:]) if reference_name is None else representation
            for target_index, target in enumerate(TARGETS):
                key = f"{target}__{representation}__seed0"
                ref_key = f"{target}__{reference_representation}__seed0"
                if key not in predictions or ref_key not in reference_predictions:
                    raise ValueError("Missing expected fresh probe prediction: " + key + " / " + ref_key)
                prediction_pairs[(representation, target_index)].append((predictions[key], reference_predictions[ref_key]))
    results = []
    for (representation, column), pairs in prediction_pairs.items():
        target = TARGETS[column]
        classification = target == "any_contact"
        truth = y[:, column]
        def score(prediction, weights):
            selected = weights > 0
            if classification:
                if len(np.unique(truth[selected])) < 2:
                    return None
                return float(average_precision_score(truth[selected], prediction[selected], sample_weight=weights[selected]))
            return float(np.average(np.abs(truth - prediction), weights=weights))
        def improvement(weights):
            differences = []
            for candidate, reference in pairs:
                a, b = score(candidate, weights), score(reference, weights)
                if a is None or b is None:
                    return None
                differences.append(a - b if classification else b - a)
            return differences
        observed = improvement(base_weights)
        sampled = [improvement(base_weights * counts[inverse]) for counts in multiplicities]
        sampled_means = [float(np.mean(values)) for values in sampled if values is not None]
        ci = np.quantile(sampled_means, [.025, .975]).tolist() if sampled_means else [None, None]
        results.append({"candidate": candidate_name, "reference": reference_name or "raw_neurons_same_coordinate_budget",
            "representation": representation, "target": target, "metric": _main_metric(target),
            "improvement_mean": float(np.mean(observed)) if observed is not None else None,
            "improvement_seed_sd": float(np.std(observed, ddof=1)) if observed is not None and len(seeds) > 1 else None,
            "bootstrap_95pct_low": ci[0], "bootstrap_95pct_high": ci[1], "valid_repetitions": len(sampled_means),
            "training_seeds": ",".join(map(str, seeds)), "validation_trajectories": len(unique),
            "positive_means": "lower MAE or higher AP", "resampling_unit": "whole trajectory"})
    safe = lambda value: "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))
    stem = f"bootstrap_{safe(candidate_name)}_vs_{safe(reference_name or 'raw')}"
    _write_csv(folder / (stem + ".csv"), results)
    _dump(folder / (stem + ".json"), {"repetitions": repetitions, "rng_seed": rng_seed, "results": results,
        "resampling": "same whole validation trajectories for all seeds/methods; seed-averaged metric difference",
        "training_seed_variation": "reported separately as sample SD, not folded into trajectory bootstrap",
        "scope": "descriptive adaptive-validation uncertainty; not independent-test evidence or multiplicity-adjusted inference"})
    return results


def pareto_candidates(root, *, min_seeds=3, destination=None):
    """Return mean-metric nondominated candidates within matched capacity groups.

    The four objectives are clipped-action MAE, Top4 distance MAE, Top4 opening
    MAE, and negative Top4 contact AP. No arbitrary composite weights or physical
    acceptance threshold are introduced. TopK/preprocessing changes remain in
    separate groups. A frontier is a development shortlist, not a significance
    test, a method ranking, or a semantic acceptance certificate.
    """
    root = Path(root).resolve()
    folder = output_path(destination or root / "analysis")
    records = _csv(folder / "configurations.csv")
    objectives = ("validation_action_clipped_mae_mean", "z_top4_distance_m_mae_mean",
                  "z_top4_opening_rad_mae_mean", "z_top4_any_contact_average_precision_mean")
    eligible, flags = [], []
    for record in records:
        n = int(record["seed_count"])
        vector = [_number(record.get(k)) for k in objectives]
        if record.get("label_mode", "true") not in ("", "true"):
            flags.append({"name": record.get("name"), "eligible": False, "seed_count": n,
                "reason": "negative label control, not a method candidate", "nondominated": False})
        elif n >= min_seeds and all(v is not None for v in vector):
            eligible.append((record, np.array(vector) * np.array([1, 1, 1, -1])))
        else:
            flags.append({"name": record.get("name"), "eligible": False, "seed_count": n,
                "reason": "insufficient seeds or missing objective", "nondominated": False})
    selected = []
    for record, vector in eligible:
        capacity = tuple(record.get(k) for k in ("layer", "width", "top_k", "preprocessing"))
        dominators = []
        for other, other_vector in eligible:
            other_capacity = tuple(other.get(k) for k in ("layer", "width", "top_k", "preprocessing"))
            if capacity == other_capacity and np.all(other_vector <= vector + 1e-12) and np.any(other_vector < vector - 1e-12):
                dominators.append(other.get("name"))
        front = not dominators
        gains = [_number(record.get(f"improvement_vs_raw_top4_{t}_mean")) for t in TARGETS]
        flag = {"name": record.get("name"), "eligible": True, "seed_count": int(record["seed_count"]),
            "capacity_group": "/".join(map(str, capacity)), "nondominated": front,
            "dominators": ",".join(dominators), "beats_raw_top4_all_three_means": all(g is not None and g > 0 for g in gains),
            **{k: record[k] for k in objectives}}
        flags.append(flag)
        if front:
            selected.append({**record, "capacity_group": flag["capacity_group"],
                             "beats_raw_top4_all_three_means": flag["beats_raw_top4_all_three_means"]})
    _write_csv(folder / "pareto_flags.csv", flags)
    _write_csv(folder / "pareto_candidates.csv", selected)
    _dump(folder / "pareto_definition.json", {"minimum_training_seeds": min_seeds,
        "grouping": ["layer", "width", "top_k", "preprocessing"], "objectives": list(objectives),
        "direction": ["minimize", "minimize", "minimize", "maximize"], "selected_candidates": len(selected),
        "scope": "development shortlist based on seed means; no composite score, significance, or physical-fidelity threshold"})
    return selected


def augment_time_top_probes(root, experiment_ids):
    """Append time+fixed-Top1/2/4 probes after candidate training has finished.

    Every coordinate, activation transform and time basis is reused from the
    completed train-fitted artifacts. The new probe regularization has exactly
    the existing development validation budget. Existing estimators, predictions
    and scores remain present and unchanged. Original files are backed up before
    atomic file replacement; a completed manifest makes a second call a no-op.
    The holdout reader uses ``with_time=True`` to prepend the stored time basis.
    """
    import joblib
    import shutil
    from datetime import datetime, timezone, timedelta
    from threadpoolctl import threadpool_limits
    from temp_scripts.activation_baseline import (WindowData, fit_probe, metrics, predict_model,
        sample_weights, summarize_prediction)
    root = output_path(root)
    root_cfg = _json(root / "configuration.json")
    layer = root_cfg.get("layer", "h1")
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as source:
        arrays = {k: source[k].copy() for k in source.files}
    if set(arrays["split"]) != {"train", "validation"}:
        raise ValueError("Time augmentation needs only the original development rows")
    rows = _csv(root / "selected_frames.csv")
    for row in rows:
        row["time_s"] = float(row["time_s"])
        row["state_index"], row["object_id"] = int(row["state_index"]), int(row["object_id"])
    data = WindowData(rows, arrays.get("h1", arrays["h"]), arrays.get("h2", arrays["h"]), arrays["y"], [])
    if not np.array_equal(data.splits, arrays["split"]) or not np.array_equal(data.groups, arrays["trajectory_id"]):
        raise ValueError("Development frame identities disagree with saved arrays")
    train, validation = data.splits == "train", data.splits == "validation"
    weights, validation_weights = sample_weights(data.groups[train]), sample_weights(data.groups[validation])
    outcomes = []
    for identifier in experiment_ids:
        experiment = output_path(root / identifier)
        if experiment.parent != root or not (experiment / "summary.json").exists():
            raise ValueError("Use a completed immediate experiment directory: " + str(identifier))
        manifest_path = experiment / "augmentation_manifest.json"
        if manifest_path.exists():
            manifest = _json(manifest_path)
            if manifest.get("kind") != "time_plus_fixed_top_coordinates":
                raise ValueError("An incompatible augmentation manifest already exists")
            current = joblib.load(experiment / "fitted_probes.joblib")
            if not all(key in current["models"] for key in manifest["added_model_keys"]):
                raise ValueError("Augmentation manifest and saved estimators disagree")
            outcomes.append({"experiment_id": identifier, "status": "already_augmented", **manifest})
            continue
        artifact = joblib.load(experiment / "fitted_probes.joblib")
        with np.load(experiment / "reconstruction.npz", allow_pickle=False) as saved:
            z = saved["z"].copy()
        transformed = {layer: artifact["transforms"][layer]["scaler"].transform(getattr(data, layer)),
                       "z": artifact["z_scaler"].transform(z)}
        spline = artifact["transforms"]["time_scaler"].transform(artifact["transforms"]["time_spline"].transform(data.time))
        files = ("fitted_probes.joblib", "probe_predictions.npz", "probe_metrics.csv", "probe_by_trajectory.csv",
                 "probe_by_object.csv", "probe_model_selection.csv")
        original_hashes = {name: _hash(experiment / name) for name in files}
        with np.load(experiment / "probe_predictions.npz", allow_pickle=False) as source:
            predictions = {k: source[k].copy() for k in source.files}
        metric_rows = _csv(experiment / "probe_metrics.csv")
        trajectory_rows = _csv(experiment / "probe_by_trajectory.csv")
        object_rows = _csv(experiment / "probe_by_object.csv")
        selection_rows = _csv(experiment / "probe_model_selection.csv")
        original_model_keys, added = set(artifact["models"]), []
        with threadpool_limits(limits=1):
            for column, target in enumerate(TARGETS):
                y, classification = data.y[:, column], target == "any_contact"
                for prefix in (layer, "z"):
                    for k in (1, 2, 4):
                        old_name = f"{prefix}_top{k}"
                        name = old_name + "_time"
                        original_key = f"{target}__{old_name}__seed0"
                        key = f"{target}__{name}__seed0"
                        if key in artifact["models"] or key in predictions:
                            raise ValueError("Time probe exists without its completed augmentation manifest: " + key)
                        coordinates = list(artifact["models"][original_key]["coordinates"])
                        values = np.column_stack([spline, transformed[prefix][:, coordinates]])
                        candidates, target_scaler, single_class = fit_probe(values, y, train, validation,
                            weights, "linear", classification, quick=False)
                        scored = []
                        for parameter, model, messages in candidates:
                            p = predict_model(model, values[validation], classification)
                            if target_scaler is not None:
                                p = target_scaler.inverse_transform(p[:, None]).ravel()
                            score = metrics(y[validation], p, classification, validation_weights)["log_loss" if classification else "mae"]
                            selection_rows.append({"target": target, "model": name, "parameter": float(parameter),
                                "validation_score": score, "warnings": " | ".join(messages)})
                            scored.append((score, parameter, model, messages))
                        _, parameter, model, messages = min(scored, key=lambda item: item[0])
                        p = predict_model(model, values, classification)
                        if target_scaler is not None:
                            p = target_scaler.inverse_transform(p[:, None]).ravel()
                        if not np.isfinite(p).all():
                            raise ValueError("Nonfinite augmented time probe prediction")
                        predictions[key] = p
                        artifact["models"][key] = {"estimator": model, "target_scaler": target_scaler,
                            "coordinates": coordinates, "with_time": True, "single_class_train": single_class,
                            "parameter": parameter, "warnings": messages,
                            "input_order": "stored train-fitted time spline, then standardized selected coordinates",
                            "coordinate_source_model": original_key}
                        summary, trajectories, objects = summarize_prediction(data, target, name, 0, p)
                        metric_rows.extend(summary)
                        trajectory_rows.extend(trajectories)
                        object_rows.extend(objects)
                        added.append(key)
        if not original_model_keys.issubset(artifact["models"]):
            raise ValueError("An original estimator disappeared during augmentation")
        backup = experiment / "pre_time_top_augmentation"
        staging = experiment / "time_top_augmentation_staging"
        backup.mkdir(exist_ok=True)
        staging.mkdir(exist_ok=True)
        for name in files:
            if not (backup / name).exists():
                shutil.copy2(experiment / name, backup / name)
            elif _hash(backup / name) != original_hashes[name]:
                raise ValueError("Pre-existing augmentation backup does not match original artifacts")
        joblib.dump(artifact, staging / "fitted_probes.joblib")
        np.savez_compressed(staging / "probe_predictions.npz", **predictions)
        for filename, values in (("probe_metrics.csv", metric_rows), ("probe_by_trajectory.csv", trajectory_rows),
                                 ("probe_by_object.csv", object_rows), ("probe_model_selection.csv", selection_rows)):
            _write_csv(staging / filename, values)
        try:
            for name in files:
                (staging / name).replace(experiment / name)
        except BaseException:
            for name in files:
                shutil.copy2(backup / name, experiment / name)
            raise
        source_snapshot = experiment / "time_top_augmentation_source.py"
        shutil.copy2(Path(__file__), source_snapshot)
        manifest = {"kind": "time_plus_fixed_top_coordinates", "experiment_id": identifier,
            "created_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "added_probe_count": len(added), "original_probe_count": len(original_model_keys),
            "added_model_keys": added, "original_artifact_sha256": original_hashes,
            "augmented_artifact_sha256": {name: _hash(experiment / name) for name in files},
            "augmentation_source_sha256": _hash(source_snapshot), "augmentation_source": str(source_snapshot),
            "original_backup": str(backup), "coordinate_selection": "reuse original train-selected coordinates, no reselection",
            "time_basis": "reuse stored train-fitted SplineTransformer and StandardScaler",
            "probe_regularization_budget": [.1, 1., 10.],
            "input_order": "time spline basis first, selected standardized h/z coordinates second",
            "validation_used_for_probe_selection": True, "test_rows_used": 0,
            "scope": "candidate-only development time-control augmentation; existing original estimators/scores retained"}
        _dump(manifest_path, manifest)
        outcomes.append({"experiment_id": identifier, "status": "augmented", **manifest})
    return outcomes


def plot_selected_summary(root, config_names, holdout_root=None):
    """Curated figures for preselected configurations, raw neurons and Ridge.

    All development configurations remain in the complete CSV. This view never
    chooses a configuration or Ridge control from holdout scores. If a holdout
    root is supplied, its three original splits are shown as separate rows.
    Returns figure paths and writes the exact plotted values in a long CSV.
    """
    from matplotlib.figure import Figure
    from temp_scripts.plots import configure_style
    root = output_path(root)
    names = list(config_names)
    if not names or len(names) > 8 or len(set(names)) != len(names):
        raise ValueError("Choose between one and eight unique configuration names")
    folder = root / "analysis"
    folder.mkdir(exist_ok=True)
    root_cfg = _json(root / "configuration.json")
    layer = root_cfg.get("layer", "h1")
    ridge_selection_path = root / "controls" / "ridge_rotation_selection.json"
    selected_ridge = _json(ridge_selection_path).get("selected_by_target", {}) if ridge_selection_path.exists() else {}
    values, fidelities, raw_values = defaultdict(list), defaultdict(list), {}
    sources = []
    source_roots = [(root, "validation")]
    if holdout_root is not None:
        holdout_root = output_path(holdout_root)
        source_roots.append((holdout_root, "holdout"))
    models = ("z_top1", "z_top2", "z_top4", "z_linear")
    for source, scope in source_roots:
        observed_names = set()
        for experiment in _experiment_paths(source):
            summary = _json(experiment / "summary.json")
            name = _spec(summary).get("name", experiment.name.rsplit("_seed", 1)[0])
            if name not in names:
                continue
            observed_names.add(name)
            sources.append({"scope": scope, "experiment": str(experiment),
                            "probe_metrics_sha256": _hash(experiment / "probe_metrics.csv")})
            for row in _csv(experiment / "probe_metrics.csv"):
                if row["weighting"] != "trajectory" or (scope == "validation" and row["split"] != "validation"):
                    continue
                metric = _main_metric(row["target"])
                value = _number(row.get(metric))
                if value is None:
                    continue
                if row["model"] in models:
                    values[(row["split"], name, row["target"], row["model"])].append(value)
                elif row["model"] in tuple(layer + m[1:] for m in models):
                    alias = "z" + row["model"][len(layer):]
                    key = (row["split"], "raw_neurons", row["target"], alias)
                    if key in raw_values and abs(raw_values[key] - value) > 1e-8:
                        raise ValueError("Raw-neuron scores disagree across selected candidates")
                    raw_values[key] = value
            for row in summary["fidelity"]:
                if row.get("weighting") == "trajectory" and row.get("quantity") == "action_clipped":
                    if scope == "validation" and row["split"] != "validation":
                        continue
                    fidelities[(row["split"], name)].append(float(row["mae"]))
        missing = set(names) - observed_names
        if missing:
            raise ValueError(f"Selected configuration(s) missing from {scope}: {sorted(missing)}")
    for key, value in raw_values.items():
        values[key] = [value]
    ridge_sources = []
    for source, scope in source_roots:
        if scope == "validation":
            candidates = [source / "controls" / "ridge_rotation_validation_metrics.csv"]
        else:
            controls_folder = source / "controls"
            candidates = sorted(set([*controls_folder.glob("*metrics.csv"), *controls_folder.rglob("probe_metrics.csv")]))
            rotation_folder = source / "rotation_controls"
            candidates += [p for p in rotation_folder.rglob("probe_metrics.csv") if p not in candidates]
            candidates += [p for p in source.glob("*rotation*metrics.csv") if p not in candidates]
        ridge_unique = {}
        for csv_path in candidates:
            if not csv_path.exists():
                continue
            for row in _csv(csv_path):
                control_id = row.get("control_id") or Path(row.get("control_path") or csv_path.parent.name).name
                target = row.get("target")
                if target not in selected_ridge or control_id != selected_ridge[target]["control_id"]:
                    continue
                if row.get("weighting") != "trajectory" or row.get("model") not in models:
                    continue
                if scope == "validation" and row["split"] != "validation":
                    continue
                value = _number(row.get(_main_metric(target)))
                if value is not None:
                    key = (row["split"], "ridge_rotation_selected_per_target", target, row["model"])
                    if key in ridge_unique and abs(ridge_unique[key] - value) > 1e-8:
                        raise ValueError("Conflicting frozen Ridge-control scores")
                    ridge_unique[key] = value
            ridge_sources.append({"scope": scope, "path": str(csv_path), "sha256": _hash(csv_path)})
        for key, value in ridge_unique.items():
            values[key] = [value]
    plotted_rows = []
    for (split, name, target, model), data in values.items():
        plotted_rows.append({"split": split, "configuration": name, "target": target, "model": model,
            "metric": _main_metric(target), "mean": float(np.mean(data)),
            "training_seed_sd": float(np.std(data, ddof=1)) if len(data) > 1 else None, "seed_count": len(data),
            "capacity": "full-dimensional dense reversible; no TopK" if name == "ridge_rotation_selected_per_target"
                else "original full layer; selected-coordinate probe" if name == "raw_neurons" else "TopK SAE"})
    _write_csv(folder / "selected_summary_metrics.csv", plotted_rows)
    _dump(folder / "selected_summary_sources.json", {"configuration_names": names, "sources": sources,
        "ridge_sources": ridge_sources,
        "ridge_selection_sha256": _hash(ridge_selection_path) if ridge_selection_path.exists() else None,
        "display_selection": "explicit representative development candidates, not all explored configurations",
        "error_bars": "training-seed sample SD; raw and deterministic Ridge controls have no seed error bar",
        "holdout_selection": "configuration names and target-wise Ridge IDs inherited from development; no holdout selection"})
    lookup = {(r["split"], r["configuration"], r["target"], r["model"]): r for r in plotted_rows}
    configure_style()
    colors = {name: color for name, color in zip(names, _colors(names))}
    colors.update(raw_neurons="#151b27", ridge_rotation_selected_per_target="#a51f65")
    labels = {name: name for name in names}
    labels.update(raw_neurons="原始神经元", ridge_rotation_selected_per_target="Ridge监督旋转（各目标预选；64维稠密）")
    split_titles = {"validation": "开发集 validation", "test_trajectory": "同物体新轨迹",
        "test_object_policy_seen": "policy-seen 留出物体", "test_object_policy_unseen": "policy-unseen 物体"}
    contexts = [("validation", ["validation"])]
    if holdout_root is not None:
        contexts.append(("holdout", ["test_trajectory", "test_object_policy_seen", "test_object_policy_unseen"]))
    paths = []
    for context, splits in contexts:
        fig = Figure(figsize=(16, 5 * len(splits) + 1.3), layout="constrained")
        axes = np.array(fig.subplots(len(splits), 3, squeeze=False))
        handles, legend_labels = {}, {}
        for r, split in enumerate(splits):
            for c, (target, metric, label, scale) in enumerate(zip(TARGETS, METRICS, TARGET_LABELS, SCALES)):
                ax = axes[r, c]
                for name in names + ["raw_neurons", "ridge_rotation_selected_per_target"]:
                    cells = [lookup.get((split, name, target, model)) for model in models]
                    if any(cell is None for cell in cells):
                        continue
                    style = "s--" if name == "raw_neurons" else "D:" if name == "ridge_rotation_selected_per_target" else "o-"
                    line = ax.errorbar(range(4), [cell["mean"] * scale for cell in cells],
                        yerr=[(cell["training_seed_sd"] or 0.) * scale for cell in cells], fmt=style,
                        color=colors[name], lw=2 if name.startswith("ridge") else 1.5, ms=4, capsize=2, label=labels[name])
                    handles[name], legend_labels[name] = line, labels[name]
                ax.set(xticks=range(4), xticklabels=["1", "2", "4", "全部"], xlabel="train选定的固定坐标数",
                    ylabel=label, title=f"{split_titles.get(split, split)} · {label}")
                ax.grid(True)
        fig.suptitle("状态辅助 SAE：代表候选与强基线\n候选来自开发集；误差线为训练种子标准差，非数据置信区间", fontsize=12)
        fig.legend(list(handles.values()), list(legend_labels.values()), loc="outside lower center", ncol=2,
                   fontsize=8, frameon=False)
        target_path = folder / f"selected_{context}_concentration.png"
        fig.savefig(target_path, dpi=160)
        paths.append(target_path)
        fig = Figure(figsize=(16, 5 * len(splits) + 1.3), layout="constrained")
        axes = np.array(fig.subplots(len(splits), 3, squeeze=False))
        for r, split in enumerate(splits):
            for c, (target, metric, label, scale) in enumerate(zip(TARGETS, METRICS, TARGET_LABELS, SCALES)):
                ax = axes[r, c]
                for name in names:
                    cell = lookup.get((split, name, target, "z_top4"))
                    action = fidelities.get((split, name), [])
                    if cell is None or not action:
                        continue
                    x = float(np.mean(action))
                    dx = float(np.std(action, ddof=1)) if len(action) > 1 else 0.
                    ax.errorbar(x, cell["mean"] * scale, xerr=dx, yerr=(cell["training_seed_sd"] or 0.) * scale,
                        fmt="o", color=colors[name], ms=6, capsize=3, label=labels[name])
                for name, ls in (("raw_neurons", "--"), ("ridge_rotation_selected_per_target", ":")):
                    cell = lookup.get((split, name, target, "z_top4"))
                    if cell is not None:
                        ax.axhline(cell["mean"] * scale, ls=ls, color=colors[name], lw=1.7, label=labels[name])
                ax.set(xlabel="裁剪动作 MAE ↓", ylabel="Top4 · " + label, title=split_titles.get(split, split) + " · " + label)
                ax.grid(True)
        handles, labels_ = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="outside lower center", ncol=2, fontsize=8, frameon=False)
        fig.suptitle("固定4特征的状态读出与动作保持\n水平线仅标示状态预测基线；监督旋转为64维稠密可逆，容量不同", fontsize=12)
        target_path = folder / f"selected_{context}_tradeoff.png"
        fig.savefig(target_path, dpi=160)
        paths.append(target_path)
    return paths
