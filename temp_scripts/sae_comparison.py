"""Compare completed SAE studies without reading source or test trajectories."""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from temp_scripts.project_paths import OUTPUT_ROOT, output_path


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(runs, destination=None):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or OUTPUT_ROOT / "sae_comparison" / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    observations, duplicate_checks, baseline = [], [], []
    identities, seen, input_identity = [], {}, None
    for run in map(lambda p: Path(p).resolve(), runs):
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        cfg = summary["configuration"]
        if summary["status"] != "development_run_completed" or cfg["quick_check"]:
            raise ValueError("Only completed non-quick development runs can be compared: " + str(run))
        if summary["test_numeric_archives_loaded"] != 0 or not summary["policy_weights_unchanged"]:
            raise ValueError("Study must leave policy frozen and test numerical archives unopened")
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        identity = {"layer": cfg["layer"], "event": cfg["event"], "radius": cfg["radius"],
                    "weighting": cfg["weighting"], "source_files": provenance["source_files_sha256"],
                    "numeric_archives": provenance["numeric_archives_sha256"]}
        if input_identity is None:
            input_identity = identity
        if identity != input_identity:
            raise ValueError("Incompatible input populations, layer, windows or weighting")
        identities.append({"run": str(run), "configuration": cfg, "provenance": str(run / "provenance.json")})
        for experiment in summary["experiments"]:
            spec = dict(experiment["configuration"])
            spec.setdefault("preprocessing", "coordinate")
            config_id = json.dumps(spec, sort_keys=True)
            key = (config_id, experiment["seed"])
            path = run / experiment["experiment_id"]
            selected = [r for r in experiment["fidelity"] if r["split"] == "validation" and r["weighting"] == "trajectory"]
            fidelity = {r["quantity"]: r for r in selected}
            row = {"configuration_id": config_id, **spec, "seed": experiment["seed"], "layer": cfg["layer"],
                   "best_epoch": experiment["best_epoch"], "epochs_completed": experiment["epochs_completed"],
                   "early_stopped": experiment["early_stopped"],
                   "validation_mse_standardized": fidelity["activation_standardized"]["mse"],
                   "validation_nmse_raw": fidelity["activation_raw"]["nmse_vs_train_mean"],
                   "validation_action_mean_mae": fidelity["action_mean"]["mae"],
                   "validation_action_clipped_mae": fidelity["action_clipped"]["mae"],
                   "validation_action_clipped_max": fidelity["action_clipped"]["max_abs"],
                   "train_unused_features": experiment["health"]["train"]["never_active_features"],
                   "validation_mean_l0": experiment["health"]["validation"]["mean_l0"],
                   "fitting_seconds": experiment["fitting_seconds"], "experiment_path": str(path)}
            probe_path = path / "probe_metrics.csv"
            if probe_path.exists():
                for probe in read_rows(probe_path):
                    if probe["split"] != "validation" or probe["weighting"] != "trajectory":
                        continue
                    name, target = probe["model"], probe["target"]
                    metric = "average_precision" if target == "any_contact" else "mae"
                    if name.startswith("z_"):
                        row[f"{name}_{target}_{metric}"] = float(probe[metric]) if probe[metric] else None
                        if target == "any_contact":
                            row[f"{name}_any_contact_log_loss"] = float(probe["log_loss"])
                    elif name in ("h1_linear", "h1_top4", "h1_pca4", "h2_linear", "h2_top4", "h2_pca4"):
                        baseline.append({"model": name, "target": target, "metric": metric,
                                         "value": float(probe[metric]), "run": str(run)})
            if key in seen:
                first = seen[key]
                numeric = [k for k, v in row.items() if isinstance(v, (int, float)) and k not in ("fitting_seconds",)]
                max_diff = max(abs(row[k] - first[k]) for k in numeric if isinstance(first.get(k), (int, float)))
                duplicate_checks.append({"configuration_id": config_id, "seed": row["seed"],
                    "first_path": first["experiment_path"], "duplicate_path": str(path), "max_metric_difference": max_diff})
                if max_diff > 1e-6:
                    raise ValueError("Same config/seed produced different results; do not silently deduplicate")
                continue
            seen[key] = row
            observations.append(row)
    if not observations:
        raise ValueError("No completed experiments")
    baseline_values = {}
    for b in baseline:
        key = (b["model"], b["target"], b["metric"])
        if key in baseline_values and abs(baseline_values[key] - b["value"]) > 1e-10:
            raise ValueError("Recomputed raw baseline differs across study runs")
        baseline_values[key] = b["value"]
    grouped = defaultdict(list)
    for row in observations:
        grouped[row["configuration_id"]].append(row)
    aggregates = []
    for config_id, rows in grouped.items():
        aggregate = {"configuration_id": config_id, **json.loads(config_id), "seeds": ",".join(str(r["seed"]) for r in rows),
                     "seed_count": len(rows)}
        for name in rows[0]:
            values = [r.get(name) for r in rows]
            if name.startswith(("validation_", "z_")) and all(isinstance(v, (float, int)) for v in values):
                aggregate[name + "_mean"] = float(np.mean(values))
                aggregate[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                aggregate[name + "_min"] = float(min(values))
                aggregate[name + "_max"] = float(max(values))
        aggregates.append(aggregate)
    aggregates.sort(key=lambda r: r["validation_action_clipped_mae_mean"])
    write_rows(folder / "experiments.csv", observations)
    write_rows(folder / "configurations.csv", aggregates)
    write_rows(folder / "raw_baselines.csv", [{"model": k[0], "target": k[1], "metric": k[2], "value": v}
                                              for k, v in baseline_values.items()])
    (folder / "sources.json").write_text(json.dumps({"runs": identities, "input_identity": input_identity,
            "comparison_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "duplicate_checks": duplicate_checks, "uncertainty": "sample standard deviation across training seeds; not a confidence interval",
            "scope": "adaptive validation-set exploration; no test score or semantic/causal claim"}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# SAE 参数探索汇总", "", "仅开发集；按相同配置的训练种子汇总。±为种子样本标准差，不是数据置信区间。",
             "预处理不同的标准化MSE不可直接比较；跨预处理主要比较原尺度NMSE和动作误差。",
             "同配置同种子的重复运行核对后去重，原始基线也不把重复拟合作为独立重复。", "",
             "|宽度|K|预处理|学习率|最大轮数|种子数|动作MAE↓|原尺度NMSE↓|4特征距离MAE(m)↓|4特征角度MAE(rad)↓|4特征接触AP↑|",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def format_metric(row, name):
        value = row.get(name + "_mean")
        std = row.get(name + "_std")
        return "—" if value is None else (f"{value:.5f} ± {std:.5f}" if std is not None else f"{value:.5f}")
    for row in aggregates:
        cells = [str(row[k]) for k in ("width", "top_k", "preprocessing", "learning_rate", "max_epochs", "seed_count")]
        cells += [format_metric(row, k) for k in ("validation_action_clipped_mae", "validation_nmse_raw",
                    "z_top4_distance_m_mae", "z_top4_opening_rad_mae", "z_top4_any_contact_average_precision")]
        lines.append("|" + "|".join(cells) + "|")
    lines += ["", "## 证据位置", "", "- experiments.csv：逐配置/种子，含原始实验绝对路径。",
              "- configurations.csv：各项均值、种子标准差及范围。", "- raw_baselines.csv：同批原神经元/PCA基线。",
              "- sources.json：固定数据/代码来源及重复运行核对。", "", "本表按动作MAE排序，不表示语义组织或闭环行为的综合优劣。"]
    lines += ["", "步骤6入口同时生成pca_fidelity.csv及pca_sources.json：PCA仅在train拟合，动作误差在同批validation评价。",
              "PCA的r个固定方向与SAE从宽字典中逐样本选择K个方向具有不同容量约束，不能视为完全同模型容量。"]
    (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder, aggregates
