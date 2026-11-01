"""Numerical checks for saved h2/h1/features -> original actor outputs."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from temp_scripts.actor_model import load_actor, parameter_fingerprint, replay
from temp_scripts.data_access import ALL_ROLES
from temp_scripts.project_paths import PROJECT_ROOT, source_path, OUTPUT_ROOT, output_path

ACTOR_HOME = OUTPUT_ROOT / "actor_verification"
COMPARISONS = {
    "A_mu": ("action_mean_raw", "A: 保存h2 -> 动作均值"),
    "A_action": ("actions", "A: 保存h2 -> 实际动作"),
    "B_h2": ("activations_l2", "B: 保存h1 -> 第二层激活"),
    "B_mu": ("action_mean_raw", "B: 保存h1 -> 动作均值"),
    "B_action": ("actions", "B: 保存h1 -> 实际动作"),
    "C_h1": ("activations", "C: 保存融合特征 -> 第一层激活"),
    "C_h2": ("activations_l2", "C: 保存融合特征 -> 第二层激活"),
    "C_mu": ("action_mean_raw", "C: 保存融合特征 -> 动作均值"),
    "C_action": ("actions", "C: 保存融合特征 -> 实际动作"),
    "D_action": ("actions", "D: 保存动作均值 -> 原始裁剪核验"),
}


def json_file(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_table(path, rows, fields=None):
    with output_path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def compare_values(prediction, reference, atol, rtol):
    pred, ref = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape or ref.ndim != 2:
        raise ValueError("Prediction/reference shapes differ or are not [steps,dimensions]")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite comparison values")
    delta = pred - ref
    error = np.abs(delta)
    tolerance = atol + rtol * np.abs(ref)
    failed = error > tolerance
    t, k = np.unravel_index(int(np.argmax(error)), error.shape)
    return {"delta": delta, "error": error, "failed": failed, "tolerance": tolerance,
            "mae": float(error.mean()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs_error": float(error[t, k]), "values": int(error.size),
            "failed_values": int(failed.sum()), "failed_steps": int(failed.any(axis=1).sum()),
            "passed": bool(not failed.any()), "worst_action_index": int(t), "worst_dimension": int(k),
            "worst_reference": float(ref[t, k]), "worst_prediction": float(pred[t, k])}


class Accumulator:
    def __init__(self, width):
        self.steps = self.values = self.failed_values = self.failed_steps = 0
        self.abs_sum = self.square_sum = 0.
        self.maximum = -1.
        self.worst = None
        self.trajectory_mae = []
        self.abs_by_dim, self.max_by_dim, self.failed_by_dim = np.zeros(width), np.zeros(width), np.zeros(width, dtype=np.int64)

    def add(self, result, trajectory_id):
        error = result["error"]
        self.steps += error.shape[0]
        self.values += error.size
        self.abs_sum += float(error.sum())
        self.square_sum += float(np.square(result["delta"]).sum())
        self.failed_values += result["failed_values"]
        self.failed_steps += result["failed_steps"]
        self.trajectory_mae.append(result["mae"])
        self.abs_by_dim += error.sum(axis=0)
        self.max_by_dim = np.maximum(self.max_by_dim, error.max(axis=0))
        self.failed_by_dim += result["failed"].sum(axis=0)
        if result["max_abs_error"] > self.maximum:
            self.maximum = result["max_abs_error"]
            self.worst = {"trajectory_id": trajectory_id, **{key: result[key] for key in ["worst_action_index", "worst_dimension", "worst_reference", "worst_prediction"]}}

    def summary(self, comparison, split="all"):
        return {"comparison": comparison, "split": split, "trajectories": len(self.trajectory_mae), "steps": self.steps,
                "values": self.values, "mae": self.abs_sum / self.values, "rmse": float(np.sqrt(self.square_sum / self.values)),
                "macro_trajectory_mae": float(np.mean(self.trajectory_mae)), "max_abs_error": self.maximum,
                "failed_values": self.failed_values, "failed_steps": self.failed_steps,
                "passed": self.failed_values == 0, **self.worst}


def verify_actor(repo, checkpoint, *, records=None, device="cpu", batch_size=256, atol=1e-5, rtol=1e-4,
                 destination=None, figures=True, progress=None, publish=True):
    if min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = repo.filter(roles=ALL_ROLES) if records is None else records
    if not records:
        raise ValueError("No trajectories selected")
    expected = {repo.source_meta[r["trajectory_id"]]["checkpoint_sha256"] for r in records}
    if len(expected) != 1:
        raise ValueError("Selected trajectories use different policy checkpoints")
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select --device cpu")
        torch.backends.cuda.matmul.allow_tf32 = False
    actor, model_info = load_actor(checkpoint, expected.pop(), device)
    before = parameter_fingerprint(actor)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or ACTOR_HOME / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "replayed").mkdir()
    (folder / "model").mkdir()
    torch.save({k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}, folder / "model" / "frozen_actor_state.pt")
    json_file(folder / "model" / "model_spec.json", model_info)
    config = {"atol": atol, "rtol": rtol, "comparison_rule": "abs(replayed-reference) <= atol + rtol*abs(reference)",
              "dtype": "float32 forward, float64 error arithmetic", "device": str(device), "batch_size": batch_size,
              "cpu_threads": torch.get_num_threads(), "hook_comparison_rule": "exact equality of full forward with/without observing hooks",
              "scope": "Saved fused features -> actor tail only; no point cloud extraction or simulation replay",
              "torch_version": torch.__version__, "numpy_version": np.__version__, "source_index": str(repo.root),
              "fitting_performed": False, "fixed_data_splits_preserved": True, "checkpoint": model_info}
    json_file(folder / "configuration.json", config)  # Freeze acceptance criteria before replaying trajectories.
    global_metrics, split_metrics, trajectory_metrics, hook_checks, terminals = {}, {}, [], [], []
    plot_bank = {name: [[], []] for name in ["C_h1", "C_h2", "C_mu", "C_action"]}
    mismatch_rows, mismatch_total = [], 0
    source_provenance = []
    columns = ["trajectory_id", "object_id", "split", "trajectory_success", "action_index", "time_s", "distance_m", "opening_rad", "strict_contact", "event_ids", "triggered_event_ids", "raw_action_clipped_dimensions"]
    columns += [metric + suffix for metric in COMPARISONS for suffix in ["_mae", "_max_abs", "_failed_dimensions"]]
    with (folder / "step_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, meta in enumerate(records):
            traj = repo.load(meta["trajectory_id"])
            predictions, hooks = replay(actor, traj.arrays, batch_size)
            hook_checks.append({"trajectory_id": traj.tid, **hooks})
            per_metric = {}
            for name, (key, _) in COMPARISONS.items():
                result = compare_values(predictions[name], traj.arrays[key], atol, rtol)
                per_metric[name] = result
                global_metrics.setdefault(name, Accumulator(predictions[name].shape[1])).add(result, traj.tid)
                split_metrics.setdefault((meta["split"], name), Accumulator(predictions[name].shape[1])).add(result, traj.tid)
                trajectory_metrics.append({"trajectory_id": traj.tid, "object_id": int(meta["object_id"]), "split": meta["split"], "comparison": name,
                                           **{key: value for key, value in result.items() if key not in ["delta", "error", "failed", "tolerance"]}})
                bad = np.argwhere(result["failed"])
                mismatch_total += len(bad)
                for t, k in bad[:max(0, 5000 - len(mismatch_rows))]:
                    mismatch_rows.append({"trajectory_id": traj.tid, "split": meta["split"], "action_index": int(t), "comparison": name,
                                          "dimension": int(k), "reference": float(traj.arrays[key][t, k]), "replayed": float(predictions[name][t, k]),
                                          "abs_error": float(result["error"][t, k]), "allowed_error": float(result["tolerance"][t, k])})
                if name in plot_bank:
                    plot_bank[name][0].append(traj.arrays[key].reshape(-1))
                    plot_bank[name][1].append(predictions[name].reshape(-1))
            event_at, event_trigger = {}, {}
            for e in traj.events:
                if e["state_index"] is not None:
                    event_at.setdefault(e["state_index"], []).append(e["event_id"])
                if e["trigger_action_index"] is not None:
                    event_trigger.setdefault(e["trigger_action_index"], []).append(e["event_id"])
            for t in range(traj.n):
                row = {"trajectory_id": traj.tid, "object_id": int(meta["object_id"]), "split": meta["split"], "trajectory_success": meta["trajectory_success"],
                       "action_index": t, "time_s": t * traj.dt, "distance_m": float(traj.state("palm_handle_distance")[t]),
                       "opening_rad": float(traj.opening[t]), "strict_contact": bool(traj.state("contact")[t]),
                       "event_ids": "|".join(event_at.get(t, [])), "triggered_event_ids": "|".join(event_trigger.get(t, [])),
                       "raw_action_clipped_dimensions": int(np.count_nonzero(np.abs(traj.arrays["actions"][t] - traj.arrays["action_mean_raw"][t]) > 1e-6))}
                for name, result in per_metric.items():
                    row[name + "_mae"] = float(result["error"][t].mean())
                    row[name + "_max_abs"] = float(result["error"][t].max())
                    row[name + "_failed_dimensions"] = int(result["failed"][t].sum())
                writer.writerow(row)
            payload = {**predictions, "reference_h1": traj.arrays["activations"], "reference_h2": traj.arrays["activations_l2"],
                       "reference_mu": traj.arrays["action_mean_raw"], "reference_action": traj.arrays["actions"],
                       "action_index": np.arange(traj.n), "time_s": traj.time[:-1], "terminal_state_index": np.asarray(traj.n),
                       "distance_m": traj.state("palm_handle_distance")[:-1], "opening_rad": traj.opening[:-1],
                       "strict_contact": traj.state("contact")[:-1],
                       "metadata_json": np.asarray(json.dumps({"trajectory": meta, "provenance": traj.provenance}, ensure_ascii=False))}
            np.savez_compressed(folder / "replayed" / (traj.tid + ".npz"), **payload)
            terminals.append({"trajectory_id": traj.tid, "split": meta["split"], "state_index": traj.n, "time_s": float(traj.time[-1]),
                              "policy_available": False, "action_index": None, "comparison_status": "not_applicable_no_policy_response"})
            source_provenance.append({"trajectory_id": traj.tid, "source_numeric_archive": meta["numeric_archive"],
                                      "source_npz": meta["source_npz"], "source_sha256": meta["source_sha256"], "split": meta["split"], "decisions": traj.n})
            if progress and ((i + 1) % 25 == 0 or i + 1 == len(records)):
                progress("Verified %d/%d trajectories" % (i + 1, len(records)))
    metrics = [global_metrics[name].summary(name) for name in COMPARISONS]
    by_split = [value.summary(name, split) for (split, name), value in split_metrics.items()]
    write_table(folder / "comparison_summary.csv", metrics)
    write_table(folder / "by_split_summary.csv", by_split)
    write_table(folder / "trajectory_errors.csv", trajectory_metrics)
    dimensions = []
    for name, acc in global_metrics.items():
        for k in range(len(acc.abs_by_dim)):
            dimensions.append({"comparison": name, "dimension": k, "steps": acc.steps, "mae": float(acc.abs_by_dim[k] / acc.steps),
                               "max_abs_error": float(acc.max_by_dim[k]), "failed_values": int(acc.failed_by_dim[k])})
    write_table(folder / "dimension_errors.csv", dimensions)
    write_table(folder / "terminal_states.csv", terminals)
    write_table(folder / "mismatches.csv", mismatch_rows, ["trajectory_id", "split", "action_index", "comparison", "dimension", "reference", "replayed", "abs_error", "allowed_error"])
    after = parameter_fingerprint(actor)
    hook_ok = all(h["hooks_removed"] and h["hook_output_unchanged_exact"] for h in hook_checks)
    ok = all(m["passed"] for m in metrics) and before == after and hook_ok
    summary = {"status": "passed" if ok else "failed", "trajectories": len(records), "decision_steps": sum(int(r["decisions"]) for r in records),
               "terminal_states_not_replayed": len(terminals), "comparisons": metrics, "by_split": by_split,
               "failed_values": mismatch_total, "mismatch_rows_exported": len(mismatch_rows), "mismatches_truncated": mismatch_total > len(mismatch_rows),
               "hooks_neutral_and_removed": hook_ok, "weights_unchanged": before == after,
               "weights_sha256_before": before, "weights_sha256_after": after, "configuration": config,
               "interpretation": "Numerical fidelity of restored actor tail only; no learned feature semantics, intervention or closed-loop behavior claim."}
    json_file(folder / "summary.json", summary)
    json_file(folder / "hook_checks.json", hook_checks)
    json_file(folder / "sources.json", source_provenance)
    if figures:
        from temp_scripts.actor_verification_plots import export_figures
        export_figures(folder, summary, plot_bank, repo, records)
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    for name in ["actor_model.py", "actor_verification.py", "actor_verification_plots.py", "actor_verification_viewer.py", "3_verify_actor_forward.py", "data_access.py", "project_paths.py", "plots.py", "__init__.py"]:
        source = source_path(name)
        target = snapshot / source.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    table = "\n".join("| %s | %.3g | %.3g | %d | %s |" % (m["comparison"], m["mae"], m["max_abs_error"], m["failed_values"], m["passed"]) for m in metrics)
    (folder / "README.md").write_text("# Actor 前向复现结果\n\n状态：**%s**；%d 条轨迹、%d 个动作步。另有 %d 个终止状态没有新激活，未参加前向复现。\n\n固定容差：abs(error) <= %.3g + %.3g * abs(reference)。\n\n| 比较 | MAE | 最大绝对误差 | 超差数值 | 通过 |\n| --- | ---: | ---: | ---: | --- |\n%s\n\n- [全局误差图](figures/01_global_errors.png)\n- [重算与记录的散点图](figures/02_replayed_vs_recorded.png)\n- [逐轨迹结果](trajectory_errors.csv)\n- [逐步误差（可按轨迹ID和动作索引关联信息总表）](step_errors.csv)\n- [逐维误差](dimension_errors.csv)\n- [超差位置](mismatches.csv)\n- [终止状态排除记录](terminal_states.csv)\n- [原权重及网络规格](model/model_spec.json)\n\nreplayed/ 中保存每条轨迹的日志参照与重算数组；summary.json 保存全部阈值、版本、来源和总体结论，hook_checks.json 记录每次挂钩数量及中立性检查。全局数值统计使用全部数据；仅散点图确定性抽样，最多每比较5000点。\n\nA 从已存h2开始，B 从已存h1开始，C 从已存融合特征开始；每次运算都向前执行。D 独立验证已存动作均值的裁剪。通过只表明融合特征之后的actor计算忠实，不包含PointNet、仿真或语义/因果证明。\n" % (summary["status"], summary["trajectories"], summary["decision_steps"], len(terminals), atol, rtol, table), encoding="utf-8")
    if publish:
        json_file(ACTOR_HOME / "latest.json", {"directory": str(folder), "status": summary["status"]})
        (ACTOR_HOME / "README.md").write_text("# 步骤3：冻结Actor前向复现\n\n最近完成的结果：[%s / %s](%s/README.md)。\n\n在项目根目录独立运行 3_verify_actor_forward.py，默认完成全部数值校验、导出图表后打开结果窗口。加 --no-gui 可只生成文件；加 --view-only 可直接查看最近一次结果。\n" % (folder.name, summary["status"], folder.name), encoding="utf-8")
    return folder, summary


def load_replayed(folder, trajectory_id):
    with np.load(Path(folder) / "replayed" / (trajectory_id + ".npz"), allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def latest_result():
    pointer = ACTOR_HOME / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("先运行 3_verify_actor_forward.py 完成一次前向复现。")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
