"""Auditable, project-local exports. Source datasets are never modified."""
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from temp_scripts.project_paths import OUTPUT_ROOT, output_path
from temp_scripts.plots import physical_figure, heatmap_figure, snapshot_figure, spatial_figure, comparison_figure


def json_file(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_file(path, rows):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_trajectory(repo, traj, run, selected_state=0, event=None, scope=None):
    folder = output_path(Path(run) / "trajectories" / traj.tid)
    folder.mkdir(parents=True, exist_ok=True)
    start, stop = scope or (0, traj.n)
    token = "states_%04d_%04d_at_%04d" % (start, stop, selected_state)
    dest = folder / token
    dest.mkdir(exist_ok=True)
    figures = {}
    for name, fn, args in [
            ("01_physical_states", physical_figure, (traj, repo.rules, selected_state, event, scope)),
            ("02_activations_actions", heatmap_figure, (traj, selected_state, event, scope)),
            ("03_selected_state_vectors", snapshot_figure, (traj, selected_state)),
            ("04_world_positions", spatial_figure, (traj, selected_state))]:
        fig = Figure(figsize=(12, 10) if name.startswith(("01", "02")) else (12, 8), layout="constrained")
        fn(fig, *args)
        fig.savefig(dest / (name + ".png"), dpi=160)
        fig.clear()
        figures[name] = name + ".png"
    steps, states = [], []
    a = traj.arrays
    for j in range(start, stop + 1):
        states.append({"trajectory_id": traj.tid, "split": traj.meta["split"], "state_index": j,
                       "time_s": j * traj.dt, "distance_m": float(traj.state("palm_handle_distance")[j]),
                       "q_rad": float(traj.state("object_qpos")[j, 0]), "opening_rad": float(traj.opening[j]),
                       "strict_contact": bool(traj.state("contact")[j]), "any_contact": bool(traj.state("loose_contact")[j]),
                       "progress": float(traj.state("progress")[j]), "policy_available": j < traj.n})
    for t in range(start, stop):
        step = {"trajectory_id": traj.tid, "split": traj.meta["split"], "action_index": t,
                "current_state_index": t, "post_state_index": t + 1,
                "distance_m": float(traj.state("palm_handle_distance")[t]),
                "post_distance_m": float(traj.state("palm_handle_distance")[t + 1]),
                "opening_rad": float(traj.opening[t]), "post_opening_rad": float(traj.opening[t + 1]),
                "opening_delta_rad": float(a["physical__opening_delta_rad"][t]),
                "contact": bool(traj.state("contact")[t]), "post_contact": bool(traj.state("contact")[t + 1]),
                "success_after": bool(a["success_after"][t]), "done": bool(a["done"][t]),
                "response_available": t + 1 < traj.n}
        for prefix, key in [("h1", "activations"), ("h2", "activations_l2"), ("mu_raw", "action_mean_raw"), ("action", "actions")]:
            step.update({"%s_%02d" % (prefix, k): float(value) for k, value in enumerate(a[key][t])})
        steps.append(step)
    csv_file(dest / "steps.csv", steps)
    csv_file(dest / "states.csv", states)
    csv_file(dest / "events.csv", traj.events)
    arrays = {key: value[start:stop + int(len(value) == traj.n + 1)] for key, value in a.items()}
    arrays["state_index"] = np.arange(start, stop + 1)
    arrays["source_row_index"] = np.arange(start, stop)
    np.savez_compressed(dest / "aligned_numeric.npz", **arrays)
    meta = {"trajectory": traj.meta, "source_provenance": traj.provenance,
            "source_index": str(repo.root), "event_rules": repo.rules,
            "selected_state": selected_state, "selected_event": event,
            "action_slice_left_closed_right_open": [start, stop], "state_slice_left_closed_right_open": [start, stop + 1],
            "figures": figures, "interpretation": "Offline recorded numerical states, not RGB/video or simulation; temporal correspondence is not causal contribution. Event rules are unreviewed v1 unless separately annotated."}
    json_file(dest / "source_and_alignment.json", meta)
    links = "\n".join("- [%s](%s)" % (name, filename) for name, filename in figures.items())
    (dest / "README.md").write_text("# %s\n\n动作行 [%d,%d)，状态 [%d,%d]，选择 s_%d。\n\n%s\n\nsteps.csv 含两层完整激活和原始/裁剪动作；states.csv 含片段末状态；来源和下标说明见 source_and_alignment.json。\n终止状态没有策略响应，不能填零当作新激活。\n" % (traj.tid, start, stop, start, stop, selected_state, links), encoding="utf-8")
    return dest


def export_comparison(repo, a, b, run, anchor="first_any_contact"):
    folder = output_path(Path(run) / "comparisons" / (a.tid + "__vs__" + b.tid))
    folder.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(12, 10), layout="constrained")
    comparison_figure(fig, a, b, anchor)
    fig.savefig(folder / (anchor + ".png"), dpi=160)
    fig.clear()
    json_file(folder / (anchor + ".json"), {"A": a.meta, "B": b.meta, "anchor": anchor,
              "A_state_anchor": 0 if anchor == "start" else a.event_anchor(anchor),
              "B_state_anchor": 0 if anchor == "start" else b.event_anchor(anchor),
              "source_index": str(repo.root), "matched_on_physical_state": False,
              "interpretation": "Descriptive examples only. Missing anchors are not synthesized; this is not a causal success/failure comparison."})
    return folder


def save_review(run, traj, event, judgement, note):
    if event is None:
        raise ValueError("先选择一个事件，再保存审阅意见。")
    path = OUTPUT_ROOT / "review_notes.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    record = {"time_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "trajectory_id": traj.tid, "split": traj.meta["split"], "event_id": event["event_id"],
              "event_type": event["event_type"], "state_index": event["state_index"],
              "automatic_status": event["status"], "human_judgement": judgement, "note": note,
              "rules_modified": False}
    data.append(record)
    json_file(path, data)
    csv_file(Path(run) / "review" / "review_notes.csv", data)
    json_file(Path(run) / "review" / "review_notes.json", data)
    return path


def record_run(repo, run, mode):
    json_file(Path(run) / "run_manifest.json", {"mode": mode, "source_index": str(repo.root),
              "output_directory": str(Path(run).resolve()), "source_rules_sha256": repo.provenance["rules_sha256"],
              "default_scope": "train + validation", "source_data_modified": False,
              "model_fitted": False, "event_rules_modified": False})
