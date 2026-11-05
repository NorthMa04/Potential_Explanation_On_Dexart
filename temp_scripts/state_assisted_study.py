"""Resumable, staged development experiments for h-only state-assisted SAEs."""
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from temp_scripts.activation_baseline import (TARGETS, WindowData, collect_windows, dump,
    sha256, write_csv)
from temp_scripts.actor_model import load_actor, parameter_fingerprint
from temp_scripts.data_access import DataRepository
from temp_scripts.project_paths import OUTPUT_ROOT, PROJECT_ROOT, default_index, output_path
from temp_scripts.sae_model import SAEConfig
from temp_scripts.sae_training import (actor_outputs, check_original_actor, export_state_probes,
    reconstruction_metrics, selected_action_references, write_union)

HOME = OUTPUT_ROOT / "state_assisted"


def experiment_configs(stage):
    from temp_scripts.state_assisted_model import AssistedConfig
    base = AssistedConfig(sae=SAEConfig(width=64, top_k=16, preprocessing="global"))
    configs = []
    def add(name, **changes):
        configs.append(replace(base, name=name, **changes))
    if stage == "core":
        for weight in (0., .03, .1, .3, 1.):
            add("dense_l" + str(weight).replace(".", "p"), state_weight=weight)
    elif stage == "losses":
        for weight in (.1, .3):
            add(f"huber_l{weight}", state_weight=weight, regression_loss="huber")
            add(f"balanced_l{weight}", state_weight=weight, classification_loss="balanced")
            add(f"entropy_l{weight}", state_weight=weight, classification_loss="entropy")
        for index, target in enumerate(("distance", "opening", "contact")):
            for weight in (.1, .3):
                add(f"single_{target}_l{weight}", targets=(index,), state_weight=weight)
    elif stage == "structure":
        for mode in ("top4", "slots"):
            for weight in (.1, .3, 1.):
                add(f"{mode}_l{weight}", head_mode=mode, state_weight=weight)
        for k in (8, 32):
            for weight in (0., .1, .3):
                add(f"k{k}_l{weight}", sae=replace(base.sae, top_k=k), state_weight=weight)
        for weight in (0., .1, .3):
            add(f"coordinate_l{weight}", sae=replace(base.sae, preprocessing="coordinate"), state_weight=weight)
    elif stage == "fidelity":
        for state_weight in (0., .1, .3):
            for action_weight in (.1, 1.):
                add(f"action_s{state_weight}_a{action_weight}", state_weight=state_weight, action_weight=action_weight)
        for weight in (0., .1, .3):
            add(f"jointselect_l{weight}", state_weight=weight, selection="joint")
    elif stage == "refine":
        add("top4_l0.03", head_mode="top4", state_weight=.03)
        for weight in (.1, .3):
            for action in (.1, 1.):
                add(f"top4_s{weight}_a{action}", head_mode="top4", state_weight=weight, action_weight=action)
            for k in (8, 32):
                add(f"top4_k{k}_l{weight}", head_mode="top4", state_weight=weight,
                    sae=replace(base.sae, top_k=k))
        for weight in (0., .1):
            add(f"top4_w128_l{weight}", head_mode="top4", state_weight=weight,
                sae=replace(base.sae, width=128))
            add(f"top4_long_l{weight}", head_mode="top4", state_weight=weight,
                sae=replace(base.sae, max_epochs=1500, patience=100))
            add(f"top4_slowlr_l{weight}", head_mode="top4", state_weight=weight,
                sae=replace(base.sae, max_epochs=1500, patience=100, learning_rate=.0003))
        add("top4_huber_l0.1", head_mode="top4", state_weight=.1, regression_loss="huber")
    elif stage == "refine_slots":
        for weight in (.1, .3):
            for action in (.1, 1.):
                add(f"slots_s{weight}_a{action}", head_mode="slots", state_weight=weight, action_weight=action)
        for action in (0., .1):
            add(f"slots_s0.03_a{action}", head_mode="slots", state_weight=.03, action_weight=action)
    elif stage == "concentration":
        for mode in ("top1", "top2", "slots1", "slots2"):
            for weight in (.1, .3):
                add(f"{mode}_l{weight}", head_mode=mode, state_weight=weight)
    elif stage == "negative":
        for mode in ("dense", "top4"):
            for weight in (.1, .3):
                add(f"shuffle_{mode}_l{weight}", head_mode=mode, state_weight=weight, label_mode="trajectory_shuffle")
    else:
        raise ValueError("Unknown study stage: " + stage)
    return configs


def prepare_study(destination=None, index=None, checkpoint=None, layer="h1", event="sustained_opening", radius=5):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or HOME / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    repo = DataRepository(index or default_index())
    data = collect_windows(repo, event, radius)
    raw = data.h1 if layer == "h1" else data.h2
    expected = {repo.source_meta[tid]["checkpoint_sha256"] for tid in set(data.groups)}
    if len(expected) != 1:
        raise ValueError("Mixed policy checkpoints")
    actor, metadata = load_actor(checkpoint or PROJECT_ROOT.parents[2] / "dexart_assets.zip", expected.pop())
    mu, action = selected_action_references(repo, data)
    original = actor_outputs(actor, raw, layer)
    dump(folder / "actor_reference_check.json", check_original_actor(original, mu, action))
    (folder / "model").mkdir()
    dump(folder / "model" / "actor_metadata.json", metadata)
    torch.save({k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}, folder / "model" / "frozen_actor_state.pt")
    np.savez_compressed(folder / "selected_arrays.npz", h=raw, h1=data.h1, h2=data.h2, y=data.y,
        trajectory_id=data.groups, split=data.splits, state_index=np.array([r["state_index"] for r in data.rows]))
    np.savez_compressed(folder / "action_references.npz", saved_mean=mu, saved_action=action, original_mean_replayed=original)
    write_csv(folder / "selected_frames.csv", [{**r, **dict(zip(TARGETS, map(float, data.y[i])))} for i, r in enumerate(data.rows)])
    write_csv(folder / "window_coverage.csv", data.coverage)
    dump(folder / "configuration.json", {"layer": layer, "event": event, "radius": radius,
        "scope": "h-only current-state supervised representation; adaptive development exploration",
        "test_evaluation": False, "physical_simulation": False, "policy_updated": False,
        "sample_weighting": "trajectory", "python": platform.python_version(), "torch": str(torch.__version__),
        "source_index": str(repo.root), "created_beijing": stamp})
    dump(folder / "provenance.json", {"numeric_archives_sha256": {tid: sha256(repo.root / repo.by_id[tid]["numeric_archive"])
        for tid in dict.fromkeys(data.groups)}, "selected_arrays_sha256": sha256(folder / "selected_arrays.npz"),
        "test_numeric_archives_loaded": 0, "labels_used_for_sae_optimization": True, "policy_updated": False})
    return folder


def load_study(folder):
    folder = output_path(folder)
    with np.load(folder / "selected_arrays.npz", allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    with (folder / "selected_frames.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("state_index", "object_id"):
            row[key] = int(row[key])
        row["time_s"] = float(row["time_s"])
    if (not rows or arrays["h"].shape != (len(rows), 64) or arrays["y"].shape != (len(rows), 3)
            or any(arrays[key].shape != (len(rows),) for key in ("split", "trajectory_id", "state_index"))):
        raise ValueError("Study arrays and CSV rows must have matching dimensions")
    if set(arrays["split"]) != {"train", "validation"}:
        raise ValueError("Study numeric arrays must contain only train and validation")
    for key in ("split", "trajectory_id", "state_index"):
        expected = np.array([row[key] for row in rows])
        if not np.array_equal(arrays[key], expected):
            raise ValueError("Study NPZ/CSV row identity mismatch: " + key)
    if len({(r["trajectory_id"], r["state_index"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate study source states")
    if any(not np.isfinite(arrays[key]).all() for key in ("h", "y")):
        raise ValueError("Nonfinite study arrays")
    for key in ("h1", "h2"):
        if key in arrays and (arrays[key].shape != (len(rows), 64) or not np.isfinite(arrays[key]).all()):
            raise ValueError("Invalid stored activation layer: " + key)
    data = WindowData(rows, arrays.get("h1", arrays["h"]), arrays.get("h2", arrays["h"]), arrays["y"], [])
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    if not np.array_equal(arrays["h"], data.h1 if config["layer"] == "h1" else data.h2):
        raise ValueError("Stored primary activation does not match the study layer")
    if set(data.splits) != {"train", "validation"}:
        raise ValueError("Development study must contain only train and validation")
    from temp_scripts.actor_model import FrozenActor
    actor = FrozenActor()
    actor.load_state_dict(torch.load(folder / "model" / "frozen_actor_state.pt", weights_only=True))
    actor.requires_grad_(False).eval()
    return data, actor, config


def run_experiments(folder, configurations, seeds=(0, 1, 2), progress=print):
    from temp_scripts.state_assisted_model import fit_assisted
    folder = output_path(folder)
    data, actor, study = load_study(folder)
    layer = study["layer"]
    raw = data.h1 if layer == "h1" else data.h2
    original = actor_outputs(actor, raw, layer)
    before = parameter_fingerprint(actor)
    planned = {"configurations": [asdict(c) for c in configurations], "seeds": list(seeds),
        "development_only": True, "selection_rule_defined_before_each_run": True}
    batch_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    dump(folder / "plans" / (batch_id + ".json"), planned)
    snapshot = folder / "source_snapshots" / batch_id
    snapshot.mkdir(parents=True)
    for source in [PROJECT_ROOT / "7_run_state_assisted_sae.py", *sorted((PROJECT_ROOT / "temp_scripts").glob("*.py"))]:
        if source.exists():
            target = snapshot / source.relative_to(PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    dump(snapshot / "hashes.json", {p.relative_to(snapshot).as_posix(): sha256(p) for p in snapshot.rglob("*.py")})
    with threadpool_limits(limits=1):
        for config in configurations:
            config.validate()
            for seed in seeds:
                identifier = f"{config.name}_seed{seed}"
                experiment = folder / identifier
                if (experiment / "summary.json").exists():
                    old = json.loads((experiment / "summary.json").read_text(encoding="utf-8"))
                    old["configuration"].setdefault("label_mode", "true")
                    if json.dumps(old["configuration"], sort_keys=True) != json.dumps(asdict(config), sort_keys=True):
                        raise ValueError("Existing experiment has a different configuration: " + identifier)
                    progress("Already completed: " + identifier)
                    continue
                if experiment.exists():
                    raise ValueError("Incomplete experiment exists; inspect before resuming: " + str(experiment))
                experiment.mkdir()
                progress("Experiment: " + identifier)
                try:
                    model, scaler, normalized, z, reconstructed, fitted = fit_assisted(raw, data.y, data.splits,
                        data.groups, config, seed, experiment, actor=actor, layer=layer, progress=progress)
                    new_mu = actor_outputs(actor, reconstructed, layer)
                    fidelity, trajectories, objects, dimensions, usage, health = reconstruction_metrics(data,
                        raw, reconstructed, normalized, scaler, z, original, new_mu)
                    np.savez_compressed(experiment / "reconstruction.npz", z=z, reconstructed_h=reconstructed,
                        reconstructed_mean=new_mu, reconstructed_action=np.clip(new_mu, -1, 1))
                    write_union(experiment / "reconstruction_metrics.csv", fidelity)
                    write_union(experiment / "reconstruction_by_trajectory.csv", trajectories)
                    write_union(experiment / "reconstruction_by_object.csv", objects)
                    write_csv(experiment / "dimension_errors.csv", dimensions)
                    write_csv(experiment / "feature_usage.csv", usage)
                    probes = export_state_probes(experiment, data, z, layer, "trajectory")
                    if parameter_fingerprint(actor) != before or any(p.grad is not None or p.requires_grad for p in actor.parameters()):
                        raise RuntimeError("Frozen actor changed or received gradients")
                    result = {"experiment_id": identifier, "configuration": asdict(config), **fitted,
                        "seed": seed, "health": health, "fidelity": fidelity, "probes": probes,
                        "policy_weights_unchanged": True, "source_snapshot": str(snapshot),
                        "test_numeric_archives_loaded": 0, "physical_simulation": False}
                    dump(experiment / "summary.json", result)
                    progress(f"Finished {identifier}; epoch {result['best_epoch']}; " +
                        "action MAE=" + str(next(r["mae"] for r in fidelity if r["split"] == "validation" and r["weighting"] == "trajectory" and r["quantity"] == "action_clipped")))
                except BaseException as exc:
                    dump(experiment / "failure.json", {"error": str(exc), "configuration": asdict(config), "seed": seed})
                    raise
    return folder
