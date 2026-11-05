"""Residual-preserving, norm-matched interventions on frozen actor readouts.

This is fixed-input offline analysis. It does not execute an action, restore a
simulator, infer a physical effect, or rank semantic causality by random-control
effect magnitude. States and directions are chosen without actor-effect scores.
"""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from temp_scripts.activation_baseline import TARGETS, dump, sample_weights, sha256, write_csv
from temp_scripts.actor_model import FrozenActor, parameter_fingerprint
from temp_scripts.data_access import DEV_ROLES
from temp_scripts.project_paths import output_path
from temp_scripts.sae_model import ActivationStandardizer
from temp_scripts.sae_training import actor_outputs, infer_sae
from temp_scripts.state_assisted_model import load_assisted_checkpoint


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_union(path, rows):
    write_csv(path, rows, list(dict.fromkeys(key for row in rows for key in row)) or ["status"])


def weighted_quantile(values, weights, quantile):
    """Inverse empirical CDF with train trajectory weights and stable sorting."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.ndim != 1 or weights.shape != values.shape or not len(values) or not 0 <= quantile <= 1:
        raise ValueError("Invalid weighted quantile inputs")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weighted quantile requires finite values and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")), len(values) - 1)
    return float(values[order[index]])


def residual_intervention(h, decoder_raw, z, modified_z):
    """h' = h + D_raw(z' - z): preserve h - reconstruction exactly."""
    h, decoder_raw, z, modified_z = map(np.asarray, (h, decoder_raw, z, modified_z))
    if decoder_raw.ndim != 2 or z.shape != modified_z.shape or z.shape[-1] != decoder_raw.shape[1]:
        raise ValueError("Code and decoder dimensions do not align")
    if h.shape[:-1] != z.shape[:-1] or h.shape[-1] != decoder_raw.shape[0]:
        raise ValueError("Activation and code rows do not align")
    return h + (modified_z - z) @ decoder_raw.T


def matched_control_deltas(delta_h, raw_coordinate, pca_direction, rng, random_pairs=5):
    """Five random unit draws and both signs = ten signed random controls."""
    delta_h, pca_direction = np.asarray(delta_h, float), np.asarray(pca_direction, float)
    if delta_h.ndim != 1 or pca_direction.shape != delta_h.shape or not np.isfinite(delta_h).all():
        raise ValueError("Control directions must match the finite activation vector")
    if not isinstance(raw_coordinate, (int, np.integer)) or not 0 <= raw_coordinate < len(delta_h):
        raise ValueError("Invalid raw neuron coordinate")
    norm = float(np.linalg.norm(delta_h))
    if not np.isfinite(pca_direction).all() or np.linalg.norm(pca_direction) < 1e-15:
        raise ValueError("PCA direction must be finite and nonzero")
    controls = []
    for pair in range(random_pairs):
        direction = rng.normal(size=len(delta_h))
        while np.linalg.norm(direction) < 1e-15:
            direction = rng.normal(size=len(delta_h))
        perturbation = direction / np.linalg.norm(direction) * norm
        controls.extend(((f"random_{pair}_plus", perturbation), (f"random_{pair}_minus", -perturbation)))
    raw = np.zeros_like(delta_h)
    raw[raw_coordinate] = norm
    pca = pca_direction / np.linalg.norm(pca_direction) * norm
    controls.extend((("raw_neuron_plus", raw), ("raw_neuron_minus", -raw),
        ("pca1_plus", pca), ("pca1_minus", -pca)))
    return controls


def evaluate_actor_perturbations(actor, h, perturbations, train_min, train_max, layer="h1"):
    """Evaluate all variants in one batch; return stats and actual delta arrays."""
    if any(parameter.requires_grad or parameter.grad is not None for parameter in actor.parameters()):
        raise ValueError("Actor must be frozen and have no parameter gradients")
    if layer not in ("h1", "h2"):
        raise ValueError("Unknown actor layer")
    h, train_min, train_max = map(lambda v: np.asarray(v, float), (h, train_min, train_max))
    if h.ndim != 1 or train_min.shape != h.shape or train_max.shape != h.shape:
        raise ValueError("Actor state and train ranges must align")
    names, deltas = zip(*perturbations)
    deltas = np.asarray(deltas, float)
    if deltas.shape != (len(names), len(h)) or not np.isfinite(deltas).all():
        raise ValueError("Invalid actor perturbation matrix")
    before_hash = parameter_fingerprint(actor)
    states = np.vstack((h, h[None, :] + deltas))
    mu = actor_outputs(actor, states, layer, batch_size=len(states))
    clipped = np.clip(mu, -1, 1)
    delta_mu, delta_clipped = mu[1:].astype(float) - mu[0], clipped[1:].astype(float) - clipped[0]
    before_saturated = np.abs(mu[0]) >= 1.
    rows = []
    for index, name in enumerate(names):
        modified = states[index + 1]
        after_saturated = np.abs(mu[index + 1]) >= 1.
        rows.append({"control": name, "delta_h_norm": float(np.linalg.norm(deltas[index])),
            "delta_mu_mae": float(np.abs(delta_mu[index]).mean()), "delta_mu_l2": float(np.linalg.norm(delta_mu[index])),
            "delta_clipped_mae": float(np.abs(delta_clipped[index]).mean()),
            "delta_clipped_l2": float(np.linalg.norm(delta_clipped[index])),
            "before_negative_coordinate_fraction": float((h < 0).mean()),
            "after_negative_coordinate_fraction": float((modified < 0).mean()),
            "before_outside_train_range_fraction": float(((h < train_min) | (h > train_max)).mean()),
            "after_outside_train_range_fraction": float(((modified < train_min) | (modified > train_max)).mean()),
            "before_saturated_action_fraction": float(before_saturated.mean()),
            "after_saturated_action_fraction": float(after_saturated.mean()),
            "saturated_action_fraction_change": float(after_saturated.mean() - before_saturated.mean()),
            "saturation_status_changed_fraction": float((before_saturated != after_saturated).mean()),
            **{f"delta_mu_{dimension:02d}": float(value) for dimension, value in enumerate(delta_mu[index])},
            **{f"delta_clipped_{dimension:02d}": float(value) for dimension, value in enumerate(delta_clipped[index])}})
    if parameter_fingerprint(actor) != before_hash or any(p.grad is not None for p in actor.parameters()):
        raise RuntimeError("Actor changed during offline intervention")
    return rows, deltas, delta_mu, delta_clipped


def _aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["experiment_id"], row["target"], row["operation"], row["control"], row["split"])].append(row)
    metrics = ("delta_h_norm", "delta_mu_mae", "delta_clipped_mae", "after_negative_coordinate_fraction",
        "after_outside_train_range_fraction", "before_saturated_action_fraction", "after_saturated_action_fraction")
    output = []
    for (experiment, target, operation, control, split), items in groups.items():
        trajectories = np.array([item["trajectory_id"] for item in items])
        weights = sample_weights(trajectories)
        output.append({"experiment_id": experiment, "target": target, "operation": operation,
            "control": control, "split": split, "states": len(items), "trajectories": len(set(trajectories)),
            "weighting": "equal trajectories, equal selected states within trajectory",
            **{metric: float(np.average([item[metric] for item in items], weights=weights)) for metric in metrics}})
    return output


def run_offline_interventions(study_root, holdout_root, experiment_ids, *, max_states_per_trajectory=3):
    """Run frozen-input readout interventions after completed holdout evaluation."""
    root, holdout = output_path(study_root), output_path(holdout_root)
    if not isinstance(max_states_per_trajectory, int) or isinstance(max_states_per_trajectory, bool) or max_states_per_trajectory < 1:
        raise ValueError("max_states_per_trajectory must be a positive integer")
    identifiers = list(experiment_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Use a nonempty unique candidate list")
    frozen = _json(root / "frozen_evaluation_plan.json")
    holdout_summary = _json(holdout / "summary.json")
    if holdout_summary.get("status") != "frozen_holdout_evaluation_completed":
        raise ValueError("Complete frozen holdout evaluation before interventions")
    if not set(identifiers).issubset(frozen["experiment_ids"]) or not set(identifiers).issubset(holdout_summary["experiment_ids"]):
        raise ValueError("Interventions may only use previously frozen and evaluated candidates")
    for identifier in identifiers:
        experiment = (root / identifier).resolve()
        if experiment.parent != root:
            raise ValueError("Candidate escaped the study directory")
        actual = {name: sha256(experiment / name) for name in ("best_model.pt", "fitted_probes.joblib", "summary.json")}
        if actual != holdout_summary["candidate_artifact_hashes"][identifier]:
            raise ValueError("Candidate changed after holdout evaluation")
    study = _json(root / "configuration.json")
    layer = study["layer"]
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as archive:
        development = {key: archive[key] for key in archive.files}
    with np.load(holdout / "selected_arrays.npz", allow_pickle=False) as archive:
        test = {key: archive[key] for key in archive.files}
    if not set(development["split"]).issubset(DEV_ROLES) or any(str(role) in DEV_ROLES for role in test["split"]):
        raise ValueError("Unexpected development or holdout rows")
    train = development["split"] == "train"
    raw_train = development.get(layer, development["h"])[train]
    raw_test = test.get(layer, test["h"])
    weights = sample_weights(development["trajectory_id"][train])
    global_scaler = ActivationStandardizer.fit(raw_train, weights, mode="global")
    centered = global_scaler.transform(raw_train)
    _, _, components = np.linalg.svd(centered * np.sqrt(weights[:, None]), full_matrices=False)
    pca_direction = components[0] * global_scaler.scale
    pca_direction /= np.linalg.norm(pca_direction)
    train_min, train_max = raw_train.min(axis=0), raw_train.max(axis=0)
    actor = FrozenActor()
    actor.load_state_dict(torch.load(root / "model" / "frozen_actor_state.pt", map_location="cpu", weights_only=True), strict=True)
    actor.requires_grad_(False).eval()
    actor_before = parameter_fingerprint(actor)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(holdout / ("offline_interventions_" + stamp))
    folder.mkdir(parents=True, exist_ok=False)
    protocol = {"experiment_ids": identifiers, "layer": layer, "physical_simulation": False,
        "scope": "fixed-input effects on frozen actor readout only",
        "state_selection": "per target and trajectory: positive Top1 code >= train weighted q80, sorted by state_index, fixed evenly spaced indices",
        "max_states_per_target_per_trajectory": max_states_per_trajectory,
        "feature_selection": "development train-only target correlation rank 1, no holdout feature selection",
        "suppression": "z_j to zero", "enhancement": "add half train active-code weighted population std; skip if exceeds train weighted q99",
        "activation_update": "h' = h + D_raw (z' - z), retaining reconstruction residual",
        "controls": "same delta_h L2 norm: 5 random directions each +/- (10 total); raw train Top1 +/-; global train PCA1 +/-",
        "random_seed": 20261102, "direction_seed_key": "target|trajectory_id|state_index; shared across models and operations",
        "training_range": "coordinate-wise train activation minimum and maximum; diagnostics, never clipping",
        "interpretation": "Effect magnitudes relative to random controls do not establish semantic causality or physical motion",
        "intervention_source_sha256": sha256(Path(__file__)),
        "study_freeze_manifest_sha256": sha256(root / "frozen_evaluation_plan.json"),
        "holdout_summary_sha256": sha256(holdout / "summary.json")}
    dump(folder / "protocol.json", protocol)
    all_rows, all_skips, all_features = [], [], []
    with threadpool_limits(limits=1):
        for identifier in identifiers:
            experiment = root / identifier
            result = _json(experiment / "summary.json")
            if result["configuration"].get("label_mode", "true") != "true":
                raise ValueError("Negative label controls are not intervention candidates")
            model, scaler, payload = load_assisted_checkpoint(experiment / "best_model.pt")
            batch_size = payload["configuration"]["sae"]["batch_size"]
            train_z, _ = infer_sae(model, scaler.transform(raw_train), batch_size)
            test_z, _ = infer_sae(model, scaler.transform(raw_test), batch_size)
            decoder_raw = model.decoder.weight.detach().cpu().numpy() * scaler.scale[:, None]
            ranking = _csv(experiment / "coordinate_ranking.csv")
            rows, skips, features, delta_arrays, mu_arrays, clipped_arrays = [], [], [], [], [], []
            destination = folder / identifier
            destination.mkdir()
            for target_index, target in enumerate(TARGETS):
                feature = int(next(row["coordinate"] for row in ranking if row["target"] == target
                    and row["representation"] == "z" and int(row["rank"]) == 1))
                raw_coordinate = int(next(row["coordinate"] for row in ranking if row["target"] == target
                    and row["representation"] == layer and int(row["rank"]) == 1))
                code_train = train_z[:, feature]
                q80 = weighted_quantile(code_train, weights, .8)
                q99 = weighted_quantile(code_train, weights, .99)
                active = code_train > 0
                active_mean = float(np.average(code_train[active], weights=weights[active])) if active.any() else 0.
                active_std = float(np.sqrt(np.average((code_train[active] - active_mean) ** 2, weights=weights[active]))) if active.any() else 0.
                feature_record = {"experiment_id": identifier, "target": target, "feature": feature,
                    "raw_top1_coordinate": raw_coordinate, "train_q80": q80, "train_q99": q99,
                    "train_active_std": active_std, "enhancement_increment": .5 * active_std,
                    "train_active_fraction": float(np.average(active.astype(float), weights=weights)),
                    "ranking_sha256": sha256(experiment / "coordinate_ranking.csv")}
                features.append(feature_record)
                eligible = (test_z[:, feature] > 0) & (test_z[:, feature] >= q80)
                for trajectory in sorted(set(test["trajectory_id"])):
                    candidate_indices = np.flatnonzero(eligible & (test["trajectory_id"] == trajectory))
                    candidate_indices = candidate_indices[np.argsort(test["state_index"][candidate_indices], kind="stable")]
                    if not len(candidate_indices):
                        continue
                    positions = np.rint(np.linspace(0, len(candidate_indices) - 1,
                        min(max_states_per_trajectory, len(candidate_indices)))).astype(int)
                    for index in candidate_indices[positions]:
                        state_index = int(test["state_index"][index])
                        original_code = test_z[index]
                        before = float(original_code[feature])
                        base = {"experiment_id": identifier, "seed": int(result["seed"]), "target": target,
                            "feature": feature, "trajectory_id": str(trajectory), "split": str(test["split"][index]),
                            "state_index": state_index, "source_row_index": int(index), "eligible_states_in_trajectory": len(candidate_indices),
                            "original_code": before, "train_q80": q80, "train_q99": q99}
                        random_key = f"{target}|{trajectory}|{state_index}"
                        direction_seed = int.from_bytes(hashlib.sha256(random_key.encode("utf-8")).digest()[:8], "little") ^ 20261102
                        for operation, after in (("suppress", 0.), ("enhance", before + .5 * active_std)):
                            if operation == "enhance" and (active_std <= 0 or after > q99):
                                skips.append({**base, "operation": operation, "requested_code": after,
                                    "status": "skipped", "reason": "no active variation" if active_std <= 0 else "would exceed train q99"})
                                continue
                            modified_code = original_code.astype(float).copy()
                            modified_code[feature] = after
                            modified_h = residual_intervention(raw_test[index], decoder_raw, original_code, modified_code)
                            delta_h = modified_h - raw_test[index]
                            norm = float(np.linalg.norm(delta_h))
                            if norm <= 1e-15:
                                skips.append({**base, "operation": operation, "requested_code": after,
                                    "status": "skipped", "reason": "zero activation perturbation"})
                                continue
                            controls = matched_control_deltas(delta_h, raw_coordinate, pca_direction,
                                np.random.default_rng(direction_seed))
                            variants = [("sae_feature", delta_h), *controls]
                            stats, deltas, delta_mu, delta_clipped = evaluate_actor_perturbations(actor,
                                raw_test[index], variants, train_min, train_max, layer)
                            if not np.allclose(np.linalg.norm(deltas, axis=1), norm, atol=1e-10, rtol=1e-10):
                                raise ValueError("Intervention controls do not match activation-space norm")
                            for row in stats:
                                row.update(base, operation=operation, modified_code=after,
                                    sae_delta_h_norm=norm, direction_seed=direction_seed,
                                    array_row=len(rows))
                                rows.append(row)
                            delta_arrays.extend(deltas)
                            mu_arrays.extend(delta_mu)
                            clipped_arrays.extend(delta_clipped)
            _write_union(destination / "interventions.csv", rows)
            _write_union(destination / "skipped_interventions.csv", skips)
            _write_union(destination / "train_feature_rules.csv", features)
            np.savez_compressed(destination / "delta_arrays.npz",
                delta_h=np.asarray(delta_arrays).reshape(-1, raw_train.shape[1]),
                delta_mu=np.asarray(mu_arrays).reshape(-1, 22), delta_clipped=np.asarray(clipped_arrays).reshape(-1, 22),
                pca1_direction_raw=pca_direction, train_activation_min=train_min, train_activation_max=train_max)
            all_rows.extend(rows)
            all_skips.extend(skips)
            all_features.extend(features)
            print("Offline readout interventions:", identifier, len(rows), "variants;", len(skips), "skips", flush=True)
    _write_union(folder / "aggregate_metrics.csv", _aggregate(all_rows))
    _write_union(folder / "train_feature_rules.csv", all_features)
    if parameter_fingerprint(actor) != actor_before or any(p.requires_grad or p.grad is not None for p in actor.parameters()):
        raise RuntimeError("Actor changed during offline intervention study")
    summary = {"status": "offline_fixed_input_interventions_completed", "protocol": protocol,
        "experiment_ids": identifiers, "evaluated_variants": len(all_rows), "skipped_operations": len(all_skips),
        "policy_weights_unchanged": True, "physical_simulation": False,
        "causal_scope": "fixed-input effect of a specified representation perturbation on actor outputs only",
        "physical_or_semantic_causality_established": False}
    dump(folder / "summary.json", summary)
    return folder, summary
