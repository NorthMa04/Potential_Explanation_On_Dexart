"""Read-only checks of mixed state/time associations in frozen SAE units.

Units come from saved train-only Top1/Top4 rankings.  High-activation rules use
train only.  Object/time matches use neither target changes nor feature values.
Correlations, matching and supervised readout performance do not certify a
single semantic concept, disentanglement, or a causal physical mechanism.
"""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from temp_scripts.activation_baseline import TARGETS, dump, sample_weights, sha256, write_csv
from temp_scripts.project_paths import output_path
from temp_scripts.state_assisted_intervention import weighted_quantile


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_union(path, rows):
    write_csv(path, rows, list(dict.fromkeys(key for row in rows for key in row)) or ["status"])


def weighted_correlation(x, y, weights):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if x.shape != y.shape or x.shape != w.shape or x.ndim != 1 or not len(x):
        raise ValueError("Correlation needs aligned nonempty one-dimensional arrays")
    if not all(np.isfinite(v).all() for v in (x, y, w)) or (w <= 0).any():
        raise ValueError("Correlation requires finite values and positive weights")
    dx, dy = x - np.average(x, weights=w), y - np.average(y, weights=w)
    vx, vy = np.average(dx * dx, weights=w), np.average(dy * dy, weights=w)
    if len(x) < 3:
        return None, "fewer_than_three_rows"
    if vx <= 1e-24 or vy <= 1e-24:
        return None, "zero_variance"
    return float(np.clip(np.average(dx * dy, weights=w) / np.sqrt(vx * vy), -1, 1)), "defined"


def fixed_object_time_pairs(rows, tolerance_seconds):
    """Greedy one-to-one cross-trajectory pairs chosen on object/time only."""
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Time tolerance must be finite and nonnegative")
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["split"], str(row["object_id"]))].append(index)
    pairs, coverage = [], []
    for (split, obj), indices in sorted(strata.items()):
        edges, supported = [], set()
        for position, a in enumerate(indices):
            for b in indices[position + 1:]:
                if rows[a]["trajectory_id"] == rows[b]["trajectory_id"]:
                    continue
                gap = abs(float(rows[a]["time_s"]) - float(rows[b]["time_s"]))
                if gap > tolerance_seconds:
                    continue
                # Orient by trajectory identity, never by state/feature change.
                if str(rows[a]["trajectory_id"]) > str(rows[b]["trajectory_id"]):
                    left, right = b, a
                else:
                    left, right = a, b
                edges.append((gap, str(rows[left]["trajectory_id"]), str(rows[right]["trajectory_id"]),
                    int(rows[left]["state_index"]), int(rows[right]["state_index"]), left, right))
                supported.update((a, b))
        used, accepted = set(), []
        for gap, ta, tb, sa, sb, a, b in sorted(edges):
            if a in used or b in used:
                continue
            used.update((a, b))
            item = {"split": split, "object_id": obj, "index_a": a, "index_b": b,
                "trajectory_a": ta, "trajectory_b": tb, "state_index_a": sa, "state_index_b": sb,
                "time_a_seconds": float(rows[a]["time_s"]), "time_b_seconds": float(rows[b]["time_s"]),
                "absolute_time_gap_seconds": gap, "trajectory_pair": ta + "|" + tb}
            pairs.append(item)
            accepted.append(item)
        coverage.append({"split": split, "object_id": obj, "states": len(indices),
            "trajectories": len({rows[i]["trajectory_id"] for i in indices}), "candidate_edges": len(edges),
            "matched_pairs": len(accepted), "matched_states": len(used),
            "states_without_any_common_support": len(indices) - len(supported),
            "supported_but_unmatched_states": len(supported - used),
            "unmatched_states": len(indices) - len(used),
            "distinct_matched_trajectory_pairs": len({r["trajectory_pair"] for r in accepted}),
            "tolerance_seconds": tolerance_seconds})
    return pairs, coverage


def _load_rows(path):
    rows = _csv(path)
    for row in rows:
        row["state_index"] = int(row["state_index"])
        row["time_s"] = float(row["time_s"])
    return rows


def _artifact_hashes(path):
    return {name: sha256(path / name) for name in ("best_model.pt", "fitted_probes.joblib", "summary.json")}


def review_semantics(study_root, experiment_ids, *, holdout_root=None, destination=None, max_examples=3):
    """Write descriptive dev checks; optionally append already-evaluated holdout."""
    root = output_path(study_root)
    identifiers = list(experiment_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Use a nonempty unique frozen-candidate list")
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 1:
        raise ValueError("max_examples must be positive")
    candidates = []
    for identifier in identifiers:
        candidate = (root / identifier).resolve()
        if candidate.parent != root or not (candidate / "summary.json").is_file():
            raise ValueError("Unknown completed candidate")
        candidates.append(candidate)
    before = {p.name: _artifact_hashes(p) for p in candidates}
    with np.load(root / "selected_arrays.npz", allow_pickle=False) as archive:
        development = {key: archive[key] for key in archive.files}
    rows = _load_rows(root / "selected_frames.csv")
    if len(rows) != len(development["y"]) or not set(development["split"]).issubset({"train", "validation"}):
        raise ValueError("Development rows must align and contain only train/validation")
    labels = development["y"]
    holdout, holdout_info = None, None
    if holdout_root is not None:
        holdout = output_path(holdout_root)
        holdout_info = _json(holdout / "summary.json")
        if holdout_info.get("status") != "frozen_holdout_evaluation_completed" or not set(identifiers).issubset(holdout_info["experiment_ids"]):
            raise ValueError("Only already-frozen holdout candidates may be reviewed")
        for identifier in identifiers:
            if before[identifier] != holdout_info["candidate_artifact_hashes"][identifier]:
                raise ValueError("Candidate changed after holdout evaluation")
        with np.load(holdout / "selected_arrays.npz", allow_pickle=False) as archive:
            test = {key: archive[key] for key in archive.files}
        test_rows = _load_rows(holdout / "selected_frames.csv")
        if len(test_rows) != len(test["y"]) or any(str(role) in ("train", "validation") for role in test["split"]):
            raise ValueError("Holdout rows must align and exclude development splits")
        rows += test_rows
        labels = np.vstack((labels, test["y"]))
    splits = np.array([r["split"] for r in rows])
    trajectories = np.array([r["trajectory_id"] for r in rows])
    objects = np.array([str(r["object_id"]) for r in rows])
    times = np.array([r["time_s"] for r in rows])
    train = splits == "train"
    weights_train = sample_weights(trajectories[train])
    target_matrix = np.column_stack((labels, times))
    target_names = (*TARGETS, "absolute_time_seconds")
    control_steps = []
    for trajectory in sorted(set(trajectories[train])):
        values = np.unique(times[train & (trajectories == trajectory)])
        control_steps.extend(np.diff(values)[np.diff(values) > 0])
    if not control_steps:
        raise ValueError("No within-train-trajectory time differences to define matching support")
    control_step = float(np.median(control_steps))
    tolerance = control_step * 1.01  # Include recorded float roundoff at one tick.
    pairs, coverage = fixed_object_time_pairs(rows, tolerance)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or root / ("semantics_" + stamp))
    folder.mkdir(parents=True, exist_ok=False)
    protocol = {"experiment_ids": identifiers, "holdout_root": str(holdout) if holdout else None,
        "feature_selection": "saved train-only Top1/Top4 for each target; same frozen coordinates for all splits",
        "sample_weighting": "equal trajectories, equal rows within trajectory",
        "high_activation": "z>0 and z>=train trajectory-weighted q80",
        "display_expected_state": "contact class or train weighted target median, oriented by train feature-target correlation sign",
        "matching": "same split/object, different trajectories, absolute time gap<=1.01*train median control tick; greedily one-to-one on time gap then IDs",
        "matching_control_step_seconds": control_step, "matching_tolerance_seconds": tolerance,
        "pair_weighting": "equal unordered trajectory pairs; within-pair frame differences are dependent",
        "no_target_or_feature_based_matching": True, "model_or_probe_refitting": False,
        "causal_claim": False, "single_semantic_certification": False,
        "limitations": "state/time correlations may share causes; pair adjustment is descriptive; target thresholds are display conventions, not semantic truth",
        "candidate_hashes_before": before, "source_sha256": sha256(Path(__file__))}
    dump(folder / "protocol.json", protocol)
    _write_union(folder / "time_pair_coverage.csv", coverage)
    pair_definitions = []
    for pair in pairs:
        a, b = pair["index_a"], pair["index_b"]
        pair_definitions.append({**pair,
            **{"delta_" + name: float(target_matrix[b, index] - target_matrix[a, index])
               for index, name in enumerate(target_names)}})
    _write_union(folder / "object_time_pairs.csv", pair_definitions)
    target_correlations = []
    for split in sorted(set(splits)):
        mask = splits == split
        weights = sample_weights(trajectories[mask])
        for ai, a in enumerate(target_names):
            for bi, b in enumerate(target_names):
                value, status = weighted_correlation(target_matrix[mask, ai], target_matrix[mask, bi], weights)
                target_correlations.append({"split": split, "a": a, "b": b, "correlation": value, "status": status})
    _write_union(folder / "state_time_correlation.csv", target_correlations)
    all_correlations, all_high, all_pair_correlations, all_object_coverage = [], [], [], []
    summaries = []
    for candidate in candidates:
        identifier = candidate.name
        result = _json(candidate / "summary.json")
        artifacts = joblib.load(candidate / "fitted_probes.joblib")
        ranking = _csv(candidate / "coordinate_ranking.csv")
        selected = defaultdict(list)
        for target in TARGETS:
            for rank in range(1, 5):
                coordinate = int(next(row["coordinate"] for row in ranking if row["target"] == target
                    and row["representation"] == "z" and int(row["rank"]) == rank))
                selected[coordinate].append((target, rank))
            expected = list(artifacts["models"][f"{target}__z_top4__seed0"]["coordinates"])
            observed = [int(next(row["coordinate"] for row in ranking if row["target"] == target
                and row["representation"] == "z" and int(row["rank"]) == rank)) for rank in range(1, 5)]
            if expected != observed:
                raise ValueError("Saved Top4 ranking differs from frozen probe coordinates")
        with np.load(candidate / "reconstruction.npz", allow_pickle=False) as archive:
            z = archive["z"]
        if holdout is not None:
            with np.load(holdout / identifier / "reconstruction.npz", allow_pickle=False) as archive:
                z = np.vstack((z, archive["z"]))
        if len(z) != len(rows):
            raise ValueError("Saved codes do not align with source states")
        destination_candidate = folder / identifier
        destination_candidate.mkdir()
        correlations, high_summary, examples, paired_correlations, pair_changes, object_coverage, rules = [], [], [], [], [], [], []
        for coordinate, selections in sorted(selected.items()):
            values = z[:, coordinate]
            threshold = weighted_quantile(values[train], weights_train, .8)
            high = (values > 0) & (values >= threshold)
            train_correlations = {name: weighted_correlation(values[train], target_matrix[train, i], weights_train)[0]
                for i, name in enumerate(target_names)}
            selection_text = ";".join(f"{target}:Top{rank}" for target, rank in selections)
            rules.append({"feature": coordinate, "selected_for": selection_text, "train_q80": threshold,
                **{"train_corr_" + name: corr for name, corr in train_correlations.items()}})
            for split in sorted(set(splits)):
                mask = splits == split
                weights = sample_weights(trajectories[mask])
                corr_row = {"experiment_id": identifier, "seed": result["seed"], "feature": coordinate,
                    "split": split, "selected_for": selection_text, "states": int(mask.sum()),
                    "trajectories": len(set(trajectories[mask])), "train_q80": threshold}
                for i, name in enumerate(target_names):
                    corr, status = weighted_correlation(values[mask], target_matrix[mask, i], weights)
                    corr_row["corr_" + name], corr_row["status_" + name] = corr, status
                correlations.append(corr_row)
                for obj in sorted(set(objects[mask])):
                    local = mask & (objects == obj)
                    object_coverage.append({"experiment_id": identifier, "feature": coordinate, "split": split,
                        "object_id": obj, "states": int(local.sum()), "trajectories": len(set(trajectories[local])),
                        "positive_states": int((local & (values > 0)).sum()), "high_states": int((local & high).sum()),
                        "high_trajectories": len(set(trajectories[local & high])),
                        "train_q80": threshold, "selected_for": selection_text})
                for target, rank in selections:
                    target_index = TARGETS.index(target)
                    corr = train_correlations[target]
                    direction = 1 if corr is None or corr >= 0 else -1
                    target_threshold = .5 if target == "any_contact" else weighted_quantile(labels[train, target_index], weights_train, .5)
                    expected_state = labels[:, target_index] >= target_threshold if direction > 0 else labels[:, target_index] < target_threshold
                    high_local, expected_local = high[mask], expected_state[mask]
                    denominator_high = float(weights[high_local].sum())
                    denominator_expected = float(weights[expected_local].sum())
                    numerator = float(weights[high_local & expected_local].sum())
                    high_summary.append({"experiment_id": identifier, "feature": coordinate, "target": target,
                        "rank": rank, "split": split, "train_q80": threshold, "train_target_threshold": target_threshold,
                        "train_correlation_sign": direction, "high_states": int(high_local.sum()),
                        "high_fraction": float(np.average(high_local, weights=weights)),
                        "expected_fraction": float(np.average(expected_local, weights=weights)),
                        "expected_given_high": numerator / denominator_high if denominator_high else None,
                        "high_given_expected": numerator / denominator_expected if denominator_expected else None,
                        "false_high_states": int((high_local & ~expected_local).sum()),
                        "expected_but_low_states": int((~high_local & expected_local).sum()),
                        "scope": "train-fixed display rule; neither precision nor recall of semantic truth"})
                    for kind, case_mask in (("high_expected", high & expected_state), ("high_opposite", high & ~expected_state),
                                           ("low_expected", ~high & expected_state)):
                        indices = np.flatnonzero(mask & case_mask)
                        order = indices[np.argsort(-values[indices] if kind.startswith("high") else values[indices], kind="stable")]
                        used = set()
                        for index in order:
                            if trajectories[index] in used:
                                continue
                            used.add(trajectories[index])
                            examples.append({"experiment_id": identifier, "feature": coordinate, "target": target,
                                "rank": rank, "case_type": kind, **rows[index], "activation": float(values[index]),
                                "distance_m": float(labels[index, 0]), "opening_rad": float(labels[index, 1]),
                                "any_contact": float(labels[index, 2]), "train_q80": threshold,
                                "train_target_threshold": target_threshold, "train_correlation_sign": direction})
                            if len(used) == max_examples:
                                break
            for pair in pair_definitions:
                a, b = pair["index_a"], pair["index_b"]
                pair_changes.append({"experiment_id": identifier, "feature": coordinate, "selected_for": selection_text,
                    **pair, "z_a": float(values[a]), "z_b": float(values[b]), "delta_z": float(values[b] - values[a])})
            for split in sorted(set(splits)):
                local_pairs = [pair for pair in pair_definitions if pair["split"] == split]
                if not local_pairs:
                    paired_correlations.append({"experiment_id": identifier, "feature": coordinate, "split": split,
                        "pairs": 0, "status": "no_common_object_time_support"})
                    continue
                pair_groups = np.array([pair["trajectory_pair"] for pair in local_pairs])
                pair_weights = sample_weights(pair_groups)
                delta_z = np.array([values[pair["index_b"]] - values[pair["index_a"]] for pair in local_pairs])
                item = {"experiment_id": identifier, "feature": coordinate, "split": split,
                    "selected_for": selection_text, "pairs": len(local_pairs),
                    "distinct_trajectory_pairs": len(set(pair_groups)),
                    "objects": len({pair["object_id"] for pair in local_pairs}),
                    "delta_z_weighted_std": float(np.sqrt(np.average((delta_z - np.average(delta_z, weights=pair_weights)) ** 2, weights=pair_weights)))}
                for name in target_names:
                    delta_y = np.array([pair["delta_" + name] for pair in local_pairs])
                    corr, status = weighted_correlation(delta_z, delta_y, pair_weights)
                    item["delta_corr_" + name], item["status_" + name] = corr, status
                    item["nonzero_changes_" + name] = int((np.abs(delta_y) > 1e-12).sum())
                paired_correlations.append(item)
        for name, table in (("feature_state_time_correlations", correlations), ("high_activation_summary", high_summary),
            ("high_activation_examples", examples), ("matched_pair_correlations", paired_correlations),
            ("matched_pair_changes", pair_changes), ("object_feature_coverage", object_coverage), ("train_feature_rules", rules)):
            _write_union(destination_candidate / (name + ".csv"), table)
        all_correlations.extend(correlations)
        all_high.extend(high_summary)
        all_pair_correlations.extend(paired_correlations)
        all_object_coverage.extend(object_coverage)
        summaries.append({"experiment_id": identifier, "seed": result["seed"], "selected_units": len(selected),
            "unit_selection": {str(feature): [f"{target}:Top{rank}" for target, rank in targets] for feature, targets in selected.items()},
            "model_or_probe_refit": False})
    for name, table in (("feature_state_time_correlations", all_correlations), ("high_activation_summary", all_high),
                       ("matched_pair_correlations", all_pair_correlations), ("object_feature_coverage", all_object_coverage)):
        _write_union(folder / (name + ".csv"), table)
    after = {p.name: _artifact_hashes(p) for p in candidates}
    if before != after:
        raise RuntimeError("Frozen model/probe artifacts changed during read-only review")
    summary = {"status": "read_only_semantic_review_completed", "experiments": summaries,
        "splits": sorted(set(splits)), "state_counts": {split: int((splits == split).sum()) for split in sorted(set(splits))},
        "time_match_coverage": coverage, "matched_pairs": len(pairs), "protocol": protocol,
        "candidate_hashes_unchanged": True, "single_semantic_or_causal_claim": False}
    dump(folder / "summary.json", summary)
    report = ["# Frozen-unit association review", "", "All units were selected on development train; models and probes are unchanged.",
        "Train-fixed high-activation display cases are associations, not semantic ground truth.",
        "Object/time matching is descriptive; shared causes and target collinearity remain possible.", "",
        f"Matched state pairs: {len(pairs)}. Unmatched and unsupported states remain in time_pair_coverage.csv.", "",
        "- feature_state_time_correlations.csv: signed trajectory-weighted associations with all states and absolute time.",
        "- matched_pair_correlations.csv: within-object, near-absolute-time pair differences with pair-group weighting.",
        "- high_activation_summary.csv and per-candidate examples: train-fixed expected/opposite/missing cases.",
        "- object_feature_coverage.csv: positive/high activation coverage across objects.", "",
        "No feature is certified disentangled by these tables; supervised readout accuracy alone is insufficient."]
    (folder / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return folder, summary
