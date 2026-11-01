"""Current-state probes on complete, event-anchored windows. No policy updates."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from temp_scripts.data_access import DEV_ROLES, EVENT_LABELS
from temp_scripts.project_paths import PROJECT_ROOT, source_path, OUTPUT_ROOT, output_path

BASELINE_HOME = OUTPUT_ROOT / "activation_baseline"
TARGETS = ("distance_m", "opening_rad", "any_contact")


def dump(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class WindowData:
    rows: list
    h1: np.ndarray
    h2: np.ndarray
    y: np.ndarray
    coverage: list

    @property
    def groups(self):
        return np.array([r["trajectory_id"] for r in self.rows])

    @property
    def splits(self):
        return np.array([r["split"] for r in self.rows])

    @property
    def time(self):
        return np.array([r["time_s"] for r in self.rows], dtype=float)[:, None]


def collect_windows(repo, event="sustained_opening", radius=5, records=None, progress=print):
    """All action rows in the FIRST observed event window; never sample frames."""
    if event not in EVENT_LABELS or radius < 0:
        raise ValueError("Unknown event or negative window radius")
    records = list(repo.filter(DEV_ROLES) if records is None else records)
    if not records or any(r["split"] not in DEV_ROLES for r in records):
        raise ValueError("Baseline development may load only train/validation trajectories")
    rows, h1, h2, targets, coverage = [], [], [], [], []
    for count, meta in enumerate(records, 1):
        traj = repo.load(meta["trajectory_id"])
        candidates = [e for e in traj.events if e["event_type"] == event]
        observed = sorted((e for e in candidates if e["status"] == "observed"), key=lambda e: e["state_index"])
        audit = {"trajectory_id": traj.tid, "object_id": meta["object_id"], "split": meta["split"],
                 "trajectory_success": meta["trajectory_success"], "event_type": event,
                 "status": "observed" if observed else "right_censored" if any(e["status"] == "right_censored" for e in candidates) else "not_observed",
                 "observed_occurrences": len(observed), "event_id": "", "event_state_index": None,
                 "confirmation_state_index": None, "center_action_index": None,
                 "action_start_inclusive": None, "action_stop_exclusive": None,
                 "selected_steps": 0, "left_truncated": False, "right_truncated": False}
        if observed:
            e = observed[0]
            j = e["state_index"]
            # Same convention as physical-event v1 clips: center on trigger a_(j-1).
            center = max(0, j - 1)
            start, stop = max(0, center - radius), min(traj.n, center + radius + 1)
            audit.update(event_id=e["event_id"], event_state_index=j,
                         confirmation_state_index=e["confirmation_state_index"], center_action_index=center,
                         action_start_inclusive=start, action_stop_exclusive=stop, selected_steps=stop-start,
                         left_truncated=center-radius < 0, right_truncated=center+radius >= traj.n)
            for t in range(start, stop):
                rows.append({"trajectory_id": traj.tid, "object_id": int(meta["object_id"]), "split": meta["split"],
                             "state_index": t, "time_s": float(t * traj.dt), "event_state_index": j,
                             "relative_state_index": t-j, "event_id": e["event_id"],
                             "strict_contact": bool(traj.state("contact")[t]),
                             "trajectory_success": meta["trajectory_success"]})
                h1.append(traj.arrays["activations"][t])
                h2.append(traj.arrays["activations_l2"][t])
                targets.append([traj.state("palm_handle_distance")[t], traj.opening[t],
                                float(np.any(traj.state("finger_palm_contact")[t]))])
        coverage.append(audit)
        if count % 25 == 0 or count == len(records):
            progress(f"Windows: {count}/{len(records)} trajectories; {len(rows)} retained steps")
    data = WindowData(rows, np.asarray(h1, dtype=float).reshape(-1, 64), np.asarray(h2, dtype=float).reshape(-1, 64),
                      np.asarray(targets, dtype=float).reshape(-1, 3), coverage)
    if len({(r["trajectory_id"], r["state_index"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate source state rows")
    if not all(np.isfinite(a).all() for a in (data.h1, data.h2, data.y)):
        raise ValueError("Nonfinite model input or target")
    return data


def sample_weights(groups, mode="trajectory"):
    """Use EVERY selected row. Weighting changes total loss, never row inclusion."""
    if mode not in ("trajectory", "frame"):
        raise ValueError("Unknown weighting")
    groups = np.asarray(groups)
    counts = Counter(groups)
    weights = np.ones(len(groups)) if mode == "frame" else np.array([1. / counts[g] for g in groups])
    return weights / weights.mean()


def coordinate_ranking(x_train, y_train, weights):
    """Train-only absolute weighted Pearson correlation (point-biserial for 0/1)."""
    x = x_train - np.average(x_train, axis=0, weights=weights)
    y = y_train - np.average(y_train, weights=weights)
    cov = np.average(x * y[:, None], axis=0, weights=weights)
    denom = np.sqrt(np.average(x*x, axis=0, weights=weights) * np.average(y*y, weights=weights))
    score = np.divide(np.abs(cov), denom, out=np.zeros_like(cov), where=denom > 1e-12)
    return np.argsort(-score, kind="stable"), score


def metrics(y, prediction, classification=False, weights=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    mean = lambda x: float(np.average(x, weights=w))
    if not classification:
        variance = mean((y - mean(y)) ** 2)
        return {"mae": mean(np.abs(prediction-y)), "rmse": float(np.sqrt(mean((prediction-y)**2))),
                "r2": 1-mean((prediction-y)**2)/variance if variance > 1e-15 else None}
    p = np.clip(prediction, 1e-12, 1-1e-12)
    label = p >= .5
    tp, fp, fn = (float(w[m].sum()) for m in (label & (y == 1), label & (y == 0), ~label & (y == 1)))
    both = len(np.unique(y)) == 2
    return {"average_precision": float(average_precision_score(y, p, sample_weight=w)) if both else None,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1], sample_weight=w)),
            "brier": mean((p-y)**2), "positive_prevalence": mean(y),
            "ap_status": "defined" if both else "undefined_single_class"}


def build_representations(data, weights):
    train = data.splits == "train"
    scaler = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")
    scaler.fit(data.time[train], sample_weight=weights)
    spline = scaler.transform(data.time)
    time_scaler = StandardScaler().fit(spline[train], sample_weight=weights)
    spline = time_scaler.transform(spline)
    reps = {"constant": np.zeros((len(data.rows), 1)), "time_spline": spline}
    transforms = {"time_spline": scaler, "time_scaler": time_scaler}
    for layer in ("h1", "h2"):
        raw = getattr(data, layer)
        scale = StandardScaler().fit(raw[train], sample_weight=weights)
        x = scale.transform(raw)
        reps[layer+"_linear"] = x
        reps[layer+"_time_linear"] = np.column_stack([spline, x])
        # Weighted PCA on the same train-only standardized activation distribution.
        _, _, vt = np.linalg.svd(x[train] * np.sqrt(weights[:, None]), full_matrices=False)
        for k in (1, 2, 4, 8):
            if k <= len(vt):
                reps[f"{layer}_pca{k}"] = x @ vt[:k].T
        transforms[layer] = {"scaler": scale, "pca_components": vt}
    return reps, transforms


def predict_model(model, x, classification):
    if classification:
        if len(model.classes_) == 1:
            return np.full(len(x), float(model.classes_[0]))
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    return model.predict(x)


def fit_probe(x, y, train, validation, weights, family, classification, seed=0, quick=False):
    """Fit only train, choose regularization only validation; no internal frame split."""
    single_class = classification and len(np.unique(y[train])) < 2
    family = "constant" if single_class else family
    target_scale = StandardScaler().fit(y[train, None], sample_weight=weights) if not classification else None
    fit_y = y if classification else target_scale.transform(y[:, None]).ravel()
    values = [1.] if quick else [.1, 1., 10.]
    if family == "mlp":
        values = [.001] if quick else [.0001, .01]
    if family == "constant":
        values = [0.]
    candidates = []
    for value in values:
        if family == "constant":
            model = DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
        elif family == "mlp":
            cls = MLPClassifier if classification else MLPRegressor
            model = cls(hidden_layer_sizes=(32,), activation="tanh", alpha=value, random_state=seed,
                        max_iter=40 if quick else 400, early_stopping=False, solver="adam", batch_size=min(128, int(train.sum())))
        else:
            model = LogisticRegression(C=value, max_iter=2000, solver="lbfgs") if classification else Ridge(alpha=value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x[train], fit_y[train], sample_weight=weights)
        candidates.append((value, model, [str(w.message) for w in caught]))
    return candidates, target_scale, single_class


def summarize_prediction(data, target, name, seed, prediction):
    classification = target == "any_contact"
    y = data.y[:, TARGETS.index(target)]
    summary, per_trajectory, per_object = [], [], []
    objects = np.array([str(r["object_id"]) for r in data.rows])
    for split in DEV_ROLES:
        mask = data.splits == split
        for weighting in ("trajectory", "frame"):
            result = metrics(y[mask], prediction[mask], classification, sample_weights(data.groups[mask], weighting))
            summary.append({"target": target, "model": name, "seed": seed, "split": split,
                            "weighting": weighting, "steps": int(mask.sum()),
                            "trajectories": len(set(data.groups[mask])), **result})
        for kind, ids, dest in (("trajectory_id", data.groups, per_trajectory), ("object_id", objects, per_object)):
            for value in sorted(set(ids[mask])):
                selected = mask & (ids == value)
                result = metrics(y[selected], prediction[selected], classification,
                                 sample_weights(data.groups[selected], "trajectory"))
                dest.append({"target": target, "model": name, "seed": seed, "split": split,
                             kind: value, "steps": int(selected.sum()), **result})
        # Equal objects, equal trajectories within each object, equal rows within each trajectory.
        w = sample_weights(data.groups[mask], "trajectory")
        for obj in set(objects[mask]):
            local = objects[mask] == obj
            w[local] /= w[local].sum()
        summary.append({"target": target, "model": name, "seed": seed, "split": split,
                        "weighting": "object", "steps": int(mask.sum()),
                        "trajectories": len(set(data.groups[mask])), **metrics(y[mask], prediction[mask], classification, w)})
    return summary, per_trajectory, per_object


def run_baseline(repo, *, event="sustained_opening", radius=5, weighting="trajectory", seeds=(0, 1, 2),
                 include_mlp=True, destination=None, records=None, quick=False, publish=True, prepare_only=False, progress=print):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or BASELINE_HOME / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    configuration = {"event": event, "occurrence": "first observed per trajectory", "radius": radius,
                     "window_center": "trigger action max(0,event_state_index-1)", "weighting": weighting,
                     "seeds": list(seeds), "include_mlp": include_mlp, "quick": quick, "prepare_only": prepare_only,
                     "targets": list(TARGETS), "roles": list(DEV_ROLES),
                     "contact_definition": "any of thumb/index/middle/ring/palm touching handle_link; not stable grasp",
                     "all_selected_frames_used": True, "test_evaluation": False,
                     "threshold": .5, "coordinate_selection": "train-only absolute weighted correlation; heuristic, not best subset",
                     "selection_metric": "validation trajectory-weighted MAE / log loss",
                     "time_input": "elapsed seconds from episode start; no total duration, relative event time or outcome",
                     "mlp": "32 tanh units, no internal validation frame split; seeds reported separately",
                     "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    dump(folder / "configuration.json", configuration)
    try:
        with threadpool_limits(limits=1):
            data = collect_windows(repo, event, radius, records, progress)
            write_csv(folder / "window_coverage.csv", data.coverage)
            train, val = data.splits == "train", data.splits == "validation"
            if not train.any() or not val.any():
                raise ValueError("Both train and validation need confirmed event windows; inspect window_coverage.csv")
            w = sample_weights(data.groups[train], weighting)
            vw = sample_weights(data.groups[val], "trajectory")
            sample_rows = [{**r, **dict(zip(TARGETS, map(float, data.y[i])))} for i, r in enumerate(data.rows)]
            write_csv(folder / "selected_frames.csv", sample_rows)
            np.savez_compressed(folder / "selected_arrays.npz", h1=data.h1, h2=data.h2, y=data.y,
                                trajectory_id=data.groups, split=data.splits,
                                state_index=np.array([r["state_index"] for r in data.rows]))
            if prepare_only:
                summary = {"status": "windows_prepared_only", "configuration": configuration,
                           "steps": len(data.rows), "trajectories": len(set(data.groups)), "model_fitted": False}
                dump(folder / "summary.json", summary)
                return folder, summary
            reps, transforms = build_representations(data, w)
            summaries, trajectories, objects, selection, ranking = [], [], [], [], []
            predictions, artifacts = {}, {"configuration": configuration, "transforms": transforms, "models": {}}
            for column, target in enumerate(TARGETS):
                y = data.y[:, column]
                classification = target == "any_contact"
                task_reps = dict(reps)
                selected_coordinates = {}
                for layer in ("h1", "h2"):
                    rank, score = coordinate_ranking(reps[layer+"_linear"][train], y[train], w)
                    for order, coordinate in enumerate(rank, 1):
                        ranking.append({"target": target, "layer": layer, "rank": order,
                                        "coordinate": int(coordinate), "abs_train_correlation": float(score[coordinate])})
                    for k in (1, 2, 4):
                        name = f"{layer}_top{k}"
                        task_reps[name] = reps[layer+"_linear"][:, rank[:k]]
                        selected_coordinates[name] = rank[:k].tolist()
                    if include_mlp:
                        task_reps[layer+"_mlp"] = reps[layer+"_linear"]
                for name, x in task_reps.items():
                    family = "constant" if name == "constant" else "mlp" if name.endswith("_mlp") else "linear"
                    for seed in seeds if family == "mlp" else (0,):
                        progress(f"Probe: {target} / {name} / seed {seed}")
                        candidates, scale, single = fit_probe(x, y, train, val, w, family, classification, seed, quick)
                        best_score, chosen, chosen_value, chosen_warnings = float("inf"), None, None, []
                        for value, model, messages in candidates:
                            p = predict_model(model, x[val], classification)
                            if scale is not None:
                                p = scale.inverse_transform(p[:, None]).ravel()
                            score = metrics(y[val], p, classification, vw)["log_loss" if classification else "mae"]
                            selection.append({"target": target, "model": name, "seed": seed,
                                              "parameter": "alpha" if family == "mlp" or not classification else "C",
                                              "value": value, "validation_score": score, "warnings": " | ".join(messages)})
                            if score < best_score:
                                best_score, chosen, chosen_value, chosen_warnings = score, model, value, messages
                        p = predict_model(chosen, x, classification)
                        if scale is not None:
                            p = scale.inverse_transform(p[:, None]).ravel()
                        if not np.isfinite(p).all():
                            raise ValueError(f"Nonfinite prediction: {target}/{name}")
                        key = f"{target}__{name}__seed{seed}"
                        predictions[key] = p
                        artifacts["models"][key] = {"estimator": chosen, "target_scaler": scale,
                                                    "coordinates": selected_coordinates.get(name),
                                                    "single_class_train": single, "parameter": chosen_value,
                                                    "warnings": chosen_warnings}
                        s, t, o = summarize_prediction(data, target, name, seed, p)
                        summaries.extend(s); trajectories.extend(t); objects.extend(o)
            # Different target types have different metric columns; keep their union explicitly.
            fields = lambda rows: list(dict.fromkeys(k for r in rows for k in r))
            write_csv(folder / "metrics.csv", summaries, fields(summaries))
            write_csv(folder / "by_trajectory.csv", trajectories, fields(trajectories))
            write_csv(folder / "by_object.csv", objects, fields(objects))
            write_csv(folder / "model_selection.csv", selection)
            write_csv(folder / "coordinate_ranking.csv", ranking)
            np.savez_compressed(folder / "predictions.npz", **predictions)
            joblib.dump(artifacts, folder / "fitted_probes.joblib")
        counts = {role: {"eligible_trajectories": sum(r["split"] == role for r in data.coverage),
                         "included_trajectories": len(set(data.groups[data.splits == role])),
                         "selected_steps": int((data.splits == role).sum()),
                         "any_contact_positives": int(data.y[data.splits == role, 2].sum())} for role in DEV_ROLES}
        summary = {"status": "development_run_completed", "scope": "event-conditioned current-state decodability; no causal or SAE result",
                   "configuration": configuration, "counts": counts, "metrics": summaries,
                   "selected_models_with_warnings": sum(bool(m["warnings"]) for m in artifacts["models"].values()),
                   "single_class_target_models": sum(bool(m["single_class_train"]) for m in artifacts["models"].values()),
                   "validation_is_used_for_model_selection": True, "test_numeric_archives_loaded": 0}
        source_files = ["trajectory_index.csv", "events.csv", "event_rules.json", "provenance.json"]
        provenance = {"source_index": str(repo.root), "source_files_sha256": {n: sha256(repo.root / n) for n in source_files},
                      "numeric_archives_sha256": {tid: sha256(repo.root / repo.by_id[tid]["numeric_archive"]) for tid in set(data.groups)},
                      "event_rules_manually_approved": False, "policy_updated": False}
        snapshot = folder / "source_snapshot"
        snapshot.mkdir()
        for name in ("4_run_activation_baseline.py", "activation_baseline.py", "activation_baseline_plots.py",
                     "activation_baseline_viewer.py", "data_access.py", "project_paths.py", "STEP4_ACTIVATION_BASELINE.md", "requirements.txt", "plots.py", "__init__.py"):
            source = source_path(name)
            target = snapshot / source.relative_to(PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        provenance["code_sha256"] = {p.relative_to(snapshot).as_posix(): sha256(p) for p in snapshot.rglob("*") if p.is_file()}
        dump(folder / "provenance.json", provenance)
        from temp_scripts.activation_baseline_plots import export_figures
        export_figures(folder, data, predictions, summaries)
        dump(folder / "summary.json", summary)
        (folder / "README.md").write_text(
            "# 原始激活 baseline（开发集）\n\n"
            f"事件：{EVENT_LABELS[event]}；每条轨迹只取首次已确认事件。窗口以触发动作 j-1 为中心，前后各{radius}步，保留全部可用动作前状态。\n\n"
            "- metrics.csv：轨迹等权、自然帧等权、物体等权结果；validation 已用于选参，不是独立测试分数。\n"
            "- window_coverage.csv：每条开发集轨迹的纳入/缺失/截断情况。未发生事件不伪造窗口。\n"
            "- selected_frames.csv / selected_arrays.npz：全部入选步骤、同一步状态标签及两层激活。\n"
            "- by_trajectory.csv / by_object.csv：分轨迹和分物体结果；coordinate_ranking.csv 为训练集相关性筛选。\n"
            "- model_selection.csv：完整候选预算和收敛警告；fitted_probes.joblib：预处理及训练好的探针。\n"
            "- predictions.npz / figures/：逐步预测、汇总图及按轨迹ID顺序挑选的验证集示例。\n\n"
            "距离单位m，开启量rad，接触目标为任一手指/手掌接触把手，不代表抓稳；严格接触另存为状态展示。AP使用average_precision_score，不是梯形积分PR-AUC；单类别分组AP留空。\n"
            "所有窗口内步骤均参与；事件、未来确认、最终结局和相对事件时间不输入探针。时间输入只有从轨迹开始的秒数。\n"
            "结果仅描述已发生该事件的局部窗口；不评价未发生事件的轨迹，不证明特征语义或动作贡献。事件v1尚待人工审阅。\n",
            encoding="utf-8")
        if publish and not quick:
            dump(BASELINE_HOME / "latest.json", {"directory": str(folder)})
        return folder, summary
    except Exception as exc:
        dump(folder / "failure.json", {"status": "failed", "error": str(exc), "configuration": configuration})
        raise


def latest_result():
    pointer = BASELINE_HOME / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("先运行 4_run_activation_baseline.py；快速检查不会更新正式结果指针。")
    return output_path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
