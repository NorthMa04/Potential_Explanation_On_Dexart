"""Leakage, full-window grain, missing events, and synthetic end-to-end regression."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temp_scripts.project_paths import bootstrap, OUTPUT_ROOT, default_index
bootstrap()
import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
from temp_scripts.data_access import DataRepository
from temp_scripts.activation_baseline import (TARGETS, collect_windows, coordinate_ranking, sample_weights,
                                  build_representations, metrics, run_baseline, predict_model)


class SyntheticRepository:
    """Small synthetic recordings, with an unopened test trajectory sentinel."""
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True)
        for name in ("trajectory_index.csv", "events.csv", "event_rules.json", "provenance.json"):
            (root / name).write_text("synthetic fixture\n", encoding="utf-8")
        self.records, self.by_id, self.trajectories, self.loaded = [], {}, {}, []
        for i, split in enumerate(("train", "train", "train", "validation", "validation", "test_trajectory")):
            tid = f"synthetic_{i}"
            meta = {"trajectory_id": tid, "object_id": str(i % 2), "episode_id": str(i), "split": split,
                    "trajectory_success": str(i % 2 == 0), "numeric_archive": tid+".npz"}
            self.records.append(meta)
            self.by_id[tid] = meta
            rng = np.random.default_rng(i)
            h = rng.uniform(0, 1, (20, 64))
            y = np.column_stack([.1 + .2*h[:, 0], .05 + h[:, 1], (h[:, 2] > .5).astype(float)])
            groups = np.zeros((21, 5), dtype=bool)
            groups[:20, 0] = y[:, 2].astype(bool)
            state = {"palm_handle_distance": np.r_[y[:, 0], 99.], "contact": np.zeros(21, dtype=bool),
                     "finger_palm_contact": groups}
            events = [{"event_type": "sustained_opening", "status": "observed", "state_index": j,
                       "confirmation_state_index": j+2, "event_id": f"{tid}_{j}"} for j in (8, 16)]
            if i == 2:
                events = [{"event_type": "sustained_opening", "status": "right_censored", "state_index": 19,
                           "confirmation_state_index": None, "event_id": tid+"_unconfirmed"}]
            traj = SimpleNamespace(tid=tid, n=20, dt=.04, meta=meta, events=events,
                                   arrays={"activations": h, "activations_l2": h*.5},
                                   opening=np.r_[y[:, 1], 99.], state=lambda key, s=state: s[key])
            self.trajectories[tid] = traj
            np.savez(root / meta["numeric_archive"], activations=h)

    def filter(self, roles):
        return [r for r in self.records if r["split"] in roles]

    def load(self, tid):
        if self.by_id[tid]["split"].startswith("test"):
            raise AssertionError("Test numeric archive was accessed")
        self.loaded.append(tid)
        return self.trajectories[tid]


class ActivationBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.area = OUTPUT_ROOT / "_test" / ("activation_baseline_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.repo = SyntheticRepository(cls.area / "source")
        cls.data = collect_windows(cls.repo, progress=lambda s: None)

    def test_first_observed_window_contains_every_frame_and_current_labels(self):
        rows = [r for r in self.data.rows if r["trajectory_id"] == "synthetic_0"]
        self.assertEqual([r["state_index"] for r in rows], list(range(2, 13)))
        self.assertEqual({r["event_state_index"] for r in rows}, {8})
        expected = self.repo.trajectories["synthetic_0"].arrays["activations"][2:13]
        np.testing.assert_array_equal(self.data.h1[:11], expected)
        np.testing.assert_allclose(self.data.y[:11, 0], .1+.2*expected[:, 0])
        np.testing.assert_array_equal(self.data.y[:11, 2], (expected[:, 2] > .5).astype(float))
        self.assertFalse(any(r["strict_contact"] for r in rows))
        self.assertEqual(len(self.data.rows), 44)

    def test_censored_event_has_no_fabricated_window_or_test_load(self):
        audit = next(r for r in self.data.coverage if r["trajectory_id"] == "synthetic_2")
        self.assertEqual(audit["status"], "right_censored")
        self.assertEqual(audit["selected_steps"], 0)
        self.assertNotIn("synthetic_5", self.repo.loaded)
        with self.assertRaisesRegex(ValueError, "train/validation"):
            collect_windows(self.repo, records=[self.repo.by_id["synthetic_5"]])

    def test_terminal_is_excluded_even_when_it_is_event_anchor(self):
        traj = self.repo.trajectories["synthetic_0"]
        original = traj.events
        try:
            traj.events = [{"event_type": "sustained_opening", "status": "observed", "state_index": 20,
                            "confirmation_state_index": 20, "event_id": "terminal"}]
            data = collect_windows(self.repo, records=[traj.meta], progress=lambda s: None)
            self.assertEqual([r["state_index"] for r in data.rows], list(range(14, 20)))
            self.assertTrue(data.coverage[0]["right_truncated"])
            self.assertTrue(np.all(data.y[:, 0] < 1))
        finally:
            traj.events = original

    def test_trajectory_weighting_equalizes_total_without_dropping_rows(self):
        groups = np.array(["a", "a", "b", "b", "b", "b"])
        weights = sample_weights(groups)
        self.assertEqual(len(weights), len(groups))
        self.assertAlmostEqual(weights[groups == "a"].sum(), weights[groups == "b"].sum())
        self.assertAlmostEqual(weights.mean(), 1.)

    def test_preprocessing_and_ranking_do_not_learn_from_validation(self):
        data = self.data
        train = data.splits == "train"
        weights = sample_weights(data.groups[train])
        _, transforms = build_representations(data, weights)
        before_rank = coordinate_ranking(data.h1[train], data.y[train, 0], weights)[0]
        changed = type(data)(data.rows, data.h1.copy(), data.h2.copy(), data.y.copy(), data.coverage)
        changed.h1[~train] += 1e6
        changed.y[~train] += 1e6
        _, after = build_representations(changed, weights)
        np.testing.assert_array_equal(transforms["h1"]["scaler"].mean_, after["h1"]["scaler"].mean_)
        np.testing.assert_array_equal(transforms["h1"]["pca_components"], after["h1"]["pca_components"])
        np.testing.assert_array_equal(before_rank, coordinate_ranking(changed.h1[train], changed.y[train, 0], weights)[0])

    def test_single_class_ap_is_missing_and_logloss_is_finite(self):
        result = metrics(np.zeros(5), np.full(5, .1), True)
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["ap_status"], "undefined_single_class")
        self.assertTrue(np.isfinite(result["log_loss"]))

    def test_synthetic_end_to_end_saves_replayable_probes_and_no_test_scores(self):
        folder, summary = run_baseline(self.repo, include_mlp=True, seeds=(0,), quick=True, publish=False,
                                       destination=self.area / "smoke", progress=lambda s: None)
        self.assertEqual(summary["status"], "development_run_completed")
        self.assertEqual(summary["test_numeric_archives_loaded"], 0)
        self.assertEqual({r["split"] for r in summary["metrics"]}, {"train", "validation"})
        self.assertTrue((folder / "figures" / "validation_summary.png").exists())
        artifacts = joblib.load(folder / "fitted_probes.joblib")
        self.assertEqual(sum("_mlp__" in k for k in artifacts["models"]), 6)
        model = artifacts["models"]["distance_m__h1_linear__seed0"]
        x = artifacts["transforms"]["h1"]["scaler"].transform(self.data.h1)
        p = model["target_scaler"].inverse_transform(predict_model(model["estimator"], x, False)[:, None]).ravel()
        with np.load(folder / "predictions.npz", allow_pickle=False) as f:
            np.testing.assert_allclose(p, f["distance_m__h1_linear__seed0"], atol=1e-10)
        self.assertTrue(json.loads((folder / "configuration.json").read_text(encoding="utf-8"))["all_selected_frames_used"])

    def test_real_event_window_matches_existing_v1_segment(self):
        repo = DataRepository(default_index())
        record = repo.by_id["object_9996__episode_00004"]
        data = collect_windows(repo, records=[record], progress=lambda s: None)
        trajectory = repo.load(record["trajectory_id"])
        event = next(e for e in trajectory.events if e["event_type"] == "sustained_opening" and e["status"] == "observed")
        segment = trajectory.segments[event["segment_id"]]
        self.assertEqual([r["state_index"] for r in data.rows],
                         list(range(segment["action_start_inclusive"], segment["action_stop_exclusive"])))


if __name__ == "__main__":
    unittest.main()
