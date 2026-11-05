"""Frozen holdout evaluation with synthetic trajectories only."""
import sys
sys.dont_write_bytecode = True
from dataclasses import asdict
from datetime import datetime
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import joblib
from sklearn.linear_model import Ridge

from test_activation_baseline import SyntheticRepository
from temp_scripts.activation_baseline import collect_windows, dump, sha256
from temp_scripts.actor_model import FrozenActor
from temp_scripts.project_paths import OUTPUT_ROOT
from temp_scripts.sae_model import SAEConfig
from temp_scripts.sae_training import actor_outputs, export_state_probes
from temp_scripts.state_assisted_model import AssistedConfig, fit_assisted
from temp_scripts.state_assisted_holdout import (TEST_ROLES, collect_holdout_windows,
    evaluate_holdout, frozen_probe_predictions, paired_trajectory_bootstrap)


class SyntheticHoldoutRepository(SyntheticRepository):
    def __init__(self, root, actor):
        super().__init__(root)
        for index, split in enumerate(TEST_ROLES[1:], 6):
            original = self.trajectories["synthetic_5"]
            trajectory = copy.deepcopy(original)
            tid = f"synthetic_{index}"
            trajectory.tid = tid
            trajectory.meta = {**original.meta, "trajectory_id": tid, "episode_id": str(index),
                "split": split, "numeric_archive": tid + ".npz"}
            self.records.append(trajectory.meta)
            self.by_id[tid] = trajectory.meta
            self.trajectories[tid] = trajectory
        # A missing-event test episode is counted in coverage, without an anchor.
        missing = copy.deepcopy(self.trajectories["synthetic_2"])
        missing.tid = "synthetic_missing"
        missing.meta = {**missing.meta, "trajectory_id": missing.tid, "split": TEST_ROLES[0],
            "numeric_archive": missing.tid + ".npz"}
        self.records.append(missing.meta)
        self.by_id[missing.tid] = missing.meta
        self.trajectories[missing.tid] = missing
        self.source_meta = {tid: {"checkpoint_sha256": "synthetic-checkpoint"} for tid in self.by_id}
        for trajectory in self.trajectories.values():
            h = trajectory.arrays["activations"].astype(np.float32)
            trajectory.arrays["activations"] = h
            with torch.no_grad():
                trajectory.arrays["activations_l2"] = torch.relu(actor.mlp_extractor.policy_net[2](torch.tensor(h))).numpy()
            mu = actor_outputs(actor, h, "h1")
            trajectory.arrays["action_mean_raw"], trajectory.arrays["actions"] = mu, np.clip(mu, -1, 1)
            np.savez(root / trajectory.meta["numeric_archive"], **trajectory.arrays)

    def load(self, tid):
        self.loaded.append(tid)
        return self.trajectories[tid]


class FrozenHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.area = OUTPUT_ROOT / "_test" / ("assisted_holdout_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.area.mkdir(parents=True)
        cls.actor = FrozenActor().requires_grad_(False).eval()
        cls.repo = SyntheticHoldoutRepository(cls.area / "source", cls.actor)

    def test_frozen_plan_required_before_repository_access(self):
        area = self.area / "no_plan"
        area.mkdir()
        with patch("temp_scripts.state_assisted_holdout.DataRepository", side_effect=AssertionError("Opened before freeze")):
            with self.assertRaisesRegex(ValueError, "Freeze"):
                evaluate_holdout(area, ["candidate"])
        dump(area / "frozen_evaluation_plan.json", {"experiment_ids": ["candidate"]})
        with patch("temp_scripts.state_assisted_holdout.DataRepository", side_effect=AssertionError("Opened before freeze")):
            with self.assertRaisesRegex(ValueError, "exactly"):
                evaluate_holdout(area, ["another_candidate"])

    def test_window_rules_exclude_terminal_and_keep_missing_coverage(self):
        data, mu, action = collect_holdout_windows(self.repo, progress=lambda _: None)
        self.assertEqual(set(data.splits), set(TEST_ROLES))
        self.assertTrue(all(r["state_index"] < self.repo.trajectories[r["trajectory_id"]].n for r in data.rows))
        missing = next(r for r in data.coverage if r["trajectory_id"] == "synthetic_missing")
        self.assertEqual(missing["selected_steps"], 0)
        self.assertEqual(missing["status"], "right_censored")
        self.assertEqual(mu.shape, action.shape)
        np.testing.assert_array_equal(action, np.clip(mu, -1, 1))

    def test_cluster_bootstrap_equal_trajectory_weight_and_single_class(self):
        y = np.array([0., 0., 0., 1.])
        candidate, reference = np.array([1., 1., 1., 1.]), np.zeros(4)
        result = paired_trajectory_bootstrap(y, candidate, reference, np.array(["a", "a", "a", "b"]), repetitions=100)
        # Candidate errors 1 on a and 0 on b; reference the reverse.
        self.assertAlmostEqual(result["candidate_minus_reference"], 0.)
        self.assertEqual(result["defined_replicates"], 100)
        single = paired_trajectory_bootstrap(np.zeros(4), candidate, reference,
            np.array(["a", "a", "a", "b"]), True, repetitions=20)
        self.assertIsNone(single["candidate_minus_reference"])
        self.assertIsNone(single["ci95_low"])
        self.assertEqual(single["defined_replicates"], 0)

    def test_synthetic_end_to_end_uses_frozen_probes_and_retains_all_splits(self):
        root = self.area / "study"
        root.mkdir()
        (root / "model").mkdir()
        torch.save(self.actor.state_dict(), root / "model" / "frozen_actor_state.pt")
        dump(root / "model" / "actor_metadata.json", {"checkpoint_sha256": "synthetic-checkpoint"})
        dump(root / "configuration.json", {"source_index": str(self.repo.root), "layer": "h1",
            "event": "sustained_opening", "radius": 5})
        data = collect_windows(self.repo, progress=lambda _: None)
        np.savez(root / "selected_arrays.npz", h=data.h1, h1=data.h1, h2=data.h2, y=data.y,
            split=data.splits, trajectory_id=data.groups)
        folder = root / "candidate_seed0"
        folder.mkdir()
        config = AssistedConfig(sae=SAEConfig(width=16, top_k=4, max_epochs=3, patience=2), name="candidate")
        _, _, _, z, _, fitted = fit_assisted(data.h1, data.y, data.splits, data.groups, config, 0,
            folder, actor=self.actor, progress=lambda _: None)
        export_state_probes(folder, data, z, "h1", "trajectory", quick=True)
        # A frozen time+Top4 artifact must prepend the saved time basis at test.
        artifacts = joblib.load(folder / "fitted_probes.joblib")
        base = artifacts["models"]["distance_m__z_top4__seed0"]
        spline = artifacts["transforms"]["time_scaler"].transform(
            artifacts["transforms"]["time_spline"].transform(data.time))
        values = np.column_stack((spline, artifacts["z_scaler"].transform(z)[:, base["coordinates"]]))
        train = data.splits == "train"
        target = base["target_scaler"].transform(data.y[:, :1]).ravel()
        with_time_model = Ridge(alpha=1.).fit(values[train], target[train])
        artifacts["models"]["distance_m__z_top4_time__seed0"] = {
            **base, "estimator": with_time_model, "with_time": True}
        joblib.dump(artifacts, folder / "fitted_probes.joblib")
        np.savez(folder / "reconstruction.npz", z=z)
        dump(folder / "summary.json", {"seed": 0, "configuration": asdict(config), **fitted})
        hashes = {folder.name: {name: sha256(folder / name) for name in ("best_model.pt", "fitted_probes.joblib", "summary.json")}}
        dump(root / "frozen_evaluation_plan.json", {"experiment_ids": [folder.name], "artifact_hashes": hashes})
        with patch("temp_scripts.state_assisted_holdout.DataRepository", return_value=self.repo), \
             patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("Refitted scaler")), \
             patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("Refitted probe")):
            destination, result = evaluate_holdout(root, [folder.name])
        self.assertEqual(result["status"], "frozen_holdout_evaluation_completed")
        self.assertEqual(set(result["counts"]), set(TEST_ROLES))
        self.assertEqual(result["counts"][TEST_ROLES[0]]["eligible_trajectories"], 2)
        self.assertEqual(result["counts"][TEST_ROLES[0]]["included_trajectories"], 1)
        self.assertTrue((destination / folder.name / "state_predictions.csv").is_file())
        self.assertTrue((destination / "paired_bootstrap.csv").is_file())
        self.assertTrue(result["no_refitting"])
        self.assertEqual(hashes, result["candidate_artifact_hashes"])
        with np.load(destination / folder.name / "probe_predictions.npz", allow_pickle=False) as predictions:
            self.assertIn("distance_m__z_top4_time__seed0", predictions.files)


if __name__ == "__main__":
    unittest.main()
