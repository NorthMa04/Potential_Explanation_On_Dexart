"""Synthetic checks for the whole-trajectory shuffled-label negative control."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from temp_scripts.activation_baseline import dump, sample_weights
from temp_scripts.project_paths import OUTPUT_ROOT
from temp_scripts.sae_model import SAEConfig
from temp_scripts.sae_training import fit_sae
from temp_scripts.state_assisted_model import (AssistedConfig, assign_training_labels,
    fit_assisted, fit_target_scaler, load_assisted_checkpoint)
from temp_scripts.state_assisted_holdout import evaluate_holdout


class ShuffledLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.area = OUTPUT_ROOT / "_test" / ("assisted_labels_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.area.mkdir(parents=True)
        rng = np.random.default_rng(53)
        cls.x = rng.normal(size=(18, 64)).astype(np.float32)
        cls.y = np.column_stack((np.arange(18) ** 2, np.arange(18) * .2, np.arange(18) % 2)).astype(float)
        cls.splits = np.array(["train"] * 12 + ["validation"] * 6)
        cls.groups = np.array(["a"] * 3 + ["b"] * 4 + ["c"] * 5 + ["v1"] * 3 + ["v2"] * 3)

    def test_shuffle_is_deterministic_whole_blocks_and_does_not_mutate_input_or_rng(self):
        original = self.y.copy()
        torch_before = torch.get_rng_state().clone()
        assigned, audit = assign_training_labels(self.y, self.splits, self.groups, "trajectory_shuffle", 7)
        again, repeat_audit = assign_training_labels(self.y, self.splits, self.groups, "trajectory_shuffle", 7)
        np.testing.assert_array_equal(assigned, again)
        self.assertEqual(audit, repeat_audit)
        np.testing.assert_array_equal(self.y, original)
        np.testing.assert_array_equal(assigned[self.splits == "validation"], original[self.splits == "validation"])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        indices = np.array(audit["train_source_indices"])
        np.testing.assert_array_equal(assigned[self.splits == "train"], self.y[indices])
        self.assertFalse(audit["holdout_eligible"])
        self.assertEqual(len(audit["trajectory_mapping"]), 3)
        for mapping in audit["trajectory_mapping"]:
            self.assertNotEqual(mapping["destination_trajectory"], mapping["source_trajectory"])
            destination = np.flatnonzero((self.splits == "train") & (self.groups == mapping["destination_trajectory"]))
            sources = indices[np.searchsorted(np.flatnonzero(self.splits == "train"), destination)]
            self.assertEqual(set(self.groups[sources]), {mapping["source_trajectory"]})
            self.assertTrue(np.all(np.diff(sources) >= 0))

    def test_default_true_reproduces_pure_and_shuffle_scaler_uses_actual_supervision(self):
        cfg = SAEConfig(width=16, top_k=4, max_epochs=4, patience=3, batch_size=5, preprocessing="global")
        pure, true, shuffled = (self.area / name for name in ("pure", "true", "shuffled"))
        for folder in (pure, true, shuffled):
            folder.mkdir()
        old = fit_sae(self.x, self.splits, self.groups, cfg, 1, pure, progress=lambda _: None)
        current = fit_assisted(self.x, self.y, self.splits, self.groups,
            AssistedConfig(sae=cfg, state_weight=0), 1, true, progress=lambda _: None)
        np.testing.assert_array_equal(old[3], current[3])
        np.testing.assert_array_equal(old[4], current[4])
        self.assertEqual(old[5]["best_epoch"], current[5]["best_epoch"])
        result = fit_assisted(self.x, self.y, self.splits, self.groups,
            AssistedConfig(sae=cfg, label_mode="trajectory_shuffle"), 1, shuffled, progress=lambda _: None)
        assigned, audit = assign_training_labels(self.y, self.splits, self.groups, "trajectory_shuffle", 1)
        train = self.splits == "train"
        expected = fit_target_scaler(assigned[train], sample_weights(self.groups[train]), cfg.scale_floor)
        self.assertEqual(expected, result[5]["target_scaler"])
        _, _, payload = load_assisted_checkpoint(shuffled / "best_model.pt")
        self.assertEqual(payload["label_assignment"], audit)
        self.assertEqual(result[5]["label_assignment"], audit)
        self.assertEqual(payload["configuration"]["label_mode"], "trajectory_shuffle")

    def test_shuffled_candidates_are_rejected_before_any_test_repository_access(self):
        root = self.area / "forbidden_holdout"
        candidate = root / "negative_seed0"
        candidate.mkdir(parents=True)
        dump(root / "frozen_evaluation_plan.json", {"experiment_ids": [candidate.name]})
        dump(candidate / "summary.json", {"configuration": {"label_mode": "trajectory_shuffle"}})
        torch.save({"configuration": {"label_mode": "trajectory_shuffle"}}, candidate / "best_model.pt")
        (candidate / "fitted_probes.joblib").write_bytes(b"unused synthetic artifact")
        np.savez(candidate / "reconstruction.npz", z=np.zeros((1, 1)))
        with patch("temp_scripts.state_assisted_holdout.DataRepository", side_effect=AssertionError("Read test data")):
            with self.assertRaisesRegex(ValueError, "negative controls"):
                evaluate_holdout(root, [candidate.name])


if __name__ == "__main__":
    unittest.main()
