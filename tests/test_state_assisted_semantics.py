"""Synthetic checks for read-only mixed-association and time-support review."""
import sys
sys.dont_write_bytecode = True
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from test_activation_baseline import SyntheticRepository
from temp_scripts.activation_baseline import collect_windows, dump, write_csv
from temp_scripts.project_paths import OUTPUT_ROOT
from temp_scripts.sae_model import SAEConfig
from temp_scripts.sae_training import export_state_probes
from temp_scripts.state_assisted_model import AssistedConfig, fit_assisted
from temp_scripts.state_assisted_semantics import (fixed_object_time_pairs, review_semantics,
    weighted_correlation)


class FrozenSemanticsTests(unittest.TestCase):
    def test_weighted_correlation_is_signed_and_constant_is_undefined(self):
        x = np.array([0., 0., 0., 1.])
        corr, status = weighted_correlation(x, -x, np.array([1 / 3, 1 / 3, 1 / 3, 1.]))
        self.assertAlmostEqual(corr, -1.)
        self.assertEqual(status, "defined")
        corr, status = weighted_correlation(x, np.ones(4), np.ones(4))
        self.assertIsNone(corr)
        self.assertEqual(status, "zero_variance")

    def test_time_pairs_are_one_to_one_cross_trajectory_and_keep_unsupported_states(self):
        rows = [{"split": "validation", "object_id": obj, "trajectory_id": trajectory,
            "state_index": i, "time_s": time, "irrelevant_target": i * 100}
            for obj, trajectory, times in (("o", "a", [0., .04, .08]), ("o", "b", [0., .04, .12]),
                                          ("unsupported", "c", [0., .04]))
            for i, time in enumerate(times)]
        pairs, coverage = fixed_object_time_pairs(rows, .0404)
        used = [row[key] for row in pairs for key in ("index_a", "index_b")]
        self.assertEqual(len(used), len(set(used)))
        self.assertTrue(all(row["trajectory_a"] != row["trajectory_b"] for row in pairs))
        self.assertTrue(all(row["absolute_time_gap_seconds"] <= .0404 for row in pairs))
        missing = next(row for row in coverage if row["object_id"] == "unsupported")
        self.assertEqual(missing["states_without_any_common_support"], 2)
        changed = [{**row, "irrelevant_target": -1e9} for row in rows]
        self.assertEqual(pairs, fixed_object_time_pairs(changed, .0404)[0])

    def test_full_review_does_not_refit_model_or_probes(self):
        torch.set_num_threads(1)
        root = OUTPUT_ROOT / "_test" / ("semantics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        root.mkdir(parents=True)
        repo = SyntheticRepository(root / "source")
        data = collect_windows(repo, progress=lambda _: None)
        np.savez(root / "selected_arrays.npz", y=data.y, split=data.splits)
        write_csv(root / "selected_frames.csv", data.rows)
        candidate = root / "candidate_seed0"
        candidate.mkdir()
        config = AssistedConfig(sae=SAEConfig(width=16, top_k=4, max_epochs=2, patience=2))
        _, _, _, z, _, result = fit_assisted(data.h1, data.y, data.splits, data.groups,
            config, 0, candidate, progress=lambda _: None)
        export_state_probes(candidate, data, z, "h1", "trajectory", quick=True)
        np.savez(candidate / "reconstruction.npz", z=z)
        dump(candidate / "summary.json", {"configuration": asdict(config), **result})
        with patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("Refitted scaler")), \
             patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("Refitted probe")):
            folder, summary = review_semantics(root, [candidate.name])
        self.assertTrue(summary["candidate_hashes_unchanged"])
        self.assertFalse(summary["single_semantic_or_causal_claim"])
        self.assertEqual(summary["state_counts"], {"train": 22, "validation": 22})
        self.assertTrue((folder / "time_pair_coverage.csv").is_file())
        self.assertTrue((folder / candidate.name / "matched_pair_changes.csv").is_file())


if __name__ == "__main__":
    unittest.main()
