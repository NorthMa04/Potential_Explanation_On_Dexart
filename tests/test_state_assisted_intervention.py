"""Synthetic residual, norm-matching and frozen-readout intervention checks."""
import sys
sys.dont_write_bytecode = True
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from temp_scripts.actor_model import FrozenActor, parameter_fingerprint
from temp_scripts.activation_baseline import dump, sha256, write_csv
from temp_scripts.project_paths import OUTPUT_ROOT
from temp_scripts.sae_model import ActivationStandardizer, SAEConfig
from temp_scripts.state_assisted_model import AssistedConfig, AssistedTopKSAE, save_assisted_checkpoint
from temp_scripts.state_assisted_intervention import (evaluate_actor_perturbations,
    matched_control_deltas, residual_intervention, run_offline_interventions)


class OfflineInterventionTests(unittest.TestCase):
    def test_zero_intervention_is_exact_identity_and_residual_is_retained(self):
        rng = np.random.default_rng(113)
        h = rng.normal(size=(4, 64))
        z = rng.normal(size=(4, 16))
        decoder = rng.normal(size=(64, 16))
        bias = rng.normal(size=64)
        unchanged = residual_intervention(h, decoder, z, z.copy())
        np.testing.assert_array_equal(h, unchanged)
        modified = z.copy()
        modified[:, 7] = 0
        changed = residual_intervention(h, decoder, z, modified)
        original_residual = h - (z @ decoder.T + bias)
        modified_residual = changed - (modified @ decoder.T + bias)
        np.testing.assert_allclose(original_residual, modified_residual, atol=1e-14, rtol=1e-14)
        self.assertFalse(np.array_equal(changed, modified @ decoder.T + bias))

    def test_controls_match_norm_and_random_signs_are_paired_deterministically(self):
        rng = np.random.default_rng(127)
        delta = rng.normal(size=64)
        pca = rng.normal(size=64)
        first = matched_control_deltas(delta, 11, pca, np.random.default_rng(47))
        second = matched_control_deltas(delta, 11, pca, np.random.default_rng(47))
        self.assertEqual(len(first), 14)
        self.assertEqual(sum(name.startswith("random") for name, _ in first), 10)
        for (name, value), (repeat_name, repeat_value) in zip(first, second):
            self.assertEqual(name, repeat_name)
            np.testing.assert_array_equal(value, repeat_value)
            self.assertAlmostEqual(np.linalg.norm(value), np.linalg.norm(delta), places=12)
        for index in range(0, 10, 2):
            np.testing.assert_array_equal(first[index][1], -first[index + 1][1])
        raw = next(value for name, value in first if name == "raw_neuron_plus")
        self.assertEqual(np.count_nonzero(raw), 1)
        self.assertEqual(int(np.flatnonzero(raw)[0]), 11)

    def test_actor_is_frozen_zero_delta_is_zero_and_domain_diagnostics_are_saved(self):
        torch.set_num_threads(1)
        actor = FrozenActor().requires_grad_(False).eval()
        before = parameter_fingerprint(actor)
        h = np.full(64, .5)
        perturbations = [("zero", np.zeros(64)), ("negative", np.full(64, -1.5))]
        rows, delta_h, delta_mu, delta_clipped = evaluate_actor_perturbations(actor, h,
            perturbations, np.zeros(64), np.ones(64))
        np.testing.assert_array_equal(delta_mu[0], np.zeros(22))
        np.testing.assert_array_equal(delta_clipped[0], np.zeros(22))
        self.assertEqual(rows[1]["after_negative_coordinate_fraction"], 1.)
        self.assertEqual(rows[1]["after_outside_train_range_fraction"], 1.)
        self.assertEqual(sum(key.startswith("delta_mu_") and key[-2:].isdigit() for key in rows[0]), 22)
        self.assertEqual(sum(key.startswith("delta_clipped_") and key[-2:].isdigit() for key in rows[0]), 22)
        self.assertEqual(parameter_fingerprint(actor), before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in actor.parameters()))
        next(actor.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate_actor_perturbations(actor, h, perturbations, np.zeros(64), np.ones(64))

    def test_complete_synthetic_intervention_run_never_needs_simulation_or_original_archives(self):
        torch.set_num_threads(1)
        root = OUTPUT_ROOT / "_test" / ("intervention_pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        holdout, experiment = root / "holdout", root / "candidate_seed0"
        (root / "model").mkdir(parents=True)
        holdout.mkdir()
        experiment.mkdir()
        rng = np.random.default_rng(139)
        h = rng.uniform(0, 1, (16, 64))
        test = rng.uniform(0, 1, (6, 64))
        np.savez(root / "selected_arrays.npz", h=h, h1=h,
            y=np.column_stack((h[:, 0], h[:, 1], h[:, 2] > .5)),
            split=np.array(["train"] * 12 + ["validation"] * 4),
            trajectory_id=np.repeat(["a", "b", "c", "v"], 4))
        np.savez(holdout / "selected_arrays.npz", h=test, h1=test,
            y=np.column_stack((test[:, 0], test[:, 1], test[:, 2] > .5)),
            split=np.array(["test_trajectory"] * 6), trajectory_id=np.repeat(["ta", "tb"], 3),
            state_index=np.tile(np.arange(3), 2))
        actor = FrozenActor().requires_grad_(False).eval()
        torch.save(actor.state_dict(), root / "model" / "frozen_actor_state.pt")
        dump(root / "configuration.json", {"layer": "h1"})
        cfg = AssistedConfig(sae=SAEConfig(width=16, top_k=4), name="candidate")
        scaler = ActivationStandardizer.fit(h[:12], np.ones(12), mode="global")
        model = AssistedTopKSAE(64, 16, 4)
        # Constant positive codes ensure deterministic eligible test states;
        # zero active std deliberately exercises the enhancement skip path.
        with torch.no_grad():
            model.encoder.weight.zero_()
            model.encoder.bias.copy_(torch.arange(1, 17, dtype=torch.float32))
        save_assisted_checkpoint(experiment / "best_model.pt", model, scaler, {}, cfg, 0, 0, 1., 1., "reconstruction")
        dump(experiment / "summary.json", {"seed": 0, "configuration": asdict(cfg)})
        (experiment / "fitted_probes.joblib").write_bytes(b"unused frozen probe artifact")
        write_csv(experiment / "coordinate_ranking.csv", [
            {"target": target, "representation": representation, "rank": 1,
             "coordinate": 15 if representation == "z" else index}
            for index, target in enumerate(("distance_m", "opening_rad", "any_contact"))
            for representation in ("z", "h1")])
        hashes = {name: sha256(experiment / name) for name in ("best_model.pt", "fitted_probes.joblib", "summary.json")}
        dump(root / "frozen_evaluation_plan.json", {"experiment_ids": [experiment.name]})
        dump(holdout / "summary.json", {"status": "frozen_holdout_evaluation_completed",
            "experiment_ids": [experiment.name], "candidate_artifact_hashes": {experiment.name: hashes}})
        destination, summary = run_offline_interventions(root, holdout, [experiment.name], max_states_per_trajectory=2)
        self.assertEqual(summary["evaluated_variants"], 3 * 2 * 2 * 15)
        self.assertEqual(summary["skipped_operations"], 3 * 2 * 2)
        self.assertTrue(summary["policy_weights_unchanged"])
        self.assertFalse(summary["physical_simulation"])
        with np.load(destination / experiment.name / "delta_arrays.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["delta_mu"].shape, (180, 22))
        self.assertTrue((destination / "aggregate_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
