"""Real-checkpoint fidelity, hook neutrality, and failure-detection regressions."""
import sys
sys.dont_write_bytecode = True
import csv
from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temp_scripts.project_paths import bootstrap, PROJECT_ROOT, OUTPUT_ROOT, default_index
bootstrap()
import numpy as np
import torch
from temp_scripts.actor_model import load_actor, HookRecorder, parameter_fingerprint, replay
from temp_scripts.actor_verification import ACTOR_HOME, COMPARISONS, compare_values, verify_actor, load_replayed
from temp_scripts.data_access import DataRepository


class ActorVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.repo = DataRepository(default_index())
        cls.traj = cls.repo.load("object_9996__episode_00004")
        cls.checkpoint = PROJECT_ROOT.parents[2] / "dexart_assets.zip"
        cls.sha = cls.traj.provenance["checkpoint_sha256"]
        cls.actor, cls.metadata = load_actor(cls.checkpoint, cls.sha)

    def test_original_weights_strictly_loaded_and_frozen(self):
        self.assertEqual(self.metadata["parameters"], 26134)
        self.assertEqual(self.metadata["checkpoint_sha256"], self.sha)
        self.assertTrue(all(not p.requires_grad for p in self.actor.parameters()))
        self.assertFalse(self.actor.training)

    def test_wrong_checkpoint_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_actor(self.checkpoint, "0" * 64)

    def test_missing_parameter_never_silently_randomly_initialized(self):
        state = dict(self.actor.state_dict())
        del state["action_net.bias"]
        with patch("temp_scripts.actor_model.torch.load", return_value=state):
            with self.assertRaisesRegex(ValueError, "actor"):
                load_actor(self.checkpoint, self.sha)

    def test_action_head_agrees_with_independent_matrix_calculation(self):
        h2 = self.traj.arrays["activations_l2"][:8]
        with torch.no_grad():
            actual = self.actor.from_h2(torch.from_numpy(h2)).numpy()
        weight = self.actor.action_net.weight.detach().numpy().astype(float)
        bias = self.actor.action_net.bias.detach().numpy().astype(float)
        expected = h2.astype(float) @ weight.T + bias
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_all_replay_entries_match_logged_values_and_preserve_weights(self):
        before = parameter_fingerprint(self.actor)
        values, checks = replay(self.actor, self.traj.arrays, batch_size=13)
        self.assertTrue(checks["hooks_removed"])
        self.assertTrue(checks["hook_output_unchanged_exact"])
        self.assertEqual(parameter_fingerprint(self.actor), before)
        for name, (key, _) in COMPARISONS.items():
            self.assertTrue(compare_values(values[name], self.traj.arrays[key], 1e-5, 1e-4)["passed"], name)
            self.assertEqual(values[name].shape[0], self.traj.n)

    def test_hooks_removed_when_forward_raises(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with HookRecorder(self.actor):
                raise RuntimeError("test failure")
        self.assertTrue(all(not m._forward_hooks for m in [self.actor.mlp_extractor.policy_net[1], self.actor.mlp_extractor.policy_net[3], self.actor.action_net]))

    def test_shifted_h1_is_detected_and_localized(self):
        arrays = dict(self.traj.arrays)
        arrays["activations"] = np.roll(arrays["activations"], 1, axis=0)
        values, _ = replay(self.actor, arrays)
        self.assertFalse(compare_values(values["B_h2"], arrays["activations_l2"], 1e-5, 1e-4)["passed"])
        self.assertTrue(compare_values(values["C_h2"], arrays["activations_l2"], 1e-5, 1e-4)["passed"])

    def test_clipping_alone_can_hide_raw_mean_error(self):
        reference, prediction = np.array([[2., -2.]]), np.array([[3., -3.]])
        self.assertFalse(compare_values(prediction, reference, 1e-5, 1e-4)["passed"])
        self.assertTrue(compare_values(np.clip(prediction, -1, 1), np.clip(reference, -1, 1), 1e-5, 1e-4)["passed"])

    def test_tolerance_uses_reference_magnitude_and_absolute_term(self):
        self.assertTrue(compare_values([[1e-6, 10000.5]], [[0., 10000.]], 1e-5, 1e-4)["passed"])
        self.assertFalse(compare_values([[1e-6, 10000.5]], [[0., 10000.]], 1e-5, 0.)["passed"])
        with self.assertRaises(ValueError):
            compare_values([[np.nan]], [[0.]], 1e-5, 1e-4)

    def test_export_retains_only_real_action_rows_and_separate_terminal_record(self):
        root = OUTPUT_ROOT / "_test" / ("actor_verification_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        pointer_before = (ACTOR_HOME / "latest.json").read_bytes() if (ACTOR_HOME / "latest.json").exists() else None
        folder, summary = verify_actor(self.repo, self.checkpoint, records=[self.traj.meta], destination=root, figures=False, publish=False)
        pointer_after = (ACTOR_HOME / "latest.json").read_bytes() if (ACTOR_HOME / "latest.json").exists() else None
        self.assertEqual(pointer_before, pointer_after)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["decision_steps"], self.traj.n)
        data = load_replayed(folder, self.traj.tid)
        self.assertEqual(int(data["terminal_state_index"]), self.traj.n)
        self.assertEqual(len(data["C_mu"]), self.traj.n)
        with (folder / "step_errors.csv").open(encoding="utf-8-sig", newline="") as f:
            steps = list(csv.DictReader(f))
        self.assertEqual(int(steps[-1]["action_index"]), self.traj.n - 1)
        with (folder / "terminal_states.csv").open(encoding="utf-8-sig", newline="") as f:
            terminal = list(csv.DictReader(f))[0]
        self.assertEqual(terminal["action_index"], "")


if __name__ == "__main__":
    unittest.main()
