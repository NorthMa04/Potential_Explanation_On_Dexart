"""Synthetic fitting only; check sparsity, leakage, reload and frozen-actor alignment."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temp_scripts.project_paths import bootstrap, OUTPUT_ROOT, PROJECT_ROOT, default_index
bootstrap()
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from test_activation_baseline import SyntheticRepository
from temp_scripts.actor_model import load_actor, parameter_fingerprint
from temp_scripts.activation_baseline import collect_windows
from temp_scripts.data_access import DataRepository
from temp_scripts.sae_model import ActivationStandardizer, SAEConfig, TopKSAE, load_checkpoint
from temp_scripts.sae_training import (
    SAE_HOME, actor_outputs, check_original_actor, fit_sae, infer_sae, run_sae)
from temp_scripts.sae_viewer import SAEViewer


class SAERepository(SyntheticRepository):
    def __init__(self, root, actor, sha):
        super().__init__(root)
        (root / "segments.csv").write_text("synthetic fixture\n", encoding="utf-8")
        self.source_meta = {r["trajectory_id"]: {"checkpoint_sha256": sha} for r in self.records}
        for trajectory in self.trajectories.values():
            h = trajectory.arrays["activations"].astype(np.float32)
            trajectory.arrays["activations"] = h
            with torch.no_grad():
                h2 = actor.mlp_extractor.policy_net[3](actor.mlp_extractor.policy_net[2](torch.tensor(h)))
            trajectory.arrays["activations_l2"] = h2.numpy()
            mu = actor_outputs(actor, h, "h1")
            trajectory.arrays.update(action_mean_raw=mu, actions=np.clip(mu, -1, 1))
            np.savez(root / trajectory.meta["numeric_archive"], **trajectory.arrays)


class SAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.area = OUTPUT_ROOT / "_test" / ("sae_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.area.mkdir(parents=True)
        real_repo = DataRepository(default_index())
        cls.sha = real_repo.source_meta["object_9996__episode_00004"]["checkpoint_sha256"]
        cls.checkpoint = PROJECT_ROOT.parents[2] / "dexart_assets.zip"
        cls.actor, _ = load_actor(cls.checkpoint, cls.sha)
        cls.repo = SAERepository(cls.area / "source", cls.actor, cls.sha)
        cls.data = collect_windows(cls.repo, progress=lambda s: None)

    def test_topk_is_nonnegative_hard_constraint_with_gradient(self):
        model = TopKSAE(6, 12, 3)
        x = torch.randn(9, 6)
        z, reconstruction = model(x)
        self.assertTrue((z >= 0).all())
        self.assertTrue(((z != 0).sum(dim=1) <= 3).all())
        reconstruction.square().mean().backward()
        self.assertIsNotNone(model.encoder.weight.grad)
        self.assertTrue(torch.isfinite(model.encoder.weight.grad).all())
        with torch.no_grad():
            model.encoder.weight.zero_()
            model.encoder.bias.fill_(-1)
        self.assertEqual(int(model.encode(x).count_nonzero()), 0)

    def test_preprocessing_is_train_only_reversible_and_protects_constant_coordinates(self):
        training = np.array([[1., 2., 3.], [1., 4., 5.]])
        transform = ActivationStandardizer.fit(training, np.array([1., 3.]))
        np.testing.assert_allclose(transform.mean, [1., 3.5, 4.5])
        self.assertEqual(transform.scale[0], 1.)
        held_out = np.array([[1000., -1000., 80.]])
        before = transform.mean.copy()
        np.testing.assert_allclose(transform.inverse_transform(transform.transform(held_out)), held_out, atol=1e-4)
        np.testing.assert_array_equal(transform.mean, before)

    def test_invalid_configuration_is_rejected_before_writing(self):
        for config in (SAEConfig(top_k=129), SAEConfig(learning_rate=float("nan")),
                       SAEConfig(batch_size=0), SAEConfig(patience=-1)):
            with self.assertRaises(ValueError):
                config.validate()
        destination = self.area / "invalid"
        with self.assertRaises(ValueError):
            run_sae(self.repo, self.checkpoint, configurations=[SAEConfig(top_k=129)], destination=destination)
        self.assertFalse(destination.exists())

    def test_fit_restores_best_checkpoint_has_unit_dictionary_and_deterministic_reload(self):
        folder = self.area / "fit"
        folder.mkdir()
        config = SAEConfig(width=16, top_k=4, learning_rate=.01, max_epochs=20, patience=5)
        model, scaler, normalized, z, reconstructed, result = fit_sae(
            self.data.h1, self.data.splits, self.data.groups, config, 12, folder, progress=lambda s: None)
        payload = json.loads(json.dumps(result))
        self.assertGreaterEqual(payload["epochs_completed"], payload["best_epoch"])
        loaded, transform, metadata = load_checkpoint(folder / "best_model.pt")
        rz, rx = infer_sae(loaded, transform.transform(self.data.h1))
        np.testing.assert_array_equal(z, rz)
        np.testing.assert_allclose(reconstructed, transform.inverse_transform(rx))
        np.testing.assert_allclose(model.decoder.weight.detach().norm(dim=0), 1., atol=1e-6)
        train = self.data.splits == "train"
        np.testing.assert_allclose(scaler.mean, self.data.h1[train].mean(axis=0), atol=1e-10)
        self.assertEqual(metadata["best_epoch"], result["best_epoch"])
        self.assertTrue(((z > 0).sum(axis=1) <= 4).all())
        # A second identical seed has the same fitted code, not only the same shape.
        second = self.area / "fit_repeat"
        second.mkdir()
        again = fit_sae(self.data.h1, self.data.splits, self.data.groups, config, 12, second, progress=lambda s: None)
        np.testing.assert_array_equal(z, again[3])

    def test_early_stop_can_preserve_initial_model_without_claiming_improvement(self):
        folder = self.area / "early_stop"
        folder.mkdir()
        *_, result = fit_sae(self.data.h1, self.data.splits, self.data.groups,
             SAEConfig(width=8, top_k=2, learning_rate=1e-50, max_epochs=8, patience=2), 0, folder,
             progress=lambda s: None)
        self.assertTrue(result["early_stopped"])
        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(result["best_epoch"], 0)
        self.assertTrue(result["best_is_initial_model"])

    def test_reference_alignment_failure_is_detected_before_training(self):
        pred = np.array([[2.] * 22, [-2.] * 22])
        self.assertTrue(check_original_actor(pred, pred, np.clip(pred, -1, 1))["passed"])
        with self.assertRaisesRegex(ValueError, "saved means"):
            check_original_actor(pred + 1, pred, np.clip(pred, -1, 1))

    def test_fit_rejects_test_rows(self):
        splits = self.data.splits.copy()
        splits[0] = "test_trajectory"
        with self.assertRaisesRegex(ValueError, "no test rows"):
            fit_sae(self.data.h1, splits, self.data.groups, SAEConfig(), 0, self.area / "forbidden")

    def test_synthetic_full_run_and_hidden_native_viewer(self):
        pointer = SAE_HOME / "latest.json"
        before_pointer = pointer.read_bytes() if pointer.exists() else None
        fingerprint = parameter_fingerprint(self.actor)
        folder, summary = run_sae(self.repo, self.checkpoint, configurations=[SAEConfig(width=16, top_k=4,
                   max_epochs=3, patience=2)], destination=self.area / "end_to_end", quick=True,
                   publish=False, progress=lambda s: None)
        self.assertEqual(summary["status"], "development_run_completed")
        self.assertEqual(summary["test_numeric_archives_loaded"], 0)
        self.assertNotIn("synthetic_5", self.repo.loaded)
        self.assertEqual(before_pointer, pointer.read_bytes() if pointer.exists() else None)
        self.assertTrue(summary["policy_weights_unchanged"])
        self.assertEqual(fingerprint, parameter_fingerprint(self.actor))
        result = summary["experiments"][0]
        experiment = folder / result["experiment_id"]
        self.assertTrue((experiment / "figures" / "validation_probe_example.png").exists())
        import joblib
        probes = joblib.load(experiment / "fitted_probes.joblib")
        self.assertIn("any_contact__z_top4__seed0", probes["models"])
        self.assertIn("distance_m__h1_top4__seed0", probes["models"])
        self.assertFalse(json.loads((folder / "provenance.json").read_text(encoding="utf-8"))["labels_used_for_sae_optimization"])
        root = tk.Tk()
        root.withdraw()
        try:
            viewer = SAEViewer(root, folder)
            viewer.dimension.set("21")
            viewer.trajectory.set("synthetic_1")
            viewer.redraw()
            viewer.save_figure("window")
            root.update_idletasks()
            self.assertTrue((experiment / "figures" / "review_window_synthetic_1_dim21.png").exists())
            self.assertGreater(len(viewer.table.get_children()), 0)
        finally:
            root.destroy()

    def test_prepare_and_failed_runs_never_update_latest(self):
        pointer = SAE_HOME / "latest.json"
        before = pointer.read_bytes() if pointer.exists() else None
        folder, summary = run_sae(self.repo, self.checkpoint, destination=self.area / "prepare", prepare_only=True,
                                  progress=lambda s: None)
        self.assertEqual(summary["status"], "prepared_only")
        self.assertFalse(any(folder.rglob("best_model.pt")))
        with patch("temp_scripts.sae_training.fit_sae", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                run_sae(self.repo, self.checkpoint, destination=self.area / "failure", progress=lambda s: None)
        self.assertTrue((self.area / "failure" / "failure.json").exists())
        self.assertEqual(before, pointer.read_bytes() if pointer.exists() else None)

    def test_numbered_entry_defaults_match_agreed_trial(self):
        spec = importlib.util.spec_from_file_location("step5", PROJECT_ROOT / "5_run_pure_sae.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.build_parser().parse_args([])
        self.assertEqual((args.layer, args.width, args.top_k, args.seeds), ("h1", 128, 8, [0]))
        self.assertEqual((args.max_epochs, args.patience), (500, 50))
        self.assertFalse(args.compare_configs)


if __name__ == "__main__":
    unittest.main()
