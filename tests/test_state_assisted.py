"""Synthetic checks for paired SAE training, label isolation and frozen actor loss.

No held-out numeric archive or simulator is opened by these tests.
"""
import sys
sys.dont_write_bytecode = True
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temp_scripts.project_paths import bootstrap, OUTPUT_ROOT
bootstrap()
import numpy as np
import torch

from temp_scripts.actor_model import FrozenActor, parameter_fingerprint
from temp_scripts.activation_baseline import sample_weights, sha256, write_csv
from temp_scripts.sae_model import SAEConfig
from temp_scripts.sae_training import export_state_probes, fit_sae, infer_sae
from temp_scripts.state_assisted_model import (
    AssistedConfig, AssistedTopKSAE, fit_assisted, fit_target_scaler,
    load_assisted_checkpoint, transform_targets)
from temp_scripts.state_assisted_controls import (evaluate_rotation_holdout,
    fit_ridge_rotation, fit_supervised_rotation, inverse_supervised_rotation,
    transform_supervised_rotation)
from temp_scripts.state_assisted_study import load_study, run_experiments


class StateAssistedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.area = OUTPUT_ROOT / "_test" / (
            "state_assisted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.area.mkdir(parents=True)
        rng = np.random.default_rng(913)
        cls.groups = np.array(["train_a"] * 6 + ["train_b"] * 9 +
                              ["train_c"] * 8 + ["val_a"] * 7 + ["val_b"] * 9)
        cls.splits = np.array(["train"] * 23 + ["validation"] * 16)
        cls.h = np.maximum(rng.normal(.6, .8, (39, 64)), 0).astype(np.float32)
        cls.y = np.column_stack([
            .1 + .03 * cls.h[:, 0] - .01 * cls.h[:, 1],
            .3 + .2 * cls.h[:, 2] + .04 * cls.h[:, 4],
            (cls.h[:, 3] > .6).astype(float)])
        cls.sae = SAEConfig(width=16, top_k=4, learning_rate=.005,
                            batch_size=8, max_epochs=14, patience=20,
                            preprocessing="global")
        cls.counter = 0

    def folder(self, label):
        type(self).counter += 1
        path = self.area / f"{self.counter:03d}_{label}"
        path.mkdir()
        return path

    def fit(self, config=None, *, y=None, splits=None, seed=4, actor=None, label="fit"):
        config = config or AssistedConfig(sae=self.sae)
        path = self.folder(label)
        result = fit_assisted(self.h, self.y if y is None else y,
            self.splits if splits is None else splits, self.groups, config, seed, path,
            actor=actor, progress=lambda _s: None)
        return path, result

    def saved_study(self, label, *, metadata_splits=None, archive_splits=None,
                    archive_state_indices=None):
        path = self.folder(label)
        metadata_splits = self.splits if metadata_splits is None else metadata_splits
        archive_splits = self.splits if archive_splits is None else archive_splits
        state_indices = np.array([list(self.groups[:i + 1]).count(group) - 1
                                  for i, group in enumerate(self.groups)])
        rows = [{"trajectory_id": group, "split": str(metadata_splits[i]),
                 "state_index": int(state_indices[i]), "object_id": 1,
                 "time_s": float(state_indices[i] * .02)} for i, group in enumerate(self.groups)]
        write_csv(path / "selected_frames.csv", rows)
        np.savez_compressed(path / "selected_arrays.npz", h=self.h, h1=self.h, h2=self.h,
            y=self.y, trajectory_id=self.groups, split=archive_splits,
            state_index=state_indices if archive_state_indices is None else archive_state_indices)
        (path / "configuration.json").write_text(json.dumps({"layer": "h1"}), encoding="utf-8")
        (path / "model").mkdir()
        with torch.random.fork_rng(devices=[]):
            actor = FrozenActor()
        torch.save(actor.state_dict(), path / "model" / "frozen_actor_state.pt")
        return path

    def test_zero_auxiliary_weight_exactly_reproduces_pure_sae(self):
        pure = fit_sae(self.h, self.splits, self.groups, self.sae, 4,
                       self.folder("pure"), progress=lambda _s: None)
        _, assisted = self.fit(AssistedConfig(sae=self.sae, state_weight=0.))
        for index in (2, 3, 4):
            np.testing.assert_array_equal(pure[index], assisted[index])
        for module in ("encoder", "decoder"):
            left, right = getattr(pure[0], module), getattr(assisted[0], module)
            torch.testing.assert_close(left.weight, right.weight, rtol=0, atol=0)
            torch.testing.assert_close(left.bias, right.bias, rtol=0, atol=0)
        self.assertEqual(pure[-1]["best_epoch"], assisted[-1]["best_epoch"])

    def test_nonzero_state_loss_reaches_the_encoder(self):
        _, zero = self.fit(AssistedConfig(sae=self.sae, state_weight=0.), label="zero")
        _, aided = self.fit(AssistedConfig(sae=self.sae, state_weight=1.), label="aided")
        delta = (zero[0].encoder.weight - aided[0].encoder.weight).detach().abs().max()
        self.assertGreater(float(delta), 1e-6)

    def test_validation_labels_do_not_enter_training_or_reconstruction_selection(self):
        config = AssistedConfig(sae=self.sae, state_weight=.3, selection="reconstruction")
        _, reference = self.fit(config, label="clean_validation")
        contaminated = self.y.copy()
        validation = self.splits == "validation"
        contaminated[validation, :2] = contaminated[validation, :2] * 1000 + 100
        contaminated[validation, 2] = 1 - contaminated[validation, 2]
        _, altered = self.fit(config, y=contaminated, label="changed_validation_labels")
        np.testing.assert_array_equal(reference[3], altered[3])
        np.testing.assert_array_equal(reference[4], altered[4])
        self.assertEqual(reference[-1]["best_epoch"], altered[-1]["best_epoch"])
        for key, value in reference[0].state_dict().items():
            torch.testing.assert_close(value, altered[0].state_dict()[key], rtol=0, atol=0)

    def test_zero_weight_labels_cannot_change_the_representation(self):
        config = AssistedConfig(sae=self.sae, state_weight=0.)
        _, reference = self.fit(config)
        other = self.y.copy()
        other[:, :2] = other[::-1, :2] * 100
        other[:, 2] = 1 - other[:, 2]
        _, altered = self.fit(config, y=other)
        np.testing.assert_array_equal(reference[3], altered[3])
        np.testing.assert_array_equal(reference[4], altered[4])

    def test_test_rows_are_rejected(self):
        splits = self.splits.astype("U32")
        splits[0] = "test_trajectory"
        with self.assertRaises(ValueError):
            self.fit(splits=splits)

    def test_checkpoint_restores_codes_dictionary_and_preprocessing(self):
        path, fitted = self.fit()
        model, standardizer, metadata = load_assisted_checkpoint(path / "best_model.pt")
        z, normalized_reconstruction = infer_sae(model, standardizer.transform(self.h),
                                                self.sae.batch_size)
        np.testing.assert_array_equal(z, fitted[3])
        np.testing.assert_array_equal(standardizer.inverse_transform(normalized_reconstruction),
                                      fitted[4])
        np.testing.assert_allclose(model.decoder.weight.detach().norm(dim=0), 1., atol=1e-6)
        self.assertTrue((z >= 0).all())
        self.assertTrue(((z > 0).sum(axis=1) <= self.sae.top_k).all())
        self.assertEqual(metadata["best_epoch"], fitted[-1]["best_epoch"])
        train = self.splits == "train"
        weights = sample_weights(self.groups[train])
        np.testing.assert_allclose(standardizer.mean,
            np.average(self.h[train], axis=0, weights=weights), rtol=0, atol=1e-12)
        target_scaler = metadata["target_scaler"]
        expected_mean = np.average(self.y[train, :2], axis=0, weights=weights)
        expected_std = np.sqrt(np.average((self.y[train, :2] - expected_mean) ** 2,
                                         axis=0, weights=weights))
        np.testing.assert_allclose(target_scaler["mean"], expected_mean, atol=1e-12)
        np.testing.assert_allclose(target_scaler["scale"], expected_std, atol=1e-12)
        self.assertAlmostEqual(target_scaler["contact_prevalence"],
                               np.average(self.y[train, 2], weights=weights))

    def test_target_scaler_preserves_binary_contact_and_protects_constant_target(self):
        y = np.array([[.1, .5, 0.], [.2, .5, 1.], [.3, .5, 1.]])
        w = np.array([1., 2., 1.])
        scaler = fit_target_scaler(y, w)
        transformed = transform_targets(y, scaler)
        np.testing.assert_allclose(scaler["mean"], [.2, .5], atol=1e-12)
        self.assertEqual(scaler["scale"][1], 1.)
        self.assertAlmostEqual(scaler["contact_prevalence"], .75)
        self.assertAlmostEqual(scaler["contact_positive_weight"], 1 / 3)
        np.testing.assert_array_equal(transformed[:, 2], y[:, 2])
        np.testing.assert_array_equal(transformed[:, 1], np.zeros(3))
        self.assertAlmostEqual(float(np.average(transformed[:, 0], weights=w)), 0.)
        before = dict(scaler)
        held_out = np.array([[100., 200., 0.]])
        transform_targets(held_out, scaler)
        self.assertEqual(before, scaler)
        for contact in (0., 1.):
            single_class = y.copy()
            single_class[:, 2] = contact
            state = fit_target_scaler(single_class, w)
            self.assertTrue(state["contact_single_class"])
            self.assertTrue(np.isfinite([state["contact_entropy"],
                                         state["contact_positive_weight"]]).all())

    def test_restricted_heads_only_backpropagate_through_their_allowed_coordinates(self):
        for mode in ("dense", "top4", "slots"):
            with self.subTest(mode=mode):
                torch.manual_seed(20)
                model = AssistedTopKSAE(64, 16, 4, head_mode=mode, head_seed=4)
                mask = model.effective_head_mask().detach()
                expected = 16 if mode == "dense" else 4
                np.testing.assert_array_equal(mask.sum(dim=1).numpy(), [expected] * 3)
                if mode == "slots":
                    self.assertTrue((mask.sum(dim=0) <= 1).all())
                    self.assertEqual(mask[0, :4].sum(), 4)
                    self.assertEqual(mask[1, 4:8].sum(), 4)
                    self.assertEqual(mask[2, 8:12].sum(), 4)
                z = torch.ones(5, 16, requires_grad=True)
                model.predict_state(z)[:, 0].sum().backward()
                self.assertGreater(float(z.grad.abs().max()), 0.)
                self.assertTrue((z.grad[:, mask[0] == 0] == 0).all())
                self.assertTrue((model.state_head.weight.grad[0, mask[0] == 0] == 0).all())
                self.assertTrue((model.state_head.weight.grad[1:] == 0).all())

    def test_head_initialization_and_checkpoint_loading_preserve_cpu_rng(self):
        from temp_scripts.sae_model import TopKSAE
        torch.manual_seed(12)
        pure = TopKSAE(64, 16, 4)
        expected_stream = torch.rand(4)
        torch.manual_seed(12)
        aided = AssistedTopKSAE(64, 16, 4, head_seed=999)
        torch.testing.assert_close(torch.rand(4), expected_stream, rtol=0, atol=0)
        torch.testing.assert_close(pure.encoder.weight, aided.encoder.weight, rtol=0, atol=0)
        path, _ = self.fit(AssistedConfig(sae=replace(self.sae, max_epochs=1)))
        torch.manual_seed(541)
        expected_stream = torch.rand(4)
        torch.manual_seed(541)
        load_assisted_checkpoint(path / "best_model.pt")
        torch.testing.assert_close(torch.rand(4), expected_stream, rtol=0, atol=0)

    def test_loss_variants_and_target_subsets_fit_finite_models(self):
        short = replace(self.sae, max_epochs=3)
        variants = [
            dict(regression_loss="huber"),
            dict(classification_loss="balanced"),
            dict(classification_loss="entropy"),
            dict(targets=(0,)), dict(targets=(1,)), dict(targets=(2,)),
            dict(head_mode="top4"), dict(head_mode="slots"),
            dict(selection="joint")]
        for options in variants:
            with self.subTest(**options):
                _, fitted = self.fit(AssistedConfig(sae=short, **options))
                self.assertTrue(np.isfinite(fitted[3]).all())
                self.assertTrue(np.isfinite(fitted[4]).all())

    def test_action_loss_updates_sae_but_keeps_actor_frozen(self):
        torch.manual_seed(310)
        actor = FrozenActor().requires_grad_(False).eval()
        fingerprint = parameter_fingerprint(actor)
        zero_config = AssistedConfig(sae=self.sae, state_weight=0.)
        _, zero = self.fit(zero_config, actor=actor)
        _, action = self.fit(replace(zero_config, action_weight=.4), actor=actor)
        self.assertEqual(parameter_fingerprint(actor), fingerprint)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in actor.parameters()))
        self.assertGreater(float((zero[0].encoder.weight - action[0].encoder.weight)
                                 .detach().abs().max()), 1e-6)

    def test_action_loss_requires_an_already_frozen_actor(self):
        config = AssistedConfig(sae=self.sae, action_weight=.1)
        with self.assertRaises(ValueError):
            self.fit(config)
        with self.assertRaises(ValueError):
            self.fit(config, actor=FrozenActor())

    def test_invalid_targets_and_losses_are_rejected(self):
        for options in (dict(targets=(3,)), dict(targets=(-1,)), dict(targets=(0, 0)),
                        dict(state_weight=-1.), dict(state_weight=float("nan")),
                        dict(regression_loss="unknown"), dict(head_mode="unknown")):
            with self.subTest(**options):
                with self.assertRaises(ValueError):
                    self.fit(AssistedConfig(sae=self.sae, **options))

    def test_supervised_rotation_is_weighted_orthogonal_reversible_and_train_fitted(self):
        train = self.splits == "train"
        weights = sample_weights(self.groups[train])
        for mode in ("global", "coordinate"):
            with self.subTest(preprocessing=mode):
                projection = fit_supervised_rotation(self.h[train], self.y[train], weights, mode)
                rotation = projection["rotation"]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(64), rtol=0, atol=1e-12)
                codes = transform_supervised_rotation(self.h, projection)
                np.testing.assert_allclose(inverse_supervised_rotation(codes, projection),
                                           self.h, rtol=0, atol=1e-12)
                np.testing.assert_allclose(projection["h_mean"],
                    np.average(self.h[train], axis=0, weights=weights), rtol=0, atol=1e-12)
                np.testing.assert_allclose(projection["y_mean"],
                    np.average(self.y[train], axis=0, weights=weights), rtol=0, atol=1e-12)
                self.assertLessEqual(int(projection["supervised_rank"]), 3)
                before = {key: value.copy() for key, value in projection.items()}
                transform_supervised_rotation(np.ones((2, 64)) * 1e8, projection)
                for key in projection:
                    np.testing.assert_array_equal(projection[key], before[key])
                rank = int(projection["supervised_rank"])
                centered_codes = codes[train] - np.average(codes[train], axis=0, weights=weights)
                covariance = centered_codes.T @ (weights[:, None] * centered_codes) / weights.sum()
                complement_covariance = covariance[rank:, rank:]
                off_diagonal = complement_covariance - np.diag(np.diag(complement_covariance))
                np.testing.assert_allclose(off_diagonal, 0., rtol=0, atol=1e-12)

    def test_supervised_rotation_handles_no_label_variation_without_fabricating_axes(self):
        train = self.splits == "train"
        constant = np.tile([.1, .2, 0.], (train.sum(), 1))
        projection = fit_supervised_rotation(self.h[train], constant,
                                             sample_weights(self.groups[train]))
        self.assertEqual(int(projection["supervised_rank"]), 0)
        self.assertTrue(np.isfinite(projection["rotation"]).all())

    def test_ridge_rotation_preserves_all_three_supervised_readouts_on_unseen_rows(self):
        train = self.splits == "train"
        weights = sample_weights(self.groups[train])
        for mode in ("coordinate", "global"):
            with self.subTest(preprocessing=mode):
                projection = fit_ridge_rotation(self.h[train], self.y[train], weights,
                                                alpha=.1, preprocessing=mode)
                rotation = projection["rotation"]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(64), rtol=0, atol=1e-12)
                codes = transform_supervised_rotation(self.h, projection)
                np.testing.assert_allclose(inverse_supervised_rotation(codes, projection),
                                           self.h, rtol=0, atol=1e-12)
                rank = int(projection["supervised_rank"])
                self.assertLessEqual(rank, 3)
                coefficients = projection["ridge_coefficients"]
                full_predictions = ((self.h - projection["h_mean"]) / projection["h_scale"]) @ coefficients
                reduced_predictions = codes[:, :rank] @ (rotation[:, :rank].T @ coefficients)
                np.testing.assert_allclose(reduced_predictions, full_predictions, rtol=0, atol=1e-12)

    def test_rotation_holdout_requires_matching_original_plan_before_numeric_reads(self):
        root = self.saved_study("rotation_no_plan")
        held = root / "holdout_synthetic"
        held.mkdir()
        with patch("temp_scripts.state_assisted_controls.np.load",
                   side_effect=AssertionError("Numeric holdout opened before manifest checks")):
            with self.assertRaisesRegex(ValueError, "Freeze"):
                evaluate_rotation_holdout(root, held)
            plan = {"rotation_controls": [{"path": "controls/nonexistent", "targets": ["distance_m"],
                                            "artifact_hashes": {}}]}
            (root / "frozen_evaluation_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "copied frozen plan"):
                evaluate_rotation_holdout(root, held)

    def test_rotation_synthetic_holdout_applies_frozen_projection_and_probes_without_fit(self):
        root = self.saved_study("rotation_frozen")
        data, _, _ = load_study(root)
        train = self.splits == "train"
        control = root / "controls" / "ridge_synthetic"
        control.mkdir(parents=True)
        projection = fit_ridge_rotation(self.h[train], self.y[train],
                                         sample_weights(self.groups[train]), alpha=.1)
        np.savez_compressed(control / "projection.npz", **projection)
        codes = transform_supervised_rotation(self.h, projection)
        export_state_probes(control, data, codes, "h1", "trajectory", quick=True)
        (control / "summary.json").write_text(json.dumps({"layer": "h1"}), encoding="utf-8")
        held = root / "holdout_synthetic"
        held.mkdir()
        validation_rows = [r for r in data.rows if r["split"] == "validation"]
        held_rows = [{**r, "split": "test_trajectory", "trajectory_id": "held_" + r["trajectory_id"]}
                     for r in validation_rows]
        write_csv(held / "selected_frames.csv", held_rows)
        np.savez_compressed(held / "selected_arrays.npz", h=self.h[~train], h1=self.h[~train],
            h2=self.h[~train], y=self.y[~train],
            split=np.array([r["split"] for r in held_rows]),
            trajectory_id=np.array([r["trajectory_id"] for r in held_rows]),
            state_index=np.array([r["state_index"] for r in held_rows]))
        entry = {"path": "controls/ridge_synthetic", "targets": ["distance_m", "opening_rad", "any_contact"],
                 "artifact_hashes": {name: sha256(control / name) for name in
                    ("projection.npz", "fitted_probes.joblib", "summary.json")}}
        plan = {"rotation_controls": [entry]}
        for path in (root, held):
            (path / "frozen_evaluation_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("Refitted scaler")), \
             patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("Refitted ridge")), \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Refitted classifier")), \
             patch("temp_scripts.state_assisted_controls.fit_ridge_rotation", side_effect=AssertionError("Refitted projection")):
            folder, summary = evaluate_rotation_holdout(root, held)
        self.assertTrue(summary["no_refitting"])
        self.assertFalse(summary["physical_simulation"])
        self.assertEqual(summary["controls"][0]["frozen_artifact_hashes"], entry["artifact_hashes"])
        with np.load(folder / "ridge_synthetic" / "codes.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["z"], transform_supervised_rotation(self.h[~train], projection))

    def test_saved_study_rejects_test_rows_in_either_archive_or_metadata(self):
        splits = self.splits.astype("U32")
        splits[0] = "test_trajectory"
        for side in ("metadata", "archive"):
            with self.subTest(side=side):
                kwargs = {"metadata_splits" if side == "metadata" else "archive_splits": splits}
                path = self.saved_study("test_" + side, **kwargs)
                with self.assertRaises(ValueError):
                    load_study(path)

    def test_saved_study_rejects_shuffled_row_identity(self):
        indices = np.array([list(self.groups[:i + 1]).count(group) - 1
                            for i, group in enumerate(self.groups)])
        indices[[0, 1]] = indices[[1, 0]]
        path = self.saved_study("misaligned_rows", archive_state_indices=indices)
        with self.assertRaises(ValueError):
            load_study(path)

    def test_saved_study_roundtrip_preserves_source_and_freezes_actor(self):
        path = self.saved_study("valid_study")
        data, actor, config = load_study(path)
        np.testing.assert_array_equal(data.h1, self.h)
        np.testing.assert_array_equal(data.h2, self.h)
        np.testing.assert_array_equal(data.y, self.y)
        np.testing.assert_array_equal(data.groups, self.groups)
        np.testing.assert_array_equal(data.splits, self.splits)
        self.assertEqual(config["layer"], "h1")
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in actor.parameters()))

    def test_resume_skips_matching_configuration_and_rejects_changed_parameters(self):
        path = self.saved_study("resume")
        config = AssistedConfig(sae=self.sae, name="resume_pair")
        experiment = path / "resume_pair_seed4"
        experiment.mkdir()
        summary = experiment / "summary.json"
        summary.write_text(json.dumps({"configuration": asdict(config)}), encoding="utf-8")
        before = summary.read_bytes()
        with patch("temp_scripts.state_assisted_model.fit_assisted",
                   side_effect=AssertionError("Completed runs must not train again")):
            run_experiments(path, [config], seeds=(4,), progress=lambda _s: None)
            self.assertEqual(summary.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                run_experiments(path, [replace(config, state_weight=.3)], seeds=(4,),
                                progress=lambda _s: None)
        self.assertEqual(summary.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
