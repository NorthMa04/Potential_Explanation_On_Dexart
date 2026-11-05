"""Synthetic concentration-head mask, gradient and checkpoint invariants."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from temp_scripts.project_paths import OUTPUT_ROOT
from temp_scripts.sae_model import SAEConfig
from temp_scripts.sae_training import infer_sae
from temp_scripts.state_assisted_model import (AssistedConfig, AssistedTopKSAE, HEAD_MODES,
    fit_assisted, load_assisted_checkpoint)


class ConcentrationHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.area = OUTPUT_ROOT / "_test" / ("assisted_heads_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.area.mkdir(parents=True)

    def test_each_head_has_exact_connections_disjoint_slots_and_masked_gradients(self):
        for mode, count in (("dense", 16), ("top1", 1), ("top2", 2), ("top4", 4),
                            ("slots1", 1), ("slots2", 2), ("slots", 4)):
            with self.subTest(mode=mode):
                model = AssistedTopKSAE(64, 16, 4, mode, 17)
                z = torch.ones(5, 16, requires_grad=True)
                mask = model.effective_head_mask()
                self.assertEqual(mask.sum(dim=1).tolist(), [count] * 3)
                if mode.startswith("slots"):
                    self.assertTrue((mask.sum(dim=0) <= 1).all())
                    expected = torch.zeros_like(mask)
                    for target in range(3):
                        expected[target, target * count:(target + 1) * count] = 1.
                    self.assertTrue(torch.equal(mask, expected))
                if mode.startswith("top"):
                    expected = torch.zeros_like(mask).scatter(1, model.state_head.weight.detach().abs().topk(count, dim=1).indices, 1.)
                    self.assertTrue(torch.equal(mask, expected))
                model.predict_state(z).sum().backward()
                self.assertTrue(torch.equal(model.state_head.weight.grad, 5. * mask))
                self.assertTrue(torch.isfinite(z.grad).all())

    def test_slot_width_constraints_follow_three_target_groups(self):
        for mode, minimum in (("slots1", 3), ("slots2", 6), ("slots", 12)):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    AssistedConfig(sae=SAEConfig(width=minimum - 1, top_k=1), head_mode=mode).validate()
                AssistedConfig(sae=SAEConfig(width=minimum, top_k=1), head_mode=mode).validate()
                with self.assertRaises(ValueError):
                    AssistedTopKSAE(64, minimum - 1, 1, mode)
                model = AssistedTopKSAE(64, minimum, 1, mode)
                self.assertEqual(model.effective_head_mask().shape, (3, minimum))

    def test_all_head_modes_reload_exactly_after_supervised_training(self):
        rng = np.random.default_rng(97)
        x = rng.normal(size=(24, 64)).astype(np.float32)
        y = np.column_stack((x[:, 0], x[:, 1], (x[:, 2] > 0).astype(float)))
        splits = np.array(["train"] * 18 + ["validation"] * 6)
        groups = np.repeat(np.arange(8).astype(str), 3)
        cfg = SAEConfig(width=16, top_k=4, max_epochs=2, patience=2, batch_size=5)
        initial_hashes = set()
        for mode in HEAD_MODES:
            with self.subTest(mode=mode):
                folder = self.area / mode
                folder.mkdir()
                model, scaler, normalized, z, reconstructed, result = fit_assisted(x, y, splits, groups,
                    AssistedConfig(sae=cfg, head_mode=mode), 7, folder, progress=lambda _: None)
                loaded, loaded_scaler, payload = load_assisted_checkpoint(folder / "best_model.pt")
                again_z, again_x = infer_sae(loaded, loaded_scaler.transform(x), cfg.batch_size)
                np.testing.assert_array_equal(z, again_z)
                np.testing.assert_array_equal(reconstructed, loaded_scaler.inverse_transform(again_x))
                with torch.no_grad():
                    self.assertTrue(torch.equal(model.predict_state(torch.tensor(z)), loaded.predict_state(torch.tensor(z))))
                self.assertEqual(payload["spec"]["head_mode"], mode)
                self.assertTrue(torch.equal(model.effective_head_mask(), loaded.effective_head_mask()))
                initial_hashes.add(result["initial_sae_sha256"])
        self.assertEqual(len(initial_hashes), 1)


if __name__ == "__main__":
    unittest.main()
