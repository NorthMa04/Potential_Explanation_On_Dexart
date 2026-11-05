import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from temp_scripts.combined_intervention import CombinedEditor, group_definitions
from temp_scripts.project_paths import OUTPUT_ROOT


class IdentitySAE:
    def __init__(self):
        self.w = {'decoder_weight': np.eye(64), 'scale': np.ones(64)}

    def infer(self, h):
        z = np.maximum(np.atleast_2d(h), 0)
        return z, z, np.zeros((len(z), 3))


class TestCombinedIntervention(unittest.TestCase):
    def setUp(self):
        parent = OUTPUT_ROOT / '_test'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        metadata = {'models': {'pure_seed0': {'opening_coordinates': list(range(6))},
                               'assisted_seed0': {'opening_coordinates': list(range(6))}},
                    'groups': group_definitions('top'), 'relative_norm_cap': .1, 'suppression_fraction': 1.}
        (folder / 'combined_intervention.json').write_text(json.dumps(metadata), encoding='utf-8')
        np.savez(folder / 'combined_directions.npz', ridge_oriented=np.eye(64)[:, :6])
        self.editor = CombinedEditor(folder, {k: IdentitySAE() for k in metadata['models']})

    def test_group_membership_and_factorial_deduplication(self):
        groups = group_definitions('factorial')
        self.assertEqual(len({tuple(v) for v in groups.values()}), len(groups))
        self.assertEqual(groups['top3'], [0, 1, 2])
        self.assertEqual(groups['AC'], [0, 1, 4, 5])

    def test_nominal_group_changes_only_requested_coordinates_in_identity_model(self):
        h = np.ones(64, dtype=np.float32)
        new, detail = self.editor.edit(h, 'pure_seed0', 'top4', 'nominal', 'target', 'origin')
        self.assertEqual(detail['status'], 'ok')
        self.assertTrue(np.all(new[:4] < h[:4]))
        np.testing.assert_array_equal(new[4:], h[4:])
        self.assertAlmostEqual(detail['actual_norm'], .1 * np.linalg.norm(h), places=6)
        self.assertEqual(detail['negative_fraction'], 0)

    def test_common_budget_is_equal_across_models_groups_and_random_controls(self):
        h = np.ones(64, dtype=np.float32)
        budgets = []
        for model in self.editor.meta['models']:
            for group in self.editor.meta['groups']:
                for control in ('target', 'random0', 'random1', 'random2', 'ridge'):
                    new, detail = self.editor.edit(h, model, group, 'common_norm', control, 'origin')
                    self.assertEqual(detail['status'], 'ok')
                    self.assertTrue(np.all(new >= 0))
                    self.assertLess(detail['norm_error'], 2e-6)
                    budgets.append(detail['budget'])
        self.assertEqual(len(set(budgets)), 1)

    def test_inactive_selected_code_is_explicit_noop(self):
        h = np.ones(64, dtype=np.float32)
        h[0] = 0
        new, detail = self.editor.edit(h, 'pure_seed0', 'top1', 'common_norm', 'random0', 'origin')
        self.assertEqual(detail['status'], 'inactive_or_no_feasible_budget')
        np.testing.assert_array_equal(new, h)
        self.assertLess(detail['common_supported_candidates'], detail['common_total_candidates'])

    def test_random_combinations_are_fixed_and_exclude_semantic_candidates(self):
        h = np.ones(64, dtype=np.float32)
        a, da = self.editor.edit(h, 'pure_seed0', 'top6', 'common_norm', 'random_group0', 'origin')
        b, db = self.editor.edit(h, 'pure_seed0', 'top6', 'common_norm', 'random_group0', 'origin')
        np.testing.assert_array_equal(a, b)
        self.assertEqual(da['control_indices'], db['control_indices'])
        self.assertEqual(len(da['control_indices']), 6)
        self.assertFalse(set(da['control_indices']) & set(range(6)))

    def test_updated_closed_loop_state_recomputes_budget(self):
        _, first = self.editor.edit(np.ones(64), 'pure_seed0', 'top2', 'common_norm', 'target', 'origin')
        _, second = self.editor.edit(np.ones(64) * 2, 'pure_seed0', 'top2', 'common_norm', 'target', 'origin')
        self.assertAlmostEqual(second['budget'], 2 * first['budget'])

    def test_zero_and_invalid_inputs(self):
        new, detail = self.editor.edit(np.zeros(64), 'pure_seed0', 'top6', 'nominal', 'target', 'origin')
        np.testing.assert_array_equal(new, np.zeros(64))
        self.assertNotEqual(detail['status'], 'ok')
        for invalid in (np.ones(63), -np.ones(64), np.full(64, np.nan)):
            with self.assertRaises(ValueError):
                self.editor.edit(invalid, 'pure_seed0', 'top1', 'nominal', 'target', 'origin')


if __name__ == '__main__':
    unittest.main()
