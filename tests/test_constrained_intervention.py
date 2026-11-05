import unittest
import numpy as np

from temp_scripts.constrained_intervention import (match_projected_norm, project_delta,
                                                  stable_random_directions)


class TestConstrainedIntervention(unittest.TestCase):
    def test_zero_relu_coordinates_cannot_move_negative(self):
        h = np.array([0., 1., 0.])
        d, status = match_projected_norm(h, np.array([-1., -1., 1.]), .7)
        self.assertEqual(status, 'ok')
        self.assertGreaterEqual((h + d).min(), 0.)
        self.assertAlmostEqual(np.linalg.norm(d), .7, places=9)
        self.assertEqual(d[0], 0.)

    def test_negative_only_ray_exposes_insufficient_capacity(self):
        h = np.array([0., .1])
        d, status = match_projected_norm(h, np.array([-1., -1.]), .3)
        self.assertEqual(status, 'insufficient_feasible_capacity')
        np.testing.assert_array_equal(d, 0.)

    def test_random_feasible_matched_controls_after_projection(self):
        rng = np.random.default_rng(2718)
        for _ in range(100):
            h = np.maximum(rng.normal(size=64), 0)
            direction = rng.normal(size=64)
            delta, status = match_projected_norm(h, direction, .5)
            self.assertEqual(status, 'ok')
            self.assertGreaterEqual((h + delta).min(), 0.)
            self.assertAlmostEqual(np.linalg.norm(delta), .5, places=8)

    def test_shared_random_keys_are_reproducible_and_do_not_touch_global_rng(self):
        before = np.random.get_state()
        a = stable_random_directions('same_origin', 64)
        b = stable_random_directions('same_origin', 64)
        after = np.random.get_state()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertFalse(np.array_equal(a, stable_random_directions('other_origin', 64)))

    def test_reject_negative_source_h(self):
        with self.assertRaises(ValueError):
            project_delta(np.array([-1., 2.]), np.ones(2))


if __name__ == '__main__':
    unittest.main()
