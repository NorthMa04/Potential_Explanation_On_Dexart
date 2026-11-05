import unittest
from temp_scripts.feature_analysis import stratified_origin_interval


class TestOriginUncertainty(unittest.TestCase):
    def test_no_false_zero_width_interval_for_one_origin_per_object(self):
        rows = [{'object': 1, 'effect_degrees': 1.}, {'object': 2, 'effect_degrees': 5.}]
        result = stratified_origin_interval(rows)
        self.assertEqual(result['mean'], 3.)
        self.assertIsNone(result['ci95'])

    def test_paired_origin_mean_and_known_bootstrap_support(self):
        rows = [{'object': 1, 'effect_degrees': 0.}, {'object': 1, 'effect_degrees': 2.}]
        result = stratified_origin_interval(rows, draws=2000)
        self.assertEqual(result['mean'], 1.)
        self.assertEqual(result['ci95'], [0., 2.])


if __name__ == '__main__':
    unittest.main()
