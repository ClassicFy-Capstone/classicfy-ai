import unittest

import numpy as np

from features.models import BeatSequence
from features.temporal_shape import adjacent_changes
from features.temporal_duration import (lagged_correlation, observed_runs,
                                        summarize_temporal_duration)


class TemporalDurationTests(unittest.TestCase):
    def sequence(self, values):
        return BeatSequence.from_optional(values)

    def test_hand_computed_level_runs_and_censoring(self):
        sequence = self.sequence([1, 2, 2, 1, -1, None, -2, -3, 0, 0])
        runs = observed_runs(sequence)
        self.assertEqual([(r.state, r.start, r.stop, r.length) for r in runs],
                         [('positive', 0, 4, 4), ('negative', 4, 5, 1),
                          ('negative', 6, 8, 2), ('zero', 8, 10, 2)])
        self.assertEqual([(r.left_censored, r.right_censored) for r in runs],
                         [(True, False), (False, True), (True, False), (False, True)])
        metrics = summarize_temporal_duration(sequence)
        self.assertEqual(metrics['level_negative_run_mean'], 1.5)
        self.assertEqual(metrics['level_negative_run_max'], 2)
        self.assertAlmostEqual(metrics['level_negative_run_p95'], 1.95)

    def test_direction_duration_counts_steps_not_values(self):
        metrics = summarize_temporal_duration(self.sequence([1, 2, 2, 1, -1, None, -2, -3, 0, 0]))
        self.assertEqual(metrics['valid_pairs'], 7)
        self.assertEqual(metrics['direction_fall_run_count'], 2)
        self.assertEqual(metrics['direction_fall_run_entries'], 3)
        self.assertEqual(metrics['direction_fall_run_median'], 1.5)
        self.assertEqual(metrics['direction_fall_left_censored_count'], 1)
        self.assertEqual(metrics['direction_fall_right_censored_count'], 1)
        monotonic = summarize_temporal_duration(self.sequence([0, 1, 2, 3, 4]))
        self.assertEqual(monotonic['direction_rise_run_max'], 4)
        self.assertEqual(monotonic['level_positive_run_max'], 4)

    def test_constant_runs_are_real_and_absent_runs_are_nan(self):
        metrics = summarize_temporal_duration(self.sequence([3] * 7))
        self.assertEqual(metrics['direction_flat_run_max'], 6)
        self.assertEqual(metrics['level_positive_run_max'], 7)
        self.assertEqual(metrics['direction_flat_left_censored_count'], 1)
        self.assertEqual(metrics['direction_flat_right_censored_count'], 1)
        self.assertEqual(metrics['direction_rise_run_count'], 0)
        self.assertTrue(np.isnan(metrics['direction_rise_run_mean']))

    def test_empty_missing_and_singleton(self):
        for values in ([], [None, None], [np.nan, np.inf]):
            seq = self.sequence(values)
            self.assertEqual(observed_runs(seq), ())
            result = lagged_correlation(seq, 1, min_pairs=3)
            self.assertEqual(result['valid_pairs'], 0)
            self.assertEqual(result['status'], 'insufficient_pairs')
            self.assertTrue(np.isnan(result['correlation']))
        single = summarize_temporal_duration(self.sequence([1]))
        self.assertEqual(single['level_positive_run_max'], 1)
        self.assertEqual(single['direction_rise_run_count'], 0)

    def test_mask_nonfinite_and_strict_tolerance(self):
        seq = BeatSequence([1, 9, 1, np.inf, 1, np.nan, 1], [1, 0, 1, 1, 1, 1, 1])
        self.assertEqual([r.length for r in observed_runs(seq)], [1, 1, 1, 1])
        self.assertTrue(all(r.left_censored and r.right_censored for r in observed_runs(seq)))
        self.assertEqual([r.state for r in observed_runs(self.sequence([-1, 0, 1]), tolerance=1)], ['zero'])

    def test_linear_persistence_is_not_a_repetition_proof(self):
        seq = self.sequence(list(range(40)))
        self.assertAlmostEqual(lagged_correlation(seq, 4)['correlation'], 1)
        result = lagged_correlation(adjacent_changes(seq), 4)
        self.assertEqual(result['status'], 'constant_pairs')
        self.assertTrue(np.isnan(result['correlation']))

    def test_periodic_and_alternating_sequences(self):
        seq = self.sequence([0, 1, 0, -1] * 20)
        self.assertAlmostEqual(lagged_correlation(seq, 4)['correlation'], 1)
        self.assertAlmostEqual(lagged_correlation(seq, 2)['correlation'], -1)
        alternating = self.sequence([1, -1] * 30)
        self.assertAlmostEqual(lagged_correlation(alternating, 1)['correlation'], -1)
        self.assertAlmostEqual(lagged_correlation(alternating, 2)['correlation'], 1)

    def test_lag_pairs_cannot_cross_an_interior_gap(self):
        seq = self.sequence([0, 2, 4, None, 2, 4, 6, 8])
        result = lagged_correlation(seq, 2, min_pairs=3)
        self.assertEqual(result['possible_pairs'], 6)
        self.assertEqual(result['valid_pairs'], 3)
        self.assertEqual(result['pair_coverage'], .5)
        self.assertAlmostEqual(result['correlation'], 1)
        delta = lagged_correlation(adjacent_changes(seq), 2, min_pairs=3)
        self.assertEqual(delta['valid_pairs'], 1)

    def test_matches_independent_pearson_and_positive_affine_transform(self):
        values = np.array([2, 5, -1, 8, 3, 0, 4, -3, 2, 7, 1, -2], dtype=float)
        for lag in (1, 2, 4):
            a = lagged_correlation(self.sequence(values), lag, min_pairs=3)
            b = lagged_correlation(self.sequence(values * 2 + 50), lag, min_pairs=3)
            self.assertAlmostEqual(a['correlation'], np.corrcoef(values[:-lag], values[lag:])[0, 1])
            self.assertAlmostEqual(a['correlation'], b['correlation'])
        before = values.copy()
        summarize_temporal_duration(self.sequence(values))
        np.testing.assert_array_equal(values, before)

    def test_support_variance_and_large_lag_status(self):
        seq = self.sequence([1] * 40)
        self.assertEqual(lagged_correlation(seq, 1)['status'], 'constant_pairs')
        self.assertEqual(lagged_correlation(seq, 16)['status'], 'insufficient_pairs')
        result = lagged_correlation(seq, 40)
        self.assertEqual(result['possible_pairs'], 0)
        self.assertTrue(np.isnan(result['pair_coverage']))
        tiny = self.sequence(np.arange(40) * 1e-14)
        self.assertEqual(lagged_correlation(tiny, 1)['status'], 'constant_pairs')

    def test_invalid_settings(self):
        seq = self.sequence([0, 1, 2, 3])
        for lag in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                lagged_correlation(seq, lag)
        for minimum in (2, 1.5, True):
            with self.assertRaises(ValueError):
                lagged_correlation(seq, 1, min_pairs=minimum)
        for tolerance in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                observed_runs(seq, tolerance=tolerance)
            with self.assertRaises(ValueError):
                lagged_correlation(seq, 1, std_tolerance=tolerance)
        with self.assertRaises(ValueError):
            observed_runs(seq, labels=('a', 'a', 'b'))


if __name__ == '__main__':
    unittest.main()
