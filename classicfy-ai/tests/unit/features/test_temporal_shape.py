import unittest

import numpy as np

from features.models import BeatSequence
from features.temporal_shape import adjacent_changes, fit_change_threshold, summarize_temporal_shape


class TemporalShapeTests(unittest.TestCase):
    def summarize(self, values, threshold=1., tolerance=1e-6):
        return summarize_temporal_shape(BeatSequence.from_optional(values),
                                        large_threshold=threshold, direction_tolerance=tolerance)

    def test_hand_computed_deltas_and_patterns(self):
        # Deltas +1,+2,0,-2,-1,+2; directions ++0--+.
        result = self.summarize([0, 1, 3, 3, 1, 0, 2])
        self.assertEqual(result['valid_pairs'], 6)
        self.assertEqual(result['large_count'], 3)
        self.assertEqual(result['large_ratio'], .5)
        self.assertEqual(result['rise_count'], 3)
        self.assertEqual(result['fall_count'], 2)
        self.assertEqual(result['flat_count'], 1)
        self.assertEqual(result['valid_triplets'], 5)
        self.assertEqual(result['moving_triplets'], 3)
        self.assertEqual(result['reversal_count'], 1)
        self.assertAlmostEqual(result['reversal_ratio'], 1 / 3)
        self.assertEqual(result['rise_to_rise_count'], 1)
        self.assertEqual(result['fall_to_rise_ratio'], .2)
        self.assertAlmostEqual(result['mean_abs_delta'], 8 / 6)
        self.assertAlmostEqual(result['rms_delta'], np.sqrt(14 / 6))
        self.assertAlmostEqual(result['delta_p50'], .5)
        self.assertEqual(sum(result[f'{d}_count'] for d in ('rise', 'flat', 'fall')), 6)
        self.assertEqual(sum(result[f'{a}_to_{b}_count'] for a in ('rise', 'flat', 'fall')
                             for b in ('rise', 'flat', 'fall')), 5)

    def test_masked_and_nonfinite_beats_break_adjacency(self):
        seq = BeatSequence([0, 1, 100, 4, 5, np.inf, 8, 9, np.nan],
                           [True, True, False, True, True, True, True, True, True])
        changes = adjacent_changes(seq)
        np.testing.assert_array_equal(changes.mask, [1, 0, 0, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(changes.values[changes.mask], [1, 1, 1])
        result = summarize_temporal_shape(seq, large_threshold=0)
        self.assertEqual(result['valid_beats'], 6)
        self.assertEqual(result['valid_triplets'], 0)
        self.assertTrue(np.isnan(result['reversal_ratio']))

    def test_empty_single_and_all_missing_are_undefined(self):
        for values in ([], [4], [None, None]):
            result = self.summarize(values)
            self.assertEqual(result['valid_pairs'], 0)
            self.assertTrue(np.isnan(result['large_ratio']))
            self.assertTrue(np.isnan(result['delta_p95']))
            self.assertTrue(np.isnan(result['rms_delta']))

    def test_constant_has_zero_changes_but_no_moving_reversals(self):
        result = self.summarize([3, 3, 3, 3], threshold=0)
        self.assertEqual(result['large_ratio'], 0)
        self.assertEqual(result['rms_delta'], 0)
        self.assertEqual(result['flat_ratio'], 1)
        self.assertEqual(result['flat_to_flat_ratio'], 1)
        self.assertTrue(np.isnan(result['reversal_ratio']))

    def test_strict_boundaries_and_flat_interrupts_reversal(self):
        result = self.summarize([0, 1, 2, 1], threshold=1, tolerance=1)
        self.assertEqual(result['large_ratio'], 0)
        self.assertEqual(result['flat_ratio'], 1)
        result = self.summarize([0, 2, 2, 0])
        self.assertEqual(result['reversal_count'], 0)
        self.assertEqual(result['moving_triplets'], 0)

    def test_offset_invariance_and_order_information(self):
        a = self.summarize([0, 1, 2, 1, 0])
        b = self.summarize([100, 101, 102, 101, 100])
        for key in a:
            np.testing.assert_allclose(a[key], b[key], equal_nan=True)
        # Same value multiset / mean / p95-p5, but different change shape.
        c = self.summarize([0, 2, 0, 1, 1])
        self.assertNotEqual(a['large_ratio'], c['large_ratio'])
        self.assertNotEqual(a['rms_delta'], c['rms_delta'])

    def test_fixed_reference_threshold_not_performance_percentile(self):
        reference = [BeatSequence.from_optional([0, 1, 3]), BeatSequence.from_optional([0, 3])]
        threshold = fit_change_threshold(reference, quantile=.5)
        self.assertEqual(threshold.value, 2)
        self.assertEqual(threshold.pair_count, 3)
        self.assertEqual(self.summarize([0, 1, 2], threshold.value)['large_ratio'], 0)
        self.assertEqual(self.summarize([0, 3, 6], threshold.value)['large_ratio'], 1)
        self.assertEqual(fit_change_threshold([BeatSequence.from_optional([2, 2])]).value, 0)

    def test_invalid_parameters_and_no_reference_pairs(self):
        for value in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                self.summarize([0, 1], threshold=value)
            with self.assertRaises(ValueError):
                self.summarize([0, 1], tolerance=value)
        for quantile in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                fit_change_threshold([], quantile=quantile)
        for reference in ([], [BeatSequence.from_optional([0, None, 1])]):
            with self.assertRaises(ValueError):
                fit_change_threshold(reference)

    def test_no_input_mutation_or_integer_subtraction_overflow(self):
        values = np.array([120, -120], dtype=np.int8)
        seq = BeatSequence(values, [True, True])
        changes = adjacent_changes(seq)
        self.assertEqual(changes.values[0], -240)
        self.assertFalse(changes.values.flags.writeable)
        np.testing.assert_array_equal(seq.values, values)
        with self.assertRaises(ValueError):
            adjacent_changes(BeatSequence([-1e308, 1e308], [True, True]))


if __name__ == '__main__':
    unittest.main()
