"""Estimated beat intervals are quality-masked independently of file acceptance."""

import unittest

import numpy as np

from features.performance_time import BeatTimeMap, mask_estimated_beat_intervals


class EstimatedBeatIntervalsTest(unittest.TestCase):
    def estimated(self, q, seconds, valid=None):
        return BeatTimeMap(np.array(q), np.array(seconds), valid,
                           {'granularity': 'beat', 'source': 'fixture', 'estimated': True})

    def test_collapsed_and_long_intervals_are_unknown_without_anchor_changes(self):
        original = self.estimated(np.arange(5), [0., .5, .5 + 1 / 240, 1., 7.])
        result = mask_estimated_beat_intervals(original)
        np.testing.assert_array_equal(result.valid, [True, False, True, False])
        np.testing.assert_array_equal(result.q, original.q)
        np.testing.assert_array_equal(result.seconds, original.seconds)
        self.assertTrue(original.valid.all())
        self.assertEqual(result.provenance['interval_validation']['out_of_range_indices'], [1, 3])
        self.assertIsNone(result.time_at(1.5))
        self.assertIsNone(result.expected_duration(0., 2.))
        self.assertIsNotNone(result.expected_duration(0., 1.))

    def test_existing_invalid_intervals_are_never_reenabled(self):
        result = mask_estimated_beat_intervals(self.estimated(
            [0, 1, 2], [0., .5, 1.], np.array([False, True])))
        np.testing.assert_array_equal(result.valid, [False, True])
        self.assertEqual(result.provenance['interval_validation']['source_invalid_indices'], [0])

    def test_bounds_are_inclusive_and_in_seconds_per_quarter(self):
        for tau in (.08, 5.):
            for width in (.25, 1., 2.):
                with self.subTest(tau=tau, width=width):
                    result = mask_estimated_beat_intervals(self.estimated([0., width], [0., tau * width]))
                    self.assertTrue(result.valid[0])

    def test_reference_annotations_require_a_different_quality_policy(self):
        annotated = BeatTimeMap(np.array([0, 1]), np.array([0., .004]),
                                provenance={'granularity': 'beat', 'source': 'reference'})
        with self.assertRaisesRegex(ValueError, 'estimated'):
            mask_estimated_beat_intervals(annotated)
