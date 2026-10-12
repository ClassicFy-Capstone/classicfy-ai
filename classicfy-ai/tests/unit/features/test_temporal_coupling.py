import sys
from pathlib import Path
import unittest

import numpy as np

from features.models import BeatSequence
from features.temporal_shape import adjacent_changes
from features.temporal_coupling import (cross_channel_correlation, joint_directions,
                                       phase_shift_control, contiguous_segments, pearson)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_temporal_coupling import stable_contrast, fit_descriptor_scales, single_temporal_distances


class TemporalCouplingTests(unittest.TestCase):
    def sequence(self, values):
        return BeatSequence.from_optional(values)

    def test_alignment_adds_information_with_identical_marginal_statistics(self):
        x = np.tile([0., 1., 0., -1.], 40)
        y = -x
        for values in (x, y):
            self.assertEqual(values.mean(), 0)
            np.testing.assert_allclose(np.percentile(values, [5, 95]), [-1, 1])
        dx, dy = adjacent_changes(self.sequence(x)), adjacent_changes(self.sequence(y))
        self.assertAlmostEqual(np.mean(dx.values ** 2), np.mean(dy.values ** 2))
        for lag in (1, 8):
            self.assertAlmostEqual(np.corrcoef(dx.values[:-lag], dx.values[lag:])[0, 1],
                                   np.corrcoef(dy.values[:-lag], dy.values[lag:])[0, 1])
        self.assertAlmostEqual(cross_channel_correlation(dx, dx)['correlation'], 1)
        self.assertAlmostEqual(cross_channel_correlation(dx, dy)['correlation'], -1)

    def test_signed_lag_convention_and_swapping(self):
        x = np.random.default_rng(11).normal(size=150)
        y = np.r_[0., 0., x[:-2]]
        left, right = self.sequence(x), self.sequence(y)
        self.assertAlmostEqual(cross_channel_correlation(left, right, 2)['correlation'], 1)
        self.assertAlmostEqual(cross_channel_correlation(right, left, -2)['correlation'], 1)
        self.assertLess(abs(cross_channel_correlation(left, right, -2)['correlation']), .3)

    def test_original_gaps_and_both_channel_interior_masks(self):
        x = self.sequence([0, 1, None, 3, 4, 5, 6])
        y = self.sequence([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(cross_channel_correlation(x, y, 2, min_pairs=3)['valid_pairs'], 2)
        self.assertEqual(cross_channel_correlation(x, y, -2, min_pairs=3)['valid_pairs'], 2)
        self.assertEqual(cross_channel_correlation(x, y, 0, min_pairs=3)['valid_pairs'], 6)
        self.assertEqual(contiguous_segments(x.mask), [(0, 2), (3, 7)])
        result = cross_channel_correlation(adjacent_changes(x), adjacent_changes(y), 0, min_pairs=3)
        self.assertEqual(result['valid_pairs'], 4)

    def test_independent_pearson_and_affine_invariance(self):
        rng = np.random.default_rng(20)
        x, y = rng.normal(size=(2, 120))
        for lag in (-4, 0, 4):
            a, b = (x[4:], y[:-4]) if lag < 0 else ((x[:-4], y[4:]) if lag else (x, y))
            r = cross_channel_correlation(self.sequence(x), self.sequence(y), lag)
            self.assertAlmostEqual(r['correlation'], np.corrcoef(a, b)[0, 1])
            transformed = cross_channel_correlation(self.sequence(3 * x + 50), self.sequence(2 * y - 30), lag)
            self.assertAlmostEqual(r['correlation'], transformed['correlation'])

    def test_constant_insufficient_nonfinite_and_empty(self):
        self.assertEqual(pearson(np.ones(40), np.arange(40))['status'], 'constant_pairs')
        self.assertEqual(pearson([1, 2], [3, 4])['status'], 'insufficient_pairs')
        self.assertEqual(cross_channel_correlation(self.sequence([]), self.sequence([]), 3)['valid_pairs'], 0)
        x = BeatSequence([1, np.inf, np.nan, 2, 3], [1] * 5)
        self.assertEqual(cross_channel_correlation(x, x, min_pairs=3)['valid_pairs'], 3)

    def test_joint_direction_marginal_control(self):
        x = self.sequence([-1, 1] * 20)
        same = joint_directions([x, x])
        self.assertEqual(same['fall_fall_probability'], .5)
        self.assertEqual(same['rise_rise_probability'], .5)
        self.assertEqual(same['fall_fall_independent_probability'], .25)
        self.assertEqual(same['product_marginal_tv'], .5)
        balanced = self.sequence([-1, -1, 1, 1] * 10)
        self.assertAlmostEqual(joint_directions([x, balanced])['product_marginal_tv'], 0)
        flat = joint_directions([self.sequence([0] * 40), x])
        self.assertEqual(flat['product_marginal_tv'], 0)

    def test_three_way_pattern_can_exist_without_pairwise_correlation(self):
        a, b = np.tile([-1., -1., 1., 1.], 20), np.tile([-1., 1., -1., 1.], 20)
        c = a * b
        sequences = [self.sequence(v) for v in (a, b, c)]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(cross_channel_correlation(sequences[i], sequences[j])['correlation'], 0)
        result = joint_directions(sequences)
        self.assertEqual(result['centered_third_moment'], 1)
        self.assertEqual(result['product_marginal_tv'], .5)
        self.assertEqual(sum(result[k] for k in result if k.endswith('_count')), 80)

    def test_joint_missing_tolerance_and_undefined(self):
        a = self.sequence([1., -1., 0., None])
        b = self.sequence([1., -1., 0., 1.])
        result = joint_directions([a, b], tolerance=1, min_pairs=3)
        self.assertEqual(result['flat_flat_count'], 3)
        self.assertEqual(result['product_marginal_tv'], 0)
        self.assertTrue(np.isnan(joint_directions([a, b])['product_marginal_tv']))

    def test_phase_control_preserves_distributions_and_matches_independent_rolls(self):
        rng = np.random.default_rng(30)
        x = rng.normal(size=100)
        y = .8 * x + .2 * rng.normal(size=100)
        # Two long runs with a gap and a short excluded tail.
        a = self.sequence([*x, None, *x, None, 0, 1, 2])
        b = self.sequence([*y, None, *y, None, 0, 2, 3])
        before = b.values.copy()
        result = phase_shift_control(a, b, samples=49, seed=12)
        self.assertEqual(result['control_pairs'], 200)
        self.assertEqual(result['control_segments'], 2)
        draw = np.random.default_rng(12)
        blocks = []
        for values in (y, y):
            offsets = draw.integers(8, len(values) - 8 + 1, size=49)
            block = np.array([np.roll(values, int(offset)) for offset in offsets])
            for shifted in block:
                np.testing.assert_array_equal(np.sort(values), np.sort(shifted))
            blocks.append(block)
        shifted = np.concatenate(blocks, axis=1)
        ref = [abs(np.corrcoef(np.r_[x, x], v)[0, 1]) for v in shifted]
        self.assertAlmostEqual(result['shift_abs_mean'], np.mean(ref))
        self.assertAlmostEqual(result['shift_abs_p95'], np.percentile(ref, 95))
        self.assertGreater(result['abs_excess_over_shift_p95'], .6)
        np.testing.assert_array_equal(b.values, before)
        self.assertEqual(result, phase_shift_control(a, b, samples=49, seed=12))

    def test_phase_undefined_is_not_zero_evidence(self):
        short = phase_shift_control(self.sequence(range(40)), self.sequence(range(40)))
        self.assertEqual(short['status'], 'insufficient_pairs')
        self.assertTrue(np.isnan(short['shift_abs_p95']))
        constant = phase_shift_control(self.sequence([1] * 100), self.sequence(range(100)))
        self.assertEqual(constant['status'], 'constant_pairs')

    def test_stable_contrast_requires_both_halves_same_sign(self):
        self.assertTrue(stable_contrast(.5, -.1, [.4, .5], [0., .1]))
        self.assertFalse(stable_contrast(.5, -.1, [.4, .1], [0., .5]))
        self.assertFalse(stable_contrast(.5, -.1, [.4, np.nan], [0., .1]))

    def test_descriptor_distance_requires_support_and_groups_pedal(self):
        data = np.zeros((20, 7, 10))
        data[:] = np.arange(20)[:, None, None]
        scales = fit_descriptor_scales(data)
        self.assertTrue(np.isfinite(scales).all())
        values = np.zeros((2, 7, 10))
        values[1, 4:] = 1
        distance, counts = single_temporal_distances(values, np.ones((7, 10)), np.array([0]), np.array([1]))
        self.assertAlmostEqual(distance[0], np.sqrt(1 / 5))
        self.assertEqual(counts[0], 70)
        values[1, 0, :3] = np.nan
        self.assertTrue(np.isnan(single_temporal_distances(values, np.ones((7, 10)), np.array([0]), np.array([1]))[0][0]))

    def test_invalid_settings_and_grids(self):
        x = self.sequence(range(100))
        for lag in (True, 1.2):
            with self.assertRaises(ValueError):
                cross_channel_correlation(x, x, lag)
        for minimum in (2, True, 1.5):
            with self.assertRaises(ValueError):
                pearson([1, 2, 3], [3, 2, 1], min_pairs=minimum)
        with self.assertRaises(ValueError):
            cross_channel_correlation(x, self.sequence(range(90)))
        for kwargs in ({'samples': 0}, {'min_shift': 0}, {'min_segment': 4}, {'samples': True}):
            with self.assertRaises(ValueError):
                phase_shift_control(x, x, **kwargs)
        with self.assertRaises(ValueError):
            joint_directions([x])
        with self.assertRaises(ValueError):
            joint_directions([x, x], tolerance=-1)


if __name__ == '__main__':
    unittest.main()
