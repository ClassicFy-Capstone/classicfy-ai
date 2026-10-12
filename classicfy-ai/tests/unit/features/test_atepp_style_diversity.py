"""Protect musical-block weighting used in ATEPP diversity comparisons."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_atepp_style_diversity import block_squared, weighted_vectors


class StyleDiversityTests(unittest.TestCase):
    def test_three_pedal_channels_have_one_feature_weight(self):
        zero = np.zeros(14)
        pedal = zero.copy(); pedal[8:] = 1
        tempo = zero.copy(); tempo[:2] = 1
        # Equal changes in a musical block have equal total weight despite
        # Pedaling having three times as many coordinates as Tempo.
        self.assertAlmostEqual(np.linalg.norm(weighted_vectors(pedal)), 1 / np.sqrt(5))
        self.assertAlmostEqual(np.linalg.norm(weighted_vectors(tempo)), 1 / np.sqrt(5))

    def test_signed_tendency_difference_survives(self):
        a, b = np.zeros(14), np.zeros(14)
        a[4], b[4] = -1, 1
        self.assertAlmostEqual(block_squared(a, b)[2], 2)
        self.assertAlmostEqual(np.linalg.norm(weighted_vectors(a) - weighted_vectors(b)), np.sqrt(.4))

    def test_common_offset_does_not_change_distance(self):
        a, b = np.arange(14.), np.arange(14.)[::-1]
        shift = np.linspace(-7, 20, 14)
        np.testing.assert_allclose(block_squared(a + shift, b + shift), block_squared(a, b))


if __name__ == '__main__':
    unittest.main()
