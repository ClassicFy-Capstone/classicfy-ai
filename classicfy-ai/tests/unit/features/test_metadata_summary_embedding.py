import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from compare_metadata_summary_embedding import (distance_matrices, choose_pairs, compare,
    preference, evaluate, aggregate, weighted_projection, select_examples)


class MetadataSummaryEmbeddingTests(unittest.TestCase):
    def row(self, key, label, family, composer='Chopin'):
        return dict(key=key, work_type=label, family_id=family,
                    work_id=family, composer=composer)

    def test_equal_musical_blocks_and_range_decomposition(self):
        x = np.zeros((2, 14))
        x[1, 8:] = 1
        d = distance_matrices(x)
        self.assertAlmostEqual(d['representative_range'][0, 1], np.sqrt(1 / 5))
        rng = np.random.default_rng(1)
        x = rng.normal(size=(6, 14))
        x[:, 1::2] = abs(x[:, 1::2])
        d = distance_matrices(x)
        np.testing.assert_allclose(d['representative_range'] ** 2,
                                   (d['representative'] ** 2 + d['range_only'] ** 2) / 2)

    def test_nearest_same_type_control_and_selection_never_uses_range(self):
        rows = [self.row('a', 'etude', 'E1'), self.row('b', 'ballade', 'B1'),
                self.row('c', 'etude', 'E2'), self.row('far_c', 'etude', 'E3')]
        x = np.zeros((4, 14));x[1, ::2] = .1;x[2, ::2] = 1;x[3, ::2] = 3
        d = distance_matrices(x)
        chosen = choose_pairs(0, [1], [2, 3], rows, d['representative'])
        self.assertEqual(chosen, [(1, 2)])
        x[1, 1::2] = 4
        changed = distance_matrices(x)
        self.assertEqual(choose_pairs(0, [1], [2, 3], rows, changed['representative']), chosen)
        result = compare(0, chosen, changed)
        self.assertEqual(result['representative_preference'], 1)
        self.assertEqual(result['representative_range_preference'], 0)
        self.assertEqual(result['range_only_preference'], 0)

    def test_all_three_families_and_composer_are_enforced(self):
        rows = [self.row('a', 'prelude', 'BWV1', 'Bach'),
                self.row('b', 'fugue', 'BWV2', 'Bach'),
                self.row('c', 'prelude', 'BWV2', 'Bach'),
                self.row('c2', 'prelude', 'BWV3', 'Bach'),
                self.row('other', 'fugue', 'X', 'Other')]
        d = np.zeros((5, 5));d[0, 3] = 1
        self.assertEqual(choose_pairs(0, [1, 4], [2, 3], rows, d), [(1, 3)])
        self.assertEqual(choose_pairs(0, [1], [2], rows, d), [])

    def test_selection_ties_are_averaged_not_resolved_by_full_result(self):
        rows = [self.row('a', 'etude', 'E1'), self.row('b1', 'ballade', 'B1'),
                self.row('b2', 'ballade', 'B2'), self.row('c', 'etude', 'E2')]
        x = np.zeros((4, 14));x[1:3, ::2] = .1;x[3, ::2] = 1;x[2, 1::2] = 4
        d = distance_matrices(x)
        pairs = choose_pairs(0, [2, 1], [3], rows, d['representative'])
        self.assertEqual(pairs, [(1, 3), (2, 3)])
        self.assertEqual(compare(0, pairs, d)['representative_range_preference'], .5)

    def test_comparison_ties_and_invalid_values(self):
        self.assertEqual(preference(1, 1 + 1e-13), .5)
        self.assertEqual(preference(1, 2), 1)
        self.assertEqual(preference(2, 1), 0)
        for a, b in [(np.nan, 1), (np.inf, 1), (-1, 1)]:
            with self.assertRaises(ValueError):
                preference(a, b)
        with self.assertRaises(ValueError):
            distance_matrices(np.zeros((3, 7)))
        bad = np.zeros((2, 14));bad[0, 1] = -1
        with self.assertRaises(ValueError):
            distance_matrices(bad)

    def test_family_pair_average_and_work_macro(self):
        rows = [self.row('e1', 'etude', 'E1'), self.row('e2', 'etude', 'E2'),
                self.row('b1', 'ballade', 'B1'), self.row('b2', 'ballade', 'B2')]
        x = np.zeros((4, 14));x[:, ::2] = np.arange(4)[:, None]
        q, f = evaluate(rows, distance_matrices(x))
        self.assertEqual(len(q), 4)
        self.assertEqual(len(f), 8)
        summaries, works, balanced = aggregate(q, f)
        self.assertEqual(len(balanced), 4)
        self.assertEqual(len(works), 24)
        self.assertTrue(all(r['queries'] == 4 for r in summaries if r['query_type'] == 'all'))
        for row in balanced:
            local = [r['representative_range_preference'] for r in f if r['key'] == row['key']]
            self.assertAlmostEqual(row['representative_range_preference'], np.mean(local))

    def test_projection_preserves_original_weighted_14D_distances(self):
        x = np.random.default_rng(22).normal(size=(3, 14));x[:, 1::2] = abs(x[:, 1::2])
        xy, _ = weighted_projection(x)
        projected = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        np.testing.assert_allclose(projected, distance_matrices(x)['representative_range'], atol=1e-12)

    def test_examples_include_both_outcomes_and_handle_absence(self):
        rows = [dict(key=str(i), representative_range_preference=float(i < 3),
                     representative_range_mean_margin=float(i)) for i in range(5)]
        selected = select_examples(rows)
        self.assertEqual(selected['cross_type_closer']['key'], '1')
        self.assertEqual(selected['same_type_closer']['key'], '3')
        self.assertIsNone(select_examples(rows[:3])['same_type_closer'])


if __name__ == '__main__':
    unittest.main()
