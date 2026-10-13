import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from compare_metadata_temporal_embeddings import (fit_scales, masked_triplet_distances,
    learned_distances, summarize_records, family_balanced)


class MetadataTemporalEmbeddingTests(unittest.TestCase):
    def test_scales_use_train_only_and_leave_constants_undefined(self):
        x = np.column_stack([np.arange(12.), np.ones(12)])
        train = np.arange(12) < 10
        before = fit_scales(x, train)
        x[10:, 0] = 1e9
        np.testing.assert_equal(before, fit_scales(x, train))
        self.assertTrue(np.isfinite(before[0]))
        self.assertTrue(np.isnan(before[1]))
        self.assertTrue(np.isnan(fit_scales(x, np.arange(12) < 9)).all())

    def test_same_triplet_intersection_prevents_incomparable_pair_distances(self):
        # Pair AB alone would include a large extra coordinate; C lacks it.
        x = np.array([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 100.],
                      [2., 2., 2., 2., np.nan]])
        db, dc, support, _ = masked_triplet_distances(x, np.ones(5), [list(range(5))], [1],
            np.array([0]), np.array([1]), np.array([2]))
        self.assertAlmostEqual(db[0], 1)
        self.assertAlmostEqual(dc[0], 2)
        self.assertEqual(support[0], 4)

    def test_support_threshold_and_nan_are_not_zero_similarity(self):
        x = np.array([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.],
                      [2., 2., 2., np.nan, np.nan]])
        db, dc, _, _ = masked_triplet_distances(x, np.ones(5), [list(range(5))], [1],
            np.array([0]), np.array([1]), np.array([2]))
        self.assertTrue(np.isnan(db[0]) and np.isnan(dc[0]))

    def test_pedal_channels_share_one_music_block(self):
        x = np.zeros((3, 7)); x[1, 4:] = 1; x[2, :4] = 1
        db, dc, _, _ = masked_triplet_distances(x, np.ones(7), [[i] for i in range(7)],
            [1 / 5] * 4 + [1 / 15] * 3, np.array([0]), np.array([1]), np.array([2]))
        self.assertAlmostEqual(db[0], np.sqrt(.2))
        self.assertAlmostEqual(dc[0], np.sqrt(.8))

    def test_relation_group_support_is_shared_and_at_least_eighty_percent(self):
        x = np.zeros((3, 5)); x[1] = 1; x[2] = 2; x[2, 4] = np.nan
        db, dc, support, groups = masked_triplet_distances(x, np.ones(5), [[i] for i in range(5)],
            [1 / 5] * 5, np.array([0]), np.array([1]), np.array([2]), required_group_fraction=.8)
        self.assertEqual(groups[0], 4)
        self.assertEqual(support[0], 4)
        self.assertEqual((db[0], dc[0]), (1, 2))
        x[2, 3] = np.nan
        db, *_ = masked_triplet_distances(x, np.ones(5), [[i] for i in range(5)], [1 / 5] * 5,
            np.array([0]), np.array([1]), np.array([2]), required_group_fraction=.8)
        self.assertTrue(np.isnan(db[0]))

    def test_cosine_and_euclidean_geometries_can_disagree(self):
        x = np.array([[1., 0.], [10., 0.], [1., 1.]])
        cosine = learned_distances(x, 'cosine')
        euclidean = learned_distances(x, 'euclidean')
        self.assertLess(cosine[0, 1], cosine[0, 2])
        self.assertGreater(euclidean[0, 1], euclidean[0, 2])
        np.testing.assert_allclose(cosine, cosine.T, atol=1e-12)
        with self.assertRaises(ValueError):
            learned_distances(np.zeros((2, 3)), 'cosine')

    def row(self, key, work, score, representative=1):
        return dict(key=key, work_id=work, work_type='etude', preference=score,
                    representative_preference=representative)

    def test_conditional_counts_and_work_macro_handle_missing_explicitly(self):
        rows = [self.row('a', 'w1', 1), self.row('b', 'w1', 1),
                self.row('c', 'w2', 0), self.row('d', 'w2', np.nan), self.row('e', 'w2', 1, 0)]
        summaries, _ = summarize_records(rows)
        r = next(s for s in summaries if s['subset'] == 'representative_b_closer')
        self.assertEqual((r['eligible'], r['supported'], r['unsupported']), (4, 3, 1))
        self.assertAlmostEqual(r['query_preference'], 2 / 3)
        self.assertAlmostEqual(r['work_preference'], .5)

    def test_family_balancing_does_not_drop_unsupported_pairs(self):
        rows = [self.row('a', 'w1', 1), self.row('a', 'w1', np.nan),
                self.row('b', 'w2', 1), self.row('b', 'w2', 0)]
        balanced = family_balanced(rows)
        self.assertTrue(np.isnan(balanced[0]['preference']))
        self.assertEqual(balanced[0]['supported_pairs'], 1)
        self.assertEqual(balanced[1]['preference'], .5)


if __name__ == '__main__':
    unittest.main()
