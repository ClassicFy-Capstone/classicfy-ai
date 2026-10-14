"""Verify shortcut isolation, metric semantics, and actual aligned batch boundaries."""
from copy import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from data import GroupCatalog, load_mapping, coordinate_token, previous as previous_data
from model import (ConditionalMetricEncoder, target_distances, within_passage_loss,
                   balanced_regularization, previous)


def targets(amplitude):
    amp = torch.tensor(amplitude, dtype=torch.float32)
    n = len(amp)
    return dict(shape=torch.zeros(n, 2, 16), shape_mask=torch.zeros(n, 2, 16, dtype=torch.bool),
                amplitude=amp, amplitude_mask=torch.ones_like(amp, dtype=torch.bool),
                lag=torch.zeros(n, 54), lag_mask=torch.zeros(n, 54, dtype=torch.bool))


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.x = torch.randn(4, 7, 17)
        self.c = torch.randn(4, 13, 17)
        self.model = ConditionalMetricEncoder().eval()

    def test_score_swap_does_not_change_primary_retrieval_but_conditions_decoder(self):
        c = self.c.clone().requires_grad_(True)
        z = self.model.encode(self.x, c)
        other = self.model.encode(self.x, self.c.flip(0))
        self.assertTrue(torch.equal(z, other))
        self.assertIsNone(torch.autograd.grad(z[:, 0].sum(), c, allow_unused=True)[0])
        a = self.model(self.x, self.c)['heads']['lag']
        b = self.model(self.x, self.c.flip(0))['heads']['lag']
        self.assertGreater(float((a - b).abs().max().detach()), 1e-5)

    def test_fused_control_retrieval_has_context_path(self):
        model = ConditionalMetricEncoder(True).eval()
        c = self.c.clone().requires_grad_(True)
        z = model.encode(self.x, c)
        self.assertGreater(float(torch.autograd.grad(z[:, 0].sum(), c)[0].abs().sum()), 0)

    def test_hidden_poison_never_enters_encoder_or_decoder(self):
        mask = previous.masks(len(self.x), torch.Generator().manual_seed(4))
        poison = self.x.clone(); poison[mask] = 1e10
        a = self.model(self.x, self.c, mask); b = self.model(poison, self.c, mask)
        self.assertTrue(torch.equal(a['unit'], b['unit']))
        for name in a['heads']:
            self.assertTrue(torch.equal(a['heads'][name], b['heads'][name]))

    def test_order_is_observable_and_unit_distance_is_bounded(self):
        z = self.model.encode(self.x)
        shuffled = self.model.encode(self.x.flip(2))
        self.assertGreater(float((z - shuffled).abs().max().detach()), 1e-5)
        self.assertTrue(torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-6))
        distance = (z[:, None] - z[None, :]).square().sum(2)
        self.assertTrue(bool((distance <= 4 + 1e-6).all()))


class MetricTests(unittest.TestCase):
    def test_observed_similarity_is_rewarded_without_performer_labels(self):
        t = targets([[0, 0], [.2, .2], [1, 1]])
        angle = torch.tensor([0., .2, 1.])
        z = torch.stack((angle.cos(), angle.sin()), dim=1)
        groups = torch.zeros(3, dtype=torch.long); owners = torch.tensor([10, 20, 30])
        good, _, _ = within_passage_loss(z, t, groups, owners)
        bad, _, _ = within_passage_loss(z[[0, 2, 1]], t, groups, owners)
        self.assertLess(float(good), float(bad) / 10)

    def test_cross_passage_and_same_recording_pairs_are_excluded(self):
        t = targets([[0, 0], [.2, .2], [.4, .4], [1, 1]])
        z = torch.eye(4)
        groups = torch.tensor([0, 0, 0, 1]); owners = torch.tensor([7, 8, 8, 9])
        a, count, ng = within_passage_loss(z, t, groups, owners)
        self.assertEqual((count, ng), (2, 1))
        z[3] = torch.tensor([-1., 0., 0., 0.])
        b, _, _ = within_passage_loss(z, t, groups, owners)
        self.assertEqual(float(a), float(b))

    def test_unsupported_shape_does_not_become_false_similarity(self):
        t = targets([[0, 0], [1, 1]])
        a, support = target_distances(t)
        t['shape'].fill_(10000); t['lag'].fill_(-10000)
        b, _ = target_distances(t)
        self.assertTrue(torch.equal(a, b))
        self.assertAlmostEqual(float(a[0, 1]), 1.)
        self.assertTrue(bool(support.all()))

    def test_targets_are_detached_and_no_support_raises(self):
        t = targets([[0, 0], [1, 1]])
        t['amplitude'].requires_grad_(True)
        wanted, _ = target_distances(t)
        self.assertFalse(wanted.requires_grad)
        t['amplitude_mask'].zero_()
        with self.assertRaisesRegex(ValueError, 'No supported'):
            within_passage_loss(torch.eye(2), t, torch.zeros(2, dtype=torch.long), torch.arange(2))

    def test_group_balancing_is_invariant_to_replicated_rows(self):
        torch.manual_seed(1)
        a = torch.randn(5, 32); b = torch.randn(5, 32); groups = torch.tensor([0, 0, 0, 1, 1])
        before = balanced_regularization(a, b, groups)
        ids = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        after = balanced_regularization(a[ids], b[ids], groups[ids])
        for name in before:
            self.assertTrue(torch.allclose(before[name], after[name], atol=1e-6))

    def test_collapsed_raw_embedding_incurs_variance_penalty(self):
        zero = torch.zeros(4, 32)
        parts = balanced_regularization(zero, zero, torch.tensor([0, 0, 1, 1]))
        self.assertGreater(float(parts['variance']), .98)


class ActualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_mapping()
        cls.catalog = GroupCatalog(0, cls.mapping)

    def test_actual_batches_have_only_train_distinct_recordings_and_equal_context(self):
        catalog = self.catalog; d = catalog.data
        family_roles = {}
        for g in catalog.groups:
            self.assertEqual(family_roles.setdefault(g.family, g.role), g.role)
        weights = catalog.group_weights('train')
        self.assertAlmostEqual(float(weights.sum()), 1.)
        for seed in range(10):
            batch = catalog.sample_train(np.random.default_rng(seed))
            self.assertTrue(set(batch['owners'].tolist()) <= set(d.gids.tolist()))
            self.assertFalse(set(batch['owners'].tolist()) & set(d.qids.tolist()))
            for group in batch['groups'].unique():
                ii = batch['groups'] == group
                self.assertEqual(len(batch['owners'][ii]), len(batch['owners'][ii].unique()))
                c = batch['context'][ii]
                self.assertTrue(torch.allclose(c, c[:1].expand_as(c), atol=1e-6, rtol=0))

    def test_test_values_cannot_influence_new_amplitude_scales(self):
        catalog = self.catalog; altered = copy(catalog.data)
        altered.arrays = dict(altered.arrays)
        altered.arrays['x'] = altered.arrays['x'].copy()
        altered.arrays['x'][altered.roles['test']] += np.arange(17)[None, None, :] * 1000
        with patch.object(previous_data, 'ViewData', return_value=altered):
            rebuilt = GroupCatalog(0, self.mapping)
        self.assertEqual(catalog.amplitude_scales, rebuilt.amplitude_scales)
        ids = catalog.data.roles['train']
        self.assertTrue(torch.equal(catalog.targets['amplitude'][ids], rebuilt.targets['amplitude'][ids]))

    def test_repeat_occurrence_is_distinct_and_performance_timing_is_not_identity(self):
        row = dict(status='coordinate_verified', valid='True', visit_index='1', measure_index='8',
                   written_start_quarters='32', written_stop_quarters='33', performance_start_seconds='10')
        token = coordinate_token(row)
        row['performance_start_seconds'] = '20'
        self.assertEqual(token, coordinate_token(row))
        row['visit_index'] = '2'
        self.assertNotEqual(token, coordinate_token(row))
        row['valid'] = 'False'
        with self.assertRaises(ValueError):
            coordinate_token(row)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
