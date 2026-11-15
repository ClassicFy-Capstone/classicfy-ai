"""Hand-calculated weak supervision, including gradients and block balance."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from embedding.note_ae_loss import triplet_style_loss
from embedding.note_ae_training import NoteAETrainingConfig, style_loss_for_batch
from embedding.style_supervision import triplet_confidence
from features.style_profile import style_quality_schema


class GStyleLossTest(unittest.TestCase):
    def latents(self, count=1):
        return [torch.tensor([row] * count, requires_grad=True) for row in
                ([1., 0.], [0., 1.], [.8, .6])]

    def test_legacy_value_and_gradients_are_exact(self):
        z = self.latents(2)
        a, p, n = [F.normalize(v, dim=-1) for v in z]
        expected = F.relu(1 - (a * p).sum(-1) - (1 - (a * n).sum(-1)) + .2).mean()
        actual = triplet_style_loss(*z, margins=torch.full((2,), .2), weights=torch.ones(2))['loss']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for x, y in zip(torch.autograd.grad(expected, z, retain_graph=True), torch.autograd.grad(actual, z)):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_confidence_scales_singleton_and_zero_gradients(self):
        for weight in (0., .1, 1.):
            z = self.latents()
            raw = triplet_style_loss(*z)['loss']
            loss = triplet_style_loss(*z, weights=torch.tensor([weight]))['loss']
            torch.testing.assert_close(loss, raw * weight)
            for x, y in zip(torch.autograd.grad(raw, z, retain_graph=True), torch.autograd.grad(loss, z)):
                torch.testing.assert_close(y, x * weight)
                self.assertTrue(torch.isfinite(y).all())

    def test_weighted_reduction_uses_batch_size_not_weight_sum(self):
        z = self.latents(2)
        result = triplet_style_loss(*z, margins=torch.tensor([.1, .3]), weights=torch.tensor([.2, .6]))
        # d_ap=1 and d_an=.2: (.2*.9 + .6*1.1)/2 = .42
        self.assertAlmostEqual(float(result['loss'].detach()), .42, places=6)

    def test_adaptive_bounds_modes_and_detached_targets(self):
        z = self.latents(3)
        gap = torch.tensor([.01, 1., 10.], dtype=torch.float64, requires_grad=True)
        confidence = torch.tensor([.2, .5, .8], requires_grad=True)
        batch = {'g_gap_normalized': gap, 'confidence': confidence}
        for mode in ('legacy', 'confidence', 'adaptive', 'adaptive_confidence'):
            result = style_loss_for_batch(z, batch, NoteAETrainingConfig(style_objective=mode))
            adaptive = mode.startswith('adaptive')
            torch.testing.assert_close(result['margins'], torch.tensor([.05, .2, .4] if adaptive else [.2] * 3))
            torch.testing.assert_close(result['weights'], confidence.detach() if 'confidence' in mode else torch.ones(3))
            self.assertAlmostEqual(result['margin_clip_low_ratio'], 1 / 3 if adaptive else 0.)
            self.assertAlmostEqual(result['margin_clip_high_ratio'], 1 / 3 if adaptive else 0.)
        result['loss'].backward()
        self.assertIsNone(gap.grad)
        self.assertIsNone(confidence.grad)

    def test_invalid_supervision_and_configuration_are_rejected(self):
        z = self.latents()
        for weights in (torch.tensor([float('nan')]), torch.tensor([-1.]), torch.tensor([1.1]), torch.ones(2)):
            with self.assertRaises(ValueError):
                triplet_style_loss(*z, weights=weights)
        for margins in (torch.tensor([2.]), torch.tensor([-1.]), torch.tensor([float('inf')])):
            with self.assertRaises(ValueError):
                triplet_style_loss(*z, margins=margins)
        for fields in ({'style_objective': 'bad'}, {'margin_max': 2.}, {'margin_min': .5}, {'margin_scale': 0.}):
            with self.assertRaises(ValueError):
                NoteAETrainingConfig(**fields)
        for mode in ('adaptive', 'confidence', 'adaptive_confidence'):
            with self.assertRaises(ValueError):
                style_loss_for_batch(z, {}, NoteAETrainingConfig(style_objective=mode))
        with self.assertRaises(ValueError):
            style_loss_for_batch(z, {'g_gap_normalized': torch.zeros(1)}, NoteAETrainingConfig(style_objective='adaptive'))

    def test_signature_block_balance_and_weakest_role(self):
        qs = [np.ones(23) for _ in range(3)]
        qs[0][9:] = .1
        profiles = [{'quality_schema': style_quality_schema(), 'g_quality': q.tolist(), 'g_mask': [True] * 23} for q in qs]
        self.assertAlmostEqual(triplet_confidence(profiles, tuple(range(23))), .82)  # (4+.1)/5
        self.assertEqual(triplet_confidence(profiles, tuple(range(9))), 1.)
        profiles[2]['g_quality'][5:7] = [.5, .5]
        self.assertAlmostEqual(triplet_confidence(profiles, tuple(range(23))), .72)
        profiles[0].pop('quality_schema')
        with self.assertRaises(ValueError):
            triplet_confidence(profiles, tuple(range(23)))
