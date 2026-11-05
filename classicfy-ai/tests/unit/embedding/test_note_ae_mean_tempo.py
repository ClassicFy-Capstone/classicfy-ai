"""Mean supervision uses unique window beats and stays outside the decoder."""

from dataclasses import replace
import unittest

import torch

from embedding.note_ae_loss import mean_tempo_loss
from embedding.note_autoencoder import ConditionalNoteAutoencoder, NoteAEConfig


class MeanTempoAuxiliaryTest(unittest.TestCase):
    def test_window_mean_ignores_unknown_beats_and_weights_windows_equally(self):
        values = torch.tensor([[2., 4., float('nan')], [float('nan')] * 3, [6., 999., 999.]])
        mask = torch.tensor([[True, True, False], [False] * 3, [True, False, False]])
        prediction = torch.tensor([1., float('nan'), 5.], requires_grad=True)
        out = mean_tempo_loss(prediction, values, mask)
        torch.testing.assert_close(out['target'], torch.tensor([3., 0., 6.]))
        self.assertEqual(out['beat_counts'].tolist(), [2, 0, 1])
        self.assertEqual(float(out['loss'].detach()), 1.)  # (Huber(-2) + Huber(-1)) / 2
        out['loss'].backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([-.5, 0., -.5]))

    def test_no_valid_beats_has_finite_zero_gradient(self):
        for width in (0, 3):
            prediction = torch.tensor([float('nan')], requires_grad=True)
            result = mean_tempo_loss(prediction, torch.full((1, width), float('nan')),
                                     torch.zeros(1, width, dtype=torch.bool))
            self.assertTrue(result['skip'])
            self.assertEqual(float(result['loss'].detach()), 0.)
            result['loss'].backward()
            self.assertEqual(float(prediction.grad[0]), 0.)

    def test_auxiliary_does_not_change_shared_initialization_or_random_stream(self):
        cfg = NoteAEConfig(context_dim=4, condition_dim=3, input_dim=4, hidden_size=3, latent_dim=2)
        torch.manual_seed(91)
        base = ConditionalNoteAutoencoder(cfg)
        base_rng = torch.get_rng_state().clone()
        torch.manual_seed(91)
        aux = ConditionalNoteAutoencoder(replace(cfg, mean_tempo_auxiliary=True))
        self.assertTrue(torch.equal(base_rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, aux.state_dict()[key], rtol=0, atol=0)

    def test_mean_gradient_reaches_encoder_but_head_never_enters_decoder(self):
        torch.manual_seed(7)
        cfg = NoteAEConfig(context_dim=4, condition_dim=3, input_dim=4, hidden_size=3,
                           latent_dim=2, layers=1, mean_tempo_auxiliary=True)
        model = ConditionalNoteAutoencoder(cfg).eval()
        C, R = torch.randn(1, 4, 4), torch.randn(1, 4, 11)
        P, M = torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, 11, dtype=torch.bool)
        ids = torch.tensor([[0, 0, 1, 2]])
        output = model(C, R, M, P, P.sum(1), ids)
        self.assertEqual(output['mean_tempo_hat'].shape, (1,))
        mean_tempo_loss(output['mean_tempo_hat'], torch.tensor([[3., 4., 5.]]),
                        torch.ones(1, 3, dtype=torch.bool))['loss'].backward()
        for module in (model.encoder, model.latent, model.mean_tempo_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        with torch.no_grad():
            model.mean_tempo_head[-1].bias.add_(100.)
        changed = model(C, R, M, P, P.sum(1), ids)
        torch.testing.assert_close(changed['R_hat'], output['R_hat'], rtol=0, atol=0)
        torch.testing.assert_close(changed['mean_tempo_hat'], output['mean_tempo_hat'] + 100.)
