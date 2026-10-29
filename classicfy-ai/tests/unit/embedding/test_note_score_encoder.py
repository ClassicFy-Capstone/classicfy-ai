import io

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from embedding.note_score_context_data import collate_note_score_context
from embedding.note_score_encoder import NoteScoreEncoder, NoteScoreEncoderConfig
from features.note_score_context import NUMERIC_FIELDS as N, CATEGORICAL_FIELDS as C, FLAG_FIELDS as F, extract_note_score_context
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, measure, note, score_xml


class NoteScoreEncoderTest(XMLFixture):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        super().setUp()
        torch.manual_seed(17)
        self.features = extract_note_score_context(self.read())
        self.batch = collate_note_score_context([self.features])
        self.model = NoteScoreEncoder(NoteScoreEncoderConfig(embedding_dim=3, note_dim=8, hidden_size=5, context_dim=12)).eval()

    def clone(self, batch=None):
        return {name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in (batch or self.batch).items()}

    def test_note_wise_output_and_same_chord_notes_can_differ(self):
        output = self.model(self.batch)
        self.assertEqual(output.shape, (1, 8, 12))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.allclose(output[0, 0], output[0, 1]))
        default = NoteScoreEncoder()
        self.assertEqual(default(self.batch).shape, (1, 8, 128))

    def test_hand_computed_mean_and_count_reach_gru(self):
        captured = {}
        h_hook = self.model.note_mlp.register_forward_hook(lambda module, args, output: captured.update(h=output.detach()))
        g_hook = self.model.gru.register_forward_pre_hook(lambda module, args: captured.update(gru=pad_packed_sequence(args[0], batch_first=True)[0].detach()))
        try:
            self.model(self.batch)
        finally:
            h_hook.remove()
            g_hook.remove()
        expected_groups = ([0, 1], [2], [3, 4, 5], [6], [7])
        for g, rows in enumerate(expected_groups):
            torch.testing.assert_close(captured['gru'][0, g, :8], captured['h'][0, rows].mean(dim=0))
            self.assertAlmostEqual(float(captured['gru'][0, g, 8]), float(torch.tensor(1 + len(rows)).log()), places=6)
        torch.testing.assert_close(captured['gru'][0, :, 9], torch.tensor([0., 1., 3., 1., 1.]))

    def test_note_permutation_is_equivariant(self):
        before = self.model(self.batch)
        perm = torch.tensor([1, 0, 2, 5, 4, 3, 6, 7])
        changed = self.clone()
        for name in ('numeric', 'categorical', 'flags', 'field_known', 'note_mask', 'note_to_onset'):
            changed[name] = changed[name][:, perm]
        after = self.model(changed)
        torch.testing.assert_close(after, before[:, perm])

    def test_mean_can_stay_same_while_note_count_changes(self):
        baseline = self.clone()
        # Duplicate the first written note with identical features, keeping two notes in group 0.
        for name in ('numeric', 'categorical', 'flags', 'field_known'):
            baseline[name][:, 1] = baseline[name][:, 0]
        captures = []
        hook = self.model.gru.register_forward_pre_hook(lambda module, args: captures.append(pad_packed_sequence(args[0], batch_first=True)[0].detach()))
        try:
            self.model(baseline)
            expanded = self.clone(baseline)
            for name in ('numeric', 'categorical', 'flags', 'field_known', 'note_mask', 'note_to_onset'):
                expanded[name] = torch.cat([expanded[name], expanded[name][:, :1]], dim=1)
            self.model(expanded)
        finally:
            hook.remove()
        torch.testing.assert_close(captures[0][0, 0, :8], captures[1][0, 0, :8])
        self.assertAlmostEqual(float(captures[0][0, 0, 8]), float(torch.tensor(3.).log()), places=6)
        self.assertAlmostEqual(float(captures[1][0, 0, 8]), float(torch.tensor(4.).log()), places=6)

    def test_known_mask_is_forwarded_and_unknown_payload_is_neutralized(self):
        batch = self.clone()
        batch['field_known'][:, 0, 0] = False
        before = self.model(batch)
        noisy = self.clone(batch)
        noisy['numeric'][:, 0, 0] = float('nan')
        torch.testing.assert_close(self.model(noisy), before)
        captured = []
        hook = self.model.note_mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        try:
            self.model(batch)
            known = self.clone(batch)
            known['numeric'][:, 0, 0] = 0
            known['field_known'][:, 0, 0] = True
            self.model(known)
        finally:
            hook.remove()
        d = len(N) + len(C) + len(F)
        self.assertEqual(captured[0][0, 0, -d], 0)
        self.assertEqual(captured[1][0, 0, -d], 1)

    def test_padding_invariance_and_zero_padding_gradient(self):
        short = extract_note_score_context(self.read(score_xml(measure(attributes() + note('x', duration=16)))))
        batch = collate_note_score_context([short, self.features])
        result = self.model(batch)
        torch.testing.assert_close(result[0, :1], self.model(collate_note_score_context([short]))[0])
        self.assertEqual(torch.count_nonzero(result[0, 1:]), 0)
        noisy = self.clone(batch)
        noisy['numeric'][~batch['note_mask']] = float('nan')
        noisy['flags'][~batch['note_mask']] = float('inf')
        noisy['categorical'][~batch['note_mask']] = -100
        noisy['note_to_onset'][~batch['note_mask']] = 999
        noisy['delta_q'][~batch['onset_mask']] = float('nan')
        torch.testing.assert_close(self.model(noisy), result)
        batch['numeric'].requires_grad_()
        self.model(batch).square().sum().backward()
        self.assertEqual(torch.count_nonzero(batch['numeric'].grad[~batch['note_mask']]), 0)

    def test_masked_reconstruction_gradient_reaches_all_encoder_stages(self):
        self.batch['numeric'].requires_grad_()
        context = self.model(self.batch)
        decoder = nn.Linear(12, 2)
        predictions = decoder(context)
        valid = self.batch['duration_owner_mask']
        loss = nn.functional.huber_loss(predictions[valid], torch.ones_like(predictions[valid]))
        loss.backward()
        for parameter in (self.model.note_mlp[0].weight, self.model.embeddings[0].weight,
                          self.model.gru.weight_ih_l0, self.model.context_mlp[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(self.batch['numeric'].grad).all())

    def test_bad_masks_mapping_and_known_nan_are_rejected(self):
        for field, index, value in [('note_mask', (0, 1), False), ('onset_mask', (0, 1), False),
                                    ('note_to_onset', (0, 0), -1), ('delta_q', (0, 1), 0.),
                                    ('numeric', (0, 0, 0), float('nan'))]:
            bad = self.clone()
            bad[field][index] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.model(bad)

    def test_checkpoint_restores_exact_context(self):
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = NoteScoreEncoder(self.model.config).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(self.batch), self.model(self.batch), rtol=0, atol=0)
