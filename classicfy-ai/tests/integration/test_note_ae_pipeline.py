"""Hand-computable AE/loss contracts and artifact-to-checkpoint integration."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from embedding.note_autoencoder import ConditionalNoteAutoencoder, NoteAEConfig, NotePerformanceAutoencoder
from embedding.note_ae_loss import reconstruction_loss, triplet_style_loss
from embedding.note_ae_data import FixedNoteTriplets, NoteAECorpus
from embedding.note_ae_training import (NoteAETrainingConfig, export_note_embeddings, latent_ablation,
    load_note_ae_model, train_note_ae)
from embedding.note_performance_data import save_note_performance
from embedding.note_score_encoder import NoteScoreEncoderConfig
from embedding.style_profile_data import extract_profile_artifact, make_window_artifact, save_style_artifact, verify_window_sources
from embedding.style_triplets import build_triplets, calibrate_distances
from features.style_profile import StyleConfig, artifact, fit_profile_scaler, fit_reference, stable_id
from unit.preprocessing.musicxml_fixtures import XMLFixture
from integration import test_style_triplets as style_helpers


class NoteAEModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2026)
        self.config = NoteAEConfig(context_dim=8, condition_dim=4, input_dim=8, hidden_size=4, latent_dim=3)
        self.model = ConditionalNoteAutoencoder(self.config).eval()
        self.P = torch.tensor([[True, True, True, False, False], [True] * 5])
        self.lengths = self.P.sum(1)
        self.C, self.R = torch.randn(2, 5, 8), torch.randn(2, 5, 11)
        self.R[:, :, 0] = 1.
        self.M = self.P[..., None].expand(-1, -1, 11).clone()
        self.ids = torch.tensor([[0, 0, 1, -1, -1], [0, 0, 1, 2, 2]])

    def output(self):
        return self.model(self.C, self.R, self.M, self.P, self.lengths, self.ids)

    def test_shapes_channel_order_and_shared_tempo(self):
        out = self.output()
        self.assertEqual(out['R_hat'].shape, (2, 5, 11))
        self.assertEqual(out['z'].shape, (2, 3))
        torch.testing.assert_close(out['e'].norm(dim=-1), torch.ones(2))
        self.assertFalse(out['R_hat'][~self.P].any())
        torch.testing.assert_close(out['R_hat'][0, 0, 0], out['R_hat'][0, 1, 0])
        for head, v in [(self.model.onset_head, 1), (self.model.velocity_head, 2), (self.model.articulation_head, 3),
                        (self.model.pedal_head, 4), (self.model.tempo_head, 5)]:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.constant_(head.bias, v)
        torch.testing.assert_close(self.output()['R_hat'][0, 0], torch.tensor([5., 1., 2., 3.] + [4.] * 7))
        self.ids[:] = -1
        out = self.output()
        self.assertEqual(out['beat_prediction'].shape, (2, 0))
        self.assertFalse(out['R_hat'][:, :, 0].any())

    def test_padding_and_missing_input_invariance(self):
        before = self.output()
        C = torch.cat([self.C, torch.full((2, 3, 8), float('nan'))], 1)
        R = torch.cat([self.R, torch.full((2, 3, 11), float('nan'))], 1)
        M = torch.cat([self.M, torch.zeros(2, 3, 11, dtype=torch.bool)], 1)
        P = torch.cat([self.P, torch.zeros(2, 3, dtype=torch.bool)], 1)
        ids = torch.cat([self.ids, torch.full((2, 3), -1)], 1)
        after = self.model(C, R, M, P, self.lengths, ids)
        torch.testing.assert_close(before['z'], after['z'])
        torch.testing.assert_close(before['R_hat'][self.P], after['R_hat'][:, :5][self.P])
        M[:, 1, 3] = False
        first = self.model(C, R, M, P, self.lengths, ids)
        R[:, 1, 3] = float('nan')
        torch.testing.assert_close(first['z'], self.model(C, R, M, P, self.lengths, ids)['z'])

    def test_huber_equal_groups_and_pedal_channels(self):
        P, ids = torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long)
        t, p = torch.zeros(1, 1, 11), torch.full((1, 1, 11), 2.)
        M = torch.zeros_like(t, dtype=torch.bool)
        M[:, :, 1] = True
        self.assertEqual(float(reconstruction_loss(p, t, M, P, ids)['loss']), 1.5)
        M[:, :, 4:] = True
        p[:, :, 4:] = 1.
        self.assertEqual(float(reconstruction_loss(p, t, M, P, ids)['loss']), 1.)
        M[:, :, 5:] = False
        self.assertEqual(float(reconstruction_loss(p, t, M, P, ids)['loss']), 1.)

    def test_tempo_once_per_beat_and_conflicting_targets(self):
        t, p = torch.zeros(1, 4, 11), torch.zeros(1, 4, 11)
        M = torch.zeros_like(t, dtype=torch.bool)
        M[:, :, 0] = True
        P, ids = torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 0, 0, 1]])
        t[0, :3, 0] = 2.
        out = reconstruction_loss(p, t, M, P, ids)
        self.assertEqual(float(out['loss']), .75)
        self.assertEqual(int(out['channel_counts'][0, 0]), 2)
        t[0, 1, 0] = 3.
        with self.assertRaisesRegex(ValueError, 'targets disagree'):
            reconstruction_loss(p, t, M, P, ids)

    def test_masked_targets_all_missing_and_gradients(self):
        self.model.train()
        self.M[:, :, 3] = False
        pred = self.output()['R_hat']
        first = reconstruction_loss(pred, self.R, self.M, self.P, self.ids)['loss']
        self.R[~self.M] = float('nan')
        second = reconstruction_loss(pred, self.R, self.M, self.P, self.ids)['loss']
        torch.testing.assert_close(first, second)
        second.backward()
        for module in (self.model.encoder, self.model.decoder, self.model.latent, self.model.tempo_head):
            gs = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gs))
            self.assertGreater(sum(float(g.abs().sum()) for g in gs), 0)
        self.M[:] = False
        missing = reconstruction_loss(pred, self.R, self.M, self.P, self.ids)
        self.assertTrue(missing['skip'])
        self.assertEqual(float(missing['loss'].detach()), 0.)
        self.assertEqual(int(missing['channel_counts'].sum()), 0)

    def test_triplet_hand_calculation_and_satisfied_margin(self):
        a, p, n = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), torch.tensor([[1., 0.]])
        self.assertAlmostEqual(float(triplet_style_loss(a, p, n)['loss']), 1.2, places=6)
        self.assertEqual(float(triplet_style_loss(a, a * 2, -a)['loss']), 0.)


class NoteAEPipelineTest(XMLFixture):
    make_score = style_helpers.StyleTripletsTest.make_score
    record = style_helpers.StyleTripletsTest.record

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        super().setUp()
        self.mc = NoteAEConfig(context_dim=8, condition_dim=4, input_dim=8, hidden_size=4, latent_dim=3)
        self.sc = NoteScoreEncoderConfig(embedding_dim=2, note_dim=4, hidden_size=4, context_dim=8)
        self.tc = NoteAETrainingConfig(epochs=3, batch_size=3, learning_rate=.01, max_steps_per_epoch=1)

    def corpus(self, many=False):
        entries = []
        for w in range(4 if many else 2):
            xml = self.make_score(prefix=f'w{w}')
            xpath = self.folder / f'score-{self.serial}.musicxml'
            styles = (-2., -1., 0., .2, 1., 2.) if many else (-1., 0., 1.) if w == 0 else (0.,)
            for i, style in enumerate(styles):
                rec = self.record(xml, f'w{w}p{i}', style, work=f'w{w}', split='train' if many or w == 0 else 'validation')
                cache = self.folder / f'R-{w}-{i}'
                save_note_performance(rec.performance, cache)
                entries.append({'performance_cache': str(cache), 'xml_path': str(xpath), 'recording_group': rec.recording_id})
        windows = make_window_artifact(entries, self.folder)
        return NoteAECorpus(windows), windows

    def test_rows_scaling_and_score_gradient_policy(self):
        corpus, _ = self.corpus()
        self.assertEqual((len(corpus.dataset('train')), len(corpus.dataset('validation'))), (3, 1))
        items = corpus.dataset('train').items[:2]
        batch = corpus.collate(items)
        self.assertEqual(batch['R'].shape, (2, 32, 11))
        r = items[0]['segment'].pair.performance
        np.testing.assert_allclose(batch['R'][0], corpus.r_scaler.transform(r))
        model = NotePerformanceAutoencoder(self.mc, self.sc)
        output = model(batch)
        reconstruction_loss(output['R_hat'], batch['R'], batch['M'], batch['P'], batch['beat_id'])['loss'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.score_encoder.parameters()))
        frozen = NotePerformanceAutoencoder(replace(self.mc, score_gradient='frozen'), self.sc).train()
        self.assertFalse(frozen.score_encoder.training)
        self.assertTrue(all(not p.requires_grad for p in frozen.score_encoder.parameters()))

    def test_small_training_checkpoint_export_and_ablation(self):
        corpus, _ = self.corpus()
        model, report = train_note_ae(corpus, self.folder / 'run', config=self.tc, model_config=self.mc, score_config=self.sc)
        self.assertLess(report['final_train']['loss'], report['initial_train']['loss'])
        self.assertFalse(report['style_training_performed'])
        loaded, _, _, _ = load_note_ae_model(self.folder / 'run/last.pt', corpus=corpus)
        batch = corpus.collate(corpus.dataset('train').items[:2])
        model.eval()
        torch.testing.assert_close(model(batch)['z'], loaded(batch)['z'], rtol=0, atol=0)
        export_note_embeddings(loaded, corpus, self.folder / 'e.npz', split='validation', checkpoint_id='fixture')
        with np.load(self.folder / 'e.npz', allow_pickle=False) as data:
            self.assertEqual(data['z'].shape, (1, 3))
            np.testing.assert_allclose(np.linalg.norm(data['e'], axis=-1), 1., atol=1e-6)
        self.assertEqual(latent_ablation(loaded, corpus)['status'], 'evaluated')
        wrong = copy.copy(corpus)
        wrong.r_scaler = replace(corpus.r_scaler, fit_manifest_id='different')
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            load_note_ae_model(self.folder / 'run/last.pt', corpus=wrong)

    def test_resume_exact_at_epoch_boundary(self):
        corpus, _ = self.corpus()
        cfg = replace(self.tc, epochs=2)
        full, _ = train_note_ae(corpus, self.folder / 'full', config=cfg, model_config=self.mc, score_config=self.sc)
        train_note_ae(corpus, self.folder / 'first', config=replace(cfg, epochs=1), model_config=self.mc, score_config=self.sc)
        # Pre-auxiliary checkpoints omitted these fields; their defaults are
        # equivalent and must remain resumable.
        path = self.folder / 'first/last.pt'
        legacy = torch.load(path, weights_only=True)
        legacy['model_config'].pop('mean_tempo_auxiliary')
        legacy['training_config'].pop('beta_mean_tempo')
        for key in tuple(legacy['training_config']):
            if key.startswith('lr_scheduler'):
                legacy['training_config'].pop(key)
        legacy.pop('lr_scheduler_state')
        torch.save(legacy, path)
        resumed, _ = train_note_ae(corpus, self.folder / 'resumed', config=cfg, model_config=self.mc, score_config=self.sc, resume=self.folder / 'first/last.pt')
        for k, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[k], rtol=0, atol=0)

    def test_mean_auxiliary_training_checkpoint_and_exact_resume(self):
        corpus, _ = self.corpus()
        cfg = replace(self.tc, epochs=2, lambda_style=0., beta_mean_tempo=1.)
        mc = replace(self.mc, mean_tempo_auxiliary=True)
        full, report = train_note_ae(corpus, self.folder / 'aux-full', config=cfg, model_config=mc, score_config=self.sc)
        self.assertTrue(report['mean_tempo_auxiliary']['enabled'])
        self.assertEqual(report['history'][0]['mean_tempo_auxiliary']['valid_windows'], 3)
        self.assertIsNotNone(report['history'][0]['validation']['mean_tempo_auxiliary']['loss'])
        train_note_ae(corpus, self.folder / 'aux-first', config=replace(cfg, epochs=1), model_config=mc, score_config=self.sc)
        resumed, _ = train_note_ae(corpus, self.folder / 'aux-resumed', config=cfg, model_config=mc,
                                 score_config=self.sc, resume=self.folder / 'aux-first/last.pt')
        loaded, _, _, _ = load_note_ae_model(self.folder / 'aux-full/last.pt', corpus=corpus)
        for other in (resumed, loaded):
            for k, value in full.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[k], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'requires.*head'):
            train_note_ae(corpus, self.folder / 'missing-head', config=cfg, model_config=self.mc, score_config=self.sc)

    def test_full_epochs_and_early_stopping_keep_actual_validation_minimum(self):
        corpus, _ = self.corpus()
        cfg = replace(self.tc, epochs=10, batch_size=2, max_steps_per_epoch=0,
                      early_stopping_patience=2, early_stopping_min_delta=.1, evaluate_train_each_epoch=True)
        scores = iter([1., .95, .94])

        def evaluate(model, dataset, collate, config):
            model.eval()
            return {'loss': next(scores) if len(dataset) == 1 else .8, 'windows': len(dataset)}

        with patch('embedding.note_ae_training.evaluate_note_ae', side_effect=evaluate):
            _, report = train_note_ae(corpus, self.folder / 'early', config=cfg, model_config=self.mc, score_config=self.sc)
        self.assertEqual(report['stop_reason'], 'validation_early_stopping')
        self.assertEqual(report['completed_epoch'], 3)
        expected = {i['window']['window_id']: 1 for i in corpus.dataset('train').items}
        for row in report['history']:
            self.assertEqual(row['reconstruction_window_exposures'], expected)
            self.assertEqual(row['updates'], 2)  # The final short batch is included.
            self.assertTrue(row['complete_train_epoch'])
            self.assertEqual(row['full_train_evaluation']['windows'], 3)
        # Small improvements do not reset patience but still select the minimum.
        _, payload, _, _ = load_note_ae_model(self.folder / 'early/best.pt', corpus=corpus)
        self.assertEqual(payload['epoch'], 3)
        self.assertEqual(payload['best_validation'], .94)
        self.assertEqual(payload['early_stopping_state']['bad_epochs'], 2)

    def test_early_stopping_state_resumes_exactly(self):
        corpus, _ = self.corpus()
        cfg = replace(self.tc, epochs=3, max_steps_per_epoch=0, early_stopping_patience=3,
                      early_stopping_min_delta=100., evaluate_train_each_epoch=True)
        full, report = train_note_ae(corpus, self.folder / 'stop-full', config=cfg, model_config=self.mc, score_config=self.sc)
        train_note_ae(corpus, self.folder / 'stop-first', config=replace(cfg, epochs=2), model_config=self.mc, score_config=self.sc)
        resumed, resumed_report = train_note_ae(corpus, self.folder / 'stop-resume', config=cfg,
            model_config=self.mc, score_config=self.sc, resume=self.folder / 'stop-first/last.pt')
        self.assertEqual(resumed_report['early_stopping_state'], report['early_stopping_state'])
        self.assertEqual(report['early_stopping_state']['bad_epochs'], 2)
        for k, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[k], rtol=0, atol=0)

    def test_fixed_triplet_training_and_reconstruction_exposure(self):
        corpus, ws = self.corpus(many=True)
        triples = self.triplet_inputs(corpus, ws)
        self.assertGreater(len(triples), 0)
        _, report = train_note_ae(corpus, self.folder / 'joint', config=replace(self.tc, epochs=1, max_steps_per_epoch=2, triplet_batch_size=100), model_config=self.mc, score_config=self.sc, triplets=triples)
        self.assertTrue(report['style_training_performed'])
        row = report['history'][0]
        self.assertEqual(row['reconstruction']['windows'], 6)
        self.assertEqual(row['triplet_exposures'], 2 * len(triples))
        self.assertEqual(row['style_loader_restarts'], 1)
        self.assertEqual(sum(row['reconstruction_window_exposures'].values()), 6)
        self.assertTrue(np.isfinite(row['style_loss']))
        self.assertEqual(report['g_dependencies']['triplets'], triples.manifest_id)

    def triplet_inputs(self, corpus, ws, legacy=False, return_artifacts=False):
        cfg = StyleConfig()
        r = fit_reference(verify_window_sources(ws), cfg)
        ref = artifact('reference', cfg, parents={'windows': ws['id']}, **{k:v for k,v in r.items() if k not in {'id','kind','schema','config','config_id','parents'}})
        ps = extract_profile_artifact(ws, ref)
        if legacy:
            for p in ps['profiles']:
                for key in ('quality_schema', 'g_quality', 'reference_support', 'roughness_reference_support'):
                    p.pop(key)
                p['profile_id'] = stable_id({k: v for k, v in p.items() if k != 'profile_id'})
            ps['id'] = stable_id({k: v for k, v in ps.items() if k != 'id'})
        s = fit_profile_scaler(ps['profiles'], profiles_id=ps['id'])
        c = calibrate_distances(ps['profiles'], s, profiles_id=ps['id'])
        t = build_triplets(ps['profiles'], s, c, profiles_id=ps['id'])
        if return_artifacts:
            return t, ps, s, c
        return FixedNoteTriplets(corpus, t, ps, s, c)

    def test_g_loss_modes_training_and_exact_epoch_resume(self):
        corpus, ws = self.corpus(many=True)
        triples = self.triplet_inputs(corpus, ws)
        batch = triples.collate([triples[0], triples[1]])
        self.assertEqual(batch['confidence'].shape, (2,))
        self.assertTrue(torch.all(batch['g_gap_normalized'] > 0))
        torch.testing.assert_close(batch['confidence'], torch.full((2,), 2 / 3, dtype=torch.float64))
        for mode in ('legacy', 'confidence', 'adaptive', 'adaptive_confidence'):
            with self.subTest(mode=mode):
                cfg = replace(self.tc, epochs=2, style_objective=mode)
                full, report = train_note_ae(corpus, self.folder / (mode + '-full'), config=cfg,
                    model_config=self.mc, score_config=self.sc, triplets=triples)
                first = self.folder / (mode + '-first')
                train_note_ae(corpus, first, config=replace(cfg, epochs=1), model_config=self.mc,
                              score_config=self.sc, triplets=triples)
                path = first / 'last.pt'
                if mode == 'legacy':
                    payload = torch.load(path, weights_only=True)
                    payload.pop('style_objective')
                    for key in ('style_objective', 'margin_scale', 'margin_min', 'margin_max'):
                        payload['training_config'].pop(key)
                    torch.save(payload, path)
                resumed, _ = train_note_ae(corpus, self.folder / (mode + '-resume'), config=cfg,
                    model_config=self.mc, score_config=self.sc, triplets=triples, resume=path)
                for k, value in full.state_dict().items():
                    torch.testing.assert_close(value, resumed.state_dict()[k], rtol=0, atol=0)
                d = report['history'][0]['style_diagnostics']
                self.assertAlmostEqual(d['confidence_mean'], 2 / 3 if 'confidence' in mode else 1., places=6)
                self.assertGreaterEqual(d['unweighted_hinge'], report['history'][0]['style_loss'])
                with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                    train_note_ae(corpus, self.folder / (mode + '-wrong'),
                        config=replace(cfg, margin_scale=.3), model_config=self.mc, score_config=self.sc,
                        triplets=triples, resume=path)
        path = self.folder / 'adaptive_confidence-first/last.pt'
        payload = torch.load(path, weights_only=True)
        payload['style_objective']['version'] = 'wrong'
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, 'objective contract'):
            load_note_ae_model(path)

    def test_old_artifacts_and_missing_supervision(self):
        corpus, ws = self.corpus(many=True)
        triples = self.triplet_inputs(corpus, ws, legacy=True)
        current = self.triplet_inputs(corpus, ws)
        # Quality metadata does not change g values, cutoffs or chosen windows.
        keys = ('anchor_window_id', 'positive_window_id', 'negative_window_id', 'D_ap', 'D_an', 'delta_g')
        self.assertEqual([[t[k] for k in keys] for t in triples.rows],
                         [[t[k] for k in keys] for t in current.rows])
        self.assertGreater(len(triples), 0)
        triples.require_supervision('legacy')
        triples.require_supervision('adaptive')
        for mode in ('confidence', 'adaptive_confidence'):
            with self.assertRaisesRegex(ValueError, 'regenerated'):
                train_note_ae(corpus, self.folder / mode, config=replace(self.tc, style_objective=mode),
                    model_config=self.mc, score_config=self.sc, triplets=triples)
        with self.assertRaisesRegex(ValueError, 'nonempty fixed triplets'):
            train_note_ae(corpus, self.folder / 'missing', config=replace(self.tc, style_objective='adaptive'))
        for fields in triples.supervision:
            fields.pop('g_gap_normalized')
        with self.assertRaisesRegex(ValueError, 'valid calibration'):
            triples.require_supervision('adaptive')

    def test_quality_and_calibration_provenance_rejected(self):
        corpus, ws = self.corpus(many=True)
        t, ps, s, c = self.triplet_inputs(corpus, ws, return_artifacts=True)
        altered = copy.deepcopy(t)
        altered['triplets'][0]['quality'][0]['g_quality'][0] = .1
        altered['id'] = stable_id({k: v for k, v in altered.items() if k != 'id'})
        with self.assertRaisesRegex(ValueError, 'quality disagrees'):
            FixedNoteTriplets(corpus, altered, ps, s, c)
        c = copy.deepcopy(c)
        for entry in c['signatures'].values():
            entry['distance_scale'] = 0.
        c['id'] = stable_id({k: v for k, v in c.items() if k != 'id'})
        t = build_triplets(ps['profiles'], s, c, profiles_id=ps['id'])
        triples = FixedNoteTriplets(corpus, t, ps, s, c)
        self.assertGreater(len(triples), 0)
        triples.require_supervision('confidence')
        with self.assertRaisesRegex(ValueError, 'valid calibration'):
            triples.require_supervision('adaptive')

    def test_g_loss_cli_config_reaches_real_training(self):
        corpus, ws = self.corpus(many=True)
        t, ps, s, c = self.triplet_inputs(corpus, ws, return_artifacts=True)
        arguments = []
        for flag, value in (('--windows', ws), ('--triplets', t), ('--g-profiles', ps),
                            ('--g-scaler', s), ('--g-calibration', c)):
            path = self.folder / (flag[2:] + '.json')
            save_style_artifact(value, path)
            arguments += [flag, str(path)]
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / 'configs/note_ae_g_loss.json').read_text(encoding='utf-8'))
        config['training'].update(epochs=1, max_steps_per_epoch=1, batch_size=3)
        config_path = self.folder / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        out = self.folder / 'cli-training'
        subprocess.run([sys.executable, str(root / 'scripts/train_note_autoencoder.py'), 'train',
                        *arguments, '--config', str(config_path), '--out', str(out)],
                       check=True, capture_output=True, text=True, timeout=120)
        report = json.loads((out / 'training.json').read_text(encoding='utf-8'))
        self.assertTrue(report['style_training_performed'])
        self.assertEqual(report['style_objective']['mode'], 'adaptive_confidence')
        self.assertAlmostEqual(report['history'][0]['style_diagnostics']['confidence_mean'], 2 / 3, places=6)

    def test_bounded_full_pipeline_prepares_and_trains_all_windows(self):
        import importlib.util
        from embedding.note_ae_indexed_data import IndexedNoteAECorpus
        from embedding.note_score_context_data import save_note_score_context
        script = Path(__file__).resolve().parents[2] / 'scripts/run_local_atepp_ae.py'
        spec = importlib.util.spec_from_file_location('local_pipeline_test', script)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        eager, windows = self.corpus(many=True)
        out = self.folder / 'local-run'
        out.mkdir()
        runner.dump(out / 'inventory.json', {'scope': 'fixture'})
        results = []
        for source in windows['sources']:
            pair = eager.pairs[source['performance_id']]
            score_cache = self.folder / ('S-' + pair.score.cache_key)
            if not score_cache.exists():
                save_note_score_context(pair.score, score_cache)
            recording = next(w['recording_id'] for w in windows['windows']
                             if w['performance_id'] == source['performance_id'])
            results.append({**source, 'score_cache': str(score_cache), 'recording_id': recording})
        index = runner.prepare_inputs(out, results, lambda *args, **kwargs: None)
        runner.pack_inputs(out, index, lambda *args, **kwargs: None)
        runner.prepare_style(out, index, lambda *args, **kwargs: None)
        packed = IndexedNoteAECorpus(out / 'packed')
        from embedding.style_profile_data import load_style_artifact
        triples = FixedNoteTriplets(packed, *(load_style_artifact(out / 'style' / name) for name in
            ('triplets.json', 'profiles.json', 'scaler.json', 'calibration.json')))
        self.assertEqual(len(packed.items), len(eager.items))
        self.assertGreater(len(triples), 0)
        triples.require_supervision('adaptive_confidence')
        _, report = train_note_ae(packed, out / 'training', config=replace(self.tc, epochs=1,
            style_objective='adaptive_confidence'), model_config=self.mc, score_config=self.sc, triplets=triples)
        self.assertTrue(report['style_training_performed'])
        self.assertAlmostEqual(report['history'][0]['style_diagnostics']['confidence_mean'], 2 / 3, places=6)

    def test_plateau_steps_after_validation_and_logs_used_and_next_lr(self):
        corpus, _ = self.corpus()
        cfg = replace(self.tc, epochs=4, max_steps_per_epoch=0, lr_scheduler='plateau',
                      lr_scheduler_patience=1, lr_scheduler_factor=.5)

        def evaluate(model, dataset, collate, config):
            model.eval()
            return {'loss': 1., 'windows': len(dataset)}

        with patch('embedding.note_ae_training.evaluate_note_ae', side_effect=evaluate):
            _, report = train_note_ae(corpus, self.folder / 'plateau', config=cfg, model_config=self.mc, score_config=self.sc)
        self.assertEqual([r['learning_rate'][0] for r in report['history']], [.01, .01, .01, .005])
        self.assertEqual([r['next_learning_rate'][0] for r in report['history']], [.01, .01, .005, .005])
        _, payload, _, _ = load_note_ae_model(self.folder / 'plateau/last.pt', corpus=corpus)
        self.assertEqual(payload['lr_scheduler_state']['last_epoch'], 4)
        self.assertEqual(payload['optimizer']['param_groups'][0]['lr'], .005)

    def test_plateau_state_resumes_exactly_across_lr_change(self):
        corpus, _ = self.corpus()
        cfg = replace(self.tc, epochs=3, max_steps_per_epoch=0, lr_scheduler='plateau',
                      lr_scheduler_patience=0, lr_scheduler_threshold=100.)
        full, _ = train_note_ae(corpus, self.folder / 'lr-full', config=cfg, model_config=self.mc, score_config=self.sc)
        train_note_ae(corpus, self.folder / 'lr-first', config=replace(cfg, epochs=2), model_config=self.mc, score_config=self.sc)
        resumed, report = train_note_ae(corpus, self.folder / 'lr-resume', config=cfg, model_config=self.mc,
            score_config=self.sc, resume=self.folder / 'lr-first/last.pt')
        self.assertEqual(report['history'][0]['learning_rate'], [.005])
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
        a = torch.load(self.folder / 'lr-full/last.pt', weights_only=True)
        b = torch.load(self.folder / 'lr-resume/last.pt', weights_only=True)
        self.assertEqual(a['lr_scheduler_state'], b['lr_scheduler_state'])
