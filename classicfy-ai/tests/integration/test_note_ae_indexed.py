"""Indexed full-window batches must equal eager batches and preserve epoch coverage."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

AI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI / 'scripts'))
from prepare_note_ae_index import prepare, xml_score
from embedding.note_ae_data import NoteAECorpus
from embedding.note_ae_source_index import IndexedNoteAECorpus, RecordBlockSampler
from embedding.note_ae_training import NoteAETrainingConfig, load_note_ae_model, train_note_ae
from embedding.note_autoencoder import NoteAEConfig
from embedding.note_performance_data import save_note_performance
from embedding.note_score_context_data import save_note_score_context
from embedding.note_score_encoder import NoteScoreEncoderConfig
from embedding.style_profile_data import make_window_artifact
from features.note_performance_normalization import fit_note_performance_scaler
from features.note_score_context import extract_note_score_context
from unit.preprocessing.musicxml_fixtures import XMLFixture
from integration import test_style_triplets as style_helpers


class IndexedNoteAETest(XMLFixture):
    make_score = style_helpers.StyleTripletsTest.make_score
    record = style_helpers.StyleTripletsTest.record

    def setUp(self):
        super().setUp()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        xml_score.cache_clear()
        torch.set_num_threads(self.threads)
        super().tearDown()

    def build(self, cache_records=2):
        pre = self.folder / 'pre'; pre.mkdir()
        results, scores, entries, training = [], [], [], []
        for w, split in enumerate(('train', 'validation', 'test')):
            xml = self.make_score(bars=17, size=w + 1, prefix=f'w{w}')
            xpath = self.folder / f'score-{self.serial}.musicxml'
            score = extract_note_score_context(xml)
            scache = self.folder / f'S-{w}'; save_note_score_context(score, scache)
            scores.append(dict(status='prepared', source_sha256=xml.source_sha256, score_cache=str(scache)))
            for i, style in enumerate((-1., 1.) if split == 'train' else (0.,)):
                rec = self.record(xml, f'w{w}p{i}', style, work=f'w{w}', split=split)
                sha = hashlib.sha256(rec.recording_id.encode()).hexdigest()
                r = replace(rec.performance, provenance={**rec.performance.provenance, 'performance': {'midi_sha256': sha}})
                cache = self.folder / f'R-{w}-{i}'; save_note_performance(r, cache)
                results.append(dict(perf_id=r.record_key, status='extracted', performance_cache=str(cache), xml_path=str(xpath)))
                entries.append(dict(performance_cache=str(cache), xml_path=str(xpath), recording_group=sha))
                if split == 'train': training.append(r)
        for name, rows in (('alignments.jsonl', results), ('scores.jsonl', scores)):
            (pre / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (pre / 'alignment_summary.json').write_text(json.dumps({'extracted': len(results)}))
        (pre / 'splits.json').write_text(json.dumps({'policy': 'work-disjoint fixture'}))
        out = self.folder / 'indexed'
        index = prepare(pre, out)
        lazy = IndexedNoteAECorpus(index, out / 'R_scaler.json', out / 'S_scale.json', cache_records=cache_records)
        eager = NoteAECorpus(make_window_artifact(entries, self.folder), r_scaler=lazy.r_scaler, score_scale=lazy.score_scale)
        return lazy, eager, training, out

    def test_same_values_masks_padding_and_train_only_scalers_as_eager(self):
        lazy, eager, training, _ = self.build()
        expected = fit_note_performance_scaler(training, fit_manifest_id=lazy.index['id'])
        np.testing.assert_allclose(lazy.r_scaler.mean, expected.mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(lazy.r_scaler.scale, expected.scale, atol=1e-12, rtol=0)
        self.assertEqual(lazy.r_scaler.valid_counts, expected.valid_counts)
        selected = [lazy.dataset(split)[0] for split in ('train', 'validation', 'test')]
        batch = lazy.collate(selected)
        reference = eager.collate([eager.items[item['window']['window_id']] for item in selected])
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, reference[key])
        self.assertTrue((~batch['P']).any())
        self.assertFalse(batch['M'][~batch['P']].any())
        self.assertEqual(len(lazy.excluded), 4)

    def test_cache_bound_and_seeded_sampler_use_every_window_once(self):
        lazy, _, _, _ = self.build(cache_records=1)
        for split in ('train', 'validation', 'test'):
            _ = lazy.dataset(split)[0]
            self.assertLessEqual(len(lazy._records), 1)
            self.assertLessEqual(len(lazy._scores), 1)
        items = lazy.dataset('train').items
        one = list(RecordBlockSampler(items, torch.Generator().manual_seed(4), 1))
        two = list(RecordBlockSampler(items, torch.Generator().manual_seed(4), 1))
        self.assertEqual(one, two)
        self.assertEqual(sorted(one), list(range(len(items))))

    def test_changed_window_file_is_rejected(self):
        lazy, _, _, _ = self.build()
        source = lazy.sources[lazy.dataset('train').items[0]['window']['performance_id']]
        path = Path(source['windows_file']); path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Window file changed'):
            _ = lazy.dataset('train')[0]

    def test_real_training_loop_checkpoint_and_progress_with_indexed_inputs(self):
        lazy, _, _, out = self.build()
        cfg = NoteAETrainingConfig(epochs=1, batch_size=3, progress_interval_steps=1)
        mc = NoteAEConfig(context_dim=8, condition_dim=4, input_dim=8, hidden_size=4, latent_dim=3)
        sc = NoteScoreEncoderConfig(embedding_dim=2, note_dim=4, hidden_size=4, context_dim=8)
        _, report = train_note_ae(lazy, out / 'training', config=cfg, model_config=mc, score_config=sc)
        self.assertTrue(report['history'][0]['complete_train_epoch'])
        self.assertEqual(sum(report['history'][0]['reconstruction_window_exposures'].values()), len(lazy.dataset('train')))
        model, _, _, _ = load_note_ae_model(out / 'training/best.pt', corpus=lazy)
        self.assertIsNotNone(model)
        self.assertEqual(json.loads((out / 'training/progress.json').read_text())['stage'], 'complete')
        self.assertEqual(len((out / 'training/epochs.jsonl').read_text().splitlines()), 1)

    def test_epoch_resume_keeps_indexed_sampler_and_weights_identical(self):
        lazy, _, _, out = self.build(cache_records=1)
        cfg = NoteAETrainingConfig(epochs=2, batch_size=3)
        mc = NoteAEConfig(context_dim=8, condition_dim=4, input_dim=8, hidden_size=4, latent_dim=3)
        sc = NoteScoreEncoderConfig(embedding_dim=2, note_dim=4, hidden_size=4, context_dim=8)
        full, _ = train_note_ae(lazy, out / 'full', config=cfg, model_config=mc, score_config=sc)
        train_note_ae(lazy, out / 'first', config=replace(cfg, epochs=1), model_config=mc, score_config=sc)
        resumed, _ = train_note_ae(lazy, out / 'resumed', config=cfg, model_config=mc, score_config=sc,
                                 resume=out / 'first/last.pt')
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
