"""Indexed inputs preserve the original AE tensors, masks and gradients."""

import json
from pathlib import Path
import unittest

import torch

from embedding.note_ae_data import NoteAECorpus
from embedding.note_ae_indexed_data import IndexedNoteAECorpus, file_sha, pack_record
from embedding.note_ae_loss import reconstruction_loss
from embedding.note_ae_training import LengthBucketBatches
from embedding.note_autoencoder import NotePerformanceAutoencoder
from embedding.note_performance_data import save_note_performance
from embedding.note_score_context_data import save_note_score_context, save_note_score_scale
from embedding.style_profile_data import make_window_artifact
from features.note_performance_normalization import save_note_performance_scaler
from features.style_profile import stable_id
from integration.test_style_triplets import StyleTripletsTest
from unit.preprocessing.musicxml_fixtures import XMLFixture


class IndexedNoteAEDataTest(XMLFixture):
    make_score = StyleTripletsTest.make_score
    record = StyleTripletsTest.record

    def setUp(self):
        super().setUp()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)

    def corpus(self):
        entries = []
        for k, split in enumerate(('train', 'validation')):
            xml = self.make_score(size=k + 1, prefix=f'work{k}')
            xml_path = self.folder / f'score-{self.serial}.musicxml'
            for j in range(2):
                record = self.record(xml, f'p{k}-{j}', j, work=f'w{k}', split=split)
                path = self.folder / f'R{k}-{j}'
                save_note_performance(record.performance, path)
                entries.append({'performance_cache': str(path), 'xml_path': str(xml_path),
                                'recording_group': record.recording_id})
        windows = make_window_artifact(entries, self.folder)
        reference = NoteAECorpus(windows)
        packed = self.folder / 'packed'
        (packed / 'scores').mkdir(parents=True)
        (packed / 'records').mkdir()
        save_note_performance_scaler(reference.r_scaler, packed / 'R_scaler.json')
        save_note_score_scale(reference.score_scale, packed / 'S_scale.json')
        results = []
        for entry in windows['sources']:
            pair = reference.pairs[entry['performance_id']]
            score_cache = self.folder / ('score-' + pair.score.cache_key)
            if not score_cache.exists():
                save_note_score_context(pair.score, score_cache)
            owned = [w for w in windows['windows'] if w['performance_id'] == entry['performance_id']]
            source_file = self.folder / (entry['record_key'] + '.json')
            source_file.write_text(json.dumps(owned))
            source = {**entry, 'windows_file': str(source_file), 'windows_sha256': file_sha(source_file)}
            results.append(pack_record((source, score_cache, packed, packed / 'R_scaler.json', packed / 'S_scale.json')))
        manifest = {'format': 'indexed-note-ae-tensors-v1', 'source_index_id': windows['id'],
                    'R_scaler': reference.r_scaler.fingerprint,
                    'score_scale': stable_id(reference.score_scale.__dict__),
                    'items': [v for r in results for v in r['items']],
                    'excluded': [], 'splits': reference.contract['split'],
                    'tensor_hashes': {kind: {p.stem: file_sha(p) for p in (packed / kind).glob('*.pt')}
                                      for kind in ('records', 'scores')}}
        manifest['id'] = stable_id(manifest)
        (packed / 'manifest.json').write_text(json.dumps(manifest))
        return reference, IndexedNoteAECorpus(packed)

    def test_collator_outputs_loss_and_gradients_match_original(self):
        original, indexed = self.corpus()
        ids = sorted(indexed.items)
        expected = original.collate([original.items[k] for k in ids])
        actual = indexed.collate([indexed.items[k] for k in ids])
        for key, value in actual.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertTrue((~actual['P']).any())
        torch.manual_seed(2026)
        model = NotePerformanceAutoencoder().eval()
        gradients = []
        for batch in (expected, actual):
            model.zero_grad()
            output = model(batch)
            loss = reconstruction_loss(output['R_hat'], batch['R'], batch['M'], batch['P'], batch['beat_id'])
            loss['loss'].backward()
            gradients.append([p.grad.clone() for p in model.parameters()])
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_corrupt_packed_tensor_is_rejected(self):
        _, indexed = self.corpus()
        path = next((indexed.path / 'records').glob('*.pt'))
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            IndexedNoteAECorpus(indexed.path)

    def test_length_buckets_cover_every_window_and_resume_generator(self):
        class Windows:
            items = [{'window': {'N': n}} for n in range(109)]
        generator = torch.Generator().manual_seed(2026)
        sampler = LengthBucketBatches(Windows(), 4, shuffle=True, generator=generator)
        first = list(sampler)
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(109)))
        self.assertEqual(len(first), len(sampler))
        state = generator.get_state()
        second = list(sampler)
        generator.set_state(state)
        self.assertEqual(second, list(sampler))
        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
