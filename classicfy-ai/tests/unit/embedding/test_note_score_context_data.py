from dataclasses import replace
import json

import numpy as np
import torch

from embedding.note_score_context_data import (
    collate_note_score_context, load_note_score_context, load_note_score_scale,
    save_note_score_context, save_note_score_scale, split_note_score_context,
)
from features.note_score_context import extract_note_score_context, fit_note_score_scale
from preprocessing.score_note_mapping import link_note_alignment
from unit.preprocessing.musicxml_fixtures import XMLFixture
from unit.preprocessing.test_score_note_mapping import alignment_for


class NoteScoreDataTest(XMLFixture):
    def setUp(self):
        super().setUp()
        self.score = self.read()
        self.features = extract_note_score_context(self.score)
        self.links = link_note_alignment(self.score, self.features.ties, alignment_for(self.score))

    def test_cache_round_trip_including_alignment_and_scale(self):
        out = self.folder / 'cache'
        save_note_score_context(self.features, out, links=self.links)
        restored, links = load_note_score_context(out)
        for name in ('numeric', 'categorical', 'flags', 'field_known', 'onset_q', 'note_to_onset'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(self.features, name))
        self.assertEqual(restored.cache_key, self.features.cache_key)
        self.assertEqual(restored.ties.occurrence_ids, self.features.ties.occurrence_ids)
        np.testing.assert_array_equal(links.attack_mask, self.links.attack_mask)
        scale = fit_note_score_scale([self.features])
        save_note_score_scale(scale, self.folder / 'scale.json')
        self.assertEqual(load_note_score_scale(self.folder / 'scale.json'), scale)

    def test_cache_rejects_overwrite_corruption_and_wrong_schema(self):
        out = self.folder / 'cache'
        save_note_score_context(self.features, out)
        with self.assertRaises(FileExistsError):
            save_note_score_context(self.features, out)
        (out / 'features.npz').write_bytes((out / 'features.npz').read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            load_note_score_context(out)
        other = self.folder / 'other'
        save_note_score_context(self.features, other)
        manifest = json.loads((other / 'manifest.json').read_text())
        manifest['schema']['pooling'] = 'wrong'
        (other / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'schema'):
            load_note_score_context(other)

    def test_pre_fix_cache_and_scale_versions_are_rejected(self):
        out = self.folder / 'old-cache'
        save_note_score_context(self.features, out)
        manifest = json.loads((out / 'manifest.json').read_text())
        manifest['schema']['version'] = 'note-score-context-v1'
        (out / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'schema'):
            load_note_score_context(out)
        scale = fit_note_score_scale([self.features])
        path = self.folder / 'old-scale.json'
        save_note_score_scale(scale, path)
        payload = json.loads(path.read_text())
        payload['schema_version'] = 'note-score-context-v1'
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'schema'):
            load_note_score_scale(path)

    def test_chunk_loss_ownership_is_disjoint_without_splitting_chords(self):
        chunks = split_note_score_context(self.features, max_onsets=2, overlap=1)
        batch = collate_note_score_context(chunks, alignment_links=[self.links] * len(chunks))
        owners = np.zeros(len(self.features.metadata), int)
        attack_targets = np.zeros_like(owners)
        for bi, meta in enumerate(batch['metadata']):
            rows = meta['note_indices']
            owners[rows] += batch['loss_note_mask'][bi, :len(rows)].numpy()
            attack_targets[rows] += batch['attack_target_mask'][bi, :len(rows)].numpy()
            groups = self.features.note_to_onset[rows]
            for group in np.unique(groups):
                self.assertEqual(int((groups == group).sum()), int((self.features.note_to_onset == group).sum()))
        np.testing.assert_array_equal(owners, 1)
        np.testing.assert_array_equal(attack_targets, self.links.attack_mask.astype(int))

    def test_crop_inside_tie_keeps_global_owner_without_reassignment(self):
        chunk = split_note_score_context(self.features, max_onsets=1, overlap=0)[1]
        batch = collate_note_score_context([chunk], alignment_links=[self.links])
        self.assertEqual(batch['metadata'][0]['notes'][0]['occurrence_id'], 'c2@1')
        self.assertEqual(batch['metadata'][0]['notes'][0]['attack_owner_occurrence_id'], 'c1@1')
        self.assertFalse(batch['attack_target_mask'].any())
        self.assertFalse(batch['duration_target_mask'].any())
        self.assertEqual(float(batch['tie_chain_duration_q'][0, 0]), 3)
        self.assertTrue(batch['note_mask'].all())

    def test_reconstruction_masks_do_not_change_score_pool_membership(self):
        batch = collate_note_score_context([self.features], alignment_links=[self.links])
        self.assertTrue(batch['note_mask'][0, 2])
        self.assertFalse(batch['attack_target_mask'][0, 2])
        self.assertTrue(torch.equal(batch['field_known'][0], torch.from_numpy(self.features.field_known.copy())))
        foreign = replace(self.links, occurrence_ids=tuple('other-' + x for x in self.links.occurrence_ids))
        with self.assertRaisesRegex(ValueError, 'different score'):
            collate_note_score_context([self.features], alignment_links=[foreign])

    def test_crop_retains_directives_and_full_score_numeric_scale(self):
        scale = fit_note_score_scale([self.features])
        full = collate_note_score_context([self.features], scale=scale)
        for chunk in split_note_score_context(self.features, max_onsets=2, overlap=1):
            batch = collate_note_score_context([chunk], scale=scale)
            rows = batch['metadata'][0]['note_indices']
            torch.testing.assert_close(batch['numeric'][0, :len(rows)], full['numeric'][0, rows])
            torch.testing.assert_close(batch['categorical'][0, :len(rows)], full['categorical'][0, rows])
            torch.testing.assert_close(batch['field_known'][0, :len(rows)], full['field_known'][0, rows])
