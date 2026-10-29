"""Hand-computable checks for the pre-AE note performance contract."""

from dataclasses import replace
import hashlib

import mido
import numpy as np

from embedding.note_performance_data import (
    NotePerformancePair, collate_note_performance, export_scaled_note_performance,
    load_note_performance, save_note_performance, split_note_performance,
)
from features.note_performance import extract_note_performance, inverse_r_physical
from features.note_performance_normalization import (
    fit_note_performance_scaler, load_note_performance_scaler,
    save_note_performance_scaler, validate_work_splits,
)
from features.note_score_context import extract_note_score_context
from features.performance_time import BeatTimeMap
from preprocessing.note_alignment import NoteAlignment, PerformedNote, ScoreNote
from preprocessing.note_performance_adapters import bind_alignment_to_raw_midi, prepare_atepp_note_performance
from preprocessing.performance_midi import (
    ControlEvent, PedalObservation, PerformanceMidi, PerformanceNote,
    load_performance_midi, score_seconds_to_quarters,
)
from preprocessing.score_note_mapping import link_note_alignment
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, measure, note, score_xml


class NotePerformancePipelineTest(XMLFixture):
    def setUp(self):
        super().setUp()
        body = (attributes() + note('a', extra='<tie type="start"/>')
                + note('b', extra='<tie type="stop"/>') + note('c', 'D') + note('d', 'E'))
        self.xml = self.read(score_xml(measure(body)))
        self.score = extract_note_score_context(self.xml)
        notes = (PerformanceNote('a', 60, .1, 1.1, 64),
                 PerformanceNote('c', 62, 1.2, 1.45, 80),
                 PerformanceNote('d', 64, 1.7, 1.9, 96))
        controls = tuple(ControlEvent(t, 64, v, order=i) for i, (t, v) in
                         enumerate(((0., 100), (.5, 20), (1.1, 0), (1.2, 80), (1.4, 0))))
        observations = {(0, 0, 64): PedalObservation('observed', True, source='fixture'),
                        (0, 0, 67): PedalObservation('confirmed_absent', True, source='fixture')}
        self.performance = PerformanceMidi(notes, controls, 2., ((0, 0),), observations)
        attacks = (self.xml.notes[i] for i in (0, 2, 3))
        self.alignment = NoteAlignment(tuple(
            (ScoreNote(s.occurrence_id, s.pitch, float(s.onset), float(s.offset), ()),
             PerformedNote(p.note_id, p.pitch, p.onset_sec, p.key_offset_sec, p.velocity))
            for s, p in zip(attacks, notes)), (), ())
        self.links = link_note_alignment(self.xml, self.score.ties, self.alignment)
        self.beats = BeatTimeMap(np.arange(5), np.arange(5) * .5,
                                 provenance={'granularity': 'beat', 'source': 'fixture'})
        self.r = self.extract()

    def extract(self, **changes):
        return extract_note_performance(self.score, changes.get('performance', self.performance),
                                        changes.get('links', self.links), changes.get('beats', self.beats),
                                        record_key='fixture', work_key='work', split='train', dataset='synthetic')

    def test_hand_computed_targets_and_tie_owner(self):
        np.testing.assert_allclose(self.r.values[0, :4], [np.log2(120), .2, 64 / 127, 0])
        self.assertAlmostEqual(self.r.expected_duration_sec[0], 1.)
        self.assertFalse(self.r.mask[1].any())
        self.assertTrue(self.r.note_mask[1])
        self.assertAlmostEqual(self.r.values[2, 3], -1.)
        physical, valid = inverse_r_physical(self.r.values, self.r.mask, self.r.local_tau)
        np.testing.assert_allclose(physical[valid], self.r.physical[valid])

    def test_closed_hold_and_open_gap_boundaries(self):
        np.testing.assert_allclose(self.r.physical[0, 4:10], [100, 0, 0, 1., 0, 0])
        self.assertFalse(self.r.mask[-1, 8:10].any())
        # An inserted, unaligned note must bound the preceding note's gap.
        extra = PerformanceNote('inserted', 70, 1.5, 1.6, 60)
        r = self.extract(performance=replace(self.performance, notes=self.performance.notes + (extra,)))
        self.assertEqual(r.metadata[2]['next_onset_sec'], 1.5)

    def test_unknown_pedal_is_not_zero_observation(self):
        r = self.extract(performance=replace(self.performance, controls=(), pedal_observations={}))
        self.assertFalse(r.mask[:, 4:].any())
        self.assertTrue(r.mask[0, :4].all())
        before_first = replace(self.performance, controls=self.performance.controls[1:])
        self.assertFalse(self.extract(performance=before_first).mask[0, 4])

    def test_missing_key_release_preserves_independent_targets(self):
        notes = (replace(self.performance.notes[0], key_offset_sec=None),) + self.performance.notes[1:]
        r = self.extract(performance=replace(self.performance, notes=notes))
        self.assertFalse(r.mask[0, 3])
        self.assertFalse(r.mask[0, 5:10].any())
        self.assertTrue(r.mask[0, [0, 1, 2, 4, 10]].all())

    def test_unavailable_and_invalid_timing_masks(self):
        r = self.extract(beats=BeatTimeMap.unavailable('fixture', 'no beats'))
        self.assertEqual(len(r.beats.valid), 0)
        self.assertFalse(r.mask[:, [0, 1, 3, 7, 9]].any())
        self.assertTrue(r.mask[0, [2, 4, 5, 6, 8, 10]].all())
        beats = replace(self.beats, valid=np.array([True, False, True, True]))
        self.assertIsNone(beats.expected_duration(0, 2))
        self.assertFalse(self.extract(beats=beats).mask[0, 3])
        varying = BeatTimeMap(np.array([0, 1, 2]), np.array([0., .5, 1.5]),
                              provenance=self.beats.provenance)
        self.assertEqual(varying.expected_duration(.5, 1), .75)

    def test_raw_binding_ambiguity_and_score_coordinate_validation(self):
        linked, audit = bind_alignment_to_raw_midi(self.xml, self.score, self.performance, self.alignment)
        self.assertEqual(audit['unresolved_midi_ids'], [])
        np.testing.assert_array_equal(linked.attack_mask, self.links.attack_mask)
        duplicate = replace(self.performance.notes[0], note_id='duplicate')
        linked, _ = bind_alignment_to_raw_midi(
            self.xml, self.score, replace(self.performance, notes=self.performance.notes + (duplicate,)), self.alignment)
        self.assertFalse(self.extract(links=linked).mask[0].any())
        first, *others = self.alignment.matches
        bad = replace(self.alignment, matches=((replace(first[0], onset_beats=9), first[1]), *others))
        with self.assertRaisesRegex(ValueError, 'coordinate'):
            bind_alignment_to_raw_midi(self.xml, self.score, self.performance, bad)

    def test_atepp_masks_extreme_beats_and_dependent_targets(self):
        # A file may pass whole-alignment acceptance while one beat collapses.
        path = self.folder / 'performance.mid'
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        events = []
        for n in self.performance.notes:
            events.extend([(round(n.onset_sec * 960), 'note_on', n.pitch, n.velocity),
                           (round(n.key_offset_sec * 960), 'note_off', n.pitch, 0)])
        events.extend((round(c.time_sec * 960), 'control_change', c.number, c.value)
                      for c in self.performance.controls)
        previous = 0
        for tick, kind, pitch, velocity in sorted(events):
            fields = ({'control': pitch, 'value': velocity} if kind == 'control_change'
                      else {'note': pitch, 'velocity': velocity})
            track.append(mido.Message(kind, **fields, time=tick - previous))
            previous = tick
        midi.save(path)
        cache = self.folder / 'alignment.npz'
        matches = self.alignment.matches
        columns = dict(
            score_ids=[s.note_id for s, _ in matches], performance_ids=[p.note_id for _, p in matches],
            score_onsets=[s.onset_beats for s, _ in matches], score_offsets=[s.offset_beats for s, _ in matches],
            performance_onsets=[p.onset for _, p in matches], performance_offsets=[p.offset for _, p in matches],
            pitch=[s.pitch for s, _ in matches], score_beats=np.arange(5),
            performance_beats=np.array([0., .5, 1., 1. + 1 / 240, 1.5 + 1 / 240]))
        score_path = self.folder / 'score.mid'
        mido.MidiFile().save(score_path)
        bridge = {'xml_sha256': self.xml.source_sha256,
                  'score_midi_sha256': hashlib.sha256(score_path.read_bytes()).hexdigest(),
                  'score_coordinate_unit': 'quarter', 'alignment_accepted': True,
                  'score_id_to_occurrence': {s.note_id: s.note_id for s, _ in matches}}
        for supplied_mask in (None, np.array([False, True, True, True])):
            with self.subTest(supplied_mask=supplied_mask):
                np.savez(cache, **columns, **({} if supplied_mask is None else {'beat_valid': supplied_mask}))
                _, r = prepare_atepp_note_performance(
                    self.folder / 'score-1.musicxml', score_path, path, cache, bridge,
                    record_key='fixture', work_key='work', split='train', score_features=self.score,
                    pedal_observations=self.performance.pedal_observations)
                self.assertFalse(r.beats.valid[2])
                self.assertEqual(r.provenance['adapter'], 'atepp-note-r-v2')
                self.assertFalse(r.mask[2, [0, 1, 3, 7, 9]].any())
                self.assertTrue(r.mask[2, [2, 4, 5, 6, 8, 10]].all())
                self.assertTrue(r.mask[3, :3].all())
                self.assertEqual(bool(r.beats.valid[0]), supplied_mask is None)
                scaler = fit_note_performance_scaler([r], fit_manifest_id='masked-estimated-beats')
                self.assertEqual(scaler.valid_counts[0], int(r.beats.valid.sum()))
                batch = collate_note_performance([NotePerformancePair(self.score, r)], scaler)
                self.assertFalse(batch['beat_mask'][0, 2])
                self.assertFalse(batch['beat_loss_mask'][0, 2])
                self.assertFalse(batch['R_mask'][0, 2, [0, 1, 3, 7, 9]].any())
                self.assertTrue(np.isfinite(batch['R_scaled'].numpy()).all())

    def test_scaler_train_only_roundtrip_and_beat_weighting(self):
        scaler = fit_note_performance_scaler([self.r, self.r], fit_manifest_id='fixture')
        self.assertEqual(scaler.valid_counts[0], 4)
        self.assertEqual(scaler.valid_counts[2], 3)
        self.assertAlmostEqual(scaler.mean[2], 80 / 127)
        scaled = scaler.transform(self.r)
        self.assertTrue(np.isfinite(scaled).all())
        self.assertTrue((scaled[~self.r.mask] == 0).all())
        np.testing.assert_allclose(scaler.inverse_transform(scaled, self.r.mask)[self.r.mask],
                                   self.r.values[self.r.mask], atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'train'):
            fit_note_performance_scaler([replace(self.r, split='test')], fit_manifest_id='fixture')
        with self.assertRaisesRegex(ValueError, 'splits'):
            validate_work_splits([self.r, replace(self.r, split='test')])
        missing = self.extract(performance=replace(self.performance, controls=(), pedal_observations={}))
        fallback = fit_note_performance_scaler([missing], fit_manifest_id='fixture')
        self.assertEqual(fallback.channel_status[4], 'unobserved')
        self.assertEqual((fallback.mean[4], fallback.scale[4]), (0, 1))

    def test_cache_scaler_export_and_corruption(self):
        out = self.folder / 'cache'
        save_note_performance(self.r, out)
        self.assertEqual(load_note_performance(out).fingerprint, self.r.fingerprint)
        scaler = fit_note_performance_scaler([self.r], fit_manifest_id='fixture')
        path = self.folder / 'scale.json'
        save_note_performance_scaler(scaler, path)
        self.assertEqual(load_note_performance_scaler(path).fingerprint, scaler.fingerprint)
        export_scaled_note_performance(self.r, scaler, self.folder / 'scaled.npz')
        with np.load(self.folder / 'scaled.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['R_scaled'], scaler.transform(self.r))
        (out / 'performance.npz').write_bytes((out / 'performance.npz').read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            load_note_performance(out)

    def test_chunk_ownership_padding_and_source_guard(self):
        scaler = fit_note_performance_scaler([self.r], fit_manifest_id='fixture')
        pair = NotePerformancePair(self.score, self.r)
        chunks = split_note_performance(pair, max_onsets=2, overlap=1)
        batch = collate_note_performance(chunks, scaler)
        self.assertEqual(int(batch['beat_loss_mask'].sum()), 4)
        self.assertEqual(int(batch['R_loss_mask'][:, :, 2].sum()), 3)
        self.assertFalse(batch['R_loss_mask'][:, :, 0].any())
        self.assertFalse(batch['R_mask'][~batch['note_mask']].any())
        for bi, meta in enumerate(batch['R_metadata']):
            if 'b@1' in meta['row_keys']:
                row = meta['row_keys'].index('b@1')
                self.assertTrue(batch['note_mask'][bi, row])
                self.assertFalse(batch['R_mask'][bi, row].any())
        with self.assertRaisesRegex(ValueError, 'S/R'):
            NotePerformancePair(self.score, replace(self.r, score_source_sha256='foreign'))

    def test_raw_midi_note_off_and_tempo_changes(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.extend([mido.Message('control_change', control=64, value=127),
                      mido.Message('note_on', note=60, velocity=64),
                      mido.Message('note_off', note=60, time=480),
                      mido.MetaMessage('set_tempo', tempo=1000000),
                      mido.Message('control_change', control=64, value=0, time=480)])
        path = self.folder / 'raw.mid'
        midi.save(path)
        performance = load_performance_midi(path)
        self.assertEqual(performance.notes[0].key_offset_sec, .5)
        self.assertEqual(performance.duration_sec, 1.5)
        self.assertEqual(performance.observation(0, 0, 67).status, 'unknown')
        np.testing.assert_allclose(score_seconds_to_quarters(path, [0, .5, 1., 1.5]), [0, 1, 1.5, 2])
