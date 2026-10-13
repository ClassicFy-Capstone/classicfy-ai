import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import mido
import numpy as np

from preprocessing import NoteAlignment, PerformedNote, ScoreNote, load_musicxml
from preprocessing.score_beat_bridge import (
    MidiScoreNote, MidiScoreTimeline, build_score_beat_bridge, load_midi_score_timeline,
)


def xml_note(identifier, step='C', extra=''):
    return (f'<note id="{identifier}"><pitch><step>{step}</step><octave>4</octave></pitch>'
            f'<duration>1</duration><voice>1</voice><staff>1</staff>{extra}</note>')


class ScoreBeatBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.score = self.load(''.join(xml_note('n'+str(i), s)
                                     for i, s in enumerate(('C', 'D', 'E', 'F'))))
        self.references = tuple(ScoreNote('n'+str(i)+'-1', p, i, i+1, ())
                                for i, p in enumerate((60, 62, 64, 65)))
        self.alignment = NoteAlignment(tuple((s, PerformedNote(str(i), s.pitch,
            i * .7, i * .7 + .5, 64)) for i, s in enumerate(self.references)), (), ())
        self.midi = MidiScoreTimeline(480, tuple(MidiScoreNote(s.pitch,
            Fraction(i), Fraction(i+1)-Fraction(1, 480), 0, 0)
            for i, s in enumerate(self.references)), ((0, Fraction(0), 500_000),), 1920)

    def load(self, body):
        path = Path(self.temp.name) / 'score.musicxml'
        path.write_text('<score-partwise><part id="P1"><measure number="0" implicit="yes">'
            '<attributes><divisions>1</divisions></attributes>'+body+
            '</measure></part></score-partwise>')
        return load_musicxml(path)

    def bridge(self, **overrides):
        args = dict(score=self.score, alignment=self.alignment, midi=self.midi,
            score_beats=[0, .5, 1, 1.5, 2], performance_beats=[0, .8, 1.7, 2.2, 3.1],
            score_beat_types=['db', 'b', 'b', 'b', 'db'])
        args.update(overrides)
        return build_score_beat_bridge(**args)

    def test_half_open_boundaries_and_original_indices(self):
        b = self.bridge(score_beats=[.5, 1, 1.5], performance_beats=[.8, 1.7, 2.2],
                        score_beat_types=['b', 'b', 'b'])
        self.assertEqual([l.beat_index for l in b.note_links], [None, 0, 1, None])
        self.assertEqual(b.note_links[-1].beat_status, 'outside_beat_range')
        self.assertEqual([i.xml_start_quarters for i in b.intervals], [1, 2])
        self.assertEqual([i.measure_number for i in b.intervals], ['0', '0'])
        exact = replace(self.alignment, matches=tuple((replace(s, note_id=s.note_id[:-2]), p)
                                                      for s, p in self.alignment.matches))
        self.assertTrue(all(l.method=='exact_xml_id' for l in self.bridge(alignment=exact).note_links))

    def test_deletions_and_performed_timing_do_not_define_symbolic_mapping(self):
        b = self.bridge()
        changed = replace(self.alignment, matches=self.alignment.matches[1:],
                          deletions=(self.references[0],))
        other = self.bridge(alignment=changed, performance_beats=[0, 2, 4, 6, 8])
        self.assertEqual(b.note_links, other.note_links)
        self.assertEqual([x.xml_start_quarters for x in b.intervals],
                         [x.xml_start_quarters for x in other.intervals])

    def test_offset_is_inferred_and_validated_instead_of_treating_quarters_as_indices(self):
        refs = tuple(replace(s, onset_beats=s.onset_beats+1, offset_beats=s.offset_beats+1)
                     for s in self.references)
        alignment = replace(self.alignment, matches=tuple((s, p) for s, (_, p) in
                            zip(refs, self.alignment.matches)))
        midi = replace(self.midi, notes=tuple(replace(n, onset_quarters=n.onset_quarters+1,
                       offset_quarters=n.offset_quarters+1) for n in self.midi.notes), end_tick=2400)
        b = self.bridge(alignment=alignment, midi=midi, score_beats=[.5, 1, 1.5, 2, 2.5])
        self.assertEqual(b.xml_to_score_offset_quarters, 1)
        self.assertEqual([l.beat_index for l in b.note_links], [0, 1, 2, 3])

    def test_tie_head_duration_and_continuation_are_separate(self):
        score = self.load(xml_note('n0', extra='<tie type="start"/>')+
                          xml_note('n1', extra='<tie type="stop"/>')+xml_note('n2', 'E'))
        refs = (ScoreNote('n0-1', 60, 0, 2, ()), ScoreNote('n2-1', 64, 2, 3, ()))
        alignment = NoteAlignment((), refs, ())
        midi = replace(self.midi, notes=(MidiScoreNote(60, Fraction(0), Fraction(2), 0, 0),
                                       MidiScoreNote(64, Fraction(2), Fraction(3), 0, 0)))
        b = self.bridge(score=score, alignment=alignment, midi=midi)
        self.assertEqual([l.status for l in b.note_links],
                         ['verified', 'tie_continuation_verified', 'verified'])
        self.assertEqual(b.note_links[1].tie_head_id, b.note_links[0].note_id)
        self.assertFalse(b.intervals[-1].valid)

    def test_pitch_duration_and_missing_midi_have_explicit_status(self):
        refs = list(self.references)
        refs[0] = replace(refs[0], pitch=61)
        refs[1] = replace(refs[1], offset_beats=2.5)
        alignment = replace(self.alignment, matches=tuple((s, p) for s, (_, p) in
                            zip(refs, self.alignment.matches)))
        b = self.bridge(alignment=alignment)
        self.assertEqual([l.status for l in b.note_links[:2]], ['pitch_mismatch', 'duration_mismatch'])
        midi = replace(self.midi, notes=self.midi.notes[1:])
        self.assertEqual(self.bridge(midi=midi).note_links[0].status, 'missing_midi')
        midi = replace(self.midi, notes=(replace(self.midi.notes[0], offset_quarters=Fraction(1,2)),)
                       + self.midi.notes[1:])
        self.assertEqual(self.bridge(midi=midi).note_links[0].status, 'verified_onset_duration_differs')

    def test_ambiguous_midi_is_not_arbitrarily_selected(self):
        midi = replace(self.midi, notes=self.midi.notes + (self.midi.notes[0],))
        self.assertEqual(self.bridge(midi=midi).note_links[0].status, 'ambiguous_midi')
        grace = replace(self.midi.notes[0], offset_quarters=Fraction(0))
        midi = replace(self.midi, notes=self.midi.notes + (grace,))
        self.assertEqual(self.bridge(midi=midi).note_links[0].status, 'verified')

    def test_rest_grace_and_broken_tie_remain_observable(self):
        score = self.load(xml_note('n0')+xml_note('n1', 'D')+
            '<note id="rest"><rest/><duration>1</duration></note>'+
            '<note id="grace"><grace/><pitch><step>G</step><octave>4</octave></pitch></note>'+
            xml_note('n3', 'F', '<tie type="start"/>'))
        refs = tuple(s for s in self.references if s.note_id != 'n2-1')
        b = self.bridge(score=score, alignment=NoteAlignment((), refs, ()))
        self.assertEqual([l.status for l in b.note_links[-3:]],
                         ['rest_position_only', 'grace_unverified', 'ambiguous_tie'])
        orphan = self.load(xml_note('n0')+xml_note('n1', 'D')+xml_note('n2', 'E')+
                           xml_note('n3', 'F', '<tie type="stop"/>'))
        self.assertEqual(self.bridge(score=orphan).note_links[-1].status, 'orphan_tie_stop')

    def test_zero_width_and_irregular_beats_are_not_deleted(self):
        b = self.bridge(score_beats=[0, .5, .5, 1.5, 2], score_beat_types=['db', 'bR', 'b', 'b', 'db'])
        self.assertEqual(len(b.intervals), 4)
        self.assertEqual(b.intervals[1].status, 'zero_width')
        self.assertFalse(b.intervals[1].valid)
        self.assertEqual(b.intervals[0].status, 'verified_irregular')
        b = self.bridge(performance_beats=[0, .8, .8, 2.2, 3.1])
        self.assertEqual(b.note_links[1].beat_status, 'zero_width')

    def test_compact_selection_preserves_gaps_and_rejects_bad_indices(self):
        b = self.bridge()
        selected = b.select_intervals(np.array([0, 2], dtype=int))
        self.assertEqual([i.index for i in selected], [0, 2])
        for bad in ([1, 1], [-1], [4], [1.5], [True], [2, 1]):
            with self.subTest(indices=bad), self.assertRaises(ValueError):
                b.select_intervals(bad)

    def test_repeats_unknown_ids_and_nonuniform_offsets_fail(self):
        score = self.load(''.join(xml_note('n'+str(i), s) for i, s in
                         enumerate(('C', 'D', 'E', 'F')))+
                         '<barline><repeat direction="backward"/></barline>')
        with self.assertRaisesRegex(ValueError, 'Repeats'):
            self.bridge(score=score)
        for ref in (replace(self.references[0], note_id='n0-2'),
                    replace(self.references[0], note_id='unknown-1'),
                    replace(self.references[0], onset_beats=.5)):
            alignment = replace(self.alignment, matches=((ref, self.alignment.matches[0][1]),)+self.alignment.matches[1:])
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.bridge(alignment=alignment)
        alignment = replace(self.alignment, matches=self.alignment.matches+(self.alignment.matches[0],))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.bridge(alignment=alignment)

    def test_invalid_beats_and_missing_midi_anchors_fail(self):
        for overrides in (dict(score_beats=[0]), dict(performance_beats=[0, float('nan'), 1, 2, 3]),
                          dict(score_beats=[0, 1, .5, 1.5, 2]),
                          dict(score_beat_types=['db', 'unknown', 'b', 'b', 'db'])):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.bridge(**overrides)
        with self.assertRaisesRegex(ValueError, 'MIDI anchors'):
            self.bridge(midi=replace(self.midi, notes=(self.midi.notes[0],)))

    def test_midi_tempo_conversion_and_zero_length_grace(self):
        path = Path(self.temp.name) / 'score.mid'
        midi = mido.MidiFile(ticks_per_beat=480)
        tempo = mido.MidiTrack()
        tempo.extend([mido.MetaMessage('set_tempo', tempo=500_000),
                      mido.MetaMessage('set_tempo', tempo=1_000_000, time=480)])
        notes = mido.MidiTrack()
        notes.extend([mido.Message('note_on', note=60, velocity=64),
                      mido.Message('note_off', note=60, time=480),
                      mido.Message('note_on', note=62, velocity=64),
                      mido.Message('note_off', note=62, time=480),
                      mido.Message('note_on', note=64, velocity=64),
                      mido.Message('note_off', note=64)])
        midi.tracks.extend([tempo, notes]); midi.save(path)
        timeline = load_midi_score_timeline(path)
        self.assertEqual(timeline.quarters_at_seconds(.25), Fraction(1, 2))
        self.assertEqual(timeline.quarters_at_seconds(.5), 1)
        self.assertEqual(timeline.quarters_at_seconds(1.5), 2)
        self.assertEqual(timeline.notes[-1].onset_quarters, timeline.notes[-1].offset_quarters)
        for bad in (-1, float('nan'), 10):
            with self.assertRaises(ValueError):
                timeline.quarters_at_seconds(bad)
        notes.append(mido.Message('note_on', note=65, velocity=64)); midi.save(path)
        with self.assertRaisesRegex(ValueError, 'Unclosed'):
            load_midi_score_timeline(path)

    def test_overlapping_same_channel_pitch_does_not_claim_unique_duration(self):
        path = Path(self.temp.name) / 'overlap.mid'
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack([
            mido.Message('note_on', note=60, velocity=64),
            mido.Message('note_on', note=60, velocity=64, time=240),
            mido.Message('note_off', note=60, time=240),
            mido.Message('note_off', note=60, time=240)])
        midi.tracks.append(track); midi.save(path)
        timeline = load_midi_score_timeline(path)
        self.assertTrue(all(n.duration_ambiguous for n in timeline.notes))
        modified = replace(self.midi, notes=(replace(self.midi.notes[0], duration_ambiguous=True),)
                           + self.midi.notes[1:])
        link = self.bridge(midi=modified).note_links[0]
        self.assertEqual(link.status, 'verified_onset_duration_ambiguous')
        self.assertIsNone(link.midi_duration_quarters)


if __name__ == '__main__':
    unittest.main()
