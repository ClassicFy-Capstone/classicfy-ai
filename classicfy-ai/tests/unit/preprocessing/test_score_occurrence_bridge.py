import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from preprocessing import NoteAlignment, ScoreNote, load_musicxml
from preprocessing.score_beat_bridge import MidiScoreNote, MidiScoreTimeline
from preprocessing.score_occurrence_bridge import build_score_occurrence_bridge


def note(identifier,pitch='C',extra=''):
    return f'<note id="{identifier}"><pitch><step>{pitch}</step><octave>4</octave></pitch><duration>1</duration>{extra}</note>'


class ScoreOccurrenceBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def score(self,bodies):
        path=Path(self.temp.name)/'score.musicxml'
        parts=''.join(f'<measure number="{i+1}"><attributes><divisions>1</divisions></attributes>{b}</measure>'
                      for i,b in enumerate(bodies))
        path.write_text(f'<score-partwise><part id="P1">{parts}</part></score-partwise>')
        return load_musicxml(path)

    def bridge(self,score,refs,beats=None,midi=None):
        if midi is None:
            midi=MidiScoreTimeline(480,tuple(MidiScoreNote(s.pitch,Fraction(str(s.onset_beats)),
                Fraction(str(s.offset_beats))-Fraction(1,480),0,0) for s in refs),
                ((0,Fraction(0),500000),),int(max(s.offset_beats for s in refs)*480))
        if beats is None:
            beats=[i*.5 for i in range(int(max(s.offset_beats for s in refs))+1)]
        return build_score_occurrence_bridge(score,NoteAlignment((),tuple(refs),()),midi,
            beats,[i*.7 for i in range(len(beats))],['b']*len(beats))

    def test_same_note_occurrences_keep_ids_order_written_position_and_break(self):
        score=self.score(['<barline location="left"><repeat direction="forward"/></barline>'+note('n0')+
            note('n1','D')+'<barline><repeat direction="backward"/></barline>',note('n2','E')])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,()),
              ScoreNote('n0-2',60,2,3,()),ScoreNote('n1-2',62,3,4,()),ScoreNote('n2-1',64,4,5,())]
        result=self.bridge(score,refs)
        self.assertEqual([v.measure_index for v in result.visits],[0,0,1])
        self.assertEqual([v.visit_number for v in result.visits],[1,2,1])
        self.assertEqual([v.transition for v in result.visits],['start','repeat_jump','consecutive'])
        repeated=[n for n in result.note_occurrences if n.written_xml_id=='n0']
        self.assertEqual(len(repeated),2)
        self.assertNotEqual(repeated[0].note_id,repeated[1].note_id)
        self.assertEqual([n.written_onset_quarters for n in repeated],[0,0])
        self.assertEqual(result.interval_written_quarters[0],result.interval_written_quarters[2])
        self.assertEqual(result.interval_break_before,(False,False,True,False,False))
        self.assertEqual([i.index for i in result.bridge.intervals],list(range(5)))
        self.assertEqual(result.adjacent_pair_mask([True]*5),(False,True,False,True,True))
        self.assertEqual(result.adjacent_pair_mask([True,False,True,True,True]),(False,False,False,True,True))
        with self.assertRaises(ValueError):result.adjacent_pair_mask([True])

    def test_rest_measure_is_inferred_only_from_consistent_neighbors(self):
        score=self.score([note('n0'),'<note id="r1"><rest/><duration>1</duration></note>',note('n2','E')])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n2-1',64,2,3,())]
        result=self.bridge(score,refs)
        self.assertEqual(result.visits[1].status,'rest_bracketed')
        self.assertEqual(result.visits[1].anchor_count,0)
        self.assertTrue(result.bridge.intervals[1].valid)
        bad=[refs[0],replace(refs[1],onset_beats=3,offset_beats=4)]
        result=self.bridge(score,bad)
        self.assertEqual(len(result.visits),2)
        self.assertEqual(result.visits[-1].transition,'unresolved_gap')
        self.assertIsNone(result.interval_written_quarters[1])
        self.assertFalse(result.bridge.intervals[1].valid)

    def test_first_second_ending_skip_is_explicit_and_does_not_create_a_visit(self):
        score=self.score(['<barline location="left"><repeat direction="forward"/></barline>'+note('n0'),
            '<barline location="left"><ending number="1" type="start"/></barline>'+note('n1','D')+
            '<barline><ending number="1" type="stop"/><repeat direction="backward"/></barline>',
            '<barline location="left"><ending number="2" type="start"/></barline>'+note('n2','E')+
            '<barline><ending number="2" type="stop"/></barline>'])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,()),
              ScoreNote('n0-2',60,2,3,()),ScoreNote('n2-1',64,3,4,())]
        result=self.bridge(score,refs)
        self.assertEqual([v.measure_index for v in result.visits],[0,1,0,2])
        self.assertEqual(result.visits[-1].transition,'ending_skip')
        self.assertEqual(result.visits[-1].visit_number,1)
        self.assertEqual(result.interval_break_before,(False,False,True,True))

    def test_source_suffix_is_a_label_not_an_assumed_visit_number(self):
        score=self.score(['<barline location="left"><repeat direction="forward"/></barline>'+note('n0')+
            note('n1','D')+'<barline><repeat direction="backward"/></barline>'])
        refs=[ScoreNote('n0-7',60,0,1,()),ScoreNote('n1-7',62,1,2,()),
              ScoreNote('n0-9',60,2,3,()),ScoreNote('n1-9',62,3,4,())]
        result=self.bridge(score,refs)
        self.assertEqual([v.visit_number for v in result.visits],[1,2])
        self.assertEqual([v.source_occurrence_label for v in result.visits],['7','9'])

    def test_unknown_jump_is_marked_and_not_silently_verified(self):
        score=self.score([note('n0'),note('n1','D')])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,()),
              ScoreNote('n0-2',60,2,3,())]
        result=self.bridge(score,refs)
        self.assertEqual(result.visits[-1].transition,'unresolved_jump')
        self.assertFalse(result.bridge.intervals[-1].valid)
        self.assertTrue(result.interval_break_before[-1])

    def test_beat_crossing_repeat_boundary_is_masked_and_zero_width_is_preserved(self):
        score=self.score(['<barline location="left"><repeat direction="forward"/></barline>'+note('n0')+
            note('n1','D')+'<barline><repeat direction="backward"/></barline>'])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,()),
              ScoreNote('n0-2',60,2,3,()),ScoreNote('n1-2',62,3,4,())]
        result=self.bridge(score,refs,beats=[0,.75,1.25,2])
        self.assertEqual(result.bridge.intervals[1].status,'crosses_repeat_boundary')
        self.assertFalse(result.bridge.intervals[1].valid)
        result=self.bridge(score,refs,beats=[0,.5,.5,1,1.5,2])
        self.assertEqual(result.bridge.intervals[1].status,'zero_width')
        self.assertEqual(len(result.bridge.intervals),5)

    def test_duplicate_nonuniform_and_unknown_id_fail_explicitly(self):
        score=self.score([note('n0')+note('n1','D')])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,())]
        for invalid in (refs+[refs[0]],[refs[0],replace(refs[1],onset_beats=1.5)],
                        [refs[0],replace(refs[1],note_id='unknown-1')]):
            with self.subTest(refs=invalid),self.assertRaises(ValueError):
                self.bridge(score,invalid)


if __name__=='__main__':
    unittest.main()
