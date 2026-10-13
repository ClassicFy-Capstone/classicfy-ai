import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from preprocessing import load_musicxml,NoteAlignment,ScoreNote
from preprocessing.score_beat_bridge import MidiScoreNote,MidiScoreTimeline
from preprocessing.score_measure_bridge import build_score_measure_bridge


def note(identifier,step='C'):
    return f'<note id="{identifier}"><pitch><step>{step}</step><octave>4</octave></pitch><duration>1</duration></note>'


class ScoreMeasureBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def score(self,bodies):
        path=Path(self.temp.name)/'score.musicxml'
        text=''.join(f'<measure number="{i+1}"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>{b}</measure>' for i,b in enumerate(bodies))
        path.write_text(f'<score-partwise><part id="P1">{text}</part></score-partwise>')
        return load_musicxml(path)

    def bridge(self,score,refs,midi_refs=None,downbeats=None,mapping=None,performance=None,beats=None):
        actual=refs if midi_refs is None else midi_refs
        end=max(s.offset_beats for s in actual)
        midi=MidiScoreTimeline(480,tuple(MidiScoreNote(n.pitch,Fraction(str(n.onset_beats)),
            Fraction(str(n.offset_beats))-Fraction(1,480),0,0) for n in actual),
            ((0,Fraction(0),500000),),round(end*480))
        if downbeats is None:downbeats=[float(m.onset_quarters)*.5 for m in score.measures]
        if mapping is None:mapping=list(range(len(score.measures)))
        if beats is None:beats=[i*.5 for i in range(int(end)+1)]
        if performance is None:performance=[i*.7 for i in range(len(beats))]
        return build_score_measure_bridge(score,NoteAlignment((),tuple(refs),()),midi,beats,performance,
                                          ['b']*len(beats),downbeats,mapping)

    def test_piecewise_match_units_preserve_original_midi_and_xml_coordinates(self):
        score=self.score([note('n0')+note('n1','D'),note('n2','E')+note('n3','F')])
        native=[ScoreNote(f'n{i}-1',p,i,i+1,()) for i,p in enumerate((60,62,64,65))]
        raw=([replace(n,onset_beats=2*n.onset_beats,offset_beats=2*n.offset_beats) for n in native[:2]]+
             [replace(n,onset_beats=n.onset_beats+2,offset_beats=n.offset_beats+2) for n in native[2:]])
        result=self.bridge(score,raw,native)
        self.assertEqual([r['match_units_per_xml_quarter'] for r in result.match_coordinate_checks],[2,1])
        part=result.parts[0][1]
        self.assertTrue(all(i.valid for i in part.bridge.intervals))
        self.assertEqual([l.score_onset_quarters for l in part.bridge.note_links],[0,1,2,3])
        self.assertEqual(part.bridge.note_links[1].match_onset_quarters,1)
        self.assertIsNone(part.bridge.note_links[1].match_duration_quarters)

    def test_combined_downbeat_range_splits_only_with_exact_duration_bracket(self):
        score=self.score([note('n0'),note('n1','D'),note('n2','E')])
        refs=[ScoreNote(f'n{i}-1',p,i,i+1,()) for i,p in enumerate((60,62,64))]
        result=self.bridge(score,refs,downbeats=[0,1],mapping=['0-1',2])
        self.assertEqual(result.inferred_downbeat_indices,(1,))
        self.assertEqual([v.score_start_quarters for v in result.parts[0][1].visits],[0,1,2])
        with self.assertRaisesRegex(ValueError,'duration bracket'):
            self.bridge(score,refs,downbeats=[0,.75],mapping=['0-1',2])

    def test_missing_pickup_downbeat_keeps_observed_repeat_visits_separate(self):
        score=self.score([note('n0'),'<barline location="left"><repeat direction="forward"/></barline>'+note('n1','D')+
            '<barline><repeat direction="backward"/></barline>'])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,()),ScoreNote('n1-2',62,2,3,())]
        result=self.bridge(score,refs,downbeats=[.5,1],mapping=float('nan'))
        self.assertEqual(result.inferred_downbeat_indices,(0,))
        part=result.parts[0][1]
        self.assertEqual([v.visit_number for v in part.visits],[1,1,2])
        self.assertEqual(part.interval_break_before,(False,False,True))
        self.assertEqual(part.adjacent_pair_mask([True]*3),(False,True,False))

    def test_local_coordinate_conflict_is_masked_without_dropping_original_indices(self):
        score=self.score([note('n0')+note('n1','D'),note('n2','E')+note('n3','F')])
        native=[ScoreNote(f'n{i}-1',p,i,i+1,()) for i,p in enumerate((60,62,64,65))]
        raw=[replace(native[0],onset_beats=.00625)]+native[1:]
        result=self.bridge(score,raw,native);part=result.parts[0][1]
        self.assertEqual(part.visits[0].status,'coordinate_conflict')
        self.assertEqual([i.index for i in part.bridge.intervals],list(range(4)))
        self.assertEqual([i.valid for i in part.bridge.intervals],[False,False,True,True])

    def test_wrong_measure_schedule_fails_midi_support_instead_of_shifting_notes(self):
        score=self.score([note('n0')+note('n1','D'),note('n2','E')+note('n3','F')])
        refs=[ScoreNote(f'n{i}-1',p,i,i+1,()) for i,p in enumerate((60,62,64,65))]
        with self.assertRaisesRegex(ValueError,'MIDI anchor support'):
            self.bridge(score,refs,downbeats=[.5,1.5])

    def test_zero_performance_interval_is_preserved_and_excluded_from_pairs(self):
        score=self.score([note('n0')+note('n1','D')+note('n2','E')])
        refs=[ScoreNote(f'n{i}-1',p,i,i+1,()) for i,p in enumerate((60,62,64))]
        part=self.bridge(score,refs,performance=[0,.7,.7,2.1]).parts[0][1]
        self.assertEqual([i.valid for i in part.bridge.intervals],[True,False,True])
        self.assertEqual(part.adjacent_pair_mask([True]*3),(False,False,False))

    def test_unknown_or_duplicate_match_ids_fail_explicitly(self):
        score=self.score([note('n0')+note('n1','D')])
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,())]
        for raw in (refs+[refs[0]],[refs[0],replace(refs[1],note_id='unknown-1')]):
            with self.subTest(raw=raw),self.assertRaises(ValueError):self.bridge(score,raw,refs)

    def test_two_parts_keep_separate_written_ids_on_one_original_grid(self):
        score=self.score([note('n0')+note('n1','D')])
        path=score.source_path;text=path.read_text()
        part=text.split('<score-partwise>')[1].split('</score-partwise>')[0]
        second=part.replace('P1','P2').replace('n0','n2').replace('n1','n3').replace('<step>C','<step>E').replace('<step>D','<step>F')
        path.write_text('<score-partwise>'+part+second+'</score-partwise>');score=load_musicxml(path)
        refs=[ScoreNote('n0-1',60,0,1,()),ScoreNote('n1-1',62,1,2,()),
              ScoreNote('n2-1',64,0,1,()),ScoreNote('n3-1',65,1,2,())]
        result=self.bridge(score,refs,downbeats=[0],mapping=[0])
        self.assertEqual([pid for pid,part in result.parts],['P1','P2'])
        self.assertTrue(all(i.valid for pid,part in result.parts for i in part.bridge.intervals))
        self.assertNotEqual(result.parts[0][1].note_occurrences[0].written_note_id,
                            result.parts[1][1].note_occurrences[0].written_note_id)

    def test_single_onset_does_not_claim_a_verified_match_unit(self):
        score=self.score([note('n0'),note('n1','D')])
        refs=[ScoreNote('n0-1',60,0,2,()),ScoreNote('n1-1',62,2,4,())]
        native=[replace(n,onset_beats=i,offset_beats=i+1) for i,n in enumerate(refs)]
        result=self.bridge(score,refs,native)
        self.assertEqual([r['status'] for r in result.match_coordinate_checks],['local_match_origin_only']*2)
        self.assertEqual([r['match_units_per_xml_quarter'] for r in result.match_coordinate_checks],[None,None])


if __name__=='__main__':unittest.main()
