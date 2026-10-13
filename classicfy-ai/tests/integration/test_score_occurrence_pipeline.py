import csv
from pathlib import Path
import unittest

from preprocessing import (ASAPLoader,load_match,load_musicxml,load_midi_score_timeline,
                           build_score_occurrence_bridge)

ROOT=Path(__file__).resolve().parents[3]


class ScoreOccurrencePipelineTest(unittest.TestCase):
    def test_fixed_six_performances_preserve_repeats_endings_and_original_intervals(self):
        work='Beethoven/Piano_Sonatas/31-2'
        asap=ROOT.parent/'datasets/ASAP';nasap=ROOT.parent/'datasets/nASAP'
        if not (nasap/work/'xml_score.musicxml').is_file() or not (asap/work/'midi_score.mid').is_file():
            self.skipTest('Beethoven 31-2 pilot XML/MIDI/match inputs are not installed')
        score=load_musicxml(nasap/work/'xml_score.musicxml')
        midi=load_midi_score_timeline(asap/work/'midi_score.mid')
        loader=ASAPLoader(asap,nasap)
        with (ROOT/'classicfy-ai/analysis/score_structure_inputs/performances.csv').open(encoding='utf-8-sig') as stream:
            keys=[r['key'] for r in csv.DictReader(stream) if r['piece']==work+'/midi_score.mid']
        self.assertEqual(len(keys),6)
        for key in keys:
            with self.subTest(key=key):
                sample=loader.get_sample(key)
                result=build_score_occurrence_bridge(score,load_match(sample.note_alignment_path),midi,
                    sample.score_beats,sample.performance_beats,sample.score_beat_types)
                self.assertEqual(len(result.bridge.intervals),459)
                self.assertEqual(len(result.visits),230)
                self.assertTrue(all(i.valid for i in result.bridge.intervals))
                self.assertEqual(sum(v.status=='rest_bracketed' for v in result.visits),13)
                self.assertEqual(sum(v.transition=='repeat_jump' for v in result.visits),3)
                self.assertEqual(sum(v.transition=='ending_skip' for v in result.visits),2)
                self.assertEqual([i for i,b in enumerate(result.interval_break_before) if b],[16,98,158,368,428])
                copies=[n for n in result.note_occurrences if n.written_xml_id=='n1']
                self.assertEqual(len(copies),2)
                self.assertEqual([n.visit_number for n in copies],[1,2])
                self.assertNotEqual(copies[0].note_id,copies[1].note_id)


if __name__=='__main__':
    unittest.main()
