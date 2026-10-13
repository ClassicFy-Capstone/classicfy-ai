"""Three fixed score pilots; no performance feature values are consulted."""

import csv
from fractions import Fraction
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'classicfy-ai/scripts'))
from extract_phrase_candidates import map_boundary
from preprocessing import ASAPLoader,load_musicxml,load_match,load_midi_score_timeline
from preprocessing.phrase_candidates import extract_phrase_candidates
from preprocessing.score_measure_bridge import build_score_measure_bridge


class PhraseCandidatePipelineTest(unittest.TestCase):
    def score(self,work):
        p=ROOT.parent/'datasets/nASAP'/work/'xml_score.musicxml'
        if not p.exists():self.skipTest('Frozen pilot XML is not installed')
        return load_musicxml(p)

    def test_bach_ornament_slur_is_preserved_but_not_a_phrase_span(self):
        score=self.score('Bach/Fugue/bwv_848');result=extract_phrase_candidates(score)
        self.assertEqual(len(result.slurs),1)
        self.assertEqual(result.slurs[0].status,'nonpositive_onset_span')
        self.assertFalse(any(e.kind=='slur_end' for c in result.candidates for e in c.cues))
        notes={n.note_id:n for n in score.notes}
        self.assertTrue(notes[result.slurs[0].start_note_id].is_grace)

    def test_beethoven_boundary_keeps_repeated_visits_and_original_indices(self):
        work='Beethoven/Piano_Sonatas/31-2';score=self.score(work)
        result=extract_phrase_candidates(score)
        candidate=next(c for c in result.candidates if c.position_quarters==4 and
                       any(e.kind=='slur_end' for e in c.cues))
        asap=ROOT.parent/'datasets/ASAP';nasap=ROOT.parent/'datasets/nASAP'
        if not (asap/'metadata.csv').exists():self.skipTest('Frozen ASAP MIDI/match inputs are not installed')
        loader=ASAPLoader(asap,nasap)
        with (ROOT/'classicfy-ai/analysis/score_structure_inputs/performances.csv').open(encoding='utf-8-sig') as f:
            row=next(r for r in csv.DictReader(f) if r['piece']==work+'/midi_score.mid')
        sample=loader.get_sample(row['key']);midi=load_midi_score_timeline(sample.score_path)
        import json
        annotations=json.loads((asap/'asap_annotations.json').read_text())
        bridge=build_score_measure_bridge(score,load_match(sample.note_alignment_path),midi,
            sample.score_beats,sample.performance_beats,sample.score_beat_types,
            sample.score_downbeats,annotations[row['key']]['downbeats_score_map'])
        mapped=map_boundary(candidate,score,bridge.parts[0],midi.ppq)
        self.assertEqual(len(mapped),2)
        self.assertEqual([r['visit_number'] for r in mapped],[1,2])
        self.assertEqual([r['original_edge_index'] for r in mapped],[4,20])
        self.assertTrue(all(not r['eligible_for_embedding'] for r in mapped))
        self.assertTrue(all(r['boundary_status']=='pending_score_review' for r in mapped))

    def test_mozart_arrival_and_release_stay_in_the_uncertainty_range(self):
        result=extract_phrase_candidates(self.score('Mozart/Piano_Sonatas/8-1'))
        candidate=next(c for c in result.candidates if any(e.kind=='harmonic_closure_hint' for e in c.cues))
        self.assertEqual(candidate.position_quarters,Fraction(65,2))
        self.assertEqual(candidate.uncertainty_start_quarters,32)
        self.assertEqual(candidate.uncertainty_stop_quarters,33)
        cue=next(e for e in candidate.cues if e.kind=='harmonic_closure_hint')
        self.assertEqual(cue.details['mode_source'],'unconfirmed_key_signature_hypothesis')


if __name__=='__main__':unittest.main()
