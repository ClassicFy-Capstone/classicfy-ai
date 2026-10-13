"""Verify the fixed BWV 848 pilot through the public loader/bridge interfaces."""

import csv
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from preprocessing import (
    ASAPLoader, load_match, load_midi, load_musicxml,
    build_score_beat_bridge, load_midi_score_timeline,
)
from features import extract_dynamics

ROOT=Path(__file__).resolve().parents[3]


class ScoreStructurePipelineTest(unittest.TestCase):
    def test_fixed_nine_performances_keep_original_beats_and_source_features(self):
        asap=ROOT.parent/'datasets/ASAP'
        nasap=ROOT.parent/'datasets/nASAP'
        work='Bach/Fugue/bwv_848'
        xml=nasap/work/'xml_score.musicxml'
        original=asap/work/'xml_score.musicxml'
        if not all(p.is_file() for p in (xml,original,asap/'metadata.csv',nasap/'metadata.csv')):
            self.skipTest('BWV 848 XML and aligned MIDI/match datasets are not installed')
        score=load_musicxml(xml)
        self.assertEqual(tuple(replace(n,xml_id=None) for n in score.notes),load_musicxml(original).notes)
        midi=load_midi_score_timeline(asap/work/'midi_score.mid')
        loader=ASAPLoader(asap,nasap)
        feature_path=ROOT/'classicfy-ai/analysis/dynamics/beat_features.csv'
        with feature_path.open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        keys=sorted({r['key'] for r in rows})
        self.assertEqual(len(keys),9)
        for key in keys:
            with self.subTest(key=key):
                sample=loader.get_sample(key)
                bridge=build_score_beat_bridge(score,load_match(sample.note_alignment_path),midi,
                    sample.score_beats,sample.performance_beats,sample.score_beat_types)
                self.assertEqual(len(bridge.intervals),len(sample.score_beats)-1)
                self.assertEqual([i.index for i in bridge.intervals],list(range(216)))
                self.assertTrue(all(i.valid for i in bridge.intervals))
                self.assertGreaterEqual(min(i.verified_onsets for i in bridge.intervals),2)
                self.assertEqual(bridge.xml_to_score_offset_quarters,0)
                self.assertTrue(any(l.status=='missing_midi' for l in bridge.note_links))
                feature=extract_dynamics(load_midi(sample.performance_path),sample.performance_beats)
                saved=sorted((r for r in rows if r['key']==key),key=lambda r:int(r['beat_index']))
                np.testing.assert_allclose(feature.sequence.values,[float(r['raw']) for r in saved],rtol=0,atol=1e-12)
                np.testing.assert_array_equal(feature.sequence.mask,[r['raw_mask']=='True' for r in saved])
                selected=bridge.select_intervals(np.flatnonzero(feature.sequence.mask))
                self.assertEqual([i.index for i in selected],[int(r['beat_index']) for r in saved if r['raw_mask']=='True'])


if __name__=='__main__':
    unittest.main()
