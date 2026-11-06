"""Synthetic musical structure and evaluation isolation, independent of ATEPP labels."""
import sys
from pathlib import Path
import unittest
import tempfile
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from preprocessing.theme_candidates import propose, interval_iou, consensus, diagonal_prefix
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from run_atepp_theme_candidates import work_split, tune_ranking
from finalize_atepp_theme_candidates import main as finalize


class ThemeCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):threadpool_limits(1)

    def test_repeated_transposed_melody_has_a_matching_first_occurrence_candidate(self):
        rng=np.random.default_rng(91)
        melody=rng.integers(60,80,48)
        middle=rng.integers(48,90,48)
        notes=[]
        for block,pitches in enumerate((melody,middle,melody+5)):
            for j,p in enumerate(pitches):
                t=block*24+j*.5
                notes.extend([[p,t,t+.3,80],[48+(j//8)%5+(5 if block==2 else 0),t,t+.4,55]])
        candidates=propose(notes)
        self.assertTrue(any(interval_iou(c['start_seconds'],c['end_seconds'],0,24)>.5 for c in candidates))
        self.assertTrue(any(c['recurrence']>.75 for c in candidates))

    def test_static_ostinato_does_not_get_high_theme_confidence(self):
        notes=[[p,float(t),float(t)+.4,70] for t in range(90) for p in (48,60,64,67)]
        candidates=propose(notes)
        self.assertLess(max(c['heuristic_confidence'] for c in candidates),.5)

    def test_few_notes_fail_explicitly_instead_of_fabricating_theme(self):
        with self.assertRaisesRegex(ValueError,'too_few_valid_notes'):
            propose([[60,0,1,80]])

    def test_diagonal_window_score_matches_direct_sequence_comparison(self):
        rng=np.random.default_rng(8);s=rng.random((20,20)).astype(np.float32)
        p=diagonal_prefix(s)
        self.assertAlmostEqual(float(p[9,14]-p[3,8]),float(sum(s[3+k,8+k] for k in range(6))),places=5)

    def test_consensus_does_not_read_annotation_fields(self):
        candidates=[dict(ranking_score=.7,relative_start=0.,relative_duration=.1,fingerprint=[1.,0.]),
                    dict(ranking_score=.65,relative_start=.3,relative_duration=.1,fingerprint=[0.,1.])]
        records=[dict(perf_id='a',candidates=candidates),dict(perf_id='b',candidates=candidates)]
        before=consensus(records)
        for record in records:record['gold_start']=999.;record['label']='incorrect'
        self.assertEqual(before,consensus(records))

    def test_work_split_keeps_inspected_work_out_of_test(self):
        primary=pd.DataFrame({'composition_id':[str(i) for i in range(17)]*2})
        audit=pd.DataFrame({'perf_id':['05943'],'composition_id':['13']})
        split=work_split(primary,audit)
        self.assertEqual(split['13'],'development')
        self.assertEqual(len(split),17)
        self.assertEqual(sum(s=='development' for s in split.values()),5)

    def test_changing_test_labels_cannot_change_ranking_calibration(self):
        ids=['05943','test1','test2','test3'];works=['13','1','2','3']
        audit=pd.DataFrame({'perf_id':ids,'composition_id':works})
        gold=pd.DataFrame({'perf_id':ids,'composition_id':works,'start_seconds':[0.]*4,'end_seconds':[30.]*4})
        candidates=[dict(start_seconds=0.,end_seconds=60.,first_note_seconds=0.,ranking_score=.8,
                         relative_start=0.,relative_duration=.2,fingerprint=[1.,0.]),
                    dict(start_seconds=0.,end_seconds=30.,first_note_seconds=0.,ranking_score=.7,
                         relative_start=0.,relative_duration=.1,fingerprint=[1.,0.])]
        predictions={pid:dict(perf_id=pid,candidates=candidates) for pid in ids}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            before=tune_ranking(predictions,audit,gold,root)
            weights_before=json.loads((root/'ranking_calibration.json').read_text())['selected']
            gold.loc[gold.perf_id!='05943','end_seconds']=9999.
            after=tune_ranking(predictions,audit,gold,root)
            weights_after=json.loads((root/'ranking_calibration.json').read_text())['selected']
            self.assertEqual(weights_before,weights_after)
            self.assertEqual(before,after)

    def test_out_of_domain_flag_moves_only_generated_clip_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'export_checks').mkdir()
            records=[]
            for pid,work,composer,track in [('00001','1','Chopin','Nocturne'),
                                           ('00002','2','Ludwig van Beethoven','Piano Sonata: I. Allegro')]:
                path=root/'midi'/'medium'/work/f'{pid}.mid';path.parent.mkdir(parents=True)
                path.write_bytes(b'unchanged-test-clip')
                records.append(dict(perf_id=pid,composition_id=work,composer=composer,track=track,status='exported',
                                    method='label_free_midi',clip_path=str(path),confidence_level='medium',
                                    low_confidence=False,review_reasons=''))
                (root/'export_checks'/f'{pid}.json').write_text('{}')
            pd.DataFrame(records).to_csv(root/'segments.csv',index=False)
            (root/'summary.json').write_text('{}')
            finalize(root)
            result=pd.read_csv(root/'segments.csv',dtype={'perf_id':str}).set_index('perf_id')
            self.assertEqual(result.loc['00001','confidence_level'],'low')
            self.assertEqual(result.loc['00002','confidence_level'],'medium')
            self.assertEqual(Path(result.loc['00001','clip_path']).read_bytes(),b'unchanged-test-clip')
            self.assertFalse((root/'midi/medium/1/00001.mid').exists())


if __name__=='__main__':unittest.main()
