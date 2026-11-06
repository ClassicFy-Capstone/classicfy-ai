"""Failure preservation, resume integrity and time-axis checks for batch alignment."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[3]/'src'/'preprocessing'))
sys.path.insert(0,str(HERE.parents[3]/'scripts'))
import youtube_sync as sync
import match_atepp_youtube as batch


class BatchSyncTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);source=root/'input';source.mkdir()
        midi=root/'source.mid';midi.write_bytes(b'fixture source')
        row=dict(perf_id='00505-2',composition_id='3',local_midi=str(midi),
                 resolved_midi=str(midi),midi_path='source.mid',sha256=sync.sha(midi),
                 youtube_links='https://www.youtube.com/watch?v=QlbiN3n-Rls',track_duration='200')
        segments=[dict(perf_id=row['perf_id'],candidate_number=str(i+1),start_seconds=a,
                       end_seconds=b,status=status,alignment_accepted='True')
                  for i,(a,b,status) in enumerate([('1','10','exported'),('20','40','exported_review'),
                                                   ('5','5','invalid_interval'),('50','55','excluded_duration')])]
        for name,rows in [('segments.csv',segments),('input_midi_audit.csv',[row])]:
            with (source/name).open('w',newline='',encoding='utf-8-sig') as stream:
                w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return source,row,segments

    def test_split_movement_id_and_upstream_review_are_retained(self):
        with tempfile.TemporaryDirectory() as td:
            source,row,segments=self.fixture(td)
            _,actual,metadata,groups=sync.load_inventory(source)
            self.assertEqual(list(groups),['00505-2'])
            self.assertEqual([r['status'] for r in groups[row['perf_id']]],['exported','exported_review'])
            self.assertEqual(len(actual),4)
            self.assertEqual(len(sync.load_inventory(source,include_excluded=True)[3][row['perf_id']]),3)

    def test_duplicate_candidates_fail_before_download(self):
        with tempfile.TemporaryDirectory() as td:
            source,_,segments=self.fixture(td)
            with (source/'segments.csv').open('a',newline='',encoding='utf-8') as stream:
                csv.DictWriter(stream,fieldnames=list(segments[0])).writerow(segments[0])
            with self.assertRaisesRegex(ValueError,'duplicate_candidate'):
                sync.load_inventory(source)

    def test_resume_fingerprint_detects_source_and_interval_changes(self):
        with tempfile.TemporaryDirectory() as td:
            _,row,segments=self.fixture(td)
            fingerprint=sync.request_fingerprint(row,segments,{'version':1})
            segments[0]['end_seconds']='11'
            self.assertNotEqual(fingerprint,sync.request_fingerprint(row,segments,{'version':1}))
            segments[0]['end_seconds']='10'
            Path(row['resolved_midi']).write_bytes(b'new source with different length')
            self.assertNotEqual(fingerprint,sync.request_fingerprint(row,segments,{'version':1}))

    def test_cached_mapping_and_clip_tampering_force_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            work=Path(td);(work/'mapping.npz').write_bytes(b'mapping')
            (work/'clip.wav').write_bytes(b'clip')
            result=dict(fingerprint='abc',status='completed',mapping_sha256=sync.sha(work/'mapping.npz'),
                rows=[dict(youtube_clip_relative_path='clip.wav',youtube_clip_sha256=sync.sha(work/'clip.wav'))])
            sync.atomic_json(work/'result.json',result)
            self.assertTrue(sync.reusable_result(work/'result.json','abc',True,work))
            (work/'clip.wav').write_bytes(b'altered clip')
            self.assertFalse(sync.reusable_result(work/'result.json','abc',True,work))
            self.assertTrue(sync.reusable_result(work/'result.json','abc'))
            (work/'mapping.npz').write_bytes(b'altered map')
            self.assertFalse(sync.reusable_result(work/'result.json','abc'))

    def test_failure_and_skipped_rows_never_disappear_or_get_fake_times(self):
        with tempfile.TemporaryDirectory() as td:
            source,row,segments=self.fixture(td)
            _,segments,metadata,groups=sync.load_inventory(source)
            out=Path(td)/'out';work=out/'jobs'/row['perf_id'];work.mkdir(parents=True)
            config={'save_clips':False};fingerprint=sync.request_fingerprint(metadata[row['perf_id']],groups[row['perf_id']],config)
            result=dict(fingerprint=fingerprint,status='video_unavailable',perf_id=row['perf_id'],
                rows=[sync.empty_match(r,'video_unavailable','deleted') for r in groups[row['perf_id']]])
            sync.atomic_json(work/'result.json',result)
            rows,counts,_=batch.gather(segments,metadata,groups,out,config)
            self.assertEqual(len(rows),4)
            self.assertEqual([r['youtube_match_status'] for r in rows],
                ['video_unavailable','video_unavailable','skipped_invalid_interval','skipped_excluded_input_status'])
            self.assertTrue(all(r['youtube_start_seconds']=='' for r in rows))
            self.assertEqual(counts['video_unavailable'],1)

    def test_worker_access_block_is_atomic_and_preserves_all_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            _,row,segments=self.fixture(td)
            work=Path(td)/'job';work.mkdir()
            request=dict(metadata=row,segments=segments[:2],options={'keep_audio':False},fingerprint='abc')
            sync.atomic_json(work/'request.json',request)
            with patch.object(sync,'acquire_audio',side_effect=RuntimeError('HTTP Error 429: Too Many Requests')), patch('traceback.print_exc'):
                self.assertEqual(sync.execute_job(work/'request.json'),2)
            result=sync.read_json(work/'result.json')
            self.assertEqual(result['status'],'access_blocked')
            self.assertEqual(len(result['rows']),2)
            self.assertTrue(all(r['youtube_end_seconds']=='' for r in result['rows']))

    def test_time_axis_aggregation_is_monotone_and_keeps_added_offset(self):
        import numpy as np
        path=np.array([[100,101,102,103,104],[0,0,0,1,2]])
        mx,my=sync.path_mapping(path)
        np.testing.assert_allclose(mx,[0,.02,.04])
        np.testing.assert_allclose(my,[2.02,2.06,2.08])
        with self.assertRaisesRegex(ValueError,'nonmonotonic'):
            sync.path_mapping(path[:,::-1])

    def test_url_and_nan_validation(self):
        self.assertEqual(sync.video_id('https://youtu.be/QlbiN3n-Rls?t=22'),'QlbiN3n-Rls')
        with self.assertRaises(ValueError):sync.video_id('https://example.com/watch?v=QlbiN3n-Rls')
        for start,end in [('NaN','10'),('1','inf'),('-1','10'),('20','10')]:
            self.assertFalse(sync.valid_interval(dict(start_seconds=start,end_seconds=end)))

    def test_runtime_fit_accounts_for_per_file_overhead(self):
        intercept,rate=batch.nonnegative_line([(100,12),(200,22),(600,62)])
        self.assertAlmostEqual(intercept,2)
        self.assertAlmostEqual(rate,.1)
        with self.assertRaises(ValueError):batch.nonnegative_line([])


if __name__=='__main__':unittest.main()
