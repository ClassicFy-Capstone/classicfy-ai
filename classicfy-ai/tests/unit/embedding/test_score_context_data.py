import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from embedding.score_context_data import (NATIVE_CHANNELS,ScoreContextSegment,
    collate_score_context,export_score_context,read_score_context_segments)


def score_midi(path, velocity=64):
    midi=mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.Message('note_on',note=60,velocity=velocity,time=0),
        mido.Message('note_off',note=60,velocity=0,time=480),
        mido.Message('note_on',note=64,velocity=velocity,time=480),
        mido.Message('note_off',note=64,velocity=0,time=480)]))
    midi.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScoreContextDataTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.score=self.root/'score.mid';self.sha=score_midi(self.score)

    def mapping(self,keys=('00001','00002'),splits=None,works=None):
        path=self.root/'score-map.csv'
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['key','work','score_path','score_sha256','split']);writer.writeheader()
            for i,key in enumerate(keys):writer.writerow(dict(key=key,work=works[i] if works else 'work',
                score_path='score.mid',score_sha256=self.sha,split=splits[i] if splits else 'unassigned'))
        return path

    def quarter(self,keys=('00001','00002'),q=None,invalid=False,all_missing=False):
        q=np.arange(5.) if q is None else q
        arrays={};records=[]
        for i,key in enumerate(keys):
            records.append(dict(key=key,channels=['tempo','dynamics'],group='score'))
            values=np.arange(2*(len(q)-1),dtype=np.float32).reshape(2,-1)+i*.25
            mask=np.ones_like(values,bool);mask[:,1]=False
            if all_missing:mask[:]=False
            values[:,1]=np.nan
            if invalid:values[0,0]=np.nan
            arrays[f'{i}/q']=q;arrays[f'{i}/values']=values;arrays[f'{i}/valid']=mask
        arrays['manifest']=np.array(json.dumps(records));path=self.root/'quarter.npz';np.savez(path,**arrays);return path

    def native(self,bad_grid=False,bad_hash=False):
        arrays={};records=[]
        for i,channel in enumerate(NATIVE_CHANNELS):
            records.append(dict(key='00001',piece='ATEPP/work/hash',channel=channel))
            arrays[f'{i}/score_beats']=np.arange(5.)+(.5 if bad_grid and i==1 else 0)
            arrays[f'{i}/standardized_values']=np.arange(4.)+i
            arrays[f'{i}/standardized_mask']=np.array([True,False,True,True])
        arrays['manifest']=np.array(json.dumps(dict(records=records,provenance=dict(dataset='ATEPP-1.2',aligner='dual_dtw',source_files=[dict(key='00001',score_sha256='0'*64 if bad_hash else self.sha)]))))
        path=self.root/'atepp_standardized.npz';np.savez(path,**arrays);return path

    def test_portable_join_deduplicates_score_preserves_q_and_missing(self):
        source=self.quarter();out=self.root/'out'
        result=export_score_context(source,self.mapping(),out,dataset='ATEPP')
        self.assertEqual(len(result['scores']),1);self.assertEqual(len(result['records']),2)
        self.assertEqual(result['records'][0]['key'],'00001')
        self.assertEqual(result['records'][0]['original_intervals'],4)
        self.assertEqual(result['records'][0]['score_intervals'],3)
        self.assertEqual(result['records'][0]['usable_intervals'],2)
        manifest,recordings=read_score_context_segments(out,max_segment_length=2)
        self.assertEqual([len(s.score) for s in recordings[0]],[2,1])
        np.testing.assert_array_equal(recordings[0][0].q,[0,1,2])
        self.assertFalse(recordings[0][0].observation_mask[1].any())
        batch=collate_score_context(recordings)
        self.assertEqual(batch['score'].shape,(2,2,2,30));self.assertEqual(batch['performance'].shape,(2,2,2,2))
        self.assertEqual(batch['keys'],['00001','00002']);self.assertFalse(batch['time_mask'][0,1,1])
        self.assertFalse(batch['observation_mask'][0,0,1].any())
        self.assertEqual(batch['q_start'][0,1,0],2)

    def test_native_atepp_standardized_cache_is_read_without_refitting(self):
        source=self.native();out=self.root/'out'
        result=export_score_context(source,self.mapping(('00001',)),out,dataset='ATEPP')
        self.assertEqual(result['performance_channels'],list(NATIVE_CHANNELS))
        with np.load(out/'dataset.npz',allow_pickle=False) as z:
            values=z['records/0/performance'];mask=z['records/0/observation_mask']
            np.testing.assert_array_equal(values[0],np.arange(7.))
            self.assertFalse(mask[1].any());np.testing.assert_array_equal(values[1],0)

    def test_alignment_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'alignment provenance'):
            export_score_context(self.native(bad_hash=True),self.mapping(('00001',)),self.root/'out',dataset='ATEPP')
        self.assertFalse((self.root/'out').exists())

    def test_channel_grid_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'grids'):
            export_score_context(self.native(bad_grid=True),self.mapping(('00001',)),self.root/'out',dataset='ATEPP')

    def test_work_or_identical_score_cannot_cross_split(self):
        source=self.quarter()
        with self.assertRaisesRegex(ValueError,'crosses split'):
            export_score_context(source,self.mapping(splits=['train','test']),self.root/'out',dataset='ATEPP',fit_scale=True)
        with self.assertRaisesRegex(ValueError,'crosses split'):
            export_score_context(source,self.mapping(splits=['train','test'],works=['one','two']),self.root/'out2',dataset='ATEPP')

    def test_fit_requires_explicit_train_and_repeated_performances_share_scale(self):
        source=self.quarter()
        with self.assertRaisesRegex(ValueError,'training'):
            export_score_context(source,self.mapping(),self.root/'missing',dataset='ATEPP',fit_scale=True)
        result=export_score_context(source,self.mapping(splits=['train','train']),self.root/'out',dataset='ATEPP',fit_scale=True)
        self.assertEqual(result['score_scale']['training_score_hashes'],[self.sha])
        loaded=export_score_context(source,self.mapping(),self.root/'loaded',dataset='ATEPP',scale_path=self.root/'out/score_scale.json')
        self.assertEqual(result['score_scale'],loaded['score_scale'])
        with self.assertRaisesRegex(ValueError,'evaluation score'):
            export_score_context(source,self.mapping(splits=['test','test']),self.root/'badscale',dataset='ATEPP',scale_path=self.root/'out/score_scale.json')

    def test_score_map_identity_and_all_keys_are_checked(self):
        source=self.quarter()
        with self.assertRaisesRegex(ValueError,'absent'):
            export_score_context(source,self.mapping(('00001',)),self.root/'out',dataset='ATEPP')
        map_path=self.mapping();map_path.write_text(map_path.read_text().replace(self.sha,'0'*64))
        with self.assertRaisesRegex(ValueError,'hash-mismatched'):
            export_score_context(source,map_path,self.root/'out2',dataset='ATEPP')

    def test_nonfinite_observation_and_nonquarter_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite valid'):
            export_score_context(self.quarter(invalid=True),self.mapping(),self.root/'bad',dataset='ATEPP')
        with self.assertRaisesRegex(ValueError,'one-quarter'):
            export_score_context(self.quarter(q=np.arange(5.)*2),self.mapping(),self.root/'wide',dataset='ATEPP')

    def test_export_does_not_overwrite_and_integrity_detects_tampering(self):
        source=self.quarter();mapping=self.mapping();out=self.root/'out'
        export_score_context(source,mapping,out,dataset='ATEPP')
        with self.assertRaisesRegex(ValueError,'empty'):export_score_context(source,mapping,out,dataset='ATEPP')
        (out/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'integrity'):read_score_context_segments(out,max_segment_length=4)

    def test_duplicate_source_provenance_is_rejected(self):
        source=self.native()
        with np.load(source,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        data=json.loads(str(arrays['manifest']));data['provenance']['source_files']*=2
        arrays['manifest']=np.array(json.dumps(data));np.savez(source,**arrays)
        with self.assertRaisesRegex(ValueError,'Duplicate recording in source provenance'):
            export_score_context(source,self.mapping(('00001',)),self.root/'out',dataset='ATEPP')

    def test_external_scale_integrity_is_checked(self):
        out=self.root/'out'
        export_score_context(self.quarter(),self.mapping(splits=['train','train']),out,dataset='ATEPP',fit_scale=True)
        (out/'score_scale.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'scale integrity'):read_score_context_segments(out,max_segment_length=4)

    def test_subset_score_coverage_keeps_a_missing_interval_as_a_segment_cut(self):
        # Explicit coverage holes are not compacted by the exported reader.
        source=self.quarter();out=self.root/'out';export_score_context(source,self.mapping(),out,dataset='ATEPP')
        with np.load(out/'dataset.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        manifest=json.loads(str(arrays['manifest']));ref=manifest['records'][0]['score_ref']
        arrays[f'scores/{ref}/valid'][1]=False
        np.savez_compressed(out/'dataset.npz',**arrays)
        integrity=json.loads((out/'integrity.json').read_text());integrity['dataset_sha256']=hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()
        (out/'integrity.json').write_text(json.dumps(integrity))
        _,segments=read_score_context_segments(out,max_segment_length=64)
        self.assertEqual([s.q.tolist() for s in segments[0]],[[0,1],[2,3]])

    def test_no_observations_are_retained_as_ineligible_not_imputed(self):
        out=self.root/'out';result=export_score_context(self.quarter(all_missing=True),self.mapping(),out,dataset='ATEPP')
        self.assertEqual(result['records'][0]['usable_intervals'],0)
        _,segments=read_score_context_segments(out,max_segment_length=4)
        self.assertEqual(segments,[[],[]])
        with self.assertRaisesRegex(ValueError,'observed segment'):collate_score_context(segments)

    def test_collator_cleans_missing_values_and_keeps_zero_observations(self):
        s=ScoreContextSegment('a',np.arange(3.),np.ones((2,30)),np.array([[0.,np.nan],[2.,3.]]),np.array([[True,False],[True,True]]))
        batch=collate_score_context([[s]])
        self.assertEqual(float(batch['performance'][0,0,0,0]),0)
        self.assertTrue(batch['observation_mask'][0,0,0,0]);self.assertFalse(batch['observation_mask'][0,0,0,1])
        self.assertEqual(float(batch['performance'][0,0,0,1]),0)
        other=ScoreContextSegment('b',s.q,s.score,s.performance,s.observation_mask)
        with self.assertRaisesRegex(ValueError,'different recordings'):collate_score_context([[s,other]])


if __name__=='__main__':unittest.main()
