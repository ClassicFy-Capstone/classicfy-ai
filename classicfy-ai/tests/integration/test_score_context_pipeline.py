import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mido
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.score_context_data import export_score_context,read_score_context_segments,collate_score_context
from embedding.score_encoder import ScoreEncoder,ScoreEncoderConfig
from preprocessing import ASAPLoader


class ScoreContextPipelineTest(unittest.TestCase):
    def test_real_asap_handoff_score_to_context(self):
        source_path=ROOT/'classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/all_asap_quarter_normalized.npz'
        asap=ROOT.parent/'datasets/ASAP';nasap=ROOT.parent/'datasets/nASAP'
        if not source_path.is_file() or source_path.stat().st_size<200 or not (asap/'metadata.csv').is_file():
            self.skipTest('Actual LFS quarter cache and ASAP source MIDI are required')
        with tempfile.TemporaryDirectory() as directory,np.load(source_path,allow_pickle=False) as source:
            folder=Path(directory);metadata=json.loads(str(source['manifest']))[:2];arrays={}
            loader=ASAPLoader(asap,nasap);mapping=[]
            for i,r in enumerate(metadata):
                sample=loader.get_sample(r['key'])
                if not sample.score_path.is_file():self.skipTest('Missing real score MIDI')
                for field in ('q','values','valid'):arrays[f'{i}/{field}']=source[f'{i}/{field}']
                mapping.append(dict(key=r['key'],work=r['work'],score_path=str(sample.score_path),score_sha256=hashlib.sha256(sample.score_path.read_bytes()).hexdigest()))
            arrays['manifest']=np.array(json.dumps(metadata));np.savez(folder/'source.npz',**arrays)
            with (folder/'map.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(mapping[0]));writer.writeheader();writer.writerows(mapping)
            export_score_context(folder/'source.npz',folder/'map.csv',folder/'out',dataset='ASAP')
            _,recordings=read_score_context_segments(folder/'out',max_segment_length=64)
            batch=collate_score_context(recordings)
            threads=torch.get_num_threads();torch.set_num_threads(1)
            try:
                torch.manual_seed(33);encoder=ScoreEncoder(ScoreEncoderConfig(30))
                with torch.no_grad():context=encoder(batch['score'],batch['time_mask'],batch['segment_mask'])
            finally:torch.set_num_threads(threads)
            self.assertEqual(context.shape[-1],128);self.assertTrue(torch.isfinite(context).all())
            torch.testing.assert_close(batch['score'][0],batch['score'][1])
            torch.testing.assert_close(context[0],context[1])
            self.assertEqual(batch['keys'],[r['key'] for r in metadata])

    def test_cli_accepts_synthetic_native_atepp_cache_and_explicit_score_map(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);midi=mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on',note=60,velocity=70,time=0),mido.Message('note_off',note=60,velocity=0,time=1920)]))
            path=folder/'score.mid';midi.save(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            channels=['tempo','rubato','dynamics','articulation','pedal_depth','pedal_down_ratio','pedal_changes']
            metadata=[];arrays={}
            for i,channel in enumerate(channels):
                metadata.append(dict(key='00001',piece='synthetic',channel=channel))
                arrays[f'{i}/score_beats']=np.arange(5.);arrays[f'{i}/standardized_values']=np.arange(4.)+i
                arrays[f'{i}/standardized_mask']=np.array([True,False,True,True])
            arrays['manifest']=np.array(json.dumps(dict(records=metadata,provenance=dict(dataset='ATEPP-1.2',aligner='dual_dtw',source_files=[dict(key='00001',score_sha256=digest)]))))
            np.savez(folder/'native.npz',**arrays)
            with (folder/'map.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['key','work','score_path','score_sha256','split']);writer.writeheader()
                writer.writerow(dict(key='00001',work='synthetic',score_path='score.mid',score_sha256=digest,split='train'))
            command=[sys.executable,str(ROOT/'classicfy-ai/scripts/export_atepp_score_context.py'),'--performance-cache',str(folder/'native.npz'),
                '--score-map',str(folder/'map.csv'),'--dataset','ATEPP','--fit-score-scale','--out',str(folder/'out')]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('Exported 1 recordings',result.stdout)
            manifest,recordings=read_score_context_segments(folder/'out',max_segment_length=3)
            self.assertEqual(manifest['records'][0]['key'],'00001');self.assertEqual(len(recordings[0]),2)
            np.testing.assert_array_equal(recordings[0][0].observation_mask[1],False)


if __name__=='__main__':unittest.main()
