from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from features.score_context import (SCORE_CHANNELS,ScoreContextScale,encode_score_features,
    extract_score_context,fit_score_context_scale,load_score_context,score_context_schema)
from preprocessing.note_alignment import ScoreNote
from preprocessing.score_midi import load_score_midi,read_score_midi


def write_score(path, *, velocity=64, tempo=500000, duplicate=False, dangling=False):
    midi=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();midi.tracks.append(track)
    track.extend([mido.MetaMessage('set_tempo',tempo=tempo,time=0),
        mido.Message('note_on',note=60,velocity=velocity,time=0),
        mido.Message('note_on',note=64,velocity=velocity,time=0),
        mido.Message('note_off',note=60,velocity=0,time=480),
        mido.Message('note_off',note=64,velocity=0,time=480),
        mido.MetaMessage('end_of_track',time=480)])
    if duplicate:midi.tracks.append(mido.MidiTrack(track))
    if dangling:midi.tracks.append(mido.MidiTrack([mido.Message('note_on',note=72,velocity=50,time=0)]))
    midi.save(path)


class ScoreContextTest(unittest.TestCase):
    def test_exact_onsets_overlap_register_and_boundary(self):
        notes=[ScoreNote('a',60,0,1,()),ScoreNote('b',64,.5,1.5,()),ScoreNote('c',67,1,2,())]
        result=extract_score_context(notes,[0,1,2,3],score_stop=3,source_sha256='score')
        a,b,rest=result.raw
        np.testing.assert_array_equal(a[:12],np.eye(12)[0]+np.eye(12)[4])
        np.testing.assert_allclose(a[12:24],np.eye(12)[0]+.5*np.eye(12)[4])
        self.assertAlmostEqual(a[25],(60+.5*64)/1.5)
        np.testing.assert_allclose(a[24:],[2,(60+.5*64)/1.5,60,64,1,0])
        self.assertEqual(b[7],1);self.assertEqual(b[0],0)
        self.assertAlmostEqual(b[12+4],.5);self.assertAlmostEqual(b[12+7],1)
        np.testing.assert_allclose(rest,np.r_[np.zeros(29),1])
        self.assertEqual(len(SCORE_CHANNELS),30);self.assertEqual(score_context_schema()['dimension'],30)

    def test_sustain_is_not_silence_and_full_duration_is_not_overlap(self):
        notes=[ScoreNote('held',48,0,4,())]
        result=extract_score_context(notes,[0,1,2,3,4],score_stop=4)
        self.assertEqual(result.raw[0,28],4)
        self.assertEqual(result.raw[1,28],0)
        np.testing.assert_array_equal(result.raw[:,29],0)
        np.testing.assert_array_equal(result.raw[:,12],1)

    def test_unsupported_extent_and_invalid_notes(self):
        result=extract_score_context([], [0,1,2],score_stop=1)
        np.testing.assert_array_equal(result.valid,[True,False]);self.assertTrue(np.isnan(result.raw[1]).all())
        np.testing.assert_array_equal(encode_score_features(result.raw,result.valid)[1],0)
        for q in ([0,0,1],[0,np.nan],[-1,0],[1,0],['0','1'],[False,True],[0,1+2j]):
            with self.subTest(q=q),self.assertRaises(ValueError):extract_score_context([],q,score_stop=2)
        for note in [ScoreNote('bad',128,0,1,()),ScoreNote('bad',60,1,1,()),ScoreNote('bad',60,0,3,())]:
            with self.assertRaises(ValueError):extract_score_context([note],[0,1],score_stop=2)

    def test_score_midi_velocity_tempo_and_duplicate_tracks(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.mid',Path(folder)/'b.mid'
            write_score(a);write_score(b,velocity=112,tempo=1000000,duplicate=True)
            aa=load_score_context(a,[0,1,2,3]);bb=load_score_context(b,[0,1,2,3])
            np.testing.assert_array_equal(aa.raw,bb.raw)
            self.assertEqual(bb.diagnostics['duplicate_notes_removed'],2)
            self.assertNotEqual(aa.source_sha256,bb.source_sha256)
            self.assertNotEqual(aa.cache_key,bb.cache_key)

    def test_loader_retains_original_pairing_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'s.mid';write_score(path,duplicate=True)
            self.assertEqual(load_score_midi(path),
                (ScoreNote('0',60,0.,1.,()),ScoreNote('1',64,0.,2.,())))

    def test_unclosed_notes_are_diagnostic_and_context_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'s.mid';write_score(path,dangling=True)
            self.assertEqual(read_score_midi(path)[2]['unclosed_notes'],1)
            with self.assertRaisesRegex(ValueError,'Unclosed'):load_score_context(path,[0,1,2,3])

    def test_duplicate_note_api_and_transform_values(self):
        note=ScoreNote('n',60,0,1,())
        feature=extract_score_context([note,note],[0,1,2],score_stop=2)
        values=encode_score_features(feature.raw,feature.valid)
        self.assertAlmostEqual(values[0,0],np.log(2));self.assertAlmostEqual(values[0,25],60/127)
        self.assertEqual(values[0,24],values[0,0]);self.assertEqual(values[1,29],1)

    def test_train_scale_is_score_balanced_and_repeated_performances_add_no_weight(self):
        a=extract_score_context([ScoreNote('a',60,0,1,())],[0,1,2],score_stop=2,source_sha256='a')
        b=extract_score_context([ScoreNote('b',84,0,2,())],[0,1,2],score_stop=2,source_sha256='b')
        first=fit_score_context_scale([a,b]);second=fit_score_context_scale([a,a,a,b])
        self.assertEqual(first,second)
        self.assertAlmostEqual(first.mean[25],((60/127)/2+84/127)/2)
        self.assertEqual(ScoreContextScale.from_dict(first.to_dict()),first)
        self.assertEqual(first.mean[29],0);self.assertEqual(first.scale[29],1)
        self.assertTrue(np.isfinite(first.transform(a)).all())
        with self.assertRaises(ValueError):fit_score_context_scale([])

    def test_schema_identity_depends_on_grid_and_unknown_mask_is_not_silence(self):
        a=extract_score_context([], [0,1,2],score_stop=1,source_sha256='a')
        b=extract_score_context([], [.5,1.5],score_stop=1,source_sha256='a')
        self.assertNotEqual(a.cache_key,b.cache_key)
        self.assertEqual(a.raw[0,29],1);self.assertFalse(a.valid[1])
        bad=a.raw.copy();bad[0,0]=np.inf
        with self.assertRaises(ValueError):encode_score_features(bad,a.valid)


if __name__=='__main__':unittest.main()
