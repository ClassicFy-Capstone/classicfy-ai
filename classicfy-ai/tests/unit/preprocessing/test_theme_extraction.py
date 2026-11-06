"""Boundary inversion and MIDI export preserve tempo, held notes and pedal state."""
import sys
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from extract_atepp_themes import clip_midi, local_boundary


class ThemeExtractionTests(unittest.TestCase):
    def test_float_accumulation_does_not_include_note_at_exclusive_end(self):
        with tempfile.TemporaryDirectory() as td:
            sequence=[(0.,mido.Message('note_on',note=60,velocity=80)),
                      (1.,mido.Message('note_off',note=60)),
                      (1.5-1e-13,mido.Message('note_on',note=64,velocity=70)),
                      (2.,mido.Message('note_off',note=64))]
            self.assertEqual(clip_midi(None,Path(td)/'clip.mid',.3125,1.5,sequence),1)

    def test_zero_duration_source_note_is_not_stretched_to_later_reattack(self):
        with tempfile.TemporaryDirectory() as td:
            source,dest=Path(td)/'source.mid',Path(td)/'clip.mid'
            midi=mido.MidiFile(ticks_per_beat=1000)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage('set_tempo',tempo=1000000,time=0),
                mido.Message('note_on',note=62,velocity=44,time=0),
                mido.Message('note_off',note=62,time=0),
                mido.Message('note_on',note=62,velocity=43,time=1000),
                mido.Message('note_off',note=62,time=500),
            ]))
            midi.save(source)
            self.assertEqual(clip_midi(source,dest,.5,1.5),1)
            ons=[m for m in mido.MidiFile(dest) if m.type=='note_on' and m.velocity]
            self.assertEqual([m.velocity for m in ons],[43])

    def test_local_boundary_rejects_repeat_jump(self):
        with self.assertRaises(ValueError):
            local_boundary(2.,np.array([0.,1.]),np.array([1.,8.]),.5)

    def test_local_boundary_interpolates_short_gap(self):
        q,gap,method=local_boundary(1.25,np.array([0.,1.]),np.array([1.,1.5]),.5)
        self.assertEqual(q,.5)
        self.assertEqual(method,'local_interpolation')

    def test_clip_integrates_tempo_and_carries_held_note_and_pedal(self):
        with tempfile.TemporaryDirectory() as td:
            source,dest=Path(td)/'source.mid',Path(td)/'clip.mid'
            midi=mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage('set_tempo',tempo=500000,time=0),
                mido.Message('control_change',control=64,value=100,time=0),
                mido.Message('note_on',note=60,velocity=90,time=0),
                mido.MetaMessage('set_tempo',tempo=1000000,time=480),
                mido.Message('note_off',note=60,velocity=0,time=480),
                mido.Message('note_on',note=64,velocity=70,time=0),
                mido.Message('note_off',note=64,velocity=0,time=480),
            ]))
            midi.save(source)
            clip_midi(source,dest,.75,1.5)
            result=mido.MidiFile(dest)
            self.assertAlmostEqual(result.length,.75)
            now=0;on=[];off=[];pedal=[]
            for msg in result:
                now+=msg.time
                if msg.type=='note_on' and msg.velocity: on.append((now,msg.note,msg.velocity))
                if msg.type=='note_off': off.append((now,msg.note))
                if msg.type=='control_change' and msg.control==64: pedal.append((now,msg.value))
            self.assertEqual(on,[(0.,60,90)])
            self.assertEqual(off,[(.75,60)])
            self.assertEqual(pedal,[(0.,100),(.75,0)])

    def test_submillisecond_boundary_fragment_does_not_create_orphan_offset(self):
        with tempfile.TemporaryDirectory() as td:
            source,dest=Path(td)/'source.mid',Path(td)/'clip.mid'
            midi=mido.MidiFile(ticks_per_beat=1000)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage('set_tempo',tempo=1000000,time=0),
                mido.Message('note_on',note=60,velocity=90,time=0),
                mido.Message('note_off',note=60,time=1000),
                mido.Message('note_on',note=64,velocity=70,time=0),
                mido.Message('note_off',note=64,time=1000),
            ]))
            midi.save(source)
            self.assertEqual(clip_midi(source,dest,.99999,1.5),1)
            notes=[m for m in mido.MidiFile(dest) if m.type in ('note_on','note_off')]
            self.assertEqual([(m.type,m.note) for m in notes],[('note_on',64),('note_off',64)])


if __name__=='__main__':
    unittest.main()
