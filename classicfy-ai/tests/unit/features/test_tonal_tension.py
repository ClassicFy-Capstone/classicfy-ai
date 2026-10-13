import unittest
from fractions import Fraction
from types import SimpleNamespace as NS

import numpy as np
from features.tonal_tension import TonalContext,pitch_point,key_center,SCALE,tension_labels


def note(step,alter=0,start=0,duration=1):
    pitch=NS(step=step,alter=Fraction(alter),midi_pitch=Fraction(60+dict(C=0,D=2,E=4,F=5,G=7,A=9,B=11)[step]+alter))
    return NS(pitch=pitch,onset_quarters=Fraction(start),offset_quarters=Fraction(start+duration),duration_quarters=Fraction(duration),is_cue=False)


def score(notes,mode='major'):
    return NS(notes=notes,measures=[NS(onset_quarters=Fraction(0),duration_quarters=Fraction(4))],
        key_signatures=[NS(onset_quarters=Fraction(0),part_id='P1',staff=None,fifths=0,mode=mode,alterations=())])


class TonalTensionTest(unittest.TestCase):
    def test_enharmonics_and_reference_scale(self):
        self.assertAlmostEqual(np.linalg.norm(pitch_point('C',0)-pitch_point('B',1))*SCALE,1)
        self.assertGreater(np.linalg.norm(pitch_point('C',1)-pitch_point('D',-1)),0)
        with self.assertRaises(ValueError):pitch_point('C',Fraction(1,2))
        with self.assertRaises(ValueError):key_center(0,None)

    def test_single_pitch_and_half_open_overlap(self):
        context=TonalContext(score([note('C'),note('G',start=1)]))
        first=context.cloud(0,1)
        self.assertEqual(first['notes'],1);self.assertEqual(first['diameter'],0)
        np.testing.assert_allclose(first['center'],pitch_point('C',0))
        second=context.cloud(.5,1.5)
        np.testing.assert_allclose(second['center'],(pitch_point('C',0)+pitch_point('G',0))/2)
        self.assertAlmostEqual(second['diameter'],np.linalg.norm(pitch_point('C',0)-pitch_point('G',0))*SCALE)

    def test_silence_and_jump_reset_momentum(self):
        context=TonalContext(score([note('C'),note('G',start=2)]))
        values,_=context.project([(0,1),(1,2),(2,3),(0,1)],[False,False,False,True])
        self.assertTrue(np.isnan(values[1]).all());self.assertTrue(np.isnan(values[:,1]).all())
        normal,_=context.project([(0,1),(2,3)],[False,False])
        self.assertAlmostEqual(normal[1,1],np.linalg.norm(pitch_point('C',0)-pitch_point('G',0))*SCALE)

    def test_unknown_mode_retains_both_hypotheses(self):
        context=TonalContext(score([note('C'),note('E'),note('G')],mode=None),mode_margin=2)
        cloud=context.cloud(0,1)
        self.assertIsNone(context.contexts[0].mode);self.assertTrue(np.isnan(cloud['strain']))
        self.assertTrue(np.isfinite(cloud['strain_major']));self.assertTrue(np.isfinite(cloud['strain_minor']))

    def test_microtone_and_invalid_interval(self):
        context=TonalContext(score([note('C',Fraction(1,2))]))
        self.assertEqual(context.cloud(0,1)['status'],'unsupported_microtone')
        self.assertEqual(context.cloud(1,1)['status'],'invalid_interval')
        self.assertTrue(np.isnan(context.cloud(0,1)['diameter']))

    def test_missing_or_conflicting_key_keeps_key_free_metrics(self):
        source=score([note('C')]);source.key_signatures=[]
        context=TonalContext(source);cloud=context.cloud(0,1)
        self.assertEqual(cloud['diameter'],0);self.assertTrue(np.isnan(cloud['strain']))
        source.key_signatures=[NS(onset_quarters=Fraction(0),part_id=p,staff=None,fifths=k,mode='major',alterations=()) for p,k in [('P1',0),('P2',1)]]
        self.assertEqual(TonalContext(source).contexts[0].status,'missing_or_conflicting_signature')

    def test_tertiles_preserve_ties_and_invalid(self):
        label,t=tension_labels([0,0,0,1,2,np.nan],np.ones(6,bool))
        self.assertEqual(label[-1],-1);self.assertEqual(label[0],label[2])
        label,t=tension_labels([0,0,0],np.ones(3,bool))
        np.testing.assert_array_equal(label,[0,0,0])


if __name__=='__main__':unittest.main()
