import unittest
import numpy as np
from fractions import Fraction
from types import SimpleNamespace as NS

from features.structure_response import (PhraseHypothesis, phase_labels, response,
    split_view, fixed_labels, phrase_hypotheses)


class StructureResponseTest(unittest.TestCase):
    def test_midpoint_fraction_and_shift_preserve_endpoints(self):
        ps=[PhraseHypothesis('a','P1',Fraction(0),Fraction(3),(),()),
            PhraseHypothesis('b','P1',Fraction(3),Fraction(6),(),())]
        labels,owners=phase_labels([(0,1),(1,2),(2,3),(3,4),None,(6,7),(2,2)],ps)
        np.testing.assert_array_equal(labels,[0,1,2,0,-1,-1,-1])
        shifted,_=phase_labels([(0,.2),(5.8,6)],ps,.5)
        np.testing.assert_array_equal(shifted,[0,2])

    def test_known_linear_response_and_owner_cut(self):
        x=np.tile(np.arange(12),(7,1)).astype(float)
        label=np.repeat(np.arange(3),4);mask=np.ones(12,bool);pairs=mask.copy()
        z,s=response(x,mask,pairs,label,minimum_beats=2,minimum_pairs=2)
        np.testing.assert_allclose(z[:7],1.5)
        np.testing.assert_allclose(z[7:21],1)
        self.assertEqual(z[21],1)
        self.assertEqual(s[0]['pairs'],3)
        owner=np.arange(12)
        zz,_=response(x,mask,pairs,label,owner,minimum_beats=2,minimum_pairs=2)
        self.assertTrue(np.isnan(zz[7:22]).all())

    def test_missing_and_jump_are_not_bridged(self):
        x=np.tile([0,1,500,3,4,5],(7,1)).astype(float)
        mask=np.array([1,1,0,1,1,1],bool);pairs=np.ones(6,bool);pairs[4]=False
        z,s=response(x,mask,pairs,np.zeros(6,int),minimum_beats=1,minimum_pairs=1)
        self.assertEqual(s[0]['pairs'],2)
        np.testing.assert_allclose(z[7:21],1)
        self.assertTrue(np.isnan(z[22:]).all())

    def test_disjoint_blocks_and_tail(self):
        mask=np.ones(150,bool)
        a,pa,ba=split_view(mask,mask,0);b,pb,bb=split_view(mask,mask,1)
        self.assertFalse((a&b).any());self.assertEqual(ba,[0]);self.assertEqual(bb,[64])
        self.assertFalse(pa[0]);self.assertFalse(pb[64]);self.assertFalse((a|b)[128:].any())

    def test_score_only_proposals_retain_fallback_identity(self):
        score=NS(part_names=(('P1','Piano'),),measures=(NS(part_id='P1',onset_quarters=Fraction(0),duration_quarters=Fraction(70)),))
        result=phrase_hypotheses(score,NS(candidates=(),slurs=()))
        self.assertEqual([(p.start,p.stop) for p in result],[(0,32),(32,64),(64,70)])
        self.assertEqual(result[0].stop_evidence,('fixed_length_cap',))
        self.assertEqual(result[-1].stop_evidence,('score_end',))

    def test_validation_and_fixed_owner(self):
        with self.assertRaises(ValueError):response(np.zeros((2,4)),np.ones(4,bool),np.ones(4,bool),np.zeros(4,int))
        with self.assertRaises(ValueError):split_view(np.ones(4,bool),np.ones(4,bool),2)
        labels,owners=fixed_labels(32)
        self.assertEqual(owners[15],0);self.assertEqual(owners[16],1)
        self.assertEqual(labels[15],2);self.assertEqual(labels[16],0)


if __name__=='__main__':unittest.main()
