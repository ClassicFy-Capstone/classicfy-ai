import sys
from pathlib import Path
import unittest

import numpy as np

AI=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(AI/'src'),str(AI/'scripts')]
from embedding.retrieval_probe import (ranked,partition,judges,finite_pairs,overlap,
    percentile_scores,weights,hand_descriptors,PAIRS,CHANNELS,LAGS)
from run_metadata_retrieval_experiment import tiers,select,conditional_sources


class RetrievalProbeTests(unittest.TestCase):
    def test_tiers_are_not_transitively_chained_and_match_reference(self):
        d=np.array([0.,.75e-12,1.5e-12,3.,np.inf]);f=np.array(['a','b','c','d','e'])
        priority=np.array([.9,.8,.1,.2,.0])
        np.testing.assert_equal(select(tiers(d),f,priority,3),ranked(d,f,priority,3))
        np.testing.assert_equal(select(tiers(d),f,priority,3),[1,0,2])

    def test_family_cap_selects_nearest_recording_not_arbitrary_work_member(self):
        d=np.array([.1,.2,.3,.4]);f=np.array(['a','a','b','c']);p=np.array([.9,.0,.8,.7])
        np.testing.assert_equal(ranked(d,f,p,3),[0,2,3])
        self.assertIsNone(ranked(d,f,p,4))
        np.testing.assert_equal(ranked(d,f,p,4,False),[0,1,2,3])

    def test_recording_and_family_overlap_are_distinct(self):
        f=np.array(['a','a','b','b'])
        result=overlap(np.array([0,2]),np.array([1,3]),f)
        self.assertEqual(result['recording_overlap'],0)
        self.assertEqual(result['family_overlap'],1)

    def test_disjoint_blocks_keep_original_positions_and_discard_tail(self):
        x=np.arange(7*270).reshape(7,270);valid=np.ones(270,bool)
        a,aa=partition(x,valid,0);b,bb=partition(x,valid,1)
        self.assertEqual(aa,[0,128]);self.assertEqual(bb,[64,192])
        self.assertFalse((a&b).any());self.assertFalse((a|b)[256:].any())
        left,right=finite_pairs(x[0],a,1)
        self.assertEqual(len(left),126);self.assertTrue(np.all(right-left==1))
        self.assertNotIn(63,left);self.assertNotIn(127,left)

    def test_judges_distinguish_opposite_coupling_and_undefined_constant(self):
        x=np.zeros((7,256));m=np.ones(256,bool);rng=np.random.default_rng(9)
        x[0]=rng.normal(size=256);x[2]=-x[0]
        mask,starts=partition(x,m,0);j=judges(x,mask,starts)
        np.testing.assert_allclose(j['E-R'],[-1,-1,-1,0],atol=1e-12)
        self.assertTrue(np.isfinite(j['E-T']).all())
        x[2]=0;self.assertTrue(np.isnan(judges(x,mask,starts)['E-R']).all())

    def test_search_descriptors_cannot_observe_judge_side(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(7,512));m=np.ones(512,bool)
        a,_=partition(x,m,0)
        before=hand_descriptors(x,a,np.ones(7),list(range(70)))
        x[:,~a]=rng.normal(100,1000,size=x[:,~a].shape)
        after=hand_descriptors(x,a,np.ones(7),list(range(70)))
        for u,v in zip(before,after):np.testing.assert_allclose(u,v,equal_nan=True,atol=1e-12)

    def test_percentile_random_expectation_and_ties(self):
        d=np.array([[1.,1.,3.,8.]]);w=np.array([.1,.2,.3,.4])
        s=percentile_scores(d,w)
        np.testing.assert_allclose(s,[[.85,.85,.55,.2]])
        self.assertAlmostEqual((s@w).item(),.5)

    def test_conditional_permutation_never_crosses_metadata_quality_strata(self):
        rows=[dict(composer='Bach' if i<4 else 'Chopin',work_type='sonata',piece=str(i//2),grid='g',family_id=str(i//2)) for i in range(8)]
        q=np.array([[1.,1.]]*4+[[2.,1.]]*4)
        sources,support=conditional_sources(rows,q,weights(rows),[1,2,3])
        self.assertEqual(support['status'],'ok')
        for source in sources:
            self.assertEqual(sorted(source),list(range(8)))
            for i,j in enumerate(source):
                self.assertEqual(rows[i]['composer'],rows[j]['composer'])
                np.testing.assert_equal(q[i],q[j])


if __name__=='__main__':unittest.main()
