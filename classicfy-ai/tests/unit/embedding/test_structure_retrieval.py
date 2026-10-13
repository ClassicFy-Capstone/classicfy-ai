import unittest
import numpy as np
from embedding.structure_retrieval import view_descriptors,heldout_judges
from features.structure_response import fixed_labels
from embedding.retrieval_probe import ranked,percentile_scores,fit_scale,distance


class StructureRetrievalTest(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(7,512));mask=np.ones(512,bool)
        phase,owner=fixed_labels(512);tension=np.tile(np.sin(np.arange(512)/5),(3,1))
        bins=np.tile(np.arange(512)%3,(3,1))
        return x,mask,mask.copy(),phase,owner,tension,bins,np.zeros(512,int)

    def test_search_view_does_not_read_other_view_values(self):
        args=list(self.data());first=view_descriptors(*args,0)
        for start in (64,192,320,448):args[0][:,start:start+64]+=1000
        second=view_descriptors(*args,0)
        for name in first:np.testing.assert_allclose(first[name],second[name],equal_nan=True)

    def test_block_cut_does_not_create_delta_outlier(self):
        args=list(self.data());args[0][:,:]=np.tile(np.arange(512)%64,(7,1))
        z=view_descriptors(*args,0)
        np.testing.assert_allclose(z['T'][:14],1)
        self.assertEqual(z['T'][14],1)

    def test_dimensions_and_deterministic_structure_shift(self):
        args=self.data();first=view_descriptors(*args,0,shift_seed=17);second=view_descriptors(*args,0,shift_seed=17)
        for name,size in [('B',14),('T',15),('F',66),('P',66),('H',198),('S',9),('Q',2)]:
            self.assertEqual(first[name].shape,(size,));np.testing.assert_allclose(first[name],second[name],equal_nan=True)
        unshifted=view_descriptors(*args,0)
        np.testing.assert_allclose(first['B'],unshifted['B'])

    def test_judges_use_disjoint_beats_and_defined_constant_delta(self):
        args=self.data();x=args[0].copy();x[4:]=0
        judge=heldout_judges(x,*args[1:5],0)
        self.assertTrue(np.isfinite(judge['E-L']).all());self.assertTrue(np.isfinite(judge['E-T']).all())
        np.testing.assert_allclose(judge['E-T'][12:],0)
        x[:,64:128]+=100
        after=heldout_judges(x,*args[1:5],0)
        for name in judge:np.testing.assert_allclose(judge[name],after[name],equal_nan=True)

    def test_family_cap_and_random_percentile(self):
        selected=ranked(np.array([0.,.1,.2,.3]),np.array(['a','a','b','c']),np.arange(4),3)
        np.testing.assert_array_equal(selected,[0,2,3])
        scores=percentile_scores(np.array([[1.,2.,3.]]),np.ones(3)/3)
        self.assertAlmostEqual(float(scores.mean()),.5)
        self.assertIsNone(ranked(np.zeros(2),np.array(['a','a']),np.arange(2),2))

    def test_training_scale_excludes_constant_coordinates(self):
        scale=fit_scale(np.array([[0.,1.],[2.,1.]]),np.array([.5,.5]))
        self.assertEqual(scale['active'],[True,False])
        np.testing.assert_allclose(distance(np.array([[0.,99.]]),np.array([[2.,-99.]]),scale),[[2.]])

    def test_shifted_tension_has_known_linear_deltas_in_every_supported_bin(self):
        args=list(self.data());args[0]=np.tile(np.arange(512),(7,1)).astype(float)
        labels,_=fixed_labels(512);args[6]=np.tile(labels,(3,1))
        z=view_descriptors(*args,0,shift_seed=17)
        self.assertTrue(np.isfinite(z['H']).all())
        for metric in range(3):
            for phase in range(3):
                start=metric*66+phase*22
                np.testing.assert_allclose(z['H'][start+7:start+22],1)


if __name__=='__main__':unittest.main()
