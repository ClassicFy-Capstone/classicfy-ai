"""Targeted checks for order, centering, valid spans, interpolation and lag."""
import unittest
import numpy as np
from compare import set_distances
from prepare import build, curve, lag_judge


class CurveTests(unittest.TestCase):
    def test_order_changes_distance_but_not_mean_or_range(self):
        a=np.array([[[0.,1.,3.,2.,0.]]])
        b=a[:,:,::-1]
        self.assertEqual(float(a.mean()),float(b.mean()))
        self.assertEqual(float(np.ptp(a)),float(np.ptp(b)))
        d=set_distances({0:a,1:b},np.array([0,1]),np.array([0,1]),(np.ones(1),np.ones(1,bool)))
        self.assertEqual(d[0,0],0.)
        self.assertGreater(d[0,1],0.)
        self.assertEqual(d[0,1],d[1,0])

    def test_centering_removes_constant_level_shift(self):
        a=np.arange(34.).reshape(1,2,17)
        shift=np.array([8.,-3.])[None,:,None]
        b=a+shift
        ac=a-a.mean(axis=(0,2))[None,:,None]
        bc=b-b.mean(axis=(0,2))[None,:,None]
        np.testing.assert_allclose(ac,bc)

    def test_interpolation_uses_score_position(self):
        positions=np.array([0.,2.,3.,7.])
        x=np.tile(2*positions,(7,1))
        actual=curve(x,0,4,positions)
        np.testing.assert_allclose(actual,np.tile(np.linspace(0,14,17),(7,1)))

    def test_spans_never_cross_views_or_missing_intervals(self):
        n=128
        x=np.tile(np.arange(n,dtype=float),(7,1)); mask=np.ones(n,bool);mask[40]=False
        pairs=np.ones(n,bool);pairs[0]=False;pairs[70]=False
        owner=np.arange(n)//16
        written=np.column_stack((np.arange(n),np.arange(n)+1))
        t=np.tile(np.sin(np.arange(n)*2*np.pi/24),(3,1))
        for view in (0,1):
            _,spans,_=build(x,mask,pairs,owner,written,t,np.zeros(n,int),view)
            for locations in spans.values():
                for a,b in locations:
                    self.assertEqual(a//64,(b-1)//64)
                    self.assertEqual(a//64%2,view)
                    self.assertTrue(mask[a:b].all())
                    self.assertTrue(pairs[a+1:b].all())

    def test_lag_sign_detects_one_beat_delayed_response(self):
        dt=np.random.default_rng(42).normal(size=16)
        dx=np.r_[0,dt[:-1]]
        t=np.tile(np.r_[0,np.cumsum(dt)],(3,1))
        x=np.tile(np.r_[0,np.cumsum(dx)],(7,1))
        spans={f'H{k}':np.array([[0,17]]) for k in range(3)}
        judge=lag_judge(x,t,spans).reshape(3,2,9)
        np.testing.assert_allclose(judge[:,:,5],1.,atol=1e-12)
        self.assertTrue((judge.argmax(axis=2)==5).all())


if __name__=='__main__':unittest.main()
