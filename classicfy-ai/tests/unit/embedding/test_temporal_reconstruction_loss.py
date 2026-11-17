import unittest
import torch

from embedding.model import error_totals, feature_loss
from embedding.temporal_reconstruction_loss import channel_error_totals, weighted_feature_loss


class TemporalReconstructionLossTest(unittest.TestCase):
    def test_default_mse_preserves_value_and_gradient(self):
        torch.manual_seed(12)
        prediction=torch.randn(2,7,5,requires_grad=True)
        target=torch.randn_like(prediction)
        mask=torch.rand_like(target)>.3
        old=feature_loss(*error_totals(prediction,target,mask))
        new=weighted_feature_loss(*channel_error_totals(prediction,target,mask))
        torch.testing.assert_close(new,old)
        torch.testing.assert_close(torch.autograd.grad(new,prediction,retain_graph=True)[0],
                                   torch.autograd.grad(old,prediction)[0])

    def test_huber_hand_value_weight_and_nan_padding(self):
        p=torch.zeros(1,7,2,requires_grad=True)
        t=torch.full_like(p,2.);t[:,4:]=4.;t[:,:,1]=float('nan')
        mask=torch.zeros_like(t,dtype=torch.bool);mask[:,:,0]=True
        sums,counts=channel_error_totals(p,t,mask,kind='huber',delta=1.)
        value=weighted_feature_loss(sums,counts,pedal_weight=2.)
        self.assertAlmostEqual(float(value.detach()),13/6,places=6)
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertFalse(p.grad[:,:,1].any())

    def test_huber_bounds_outlier_gradient_without_clipping_targets(self):
        p=torch.zeros(1,7,1,requires_grad=True)
        t=torch.full_like(p,400.);mask=torch.ones_like(t,dtype=torch.bool)
        loss=weighted_feature_loss(*channel_error_totals(p,t,mask,kind='huber'))
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()),399.5,places=4)
        self.assertLessEqual(float(p.grad.abs().max()),.200001)
        self.assertTrue((t==400).all())

    def test_rejects_invalid_parameters(self):
        p=torch.zeros(1,7,1);mask=torch.ones_like(p,dtype=torch.bool)
        for delta in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):channel_error_totals(p,p,mask,kind='huber',delta=delta)
        with self.assertRaises(ValueError):weighted_feature_loss(torch.ones(7),torch.ones(7),pedal_weight=0.)


if __name__=='__main__':unittest.main()
