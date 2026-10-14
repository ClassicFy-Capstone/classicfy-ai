"""Checks for leakage, temporal order, collapse and lag semantics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import data
from data import response_targets,lag_targets,ViewData
from model import ResponseEncoder,masks,vicreg,objective


class ModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(3);torch.set_num_threads(2)

    def test_hidden_values_never_reach_encoder(self):
        model=ResponseEncoder().eval();x=torch.randn(4,7,17);c=torch.randn(4,13,17)
        h=masks(4,torch.Generator().manual_seed(1));poison=x.clone();poison[h]=float('nan')
        with torch.no_grad():a=model.encode(x,c,h);b=model.encode(poison,c,h)
        torch.testing.assert_close(a,b)

    def test_context_only_is_independent_of_performance(self):
        model=ResponseEncoder(True).eval();c=torch.randn(4,13,17)
        with torch.no_grad():a=model.encode(torch.randn(4,7,17),c);b=model.encode(torch.randn(4,7,17)*500,c)
        torch.testing.assert_close(a,b)

    def test_encoder_retains_order(self):
        model=ResponseEncoder().eval();x=torch.randn(4,7,17);c=torch.randn(4,13,17)
        with torch.no_grad():a=model.encode(x,c);b=model.encode(x.flip(2),c)
        self.assertGreater(float((a-b).abs().max()),1e-4)

    def test_constant_representation_has_variance_penalty(self):
        constant=torch.zeros(128,64);spread=torch.randn(128,64)*2
        self.assertGreater(float(vicreg(constant,constant)['variance']),.9)
        self.assertLess(float(vicreg(spread,spread)['variance']),.01)

    def test_lag_units_and_invalid_targets(self):
        dt=np.random.default_rng(1).normal(size=16);dx=np.r_[0,dt[:-1]]
        t=np.tile(np.r_[0,np.cumsum(dt)],(3,1));x=np.tile(np.r_[0,np.cumsum(dx)],(7,1))
        y,mask=lag_targets(x,t,np.ones_like(t,bool),2)
        self.assertTrue(mask[36+5]);self.assertAlmostEqual(y[36+5],1.)
        self.assertFalse(mask[:36].any())
        y,mask=lag_targets(x,t,np.zeros_like(t,bool),2);self.assertFalse(mask.any())

    def test_targets_distinguish_shape_from_amplitude(self):
        x=np.random.default_rng(5).normal(size=(7,17));t=np.ones((3,17));valid=np.ones_like(t,bool)
        a=response_targets(x,t,valid,-1);b=response_targets(x*3+50,t,valid,-1)
        np.testing.assert_allclose(a['shape'],b['shape'],atol=1e-12)
        np.testing.assert_array_equal(a['direction'],b['direction']);np.testing.assert_array_equal(a['peaks'],b['peaks'])

    def test_input_scales_ignore_validation_and_test(self):
        original=ViewData(0);arrays=original.arrays
        selected=[int(arrays['owner'][original.roles['train'][0]]),int(arrays['owner'][original.roles['validation'][0]]),int(arrays['owner'][original.roles['test'][0]])]
        ids=np.flatnonzero(np.isin(arrays['owner'],selected))
        old_out=data.OUT
        with tempfile.TemporaryDirectory(prefix='classicfy-model-scale-') as directory:
            try:
                data.OUT=Path(directory)
                small={'0/'+k:v[ids].copy() for k,v in arrays.items()}
                small.update(keys=np.asarray(original.keys),eligible=np.array(selected),gallery_indices=np.array(selected[:1]),query_indices=np.array(selected[1:]))
                np.savez_compressed(data.OUT/'examples.npz',**small);a=ViewData(0).scales
                other=small['0/owner']!=selected[0]
                for key in ('x','context','duration','lag'):small['0/'+key][other]+=1000
                np.savez_compressed(data.OUT/'examples.npz',**small);b=ViewData(0).scales
                self.assertEqual(a,b)
            finally:data.OUT=old_out

    def test_all_objectives_have_finite_gradients(self):
        d=ViewData(0);ii=d.roles['train'][:64];x=d.x[ii];c=d.context[ii];targets={k:v[ii] for k,v in d.targets.items()}
        for mode in ('reconstruction_only','response','response_aux','context_only'):
            model=ResponseEncoder(mode=='context_only');g=torch.Generator().manual_seed(9);a,b=masks(len(ii),g),masks(len(ii),g)
            loss,_=objective(mode,model(x,c,a),model(x,c,b),x,a,b,targets)
            loss.backward();self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
