import io
import unittest

import torch
from torch import nn

from embedding.score_encoder import ScoreEncoder,ScoreEncoderConfig


class ScoreEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def batch(self):
        torch.manual_seed(5)
        score=torch.randn(2,3,5,4)
        segments=torch.tensor([[True,False,True],[False,True,False]])
        time=torch.zeros(2,3,5,dtype=torch.bool)
        time[0,0,:3]=True;time[0,2,:5]=True;time[1,1,:1]=True
        return score,time,segments

    def test_default_dimensions_and_variable_input_shape(self):
        model=ScoreEncoder(ScoreEncoderConfig(4));score,time,segments=self.batch()
        result=model(score,time,segments)
        self.assertEqual(result.shape,(2,3,5,128));self.assertTrue(torch.isfinite(result).all())
        valid=time&segments.unsqueeze(-1);self.assertEqual(torch.count_nonzero(result[~valid]),0)
        for ds in (1,7,30):
            small=ScoreEncoder(ScoreEncoderConfig(ds,3))
            self.assertEqual(small(torch.randn(1,1,2,ds),torch.ones(1,1,2,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool)).shape,(1,1,2,6))

    def test_packed_segments_equal_individual_encodings_and_preserve_owners(self):
        model=ScoreEncoder(ScoreEncoderConfig(4,3));score,time,segments=self.batch();output=model(score,time,segments)
        for i,j in torch.nonzero(segments).tolist():
            n=int(time[i,j].sum());individual=model(score[i:i+1,j:j+1,:n],torch.ones(1,1,n,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool))
            torch.testing.assert_close(output[i,j,:n],individual[0,0])

    def test_right_time_and_segment_padding_invariance(self):
        model=ScoreEncoder(ScoreEncoderConfig(4,4));score,time,segments=self.batch();before=model(score,time,segments)
        padded=torch.full((2,5,9,4),float('nan'));padded[:,:3,:5]=score
        tm=torch.zeros(2,5,9,dtype=torch.bool);tm[:,:3,:5]=time
        sm=torch.zeros(2,5,dtype=torch.bool);sm[:,:3]=segments
        after=model(padded,tm,sm)
        torch.testing.assert_close(before,after[:,:3,:5]);self.assertEqual(torch.count_nonzero(after[:,3:]),0)
        modified=score.clone();modified[~(time&segments.unsqueeze(-1))]=float('inf')
        torch.testing.assert_close(model(modified,time,segments),before)

    def test_masks_and_finite_observation_validation(self):
        model=ScoreEncoder(ScoreEncoderConfig(4,3));score,time,segments=self.batch()
        bad=score.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):model(bad,time,segments)
        hole=time.clone();hole[0,0]=torch.tensor([True,False,True,False,False])
        with self.assertRaisesRegex(ValueError,'prefix'):model(score,hole,segments)
        empty=segments.clone();empty[1]=False
        with self.assertRaisesRegex(ValueError,'recording'):model(score,time,empty)
        no_time=time.clone();no_time[0,0]=False
        with self.assertRaisesRegex(ValueError,'time'):model(score,no_time,segments)
        with self.assertRaises(ValueError):model(score,time.float(),segments)
        with self.assertRaises(ValueError):model(score.double(),time,segments)
        for dims in [(0,1),(4,0),(True,4)]:
            with self.assertRaises(ValueError):ScoreEncoderConfig(*dims)

    def test_context_changes_with_score_and_loss_gradient_reaches_encoder(self):
        model=ScoreEncoder(ScoreEncoderConfig(4,4));score,time,segments=self.batch();score.requires_grad_()
        context=model(score,time,segments);changed=score.detach().clone();changed[0,0,0]+=2
        self.assertFalse(torch.allclose(model(changed,time,segments),context))
        # Test-only stand-in for the future conditional decoder, never a trained AE claim.
        decoder=nn.GRU(8+3,5,batch_first=True);head=nn.Linear(5,2)
        z=torch.randn(2,3,3,requires_grad=True)
        inputs=torch.cat([context,z.unsqueeze(2).expand(-1,-1,5,-1)],dim=-1).reshape(6,5,11)
        prediction=head(decoder(inputs)[0]).reshape(2,3,5,2)
        valid=time&segments.unsqueeze(-1)
        loss=nn.functional.huber_loss(prediction[valid],torch.ones_like(prediction[valid]));loss.backward()
        self.assertGreater(float(model.gru.weight_ih_l0.grad.abs().sum()),0)
        self.assertGreater(float(z.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(score.grad).all());self.assertEqual(torch.count_nonzero(score.grad[~valid]),0)

    def test_saved_state_restores_exact_context(self):
        torch.manual_seed(9);model=ScoreEncoder(ScoreEncoderConfig(4,3));batch=self.batch();before=model(*batch)
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        restored=ScoreEncoder(model.config);restored.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(before,restored(*batch),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
