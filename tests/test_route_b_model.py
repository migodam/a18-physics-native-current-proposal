import unittest
import numpy as np
import torch
from route_b.model import CoefficientMLP
from route_b.loss import expected_best,physical_loss
from route_b.normalizer import Normalizer
from route_b.basis import build_basis
from route_b.small_physics import SmallPhysics,teacher_best
from test_shared_backend import dense_backend

class ModelTests(unittest.TestCase):
    def test_architecture_rank_padding_and_logits(self):
        torch.manual_seed(1);model=CoefficientMLP();mask=torch.zeros((2,24));mask[:,:3]=1
        t,l,n,active=model(torch.randn(2,1552),torch.randn(2,8,32),mask,torch.ones((2,8),dtype=torch.bool))
        self.assertEqual(model.parameter_count,249703);self.assertEqual(tuple(t.shape),(2,8,2,25));self.assertEqual(float(t[...,4:].abs().max().detach()),0.);torch.testing.assert_close(torch.linalg.vector_norm(t,dim=-1),torch.ones(2,8,2))
    def test_expected_best_retains_coefficient_gain_gradient(self):
        g=torch.tensor([0.,1.,2.],dtype=torch.float64,requires_grad=True);p=torch.tensor([.2,.3,.5],dtype=torch.float64);v=expected_best(g,p,2);v.backward();self.assertAlmostEqual(float(v.detach()),1.71);self.assertGreater(float(g.grad[-1]),0.)
    def test_phase_matching_and_zero_reference(self):
        torch.set_num_threads(1);b,_,_,_=dense_backend(n=30);s=b.freeze_state(b.get_receiver_basis(4),object_id='x',state_id='0');basis=build_basis(b,s);sm=SmallPhysics(b,s,basis.Q,b.full_J_action(np.eye(4)));labels=[teacher_best(sm,(i,),starts=2,steps=2)['modes'] for i in range(4)];rng=np.random.default_rng(71);a=rng.normal(size=(4,2,sm.rank+1))+1j*rng.normal(size=(4,2,sm.rank+1));a/=np.linalg.norm(a,axis=-1,keepdims=True);t=torch.tensor(a,requires_grad=True);logits=torch.zeros(4,2,requires_grad=True);noop=torch.tensor(0.,requires_grad=True);ref=max([m['gain'] for group in labels for m in group],default=1.)
        x=physical_loss(t,logits,noop,sm,labels,ref,basis.layers);y=physical_loss(t*torch.exp(torch.tensor(.71j)),logits,noop,sm,labels,ref,basis.layers);self.assertTrue(torch.isfinite(x['loss']));torch.testing.assert_close(x['projective'],y['projective']);x['loss'].backward();self.assertTrue(torch.isfinite(t.grad).all())
        zero=physical_loss(t,logits,noop,sm,[[]]*4,0.,basis.layers);zero['loss'].backward();self.assertTrue(zero['reference_zero']);self.assertLess(float(noop.grad),0.)
    def test_normalizer_train_only_and_masks(self):
        rng=np.random.default_rng(5);items=[{'split':'train','object_id':str(i),'xi':np.abs(rng.normal(size=1552)),'outgoing':np.abs(rng.normal(size=(8,32)))} for i in range(3)];norm=Normalizer.fit(items);x,o=norm.transform(items[0]['xi'],items[0]['outgoing']);np.testing.assert_array_equal(x[1434:1536],items[0]['xi'][1434:1536]);np.testing.assert_array_equal(o[:,:16],items[0]['outgoing'][:,:16])
        with self.assertRaises(ValueError):Normalizer.fit([{**items[0],'split':'validation'}])

if __name__=='__main__':unittest.main()
