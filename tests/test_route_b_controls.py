import unittest
import numpy as np
from scipy import linalg as la
from test_shared_backend import dense_backend
from route_b.basis import build_basis
from route_b.small_physics import SmallPhysics
from route_b.controls import normalize,currents_from_projector,positive_tau_bound
from shared_backend.core import pack,directional_check

class ControlTests(unittest.TestCase):
    def test_normalizer_does_not_fit_held_out_values(self):
        rng=np.random.default_rng(42);train=rng.normal(size=(8,1584));test=rng.normal(size=(2,1584));a,b=normalize(train,test)
        aa,bb=normalize(train,1000*test)
        np.testing.assert_array_equal(a,aa)
        np.testing.assert_allclose(a[:,1568:].mean(0),0,atol=1e-13)
        np.testing.assert_array_equal(b[:,:1536][:,1434:],test[:,:1536][:,1434:])
    def test_projector_padding_and_phase(self):
        t=np.zeros(25,complex);t[1:3]=[1,1j];t/=la.norm(t);P=np.outer(t,t.conj());P[24,24]=100
        modes=currents_from_projector(np.r_[P.real.ravel(),P.imag.ravel()],np.r_[np.ones(3),np.zeros(22)],1)
        self.assertEqual(len(modes),1);self.assertAlmostEqual(abs(np.vdot(modes[0][0],t)),1)
        self.assertEqual(modes[0][0][24],0)
    def test_conservative_tau_bounds_positive_original_tau(self):
        b,_,_,_=dense_backend(n=30);s=b.freeze_state(b.get_receiver_basis(4),object_id='x',state_id='0');basis=build_basis(b,s);J=b.full_J_action(np.eye(4));cache=SmallPhysics(b,s,basis.Q,J);rng=np.random.default_rng(81);seen=0
        for _ in range(50):
            t=rng.normal(size=(cache.rank+1,))+1j*rng.normal(size=(cache.rank+1,));t/=la.norm(t)
            import torch
            with torch.no_grad():g,v,x,_=cache.evaluate(torch.as_tensor(t[None]),(0,))
            if v[0] and g[0]>0:
                d=x[0].numpy()-s.step_old;check=directional_check(pack(s.r),pack(s.r)+J@s.step_old,J@d,s.step_old,d,s.ell,s.Lambda)
                self.assertGreaterEqual(positive_tau_bound(cache),check['tau']);seen+=1
        self.assertGreater(seen,0)

if __name__=='__main__':unittest.main()
