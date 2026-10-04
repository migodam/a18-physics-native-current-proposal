import unittest
import numpy as np
from scipy import linalg as la
from test_shared_backend import dense_backend
from route_b.basis import Root,build_basis,feature_vector
from route_b.policy import select_top4
from route_b.small_physics import SmallPhysics,teacher_best
from shared_backend.core import pack,endpoint_from_maps,directional_check

class RouteBTests(unittest.TestCase):
    def state(self):
        b,L,S,B=dense_backend(n=30);s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0");return b,s,L
    def test_schur_and_adjoint_real_actions(self):
        b,s,L=self.state();root=Root(b,s);P=np.eye(30)-s.U@s.U.conj().T;F=P-P@(L-s.LU@la.solve(s.D,s.U.conj().T@L))@P
        rng=np.random.default_rng(2);q=rng.normal(size=30)+1j*rng.normal(size=30)
        before=b.receipt.snapshot();np.testing.assert_allclose(root.action(q),F@q,rtol=1e-12,atol=1e-12)
        self.assertEqual(b.receipt.delta(before)["counters"]["L_rhs"],1)
        before=b.receipt.snapshot();np.testing.assert_allclose(root.action(q,True),F.conj().T@q,rtol=1e-12,atol=1e-12)
        self.assertEqual(b.receipt.delta(before)["counters"]["L_adjoint_rhs"],1)
    def test_joint_degree_and_no_extra_H_action(self):
        b,s,_=self.state();basis=build_basis(b,s);Q=basis.Q
        np.testing.assert_allclose(Q.conj().T@Q,np.eye(Q.shape[1]),atol=1e-12);np.testing.assert_allclose(s.U.conj().T@Q,0,atol=1e-12)
        self.assertTrue(np.all(np.diff(basis.layers)>=0));self.assertLessEqual(Q.shape[1],18)
        counts=basis.receipt["counters"];self.assertEqual(counts["L_rhs"],8);self.assertEqual(counts["L_adjoint_rhs"],6)
        before=b.receipt.snapshot();self.assertEqual(feature_vector(b,s,basis)[0].shape,(1552,));self.assertEqual(b.receipt.delta(before)["counters"].get("L_rhs",0),0)
    def test_raw_variant(self):
        b,s,L=self.state();root=Root(b,s,False);P=np.eye(30)-s.U@s.U.conj().T;q=np.arange(30)
        np.testing.assert_allclose(root.action(q),P@(np.eye(30)-L)@P@q,atol=1e-12)
    def test_zero_active_does_not_fill(self):
        b,s,_=self.state();seed=b.get_opm_seeds(s);object.__setattr__(seed,"active",np.zeros(6,bool));basis=build_basis(b,s);self.assertEqual(basis.Q.shape[1],0)
    def test_noop_and_duplicates(self):
        b,s,_=self.state();basis=build_basis(b,s);c=np.zeros((4,2,25),complex);c[:,:,1]=1;logits=np.ones((4,2))
        self.assertEqual(select_top4(logits,2,c,s,basis),[])
        records=select_top4(logits,0,c,s,basis);self.assertEqual(len(records),4);self.assertEqual([r.drop_indices[0] for r in records],[0,1,2,3])
    def test_small_cache_matches_real_endpoint(self):
        import torch
        b,s,_=self.state();basis=build_basis(b,s);J=b.full_J_action(np.eye(4));cache=SmallPhysics(b,s,basis.Q,J);rng=np.random.default_rng(4);t=rng.normal(size=1+cache.rank)+1j*rng.normal(size=1+cache.rank);t/=la.norm(t)
        gain,valid=cache.numpy_gain(t,2);W=np.column_stack((np.delete(s.U,2,axis=1),np.column_stack((s.U[:,2],basis.Q))@t))
        ep=endpoint_from_maps(W.conj().T@b.apply_L(W),b.apply_S(W),b.project_B(W),s.r,s.ell,s.Lambda);d=ep["step"]-s.step_old;g=directional_check(pack(s.r),pack(s.r)+J@s.step_old,J@d,s.step_old,d,s.ell,s.Lambda)
        self.assertTrue(valid);self.assertAlmostEqual(gain,g["gain"],places=9)
    def test_rank_two_is_joint_endpoint(self):
        import torch
        b,s,_=self.state();basis=build_basis(b,s);cache=SmallPhysics(b,s,basis.Q,b.full_J_action(np.eye(4)));rng=np.random.default_rng(9);T=la.qr(rng.normal(size=(cache.rank+2,2))+1j*rng.normal(size=(cache.rank+2,2)),mode="economic")[0]
        gain,valid,step,_=cache.evaluate(torch.as_tensor(T[None]),(0,1));W=np.column_stack((s.U[:,2:],np.column_stack((s.U[:,:2],basis.Q))@T));ep=endpoint_from_maps(W.conj().T@b.apply_L(W),b.apply_S(W),b.project_B(W),s.r,s.ell,s.Lambda)
        self.assertTrue(bool(valid[0]));np.testing.assert_allclose(step[0].detach().numpy(),ep["step"],atol=1e-10)
    def test_teacher_is_bounded_best_found(self):
        import torch
        torch.set_num_threads(1);b,s,_=self.state();basis=build_basis(b,s);cache=SmallPhysics(b,s,basis.Q,b.full_J_action(np.eye(4)));r=teacher_best(cache,(0,),starts=2,steps=2)
        self.assertEqual(r["steps"],2);self.assertFalse(r["global_optimality"])

if __name__=="__main__":unittest.main()
