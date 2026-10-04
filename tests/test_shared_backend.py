import unittest
import numpy as np
from shared_backend.core import (OperatorBackend,ActionSpec,VerifiedExchange,
    pack,unpack,endpoint_from_maps,directional_check,array_hash)

def dense_backend(L=None,n=12):
    rng=np.random.default_rng(1818);p=4;m=3
    L=np.eye(n)+.08*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))) if L is None else L
    n=len(L);S=rng.normal(size=(m,n))+1j*rng.normal(size=(m,n));B=rng.normal(size=(p,4,n))+1j*rng.normal(size=(p,4,n));r=rng.normal(size=(4,m))+1j*rng.normal(size=(4,m))
    inj=lambda z:np.einsum("ptn,p...->nt...",B,z)
    def dual(W,i):return (np.einsum("ptn,nt...->p...",B.conj(),W) if i is None else np.einsum("pn,n...->p...",B[:,i].conj(),W)).real
    def j(z):
        rhs=inj(z);out=np.stack([S@np.linalg.solve(L,rhs[:,i]) for i in range(4)]);return pack(out)
    meta={"chi":np.ones(n//3,complex),"a":np.ones(n//3,complex),"points":np.arange(n).reshape(-1,3),"grid_shape":(n//3,1,1),"cell_volume":1.,"wavenumber":1.,"r":r,"forcing":rng.normal(size=(n,4)),"ell":np.zeros(p),"Lambda":.1*np.eye(p),"state_hash":array_hash(L),"material_gauge_sha256":"gauge","receiver_geometry_sha256":"rx"}
    backend=OperatorBackend(lambda c:L@c,lambda c:L.conj().T@c,lambda c:S@c,lambda w:S.conj().T@w,inj,dual,lambda c:np.einsum("nk,ptn->tkp",c.conj(),B),j,lambda k:np.eye(n,dtype=complex)[:,:k],meta)
    return backend,L,S,B

class SharedTests(unittest.TestCase):
    def test_packing_not_all_real_all_imag(self):
        z=np.arange(24).reshape(4,6)+1j*np.arange(24,48).reshape(4,6)
        np.testing.assert_array_equal(unpack(pack(z),4,6),z)
        self.assertFalse(np.array_equal(pack(z),np.r_[z.real.ravel(),z.imag.ravel()]))
    def test_general_real_endpoint(self):
        b,L,S,B=dense_backend();U=b.get_receiver_basis(4);s=b.freeze_state(U,object_id="x",state_id="0")
        Jc=np.stack([S@U@np.linalg.solve(U.conj().T@L@U,U.conj().T@B[:,i].T) for i in range(4)])
        J=pack(Jc);expected=-np.linalg.solve(J.T@J+s.Lambda,J.T@pack(s.r)+s.ell)
        np.testing.assert_allclose(s.step_old,expected,rtol=1e-12,atol=1e-12)
    def test_material_adjoint(self):
        b,_,_,_=dense_backend();rng=np.random.default_rng(3);z=rng.normal(size=4);w=rng.normal(size=(12,4))+1j*rng.normal(size=(12,4))
        self.assertAlmostEqual(np.vdot(b.apply_B(z),w).real,z@b.material_dual(w),places=10)
    def test_online_is_readonly_no_teacher(self):
        b,_,_,_=dense_backend();s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0")
        with self.assertRaises(ValueError):s.U[0,0]=0
        for name in ("full_J","full_H","truth","teacher","anchor64","Ar_inv"):self.assertFalse(hasattr(s,name))
    def test_selected_only_and_invalidated_cache(self):
        b,_,_,_=dense_backend();s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0");calls=[]
        recipes={str(i):lambda i=i:(calls.append(i) or np.eye(12,dtype=complex)[:,4+i]) for i in range(6)}
        v=VerifiedExchange(b,s,recipes);a=ActionSpec("A","chosen",s.configuration_id,(3,),2.,"2");before=b.receipt.snapshot();result=v.evaluate(a)
        self.assertEqual(result["status"],"OK");self.assertEqual(calls,[2]);self.assertEqual(b.receipt.delta(before)["counters"]["current_materializations"],1)
        newer=b.freeze_state(result["new_U"],object_id="x",state_id="0");self.assertNotEqual(newer.configuration_id,s.configuration_id)
        self.assertEqual(VerifiedExchange(b,newer,recipes).evaluate(a)["status"],"INVALID_ENDPOINT")
    def test_singular_core_stable_direct_endpoint(self):
        L=np.eye(6,dtype=complex);L[:2,:2]=[[0,1],[1,0]];b,_,_,_=dense_backend(L);s=b.freeze_state(np.eye(6,dtype=complex)[:,:2],object_id="x",state_id="0")
        a=ActionSpec("A","direct",s.configuration_id,(1,),0.,"one");result=VerifiedExchange(b,s,{"one":lambda:np.eye(6,dtype=complex)[:,1]}).evaluate(a)
        self.assertEqual(result["status"],"OK");self.assertTrue(result["singular_core_fallback"])
    def test_zero_seeds_inactive(self):
        b,_,_,_=dense_backend();b._S=lambda c:np.zeros((3,c.shape[1]),complex);b._SH=lambda w:np.zeros((12,w.shape[1]),complex);b._B=lambda z:np.zeros((12,4,z.shape[1]),complex);b.metadata["r"]*=0;b.metadata["forcing"]*=0
        s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0");seed=b.get_opm_seeds(s)
        self.assertFalse(seed.active.any());self.assertEqual(np.linalg.norm(seed.vectors),0.)
    def test_finite_gain_identity(self):
        b,_,_,_=dense_backend();s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0");x=s.step_old;d=np.arange(4)*.001;u=pack(s.r)+b.full_J_action(x);z=b.full_J_action(d)
        g=directional_check(pack(s.r),u,z,x,d,s.ell,s.Lambda)
        phi=lambda w:.5*np.linalg.norm(pack(s.r)+b.full_J_action(w))**2+s.ell@w+.5*w@s.Lambda@w
        self.assertAlmostEqual(g["gain"],phi(x)-phi(x+d),places=9)
    def test_coefficient_gauge_and_phase(self):
        b,_,_,_=dense_backend();s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0");Q=np.eye(12,dtype=complex)[:,4:6];h=b.register_basis(s,Q);t=np.array([.1,.4,.7],complex);t/=np.linalg.norm(t)
        action=lambda ts,hs:ActionSpec("B",hs,s.configuration_id,(3,),1.,coefficient_real=tuple(ts.real),coefficient_imag=tuple(ts.imag),basis_gauge_sha256=hs)
        r1=VerifiedExchange(b,s).evaluate(action(t,h));phase=np.exp(.3j);h2=b.register_basis(s,Q*phase);t2=t.copy();t2[1:]/=phase;r2=VerifiedExchange(b,s).evaluate(action(t2,h2))
        self.assertAlmostEqual(r1["gain"],r2["gain"],places=9)
    def test_cache_hash_depends_on_residual_and_prior(self):
        b,_,_,_=dense_backend();U=b.get_receiver_basis(4);s=b.freeze_state(U,object_id="x",state_id="0");b.metadata["r"]*=2
        s2=b.freeze_state(U,object_id="x",state_id="0");self.assertNotEqual(s.configuration_id,s2.configuration_id)
        b.metadata["ell"]=np.ones(4);s3=b.freeze_state(U,object_id="x",state_id="0");self.assertNotEqual(s2.state_sha256,s3.state_sha256)
    def test_known_source_phase_canonicalized_seeds(self):
        b,_,_,_=dense_backend();U=b.get_receiver_basis(4);s=b.freeze_state(U,object_id="x",state_id="0");seed=b.get_opm_seeds(s);ph=np.exp(1j*np.array([.2,.5,.8,1.1]));old=b._B
        b.metadata["r"]*=ph[:,None];b.metadata["forcing"]=b.metadata["forcing"]*ph[None,:];b.metadata["source_phase"]=ph;b._B=lambda z:old(z)*ph[None,:,None]
        s2=b.freeze_state(U,object_id="x",state_id="0");seed2=b.get_opm_seeds(s2);np.testing.assert_allclose(seed.vectors,seed2.vectors,atol=1e-12)
    def test_sketch_stable_channel_order(self):
        from dataclasses import replace
        b,_,_,_=dense_backend();s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0");x=b.shared_state_features(s)[0];order=[2,0,1]
        s2=replace(s,r=s.r[:,order],SU=s.SU[order],geometry={"channel_ids":tuple(order)})
        x2=b.shared_state_features(s2)[0];np.testing.assert_allclose(x[64:128],x2[64:128],atol=1e-12)
    def test_accept_reuses_paid_maps_and_preserves_protection(self):
        from dataclasses import replace
        b,_,_,_=dense_backend();s=b.freeze_state(b.get_receiver_basis(4),object_id="x",state_id="0",protected_slots=(0,2));v=VerifiedExchange(b,s,{"new":lambda:np.eye(12,dtype=complex)[:,5]})
        a=ActionSpec("A","new",s.configuration_id,(1,),0.,"new");r=v.evaluate(a);r["accepted"]=True # Contract plumbing, not scientific acceptance.
        before=b.receipt.snapshot();new=v.accept(r);delta=b.receipt.delta(before)["counters"]
        self.assertEqual(delta.get("L_rhs",0),0);self.assertEqual(new.protected_slots,(0,1));self.assertEqual(new.receiver_mode_ids,(0,2,3,-1));self.assertNotEqual(s.configuration_id,new.configuration_id)
        np.testing.assert_allclose(v._u,pack(new.r)+b.full_J_action(new.step_old),atol=1e-12)

if __name__=="__main__":unittest.main()
