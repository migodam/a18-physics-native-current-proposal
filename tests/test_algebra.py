"""Small dense algebra tests only. No Maxwell empirical claim is made."""
from __future__ import annotations
import itertools
import unittest
import numpy as np
from src.algebra import (
    SchurFeedback, adj, project, orth_columns, galerkin_complex,
    realify_jacobians, realify_data, frozen_step, phi, finite_gain,
    compressed_endpoint, joint_flag, order_penalty, projective_distance,
    expected_best_gain,
)


class AlgebraTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1804)
        self.n, self.k, self.m, self.p = 22, 4, 7, 5
        r = self.rng
        def cx(shape): return (r.normal(size=shape)+1j*r.normal(size=shape))/np.sqrt(2)
        self.cx = cx
        F = np.triu(cx((self.n, self.n))) * 0.08
        self.L = np.eye(self.n) - F
        self.U = np.linalg.qr(cx((self.n, self.k)))[0]
        self.S = cx((self.m, self.n))
        self.B = cx((self.n, self.p))
        self.fb = SchurFeedback(self.L, self.U)
        self.C = orth_columns(cx((self.n, 5)), self.U)
        self.eye = np.eye(self.n, dtype=complex)

    def close(self, a, b, tol=2e-10):
        err = np.linalg.norm(a-b) / max(1.0, np.linalg.norm(a), np.linalg.norm(b))
        self.assertLess(err, tol)

    def test_schur_formula_and_adjoint(self):
        F = self.eye-self.L
        P = self.eye-self.U@adj(self.U)
        expected = P@F@P+P@F@self.U@np.linalg.solve(self.fb.D, adj(self.U)@F@P)
        self.close(self.fb.F(self.eye), expected)
        self.close(self.fb.A_adj(self.eye), adj(self.fb.A(self.eye)))
        for _ in range(16):
            x, y = self.cx((self.n,)), self.cx((self.n,))
            self.close(np.vdot(x,self.fb.F(y)), np.vdot(self.fb.F_adj(x),y))

    def test_full_input_output_elimination(self):
        E = orth_columns(self.eye, self.U)
        A = adj(E)@self.fb.A(E)
        base = self.S@self.fb.R(self.B)
        correction = self.S@self.fb.dress(E)@np.linalg.solve(A, adj(E)@self.fb.effective_injection(self.B))
        self.close(base+correction, self.S@np.linalg.solve(self.L,self.B))

    def test_increment_and_retained_equations(self):
        C = self.C
        cb = self.fb.dress(C)
        gamma = adj(C)@self.L@cb
        N = adj(C)@(self.B-self.L@self.fb.R(self.B))
        delta = self.S@cb@np.linalg.solve(gamma,N)
        old = galerkin_complex(self.L,self.S,self.B,self.U)
        new = galerkin_complex(self.L,self.S,self.B,np.column_stack([self.U,C]))
        self.close(new-old, delta)
        self.close(adj(self.U)@self.L@cb, np.zeros((self.k,C.shape[1])))

    def test_phase_and_block_gauge_invariance(self):
        W = np.column_stack([self.U,self.C])
        rotation = np.linalg.qr(self.cx((W.shape[1],W.shape[1])))[0]
        J = galerkin_complex(self.L,self.S,self.B,W)
        self.close(J,galerkin_complex(self.L,self.S,self.B,W@rotation))
        phase = np.exp(1j*np.arange(W.shape[1]))
        self.close(J,galerkin_complex(self.L,self.S,self.B,W*phase))

    def test_compressed_teacher_endpoint(self):
        Z = np.column_stack([self.U,self.C])
        E = np.linalg.qr(self.cx((Z.shape[1],self.k)))[0]
        direct = galerkin_complex(self.L,self.S,self.B,Z@E)
        cached = compressed_endpoint(adj(Z)@self.L@Z,self.S@Z,adj(Z)@self.B,E)
        self.close(direct,cached)

    def test_multiple_illuminations_real_material(self):
        Bs = [self.B,self.cx(self.B.shape),self.cx(self.B.shape)]
        Js = [self.S@np.linalg.solve(self.L,b) for b in Bs]
        J = realify_jacobians(Js)
        s = self.rng.normal(size=self.p)
        y = realify_data([j@s for j in Js])
        self.close(J@s,y)
        residuals = [self.cx((self.m,)) for _ in Js]
        real_gradient = J.T@realify_data(residuals)
        complex_formula = sum((adj(j)@r).real for j,r in zip(Js,residuals))
        self.close(real_gradient,complex_formula)

    def test_finite_gain_identity_and_cache_update(self):
        J = realify_jacobians([self.S@np.linalg.solve(self.L,self.B)])
        r = self.rng.normal(size=J.shape[0])
        lam, ell = np.eye(self.p)*0.2,self.rng.normal(size=self.p)*0.1
        x,s = self.rng.normal(size=self.p),self.rng.normal(size=self.p)
        G = finite_gain(J,r,lam,ell,x,s)
        self.close(G,phi(J,r,lam,ell,x)-phi(J,r,lam,ell,s))
        self.close(r+J@s,(r+J@x)+J@(s-x))
        full = frozen_step(J,r,lam,ell)
        self.assertGreaterEqual(finite_gain(J,r,lam,ell,x,full),-1e-10)

    def test_metric_whitening(self):
        weights = np.exp(self.rng.normal(size=self.n))
        Wm = np.diag(weights); root=np.diag(np.sqrt(weights)); inv=np.diag(1/np.sqrt(weights))
        Vphys = inv@self.U
        Jphys = self.S@Vphys@np.linalg.solve(adj(Vphys)@Wm@self.L@Vphys,adj(Vphys)@Wm@self.B)
        Jwhite = galerkin_complex(root@self.L@inv,self.S@inv,root@self.B,self.U)
        self.close(Jphys,Jwhite)

    def test_nonnormal_eigenvalues_do_not_bound_action(self):
        F = np.array([[0,20],[0,0]],complex)
        self.assertAlmostEqual(max(abs(np.linalg.eigvals(F))),0.0)
        self.assertAlmostEqual(np.linalg.norm(F,2),20.0)
        self.close(F@F,np.zeros((2,2)))
        self.assertAlmostEqual(np.linalg.norm(np.eye(2)+F,2)>19,True)

    def test_affine_krylov_equivalence(self):
        b = self.cx((self.n,))
        F = np.eye(self.n)-self.L
        def basis(A):
            cols=[b]
            for _ in range(3): cols.append(A@cols[-1])
            return orth_columns(np.column_stack(cols))
        Q,R = basis(F),basis(self.L)
        self.close(Q@adj(Q),R@adj(R),tol=1e-8)

    def test_joint_flag_orthogonality_and_layer_penalty(self):
        seeds = [('O',self.cx((self.n,))),('P',self.cx((self.n,))),('M',self.cx((self.n,)))]
        Q,layers = joint_flag(self.fb,seeds,depth=2)
        self.close(adj(Q)@Q,np.eye(Q.shape[1]))
        self.close(adj(self.U)@Q,np.zeros((self.k,Q.shape[1])))
        a = self.cx((Q.shape[1],)); a/=np.linalg.norm(a)
        W = np.eye(Q.shape[1],dtype=complex)
        for d in set(layers):
            ids = np.flatnonzero(np.array(layers)==d)
            W[np.ix_(ids,ids)] = np.linalg.qr(self.cx((len(ids),len(ids))))[0]
        self.close(order_penalty(a,layers),order_penalty(adj(W)@a,layers))
        c=Q@a
        tail=0.0
        for d in [1,2]:
            low=Q[:,np.array(layers)<d]
            tail += (((d+1)**2-1)-(d**2-1))*np.linalg.norm(project(low,c))**2
        self.close(tail,order_penalty(a,layers))

    def test_dark_observation_retained_loop(self):
        F=np.array([[.2,.6],[.4,.1]],complex); L=np.eye(2)-F
        U=np.eye(2,dtype=complex)[:,:1]; c=np.array([0,1],complex); S=np.array([[1,0]],complex)
        fb=SchurFeedback(L,U)
        self.close(S@c,np.array([0]))
        self.close(S@fb.dress(c),np.array([.75]))
        self.close(fb.F(c)[1],.4)
        self.close(project(U,F@c)[1],.1)
        self.assertGreater(np.linalg.norm(fb.observation_seed(S,np.array([1],complex))),0.7)

    def test_singular_common_core_valid_endpoints(self):
        L=np.array([[0,1,1],[1,1,0],[1,0,2]],complex)
        V=np.eye(3,dtype=complex)[:,:1]
        with self.assertRaises(np.linalg.LinAlgError): SchurFeedback(L,V)
        S=np.array([[1,2,3]],complex); B=np.ones((3,1),complex)
        old=np.eye(3,dtype=complex)[:,[0,1]]
        new=np.eye(3,dtype=complex)[:,[0,2]]
        self.assertTrue(np.all(np.isfinite(galerkin_complex(L,S,B,old))))
        self.assertTrue(np.all(np.isfinite(galerkin_complex(L,S,B,new))))

    def test_leave_out_lift_keeps_partial_rotation(self):
        i=1; V=np.delete(self.U,i,axis=1)
        Z=np.column_stack([self.U[:,i],self.C])
        t=self.cx((Z.shape[1],)); t/=np.linalg.norm(t); c=Z@t
        self.close(adj(V)@c,np.zeros(V.shape[1]))
        self.close(np.linalg.norm(c),1.0)
        self.assertGreater(abs(np.vdot(self.U[:,i],c)),1e-3)
        self.close(projective_distance(c,c*np.exp(1j*.731)),0.0)

    def test_negative_singletons_positive_pair(self):
        # Even L=I can have non-additive endpoint utility: no modal-mixture claim.
        L=np.eye(5,dtype=complex); S=np.ones((1,5),complex)
        B=np.array([.25,.25,5.25,-4.25,-.5],complex)[:,None]
        basis=np.eye(5,dtype=complex)
        Jfull=realify_jacobians([S@B]); r=np.array([1.,0.]); lam=np.eye(1); ell=np.zeros(1)
        def step(ids):
            J=realify_jacobians([galerkin_complex(L,S,B,basis[:,ids])])
            return frozen_step(J,r,lam,ell)
        old=step([0,1])
        gains=[finite_gain(Jfull,r,lam,ell,old,step(ids)) for ids in ([2,1],[0,3],[2,3])]
        self.assertLess(gains[0],0); self.assertLess(gains[1],0); self.assertGreater(gains[2],0)
        self.assertAlmostEqual(gains[2],.01,places=12)

    def test_expected_coverage_surrogate_exact_iid(self):
        g=np.array([0.,.2,.8]); p=np.array([.2,.3,.5]); k=3
        exact=0.0
        for inds in itertools.product(range(3),repeat=k):
            exact += np.prod(p[list(inds)])*max(g[list(inds)])
        self.close(expected_best_gain(g,p,k),exact)
        with self.assertRaises(ValueError): expected_best_gain(g,p*2,k)

    def test_zero_seed_is_not_randomly_replaced(self):
        Q,layers=joint_flag(self.fb,[('O',np.zeros(self.n,complex))],2)
        self.assertEqual(Q.shape,(self.n,0)); self.assertEqual(layers,[])


if __name__=='__main__': unittest.main(verbosity=2)
