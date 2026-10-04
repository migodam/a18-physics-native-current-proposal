"""Original A17 anchor/pool generation with explicit real-material adapter.

This retains the original generator and analytic finite-gain ranking. It does
not reuse the historical A18-B anchor or closure. The 64-column anchor, primal
solve, randomized GD bank, and every projection are paid in this backend.
"""
from types import SimpleNamespace
import time
import numpy as np
from scipy import linalg as la
from .core import endpoint_from_maps,pack,unpack
from .legacy import pinned_a17

class Reduced:
    def __init__(self,ctx,U):
        self.ctx,self.U=ctx,U;self.LU=ctx.apply_L(U);self.SU=ctx.b.apply_S(U);self.D=U.conj().T@self.LU;self.N=ctx.b.project_B(U)
        ep=endpoint_from_maps(self.D,self.SU,self.N,ctx.state.r,ctx.ell,ctx.state.Lambda);self.J=ep["J"];self.C=ep["coefficients"];self.H=self.J.T@self.J+ctx.state.Lambda
    def solve_H(self,h):return la.cho_solve(la.cho_factor(self.H,lower=True),h)
    def step(self):return -self.solve_H(self.J.T@self.ctx.r+self.ctx.ell)
    def j(self,s):return self.J@s
    def jt(self,w):return self.J.T@w
    def current_jvp(self,s):return np.einsum("nk,pkd,d...->pn...",self.U,self.C,s,optimize=True)
    def R_adjoint(self,W):return self.U@la.solve(self.D.conj().T,self.U.conj().T@W)

class GDAction:
    __array_priority__=10000
    def __init__(self,b,adj=False):self.b,self.adj=b,adj
    def __matmul__(self,X):return self.b.apply_GD(X,self.adj)
    def __rmatmul__(self,X):return self.b.apply_GD(X.conj().T,not self.adj).conj().T
    def conj(self):return self
    @property
    def T(self):return GDAction(self.b,not self.adj)

class Context:
    def __init__(self,b,state):
        self.b,self.state=b,state;self.n_current=state.U.shape[0];self.P=state.r.shape[0];self.m=state.r.shape[1];self.dim_material=len(state.ell);self.q=self.dim_material
        self.points=state.points.reshape(-1,3);self.r=pack(state.r);self.ell=state.ell;self.lam=state.lambda_scalar
        self.ledger=SimpleNamespace(add=lambda name,count=1:b.receipt.add(name,count) if isinstance(count,int) else b.receipt.add(name,0,count))
        self.current_forward=b.solve_current(state.forcing).T;self.internal_cache={}
    def records(self):return self.b.receipt.snapshot()
    def apply_L(self,C):return self.b.apply_L(C)
    def receiver_modes(self,count):return self.b.get_receiver_basis(count)
    def observe_adjoint(self,W):return self.b.apply_S_adjoint(W)
    def make_model(self,U):return Reduced(self,U)
    def B(self,Z):return self.b.apply_B(Z).transpose(1,0,2)
    def internal_modes(self,count,seed):
        from cair import domain_bank
        key=(count,seed)
        if key not in self.internal_cache:
            self.internal_cache[key]=domain_bank(SimpleNamespace(n=self.n_current,GD=GDAction(self.b)),count,seed)
        return self.internal_cache[key]

def prepare_dictionary(backend,state,a17_root):
    pinned_a17(a17_root)
    from selectors_current import build_independent_anchor,build_candidates,SelectorConfig
    start=time.perf_counter();before=backend.receipt.snapshot();ctx=Context(backend,state)
    cfg=SelectorConfig(seed=20260930,pool_count=32,residual_up_count=16,random_pool_count=10)
    anchor,meta=build_independent_anchor(ctx,64,seed=cfg.seed);pool=build_candidates(ctx,anchor,cfg)
    backend.receipt.add("current_materializations",pool.vectors.shape[1]);backend.receipt.add("dictionary_builds")
    return {"vectors":pool.vectors,"labels":pool.labels,"families":pool.families,"anchor_J":anchor.J,"anchor":meta,"wall_seconds":time.perf_counter()-start,"cost":backend.receipt.delta(before),"original_generator":"A17 pinned selectors_current.build_independent_anchor/build_candidates","actual_pool_columns":pool.vectors.shape[1],"historical_anchor_loaded":False}
