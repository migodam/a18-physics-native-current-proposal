from dataclasses import dataclass
import time
import numpy as np
from scipy import linalg as la
from shared_backend.core import stability,array_hash

def phase(q):
    return np.exp(-1j*np.angle(q[np.argmax(abs(q))])) if np.any(q) else 1+0j

class Root:
    """Cached LU; every forward/adjoint action has exactly one large L/L*."""
    def __init__(self,backend,state,schur=True):
        self.b,self.s,self.schur=backend,state,schur
        if not stability(state.D)["stable"]:raise ValueError("ROOT_UNSTABLE")
        self.factor=la.lu_factor(state.D)
    def P(self,q):return q-self.s.U@(self.s.U.conj().T@q)
    def action(self,q,adjoint=False):
        s=self.s;q=self.P(q)
        if adjoint:
            if self.schur:q2=q-s.U@la.lu_solve(self.factor,s.LU.conj().T@q,trans=2)
            else:q2=q
            A=self.P(self.b.apply_L_adjoint(q2))
        else:
            lq=self.b.apply_L(q)
            A=self.P(lq-s.LU@la.lu_solve(self.factor,s.U.conj().T@lq)) if self.schur else self.P(lq)
        return q-A

@dataclass
class Basis:
    Q: np.ndarray
    layers: np.ndarray
    chain_columns: np.ndarray
    H: np.ndarray
    action_norms: np.ndarray
    residual_ratios: np.ndarray
    seed_bundle: object
    basis_gauge_sha256: str
    configuration_id: str
    degree: int
    schur: bool
    side: str
    receipt: dict

def build_basis(backend,state,degree=2,schur=True,side="joint",joint_against=None):
    if degree not in (0,1,2) or side not in ("joint","O","P","M"):raise ValueError("basis contract")
    before=backend.receipt.snapshot();t=time.perf_counter();seed=backend.get_opm_seeds(state);root=Root(backend,state,schur)
    n=state.U.shape[0];chain=np.zeros((n,18),complex);H=np.zeros((6,3,2),complex);an=np.zeros((6,2));ratios=[]
    for i in range(6):
        if not seed.active[i] or side!="joint" and i//2!={"O":0,"P":1,"M":2}[side]:continue
        chain[:,3*i]=seed.vectors[:,i];local=[chain[:,3*i]]
        for d in range(degree):
            f=root.action(local[-1],adjoint=i<2);an[i,d]=la.norm(f);v=f.copy()
            for _ in range(2):
                for j,q in enumerate(local):
                    h=np.vdot(q,v);H[i,j,d]+=h;v-=q*h
            v=root.P(v);h=la.norm(v);ratio=h/max(an[i,d],1e-300);ratios.append(ratio)
            if ratio<=1e-12:break
            ph=phase(v);q=v/h*ph;H[i,d+1,d]=h/ph;local.append(q);chain[:,3*i+d+1]=q
    against=state.U if joint_against is None else joint_against
    Q=[];layers=[]
    for d in range(degree+1):
        for i in range(6):
            v=chain[:,3*i+d].copy();norm=la.norm(v)
            if norm==0:continue
            for _ in range(2):
                v-=against@(against.conj().T@v)
                for q in Q:v-=q*np.vdot(q,v)
            h=la.norm(v)
            if h<=1e-12*norm:continue
            Q.append(v/h*phase(v));layers.append(d)
            if len(Q)>24:raise ValueError("rank hard cap24 exceeded")
    Q=np.column_stack(Q) if Q else np.empty((n,0),complex)
    gauge=backend.register_basis(state,Q)
    receipt={"origin":"route_b.root_basis","dependencies":["current_U","current_LU","current_D","current_opm_seeds"],"exclusive_seconds":time.perf_counter()-t,**backend.receipt.delta(before)}
    return Basis(Q,np.asarray(layers,int),chain,H,an,np.asarray(ratios),seed,gauge,state.configuration_id,degree,schur,side,receipt)

def feature_vector(backend,state,basis):
    if basis.configuration_id!=state.configuration_id:raise ValueError("STALE_CONFIGURATION")
    xi,receipts=backend.shared_state_features(state);r=basis.Q.shape[1];ct=np.zeros((24,18),complex);sc=np.zeros((24,6),complex)
    ct[:r]=basis.Q.conj().T@basis.chain_columns
    sc[:r]=basis.Q.conj().T@(basis.seed_bundle.vectors*basis.seed_bundle.norms)
    mask=np.r_[np.ones(r),np.zeros(24-r)];one=np.zeros((24,3));one[np.arange(r),basis.layers]=1
    seed=basis.seed_bundle;rat=np.divide(seed.norms,seed.source_scales,out=np.zeros(6),where=seed.source_scales>0)
    diagnostics=np.r_[r,[np.sum(basis.layers==i) for i in range(3)],rat,la.norm(basis.Q.conj().T@basis.Q-np.eye(r)),la.norm(state.U.conj().T@basis.Q),min(basis.residual_ratios,default=0.),np.max(basis.action_norms),stability(state.D)["relative"],(24-r)/24]
    complex_flat=lambda z:np.r_[z.real.ravel(),z.imag.ravel()]
    x=np.r_[xi,complex_flat(basis.H),complex_flat(ct),complex_flat(sc),np.log1p(seed.norms),np.log1p(basis.action_norms.ravel()),mask,one.ravel(),seed.active.astype(float),diagnostics]
    if x.shape!=(1552,) or not np.isfinite(x).all():raise ValueError("1552 feature contract")
    return x,(*receipts,basis.receipt)
