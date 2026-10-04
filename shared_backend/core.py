from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable, Callable
from hashlib import sha256
import json
import time
import numpy as np
from scipy import linalg as la

TOL = 1e-10
SEED_SCHEMA = "opm-six-current-u-v1"

def array_hash(a):
    a = np.ascontiguousarray(a)
    return sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()

def readonly(a, dtype=None):
    a = np.array(a, dtype=dtype, copy=True); a.setflags(write=False); return a

def pack(z):
    """Complex (P,m,...) to real illumination-major [Re(m),Im(m)]."""
    z = np.asarray(z)
    if z.ndim < 2: raise ValueError("packing requires illumination and channel axes")
    return np.concatenate((z.real, z.imag), axis=1).reshape((-1,) + z.shape[2:])

def unpack(r, P, m):
    r = np.asarray(r)
    if np.iscomplexobj(r) or r.shape[0] != 2*P*m: raise ValueError("real data packing")
    a = r.reshape((P, 2*m) + r.shape[1:]); return a[:, :m] + 1j*a[:, m:]

def stability(D):
    s = la.svdvals(D)
    ratio = float(s[-1]/s[0]) if len(s) and s[0] else (1. if not len(s) else 0.)
    return {"stable": ratio > TOL, "relative": ratio,
            "sigma_min": float(s[-1]) if len(s) else None,
            "sigma_max": float(s[0]) if len(s) else None}

def endpoint_from_maps(D, SU, NB, r, ell, Lambda):
    """Exact endpoint with arbitrary shared REAL material injection (P,k,p)."""
    info = stability(D)
    if not info["stable"]: raise ValueError("CURRENT_NEAR_SINGULAR; no jitter")
    P, k, p = NB.shape
    C = la.solve(D, NB.transpose(1,0,2).reshape(k,-1)).reshape(k,P,p).transpose(1,0,2) if k else np.zeros((P,0,p),complex)
    J = pack(np.einsum("mk,pkd->pmd", SU, C, optimize=True))
    H = J.T@J + Lambda; h = J.T@pack(r) + ell
    s = -la.cho_solve(la.cho_factor(H, lower=True), h)
    nr = float(la.norm(H@s+h)/max(la.norm(h), np.finfo(float).tiny))
    if not np.isfinite(nr) or nr > TOL: raise ValueError("NORMAL_RESIDUAL_FAILED")
    return {"step": s, "J": J, "Jstep": J@s, "stability": info,"reduced_gradient_norm":float(la.norm(H@s+h)),
            "normal_relative_residual": nr, "coefficients": C}

def directional_check(r,u,z,x,d,ell,Lambda):
    """A17 SAFETY=32 gamma_N significance rule, generalized symmetric prior.

    Scalar lambda is exactly the original rule; this is not an output-error
    certificate. Current solve residuals are tested independently.
    """
    if any(np.iscomplexobj(v) for v in (r,x,d,ell)): raise ValueError("real packing")
    n = len(r)+3*len(x)+16; f=np.finfo(np.float64); gamma=n*f.eps/(1-n*f.eps)
    obs=float(np.vdot(u,z).real); pl=float((ell+Lambda@x)@d)
    dc=.5*float(np.vdot(z,z).real); pc=.5*float(d@Lambda@d)
    g=-obs-pl-dc-pc
    terms=float(np.abs(u)@np.abs(z)+np.abs(ell+Lambda@x)@np.abs(d)+dc+pc)
    base=float(np.vdot(r,r).real+np.vdot(u,u).real+abs(ell@x)+x@Lambda@x)
    scale=terms+base; tau=32*f.tiny+32*gamma*scale
    return {"gain":g,"tau":tau,"scale":scale,"tau_rel":32*gamma,
            "tau_abs":32*f.tiny,"accumulation_length":n,
            "deterministic_certificate":False}

@dataclass
class CostReceipt:
    counters: dict = field(default_factory=dict)
    wall_seconds: dict = field(default_factory=dict)
    def add(self,name,count=1,seconds=0.):
        self.counters[name]=self.counters.get(name,0)+int(count)
        self.wall_seconds[name]=self.wall_seconds.get(name,0.)+float(seconds)
    def snapshot(self): return {"counters":dict(self.counters),"wall_seconds":dict(self.wall_seconds)}
    def delta(self,before):
        return {"counters":{k:v-before["counters"].get(k,0) for k,v in self.counters.items()},
                "wall_seconds":{k:v-before["wall_seconds"].get(k,0.) for k,v in self.wall_seconds.items()}}

@dataclass(frozen=True)
class FrozenState:
    object_id: str
    state_id: str
    split: str
    configuration_id: str
    material_gauge_sha256: str
    receiver_geometry_sha256: str
    chi: np.ndarray
    current_polarizability: np.ndarray
    points: np.ndarray
    grid_shape: tuple
    cell_volume: float
    wavenumber: float
    r: np.ndarray
    forcing: np.ndarray
    U: np.ndarray
    LU: np.ndarray
    SU: np.ndarray
    D: np.ndarray
    step_old: np.ndarray
    ell: np.ndarray
    Lambda: np.ndarray
    source_phase: np.ndarray
    normal_relative_residual: float
    geometry: dict = field(default_factory=dict)
    protected_slots: tuple = ()
    receiver_mode_ids: tuple = ()
    accepted_count: int = 0
    online_provenance: dict = field(default_factory=lambda:{"origin":"FrozenState.current_estimate"})
    reduced_gradient_norm: float = 0.
    def __post_init__(self):
        for key in ("chi","current_polarizability","points","r","forcing","U","LU","SU","D","step_old","ell","Lambda","source_phase"):
            object.__setattr__(self,key,readonly(getattr(self,key)))
        if self.r.ndim!=2 or self.r.shape[0]!=4 or self.forcing.shape!=(self.U.shape[0],4):
            raise ValueError("v1 requires four illuminations")
        if la.norm(self.U.conj().T@self.U-np.eye(self.U.shape[1]))>TOL:
            raise ValueError("nonorthonormal current basis")
    @property
    def lambda_scalar(self):
        v=float(self.Lambda[0,0])
        if not np.allclose(self.Lambda,np.eye(len(self.ell))*v,rtol=0,atol=1e-14): raise ValueError("v1 scalar prior required")
        return v
    @property
    def state_sha256(self):return self.online_provenance["state_sha256"]
    @property
    def state_hash(self):return self.state_sha256

@dataclass(frozen=True)
class SeedBundle:
    vectors: np.ndarray
    norms: np.ndarray
    source_scales: np.ndarray
    active: np.ndarray
    phases: np.ndarray
    receipts: tuple = ()
    def __getitem__(self,k): return getattr(self,k)

@dataclass(frozen=True)
class ActionSpec:
    route: str
    action_id: str
    configuration_id: str
    drop_indices: tuple[int,...]
    prior_logit: float
    recipe_id: str | None = None
    coefficient_real: tuple | None = None
    coefficient_imag: tuple | None = None
    basis_gauge_sha256: str | None = None
    def __post_init__(self):
        if self.route not in ("A","B","D"): raise ValueError("invalid route")
        if not self.configuration_id or not np.isfinite(self.prior_logit): raise ValueError("invalid identity/logit")
        if len(set(self.drop_indices))!=len(self.drop_indices) or any(i<0 for i in self.drop_indices): raise ValueError("illegal drop")
        if self.route=="B" and (self.recipe_id is not None or self.coefficient_real is None or not self.basis_gauge_sha256): raise ValueError("B coefficient/gauge required")

@runtime_checkable
class BackendProtocol(Protocol):
    receipt: CostReceipt
    def apply_L(self,C): ...
    def apply_L_adjoint(self,C): ...
    def apply_S(self,C): ...
    def apply_S_adjoint(self,W): ...
    def apply_B(self,z,illumination=None): ...
    def material_dual(self,W,illumination=None): ...
    def project_B(self,C): ...
    def full_J_action(self,z): ...
    def get_opm_seeds(self,state): ...
    def shared_state_features(self,state): ...
    def outgoing_features(self,state): ...

class OperatorBackend:
    """Counted physical callbacks. Full directional solver is verifier-only."""
    def __init__(self,L,Ladj,S,Sadj,B,Bdual,project_B,Jaction,receiver,metadata):
        self._L,self._LH,self._S,self._SH=L,Ladj,S,Sadj
        self._B,self._BH,self._PB=B,Bdual,project_B
        self._J,self._receiver=Jaction,receiver
        self.metadata=dict(metadata);self.receipt=CostReceipt();self._seeds={};self._basis={}
    def _call(self,name,fn,a,axis=1):
        a=np.asarray(a); count=1 if a.ndim==1 else a.shape[axis]
        t=time.perf_counter(); out=fn(a); self.receipt.add(name,count,time.perf_counter()-t);return out
    def apply_L(self,C): return self._call("L_rhs",self._L,C)
    def apply_L_adjoint(self,C): return self._call("L_adjoint_rhs",self._LH,C)
    def apply_S(self,C): return self._call("S_rhs",self._S,C)
    def apply_S_adjoint(self,W): return self._call("S_adjoint_rhs",self._SH,W)
    def apply_B(self,z,illumination=None):
        z=np.asarray(z);n=1 if z.ndim==1 else z.shape[1];t=time.perf_counter();out=self._B(z)
        self.receipt.add("B_illumination_rhs",4*n,time.perf_counter()-t)
        return out if illumination is None else out[:,illumination]
    def material_dual(self,W,illumination=None):
        t=time.perf_counter();out=self._BH(W,illumination);self.receipt.add("B_adjoint_illumination_rhs",4 if illumination is None else 1,time.perf_counter()-t);return out
    def project_B(self,C): return self._call("B_projection_current_columns",self._PB,C)
    def full_J_action(self,z):
        t=time.perf_counter();z=np.asarray(z);n=1 if z.ndim==1 else z.shape[1]
        out=self._J(z);self.receipt.add("full_J_directions",n,time.perf_counter()-t);self.receipt.add("full_J_illumination_rhs",4*n);return out
    def get_receiver_basis(self,count):
        t=time.perf_counter();out=self._receiver(count);self.receipt.add("receiver_mode_accesses",1,time.perf_counter()-t);return out
    def freeze_state(self,U,*,object_id,state_id,split="development",receiver_mode_ids=(),accepted_count=0):
        U=np.asarray(U,complex);LU=self.apply_L(U);SU=self.apply_S(U);D=U.conj().T@LU
        m=self.metadata;ep=endpoint_from_maps(D,SU,self.project_B(U),m["r"],m["ell"],m["Lambda"])
        phases=m.get("source_phase",np.ones(4,complex))
        state_hash=sha256((m["state_hash"]+"".join(array_hash(m[k]) for k in ("chi","a","r","forcing","ell","Lambda"))+array_hash(phases)+m["material_gauge_sha256"]+m["receiver_geometry_sha256"]).encode()).hexdigest()
        config=sha256((state_hash+array_hash(U)+SEED_SCHEMA+"seed=1e-10;breakdown=1e-12;orth=1e-10;phase=maxabs;schema=192v2").encode()).hexdigest()
        if not receiver_mode_ids:receiver_mode_ids=tuple(range(U.shape[1])) if np.allclose(U,self.get_receiver_basis(U.shape[1]),rtol=0,atol=TOL) else (-1,)*U.shape[1]
        return FrozenState(object_id,state_id,split,config,m["material_gauge_sha256"],m["receiver_geometry_sha256"],m["chi"],m["a"],m["points"],tuple(m["grid_shape"]),m["cell_volume"],m["wavenumber"],m["r"],m["forcing"],U,LU,SU,D,ep["step"],m["ell"],m["Lambda"],phases,ep["normal_relative_residual"],m.get("geometry",{}),(),tuple(receiver_mode_ids),accepted_count,{"origin":"FrozenState.current_estimate","state_sha256":state_hash},ep["reduced_gradient_norm"])
    def get_old_endpoint(self,state):
        return endpoint_from_maps(state.D,state.SU,self.project_B(state.U),state.r,state.ell,state.Lambda)
    def get_opm_seeds(self,state):
        if state.configuration_id in self._seeds:return self._seeds[state.configuration_id]
        before=self.receipt.snapshot();t=time.perf_counter();U,LU,D=state.U,state.LU,state.D
        if not stability(D)["stable"]:raise ValueError("ROOT_UNSTABLE")
        rng=np.random.default_rng(180018);mix=(rng.choice([-1,1],(4,2))+1j*rng.choice([-1,1],(4,2)))/np.sqrt(8)
        p=lambda v:v-U@(U.conj().T@v)
        O=self.apply_S_adjoint((state.r/state.source_phase[:,None]).T@mix)
        Ocor=self.apply_L_adjoint(U@la.solve(D.conj().T,U.conj().T@O))
        z=rng.choice([-1.,1.],(len(state.ell),2))/np.sqrt(len(state.ell))
        Bz=self.apply_B(z);M=np.einsum("npi,pi->ni",Bz/state.source_phase[None,:,None],mix,optimize=True)
        Psrc=(state.forcing/state.source_phase[None,:])@mix;srcs=[O,Psrc,M];vec=[];norms=[];scales=[];active=[];phases=[]
        for family,src in enumerate(srcs):
            cor=Ocor if family==0 else LU@la.solve(D,U.conj().T@src)
            v=p(src-cor)
            for i in range(2):
                sc=la.norm(src[:,i])+la.norm(cor[:,i]);nm=la.norm(v[:,i]);on=sc>0 and nm/sc>1e-10
                q=v[:,i]/nm if on else np.zeros(U.shape[0],complex)
                ph=np.exp(-1j*np.angle(q[np.argmax(np.abs(q))])) if on else 1+0j
                vec.append(q*ph);norms.append(nm);scales.append(sc);active.append(on);phases.append(ph)
        receipt={"origin":"shared_backend.opm_seeds","dependencies":["current_U","current_LU","current_D","residual","forcing","material_injection"],"exclusive_seconds":time.perf_counter()-t,**self.receipt.delta(before)}
        b=SeedBundle(readonly(np.column_stack(vec)),readonly(norms),readonly(scales),readonly(active,bool),readonly(phases), (receipt,));self._seeds[state.configuration_id]=b;return b
    def shared_state_features(self,state):
        from .features import shared_features
        return shared_features(self,state)
    def outgoing_features(self,state):
        from .features import outgoing_features
        return outgoing_features(self,state)
    def register_basis(self,state,Q):
        h=array_hash(Q);self._basis[(state.configuration_id,h)]=readonly(Q);return h

class VerifiedExchange:
    """Only selected records enter physical materialization and full Jd.

    Atomic rank-two records use a joint endpoint. No singleton gain addition.
    Callback recipes are registered externally; no teacher ranker is used.
    """
    def __init__(self,backend,state,recipes=None):
        self.backend,self.state=backend,state;self.recipes=recipes or {};self._u=None
    def _materialize(self,action):
        b,s=self.backend,self.state
        if action.configuration_id!=s.configuration_id:raise ValueError("STALE_CONFIGURATION")
        ds=action.drop_indices
        if not ds or any(i>=s.U.shape[1] or i in s.protected_slots for i in ds):raise ValueError("ILLEGAL_DROP")
        if action.route=="B":
            Q=b._basis.get((s.configuration_id,action.basis_gauge_sha256))
            if Q is None:raise ValueError("STALE_BASIS_GAUGE")
            coeff=np.asarray(action.coefficient_real)+1j*np.asarray(action.coefficient_imag)
            Z=np.column_stack((s.U[:,list(ds)],Q))
            if coeff.ndim==1: coeff=coeff[:,None]
            if coeff.shape!=(Z.shape[1],len(ds)) or not np.isfinite(coeff).all():raise ValueError("COEFFICIENT_SHAPE")
            C=Z@coeff
        else:
            recipe=self.recipes[action.recipe_id];C=recipe() if callable(recipe) else recipe.materialize(s,b)
            C=np.asarray(C);C=C[:,None] if C.ndim==1 else C
        b.receipt.add("current_materializations",len(ds));raw_norm=la.norm(C)
        retained=np.delete(s.U,ds,axis=1)
        for _ in range(2): C=C-retained@(retained.conj().T@C)
        Q,R=la.qr(C,mode="economic")
        if Q.shape[1]!=len(ds) or min(abs(np.diag(R)),default=0)<=TOL*raw_norm:raise ValueError("DEGENERATE_ACTION")
        return retained,Q
    def evaluate(self,action):
        b,s=self.backend,self.state;before=b.receipt.snapshot();t=time.perf_counter()
        try:
            retained,C=self._materialize(action);drops=action.drop_indices;keep=[i for i in range(s.U.shape[1]) if i not in drops]
            # Only selected new columns are acted on. Retained maps are reused.
            LC=b.apply_L(C);SC=b.apply_S(C);NC=b.project_B(C);NR=b.project_B(retained)
            W=np.column_stack((retained,C));LW=np.column_stack((s.LU[:,keep],LC));SW=np.column_stack((s.SU[:,keep],SC))
            ep=endpoint_from_maps(W.conj().T@LW,SW,np.concatenate((NR,NC),axis=1),s.r,s.ell,s.Lambda)
            b.receipt.add("endpoint_solves")
            if self._u is None:self._u=pack(s.r)+b.full_J_action(s.step_old)
            d=ep["step"]-s.step_old;z=b.full_J_action(d)
            g=directional_check(pack(s.r),self._u,z,s.step_old,d,s.ell,s.Lambda)
            return {"status":"OK","action_id":action.action_id,"accepted":g["gain"]>g["tau"],**g,
                    "step":ep["step"],"new_U":W,"normal_relative_residual":ep["normal_relative_residual"],"stability":ep["stability"],
                    "singular_core_fallback":not stability(s.D[np.ix_(keep,keep)])["stable"],"cost":b.receipt.delta(before),"wall_seconds":time.perf_counter()-t}
        except (ValueError,la.LinAlgError) as e:
            return {"status":"INVALID_ENDPOINT","reason":str(e),"accepted":False,"action_id":action.action_id,"cost":b.receipt.delta(before),"wall_seconds":time.perf_counter()-t}
    def accept(self,result):
        if result.get("status")!="OK" or not result.get("accepted"):raise ValueError("unverified accept")
        if self.state.accepted_count>=3:raise ValueError("accept cap")
        s=self.state;return self.backend.freeze_state(result["new_U"],object_id=s.object_id,state_id=s.state_id,split=s.split,accepted_count=s.accepted_count+1,receiver_mode_ids=(-1,)*s.U.shape[1])
