"""Fixed coefficient policies. Selection never reads physical future gains."""
import hashlib,itertools
import numpy as np
from scipy import linalg as la
from shared_backend import ActionSpec

def poly_coefficients(state,basis):
    Q=basis.Q;r=Q.shape[1];out=[]
    for family in range(6):
        if not basis.seed_bundle.active[family]:continue
        q=basis.chain_columns[:,3*family:3*family+3];H=basis.H[family]
        for degree in range(3):
            # Neumann current sum expressed using the two paid Arnoldi actions.
            v=q[:,0].copy()
            if degree>=1:v+=q@H[:,0]
            if degree>=2:v+=q@(H[:,:2]@H[:2,0])
            c=Q.conj().T@v
            if la.norm(c)>1e-8:out.append((family,degree,c/la.norm(c),"neumann"))
        # Least squares residual minimizer of (I-H) with the 3x2 Arnoldi relation.
        if la.norm(q[:,1])>0:
            E=np.eye(3,2);y=la.lstsq(E-H,np.array([1.,0.,0.]))[0];c=Q.conj().T@(q[:,:2]@y)
            if la.norm(c)>1e-8:out.append((family,3,c/la.norm(c),"gmres2"))
    return out

def propose_poly(state,basis,*,family_order=0,degree=2,mixed=False,drop_rule=0,K=4):
    order=((4,0,2,5),(0,4,2,1),(2,4,0,3))[family_order]
    legal=[i for i in range(state.U.shape[1]) if i not in state.protected_slots]
    drops=sorted(legal,key=lambda i:(la.norm(state.SU[:,i]),i))[:4] if drop_rule==0 else list(reversed(legal))[:4]
    bank=poly_coefficients(state,basis);records=[];rank={i:j for j,i in enumerate(order)}
    bank=sorted([b for b in bank if b[1]==degree and b[0] in rank],key=lambda b:rank[b[0]])
    for j,(fam,d,c,kind) in enumerate(bank[:K]):
        if not drops:break
        drop=drops[j%len(drops)];t=np.r_[float(mixed),c];t/=la.norm(t)
        key=f"{state.configuration_id}|{drop}|{fam}|{d}|{kind}|{mixed}"
        records.append(ActionSpec("B",hashlib.sha256(key.encode()).hexdigest(),state.configuration_id,(drop,),float(-j),coefficient_real=tuple(t.real),coefficient_imag=tuple(t.imag),basis_gauge_sha256=basis.basis_gauge_sha256))
    return records

POLICIES=tuple(itertools.product(range(3),range(4),(False,True),range(2)))

def select_top4(logits,noop,coefficients,state,basis,K=4):
    """No-op highest => empty; stable legal unique records only, no filling."""
    if not np.isfinite(noop):return []
    legal_logits=np.where(np.isfinite(logits),logits,-np.inf).copy()
    for drop in state.protected_slots:legal_logits[drop]=-np.inf
    if noop>=float(np.max(legal_logits,initial=-np.inf)):return []
    records=[];seen=[]
    order=sorted(np.ndindex(logits.shape),key=lambda i:(-float(logits[i]),i))
    for drop,mode in order:
        if drop in state.protected_slots or not np.isfinite(logits[drop,mode]):continue
        t=np.asarray(coefficients[drop,mode],complex)[:1+basis.Q.shape[1]];norm=la.norm(t)
        if not np.isfinite(t).all() or norm<1e-8:continue
        t/=norm
        if any(drop==d and abs(np.vdot(c,t))>1-1e-10 for d,c in seen):continue
        seen.append((drop,t));key=f"{state.configuration_id}|{drop}|{mode}"
        records.append(ActionSpec("B",hashlib.sha256(key.encode()).hexdigest(),state.configuration_id,(drop,),float(logits[drop,mode]),coefficient_real=tuple(t.real),coefficient_imag=tuple(t.imag),basis_gauge_sha256=basis.basis_gauge_sha256))
        if len(records)==K:break
    return records
