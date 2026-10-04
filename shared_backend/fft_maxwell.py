"""Whitelisted full-vector A8 frozen data; old anchor/closure never loaded.

FFT L/L* and GMRES settings reproduce A18_B_PILOT/code/physics.py. The original
material B is real-parameter (24), not a holomorphic 12-column surrogate.
"""
from pathlib import Path
import hashlib,time
import numpy as np
import scipy.fft as fft
from scipy.sparse.linalg import LinearOperator,gmres
from scipy import linalg as la
from .core import OperatorBackend,pack,array_hash

WHITELIST=("a","chi","S","B","xyz","kernel","r","ell","Lambda","shape","pad","k","h","theta","scale")
FORBIDDEN=("V","Ar_inv","IA","O","JA","UA","common_total_fields","common_current_fields")

def fibonacci(n):
    z=1-2*(np.arange(n)+.5)/n;a=np.arange(n)*np.pi*(3-np.sqrt(5));return np.c_[np.sqrt(1-z*z)*np.cos(a),np.sqrt(1-z*z)*np.sin(a),z]

def load_backend(path,*,rtol=1e-10,device="cpu"):
    path=Path(path);begin=time.perf_counter()
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in WHITELIST}
    return backend_from_arrays(a,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rtol=rtol,device=device,load_wall=time.perf_counter()-begin)

def backend_from_arrays(a,*,source_sha256="generated",rtol=1e-10,device="cpu",load_wall=0.):
    shape=tuple(a["shape"].astype(int));pad=tuple(a["pad"].astype(int));n=int(np.prod(shape)*3);P=a["B"].shape[1];m=a["S"].shape[0]
    av=a["a"].ravel().repeat(3);kernel=a["kernel"];B=a["B"];S=a["S"]
    torch_cache={}
    if device=="cuda":
        import torch
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        torch_cache={"a":torch.as_tensor(av,device="cuda"),"kernel":torch.as_tensor(kernel,device="cuda"),"S":torch.as_tensor(S,device="cuda")}
    elif device!="cpu":raise ValueError("explicit cpu/cuda device")
    def conv(C,adj=False):
        one=C.ndim==1;C=C[:,None] if one else C
        z=C.T.reshape((-1,)+shape+(3,));kg=kernel.swapaxes(0,1).conj() if adj else kernel
        hat=fft.fftn(z,s=pad,axes=(1,2,3));out=np.einsum("ijxyz,bxyzj->bxyzi",kg,hat,optimize=True)
        out=fft.ifftn(out,axes=(1,2,3))[:,:shape[0],:shape[1],:shape[2],:].reshape(-1,n).T
        return out[:,0] if one else out
    def L(C,adj=False):
        if device=="cuda":
            import torch
            one=C.ndim==1;cm=C[:,None] if one else C;v=torch.as_tensor(np.ascontiguousarray(cm),device="cuda",dtype=torch.complex128)
            av2=torch_cache["a"][:,None];src=v*av2.conj() if adj else v
            z=src.T.reshape((-1,)+shape+(3,));hat=torch.fft.fftn(z,s=pad,dim=(1,2,3));kg=torch_cache["kernel"].transpose(0,1).conj() if adj else torch_cache["kernel"]
            out=torch.einsum("ijxyz,bxyzj->bxyzi",kg,hat);y=torch.fft.ifftn(out,dim=(1,2,3))[:,:shape[0],:shape[1],:shape[2],:].reshape(-1,n).T
            y=v-y if adj else v-av2*y;torch.cuda.synchronize();out=y.cpu().numpy();return out[:,0] if one else out
        ax=av if C.ndim==1 else av[:,None];return C-conv(C*ax.conj(),True) if adj else C-ax*conv(C)
    def inj(z):return np.einsum("ptn,p...->nt...",B,z,optimize=True)
    def dual(W,illumination=None):
        if illumination is not None:return np.einsum("pn,n...->p...",B[:,illumination].conj(),W,optimize=True).real
        return np.einsum("ptn,nt...->p...",B.conj(),W,optimize=True).real
    def proj(C):
        if C.ndim==1:C=C[:,None]
        return np.einsum("nk,ptn->tkp",C.conj(),B,optimize=True)
    # Original fixed acquisition, derived from public geometry/known plane waves.
    pts=a["xyz"].reshape(-1,3);dirs=fibonacci(4);pol=np.cross(dirs,[0.,0.,1.]);pol/=la.norm(pol,axis=1)[:,None]
    inc=np.exp(1j*float(a["k"])*pts@dirs.T)[:,None,:]*pol.T[None,:,:]
    forcing=(a["a"].ravel()[:,None,None]*inc).reshape(n,4)
    basis_cache={}
    def receiver(count):
        if "all" not in basis_cache:basis_cache["all"]=la.svd(S,full_matrices=False)[2].conj().T
        return basis_cache["all"][:,:count].copy()
    holder={};solve_records=[]
    def solve_current(rhs):
        one=rhs.ndim==1;rhs=rhs[:,None] if one else rhs;out=np.empty_like(rhs,dtype=complex)
        for j in range(rhs.shape[1]):
            v=rhs[:,j];op=LinearOperator((n,n),matvec=lambda c:holder["b"].apply_L(c),dtype=complex)
            c,info=gmres(op,v,rtol=rtol,atol=0,restart=160,maxiter=150)
            rr=la.norm(holder["b"].apply_L(c)-v)/max(la.norm(v),1e-300)
            if info or rr>max(5*rtol,1e-12):raise RuntimeError("PRIMAL_SOLVE_FAILED")
            out[:,j]=c
        holder["b"].receipt.add("full_primal_illumination_rhs",rhs.shape[1])
        return out[:,0] if one else out
    def jaction(z):
        rhs=inj(z);one=z.ndim==1;rhs=rhs[:,:,None] if one else rhs;out=np.empty((P,m,rhs.shape[2]),complex)
        for i in range(P):
            for j in range(rhs.shape[2]):
                v=rhs[:,i,j];op=LinearOperator((n,n),matvec=lambda c:holder["b"].apply_L(c),dtype=complex);t=time.perf_counter()
                c,info=gmres(op,v,rtol=rtol,atol=0,restart=160,maxiter=150)
                rr=la.norm(holder["b"].apply_L(c)-v)/max(la.norm(v),1e-300)
                solve_records.append({"info":int(info),"relative_residual":float(rr),"wall_seconds":time.perf_counter()-t})
                if info or rr>max(5*rtol,1e-12):raise RuntimeError("FULL_J_SOLVE_FAILED")
                out[i,:,j]=S@c
        out=pack(out);return out[:,0] if one else out
    geom={"receivers":7*fibonacci(8),"channel_ids":tuple(range(m)),"historical_anchor_loaded":False,"current_dof":n}
    state_hash=hashlib.sha256("".join(array_hash(v) for k,v in sorted(a.items())).encode()).hexdigest()
    meta={"chi":a["chi"],"a":a["a"],"points":a["xyz"],"grid_shape":shape,"cell_volume":float(a["h"])**3,"wavenumber":float(a["k"]),"r":a["r"].reshape(P,m),"ell":a["ell"],"Lambda":a["Lambda"],"forcing":forcing,"source_phase":np.ones(P,complex),"geometry":geom,"state_hash":state_hash,"material_gauge_sha256":array_hash(a["scale"]),"receiver_geometry_sha256":array_hash(geom["receivers"]),"public_source_sha256":source_sha256,"historical_anchor_loaded":False,"device":device}
    backend=OperatorBackend(lambda C:L(C),lambda C:L(C,True),lambda C:S@C,lambda W:S.conj().T@W,inj,dual,proj,jaction,receiver,meta);holder["b"]=backend
    backend.solve_records=solve_records;backend.solve_current=solve_current
    def gd(C,adj=False):
        t=time.perf_counter();out=conv(C,adj)*float(a["h"])**3;backend.receipt.add("GD_adjoint_rhs" if adj else "GD_rhs",1 if C.ndim==1 else C.shape[1],time.perf_counter()-t);return out
    backend.apply_GD=gd
    backend.receipt.add("whitelisted_state_load",1,load_wall);backend.receipt.add("forcing_construction",4)
    return backend
