"""Current-estimate-only 192/32 schema; no teacher capabilities are called."""
import time
import hashlib
import numpy as np
from scipy import linalg as la

NORM_CHANNELS=(32,*range(36,48),51,52,59,60,61)

def stats(x):return [float(np.mean(x)),float(np.std(x)),float(np.min(x)),float(np.max(x))]

def shared_features(backend,s):
    t=time.perf_counter();before=backend.receipt.snapshot();x=np.zeros(192)
    chi=s.chi.ravel();ab=abs(chi);a=abs(s.current_polarizability.ravel());pts=s.points.reshape(-1,3)
    sides=np.ptp(pts,axis=0);ctr=(pts.max(0)+pts.min(0))/2
    rx=np.asarray(s.geometry.get("receivers",np.zeros((1,3))))
    x[:16]=stats(chi.real)+stats(chi.imag)+stats(ab)+[float(np.mean(ab>0)),float(a.mean()),float(a.max()),s.cell_volume*len(chi)]
    x[16:32]=[*sides,*ctr,*(rx.mean(0)-ctr),float(np.sqrt(np.mean(np.sum((rx-ctr)**2,axis=1)))),s.r.shape[1],4,s.wavenumber,s.cell_volume**(1/3),len(chi),1.]
    # Construct the declared retained source residual, never a full solve.
    from .core import stability
    reduced=s.forcing-s.LU@la.solve(s.D,s.U.conj().T@s.forcing)
    x[32:48]=[la.norm(s.r),float(abs(s.r).mean()),float(abs(s.r).std()),float(abs(s.r).max()),*la.norm(s.r,axis=1),*la.norm(s.forcing,axis=0),*la.norm(reduced,axis=0)]
    dm=s.D
    x[48:64]=[s.U.shape[1],len(s.ell),s.lambda_scalar,la.norm(s.ell),la.norm(s.step_old),stability(dm)["relative"],s.normal_relative_residual,float(abs(dm).mean()),float(abs(dm).max()),float(np.trace(dm).real),float(np.trace(dm).imag),la.norm(s.SU),la.norm(s.LU),s.reduced_gradient_norm,s.accepted_count,0.]
    x[list(NORM_CHANNELS)]=np.log1p(x[list(NORM_CHANNELS)])
    ids=s.geometry.get("channel_ids",tuple(range(s.r.shape[1])));order=np.argsort(ids,kind="stable");r=(s.r[:,order]/s.source_phase[:,None]).ravel();w=np.empty((32,len(r)),complex)
    for j in range(32):
        for p in range(4):
            for col,index in enumerate(order):
                h=hashlib.sha256(f"180018|{s.receiver_geometry_sha256}|{p}:{ids[index]}|{j}".encode()).digest()[0]
                w[j,p*s.r.shape[1]+col]=((1 if h&1 else -1)+1j*(1 if h&2 else -1))/np.sqrt(2*len(r))
    sk=w.conj()@r
    x[64:128]=np.r_[sk.real,sk.imag]
    freq=sorted([(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4)],key=lambda a:(sum(v*v for v in a),a))[:32]
    coord=(pts-ctr)/np.where(sides>0,sides,1);tests=np.exp(2j*np.pi*np.asarray(freq)@coord.T);ms=tests.conj()@chi*s.cell_volume
    x[128:192]=np.r_[ms.real,ms.imag]
    if not np.isfinite(x).all():raise ValueError("nonfinite shared features")
    return x,({"origin":"shared_backend.shared192","dependencies":["current_chi_estimate","current_U","current_LU","current_D","residual","forcing","geometry"],"exclusive_seconds":time.perf_counter()-t,"norm_encoding":"log1p(raw_norm), scale=1",**backend.receipt.delta(before)},)

def outgoing_features(backend,s):
    t=time.perf_counter();k=s.U.shape[1]
    if k>16:raise ValueError("outgoing k cap")
    x=np.zeros((k,32));corr=s.SU.conj().T@(s.r/s.source_phase[:,None]).sum(axis=0)
    for i in range(k):
        x[i,i]=1;x[i,16:]=[la.norm(s.LU[:,i]),la.norm(s.SU[:,i]),s.D[i,i].real,s.D[i,i].imag,float(abs(s.D[i]).mean()),float(abs(s.D[i]).max()),float(abs(s.D[:,i]).mean()),float(abs(s.D[:,i]).max()),corr[i].real,corr[i].imag,0.,0.,s.receiver_mode_ids[i] if s.receiver_mode_ids else -1.,float(i in s.protected_slots),0.,1.]
    return x,({"origin":"shared_backend.outgoing32","dependencies":["current_U","current_LU","current_SU","current_D","residual"],"exclusive_seconds":time.perf_counter()-t,"injection_row_present":False},)
