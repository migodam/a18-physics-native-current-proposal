"""Batched direct endpoints for the SAME paid A17 dictionary and real gauge.

This removes Python/per-atom dense-vector projection overhead. No candidates,
operators, stability rules, prior, reference or gain rule are changed.
"""
import time
import numpy as np
from .core import pack

def audit_bank_batched(backend,state,bank,Jfull,anchor_J=None):
    begin=time.perf_counter();before=backend.receipt.snapshot();bank=np.asarray(bank,complex);n=bank.shape[1];k=state.U.shape[1];P,m=state.r.shape;p=len(state.ell)
    LB=backend.apply_L(bank);SB=backend.apply_S(bank);NB=backend.project_B(bank);NU=backend.project_B(state.U)
    gains=np.full((n,k),np.nan);taus=np.full((n,k),np.nan);scores=np.full((n,k),-np.inf);valid=np.zeros((n,k),bool);failures=[]
    r=pack(state.r);u=r+Jfull@state.step_old;h=state.ell+state.Lambda@state.step_old
    acc=len(r)+3*p+16;eps=np.finfo(float).eps;rel=32*acc*eps/(1-acc*eps)
    base=r@r+u@u+abs(state.ell@state.step_old)+state.step_old@state.Lambda@state.step_old
    def gains_tau(step,J):
        d=step-state.step_old;z=d@J.T;uu=r+J@state.step_old
        dc=.5*np.sum(z*z,axis=1);pc=.5*np.einsum('bi,ij,bj->b',d,state.Lambda,d)
        gain=-(z@uu)-(d@h)-dc-pc
        terms=np.abs(z)@np.abs(uu)+np.abs(d)@np.abs(h)+dc+pc
        bb=r@r+uu@uu+abs(state.ell@state.step_old)+state.step_old@state.Lambda@state.step_old
        return gain,32*np.finfo(float).tiny+rel*(terms+bb)
    for drop in range(k):
        keep=[i for i in range(k) if i!=drop];V=state.U[:,keep];coef=V.conj().T@bank;C=bank-V@coef;norm=np.linalg.norm(C,axis=0);good=norm>1e-10;safe=np.where(good,norm,1.)
        C/=safe;LC=(LB-state.LU[:,keep]@coef)/safe;SC=(SB-state.SU[:,keep]@coef)/safe;NC=(NB-np.einsum('ij,pid->pjd',coef.conj(),NU[:,keep],optimize=True))/safe[None,:,None]
        D=np.empty((n,k,k),complex);D[:,:-1,:-1]=state.D[np.ix_(keep,keep)];D[:,:-1,-1]=(V.conj().T@LC).T;D[:,-1,:-1]=C.conj().T@state.LU[:,keep];D[:,-1,-1]=np.einsum('nj,nj->j',C.conj(),LC)
        sv=np.linalg.svd(D,compute_uv=False);stable=(sv[:,-1]>1e-10*sv[:,0])&np.isfinite(sv).all(axis=1);good&=stable
        D[~good]=np.eye(k);N=np.empty((n,k,P,p),complex);N[:,:-1]=NU[:,keep].transpose(1,0,2)[None];N[:,-1]=NC.transpose(1,0,2)
        coeff=np.linalg.solve(D,N.reshape(n,k,P*p)).reshape(n,k,P,p);O=np.empty((n,m,k),complex);O[:,:,:-1]=state.SU[:,keep];O[:,:,-1]=SC.T
        Jc=np.einsum('bmk,bkpd->bpmd',O,coeff,optimize=True);J=np.concatenate((Jc.real,Jc.imag),axis=2).reshape(n,2*P*m,p)
        H=J.transpose(0,2,1)@J+state.Lambda;rhs=np.einsum('bmi,m->bi',J,r)+state.ell
        try:step=np.linalg.solve(H,-rhs[:,:,None])[:,:,0]
        except np.linalg.LinAlgError:
            # Same direct implementation, explicit per-endpoint fallback. No jitter.
            from .discrete_audit import audit_bank
            out=audit_bank(backend,state,bank,Jfull,anchor_J);out['batch_fallback']='NORMAL_SOLVE_FAILURE';return out
        nr=np.linalg.norm((H@step[:,:,None])[:,:,0]+rhs,axis=1)/np.maximum(np.linalg.norm(rhs,axis=1),np.finfo(float).tiny)
        good&=(nr<=1e-10)&np.isfinite(step).all(axis=1);gain,tau=gains_tau(step,Jfull)
        gains[good,drop]=gain[good];taus[good,drop]=tau[good];valid[:,drop]=good
        if anchor_J is not None:scores[good,drop]=gains_tau(step,anchor_J)[0][good]
        for j in np.flatnonzero(~good):failures.append({'atom':int(j),'drop':drop,'reason':'DEGENERATE_RELATIVE_CURRENT' if norm[j]<=1e-10 else 'CURRENT_NEAR_SINGULAR' if not stable[j] else 'NORMAL_RESIDUAL_FAILED'})
        backend.receipt.add('endpoint_solves',int(np.count_nonzero(norm>1e-10)));backend.receipt.add('batched_endpoint_batches')
    positive=np.where(valid&(gains>taus),gains,0.)
    return {'gains':gains,'tau':taus,'valid':valid,'scores':scores,'positive':positive,'best_gain':float(positive.max(initial=0)),'failures':failures,'wall_seconds':time.perf_counter()-begin,'cost':backend.receipt.delta(before),'implementation':'batched same direct endpoints'}
