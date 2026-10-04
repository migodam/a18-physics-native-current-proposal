"""Offline full-quadratic labels on a declared current bank.

All bank L/S/B actions are paid here and unavailable to proposal features.
Analytic online ranking uses its explicitly paid independent anchor, not Jfull.
"""
import time
import numpy as np
from scipy import linalg as la
from .core import endpoint_from_maps,pack,directional_check

def audit_bank(backend,state,bank,Jfull,anchor_J=None):
    t=time.perf_counter();before=backend.receipt.snapshot();bank=np.asarray(bank,complex);n=bank.shape[1];k=state.U.shape[1]
    LB=backend.apply_L(bank);SB=backend.apply_S(bank);NB=backend.project_B(bank);NU=backend.project_B(state.U)
    gains=np.full((n,k),np.nan);tau=np.full((n,k),np.nan);scores=np.full((n,k),-np.inf);valid=np.zeros((n,k),bool);failures=[]
    u=pack(state.r)+Jfull@state.step_old
    for drop in range(k):
        keep=[i for i in range(k) if i!=drop];V=state.U[:,keep];coef=V.conj().T@bank
        C=bank-V@coef;norm=la.norm(C,axis=0);LC=LB-state.LU[:,keep]@coef;SC=SB-state.SU[:,keep]@coef
        NC=NB-np.einsum("ij,pid->pjd",coef.conj(),NU[:,keep],optimize=True)
        for j in range(n):
            if norm[j]<=1e-10:failures.append({"atom":j,"drop":drop,"reason":"DEGENERATE_RELATIVE_CURRENT"});continue
            c=C[:,j]/norm[j];lc=LC[:,j]/norm[j];sc=SC[:,j]/norm[j];nc=NC[:,j]/norm[j]
            D=np.empty((k,k),complex);D[:-1,:-1]=state.D[np.ix_(keep,keep)];D[:-1,-1]=V.conj().T@lc;D[-1,:-1]=c.conj()@state.LU[:,keep];D[-1,-1]=np.vdot(c,lc)
            try:
                backend.receipt.add("endpoint_solves")
                ep=endpoint_from_maps(D,np.column_stack((state.SU[:,keep],sc)),np.concatenate((NU[:,keep],nc[:,None]),axis=1),state.r,state.ell,state.Lambda)
                d=ep["step"]-state.step_old;z=Jfull@d;g=directional_check(pack(state.r),u,z,state.step_old,d,state.ell,state.Lambda)
                gains[j,drop]=g["gain"];tau[j,drop]=g["tau"];valid[j,drop]=True
                if anchor_J is not None:
                    ur=pack(state.r)+anchor_J@state.step_old;zr=anchor_J@d
                    scores[j,drop]=directional_check(pack(state.r),ur,zr,state.step_old,d,state.ell,state.Lambda)["gain"]
            except (ValueError,la.LinAlgError) as exc:failures.append({"atom":j,"drop":drop,"reason":str(exc)})
    positive=np.where(valid&(gains>tau),gains,0.)
    return {"gains":gains,"tau":tau,"valid":valid,"scores":scores,"positive":positive,"best_gain":float(positive.max(initial=0)),"failures":failures,"wall_seconds":time.perf_counter()-t,"cost":backend.receipt.delta(before)}

def ranked_records(bank,state,labels,K=4):
    from .core import ActionSpec
    legal=[(j,d) for j,d in np.ndindex(labels["scores"].shape) if labels["valid"][j,d] and d not in state.protected_slots]
    legal.sort(key=lambda x:(-labels["scores"][x],x));records=[]
    for j,d in legal[:K]:
        if labels["scores"][j,d]<=0:continue
        records.append(ActionSpec("D",f"D:{state.configuration_id}:{j}:{d}",state.configuration_id,(d,),float(labels["scores"][j,d]),recipe_id=str(j)))
    return records
