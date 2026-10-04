"""Offline teacher/NN cache. Never passed to an online FrozenState or extractor."""
import time
import numpy as np
import torch
from shared_backend.core import pack,endpoint_from_maps,directional_check

class SmallPhysics:
    def __init__(self,backend,state,Q,Jfull,device="cpu"):
        self.state=state;self.Q=Q;self.Z=np.column_stack((state.U,Q));self.k=state.U.shape[1];self.rank=Q.shape[1];self.device=device
        before=backend.receipt.snapshot();t=time.perf_counter()
        LZ=backend.apply_L(self.Z);O=backend.apply_S(self.Z);N=backend.project_B(self.Z);A=self.Z.conj().T@LZ
        self.numpy={"A":A,"O":O,"N":N,"Jfull":np.asarray(Jfull,float),"r":pack(state.r),"ell":state.ell,"Lambda":state.Lambda,"old_step":state.step_old}
        self.cost={"wall_seconds":time.perf_counter()-t,**backend.receipt.delta(before)}
        for key,v in self.numpy.items():setattr(self,key,torch.as_tensor(np.array(v,copy=True),device=device,dtype=torch.complex128 if np.iscomplexobj(v) else torch.float64))
        self.u=self.r+self.Jfull@self.old_step;self.h=self.ell+self.Lambda@self.old_step
    def evaluate(self,T,drops):
        """Batch normalized singleton or Stiefel rank-two joint coordinates."""
        drops=tuple(drops);batch=T.shape[0];rank=len(drops);n=self.k+self.rank
        if T.ndim==2:T=T[:,:,None]
        if T.shape[1:]!=(self.rank+rank,rank):raise ValueError("coefficient shape")
        keep=[i for i in range(self.k) if i not in drops]
        E=torch.zeros((batch,n,self.k),device=self.device,dtype=torch.complex128)
        for j,i in enumerate(keep):E[:,i,j]=1
        inds=list(drops)+list(range(self.k,n));E[:,inds,len(keep):]=T
        EH=E.mH;D=EH@self.A@E
        sv=torch.linalg.svdvals(D.detach());valid=(sv[:,-1]>1e-10*sv[:,0])&torch.isfinite(sv).all(dim=1)
        # Singular blocks do not enter a solve. Invalid endpoint receives zero
        # gain and an explicit mask; loss code applies its registered penalty.
        safe=D.clone();safe[~valid]=torch.eye(self.k,device=self.device,dtype=torch.complex128)
        N=torch.einsum("bkn,pnd->bpkd",EH,self.N)
        C=torch.linalg.solve(safe[:,None],N);O=self.O@E
        Jc=torch.einsum("bmk,bpkd->bpmd",O,C)
        J=torch.cat((Jc.real,Jc.imag),dim=2).reshape(batch,-1,len(self.ell))
        H=J.mT@J+self.Lambda;g=J.mT@self.r+self.ell
        step=torch.linalg.solve(H,-g);normal=(H@step[:,:,None]).squeeze(-1)+g
        nr=torch.linalg.vector_norm(normal,dim=1)/torch.linalg.vector_norm(g,dim=1).clamp_min(torch.finfo(torch.float64).tiny)
        valid=valid&(nr<=1e-10)&torch.isfinite(step).all(dim=1)
        d=step-self.old_step;z=d@self.Jfull.mT
        gain=-(z@self.u)-(d@self.h)-.5*(z.square().sum(1)+torch.einsum("bi,ij,bj->b",d,self.Lambda,d))
        gain=torch.where(valid,gain,torch.zeros_like(gain))
        return gain,valid,step,nr
    def numpy_gain(self,t,drop):
        z=torch.as_tensor(np.asarray(t)[None],dtype=torch.complex128,device=self.device)
        with torch.no_grad():g,valid,_,_=self.evaluate(z,(drop,))
        return float(g[0]),bool(valid[0])
    def numpy_metrics(self,t,drops):
        z=torch.as_tensor(np.asarray(t)[None],dtype=torch.complex128,device=self.device)
        with torch.no_grad():g,valid,step,nr=self.evaluate(z,tuple(drops))
        a=self.numpy;d=step[0].cpu().numpy()-a['old_step'];check=directional_check(a['r'],a['r']+a['Jfull']@a['old_step'],a['Jfull']@d,a['old_step'],d,a['ell'],a['Lambda'])
        return {**check,'valid':bool(valid[0]),'normal_relative_residual':float(nr[0]),'step':step[0].cpu().numpy()}

def teacher_best(cache,drops,*,starts=8,steps=40,seed=1801,deadline=None):
    """Bounded Adam + sphere/Stiefel retraction; every result is best-found."""
    start=time.perf_counter();rank=len(drops);dim=cache.rank+rank;rng=np.random.default_rng(seed)
    c=(rng.normal(size=(starts,dim,rank))+1j*rng.normal(size=(starts,dim,rank)))
    c[0]=0;c[0,:rank]=np.eye(rank)
    retract=lambda x:torch.linalg.qr(x,mode="reduced")[0] if rank>1 else x/torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1e-300)
    T=torch.nn.Parameter(retract(torch.as_tensor(c,dtype=torch.complex128,device=cache.device)))
    optimizer=torch.optim.Adam([T],lr=.03);best=np.zeros(starts);bestT=T.detach().clone();failures=0
    for it in range(steps):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError("shared CPU teacher cap")
        optimizer.zero_grad();normalized=retract(T);gain,valid,_,_=cache.evaluate(normalized,drops)
        loss=-gain.sum();loss.backward();old=normalized.detach().clone();optimizer.step()
        with torch.no_grad():
            proposed=retract(T);gg,ok,_,_=cache.evaluate(proposed,drops)
            for half in range(5):
                bad=~ok
                if not bad.any():break
                failures+=int(bad.sum());proposed[bad]=retract(.5*(old[bad]+proposed[bad]));gg,ok,_,_=cache.evaluate(proposed,drops)
            proposed[~ok]=old[~ok];T.copy_(proposed)
            improve=ok&(gg>torch.as_tensor(best,device=cache.device))
            best[improve.cpu().numpy()]=gg[improve].cpu().numpy();bestT[improve]=proposed[improve]
    order=np.argsort(-best,kind="stable");modes=[]
    for j in order:
        t=bestT[j].detach().cpu().numpy();g=float(best[j])
        check=cache.numpy_metrics(t,drops)
        if not check['valid'] or check['gain']<=check['tau']:continue
        if rank==1 and any(abs(np.vdot(np.asarray(m["coefficients"]),t))>=.9 for m in modes):continue
        modes.append({"gain":check['gain'],"tau":check['tau'],"coefficients":t,"start_index":int(j)})
        if len(modes)==(2 if rank==1 else 1):break
    return {"best_gain":max([m['gain'] for m in modes],default=0.),"raw_best_gain":float(best.max(initial=0)),"modes":modes,"starts":starts,"steps":steps,"wall_seconds":time.perf_counter()-start,"invalid_backtracks":failures,"status":"BEST_FOUND","global_optimality":False,"drops":list(drops),"positive_rule":"valid and finite gain > original A17 tau"}
