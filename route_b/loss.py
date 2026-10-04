"""Exact cached endpoint gain gradients and phase-independent mode matching."""
import torch

def expected_best(g,p,K=4):
    # Freeze only the sorting permutation. Detaching the gains would remove
    # the physics gradient to current coefficients (A's labels are different).
    idx=torch.argsort(g.detach(),stable=True);v=g[idx];q=p[idx];tails=torch.flip(torch.cumsum(torch.flip(q,[0]),0),[0]).clamp(0,1);increments=v-torch.cat((v.new_zeros(1),v[:-1]));chance=tails*sum((1-tails).pow(j) for j in range(K));return (increments*chance).sum()

def physical_loss(t,logits,noop,cache,teachers,reference,layers,lambda_order=.001,K=4):
    ref=torch.as_tensor(reference,dtype=torch.float64,device=t.device).detach()
    if not torch.isfinite(ref) or ref<0:raise ValueError('invalid reference')
    combined=torch.cat((logits.flatten().to(torch.float64),noop.reshape(1).to(torch.float64)));logp=torch.log_softmax(combined,0);prob=logp.exp()
    if ref==0:return {'loss':-logp[-1],'cover':-logp[-1],'list':logp.new_zeros(()),'projective':logp.new_zeros(()),'diversity':logp.new_zeros(()),'order':logp.new_zeros(()),'raw_gains':[],'invalid':0,'reference_zero':True}
    gs=[];signed=[];valids=[];projective=[];div=[];target=[];rank=cache.rank;weight=torch.as_tensor([0]+[{0:0,1:3,2:8}[int(x)] for x in layers],device=t.device,dtype=torch.float64)
    for drop in range(cache.k):
        ti=t[drop,:,:rank+1].to(torch.complex128);gain,valid,step,_=cache.evaluate(ti,(drop,));d=step-cache.old_step;z=d@cache.Jfull.mT;dc=.5*z.square().sum(1);pc=.5*torch.einsum('bi,ij,bj->b',d,cache.Lambda,d);base=cache.r.square().sum()+cache.u.square().sum()+(cache.ell@cache.old_step).abs()+cache.old_step@cache.Lambda@cache.old_step;n=len(cache.r)+3*len(cache.old_step)+16;f=torch.finfo(torch.float64);rel=32*n*f.eps/(1-n*f.eps);tau=32*f.tiny+rel*(z.abs()@cache.u.abs()+d.abs()@cache.h.abs()+dc+pc+base)
        signed.append(gain.detach());gs.append(torch.where(valid&(gain.detach()>tau.detach()),gain,gain.new_zeros(2)));valids.append(valid);modes=teachers[drop];target.append(max([m['gain'] for m in modes],default=0.))
        if modes:
            tt=torch.stack([torch.as_tensor(m['coefficients'],dtype=torch.complex128,device=t.device).reshape(rank+1) for m in modes]);distance=1-(ti.conj()@tt.mT).abs().square()
            projective.append(distance[:,0].min() if len(modes)==1 else torch.minimum(distance[0,0]+distance[1,1],distance[0,1]+distance[1,0])/2)
            if len(modes)==2:div.append(torch.relu((ti[0].conj()@ti[1]).abs().square()-.9).square())
    g=torch.cat(gs)/ref.clamp_min(torch.finfo(torch.float64).tiny);valid=torch.cat(valids);cover=1-expected_best(torch.cat((g,g.new_zeros(1))),prob,K)+(prob[:-1]*~valid).sum()
    mass=torch.as_tensor(target,dtype=torch.float64,device=t.device);slot=prob[:-1].reshape(cache.k,2).sum(1);positive=mass>0;listloss=-(mass[positive]/mass[positive].sum()*slot[positive].clamp_min(torch.finfo(torch.float64).tiny).log()).sum() if positive.any() else -logp[-1]
    zero=cover.new_zeros(());proj=torch.stack(projective).mean() if projective else zero;diversity=torch.stack(div).mean() if div else zero;order=(t[:,:,:rank+1].abs().square().to(torch.float64)*weight).sum(-1).mean();total=cover+.25*listloss+.2*proj+.01*diversity+lambda_order*order
    return {'loss':total,'cover':cover,'list':listloss,'projective':proj,'diversity':diversity,'order':order,'raw_gains':torch.cat(signed).cpu().tolist(),'invalid':int((~valid).sum()),'reference_zero':False}
