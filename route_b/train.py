"""Integrator-authorized cached training; zero Maxwell actions per epoch."""
import argparse,hashlib,json,os,random,socket,time
from pathlib import Path
import numpy as np
import torch
from shared_backend.isolation import cached_only_epoch
from .controls import cached_physics,decode_coefficients
from .model import CoefficientMLP
from .normalizer import Normalizer
from .loss import physical_loss
from .preflight import write,cov

def load_items(root):
    root=Path(root);index=json.loads((root/'index.json').read_text())
    if index['status']!='PASS' or index.get('locked_test_opened'):raise RuntimeError('training requires complete train/validation cache with no locked access')
    items=[]
    for record in index['records']:
        if record['split'] not in ('train','validation'):raise RuntimeError('locked data in training index')
        
        for field,file_key in (('cache_sha256','cache'),('metadata_sha256','metadata')):
            if hashlib.sha256((root/record[file_key]).read_bytes()).hexdigest()!=record[field]:raise RuntimeError('cached training data hash mismatch')
        row=json.loads((root/record['metadata']).read_text());sm=cached_physics(root/record['cache']);row['dataset_sha256']=index['dataset_sha256'];row['feature_schema']=index['feature_schema']
        with np.load(root/record['cache'],allow_pickle=False) as z:row.update({k:z[k].copy() for k in ('xi','outgoing','basis_layers','rank_mask','legal_slots')})
        row['cache']=sm;row['teachers']=[[{**m,'coefficients':decode_coefficients(m['coefficients'])} for m in group['modes']] for group in row['teachers']];items.append(row)
    train=[s for s in items if s['split']=='train'];val=[s for s in items if s['split']=='validation']
    if len({s['object_id'] for s in train})!=32 or len({s['object_id'] for s in val})!=8:raise RuntimeError('required 32/8 object split incomplete')
    if {s['object_id'] for s in train}&{s['object_id'] for s in val}:raise RuntimeError('parent leakage')
    return train,val

def to_device(cache,device):
    cache.device=device
    for key,v in cache.numpy.items():setattr(cache,key,torch.as_tensor(v,device=device,dtype=torch.complex128 if np.iscomplexobj(v) else torch.float64))
    cache.u=cache.r+cache.Jfull@cache.old_step;cache.h=cache.ell+cache.Lambda@cache.old_step;return cache

def pack(items,norm,device):
    xs=[];outs=[]
    for row in items:x,o=norm.transform(row['xi'],row['outgoing']);xs.append(x);outs.append(o)
    return [torch.as_tensor(x,device=device,dtype=d) for x,d in ((np.stack(xs),torch.float32),(np.stack(outs),torch.float32),(np.stack([s['rank_mask'] for s in items]),torch.float32),(np.stack([s['legal_slots'] for s in items]),torch.bool))]

def top_indices(t,logits,noop,rank,K=4):
    if float(noop)>=float(np.max(logits,initial=-np.inf)):return []
    selected=[]
    for i,h in sorted(np.ndindex(logits.shape),key=lambda k:(-float(logits[k]),k)):
        if not np.isfinite(logits[i,h]):continue
        c=np.asarray(t[i,h,:rank+1],complex);length=np.linalg.norm(c)
        if not np.isfinite(c).all() or length<1e-8:continue
        c/=length
        if any(i==j and abs(np.vdot(c,a))>1-1e-10 for j,m,a in selected):continue
        selected.append((i,h,c))
        if len(selected)==K:break
    return selected

def validate(model,items,norm,K=4):
    model.eval();device=next(model.parameters()).device;rows=[]
    for s in items:
        with torch.no_grad():t,l,n,_=model(*pack([s],norm,device))
        actions=top_indices(t[0].cpu().numpy(),l[0].cpu().numpy(),n[0],s['rank'],K);gain=0.;details=[];cache=s['cache'];prior_device=cache.device;to_device(cache,'cpu')
        for i,h,c in actions:
            result=cache.numpy_metrics(c,(i,));accepted=result['valid'] and result['gain']>result['tau'];gain=max(gain,result['gain'] if accepted else 0.)
            details.append({'drop':i,'mode':h,'gain':result['gain'],'tau':result['tau'],'feasible':result['valid'],'positive':accepted,'normal_relative_residual':result['normal_relative_residual']})
        to_device(cache,prior_device);rows.append({'object_id':s['object_id'],'state_id':s['state_id'],'family':s['family'],'union_reference':s['reference'],'gain':gain,'selected':details,'scope':'exact FP64 cached initial frozen endpoints; not nonlinear or online cost'})
    near=[r for r in rows if r['family']=='near'];return {'macro':cov(rows,'gain'),'near':cov(near,'gain'),'rows':rows}

def train_one(train,val,config,out,seed=1801,lambda_order=.001,device='cuda',seconds_cap=1350,epochs_max=80,baseline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    norm=Normalizer.fit(train);model=CoefficientMLP().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5);rng=np.random.default_rng(seed)
    if baseline is None or baseline.get('status')!='HEADROOM':raise RuntimeError('complete pilot polynomial/ridge/KNN validation controls required before NN')
    for s in train+val:to_device(s['cache'],device)
    started=time.perf_counter();best=-np.inf;last=0;history=[];reason='MAX_EPOCHS';cfg=config['route_B']
    for epoch in range(1,epochs_max+1):
        if time.perf_counter()-started>=seconds_cap:reason='TIME_CAP';break
        model.train();order=rng.permutation(len(train));losses=[];invalid=0;signed=[]
        with cached_only_epoch() as counter:
            for first in range(0,len(order),8):
                batch=[train[int(i)] for i in order[first:first+8]];t,logits,noop,_=model(*pack(batch,norm,device));terms=[physical_loss(t[j],logits[j],noop[j],s['cache'],s['teachers'],s['reference'],s['basis_layers'],lambda_order) for j,s in enumerate(batch)];loss=torch.stack([x['loss'] for x in terms]).mean()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite physical loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()));invalid+=sum(x['invalid'] for x in terms);signed.extend(g for x in terms for g in x['raw_gains'])
        assert counter['Maxwell_action_count']==0 and not counter['attempted_Maxwell_operations']
        record={'epoch':epoch,'loss':float(np.mean(losses)),'invalid_endpoints':invalid,'raw_signed_gains':signed,**counter,'seconds':time.perf_counter()-started}
        if epoch%5==0:
            metrics=validate(model,val,norm);record.update(validation_macro=metrics['macro'],validation_near=metrics['near'])
            if metrics['macro']>best:
                best=metrics['macro'];last=epoch;torch.save({'weights':{k:v.detach().cpu() for k,v in model.state_dict().items()},'normalizer':norm.as_dict(),'seed':seed,'lambda_order':lambda_order,'epoch':epoch,'parameters':model.parameter_count,'config':config,'dataset_sha256':train[0]['dataset_sha256'],'feature_schema':train[0]['feature_schema'],'train_objects':norm.train_objects},out/'best.pt');write(out/'best_validation.json',metrics)
        history.append(record);print(json.dumps(record),flush=True)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if epoch>=20 and best<=baseline['macro']:reason='EPOCH20_NO_VALIDATION_HEADROOM';break
        if epoch>=20 and epoch-last>=15:reason='PATIENCE15';break
    result={'status':'CACHED_TRAINING_COMPLETE_ONLINE_G3_PENDING','reason':reason,'epochs':len(history),'seed':seed,'lambda_order':lambda_order,'parameters':model.parameter_count,'best_validation_macro':None if not np.isfinite(best) else best,'baseline':baseline,'elapsed_seconds':time.perf_counter()-started,'same_budget_seconds':seconds_cap,'same_epoch_cap':epochs_max,'Maxwell_action_count':0,'locked_test_opened':False,'device':device,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0};write(out/'training_summary.json',result);return result

def authorize(config,gate,owner,job,seed=1801):
    g=json.loads(Path(gate).read_text());o=json.loads(Path(owner).read_text())
    if g['config_sha256']!=hashlib.sha256(Path(config).read_bytes()).hexdigest():raise RuntimeError('config changed after gates')
    if any(g['gates'][k]['status']!='PASS' for k in ('G0','G1B','G2')):raise RuntimeError('B training gates not PASS')
    if job=='J2' and seed!=1801:raise RuntimeError('J2 is registered first seed only')
    if job=='J5' and (seed not in (1802,1803) or g.get('winner')!='B' or g['gates'].get('G3',{}).get('status')!='PASS' or g['gates'].get('G4',{}).get('status')!='PASS'):raise RuntimeError('winner seed extension not authorized by scientific gates')
    if o.get('job')!=job or o.get('host')!=socket.gethostname() or Path(owner).parent.name!='gpu.lock':raise RuntimeError('wrong shared queue owner')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/pilot.json');p.add_argument('--job',choices=['J2','J5'],required=True);p.add_argument('--seed',type=int,default=1801);p.add_argument('--cache-root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--gate-receipt',required=True);p.add_argument('--queue-owner',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--seconds-cap',type=float,default=1350);p.add_argument('--epochs-max',type=int,default=80);p.add_argument('--lambda-order',type=float,default=.001);args=p.parse_args();authorize(args.config,args.gate_receipt,args.queue_owner,args.job,args.seed);
    if args.seconds_cap>1350 or args.epochs_max>80:raise RuntimeError('registered training cap exceeded')
    cfg=json.loads(Path(args.config).read_text());tr,va=load_items(args.cache_root);train_one(tr,va,cfg,args.output,args.seed,args.lambda_order,args.device,args.seconds_cap,args.epochs_max,json.loads(Path(args.baseline).read_text()))
