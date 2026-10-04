"""Frozen k4/k16 quality using paid, exact FP64 compressed endpoints.

No full deployment timing or nonlinear quality claim is made from this replay.
"""
import argparse,json,hashlib,time,platform
from pathlib import Path
import numpy as np
import torch
from route_b.controls import cached_physics
from route_b.train import top_indices
from route_b.model import CoefficientMLP
from route_b.normalizer import Normalizer
from route_b.fitted_controls import FittedControls
from route_b.preflight import write,cov

def run(args):
    torch.set_num_threads(4);start=time.perf_counter();freeze=json.loads(Path(args.freeze_receipt).read_text());models={}
    if freeze['status']!='FROZEN':raise RuntimeError('unfrozen transfer')
    def verified_path(record):
        path=Path(record['path'])
        if not path.is_file():path=Path(record['artifact_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('changed frozen artifact')
        return path
    for record in freeze['checkpoints']:
        path=verified_path(record)
        if record['name'].startswith('B-'):
            ck=torch.load(path,map_location='cpu',weights_only=False);model=CoefficientMLP();model.load_state_dict(ck['weights']);model.eval();models[record['name']]=(model,Normalizer.from_dict(ck['normalizer']))
    control=FittedControls(verified_path(next(r for r in freeze['checkpoints'] if r['name']=='train_fitted_controls')));rules=json.loads(verified_path(freeze['rules'][0]).read_text());output={}
    for k in args.ranks:
        root=Path(args.cache_root)/('k'+str(k));index=json.loads((root/'index.json').read_text());rows=[]
        if index['status']!='PASS' or index['current_rank']!=k or not index['locked_test_opened']:raise RuntimeError('incomplete or wrong transfer reference')
        for record in index['records']:
            for field,key in (('cache_sha256','cache'),('metadata_sha256','metadata')):
                if hashlib.sha256((root/record[key]).read_bytes()).hexdigest()!=record[field]:raise RuntimeError('transfer cache hash drift')
            meta=json.loads((root/record['metadata']).read_text());cache=cached_physics(root/record['cache'],k=k)
            with np.load(root/record['cache'],allow_pickle=False) as z:x=z['xi'].copy();out=z['outgoing'].copy();mask=z['rank_mask'].copy();legal=z['legal_slots'].copy()
            row={'object_id':record['object_id'],'state_id':record['state_id'],'family':meta['family'],'near':meta['near'],'union_reference':meta['reference'],'original_reference':meta['original_reference'],'B-poly':meta['B_poly_policy_gains'][rules['B_poly_policy_id']]}
            for name,(model,norm) in models.items():
                xi,o=norm.transform(x,out)
                with torch.no_grad():t,l,n,_=model(torch.as_tensor(xi[None],dtype=torch.float32),torch.as_tensor(o[None],dtype=torch.float32),torch.as_tensor(mask[None],dtype=torch.float32),torch.as_tensor(legal[None]))
                actions=top_indices(t[0].numpy(),l[0].numpy(),n[0],cache.rank);details=[];gain=0.
                for drop,mode,c in actions:
                    r=cache.numpy_metrics(c,(drop,));positive=r['valid'] and r['gain']>r['tau'];gain=max(gain,r['gain'] if positive else 0.);details.append({'drop':drop,'mode':mode,'gain':r['gain'],'tau':r['tau'],'valid':r['valid'],'positive':positive})
                row[name]=gain;row[name+'_actions']=details
            for method in ('ridge_0.01_m1','KNN3_objects_m1'):
                gain=0.
                for drop,mode,c,score in control.actions(x,out,cache.rank,legal,method):
                    r=cache.numpy_metrics(c,(drop,))
                    if r['valid'] and r['gain']>r['tau']:gain=max(gain,r['gain'])
                row['control:'+method]=gain
            rows.append(row)
        methods=['B-main','B-order0','B-poly','control:ridge_0.01_m1','control:KNN3_objects_m1'];metrics={m:{'macro_union':cov(rows,m),'macro_original':cov(rows,m,'original_reference'),'near_union':cov([r for r in rows if r['near']],m)} for m in methods};diff=[]
        for oid in sorted({r['object_id'] for r in rows}):
            group=[r for r in rows if r['object_id']==oid]
            if sum(r['union_reference'] for r in group)>0:diff.append(cov(group,'B-main')-cov(group,'B-poly'))
        rng=np.random.default_rng(2026100418);samples=rng.choice(diff,(2000,len(diff)),replace=True).mean(1) if diff else [];ci=np.percentile(samples,[2.5,97.5]).tolist() if len(samples) else [None,None];output[str(k)]={'status':'COMPLETE_CACHED_FP64_QUALITY_ONLY','metrics':metrics,'rows':rows,'object_count':len(set(r['object_id'] for r in rows)),'paired_bootstrap_main_minus_poly_CI95':ci,'replicates':2000,'deployment_Tinc':'NOT_RUN_AFTER_LEARNING_STOP_GATE','nonlinear_quality':'NOT_RUN_NO_WINNER','Maxwell_calls_during_replay':0,'paid_physics_source':'Earlier gated shared rank-specific cache preparation; features formed before full matrices/labels.'}
    result={'status':'FROZEN_TRANSFER_QUALITY_COMPLETE','ranks':output,'elapsed_seconds':time.perf_counter()-start,'freeze_sha256':hashlib.sha256(Path(args.freeze_receipt).read_bytes()).hexdigest(),'model_or_rule_selection_after_test':False,'runtime':{'platform':platform.system(),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}};write(args.output,result);print(json.dumps({'status':result['status'],'metrics':{k:v['metrics'] for k,v in output.items()}}));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-receipt',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--ranks',type=int,nargs='+',default=[4,16]);p.add_argument('--output',required=True);run(p.parse_args())
