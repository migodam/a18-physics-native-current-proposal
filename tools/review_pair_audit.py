"""Recheck stored joint rank-2 endpoints; never sum singleton gains."""
import json,time
from pathlib import Path
import numpy as np
import torch
from route_b.controls import cached_physics
from route_b.preflight import write,cov,jsonable

def run():
    torch.set_num_threads(4);begin=time.perf_counter();rows=[];pairs=[]
    for path in sorted(Path('results/route_b/preflight').glob('*_state*.json')):
        s=json.loads(path.read_text());cache=cached_physics(Path('cache/development/teacher_only')/(path.stem+'.npz'));joint=[]
        for pair in s['pairs']:
            drops=pair['drops'];modes=pair['modes']
            if not modes:continue
            a=modes[0]['coefficients'];T=np.asarray(a['real'])+1j*np.asarray(a['imag']);r=cache.numpy_metrics(T,drops);singles=[]
            for h,drop in enumerate(drops):
                c=np.r_[T[h,h],T[2:,h]];length=np.linalg.norm(c)
                singles.append(cache.numpy_metrics(c/length,(drop,)) if length>=1e-8 else {'gain':0.,'tau':0.,'valid':False})
            positive=r['valid'] and r['gain']>r['tau'];projected_only=positive and all(not q['valid'] or q['gain']<=q['tau'] for q in singles)
            optimized_singletons=[]
            for drop in drops:
                best=0.
                for mode in s['variants']['degree2']['drops'][drop]['modes']:
                    a=mode['coefficients'];c=(np.asarray(a['real'])+1j*np.asarray(a['imag'])).ravel();check=cache.numpy_metrics(c,(drop,))
                    if check['valid'] and check['gain']>check['tau']:best=max(best,check['gain'])
                optimized_singletons.append(best)
            only=positive and not any(g>0 for g in optimized_singletons)
            entry={'object_id':s['object_id'],'state_id':s['state_id'],'family':s['family'],'drops':drops,'joint_gain':r['gain'],'joint_tau':r['tau'],'joint_valid':r['valid'],'joint_normal_relative_residual':r['normal_relative_residual'],'joint_positive':positive,'projected_singleton_gains':[q['gain'] for q in singles],'projected_singleton_taus':[q['tau'] for q in singles],'projected_columns_joint_only':projected_only,'same_drops_best_found_singleton_gains':optimized_singletons,'joint_only':only};pairs.append(entry)
            if positive:joint.append(r['gain'])
        best=max(joint,default=0.);rows.append({'object_id':s['object_id'],'state_id':s['state_id'],'family':s['family'],'union_reference':s['union_reference'],'best_joint_gain':best,'extra_gain':max(0.,best-s['union_reference'])})
    extra=cov(rows,'extra_gain');rate=len({(p['object_id'],p['state_id']) for p in pairs if p['joint_only']})/len(rows);result={'schema':'a18.G7.rank2.v1','status':'NEED_BLOCK_PROPOSAL' if extra>=.10 or rate>.20 else 'NO_BLOCK_TRIGGER_FOUND','joint_gain_over_singleton_reference_macro':cov(rows,'best_joint_gain'),'positive_extra_gain_over_singleton_reference_macro':extra,'joint_only_pair_count':sum(p['joint_only'] for p in pairs),'joint_only_state_rate':rate,'projected_columns_joint_only_pairs':int(sum(p['projected_columns_joint_only'] for p in pairs)),'positive_joint_pair_count':sum(p['joint_positive'] for p in pairs),'total_pairs':len(pairs),'states':rows,'pairs':pairs,'search':'4 starts x 20 steps x 6 registered pairs per development state; BEST_FOUND, not optimum','gate_singletons':'Same registered drops, independently best-found degree-2 singleton endpoints, exact original tau. Projecting an arbitrary joint QR frame into single columns is gauge dependent and is only a separate diagnostic.','singleton_projection_definition':'Each joint column projected to [own deleted column,Q]; the other deleted column is removed because it is retained in the singleton endpoint.','scope':'historically exposed development; true joint rank-2 endpoint. This diagnoses singleton scope, not a distribution or nonlinear imaging claim.','elapsed_seconds':time.perf_counter()-begin,'locked_test_opened':False};write('results/preflight_review/G7_PAIR_REVIEW.json',result);print(json.dumps(jsonable({k:v for k,v in result.items() if k not in ('states','pairs')})))

if __name__=='__main__':run()
