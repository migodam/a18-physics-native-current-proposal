"""Integrator review of immutable T0 labels with improved fair controls.

No teacher search is repeated. A legal recipe mask and equivalent batched
dictionary endpoints are re-timed together with B under one fixed BLAS setup.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from route_b.controls import cached_physics,positive_tau_bound,decode_coefficients,run as run_controls
from route_b.preflight import cov,write,benchmark,jsonable
from route_b.policy import POLICIES
from .fft_maxwell import load_backend

def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def review(args):
    start=time.perf_counter();raw=Path(args.raw_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    g0_files=[Path(args.g0_root)/n for n in ('G0.json','G0_FFT.json','G0_DICTIONARY_BATCH.json')]
    if any(json.loads(p.read_text()).get('status')!='PASS' for p in g0_files):raise RuntimeError('actual shared G0 evidence is not PASS')
    from route_a.bridge import ReceiptBackend
    from route_a.runtime import RouteAPlugin
    from route_a.baselines import all_registered_policies,analytic_actions
    a_policies=all_registered_policies()
    for path in sorted(raw.glob('*_state*.json')):
        row=json.loads(path.read_text());name=row['object_id']+'_'+row['state_id'];cache_path=Path(args.cache_root)/(name+'.npz');cache=cached_physics(cache_path);bound=float(positive_tau_bound(cache))
        row['raw_T0_sha256']=hash_file(path);row['tau_positive_upper_bound']=bound
        for key,variant in row['variants'].items():
            for drop in variant['drops']:
                modes=[]
                for mode in drop['modes']:
                    mode=dict(mode)
                    if key=='degree2':
                        check=cache.numpy_metrics(decode_coefficients(mode['coefficients']),drop['drops']);mode.update(gain=check['gain'],tau=check['tau'])
                        if not check['valid'] or check['gain']<=check['tau']:continue
                    elif mode['gain']<=bound:continue
                    else:mode['positive_significance_rule']='gain exceeds conservative upper bound on original tau'
                    modes.append(mode)
                drop['modes']=modes;drop['best_gain']=max([m['gain'] for m in modes],default=0.)
            variant['best_gain']=max(d['best_gain'] for d in variant['drops'])
        row['B_poly_policy_gains']=[g if g>bound else 0. for g in row['B_poly_policy_gains']]
        row['B_oracle']=max(row['variants']['degree2']['best_gain'],max(row['B_poly_policy_gains']))
        for pair in row['pairs']:
            for mode in pair['modes']:
                value=mode['coefficients'];T=np.asarray(value['real'])+1j*np.asarray(value['imag']);check=cache.numpy_metrics(T,pair['drops']);mode.update(gain=check['gain'],tau=check['tau'],valid=check['valid'])
            pair['best_gain']=max([m['gain'] for m in pair['modes'] if m['valid'] and m['gain']>m['tau']],default=0.)
        row['union_reference']=max(row['original_reference'],row['A_oracle'],row['B_oracle'],row['B_per_core'],max(v['best_gain'] for v in row['variants'].values()))
        b=load_backend(Path(args.public_root)/(name+'.npz'));s=b.freeze_state(b.get_receiver_basis(8),object_id=row['object_id'],state_id=row['state_id']);plugin=RouteAPlugin(ReceiptBackend(b));features,audit=plugin.prepare(s)
        with np.load(cache_path,allow_pickle=False) as z:gains=z['A_gains'];tau=z['A_tau'];valid=z['A_valid']
        positive=np.where(valid&(gains>tau),gains,0.);ids={rid:i for i,rid in enumerate(features.recipe_ids)}
        row['A_policy_gains_corrected']=[max([positive[ids[a.recipe_id],a.drop_index] for a in analytic_actions(features,**p)],default=0.) for p in a_policies]
        row['A_legal_node_mask_count']=int(features.node_mask.sum());row['locked_test_accessed']=False;rows.append(row);write(out/(name+'.json'),row)
    if len(rows)!=12 or len({r['object_id'] for r in rows})!=6:raise RuntimeError('incomplete object-paired development audit')
    scores=[cov([{**r,'value':r['B_poly_policy_gains'][i]} for r in rows],'value') for i in range(48)];b_id=int(np.argmax(scores));policy=POLICIES[b_id]
    a_scores=[cov([{**r,'value':r['A_policy_gains_corrected'][i]} for r in rows],'value') for i in range(len(a_policies))];a_id=int(np.argmax(a_scores));a_policy=a_policies[a_id]
    for row in rows:row['B_poly']=row['B_poly_policy_gains'][b_id];row['A_analytic']=row['A_policy_gains_corrected'][a_id]
    controls=run_controls(out,args.cache_root,out/'G2_controls.json');timings=[]
    for i,row in enumerate(rows):
        name=row['object_id']+'_'+row['state_id'];ts=benchmark(Path(args.public_root)/(name+'.npz'),row['object_id'],row['state_id'],args.a17_root,policy,5,a_policy);timings.extend(ts);write(out/'timings.json',timings);print(json.dumps({'event':'fair_timing','states_complete':i+1,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    ratios={'B-poly':[],'A-analytic':[]}
    for row in rows:
        ts=[t for t in timings if t['object_id']==row['object_id'] and t['state_id']==row['state_id']];D=np.median([t['Tacq_cold'] for t in ts if t['method']=='D-full'])
        for method in ratios:ratios[method].append(float(np.median([t['Tacq_cold'] for t in ts if t['method']==method])/D))
    near=[r for r in rows if r['family']=='near'];B=cov(rows,'B_oracle');bn=cov(near,'B_oracle');loss=cov(rows,'B_per_core')-B;A=cov(rows,'A_oracle');an=cov(near,'A_oracle');checksB={'macro_coverage':B>=.90,'near_coverage':bn>=.75,'shared_root_loss':loss<=.05,'acq_median_ratio':np.median(ratios['B-poly'])<=.5,'acq_P95_ratio':np.percentile(ratios['B-poly'],95)<=1.};checksA={'macro_coverage':A>=.85,'near_coverage':an>=.70,'acq_median_ratio':np.median(ratios['A-analytic'])<=.5,'acq_P95_ratio':np.percentile(ratios['A-analytic'],95)<=1.}
    g2=max(cov(rows,'B_poly'),controls['best_coverage']);gates={'G0':{'status':'PASS','evidence':[{'path':'results/shared/'+p.name,'sha256':hash_file(p)} for p in g0_files]},'G1A':{'status':'PASS' if all(checksA.values()) else 'FAIL','checks':checksA,'coverage_macro_union':A,'coverage_near_union':an},'G1B':{'status':'PASS' if all(checksB.values()) else 'FAIL','checks':checksB,'coverage_macro_union':B,'coverage_near_union':bn,'shared_root_loss_union':loss},'G2':{'status':'PASS' if g2<.95 else 'FAIL','meaning':'positive learning headroom','strongest_same_space_nonNN_coverage':g2}}
    summary={'schema':'a18.integrated_preflight_review.v1','gates':gates,'config_sha256':hash_file(args.config),'new_cohort_plan_sha256':hash_file(args.plan),'evidence_scope':'real_vector_maxwell','state_count':12,'object_count':6,'historical_exposure':True,'B_poly_policy_id':b_id,'B_poly_policy':policy,'B_poly_coverage_union':cov(rows,'B_poly'),'A_analytic_policy_id':a_id,'A_analytic_policy':a_policy,'A_analytic_coverage_union':cov(rows,'A_analytic'),'coverage_original_dictionary':{'B':cov(rows,'B_oracle','original_reference'),'A':cov(rows,'A_oracle','original_reference')},'Tacq_ratios':{k:{'median':float(np.median(v)),'P95':float(np.percentile(v,95)),'per_state':v} for k,v in ratios.items()},'teacher_reference':'BEST_FOUND, not global optimum','original_T0_status':'immutable raw evidence; weak/slow controls retained','controls':controls['coverage_union'],'timing_protocol':'five rotated fresh-backend repetitions; equivalent batched D endpoints; one BLAS thread setting across all methods','cold_warm_limit':'warm timing is same-configuration proposal-record replay only, not full warm verifier deployment','locked_test_accessed':False,'rows':rows,'elapsed_seconds':time.perf_counter()-start};write(out/'INTEGRATED_PREFLIGHT.json',summary);print(json.dumps(jsonable({k:v for k,v in summary.items() if k not in ('rows','controls')})));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--public-root',required=True);p.add_argument('--a17-root',required=True);p.add_argument('--g0-root',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='config/pilot.json');p.add_argument('--plan',default='manifests/new_cohort_plan.json');review(p.parse_args())
