"""Single shared offline teacher cache; locked split requires model freeze."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from route_b.basis import build_basis,feature_vector
from route_b.small_physics import SmallPhysics,teacher_best
from route_b.policy import POLICIES,propose_poly
from route_b.preflight import write,action_cached_gain
from .fft_maxwell import load_backend
from .core import pack,array_hash
from .dictionary import prepare_dictionary
from .batched_dictionary import audit_bank_batched

def prepare(args):
    torch.set_num_threads(4)
    start=time.perf_counter();cohort=Path(args.cohort_root);manifest=json.loads((cohort/'generation_manifest.json').read_text())
    if manifest['status']!='PASS':raise RuntimeError('shared cohort incomplete')
    if args.split=='locked_test':
        if not args.freeze_receipt:raise RuntimeError('locked teacher requires frozen models and rules')
        freeze=json.loads(Path(args.freeze_receipt).read_text())
        if freeze.get('status')!='FROZEN' or not freeze.get('checkpoints') or not freeze.get('rules'):raise RuntimeError('invalid freeze receipt')
    output=Path(args.output);records=[];selected=[r for r in manifest['objects'] if r['split'] in (('train','validation') if args.split=='train_validation' else ('locked_test',))]
    for obj in selected:
        for item in obj['states']:
            if time.perf_counter()-start>args.minutes_cap*60:raise TimeoutError('shared teacher remaining budget cap')
            name=obj['object_id']+'_'+item['state_id'];target=output/(name+'.npz');meta=output/(name+'.json')
            if target.exists() or meta.exists():raise RuntimeError('teacher cache already exists; never silently repeat')
            begin=time.perf_counter();b=load_backend(cohort/item['public']);s=b.freeze_state(b.get_receiver_basis(args.rank),object_id=obj['object_id'],state_id=item['state_id'],split=obj['split']);basis=build_basis(b,s);xi,_=feature_vector(b,s,basis);out,_=b.outgoing_features(s)
            with np.load(cohort/item['teacher'],allow_pickle=False) as z:J=pack(z['JF'].reshape(4,16,24))
            sm=SmallPhysics(b,s,basis.Q,J);dictionary=prepare_dictionary(b,s,args.a17_root);dl=audit_bank_batched(b,s,dictionary['vectors'],J,dictionary['anchor_J']);labels=[]
            for drop in range(args.rank):
                j=int(np.argmax(dl['positive'][:,drop]));v=dictionary['vectors'][:,j];t=np.r_[np.vdot(s.U[:,drop],v),basis.Q.conj().T@v];initials=[t] if np.linalg.norm(t)>1e-8 else []
                seed=basis.Q.conj().T@basis.seed_bundle.vectors[:,4];initials.append(np.r_[0.,seed]);labels.append(teacher_best(sm,(drop,),starts=4,steps=24,seed=1801+drop,deadline=time.monotonic()+max(0.,args.minutes_cap*60-(time.perf_counter()-start)),initials=initials))
            poly=[max([action_cached_gain(sm,a) for a in propose_poly(s,basis,family_order=p[0],degree=p[1],mixed=p[2],drop_rule=p[3])],default=0.) for p in POLICIES]
            from route_a.bridge import ReceiptBackend
            from route_a.runtime import RouteAPlugin
            from route_a.recipes import InvalidCurrent
            plugin=RouteAPlugin(ReceiptBackend(b));af,aa=plugin.prepare(s);raw=[];a_failures=[]
            for recipe in plugin.recipes:
                try:raw.append(plugin.materialize_selected(recipe.id))
                except InvalidCurrent as exc:raw.append(np.zeros(len(s.U),complex));a_failures.append({'recipe_id':recipe.id,'reason':str(exc)})
            abank=np.column_stack(raw);al=audit_bank_batched(b,s,abank,J)
            ref=max(dl['best_gain'],al['best_gain'],max(x['best_gain'] for x in labels),max(poly));row={'object_id':obj['object_id'],'state_id':item['state_id'],'split':obj['split'],'family':obj['family'],'near':obj['near'],'configuration_id':s.configuration_id,'basis_gauge_sha256':basis.basis_gauge_sha256,'source_public_sha256':item['public_sha256'],'source_teacher_sha256':item['teacher_sha256'],'reference':ref,'original_reference':dl['best_gain'],'B_oracle':max(x['best_gain'] for x in labels),'A_oracle':al['best_gain'],'A_inactive_recipes':a_failures,'B_poly_policy_gains':poly,'teachers':labels,'rank':sm.rank,'current_rank':args.rank,'cost':b.receipt.snapshot(),'wall_seconds':time.perf_counter()-begin,'teacher_status':'BEST_FOUND','teacher_search_bound':'4 starts x 24 steps/drop; shared once','feature_source':'current public state and paid OPM root, before offline matrices/labels','locked_test_opened':args.split=='locked_test'}
            output.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**sm.numpy,xi=xi,outgoing=out,basis_layers=basis.layers,rank_mask=np.r_[np.ones(sm.rank),np.zeros(24-sm.rank)],legal_slots=np.ones(args.rank,bool));write(meta,row);records.append({'object_id':obj['object_id'],'state_id':item['state_id'],'split':obj['split'],'family':obj['family'],'cache':name+'.npz','metadata':name+'.json','reference':ref,'rank':sm.rank,'cache_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest()});write(output/'index.json',{'status':'BUILDING','records':records});print(json.dumps({'event':'teacher_cached','state':name,'seconds':time.perf_counter()-start}),flush=True)
    result={'status':'PASS','records':records,'elapsed_seconds':time.perf_counter()-start,'cohort_plan_sha256':manifest['plan_sha256'],'locked_test_opened':args.split=='locked_test','count':len(records),'current_rank':args.rank,'feature_schema':'a18.B.OPM1552.v1','dataset_sha256':hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'reference_scope':'Singleton union: original full dictionary, all A bank recipes, degree-2 shared-root 4x24 best-found teacher, 48 registered B polynomial policies. Rank-2 endpoint audit is a separate action domain.'};write(output/'index.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort-root',required=True);p.add_argument('--rank',type=int,choices=[4,8,16],default=8);p.add_argument('--output',required=True);p.add_argument('--a17-root',required=True);p.add_argument('--split',choices=['train_validation','locked_test'],default='train_validation');p.add_argument('--freeze-receipt');p.add_argument('--minutes-cap',type=float,required=True);prepare(p.parse_args())
