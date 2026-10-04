"""Frozen physical proposals, paid full Jd, actual cold/warm verification and traces."""
import argparse,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from shared_backend.core import ActionSpec,VerifiedExchange,array_hash
from shared_backend.fft_maxwell import load_backend
from shared_backend.dictionary import prepare_dictionary
from shared_backend.batched_dictionary import audit_bank_batched
from shared_backend.discrete_audit import ranked_records
from .basis import build_basis,feature_vector
from .policy import POLICIES,propose_poly,select_top4
from .model import CoefficientMLP
from .normalizer import Normalizer
from .fitted_controls import FittedControls
from .preflight import write,jsonable

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);model=CoefficientMLP();model.load_state_dict(checkpoint['weights']);model.eval();return model,Normalizer.from_dict(checkpoint['normalizer'])

class Proposals:
    def __init__(self,rules):
        self.rules=rules;self.models={name:load_model(path) for name,path in rules.get('checkpoints',{}).items()};self.controls=FittedControls(rules['control_weights']) if rules.get('control_weights') else None
    def prepare(self,method,b,s,K=None):
        K=K or (2 if method.endswith('_K2') else 4);recipes={}
        if method.startswith('D-'):
            d=prepare_dictionary(b,s,self.rules['a17_root']);proxy=audit_bank_batched(b,s,d['vectors'],d['anchor_J'],d['anchor_J']);actions=ranked_records(d['vectors'],s,proxy,1 if method=='D-recommended' else K);recipes={str(j):(lambda j=j:d['vectors'][:,j]) for j in range(d['vectors'].shape[1])};return actions,recipes
        if method=='A-analytic':
            from route_a.features import extract
            from route_a.bridge import ReceiptBackend
            from route_a.recipes import LazyCache
            from route_a.audit import AcquisitionAudit
            from route_a.baselines import analytic_actions
            bridge=ReceiptBackend(b);cache=LazyCache();audit=AcquisitionAudit();view=bridge.online_view(s,audit);features,rs,_=extract(view,bridge,cache,audit);actions=[a.to_shared() for a in analytic_actions(features,K=K,**self.rules['A_policy'])];recipes={r.id:(lambda r=r:r.materialize(view,bridge,cache,audit)) for r in rs};return actions,recipes
        basis=build_basis(b,s);xi,_=feature_vector(b,s,basis);out,_=b.outgoing_features(s)
        if basis.Q.shape[1]==0:return [],{}
        if method=='B-poly':
            p=POLICIES[self.rules['B_poly_policy_id']];return propose_poly(s,basis,family_order=p[0],degree=p[1],mixed=p[2],drop_rule=p[3],K=K),{}
        if method.startswith('control:'):
            predicted=self.controls.actions(xi,out,basis.Q.shape[1],[i not in s.protected_slots for i in range(len(out))],method.split(':',1)[1],K);actions=[]
            for i,m,c,score in predicted:
                actions.append(ActionSpec('B',hashlib.sha256(f'{s.configuration_id}|{method}|{i}|{m}'.encode()).hexdigest(),s.configuration_id,(i,),score,coefficient_real=tuple(c.real),coefficient_imag=tuple(c.imag),basis_gauge_sha256=basis.basis_gauge_sha256))
            b.receipt.add('coefficient_records',2*len(out));return actions,{}
        name=method.removesuffix('_K2');model,norm=self.models[name];x,o=norm.transform(xi,out);mask=np.r_[np.ones(basis.Q.shape[1]),np.zeros(24-basis.Q.shape[1])];legal=np.asarray([i not in s.protected_slots for i in range(len(out))])
        with torch.no_grad():t,l,n,_=model(torch.as_tensor(x[None],dtype=torch.float32),torch.as_tensor(o[None],dtype=torch.float32),torch.as_tensor(mask[None],dtype=torch.float32),torch.as_tensor(legal[None]))
        b.receipt.add('coefficient_records',2*len(out));return select_top4(l[0].numpy(),float(n[0]),t[0].numpy(),s,basis,K),{}

def trace(action,result,applied=False):
    return {'action':asdict(action),'verification':{k:jsonable(result[k]) for k in ('status','reason','gain','tau','accepted','normal_relative_residual','stability','singular_core_fallback','wall_seconds','cost') if k in result},'step_sha256':array_hash(result['step']) if 'step' in result else None,'new_U_sha256':array_hash(result['new_U']) if 'new_U' in result else None,'actually_applied':applied}

def benchmark_state(public,oid,sid,rank,methods,proposer,repetitions=5,sequence=True):
    rows=[]
    for rep in range(repetitions):
        rotated=methods[rep%len(methods):]+methods[:rep%len(methods)]
        for method in rotated:
            begin=time.perf_counter();b=load_backend(public);s=b.freeze_state(b.get_receiver_basis(rank),object_id=oid,state_id=sid);state_wall=time.perf_counter()-begin;before=b.receipt.snapshot();at=time.perf_counter();actions,recipes=proposer.prepare(method,b,s);cold_acq=time.perf_counter()-at;cold_cost=b.receipt.delta(before);v=VerifiedExchange(b,s,recipes)
            vt=time.perf_counter();results=[v.evaluate(a) for a in actions];verify_wall=time.perf_counter()-vt;gain=max([r.get('gain',0.) for r in results if r.get('accepted')],default=0.)
            row={'object_id':oid,'state_id':sid,'rank':rank,'method':method,'repetition':rep,'cache_mode':'cold','Tacq':cold_acq,'Tverify':verify_wall,'Tinc':cold_acq+verify_wall,'state_seconds':state_wall,'Ttotal':state_wall+cold_acq+verify_wall,'gain':gain,'selected_count':len(actions),'acquisition_cost':cold_cost,'cost':b.receipt.delta(before),'actions':[trace(a,r) for a,r in zip(actions,results)],'scope':'initial frozen singleton top-K'};rows.append(row)
            # Paid root/inference/records may be reused only in identical configuration.
            # Unlike T0/T1 memo timing, this includes actual selected materialization,
            # endpoint solves and every new full Jd. Only old full Jx is reused.
            before=b.receipt.snapshot();wt=time.perf_counter();warm_actions=list(actions);warm_acq=time.perf_counter()-wt;vt=time.perf_counter();warm_results=[v.evaluate(a) for a in warm_actions];warm_verify=time.perf_counter()-vt
            rows.append({**row,'cache_mode':'warm_same_configuration','Tacq':warm_acq,'Tverify':warm_verify,'Tinc':warm_acq+warm_verify,'Ttotal':warm_acq+warm_verify,'state_seconds':0.,'gain':max([r.get('gain',0.) for r in warm_results if r.get('accepted')],default=0.),'cost':b.receipt.delta(before),'acquisition_cost':{'counters':{},'seconds':{}},'actions':[trace(a,r) for a,r in zip(actions,warm_results)],'warm_dependencies':'Paid root/feature/inference/records and old full Jx; all selected new currents/full Jd rerun.'})
            if sequence and rep==0:
                # Separate action domain: cumulative cap-3 gain is never used as
                # singleton coverage. Each acceptance invalidates U-dependent caches.
                st=time.perf_counter();before=b.receipt.snapshot();total=0.;applied=[];rounds=[];current_results=results;current_actions=actions;extra_acq=0.
                for turn in range(3):
                    eligible=[(j,r) for j,r in enumerate(current_results) if r.get('accepted')]
                    if not eligible:break
                    j,best=max(eligible,key=lambda jr:jr[1]['gain']);total+=best['gain'];s=v.accept(best);applied.append(trace(current_actions[j],best,True));rounds.append({'round':turn+1,'configuration_id':s.configuration_id,'gain':best['gain'],'selected':[trace(a,r,jj==j) for jj,(a,r) in enumerate(zip(current_actions,current_results))]})
                    if turn==2:break
                    acq=time.perf_counter();current_actions,recipes=proposer.prepare(method,b,s);extra_acq+=time.perf_counter()-acq;v.recipes=recipes;current_results=[v.evaluate(a) for a in current_actions]
                rows.append({'object_id':oid,'state_id':sid,'rank':rank,'method':method,'repetition':0,'cache_mode':'cap3_sequence','gain':total,'accepted_count':len(applied),'Tinc':cold_acq+verify_wall+(time.perf_counter()-st),'Tacq':cold_acq+extra_acq,'Tverify':None,'cost_initial':row['cost'],'cost_additional':b.receipt.delta(before),'rounds':rounds,'scope':'separate cap-3 sequential finite-gain domain; first round cold cost included, warm audit excluded'})
    return rows

def validate_freeze(path):
    f=json.loads(Path(path).read_text())
    if f['status']!='FROZEN':raise RuntimeError('freeze receipt required')
    for group in ('checkpoints','rules'):
        for record in f[group]:
            if sha(record['path'])!=record['sha256']:raise RuntimeError('changed frozen artifact')
    return f

def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);freeze=validate_freeze(args.freeze_receipt);rules=json.loads(Path(args.rules).read_text());cohort=Path(args.cohort_root);manifest=json.loads((cohort/'generation_manifest.json').read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.split=='locked_test':
        marker=cohort/'locked_test_evaluation.json'
        if marker.exists():raise RuntimeError('locked test already opened for frozen evaluation; preserve partial results, do not rerun')
        write(marker,{'status':'STARTED','freeze_sha256':sha(args.freeze_receipt),'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    proposer=Proposals(rules);methods=rules['methods'];objects=[o for o in manifest['objects'] if o['split']==args.split];count=0;begin=time.perf_counter()
    for rank in args.ranks:
        for obj in objects:
            for s in obj['states']:
                rows=benchmark_state(cohort/s['public'],obj['object_id'],s['state_id'],rank,methods,proposer,args.repetitions,not args.no_sequence)
                name=f"{obj['object_id']}_{s['state_id']}_k{rank}";write(out/(name+'.json'),{'rows':rows,'freeze_sha256':sha(args.freeze_receipt),'family':obj['family'],'near':obj['near']});count+=1;print(json.dumps({'event':'frozen_physical_evaluation','state':name,'completed_state_ranks':count,'seconds':time.perf_counter()-begin}),flush=True)
                write(out/'progress.json',{'status':'BUILDING','state_rank_count':count,'elapsed_seconds':time.perf_counter()-begin,'freeze_sha256':sha(args.freeze_receipt)})
    result={'status':'PASS','split':args.split,'state_rank_count':count,'ranks':args.ranks,'repetitions':args.repetitions,'methods':methods,'freeze_sha256':sha(args.freeze_receipt),'elapsed_seconds':time.perf_counter()-begin,'scope':'Raw physical timing/evidence completion; scientific gates are judged separately.'};write(out/'index.json',result)
    if args.split=='locked_test':write(marker,{**result,'status':'COMPLETE'})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort-root',required=True);p.add_argument('--rules',required=True);p.add_argument('--freeze-receipt',required=True);p.add_argument('--split',choices=['validation','locked_test'],required=True);p.add_argument('--output',required=True);p.add_argument('--ranks',type=int,nargs='+',default=[8]);p.add_argument('--repetitions',type=int,default=5);p.add_argument('--no-sequence',action='store_true');a=p.parse_args();print(json.dumps(run(a)))
