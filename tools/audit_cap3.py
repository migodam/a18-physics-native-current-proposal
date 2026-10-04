"""Frozen policies on two TRAIN states: actual acceptance/cache rebuild audit.

This changes the current subspace at fixed material, not a nonlinear transfer.
"""
import argparse,json,time
from pathlib import Path
import torch
from route_b.evaluate import benchmark_state,Proposals,validate_freeze
from route_b.preflight import write

def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    validate_freeze(args.freeze_receipt)
    rules=json.loads(Path(args.rules).read_text());proposer=Proposals(rules)
    root=Path(args.cohort_root);manifest=json.loads((root/'generation_manifest.json').read_text())
    objects=[next(o for o in manifest['objects'] if o['split']=='train' and o['family']==f) for f in ('separated','near')]
    rows=[];begin=time.perf_counter();checks=[]
    for obj in objects:
        state=obj['states'][0]
        actual=benchmark_state(root/state['public'],obj['object_id'],state['state_id'],8,['B-main','B-poly','A-analytic'],proposer,repetitions=1,sequence=True)
        rows.extend(actual)
        for row in actual:
            if row['cache_mode']!='cap3_sequence':continue
            rounds=row['rounds'];assert row['accepted_count']==len(rounds)<=3
            ids=[r['configuration_id'] for r in rounds];assert len(set(ids))==len(ids)
            for i,r in enumerate(rounds):
                selected=r['selected'];assert len(selected)<=4
                chosen=[a for a in selected if a['actually_applied']];assert len(chosen)==1
                v=chosen[0]['verification'];assert v['accepted'] and v['gain']>v['tau']
                assert chosen[0]['action']['configuration_id']!=r['configuration_id']
                if i:assert all(a['action']['configuration_id']==ids[i-1] for a in selected)
                action_ids=[a['action']['action_id'] for a in selected];assert len(set(action_ids))==len(action_ids)
            checks.append({'object_id':obj['object_id'],'method':row['method'],'accepted_count':len(rounds),'changed_configuration_per_acceptance':True,'selected_count_per_round':[len(r['selected']) for r in rounds]})
    assert any(c['accepted_count']>1 for c in checks),'real rebuild path never exercised'
    result={'status':'PASS','checks':checks,'rows':rows,'elapsed_seconds':time.perf_counter()-begin,'split':'TRAIN_ONLY','scope':'Actual cap-3 current-subspace implementation audit at fixed material; not held-out coverage or nonlinear transfer','nonlinear_material_updates':0,'model_or_rule_changes':False}
    write(args.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort-root',required=True);p.add_argument('--rules',required=True);p.add_argument('--freeze-receipt',required=True);p.add_argument('--output',required=True);run(p.parse_args())
