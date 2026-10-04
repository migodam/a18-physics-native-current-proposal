"""Two real new-cohort states: paid Schur adjoint and verifier/cache equality."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from shared_backend.fft_maxwell import load_backend
from shared_backend.core import ActionSpec,VerifiedExchange
from route_b.audit import audit_state
from route_b.basis import build_basis
from route_b.controls import cached_physics,decode_coefficients
from route_b.preflight import write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort-root',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4);root=Path(a.cohort_root);m=json.loads((root/'generation_manifest.json').read_text());objects=[next(o for o in m['objects'] if o['split']=='train' and o['family']==family) for family in ('separated','near')];rows=[]
    for o in objects:
        item=o['states'][0];path=root/item['public'];audit=audit_state(path,o['object_id'],item['state_id']);meta=json.loads((Path(a.cache_root)/(o['object_id']+'_'+item['state_id']+'.json')).read_text());b=load_backend(path);s=b.freeze_state(b.get_receiver_basis(8),object_id=o['object_id'],state_id=item['state_id'],split='train');basis=build_basis(b,s);cache=cached_physics(Path(a.cache_root)/(o['object_id']+'_'+item['state_id']+'.npz'));drop,maxgroup=max(enumerate(meta['teachers']),key=lambda q:q[1]['best_gain']);t=decode_coefficients(maxgroup['modes'][0]['coefficients']);action=ActionSpec('B','audit',s.configuration_id,(drop,),0.,coefficient_real=tuple(t.real),coefficient_imag=tuple(t.imag),basis_gauge_sha256=basis.basis_gauge_sha256);r=VerifiedExchange(b,s).evaluate(action);exact=cache.numpy_metrics(t,(drop,));error=abs(r['gain']-exact['gain'])/max(abs(exact['gain']),1e-12);step_error=np.linalg.norm(r['step']-exact['step'])/max(np.linalg.norm(exact['step']),1e-12)
        audit.update(selected_gain_relative_error=error,selected_step_relative_error=step_error,selected_verifier_cost=r['cost'],selected_gain=r['gain'],selected_tau=r['tau'],selected_normal_relative_residual=r['normal_relative_residual'],status='PASS' if audit['status']=='PASS' and max(error,step_error)<=1e-9 else 'FAIL');rows.append(audit)
    write(a.output,{'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','rows':rows,'scope':'Real new-cohort two-state physical implementation audit; not held-out quality','locked_test_opened':False});print(json.dumps({'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'}))
