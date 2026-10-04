"""J0: actual cached mini-batches; discard first five of twenty timing samples."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from shared_backend.isolation import cached_only_epoch
from .train import load_items,to_device,pack,authorize
from .model import CoefficientMLP
from .normalizer import Normalizer
from .loss import physical_loss
from .fitted_controls import compare
from .preflight import write

def run(args):
    authorize(args.config,args.gate_receipt,args.queue_owner,'J0');torch.set_num_threads(4);torch.manual_seed(1801);torch.use_deterministic_algorithms(True)
    tr,va=load_items(args.cache_root);out=Path(args.output);baseline=compare(tr,va,args.cache_root,out)
    if baseline['status']=='STOP_NN':
        result={'status':'NOT_RUN_G2_PILOT_NO_HEADROOM','baseline':baseline['macro']};write(out/'J0.json',result);return result
    norm=Normalizer.fit(tr);batch=tr[:8];times={};accuracy=[];peaks={}
    for device in ('cpu','cuda'):
        if device=='cuda' and not torch.cuda.is_available():continue
        torch.manual_seed(1801);model=CoefficientMLP().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);elapsed=[]
        for s in batch:to_device(s['cache'],device)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with cached_only_epoch() as counter:
            for step in range(20):
                if device=='cuda':torch.cuda.synchronize()
                begin=time.perf_counter();t,l,n,_=model(*pack(batch,norm,device));loss=torch.stack([physical_loss(t[j],l[j],n[j],s['cache'],s['teachers'],s['reference'],s['basis_layers'])['loss'] for j,s in enumerate(batch)]).mean();optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if device=='cuda':torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-begin)
            assert counter['Maxwell_action_count']==0
        times[device]={'samples':20,'discarded':5,'measured_median_seconds':float(np.median(elapsed[5:])),'raw_seconds':elapsed,'Maxwell_action_count':counter['Maxwell_action_count']}
        peaks[device]=torch.cuda.max_memory_allocated() if device=='cuda' else 0
        s=batch[0];c=t[0,0,0,:s['rank']+1].detach().to(torch.complex128);g,valid,_,_=s['cache'].evaluate(c,(0,));exact=s['cache'].numpy_metrics(c.cpu().numpy(),(0,));err=abs(float(g.detach())-exact['gain'])/max(abs(exact['gain']),1e-12);accuracy.append({'device':device,'FP64_cached_gain_relative_error':err,'valid_equal':bool(valid)==exact['valid']})
    chosen=min(times,key=lambda d:times[d]['measured_median_seconds']);result={'status':'PASS' if all(x['FP64_cached_gain_relative_error']<=1e-8 and x['valid_equal'] for x in accuracy) else 'FAIL','chosen_device':chosen,'measurements':times,'accuracy':accuracy,'peak_gpu_allocated_bytes':peaks,'parameters':249703,'baseline_macro':baseline['macro'],'baseline_method':baseline['best_method'],'locked_test_opened':False,'epoch_cap':80,'each_J2_seconds_cap':1350};write(out/'J0.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/pilot.json');p.add_argument('--gate-receipt',required=True);p.add_argument('--queue-owner',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a);print(json.dumps(r));raise SystemExit(0 if r['status'] in ('PASS','NOT_RUN_G2_PILOT_NO_HEADROOM') else 1)
