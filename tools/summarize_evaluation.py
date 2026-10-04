"""Object-level paired inference from saved frozen evaluations; no physical calls."""
import argparse,json
from pathlib import Path
import numpy as np
from route_b.preflight import write,cov

def summarize(root,reference_root,baseline,candidate,output):
    files=sorted(Path(root).glob('*_k*.json'));raw=[];states={}
    for file in files:
        record=json.loads(file.read_text());rows=record['rows'];raw.extend(rows)
        for row in rows:
            key=(row['object_id'],row['state_id'],row['rank'],row['method'])
            ref=Path(reference_root);ref=ref/('k'+str(row['rank'])) if (ref/('k'+str(row['rank']))).exists() else ref;meta=json.loads((ref/(row['object_id']+'_'+row['state_id']+'.json')).read_text());
            if meta.get('current_rank',8)!=row['rank']:raise RuntimeError('reference current rank mismatch')
            states[key]={'object_id':row['object_id'],'state_id':row['state_id'],'rank':row['rank'],'method':row['method'],'family':record['family'],'near':record['near'],'union_reference':meta['reference'],'original_reference':meta['original_reference']}
    for key,s in states.items():
        rows=[r for r in raw if (r['object_id'],r['state_id'],r['rank'],r['method'])==key];cold=[r for r in rows if r['cache_mode']=='cold'];warm=[r for r in rows if r['cache_mode']=='warm_same_configuration'];seq=[r for r in rows if r['cache_mode']=='cap3_sequence'];s.update(gain=float(np.median([r['gain'] for r in cold])),Tacq=float(np.median([r['Tacq'] for r in cold])),Tinc=float(np.median([r['Tinc'] for r in cold])),Tinc_warm=float(np.median([r['Tinc'] for r in warm])),cold_repetitions=len(cold),warm_repetitions=len(warm),sequence_gain=seq[0]['gain'] if seq else None,sequence_Tinc=seq[0]['Tinc'] if seq else None)
    allstates=list(states.values());ranks=sorted({s['rank'] for s in allstates});summaries={};bootstrap={}
    for rank in ranks:
        methods=sorted({s['method'] for s in allstates if s['rank']==rank});rs=[s for s in allstates if s['rank']==rank];summaries[str(rank)]={}
        for method in methods:
            group=[s for s in rs if s['method']==method];near=[s for s in group if s['near']];summaries[str(rank)][method]={'macro_union':cov(group,'gain'),'macro_original':cov(group,'gain','original_reference'),'near_union':cov(near,'gain'),'state_count':len(group),'object_count':len({s['object_id'] for s in group}),'all_five_cold_warm':all(s['cold_repetitions']==5 and s['warm_repetitions']==5 for s in group),'Tacq_median':float(np.median([s['Tacq'] for s in group])),'Tinc_median':float(np.median([s['Tinc'] for s in group])),'Tinc_warm_median':float(np.median([s['Tinc_warm'] for s in group]))}
        if baseline in methods and candidate in methods:
            objs=sorted({s['object_id'] for s in rs});differences=[]
            for oid in objs:
                a=[s for s in rs if s['object_id']==oid and s['method']==candidate];b=[s for s in rs if s['object_id']==oid and s['method']==baseline]
                if len(a)!=2 or len(b)!=2:continue
                differences.append(cov(a,'gain')-cov(b,'gain'))
            rng=np.random.default_rng(2026100418);samples=rng.choice(differences,(2000,len(differences)),replace=True).mean(1) if differences else np.array([]);ci=np.percentile(samples,[2.5,97.5]).tolist() if len(samples) else [None,None];bootstrap[str(rank)]={'candidate':candidate,'baseline':baseline,'object_count':len(differences),'replicates':2000,'point_difference':float(np.mean(differences)) if differences else None,'CI95':ci,'status':'HOLD' if len(differences)!=8 or ci[0]<=-.02 else 'POINT_THRESHOLDS_STILL_REQUIRED','scope':'Paired parent objects; two states aggregated before resampling. Eight parents are tiny feasibility only.'}
    result={'status':'RAW_EVIDENCE_SUMMARIZED','summaries':summaries,'paired_bootstrap':bootstrap,'states':allstates,'frozen_physical_rows':len(raw),'gain_scope':'Initial frozen singleton coverage; cap3 sequence fields reported separately.'};write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=summarize(a.root,a.reference_root,a.baseline,a.candidate,a.output);print(json.dumps({k:v for k,v in r.items() if k!='states'}))
