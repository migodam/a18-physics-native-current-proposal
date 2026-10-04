"""Train-parent fitted projector controls, sharing the MLP's allowed inputs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import linalg as la
from .normalizer import Normalizer
from .controls import currents_from_projector
from .preflight import cov,write

METHODS=tuple('ridge_'+str(l)+'_m'+str(m) for l in (1e-4,.01,1.) for m in (1,2))+('KNN3_objects_m1','KNN3_objects_m2')

def inputs(xi,out,norm):
    x,o=norm.transform(xi,out)
    return np.column_stack((np.repeat(x[None],len(o),axis=0),o))

def fit(train,output,dataset_sha256):
    if any(s['split']!='train' for s in train):raise ValueError('controls fit train parents only')
    norm=Normalizer.fit(train);X=[];Y=[];parents=[]
    for s in train:
        X.append(inputs(s['xi'],s['outgoing'],norm));targets=np.zeros((len(s['teachers']),1251))
        for i,modes in enumerate(s['teachers']):
            modes=[m for m in modes if m['gain']>0]
            total=sum(m['gain'] for m in modes)
            if total==0:continue
            P=np.zeros((25,25),complex)
            for m in modes:
                c=np.zeros(25,complex);c[:s['rank']+1]=m['coefficients'];P+=m['gain']/total*np.outer(c,c.conj())
            targets[i]=np.r_[P.real.ravel(),P.imag.ravel(),total/max(s['reference'],np.finfo(float).tiny)]
        Y.append(targets);parents.extend([s['object_id']]*len(targets))
    X=np.concatenate(X);Y=np.concatenate(Y);arrays={'X':X,'Y':Y,'parents':np.asarray(parents)}
    for lam in (1e-4,.01,1.):arrays['W_'+str(lam)]=la.solve(X@X.T+lam*np.eye(len(X)),Y,assume_a='pos')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    write(output.with_suffix('.json'),{'normalizer':norm.as_dict(),'train_parent_objects':sorted(set(parents)),'dataset_sha256':dataset_sha256,'schema':'a18.train_fitted_projectors.v1','weights_sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    return FittedControls(output)

class FittedControls:
    def __init__(self,path):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('control weights hash')
        self.norm=Normalizer.from_dict(meta['normalizer']);self.meta=meta
        with np.load(path,allow_pickle=False) as z:self.data={k:z[k].copy() for k in z.files}
    def predict(self,xi,out,method):
        T=inputs(xi,out,self.norm);d=self.data
        if method.startswith('ridge_'):return T@d['X'].T@d['W_'+method.split('_')[1]]
        predictions=[]
        for t in T:
            dist=la.norm(d['X']-t,axis=1);ids=[];parents=set()
            for i in np.argsort(dist,kind='stable'):
                if d['parents'][i] in parents:continue
                ids.append(i);parents.add(d['parents'][i])
                if len(ids)==3:break
            ds=dist[ids];w=(ds==0).astype(float) if np.any(ds==0) else 1/np.maximum(ds,1e-300);w/=w.sum();predictions.append(w@d['Y'][ids])
        return np.stack(predictions)
    def actions(self,xi,out,rank,legal,method,K=4):
        pred=self.predict(xi,out,method);mask=np.r_[np.ones(1+rank),np.zeros(24-rank)];records=[];modes=int(method[-1])
        for i,row in enumerate(pred):
            if not legal[i] or row[-1]<=0:continue
            for mode,(c,weight) in enumerate(currents_from_projector(row[:-1],mask,modes)):records.append((i,mode,c[:rank+1],float(row[-1]*weight)))
        return sorted(records,key=lambda r:(-r[3],r[0],r[1]))[:K]

def compare(train,val,root,output):
    start=time.perf_counter();root=Path(root);index=json.loads((root/'index.json').read_text());model=fit(train,Path(output)/'fitted_controls.npz',index['dataset_sha256']);rows=[]
    for s in val:
        row={'object_id':s['object_id'],'state_id':s['state_id'],'family':s['family'],'union_reference':s['reference']}
        for method in METHODS:
            gain=0.
            for i,m,c,score in model.actions(s['xi'],s['outgoing'],s['rank'],s['legal_slots'],method):
                r=s['cache'].numpy_metrics(c,(i,));gain=max(gain,r['gain'] if r['valid'] and r['gain']>r['tau'] else 0.)
            row[method]=gain
        for i,g in enumerate(s['B_poly_policy_gains']):row['poly_'+str(i)]=g
        rows.append(row)
    scores={key:cov(rows,key) for key in (*METHODS,*('poly_'+str(i) for i in range(48)))};best=max(scores,key=scores.get);near=[r for r in rows if r['family']=='near'];result={'status':'STOP_NN' if scores[best]>=.95 else 'HEADROOM','macro':scores[best],'near':cov(near,best),'best_method':best,'coverage_union':scores,'rows':rows,'train_parent_objects':model.meta['train_parent_objects'],'validation_parent_objects':sorted({s['object_id'] for s in val}),'dataset_sha256':index['dataset_sha256'],'locked_test_opened':False,'elapsed_seconds':time.perf_counter()-start,'scope':'All controls fitted on train parents; registered policies and control hyperparameters selected on validation.'};write(Path(output)/'G2_PILOT.json',result);return result

if __name__=='__main__':
    from .train import load_items
    p=argparse.ArgumentParser();p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True);args=p.parse_args();tr,va=load_items(args.cache_root);print(json.dumps({k:v for k,v in compare(tr,va,args.cache_root,args.output).items() if k not in ('rows','coverage_union')}))
