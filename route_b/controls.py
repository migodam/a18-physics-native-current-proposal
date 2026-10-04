"""Training-object-only ridge/projector KNN controls; development LOPO audit."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy import linalg as la
from .small_physics import SmallPhysics

def cached_physics(path,k=8):
    with np.load(path,allow_pickle=False) as z:data={key:z[key].copy() for key in ("A","O","N","Jfull","r","ell","Lambda","old_step")}
    obj=object.__new__(SmallPhysics);obj.numpy=data;obj.k=k;obj.rank=data["A"].shape[0]-k;obj.device="cpu"
    for key,v in data.items():setattr(obj,key,torch.as_tensor(v,dtype=torch.complex128 if np.iscomplexobj(v) else torch.float64))
    obj.u=obj.r+obj.Jfull@obj.old_step;obj.h=obj.ell+obj.Lambda@obj.old_step;return obj

def decode_coefficients(value):return (np.asarray(value["real"])+1j*np.asarray(value["imag"])).ravel()

def positive_tau_bound(cache):
    """Conservative upper bound on ORIGINAL tau for any positive finite gain.

    Complete the square in the common prior: positive gain bounds new data and
    prior energies by E_old, giving abs_terms <=16 E_old. This is used only to
    reject roundoff-sized offline opportunities; deployment retains exact tau.
    """
    a=cache.numpy;x=a["old_step"];u=a["r"]+a["Jfull"]@x;ell=a["ell"];L=a["Lambda"]
    shift=la.solve(L,ell,assume_a="pos");E=.5*(u@u+(x+shift)@L@(x+shift));base=a["r"]@a["r"]+u@u+abs(ell@x)+x@L@x;n=len(a["r"])+3*len(x)+16;f=np.finfo(float);gamma=n*f.eps/(1-n*f.eps)
    return 32*f.tiny+32*gamma*(base+16*E)

def records(root,cache_root):
    states=[]
    for p in sorted(Path(root).glob("*_state*.json")):
        row=json.loads(p.read_text());name=row["object_id"]+"_"+row["state_id"]
        path=Path(cache_root)/(name+".npz");sm=cached_physics(path)
        with np.load(path,allow_pickle=False) as z:xi=z["xi"].copy();out=z["outgoing"].copy()
        X=np.column_stack((np.repeat(xi[None],8,axis=0),out));Y=np.zeros((8,1250));mass=np.zeros(8);mask=np.r_[np.ones(1+sm.rank),np.zeros(24-sm.rank)]
        tau=positive_tau_bound(sm)
        for i,r in enumerate(row["variants"]["degree2"]["drops"]):
            modes=[m for m in r["modes"] if m["gain"]>tau]
            if not modes:continue
            P=np.zeros((25,25),complex);total=sum(m["gain"] for m in modes)
            for mode in modes:
                t=np.zeros(25,complex);t[:1+sm.rank]=decode_coefficients(mode["coefficients"]);P+=(mode["gain"]/total)*np.outer(t,t.conj())
            Y[i]=np.r_[P.real.ravel(),P.imag.ravel()];mass[i]=total/max(row["union_reference"],np.finfo(float).tiny)
        states.append({"row":row,"X":X,"Y":Y,"mass":mass,"mask":mask,"cache":sm})
    return states

def normalize(trainX,testX):
    mean=trainX.mean(0);std=trainX.std(0);std[std<1e-12]=1
    # Literal rank/layer/seed masks and outgoing slot IDs stay discrete.
    idx=np.r_[np.arange(1434,1536),np.arange(1552,1568)];mean[idx]=0;std[idx]=1
    return (trainX-mean)/std,(testX-mean)/std

def currents_from_projector(y,mask,modes=2):
    P=y[:625].reshape(25,25)+1j*y[625:].reshape(25,25);P=.5*(P+P.conj().T);P*=mask[:,None]*mask[None,:]
    if not np.isfinite(P).all():return []
    eig,v=la.eigh(P);out=[]
    for i in range(1,modes+1):
        t=v[:,-i]*mask
        if eig[-i]>0 and la.norm(t)>=1e-8:out.append((t/la.norm(t),float(eig[-i])))
    return out

def selected_gain(state,Y,mass,modes=2):
    actions=[];gain=0.;tau=positive_tau_bound(state["cache"])
    for i in range(8):
        if mass[i]<=0:continue
        for j,(t,weight) in enumerate(currents_from_projector(Y[i],state["mask"],modes)):actions.append((mass[i]*weight,i,j,t))
    for _,i,j,t in sorted(actions,key=lambda a:(-a[0],a[1],a[2]))[:4]:
        g,valid=state["cache"].numpy_gain(t[:1+state["cache"].rank],i)
        if valid and g>tau:gain=max(gain,g)
    return gain

def run(root,cache_root,output):
    from .preflight import write,cov
    torch.set_num_threads(4);start=time.perf_counter();states=records(root,cache_root);objects=sorted({s["row"]["object_id"] for s in states});rows=[]
    for s in states:
        train=[t for t in states if t["row"]["object_id"]!=s["row"]["object_id"]];X=np.concatenate([t["X"] for t in train]);Y=np.concatenate([t["Y"] for t in train]);mass=np.concatenate([t["mass"] for t in train]);obj=np.concatenate([np.repeat(t["row"]["object_id"],8) for t in train]);XX,XT=normalize(X,s["X"])
        row={"object_id":s["row"]["object_id"],"state_id":s["row"]["state_id"],"family":s["row"]["family"],"union_reference":s["row"]["union_reference"],"train_parent_objects":sorted(set(obj)),"test_parent_in_train":False}
        for lam in (1e-4,1e-2,1.):
            W=la.solve(XX@XX.T+lam*np.eye(len(XX)),np.column_stack((Y,mass)),assume_a="pos");pred=XT@XX.T@W
            for modes in (1,2):row["ridge_"+str(lam)+"_m"+str(modes)]=selected_gain(s,pred[:,:-1],pred[:,-1],modes)
        ys=[];ms=[]
        for x in XT:
            dist=la.norm(XX-x,axis=1);order=np.argsort(dist,kind="stable");ids=[];parents=set()
            for i in order:
                if obj[i] in parents:continue
                ids.append(i);parents.add(obj[i])
                if len(ids)==3:break
            ds=dist[ids];w=(ds==0).astype(float) if np.any(ds==0) else 1/np.maximum(ds,1e-300);w/=w.sum();ys.append(w@Y[ids]);ms.append(w@mass[ids])
        for modes in (1,2):row["KNN3_objects_m"+str(modes)]=selected_gain(s,np.stack(ys),np.asarray(ms),modes)
        rows.append(row)
    keys=["ridge_"+str(lam)+"_m"+str(modes) for lam in (1e-4,1e-2,1.) for modes in (1,2)]+["KNN3_objects_m"+str(modes) for modes in (1,2)]
    coverage={m:cov(rows,m) for m in keys};best=max(coverage,key=coverage.get)
    summary={"schema":"a18.G2.controls.development.v1","rows":rows,"coverage_union":coverage,"best_control":best,"best_coverage":coverage[best],"status":"STOP_NN" if coverage[best]>=.95 else "HEADROOM","method":"leave-one-parent-object-out; normalization fits remaining parents only; hyperparameters compared on development, not independent test","teacher_labels":"positive separated modes, Hermitian projectors; original-tau conservative bound filters numerical zeros","offline_seconds":time.perf_counter()-start,"locked_test_accessed":False,"actual_online_cost_status":"NOT_RUN_UNTIL_FROZEN_DEPLOYMENT_EVALUATION"}
    write(output,summary);print(json.dumps({"G2_controls":summary["status"],"coverage":coverage}));return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default="results/route_b/preflight");p.add_argument("--cache-root",default="cache/development/teacher_only");p.add_argument("--output",default="results/route_b/preflight/G2_controls.json");a=p.parse_args();run(a.root,a.cache_root,a.output)
