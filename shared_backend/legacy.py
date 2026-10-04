"""Pinned A17 G0; numerical sources are read-only external dependencies."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from .core import endpoint_from_maps,pack,unpack,directional_check

def pinned_a17(root):
    root=Path(root).resolve();sys.path.insert(0,str(root/"code"))
    import pinned_runtime
    if pinned_runtime.ROOT!=root:raise RuntimeError("different A17 runtime already imported")
    from maxwell_state import frozen
    return frozen,pinned_runtime

def run_g0(root,output,device="cuda"):
    frozen,runtime=pinned_a17(root)
    from endpoint_batch import EndpointBatch
    from exchange_core import gain_directional
    # Original A17 online rule, kept separate from the earlier R1 reference-floor rule.
    rows=[]
    for oid,phase in ((2007,"early"),(2012,"middle")):
        start=time.perf_counter();b=frozen(oid,phase,device);ctx=b["ctx"]
        Z=ctx.receiver_modes(10);AZ=Z.conj().T@ctx.apply_L(Z);OZ=ctx.apply_S(Z);PB=ctx.project_B(Z)
        NB=np.concatenate((PB,1j*PB),axis=2)
        r=unpack(ctx.r,ctx.P,ctx.m);Lambda=np.eye(ctx.dim_material)*ctx.lam
        old_ids=list(range(8));new_ids=list(range(7))+[8]
        coefficients=np.stack([np.eye(10,dtype=complex)[:,i] for i in (old_ids,new_ids)])
        ref=EndpointBatch(AZ,OZ,PB,ctx.r,ctx.ell,ctx.lam,device="cpu").evaluate(coefficients)
        if ref["status"]!=["OK","OK"]:raise RuntimeError("original endpoint invalid: "+str(ref["status"]))
        adapted=[]
        for ids in (old_ids,new_ids):
            adapted.append(endpoint_from_maps(AZ[np.ix_(ids,ids)],OZ[:,ids],NB[:,ids],r,ctx.ell,Lambda))
        rel=lambda a,c:float(np.linalg.norm(a-c)/max(np.linalg.norm(c),np.finfo(float).tiny))
        x=adapted[0]["step"];d=adapted[1]["step"]-x
        u=ctx.r+b["ev"].j(x);z=b["ev"].j(d)
        g=directional_check(ctx.r,u,z,x,d,ctx.ell,Lambda)
        original=gain_directional(u,z,x,d,Lambda,ctx.ell)
        er=[rel(a["step"],ref["steps"][i]) for i,a in enumerate(adapted)]
        ej=[rel(a["Jstep"],ref["j_steps"][i]) for i,a in enumerate(adapted)]
        gain_error=abs(g["gain"]-original)/max(abs(original),np.finfo(float).tiny)
        rng=np.random.default_rng(1801);v=rng.normal(size=ctx.dim_material);w=rng.normal(size=(ctx.P,ctx.n_current))+1j*rng.normal(size=(ctx.P,ctx.n_current))
        lhs=np.vdot(ctx.B(v),w).real;rhs=v@ctx.B_adjoint(w)
        ba=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-300)
        rows.append({"object_id":oid,"phase":phase,"device":device,"illuminations":ctx.P,"real_material_dimension":ctx.dim_material,"endpoint_relative_error":er,"Jstep_relative_error":ej,"finite_gain_relative_error":gain_error,"material_adjoint_relative_error":ba,"gain":g["gain"],"tau":g["tau"],"stability":[a["stability"] for a in adapted],"normal_relative_residual":[a["normal_relative_residual"] for a in adapted],"material_gauge":ctx.material_gauge,"inputs":b["inputs"],"state_hash":b["state_hash"],"wall_seconds":time.perf_counter()-start,"status":"PASS" if max(er+ej+[gain_error,ba])<=1e-9 else "FAIL"})
    # Exact scalar-lambda tau comparison to source-preserved efficiency rule.
    tolerance=Path(root).parent/"A17_EFFICIENCY_R1/code/online_tolerance.py"
    import importlib.util
    spec=importlib.util.spec_from_file_location("a17_tau_original",tolerance);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    tg=mod.directional_check(ctx.r,u,z,x,d,ctx.ell,ctx.lam)
    tau_err=abs(g["tau"]-float(tg["tau"][0]))/max(g["tau"],1e-300)
    out={"schema":"a18.G0.v1","status":"PASS" if all(r["status"]=="PASS" for r in rows) and tau_err<=1e-9 else "FAIL","threshold_relative":1e-9,"rows":rows,"tau_relative_error":tau_err,"tau_source_sha256":hashlib.sha256(tolerance.read_bytes()).hexdigest(),"source_manifest_sha256":hashlib.sha256((Path(root)/"SOURCE_MANIFEST.json").read_bytes()).hexdigest(),"original_sources_modified":False,"scope":"Two actual A17 frozen states, exact saved gauge; implementation G0, not proposal coverage or NN evidence"}
    q=Path(output);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)))
    print(json.dumps({"status":out["status"],"output":str(q),"tau_relative_error":tau_err}));return out

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();run_g0(a.root,a.output,a.device)
