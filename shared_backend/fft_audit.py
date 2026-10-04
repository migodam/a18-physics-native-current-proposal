"""Same-source G0 for the new FFT capability, separate from the A17 G0.

Only original numerical function/method AST nodes are executed. Platform-only
runtime and historical anchor/closure constructors are excluded, not patched.
"""
import ast,json,hashlib,time,contextlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.fft as fft
import scipy.linalg as la
import scipy.sparse.linalg as spla
from .fft_maxwell import load_backend
from .core import endpoint_from_maps,pack,directional_check,array_hash

def original_reference(source):
    tree=ast.parse(Path(source).read_text());nodes=[]
    names=("conv_batch","realify","gn_step","gain")
    methods=("L","observe","injection","material_pullback","solve")
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
        if isinstance(n,ast.ClassDef) and n.name=="FrozenPhysics":
            nodes.append(ast.ClassDef(name=n.name,bases=[],keywords=[],body=[m for m in n.body if isinstance(m,ast.FunctionDef) and m.name in methods],decorator_list=[]))
    env={"np":np,"fft":fft,"la":la,"spla":spla,"time":time,"CONFIG":{"physics":{"teacher_rtol":1e-10}}}
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));exec(compile(module,str(source),"exec"),env);return env

def run(public_root,source,output,devices=("cpu","cuda")):
    env=original_reference(source);rows=[]
    for oid in ("train_00","train_08"):
        path=Path(public_root)/(oid+"_state0.npz")
        for device in devices:
            start=time.perf_counter();b=load_backend(path,device=device);U=b.get_receiver_basis(9);old=U[:,:8];new=np.column_stack((U[:,:7],U[:,8]));s=b.freeze_state(old,object_id=oid,state_id="state0",receiver_mode_ids=tuple(range(8)))
            with np.load(path,allow_pickle=False) as data:raw={k:data[k].copy() for k in ("a","B","S","shape","pad","kernel","r","ell","Lambda")}
            ref=env["FrozenPhysics"]()
            for k,v in raw.items():setattr(ref,k,v)
            ref.shape=tuple(raw["shape"]);ref.pad=tuple(raw["pad"]);ref.n=len(U);ref.counter=SimpleNamespace(action=lambda *args:contextlib.nullcontext())
            rng=np.random.default_rng(1801);v=rng.normal(size=(len(U),2))+1j*rng.normal(size=(len(U),2));d=rng.normal(size=24);w=rng.normal(size=(len(U),4))+1j*rng.normal(size=(len(U),4))
            rel=lambda a,c:float(la.norm(a-c)/max(la.norm(c),1e-300))
            errors={"L":rel(b.apply_L(v),ref.L(v.T).T),"L_adjoint":rel(b.apply_L_adjoint(v),ref.L(v.T,True).T),"S":rel(b.apply_S(v),ref.observe(v.T).T),"B":rel(b.apply_B(d),ref.injection(d,False).T),"material_dual":rel(b.material_dual(w),ref.material_pullback(w.T,False))}
            eps=[];original_Js=[]
            for W in (old,new):
                D=W.conj().T@ref.L(W.T).T;NB=np.einsum("nk,ptn->tkp",W.conj(),ref.B);SW=ref.S@W
                C=np.stack([la.solve(D,n) for n in NB]);Jc=np.einsum("mk,pkd->pmd",SW,C).reshape(-1,24)
                step,_,record=env["gn_step"](Jc,ref.r,ref.ell,ref.Lambda)
                ep=endpoint_from_maps(W.conj().T@b.apply_L(W),b.apply_S(W),b.project_B(W),s.r,s.ell,s.Lambda)
                errors["endpoint_"+str(len(eps))]=rel(ep["step"],step);eps.append(ep);original_Js.append(step)
            # Original full GMRES vs new verifier directional capability.
            current,records=ref.solve(ref.injection(d,False));jref=pack(ref.observe(current).reshape(4,-1));jnew=b.full_J_action(d);errors["full_Jd"]=rel(jnew,jref)
            x=eps[0]["step"];dd=eps[1]["step"]-x
            cx,_=ref.solve(ref.injection(x,False));cd,_=ref.solve(ref.injection(dd,False));u=pack(s.r)+b.full_J_action(x);z=b.full_J_action(dd)
            un=pack(s.r)+pack(ref.observe(cx));zn=pack(ref.observe(cd));g=directional_check(pack(s.r),u,z,x,dd,s.ell,s.Lambda);gr=directional_check(pack(s.r),un,zn,x,dd,s.ell,s.Lambda)
            errors["finite_gain"]=abs(g["gain"]-gr["gain"])/max(abs(gr["gain"]),1e-300);errors["tau"]=abs(g["tau"]-gr["tau"])/max(gr["tau"],1e-300)
            rows.append({"object_id":oid,"state_id":"state0","device":device,"current_dof":len(U),"errors":errors,"gain":g["gain"],"tau":g["tau"],"normal_relative_residual":[ep["normal_relative_residual"] for ep in eps],"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"material_gauge_sha256":s.material_gauge_sha256,"wall_seconds":time.perf_counter()-start,"cost":b.receipt.snapshot(),"status":"PASS" if max(errors.values())<=1e-9 else "FAIL"})
    out={"schema":"a18.G0_FFT.v1","status":"PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL","threshold_relative":1e-9,"rows":rows,"reference_physics_source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),"reference_execution":"unaltered numerical AST nodes; portable import isolation only","historical_anchor_or_closure_used":False,"new_backend_online_teacher_members":False}
    q=Path(output);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2));print(json.dumps({"status":out["status"],"output":str(q)}));return out

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--public-root",required=True);p.add_argument("--source",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.public_root,a.source,a.output)
