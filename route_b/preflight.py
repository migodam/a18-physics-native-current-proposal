"""Bounded physical development audit; no locked-test access or NN training."""
import argparse,hashlib,json,time,itertools,sys
from pathlib import Path
import numpy as np
import torch
from scipy import linalg as la
from shared_backend.fft_maxwell import load_backend
from shared_backend.core import pack,VerifiedExchange,directional_check
from shared_backend.dictionary import prepare_dictionary
from shared_backend.discrete_audit import audit_bank,ranked_records
from .basis import build_basis,feature_vector
from .small_physics import SmallPhysics,teacher_best
from .policy import propose_poly,POLICIES

DEVELOPMENT=(("train_00","separated"),("train_01","separated"),("train_08","near"),("train_09","near"),("train_16","high"),("test_geometry_00","geometry"))

def jsonable(x):
    if isinstance(x,np.ndarray):
        if np.iscomplexobj(x):return {"real":jsonable(x.real),"imag":jsonable(x.imag)}
        return jsonable(x.tolist())
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,complex):return {"real":x.real,"imag":x.imag}
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    return x

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(jsonable(obj),ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def opportunities(cache,deadline,seed):
    rows=[teacher_best(cache,(i,),seed=seed+101*i,deadline=deadline) for i in range(cache.k)]
    return {"best_gain":max(x["best_gain"] for x in rows),"drops":rows,"status":"BEST_FOUND"}

def action_cached_gain(cache,action):
    t=np.asarray(action.coefficient_real)+1j*np.asarray(action.coefficient_imag)
    return cache.numpy_gain(t,action.drop_indices[0])[0]

def a_data(b,s,Jfull):
    from route_a.features import extract
    from route_a.recipes import LazyCache,InvalidCurrent
    from route_a.bridge import ReceiptBackend
    from route_a.audit import AcquisitionAudit
    from route_a.baselines import analytic_actions,all_registered_policies
    bridge=ReceiptBackend(b);lazy=LazyCache();audit=AcquisitionAudit();start=time.perf_counter();before=b.receipt.snapshot()
    view=bridge.online_view(s,audit);features,recipes,audit=extract(view,bridge,lazy,audit)
    acquisition={"seconds":time.perf_counter()-start,"cost":b.receipt.delta(before)}
    bank=[];mask=[]
    for r in recipes:
        try:v=r.materialize(view,bridge,lazy,audit);bank.append(v);mask.append(True)
        except InvalidCurrent:bank.append(np.zeros(len(s.U),complex));mask.append(False)
    labels=audit_bank(b,s,np.column_stack(bank),Jfull);policies=all_registered_policies();policy_gains=[]
    ids={r.id:i for i,r in enumerate(recipes)}
    for p in policies:
        actions=analytic_actions(features,**p);policy_gains.append(max([labels["positive"][ids[a.recipe_id],a.drop_index] for a in actions],default=0.))
    return {"features":features,"recipes":recipes,"bank":np.column_stack(bank),"labels":labels,"policy_gains":policy_gains,"policies":policies,"acquisition":acquisition}

def benchmark(public,oid,sid,a17_root,policy,repetitions=5):
    """Fresh method order is rotated. Verification uses actual full GMRES Jd."""
    methods=("B-poly","D-full","D-recommended","A-analytic");rows=[]
    for rep in range(repetitions):
        for method in methods[rep%len(methods):]+methods[:rep%len(methods)]:
            start=time.perf_counter();b=load_backend(public);s=b.freeze_state(b.get_receiver_basis(8),object_id=oid,state_id=sid);state_seconds=time.perf_counter()-start;before=b.receipt.snapshot();acq=time.perf_counter();recipes={}
            if method=="B-poly":
                basis=build_basis(b,s);feature_vector(b,s,basis);actions=propose_poly(s,basis,family_order=policy[0],degree=policy[1],mixed=policy[2],drop_rule=policy[3])
            elif method.startswith("D-"):
                d=prepare_dictionary(b,s,a17_root);proxy=audit_bank(b,s,d["vectors"],d["anchor_J"],d["anchor_J"]);actions=ranked_records(d["vectors"],s,proxy,1 if method=="D-recommended" else 4);recipes={str(j):(lambda j=j:d["vectors"][:,j]) for j in range(d["vectors"].shape[1])}
            else:
                from route_a.features import extract
                from route_a.bridge import ReceiptBackend
                from route_a.recipes import LazyCache
                from route_a.audit import AcquisitionAudit
                from route_a.baselines import analytic_actions
                bridge=ReceiptBackend(b);cache=LazyCache();audit=AcquisitionAudit();view=bridge.online_view(s,audit);features,rs,audit=extract(view,bridge,cache,audit);actions=[a.to_shared() for a in analytic_actions(features)];recipes={r.id:(lambda r=r:r.materialize(view,bridge,cache,audit)) for r in rs}
            Tacq=time.perf_counter()-acq;acqcost=b.receipt.delta(before);verifier=VerifiedExchange(b,s,recipes);begin=time.perf_counter();results=[verifier.evaluate(a) for a in actions];verification=time.perf_counter()-begin
            # An exact-same-configuration cache hit only replays prior logits /
            # coefficient records. This is explicitly distinct from cold deploy.
            memo={s.configuration_id:tuple(actions)};wt=time.perf_counter();warm_actions=list(memo[s.configuration_id]);Twarm=time.perf_counter()-wt
            rows.append({"object_id":oid,"state_id":sid,"method":method,"repetition":rep,"state_seconds":state_seconds,"Tacq_cold":Tacq,"Tacq_warm_same_configuration":Twarm,"Tverify":verification,"Tinc":Tacq+verification,"Ttotal":state_seconds+Tacq+verification,"selected_count":len(actions),"gain":max([r.get("gain",0.) if r.get("accepted") else 0. for r in results],default=0.),"cost":b.receipt.snapshot(),"acquisition_cost":acqcost,"invalid_endpoints":[r for r in results if r["status"]!="OK"],"device":"native CPU full-vector FFT; GPU NN not invoked","scope":"initial frozen topK; D-recommended top1, cap3 transfer assessed separately"})
    return rows

def cov(rows,key,denom="union_reference"):
    objects={}
    for r in rows:
        ref=r[denom]
        if ref>0:objects.setdefault(r["object_id"],[]).append(r[key]/ref)
    return float(np.mean([np.mean(v) for v in objects.values()])) if objects else None

def run(args):
    torch.set_num_threads(4);root=Path(args.output);root.mkdir(parents=True,exist_ok=True);start=time.monotonic();deadline=start+60*args.teacher_budget_minutes
    if args.a_path:sys.path.insert(0,args.a_path)
    allrows=[];timings=[];saved=[]
    for obj,family in DEVELOPMENT:
        for st in range(2):
            if time.monotonic()>deadline:raise TimeoutError("shared CPU teacher120min cap")
            name=f"{obj}_state{st}";print(json.dumps({"event":"state_start","state":name,"elapsed":time.monotonic()-start}),flush=True)
            public=Path(args.public_root)/(name+".npz");b=load_backend(public);s=b.freeze_state(b.get_receiver_basis(8),object_id=obj,state_id=f"state{st}")
            with np.load(Path(args.teacher_root)/(name+".npz"),allow_pickle=False) as z:Jfull=pack(z["JF"].reshape(4,s.r.shape[1],len(s.ell)))
            full=build_basis(b,s);xi,_=feature_vector(b,s,full);cache=SmallPhysics(b,s,full.Q,Jfull)
            variants={};variant_specs=(("degree0",0,True,"joint"),("degree1",1,True,"joint"),("degree2",2,True,"joint"),("raw_PFP",2,False,"joint"),("O_only",2,True,"O"),("P_only",2,True,"P"),("M_only",2,True,"M"))
            for index,(key,degree,schur,side) in enumerate(variant_specs):
                basis=full if key=="degree2" else build_basis(b,s,degree,schur,side);sm=cache if key=="degree2" else SmallPhysics(b,s,basis.Q,Jfull)
                variants[key]={"rank":basis.Q.shape[1],**opportunities(sm,deadline,1801+index*1000)}
            percore=[]
            for drop in range(8):
                V=np.delete(s.U,drop,axis=1)
                try:
                    core=b.freeze_state(V,object_id=obj,state_id=f"state{st}");bc=build_basis(b,core,joint_against=s.U);cc=SmallPhysics(b,s,bc.Q,Jfull);r=teacher_best(cc,(drop,),seed=1901+drop,deadline=deadline);percore.append(r)
                except ValueError as exc:percore.append({"status":"CORE_UNSTABLE_DIRECT_FALLBACK","best_gain":variants["degree2"]["drops"][drop]["best_gain"],"reason":str(exc),"drops":[drop]})
            pairs=[]
            for i,pair in enumerate(itertools.combinations(range(4),2)):
                pairs.append(teacher_best(cache,pair,starts=4,steps=20,seed=2001+i,deadline=deadline))
            poly=[max([action_cached_gain(cache,a) for a in propose_poly(s,full,family_order=p[0],degree=p[1],mixed=p[2],drop_rule=p[3])],default=0.) for p in POLICIES]
            dictionary=prepare_dictionary(b,s,args.a17_root);dl=audit_bank(b,s,dictionary["vectors"],Jfull,dictionary["anchor_J"])
            aa=a_data(b,s,Jfull);B=max(variants["degree2"]["best_gain"],max(poly));per=max(r["best_gain"] for r in percore);original=dl["best_gain"];A=aa["labels"]["best_gain"];union=max(original,A,B,per,max(v["best_gain"] for v in variants.values()))
            row={"object_id":obj,"state_id":f"state{st}","family":family,"historical_split":"test" if obj.startswith("test") else "train","historical_exposure":True,"original_reference":original,"union_reference":union,"B_oracle":B,"B_per_core":per,"A_oracle":A,"B_poly_policy_gains":poly,"A_policy_gains":aa["policy_gains"],"dictionary_actual_pool_columns":dictionary["actual_pool_columns"],"basis_rank":full.Q.shape[1],"configuration_id":s.configuration_id,"basis_gauge_sha256":full.basis_gauge_sha256,"variants":variants,"per_core":percore,"pairs":pairs,"failures":{"dictionary":dl["failures"],"A":aa["labels"]["failures"]},"teacher_status":"BEST_FOUND","reference_scope":"audited singleton union: original dictionary, A1, all B basis variants, per-core and registered poly; pairs excluded as different action domain","locked_test_accessed":False}
            write(root/(name+".json"),row);allrows.append(row)
            outgoing=b.outgoing_features(s)[0];saved.append({"row":row,"xi":xi,"outgoing":outgoing,"basis":full,"cache":cache,"A":aa})
            # Teacher-only matrices are kept in a separate, ignored cache.
            q=Path(args.cache_root)/"teacher_only";q.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(q/(name+".npz"),**cache.numpy,basis_Q=full.Q,basis_layers=full.layers,xi=xi,outgoing=outgoing,A_gains=aa["labels"]["gains"],A_tau=aa["labels"]["tau"],A_valid=aa["labels"]["valid"])
            write(root/"progress.json",{"states_complete":len(allrows),"states_expected":12,"elapsed_seconds":time.monotonic()-start,"locked_test_accessed":False})
            print(json.dumps({"event":"state_complete","state":name,"B":B,"A":A,"D":original,"per_core":per,"elapsed":time.monotonic()-start}),flush=True)
    scores=[cov([{**r,"value":r["B_poly_policy_gains"][i]} for r in allrows],"value") for i in range(48)];best=int(np.argmax(scores));policy=POLICIES[best]
    for r in allrows:r["B_poly"]=r["B_poly_policy_gains"][best]
    for r in allrows:
        ts=benchmark(Path(args.public_root)/(r["object_id"]+"_"+r["state_id"]+".npz"),r["object_id"],r["state_id"],args.a17_root,policy,args.timing_repetitions);timings.extend(ts);write(root/"timings.json",timings)
    ratios=[]
    for r in allrows:
        rows=[t for t in timings if t["object_id"]==r["object_id"] and t["state_id"]==r["state_id"]];B=np.median([x["Tacq_cold"] for x in rows if x["method"]=="B-poly"]);D=np.median([x["Tacq_cold"] for x in rows if x["method"]=="D-full"]);ratios.append(float(B/D))
    near=[r for r in allrows if r["family"]=="near"];macro=cov(allrows,"B_oracle");near_cov=cov(near,"B_oracle");loss=cov(allrows,"B_per_core")-macro
    required={"macro_coverage":macro>=.90,"near_contact_coverage":near_cov>=.75,"shared_root_loss":loss<=.05,"Tacq_median_ratio":np.median(ratios)<=.5,"Tacq_P95_ratio":np.percentile(ratios,95)<=1.}
    summary={"schema":"a18.G1B.development.v1","status":"PASS" if all(required.values()) else "FAIL","requirements":required,"coverage_macro_union":macro,"coverage_near_union":near_cov,"coverage_macro_original_dictionary":cov(allrows,"B_oracle","original_reference"),"shared_root_loss_union":loss,"Tacq_median_ratio":float(np.median(ratios)),"Tacq_P95_ratio":float(np.percentile(ratios,95)),"B_poly_coverage_union":cov(allrows,"B_poly"),"frozen_B_poly_policy_id":best,"frozen_B_poly_policy":policy,"A_oracle_macro_union":cov(allrows,"A_oracle"),"A_oracle_near_union":cov(near,"A_oracle"),"A_oracle_macro_original":cov(allrows,"A_oracle","original_reference"),"states":allrows,"time_seconds":time.monotonic()-start,"teacher_budget_minutes":args.teacher_budget_minutes,"teacher_reference":"continuous BEST_FOUND, never global optimum","development_policy_selection":"48 policies selected on these development states; screening only, not unbiased generalization","locked_test_accessed":False,"G2":{"status":"STOP_NN" if cov(allrows,"B_poly")>=.95 else "HEADROOM","B_poly_coverage_union":cov(allrows,"B_poly"),"ridge_KNN":"PENDING_SEPARATE_CONTROL_AUDIT"}}
    write(root/"G1B.json",summary);print(json.dumps({"event":"preflight_complete","G1B":summary["status"],"macro":macro,"near":near_cov,"root_loss":loss,"acq_median":summary["Tacq_median_ratio"]}),flush=True);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="config/pilot.json");p.add_argument("--split",default="development",choices=["development"]);p.add_argument("--teacher-budget-minutes",type=float,default=120);p.add_argument("--public-root",required=True);p.add_argument("--teacher-root",required=True);p.add_argument("--a17-root",required=True);p.add_argument("--a-path");p.add_argument("--output",default="results/route_b/preflight");p.add_argument("--cache-root",default="cache/development");p.add_argument("--timing-repetitions",type=int,default=5);args=p.parse_args()
    try:run(args)
    except Exception as exc:
        import traceback
        write(Path(args.output)/"failure.json",{"status":"FAILURE","error":str(exc),"traceback":traceback.format_exc(),"locked_test_accessed":False});raise

if __name__=="__main__":main()
