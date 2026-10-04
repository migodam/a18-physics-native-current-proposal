import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import linalg as la
from shared_backend.fft_maxwell import load_backend
from shared_backend.core import array_hash
from .basis import Root,build_basis,feature_vector

def audit_state(path,object_id,state_id):
    b=load_backend(path);s=b.freeze_state(b.get_receiver_basis(8),object_id=object_id,state_id=state_id);rng=np.random.default_rng(1801)
    q=rng.normal(size=len(s.U))+1j*rng.normal(size=len(s.U));w=rng.normal(size=len(s.U))+1j*rng.normal(size=len(s.U));root=Root(b,s)
    before=b.receipt.snapshot();fq=root.action(q);aw=root.action(w,True);err=abs(np.vdot(w,fq)-np.vdot(aw,q))/max(abs(np.vdot(w,fq)),abs(np.vdot(aw,q)),1e-300)
    action_cost=b.receipt.delta(before);basis=build_basis(b,s);x,_=feature_vector(b,s,basis)
    z=rng.normal(size=len(s.ell));v=rng.normal(size=(len(s.U),4))+1j*rng.normal(size=(len(s.U),4));lhs=np.vdot(b.apply_B(z),v).real;rhs=z@b.material_dual(v);adj=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-300)
    return {"object_id":object_id,"state_id":state_id,"current_dof":len(s.U),"feature_dimension":len(x),"basis_rank":basis.Q.shape[1],"Feff_adjoint_relative_error":float(err),"material_adjoint_relative_error":float(adj),"one_forward_adjoint_action_cost":action_cost,"orthogonality":float(la.norm(basis.Q.conj().T@basis.Q-np.eye(basis.Q.shape[1]))),"U_overlap":float(la.norm(s.U.conj().T@basis.Q)),"configuration_id":s.configuration_id,"basis_gauge_sha256":basis.basis_gauge_sha256,"forbidden_online_members_present":False,"historical_anchor_loaded":False,"status":"PASS" if max(err,adj)<1e-10 else "FAIL"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="config/pilot.json");p.add_argument("--public-root",required=True);p.add_argument("--output",default="results/route_b/physical_audit.json");a=p.parse_args();ids=("train_00","train_08");rows=[audit_state(Path(a.public_root)/(i+"_state0.npz"),i,"state0") for i in ids];q=Path(a.output);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps({"status":"PASS" if all(x["status"]=="PASS" for x in rows) else "FAIL","rows":rows},indent=2));print(q)

if __name__=="__main__":main()
