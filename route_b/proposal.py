"""Cheap coefficient records; selected-only shared verifier materialization."""
import numpy as np
import torch
from .basis import build_basis,feature_vector
from .policy import select_top4

def propose(backend,state,model,normalizer,K=4):
    basis=build_basis(backend,state)
    if basis.Q.shape[1]==0:return [],basis
    xi,_=feature_vector(backend,state,basis);out,_=backend.outgoing_features(state);xi,out=normalizer.transform(xi,out);mask=np.r_[np.ones(basis.Q.shape[1]),np.zeros(24-basis.Q.shape[1])];legal=np.asarray([i not in state.protected_slots for i in range(state.U.shape[1])]);device=next(model.parameters()).device
    with torch.no_grad():t,logits,noop,_=model(torch.as_tensor(xi[None],dtype=torch.float32,device=device),torch.as_tensor(out[None],dtype=torch.float32,device=device),torch.as_tensor(mask[None],dtype=torch.float32,device=device),torch.as_tensor(legal[None],device=device))
    backend.receipt.add('coefficient_records',2*len(out));return select_top4(logits[0].cpu().numpy(),float(noop[0]),t[0].cpu().numpy(),state,basis,K),basis
