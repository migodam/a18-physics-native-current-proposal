"""Fixed coefficient-only MLP. Physical currents are instantiated after top-K."""
import torch
from torch import nn

class CoefficientMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(1552,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128),nn.GELU())
        self.outgoing=nn.Sequential(nn.Linear(160,128),nn.GELU(),nn.Linear(128,102))
        self.noop=nn.Linear(128,1)
    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def forward(self,xi,outgoing,rank_mask,legal_slots=None):
        h=self.encoder(xi);k=outgoing.shape[1];y=self.outgoing(torch.cat((h[:,None,:].expand(-1,k,-1),outgoing),-1)).reshape(len(xi),k,2,51)
        mask=torch.cat((rank_mask.new_ones((len(xi),1)),rank_mask),-1)[:,None,None,:]
        t=torch.complex(y[...,:25],y[...,25:50])*mask;norm=torch.linalg.vector_norm(t,dim=-1,keepdim=True);active=norm[...,0]>=1e-8;t=t/norm.clamp_min(1e-8)
        if legal_slots is not None:active&=legal_slots[:,:,None]
        return t,y[...,50].masked_fill(~active,-torch.inf),self.noop(h)[:,0],active
