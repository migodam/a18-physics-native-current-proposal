"""Train-parent-only scaling, with literal masks and slot IDs preserved."""
from dataclasses import dataclass,asdict
import numpy as np

LOG_NORM=np.r_[32,np.arange(36,48),51,52,59,60,61,np.arange(1416,1434)]
DISCRETE=np.r_[26,27,31,48,49,62,63,np.arange(1434,1540)]

def nonzero_medians(x):
    return np.asarray([np.median(c[c>0]) if np.any(c>0) else 1. for c in x.T])

@dataclass
class Normalizer:
    mean:np.ndarray
    std:np.ndarray
    scales:np.ndarray
    action_scale:float
    out_mean:np.ndarray
    out_std:np.ndarray
    out_scales:np.ndarray
    train_objects:tuple
    def scaled(self,x):
        x=np.array(x,copy=True);x[...,LOG_NORM]=np.log1p(np.expm1(x[...,LOG_NORM])/self.scales);x[...,1549]=np.log1p(x[...,1549]/self.action_scale);return x
    @classmethod
    def fit(cls,items):
        if any(s['split']!='train' for s in items):raise ValueError('normalizer may only fit train parents')
        x=np.stack([s['xi'] for s in items]);out=np.concatenate([s['outgoing'] for s in items]);scales=nonzero_medians(np.expm1(x[:,LOG_NORM]));action=nonzero_medians(x[:,1549,None])[0];os=nonzero_medians(out[:,16:18])
        obj=cls(np.zeros(1552),np.ones(1552),scales,float(action),np.zeros(32),np.ones(32),os,tuple(sorted({s['object_id'] for s in items})))
        x=obj.scaled(x);obj.mean=x.mean(0);obj.std=x.std(0);obj.std[obj.std<1e-12]=1;obj.mean[DISCRETE]=0;obj.std[DISCRETE]=1
        out=out.copy();out[:,16:18]=np.log1p(out[:,16:18]/os);obj.out_mean=out.mean(0);obj.out_std=out.std(0);obj.out_std[obj.out_std<1e-12]=1;obj.out_mean[:16]=0;obj.out_std[:16]=1;obj.out_mean[[28,29,30,31]]=0;obj.out_std[[28,29,30,31]]=1;return obj
    def transform(self,xi,outgoing):
        out=np.array(outgoing,copy=True);out[...,16:18]=np.log1p(out[...,16:18]/self.out_scales);return ((self.scaled(xi)-self.mean)/self.std,(out-self.out_mean)/self.out_std)
    def as_dict(self):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(self).items()}
    @classmethod
    def from_dict(cls,d):return cls(**{k:tuple(v) if k=='train_objects' else float(v) if k=='action_scale' else np.asarray(v) for k,v in d.items()})
