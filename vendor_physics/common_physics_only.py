"""A8 ordinary-parameter, fixed-data-space contract. Units are physical, not fitted."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];A8=ROOT/'Gaussian/A8'
sys.path.insert(0,str(ROOT/'Gaussian/A6/code'))
# A18_B physics-only bridge: unrelated N-Port analytic_translation import omitted.
import numpy as np
from dataclasses import dataclass
import json,hashlib,time,platform,os
@dataclass
class Scene:
    k:float
    theta:np.ndarray # N by 6: Re alpha, Im alpha, log_s, cx,cy,cz
    radii:np.ndarray # port sphere
    material_radii:np.ndarray
    dirs:np.ndarray
    pols:np.ndarray
    receivers:np.ndarray
    obs_basis:np.ndarray # Rx, 2, xyz
    axes:np.ndarray # dimensionless positive shape axes, max <=1
    rotations:np.ndarray # N,3,3
    center_margin:float=.12
    family:str='easy'
    def scales(self):return np.tile(np.array([1.,.2,.2,.07,.07,.07]),(len(self.theta),1))
    def project(self,f):return np.einsum('rqc,rct->rqt',self.obs_basis,f).reshape(-1,f.shape[-1])
    def unproject(self,q):return np.einsum('rqc,rqt->rct',self.obs_basis,q.reshape(len(self.receivers),2,-1))
    def valid(self,theta):
        t=np.asarray(theta);d=t[:,3:]-self.theta[:,3:]
        if t.shape!=self.theta.shape or not np.isfinite(t).all():return False
        if np.any(t[:,0]<.05) or np.any(t[:,0]>30) or np.any(t[:,1]<.001) or np.any(t[:,1]>5) or np.any(t[:,2]<-2) or np.any(t[:,2]>.7) or np.any(np.abs(d)>self.center_margin):return False
        c=t[:,3:];dist=np.linalg.norm(c[:,None]-c[None,:],axis=-1);np.fill_diagonal(dist,np.inf)
        return bool(np.all(dist>self.radii[:,None]+self.radii[None,:]+1e-4))
    def save(self,path):
        path=Path(path);np.savez_compressed(path,**{k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)},k=self.k,center_margin=self.center_margin,family=self.family)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:return cls(**{k:(z[k].item() if z[k].ndim==0 else z[k]) for k in z.files})
def fibonacci(n,aperture='good'):
    z=1-2*(np.arange(n)+.5)/n
    if aperture=='limited':z=.6+.4*(z+1)/2
    if aperture=='weak':z=.95+.05*(z+1)/2
    a=np.arange(n)*np.pi*(3-np.sqrt(5));return np.c_[np.sqrt(1-z*z)*np.cos(a),np.sqrt(1-z*z)*np.sin(a),z]
def make_scene(N=16,Tx=24,seed=81001,family='easy',k=1.,gap=.4,rx=32):
    rng=np.random.default_rng(seed);side=int(np.ceil(np.sqrt(N)));i=np.arange(N);pitch=2+gap
    c=np.c_[(i%side)*pitch,(i//side)*pitch,np.zeros(N)];c-=c.mean(0)
    # Planar localized array is one important class; small fixed z disorder prevents exact plane symmetries.
    c[:,2]=rng.uniform(-.035,.035,N)
    alpha=1.5+.1*(i%4);loss=np.full(N,.1);s=.7+.03*(i%4)
    if family in ['hard','all-changing']:alpha=rng.uniform(1.3,2.1,N);s=rng.uniform(.65,.83,N)
    if family=='strong':alpha=np.full(N,10.);loss=np.full(N,.18)
    theta=np.c_[alpha,loss,np.log(s),c];ap='limited' if family=='hard' else ('weak' if family=='weak' else 'good');dirs=fibonacci(Tx,ap);pol=np.cross(dirs,np.array([0.,0.,1.]));pol/=np.linalg.norm(pol,axis=1)[:,None]
    rr=fibonacci(rx,ap);r0=2.5*max(np.max(np.linalg.norm(c,axis=1))+1.,2.);receivers=r0*rr
    b1=np.cross(rr,[0.,0.,1.]);b1/=np.linalg.norm(b1,axis=1)[:,None];b2=np.cross(rr,b1)
    return Scene(k,theta,np.ones(N),np.full(N,.7),dirs,pol,receivers,np.stack([b1,b2],axis=1),np.ones((N,3)),np.tile(np.eye(3),(N,1,1)),family=family)
class Costs:
    def __init__(self):self.seconds={};self.calls={};self.events=[]
    def add(self,key,t):self.seconds[key]=self.seconds.get(key,0.)+float(t);self.calls[key]=self.calls.get(key,0)+1
    def snapshot(self):return {'seconds':self.seconds.copy(),'calls':self.calls.copy(),'events':list(self.events)}
def realdot(a,b):return float(np.real(np.vdot(a,b)))
def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
def write_json(p,obj):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()));q.replace(p)
def write_npz(path,**arrays):
    p=Path(path);q=p.with_suffix(p.suffix+'.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**arrays)
    q.replace(p)
def provenance(out,config,files):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);write_json(out/'config.json',config)
    manifest={}
    dependencies=set(Path(p).resolve() for p in files)
    allowed=[ROOT/'Gaussian/A8/code',ROOT/'Gaussian/A6/code',ROOT/'Gaussian/A6/next_round/code']
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f:
            q=Path(f).resolve()
            if q.suffix=='.py' and any(q.is_relative_to(base) for base in allowed):dependencies.add(q)
    for p in sorted(dependencies):
        p=Path(p).resolve(); content=p.read_bytes(); rel=p.relative_to(ROOT); target=out/'source_snapshot'/rel; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(content); manifest[str(rel)]=hashlib.sha256(content).hexdigest()
    write_json(out/'source_manifest.json',manifest)
    write_json(out/'environment.json',{'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,'library_versions':{k:getattr(sys.modules.get(k),'__version__',None) for k in ['scipy','treams','torch']},'hardware':{'machine':platform.machine(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'host':platform.node()},'numpy_build':getattr(np.__config__,'CONFIG',None),'platform':platform.platform(),'pid':os.getpid(),'command':sys.argv,'blas_threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']},'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
