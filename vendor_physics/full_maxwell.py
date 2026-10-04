"""Fixed-grid FFT-DDA with analytic ordinary-parameter JVP and real VJP.
Finite discretization, not a continuous error certificate. No full measurement Jacobian.
"""
from common import *
from scipy.fft import fftn,ifftn
from scipy.sparse.linalg import LinearOperator,gmres
sys.path.insert(0,str(ROOT/'Gaussian/A6/next_round/code'))
from maxwell_kernel import maxwell_kernel_fft as fast_kernel_fft
class FullMaxwell:
    def __init__(self,scene,h=.175,rtol=1e-8):
        self.scene=scene;self.h=h;self.rtol=rtol;self.cost=Costs();t=time.perf_counter();c=scene.theta[:,3:];extent=scene.material_radii[:,None]*np.sqrt(np.einsum('nij,nj->ni',scene.rotations**2,scene.axes**2))+scene.center_margin
        lo=np.floor(np.min(c-extent,axis=0)/h).astype(int);hi=np.ceil(np.max(c+extent,axis=0)/h).astype(int);self.shape=tuple(hi-lo);self.memory_preflight();axes=[(np.arange(lo[i],hi[i])+.5)*h for i in range(3)];self.points=np.stack(np.meshgrid(*axes,indexing='ij'),-1);self.n=int(np.prod(self.shape)*3)
        self.fft,self.pad=fast_kernel_fft(self.shape,h,scene.k);self._theta=None;self._state=None;self.cost.add('kernel_build',time.perf_counter()-t);t=time.perf_counter();self.R=self._receiver();self.cost.add('observation_build',time.perf_counter()-t)
    def memory_preflight(self):
        try:
            import psutil
            available=int(psutil.virtual_memory().available)
        except ImportError:available=12*1024**3
        from scipy.fft import next_fast_len
        n=int(np.prod(self.shape));pad=int(np.prod([next_fast_len(2*x-1) for x in self.shape]));rx=len(self.scene.receivers)*2;tx=len(self.scene.dirs)
        # Peak alternatives: dyadic FFT build, observation rows+concatenate,
        # retained state fields and iterative solve workspace. Inclusive estimates.
        kernel=pad*(9*16*2+6*8+1); observation=pad*9*16+2*rx*3*n*16+15*n*16
        # CPU GMRES restart=160 retains a Krylov basis plus orthogonalization
        # workspace; this conservative 170-vector allowance is charged in the
        # host preflight rather than treated as free fallback memory.
        solve=pad*9*16+rx*3*n*16+2*tx*3*n*16+170*3*n*16
        estimate=int(max(kernel,observation,solve));limit=int(os.environ.get('A8_HOST_MEMORY_LIMIT_BYTES',str(int(.85*available))))
        self.cost.events.append({'kind':'host_memory_preflight','shape':self.shape,'estimated_peak_bytes':estimate,'available_or_conservative_default':available,'limit':limit})
        if estimate>limit:raise MemoryError(f'Full Maxwell shape {self.shape} estimated {estimate} bytes exceeds host budget {limit}')
    def conv(self,x,adjoint=False):
        x=x.reshape(self.shape+(3,));hat=[fftn(x[...,j],s=self.pad) for j in range(3)];cut=tuple(slice(0,n) for n in self.shape)
        out=[ifftn(sum((self.fft[j][i].conj() if adjoint else self.fft[i][j])*hat[j] for j in range(3)),s=self.pad)[cut] for i in range(3)]
        return np.stack(out,-1)
    def _receiver(self):
        src=self.points.reshape(-1,3);rows=[];k=self.scene.k
        for r,b in zip(self.scene.receivers,self.scene.obs_basis):
            dv=r-src;d=np.linalg.norm(dv,axis=1)
            if np.any(d==0):raise ValueError('receiver on source grid')
            n=dv/d[:,None];nn=n[:,:,None]*n[:,None,:];I=np.eye(3)
            green=np.exp(1j*k*d)[:,None,None]/(4*np.pi*d[:,None,None])*(k*k*(I-nn)+(1j*k/d-1/d**2)[:,None,None]*(I-3*nn));rows.append(np.einsum('qc,ncd->qnd',b,green).reshape(2,-1))
        return np.concatenate(rows,axis=0)
    def _material(self,theta):
        chi=np.zeros(self.shape,complex);parts=[];xyz=self.points
        for i,z in enumerate(theta):
            alpha=z[0]+1j*z[1];s=np.exp(z[2]);r=self.scene.material_radii[i];rot=self.scene.rotations[i];axes=self.scene.axes[i]
            # Work only on the compact support bounding box; retain sparse parameter derivatives.
            extent=r*np.sqrt((rot**2)@(axes**2));lo=xyz[0,0,0];a=np.maximum(0,np.ceil((z[3:]-extent-lo)/self.h).astype(int));b=np.minimum(self.shape,np.floor((z[3:]+extent-lo)/self.h).astype(int)+1)
            sl=tuple(slice(int(x),int(y)) for x,y in zip(a,b));pts=xyz[sl];q=(pts-z[3:])@rot;Q=rot@np.diag(1/(r*axes)**2)@rot.T;rho2=np.sum((q/(r*axes))**2,axis=-1);u=np.maximum(1-rho2,0);ex=np.exp(-rho2/(2*s*s));base=ex*u**3;chi[sl]+=alpha*base
            dbdr=ex*(-u**3/(2*s*s)-3*u*u)*(rho2<1);dcenter=-2*np.einsum('...j,jk->...k',pts-z[3:],Q)
            parts.append((sl,base,alpha*base*rho2/(s*s),alpha*dbdr[...,None]*dcenter))
        a0=3*self.h**3*chi/(chi+3);den=1-1j*self.scene.k**3*a0/(6*np.pi);a=a0/den;da=9*self.h**3/(chi+3)**2/den**2
        return a,da,parts
    def _inc(self,t):return np.exp(1j*self.scene.k*np.einsum('...j,j->...',self.points,self.scene.dirs[t]))[...,None]*self.scene.pols[t]
    def _op(self,x,adj=False):
        z=x.reshape(self.shape+(3,))
        return (z-self.conv(self.a.conj()[...,None]*z,True) if adj else z-self.a[...,None]*self.conv(z)).ravel()
    def _solve(self,b,adj=False,key='state_solve'):
        ts=time.perf_counter();count=[0];op=LinearOperator((self.n,self.n),matvec=lambda x:self._op(x,adj),dtype=complex)
        def callback(_):count[0]+=1
        x,info=gmres(op,b.ravel(),rtol=self.rtol,atol=0,restart=160,maxiter=150,callback=callback,callback_type='pr_norm');res=np.linalg.norm(op@x-b.ravel())/max(np.linalg.norm(b),1e-300)
        self.cost.add(key,time.perf_counter()-ts);self.cost.events.append({'kind':key,'krylov':count[0],'residual':float(res),'info':int(info),'restart':160,'maxiter':150})
        if info or not np.isfinite(res) or res>max(5*self.rtol,1e-12):raise RuntimeError(f'{key} residual {res} info {info}')
        return x.reshape(self.shape+(3,))
    def field(self,theta):
        theta=np.asarray(theta).reshape(self.scene.theta.shape)
        if self._theta is not None and np.array_equal(theta,self._theta):return self.F
        ts=time.perf_counter();self.a,self.da,self.parts=self._material(theta);self.cost.add('material_DT',time.perf_counter()-ts);ps=[];ws=[];fs=[]
        for t in range(len(self.scene.dirs)):
            inc=self._inc(t);p=self._solve(self.a[...,None]*inc);ps.append(p);ws.append(inc+self.conv(p));fs.append(self.R@p.ravel())
        self.p=np.stack(ps);self.w=np.stack(ws);self.F=np.column_stack(fs);self._theta=theta.copy();return self.F
    def _dchi(self,v):
        v=np.asarray(v).reshape(self.scene.theta.shape);dc=np.zeros(self.shape,complex)
        for d,(sl,base,ds,dcen) in zip(v,self.parts):dc[sl]+=(d[0]+1j*d[1])*base+d[2]*ds+np.einsum('...j,j->...',dcen,d[3:])
        return dc
    def jvp(self,theta,v):
        self.field(theta);ts=time.perf_counter();da=self.da*self._dchi(v);out=[]
        for w in self.w:out.append(self.R@self._solve(da[...,None]*w,key='jvp_solve').ravel())
        self.cost.add('jvp',time.perf_counter()-ts);return np.column_stack(out)
    def vjp(self,theta,q):
        self.field(theta);ts=time.perf_counter();q=np.asarray(q)
        if q.shape!=self.F.shape:raise ValueError('cotangent shape')
        weight=np.zeros(self.shape,complex)
        for t,w in enumerate(self.w):
            lam=self._solve((self.R.conj().T@q[:,t]).reshape(self.shape+(3,)),adj=True,key='adjoint_solve');weight+=np.sum(lam.conj()*w,axis=-1)*self.da
        g=np.zeros_like(self.scene.theta)
        for i,(sl,base,ds,dcen) in enumerate(self.parts):
            wi=weight[sl];g[i,0]=np.real(np.sum(wi*base));g[i,1]=np.real(np.sum(wi*(1j*base)));g[i,2]=np.real(np.sum(wi*ds));g[i,3:]=np.real(np.einsum('n,nj->j',wi.ravel(),dcen.reshape(-1,3)))
        self.cost.add('vjp',time.perf_counter()-ts);return g
