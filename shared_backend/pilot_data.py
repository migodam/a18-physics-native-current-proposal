"""One shared, gated cohort using the original A18-B full-vector rules.

Truth, noisy measurements and full J are isolated from public online states.
Locked-test states are generated into a sealed subtree; cache preparation and
training are forbidden from reading that subtree before the freeze receipt.
"""
import argparse,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy import linalg as la
from .fft_maxwell import backend_from_arrays
from .core import pack,unpack

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));tmp.replace(path)

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_original(root):
    root=Path(root)
    for row in json.loads((root/'SOURCE_MANIFEST.json').read_text()):
        if file_hash(root/row['file'])!=row['sha256']:raise RuntimeError('original source hash mismatch')
    for name,filename in (('common','common_physics_only.py'),('maxwell_kernel','maxwell_kernel.py'),('a18_original_full_maxwell','full_maxwell.py')):
        spec=importlib.util.spec_from_file_location(name,root/filename);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules['common'],sys.modules['a18_original_full_maxwell'].FullMaxwell

def generate_scene(spec,cfg,common):
    rng=np.random.default_rng(spec['seed']);near=spec['family']=='near' or spec.get('near',False)
    sc=common.make_scene(N=4,Tx=4,seed=spec['seed'],k=1.,gap=.05 if near else .4,rx=8)
    if near:sc.material_radii[:]=.95
    if spec['family']=='geometry':
        sc.axes[:]=cfg['geometry_axes']
        for i,angle in enumerate(rng.uniform(-np.pi,np.pi,4)):
            c=np.cos(angle);s=np.sin(angle);sc.rotations[i]=[[c,-s,0],[s,c,0],[0,0,1]]
    bounds=cfg['alpha_test_high'] if spec['family']=='test_high' else cfg['alpha_train_high'] if spec['family']=='high' else cfg['alpha_ordinary']
    sc.theta[:,0]=rng.uniform(*bounds,4);sc.theta[:,1]=rng.uniform(*cfg['loss'],4);sc.theta[:,2]=np.log(rng.uniform(*cfg['width'],4))
    sc.dirs=common.fibonacci(4);sc.pols=np.cross(sc.dirs,[0.,0.,1.]);sc.pols/=la.norm(sc.pols,axis=1)[:,None];rr=common.fibonacci(8);sc.receivers=7*rr;b1=np.cross(rr,[0.,0.,1.]);b1/=la.norm(b1,axis=1)[:,None];sc.obs_basis=np.stack((b1,np.cross(rr,b1)),axis=1)
    if not sc.valid(sc.theta):raise ValueError('invalid declared truth')
    perturb=rng.uniform(-1,1,sc.theta.shape)*np.asarray(cfg['initial_perturbation_bounds'])
    for halves in range(cfg['initial_feasibility_halvings']+1):
        initial=sc.theta+perturb*2.**(-halves)
        if sc.valid(initial):return sc,initial,halves
    raise ValueError('initial feasibility failure')

def frozen_arrays(engine,theta,noisy,whitening,prior,reg):
    field=engine.field(theta).copy();shape=engine.shape;p=theta.size;scale=engine.scene.scales().ravel();chi=np.zeros(shape,complex);B=np.empty((p,4,engine.n),complex)
    for i,(sl,base,ds,dcen) in enumerate(engine.parts):chi[sl]+=(theta[i,0]+1j*theta[i,1])*base
    for j in range(p):
        d=np.zeros(p);d[j]=scale[j];derivative=engine.da*engine._dchi(d.reshape(theta.shape));B[j]=(derivative[None,...,None]*engine.w).reshape(4,engine.n)
    return {'a':engine.a.copy(),'chi':chi,'S':engine.R*whitening,'B':B,'xyz':engine.points.copy(),'kernel':np.asarray(engine.fft),'shape':np.asarray(shape),'pad':np.asarray(engine.pad),'k':np.asarray(engine.scene.k),'h':np.asarray(engine.h),'r':((field-noisy)*whitening).T.ravel(),'ell':reg*(theta-prior).ravel()/scale,'Lambda':reg*np.eye(p),'theta':theta.copy(),'scale':scale}

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path);return file_hash(path)

def generate(args):
    start=time.perf_counter();plan=json.loads(Path(args.plan).read_text());gate=json.loads(Path(args.gate_receipt).read_text())
    if gate['gates']['G0']['status']!='PASS' or not any(gate['gates'][k]['status']=='PASS' for k in ('G1A','G1B')):raise RuntimeError('cohort requires real G0 and representation/cost PASS')
    if gate['new_cohort_plan_sha256']!=file_hash(args.plan):raise RuntimeError('manifest plan changed')
    physical=json.loads(Path(args.physical_config).read_text());cfg=physical['dataset'];reg=physical['physics']['lambda'];common,FullMaxwell=load_original(args.original_root);root=Path(args.output);index=[]
    if (root/'generation_manifest.json').exists():raise RuntimeError('cohort already generated; do not regenerate or overwrite')
    for spec in plan['objects']:
        if time.perf_counter()-start>args.minutes_cap*60:raise TimeoutError('physics preparation cap')
        oid=spec['object_id'];split=spec['split'];entry={**spec,'states':[]};obj_start=time.perf_counter()
        try:
            sc,initial,halves=generate_scene(spec,cfg,common);engine=FullMaxwell(sc,h=.35,rtol=1e-10);clean=engine.field(sc.theta).copy();sigma=.01*np.sqrt(np.mean(abs(clean)**2));rng=np.random.default_rng(spec['seed']+90000);noisy=clean+sigma*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))/np.sqrt(2);whitening=np.sqrt(2)/sigma;theta=initial.copy();evaluator=root/split/'evaluator';evaluator.mkdir(parents=True,exist_ok=True);sc.save(evaluator/(oid+'_scene.npz'));save_npz(evaluator/(oid+'_acquisition.npz'),noisy=noisy,clean=clean,initial=initial,whitening=whitening)
            trials=[]
            for st in range(2):
                arrays=frozen_arrays(engine,theta,noisy,whitening,initial,reg);b=backend_from_arrays(arrays);J=b.full_J_action(np.eye(24));r=pack(arrays['r'].reshape(4,16));H=J.T@J+arrays['Lambda'];rhs=J.T@r+arrays['ell'];step=-la.cho_solve(la.cho_factor(H,lower=True),rhs);nr=la.norm(H@step+rhs)/max(la.norm(rhs),np.finfo(float).tiny)
                # One independent original JVP per state verifies the source-preserving fast assembly.
                probe=np.random.default_rng(spec['seed']+st).normal(size=24);probe/=la.norm(probe);original=pack((engine.jvp(theta,(probe*arrays['scale']).reshape(4,6))*whitening).T);error=la.norm(J@probe-original)/max(la.norm(original),np.finfo(float).tiny)
                if error>1e-9 or nr>1e-10:raise RuntimeError('full J/normal residual accuracy gate')
                name=oid+'_state'+str(st)+'.npz';public=root/split/'public'/name;teacher=root/split/'teacher'/name;ph=save_npz(public,**arrays);th=save_npz(teacher,JF=unpack(J,4,16).reshape(64,24))
                entry['states'].append({'state_id':'state'+str(st),'public':public.relative_to(root).as_posix(),'teacher':teacher.relative_to(root).as_posix(),'public_sha256':ph,'teacher_sha256':th,'current_dof':engine.n,'original_JVP_relative_error':float(error),'normal_relative_residual':float(nr),'full_J_cost':b.receipt.snapshot(),'teacher_max_relative_residual':max(x['relative_residual'] for x in b.solve_records)})
                if st==0:
                    trust=min(1.,cfg['trust_scaled_inf']/max(np.max(abs(step)),1e-300));phi0=.5*r@r;slope=rhs@step;accepted=None
                    for j in range(cfg['line_search_halvings']):
                        alpha=trust*.5**j;candidate=initial+(alpha*step*arrays['scale']).reshape(4,6);trial={'halving':j,'alpha':float(alpha),'valid':sc.valid(candidate)}
                        if trial['valid']:
                            fc=engine.field(candidate).copy();rr=((fc-noisy)*whitening).T.ravel();xi=(candidate-initial).ravel()/arrays['scale'];phi=.5*np.vdot(rr,rr).real+.5*xi@arrays['Lambda']@xi;trial['objective']=float(phi);trial['armijo_limit']=float(phi0+cfg['armijo']*alpha*slope)
                            if phi<=trial['armijo_limit']:accepted=candidate;trials.append(trial);break
                        trials.append(trial)
                    if accepted is None:raise RuntimeError('no legal true-forward intermediate state')
                    theta=accepted
            entry.update(generation_status='COMPLETE',initial_feasibility_halvings=halves,line_search=trials,source_engine_cost=engine.cost.snapshot(),wall_seconds=time.perf_counter()-obj_start)
        except Exception as exc:
            entry.update(generation_status='FAILURE',error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-obj_start)
        index.append(entry);write(root/'generation_progress.json',{'objects':index,'elapsed_seconds':time.perf_counter()-start,'locked_test_opened':False});print(json.dumps({'event':'object_generated','object_id':oid,'status':entry['generation_status'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
    result={'schema':'a18.shared_cohort.v1','master_seed':plan['master_seed'],'plan_sha256':file_hash(args.plan),'counts':plan['counts'],'objects':index,'status':'PASS' if all(x['generation_status']=='COMPLETE' for x in index) else 'FAIL','elapsed_seconds':time.perf_counter()-start,'locked_test_opened':False,'locked_test_scope':'physically generated and sealed; no feature, rule, model or metric selection from locked data','no_old_anchor_or_closure':True};write(root/'generation_manifest.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='manifests/new_cohort_plan.json');p.add_argument('--physical-config',default='config/physical_generation.json');p.add_argument('--original-root',default='vendor_physics');p.add_argument('--gate-receipt',required=True);p.add_argument('--output',required=True);p.add_argument('--minutes-cap',type=float,default=55);a=p.parse_args();generate(a)
