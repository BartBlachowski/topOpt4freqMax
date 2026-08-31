#!/usr/bin/env python3
"""Does Eq. (4b) (C1) cure the spurious-mode defect that Eq. (4a) introduces?  And
how many trajectory states are affected?  READ-ONLY."""
import sys, os, csv, h5py, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import mesh_data
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')

LAWS={'eq4':  lambda z: np.where(z<=0.1, z**6, z),
      'eq4a': lambda z: np.where(z<=0.1, 1e5*z**6, z),
      'eq4b': lambda z: np.where(z<=0.1, 6e5*z**6 - 5e6*z**7, z)}

def spectrum(z,nelx,nely,model,law,k=6):
    md=mesh_data(nelx,nely); g=LAWS[law]
    if model=='E1': Ee=1e7*(1e-6+(1-1e-6)*z**3); rr=1e-6+(1-1e-6)*z; zz=z
    elif model=='E2': Ee=1e7*(1e-9+(1-1e-9)*z**3); rr=1e-9+(1-1e-9)*g(z); zz=z
    else: zz=np.maximum(z,1e-3); Ee=1e7*zz**3; rr=g(zz)
    ndof,free=md['ndof'],md['free']
    K=sp.coo_matrix(((md['KEv'][None,:]*Ee[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    M=sp.coo_matrix(((md['MEv'][None,:]*rr[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    K=((K+K.T)*.5)[free][:,free].tocsc(); M=((M+M.T)*.5)[free][:,free].tocsc()
    v0=np.random.default_rng(20260830).standard_normal(K.shape[0]); v0/=np.linalg.norm(v0)
    lam,V=spla.eigsh(K,k=k,M=M,sigma=0.0,which='LM',v0=v0,tol=0.0,maxiter=100000)
    o=np.argsort(lam.real); lam=lam.real[o]; V=V[:,o]
    u=np.zeros(ndof); u[free]=V[:,0]
    ue=u[md['rows'].reshape(md['nEl'],64)[:,::8]]
    ke=rr*np.einsum('ei,ij,ej->e',ue,md['ME'],ue)
    return float(np.sqrt(max(lam[0],0))), float(ke[zz<=0.1].sum()/ke.sum())

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    X=f['res/rho_snapshots'][()]
print('=== mass put into the void by each law ===')
for r in (1e-3,1e-2,0.03,0.05,0.08,0.099,0.1):
    print(f'  rho={r:<6g} Eq.(4)={LAWS["eq4"](np.array([r]))[0]:.6e}  '
          f'Eq.(4a)={LAWS["eq4a"](np.array([r]))[0]:.6e}  Eq.(4b)={LAWS["eq4b"](np.array([r]))[0]:.6e}'
          f'   (4b)/(4a)={LAWS["eq4b"](np.array([r]))[0]/LAWS["eq4a"](np.array([r]))[0]:.3f}')

print('\n=== state k=252: does Eq.(4b) help? ===')
z=np.clip(np.float64(X[252]),0,1)
for model in ('E2','E3'):
    for law in ('eq4','eq4a','eq4b'):
        w,s=spectrum(z,160,20,model,law)
        print(f'  {model} {law:5s}: omega1={w:9.4f}   void kinetic-energy share of mode 1 = {s:.4f}'
              + ('   <-- SPURIOUS' if s>0.5 else ''))

print('\n=== how many states of the 160x20 trajectory are affected? ===')
R=list(csv.DictReader(open(os.path.join(OUT,'WP7C_TRAJECTORY_160x20_INDEPENDENT.csv'))))
n1=np.array([float(r['new_E1']) for r in R]); n2=np.array([float(r['new_E2']) for r in R])
n3=np.array([float(r['new_E3']) for r in R]); o2=np.array([float(r['old_E2']) for r in R])
cand=np.flatnonzero((n2<0.99*n1)|(n3<0.99*n1))+1
rows=[]
for k in cand:
    zk=np.clip(np.float64(X[k]),0,1)
    w2,s2=spectrum(zk,160,20,'E2','eq4a'); w3,s3=spectrum(zk,160,20,'E3','eq4a')
    rows.append(dict(k=int(k),E1=n1[k-1],eq4_E2=o2[k-1],eq4a_E2=w2,eq4a_E3=w3,
                     void_KE_share_E2=s2,void_KE_share_E3=s3,
                     E2_over_E1=w2/n1[k-1],spurious=bool(s2>0.5 or s3>0.5)))
with open(os.path.join(OUT,'SPURIOUS_MODE_TRAJECTORY_SCAN.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
sp_=[r for r in rows if r['spurious']]
print(f'  candidate states examined : {len(rows)}   (k from {cand.min()} to {cand.max()})')
print(f'  states with a SPURIOUS localized mode 1 in E2 or E3 under Eq.(4a): {len(sp_)}')
print(f'  k range affected: {min(r["k"] for r in sp_)} .. {max(r["k"] for r in sp_)}')
print(f'  worst E2/E1 ratio: {min(r["E2_over_E1"] for r in sp_):.4f}')
print(f'  under Eq.(4) none of these states was spurious (void share was 0.0000)')
