#!/usr/bin/env python3
"""WP24 - remedy feasibility probe (ARPACK shift-invert; the same quantities were
cross-validated against dense LAPACK for the eq4/eq4a cases in wp_spurious2.py).
Bounds the remedy space with evidence.  READ-ONLY, no optimizer."""
import sys, os, csv, h5py, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import mesh_data
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')
G={'linear':lambda x:x,
   'eq4':   lambda x:np.where(x<=0.1,x**6,x),
   'eq4a':  lambda x:np.where(x<=0.1,1e5*x**6,x),
   'eq4b':  lambda x:np.where(x<=0.1,6e5*x**6-5e6*x**7,x)}

def probe(z,nelx,nely,model,law,k=6):
    md=mesh_data(nelx,nely)
    if model=='E1': Ee=1e7*(1e-6+(1-1e-6)*z**3); rr=1e-6+(1-1e-6)*G[law](z); zz=z
    elif model=='E2': Ee=1e7*(1e-9+(1-1e-9)*z**3); rr=1e-9+(1-1e-9)*G[law](z); zz=z
    else: zz=np.maximum(z,1e-3); Ee=1e7*zz**3; rr=G[law](zz)
    ndof,free=md['ndof'],md['free']
    K=sp.coo_matrix(((md['KEv'][None,:]*Ee[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    M=sp.coo_matrix(((md['MEv'][None,:]*rr[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    K=((K+K.T)*.5)[free][:,free].tocsc(); M=((M+M.T)*.5)[free][:,free].tocsc()
    v0=np.random.default_rng(20260830).standard_normal(K.shape[0]); v0/=np.linalg.norm(v0)
    lam,V=spla.eigsh(K,k=k,M=M,sigma=0.0,which='LM',v0=v0,tol=0.0,maxiter=200000)
    o=np.argsort(lam.real); lam=lam.real[o]; V=V[:,o]
    u=np.zeros(ndof); u[free]=V[:,0]
    ue=u[md['rows'].reshape(md['nEl'],64)[:,::8]]
    ke=rr*np.einsum('ei,ij,ej->e',ue,md['ME'],ue)
    return float(np.sqrt(max(lam[0],0))), float(ke[zz<=0.1].sum()/ke.sum())

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    X=f['res/rho_snapshots'][()]
# the six worst affected states plus a converged control
KS=[252,254,253,256,255,257,1600]
rows=[]
print(f"{'k':>5} {'field':>8} {'model':>5} {'mass law':>8} {'omega_1':>10} {'void KE':>9}  verdict")
for k in KS:
    z=np.clip(np.float64(X[k]),0,1)
    nS=int(round(0.5*z.size)); order=np.lexsort((np.arange(z.size),-z))
    zb=np.zeros_like(z); zb[order[:nS]]=1.0
    for model in ('E1','E2','E3'):
        for law in ('eq4','eq4a','eq4b','linear'):
            w,s=probe(z,160,20,model,law)
            v='SPURIOUS' if s>0.5 else 'structural'
            rows.append(dict(k=k,field='raw',model=model,mass_law=law,omega1=w,void_KE_share=s,verdict=v))
            print(f"{k:5d} {'raw':>8} {model:>5} {law:>8} {w:10.4f} {s:9.4f}  {v}")
        w,s=probe(zb,160,20,model,'eq4a')
        v='SPURIOUS' if s>0.5 else 'structural'
        rows.append(dict(k=k,field='binary_exact_count',model=model,mass_law='eq4a',omega1=w,void_KE_share=s,verdict=v))
        print(f"{k:5d} {'binary':>8} {model:>5} {'eq4a':>8} {w:10.4f} {s:9.4f}  {v}")
    print()
with open(os.path.join(OUT,'REMEDY_FEASIBILITY_PROBE.csv'),'w',newline='') as f:
    w_=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
R=list(csv.DictReader(open(os.path.join(OUT,'WP7C_TRAJECTORY_160x20_INDEPENDENT.csv'))))
o1=np.array([float(r['old_E1']) for r in R]); o2=np.array([float(r['old_E2']) for r in R])
o3=np.array([float(r['old_E3']) for r in R]); n2=np.array([float(r['new_E2']) for r in R])
print('=== control over all 1600 states ===')
print(f'  min old_E2/old_E1 = {(o2/o1).min():.4f}   min old_E3/old_E1 = {(o3/o1).min():.4f}  (Eq.(4) never collapses)')
print(f'  min new_E2/old_E1 = {(n2/o1).min():.4f}                                             (Eq.(4a) collapses to 19%)')
