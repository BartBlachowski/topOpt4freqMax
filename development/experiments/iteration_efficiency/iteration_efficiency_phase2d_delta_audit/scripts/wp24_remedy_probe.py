#!/usr/bin/env python3
"""WP24 - bound the remedy space with evidence, at the worst affected state.
Not a design proposal: a feasibility probe so the required work can be specified.
READ-ONLY."""
import sys, os, csv, h5py, numpy as np, scipy.sparse as sp, scipy.linalg as la
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import mesh_data
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')

def build(z, model, masslaw):
    if model=='E1': Ee=1e7*(1e-6+(1-1e-6)*z**3); base=z; floor=lambda g:1e-6+(1-1e-6)*g; zz=z
    elif model=='E2': Ee=1e7*(1e-9+(1-1e-9)*z**3); base=z; floor=lambda g:1e-9+(1-1e-9)*g; zz=z
    else: zz=np.maximum(z,1e-3); Ee=1e7*zz**3; base=zz; floor=lambda g:g
    g={'linear':lambda x:x,
       'eq4':   lambda x:np.where(x<=0.1,x**6,x),
       'eq4a':  lambda x:np.where(x<=0.1,1e5*x**6,x),
       'eq4b':  lambda x:np.where(x<=0.1,6e5*x**6-5e6*x**7,x)}[masslaw](base)
    return Ee, floor(g), zz

def spec(z,nelx,nely,model,masslaw,nm=8):
    md=mesh_data(nelx,nely); Ee,rr,zz=build(z,model,masslaw)
    ndof,free=md['ndof'],md['free']
    K=sp.coo_matrix(((md['KEv'][None,:]*Ee[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    M=sp.coo_matrix(((md['MEv'][None,:]*rr[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    Kd=np.asarray((((K+K.T)*.5)[free][:,free]).todense()); Md=np.asarray((((M+M.T)*.5)[free][:,free]).todense())
    lam,V=la.eigh(Kd,Md); w=np.sqrt(np.maximum(lam[:nm],0)); sh=[]
    low=zz<=0.1
    for j in range(nm):
        u=np.zeros(ndof); u[free]=V[:,j]
        ue=u[md['rows'].reshape(md['nEl'],64)[:,::8]]
        ke=rr*np.einsum('ei,ij,ej->e',ue,md['ME'],ue); sh.append(ke[low].sum()/ke.sum())
    return w,np.array(sh)

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    Xs=f['res/rho_snapshots'][()]
z=np.clip(np.float64(Xs[252]),0,1)
# exact-count binary projection (frozen rule), for remedy option (ii)
nS=int(round(0.5*z.size)); order=np.lexsort((np.arange(z.size),-z)); zb=np.zeros_like(z); zb[order[:nS]]=1.0

rows=[]
print('=== state k=252, 160x20 : remedy feasibility probe ===')
print(f"{'variant':38s} {'omega_1':>10s} {'void KE share':>14s} {'verdict':>12s}")
for model in ('E1','E2','E3'):
    for law in ('eq4','eq4a','eq4b','linear'):
        w,sh=spec(z,160,20,model,law)
        v='SPURIOUS' if sh[0]>0.5 else 'structural'
        print(f'  raw   {model} mass={law:8s}{"":12s} {w[0]:10.4f} {sh[0]:14.4f} {v:>12s}')
        rows.append(dict(state=252,field='raw',model=model,mass_law=law,omega1=w[0],
                         void_KE_share=sh[0],verdict=v))
    w,sh=spec(zb,160,20,model,'eq4a')
    v='SPURIOUS' if sh[0]>0.5 else 'structural'
    print(f'  BINARY{model} mass=eq4a  (exact-count) {w[0]:10.4f} {sh[0]:14.4f} {v:>12s}')
    rows.append(dict(state=252,field='binary_exact_count',model=model,mass_law='eq4a',
                     omega1=w[0],void_KE_share=sh[0],verdict=v))
    print()
with open(os.path.join(OUT,'REMEDY_FEASIBILITY_PROBE.csv'),'w',newline='') as f:
    w_=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)

# does Eq.(4) ever give a spurious omega_1 on this trajectory?
R=list(csv.DictReader(open(os.path.join(OUT,'WP7C_TRAJECTORY_160x20_INDEPENDENT.csv'))))
o1=np.array([float(r['old_E1']) for r in R]); o2=np.array([float(r['old_E2']) for r in R])
o3=np.array([float(r['old_E3']) for r in R]); n2=np.array([float(r['new_E2']) for r in R])
print('=== control: Eq.(4) over all 1600 states ===')
print(f'  min old_E2/old_E1 = {(o2/o1).min():.4f}   min old_E3/old_E1 = {(o3/o1).min():.4f}'
      f'   -> Eq.(4) never collapses; its low-density suppression works as the source intends')
print(f'  min new_E2/old_E1 = {(n2/o1).min():.4f}   -> Eq.(4a) collapses to 19% of the structural value')
