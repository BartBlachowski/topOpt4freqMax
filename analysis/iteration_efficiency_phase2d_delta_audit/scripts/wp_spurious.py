#!/usr/bin/env python3
"""Independent check NOT performed by Phase 2D: does Eq. (4a) reintroduce the
spurious localized eigenmodes that Du & Olhoff Eq. (4) exists to suppress, on the
GRAY INTERMEDIATE states this study measures?  READ-ONLY."""
import sys, os, csv, h5py, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import mesh_data, interp
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')

def modes(x, nelx, nely, model, eq4a, k=6):
    md=mesh_data(nelx,nely); z=np.clip(np.asarray(x,float).ravel(),0,1)
    Ee,rr=interp(z,model,eq4a); ndof,free=md['ndof'],md['free']
    K=sp.coo_matrix(((md['KEv'][None,:]*Ee[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    M=sp.coo_matrix(((md['MEv'][None,:]*rr[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
    K=((K+K.T)*.5)[free][:,free].tocsc(); M=((M+M.T)*.5)[free][:,free].tocsc()
    v0=np.random.default_rng(20260830).standard_normal(K.shape[0]); v0/=np.linalg.norm(v0)
    lam,V=spla.eigsh(K,k=k,M=M,sigma=0.0,which='LM',v0=v0,tol=0.0,maxiter=100000)
    o=np.argsort(lam.real); lam=lam.real[o]; V=V[:,o]
    # element-wise kinetic-energy participation of each mode
    part=[]
    for j in range(V.shape[1]):
        u=np.zeros(ndof); u[free]=V[:,j]
        ue=u[md['rows'].reshape(md['nEl'],64)[:,::8]]        # 8 dofs per element
        # kinetic energy per element = rr_e * ue' ME ue
        ME=md['ME']; ke=rr*np.einsum('ei,ij,ej->e',ue,ME,ue)
        part.append(ke/ke.sum())
    return np.sqrt(np.maximum(lam,0)), np.array(part), z

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    X=f['res/rho_snapshots'][()]
R=list(csv.DictReader(open(os.path.join(OUT,'WP7C_TRAJECTORY_160x20_INDEPENDENT.csv'))))
sh=np.array([float(r['level_shift_rel_E2']) for r in R])
worst=np.argsort(-sh)[:6]
rows=[]
print(f"{'k':>5} {'law':8s} {'model':4s} {'w1':>10s} {'w2':>10s} {'w3':>10s} "
      f"{'lowdens_KE_share(mode1)':>24s} {'n(rho<=0.1)':>12s}")
for i in list(worst)+[1599]:
    k=i+1; x=np.float64(X[k])
    nlow=int((x<=0.1).sum())
    for model in ('E1','E2','E3'):
        for law,tag in ((False,'Eq.(4) '),(True,'Eq.(4a)')):
            w,part,z=modes(x,160,20,model,law)
            zz=np.maximum(z,1e-3) if model=='E3' else z
            share=float(part[0][zz<=0.1].sum())
            rows.append(dict(k=k,model=model,law=tag.strip(),w1=w[0],w2=w[1],w3=w[2],
                             lowdensity_KE_share_mode1=share,n_low_density=nlow,
                             level_shift_rel=sh[i]))
            print(f"{k:5d} {tag:8s} {model:4s} {w[0]:10.4f} {w[1]:10.4f} {w[2]:10.4f} "
                  f"{share:24.4f} {nlow:12d}")
    print()
with open(os.path.join(OUT,'SPURIOUS_MODE_CHECK.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
