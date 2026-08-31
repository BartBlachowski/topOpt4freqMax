#!/usr/bin/env python3
"""Confirmation of the Eq.(4a) spurious-localized-mode finding by a THIRD,
non-iterative route: dense LAPACK generalized symmetric eigensolve (scipy.linalg.eigh),
which shares no code path with ARPACK shift-invert or with MATLAB eigs.  READ-ONLY."""
import sys, os, h5py, numpy as np, scipy.sparse as sp, scipy.linalg as la
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import mesh_data, interp
REPO='/Users/piotrek/Programming/topOpt4freqMax'

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    x=np.float64(f['res/rho_snapshots'][252])          # state k = 252
nelx,nely=160,20
md=mesh_data(nelx,nely); free=md['free']; ndof=md['ndof']
z=np.clip(x,0,1)
print(f'state k=252: n(rho<=0.1)={int((z<=0.1).sum())} of {z.size}; '
      f'min={z.min():.6g} max={z.max():.6g} mean={z.mean():.6g}')

for model in ('E1','E2','E3'):
    for eq4a,tag in ((False,'Eq.(4) '),(True,'Eq.(4a)')):
        Ee,rr=interp(z,model,eq4a)
        K=sp.coo_matrix(((md['KEv'][None,:]*Ee[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
        M=sp.coo_matrix(((md['MEv'][None,:]*rr[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
        Kd=np.asarray((((K+K.T)*.5)[free][:,free]).todense())
        Md=np.asarray((((M+M.T)*.5)[free][:,free]).todense())
        lam,V=la.eigh(Kd,Md)                      # dense LAPACK, no iteration, no shift
        w=np.sqrt(np.maximum(lam[:14],0))
        # kinetic-energy localisation of each of the lowest modes
        zz=np.maximum(z,1e-3) if model=='E3' else z
        low=zz<=0.1
        ME=md['ME']; shares=[]
        for j in range(14):
            u=np.zeros(ndof); u[free]=V[:,j]
            ue=u[md['rows'].reshape(md['nEl'],64)[:,::8]]
            ke=rr*np.einsum('ei,ij,ej->e',ue,ME,ue)
            shares.append(ke[low].sum()/ke.sum())
        struct=[j for j in range(14) if shares[j]<0.5]
        print(f'\n{model} {tag} (dense LAPACK):')
        print('   omega  : '+' '.join(f'{v:8.3f}' for v in w))
        print('   lowdens: '+' '.join(f'{s:8.4f}' for s in shares))
        print(f'   -> lowest mode with <50% void kinetic energy: '
              f'{("index %d, omega=%.4f"%(struct[0]+1,w[struct[0]])) if struct else "NONE in the lowest 14"}')
