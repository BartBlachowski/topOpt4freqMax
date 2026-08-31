#!/usr/bin/env python3
"""Is the Eq.(4a) spurious-mode defect specific to 160x20, or general?  Screens
two further stored Olhoff production trajectories.  READ-ONLY."""
import sys, os, csv, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import omega1
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')
rows=[]
for mesh,nx,ny,stride in (('240x30',240,30,2),('320x40',320,40,4)):
    with h5py.File(os.path.join(REPO,f'examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat'),'r') as f:
        X=f['res/rho_snapshots'][()]
    n=X.shape[0]-1; hit=0; worst=1.0; wk=None
    for k in range(1,n+1,stride):
        x=np.float64(X[k])
        e1=omega1(x,nx,ny,'E1',False)
        e2a=omega1(x,nx,ny,'E2',True)
        r=e2a/e1
        if r<0.99:
            hit+=1
            e2=omega1(x,nx,ny,'E2',False)
            rows.append(dict(mesh=mesh,k=k,E1=e1,eq4_E2=e2,eq4a_E2=e2a,
                             eq4a_E2_over_E1=r,eq4_E2_over_E1=e2/e1))
            if r<worst: worst, wk = r, k
    print(f'{mesh} (stride {stride}, {len(range(1,n+1,stride))} states screened): '
          f'{hit} states with amended E2 < 0.99*E1; worst ratio {worst:.4f} at k={wk}',flush=True)
if rows:
    with open(os.path.join(OUT,'SPURIOUS_MODE_OTHER_MESHES.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('done')
