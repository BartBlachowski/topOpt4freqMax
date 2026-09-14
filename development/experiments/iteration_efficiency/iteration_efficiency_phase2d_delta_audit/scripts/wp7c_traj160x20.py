#!/usr/bin/env python3
"""WP7(c)/WP9/WP10 - independent re-evaluation of the full 160x20 production
trajectory (1600 states) under Eq.(4) and Eq.(4a), as stored (float32) and with
every at-risk element forced onto the low branch.  READ-ONLY, no optimizer."""
import sys, os, csv, time, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import Qvec
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    X=f['res/rho_snapshots'][()]                    # (1601, 3200) float32
s01=np.float32(0.1)
n=X.shape[0]-1
rows=[]; t0=time.time()
for k in range(1,n+1):
    xs=X[k]; hit=(xs==s01); nat=int(hit.sum()); x=np.float64(xs)
    qo=Qvec(x,160,20,False); qn=Qvec(x,160,20,True)
    if nat:
        xb=x.copy(); xb[hit]=0.1                    # force the low (x^6 / 1e5 x^6) branch
        qof=Qvec(xb,160,20,False); qnf=Qvec(xb,160,20,True)
    else:
        qof=qo; qnf=qn
    ro=np.abs(qo-qof)/np.abs(qo); rn=np.abs(qn-qnf)/np.abs(qn)
    rows.append(dict(k=k,n_atrisk=nat,
        old_E1=qo[0],old_E2=qo[1],old_E3=qo[2],
        new_E1=qn[0],new_E2=qn[1],new_E3=qn[2],
        oldf_E1=qof[0],oldf_E2=qof[1],oldf_E3=qof[2],
        newf_E1=qnf[0],newf_E2=qnf[1],newf_E3=qnf[2],
        old_branch_rel_E1=ro[0],old_branch_rel_E2=ro[1],old_branch_rel_E3=ro[2],
        new_branch_rel_E1=rn[0],new_branch_rel_E2=rn[1],new_branch_rel_E3=rn[2],
        level_shift_rel_E1=abs(qn[0]-qo[0])/qo[0],
        level_shift_rel_E2=abs(qn[1]-qo[1])/qo[1],
        level_shift_rel_E3=abs(qn[2]-qo[2])/qo[2]))
    if k%200==0: print(f'  {k}/{n}  {time.time()-t0:.0f}s',flush=True)
with open(os.path.join(OUT,'WP7C_TRAJECTORY_160x20_INDEPENDENT.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
np.save(os.path.join(OUT,'scripts','wp7c_arrays.npy'),
        {'stored_old':np.array([[r['old_E1'],r['old_E2'],r['old_E3']] for r in rows]),
         'stored_new':np.array([[r['new_E1'],r['new_E2'],r['new_E3']] for r in rows]),
         'forced_old':np.array([[r['oldf_E1'],r['oldf_E2'],r['oldf_E3']] for r in rows]),
         'forced_new':np.array([[r['newf_E1'],r['newf_E2'],r['newf_E3']] for r in rows]),
         'n_atrisk':np.array([r['n_atrisk'] for r in rows])},allow_pickle=True)
mx=lambda k: max(r[k] for r in rows)
print(f'\nstates={len(rows)}  states with >=1 at-risk element={sum(1 for r in rows if r["n_atrisk"]>0)}'
      f'  max at-risk in one state={mx("n_atrisk")}')
for tag in ('E1','E2','E3'):
    print(f'  branch-side max rel {tag}: OLD {mx("old_branch_rel_"+tag):.4e}   NEW {mx("new_branch_rel_"+tag):.4e}')
for tag in ('E1','E2','E3'):
    print(f'  Eq.(4)->Eq.(4a) LEVEL shift {tag}: max {mx("level_shift_rel_"+tag):.4e}')
print(f'  total {time.time()-t0:.0f}s')
