#!/usr/bin/env python3
"""WP9 - independent reproduction of the binding-evaluator instability, and
reconciliation of Phase-2C's 751/3200 with Phase-2D's 150/1600.  READ-ONLY."""
import sys, os, csv, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_engines import reference_phase
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')
QR=os.path.join(REPO,'analysis/iteration_efficiency_phase2b_recheck/qualification_runs')
rows=[]

# ---- dataset 1: the Phase-2C dataset, 96x12 H3200, Eq.(4) only -------------------
with h5py.File(os.path.join(QR,'probe_96x12_H3200.mat'),'r') as f:
    Qhi=f['Qhi'][()].T; Qlo=f['Qlo'][()].T; H0=f['H0'][()].ravel().astype(bool)
# binding evaluator under the FROZEN ratio normalisation r_e = Q_e / Q_ref_e
rhi=reference_phase(Qhi,H0); rlo=reference_phase(Qlo,H0)
for tag,Qref in (('own_reference_per_branch',None),('shared_reference_hi',rhi['Q_ref'])):
    if Qref is None:
        bhi=np.argmin(Qhi/rhi['Q_ref'],axis=1); blo=np.argmin(Qlo/rlo['Q_ref'],axis=1)
    else:
        bhi=np.argmin(Qhi/Qref,axis=1); blo=np.argmin(Qlo/Qref,axis=1)
    n=int((bhi!=blo).sum())
    rows.append(dict(dataset='96x12 H3200 (Phase-2C)',evaluator_law='Eq.(4) only',
        normalisation=('frozen Q_ref, recomputed per branch' if Qref is None else 'frozen Q_ref of the hi branch'),
        perturbation='branch side (all at-risk elements forced low vs as-stored single)',
        horizon=3200,numerator=n,denominator=3200,pct=100*n/3200,
        binding_E1=int((bhi==0).sum()),binding_E2=int((bhi==1).sum()),binding_E3=int((bhi==2).sum())))
    print(f'  96x12 H3200, {rows[-1]["normalisation"]}: binding changes {n}/3200 = {100*n/3200:.2f}%'
          f'   shares hi(E1,E2,E3)=({rows[-1]["binding_E1"]},{rows[-1]["binding_E2"]},{rows[-1]["binding_E3"]})')
# and with the Phase-2C surrogate (own trajectory max) for comparability
nrm=Qhi.max(axis=0)
bhi=np.argmin(Qhi/nrm,axis=1); blo=np.argmin(Qlo/nrm,axis=1)
print(f'  96x12 H3200, surrogate max-normalisation: binding changes {(bhi!=blo).sum()}/3200')

# ---- dataset 2: the Phase-2D dataset, 160x20 production, Eq.(4) and Eq.(4a) -------
p=os.path.join(OUT,'scripts','wp7c_arrays.npy')
if os.path.exists(p):
    A=np.load(p,allow_pickle=True).item()
    for law,so,fo in (('Eq.(4)',A['stored_old'],A['forced_old']),('Eq.(4a)',A['stored_new'],A['forced_new'])):
        for nname,nrm in (('surrogate: own trajectory maximum',so.max(axis=0)),
                          ('surrogate: final-state value',so[-1])):
            b1=np.argmin(so/nrm,axis=1); b2=np.argmin(fo/nrm,axis=1)
            n=int((b1!=b2).sum())
            rows.append(dict(dataset='160x20 production (Phase-2D)',evaluator_law=law,
                normalisation=nname,perturbation='branch side (float32 0.1 elements forced to exact 0.1)',
                horizon=1600,numerator=n,denominator=1600,pct=100*n/1600,
                binding_E1=int((b1==0).sum()),binding_E2=int((b1==1).sum()),binding_E3=int((b1==2).sum())))
            print(f'  160x20 {law:8s} [{nname}]: binding changes {n}/1600 = {100*n/1600:.2f}%'
                  f'   shares=({rows[-1]["binding_E1"]},{rows[-1]["binding_E2"]},{rows[-1]["binding_E3"]})')
else:
    print('  (160x20 arrays not ready yet)')
with open(os.path.join(OUT,'BINDING_EVALUATOR_RECHECK.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
