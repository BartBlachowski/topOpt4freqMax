#!/usr/bin/env python3
"""
WP11/WP12/WP13 - candidate D: exact-volume binary-field evaluation.
Modal validity, projection/tie stability, and agreement with the gray structural
frequency.  READ-ONLY.  Sparse solver throughout.
"""
import sys, os, csv, time, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_engine import modes, exact_count_binary
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
QR=os.path.join(REPO,'analysis/iteration_efficiency_phase2b_precision/qualification_runs')

def rank_margin(x, volfrac=0.5):
    """Density gap across the exact-count cutoff, and how many elements tie there."""
    x=np.asarray(x,float).ravel(); n=x.size; nS=int(round(volfrac*n))
    s=np.sort(x)[::-1]
    gap=float(s[nS-1]-s[nS]) if 0<nS<n else float('nan')
    cutval=float(s[nS-1])
    nties=int(np.count_nonzero(x==cutval))
    return gap,cutval,nties

def binary_omega(x,nelx,nely,model,law,k=6):
    xb=exact_count_binary(x)
    r=modes(xb,nelx,nely,model,law,k=k)
    low=r['zeff']<=0.1
    return r['omega'][0],float(r['ke_n'][low,0].sum()),xb,r

# ---------------- WP11 + WP13 : binary vs gray along trajectories -------------
PLAN=[('160x20',160,20,1),('240x30',240,30,2),('320x40',320,40,4),
      ('400x50',400,50,8),('480x60',480,60,4),('560x70',560,70,4),
      ('640x80',640,80,16),('720x90',720,90,20)]
rows=[]; t0=time.time()
for mesh,nx,ny,stride in PLAN:
    with h5py.File(os.path.join(REPO,f'examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat'),'r') as f:
        X=f['res/rho_snapshots'][()]
    n=X.shape[0]-1
    for k in range(1,n+1,stride):
        z=np.clip(np.float64(X[k]),0,1)
        gap,cutval,nties=rank_margin(z)
        rec=dict(mesh=mesh,state=k,grayness=float(np.mean(4*z*(1-z))),
                 n_low=int((z<=0.1).sum()),cut_density=cutval,cut_gap=gap,n_tied_at_cut=nties)
        for model in ('E1','E2','E3'):
            law='linear' if model=='E1' else 'eq4a'
            ob,vb,xb,rb=binary_omega(z,nx,ny,model,law)
            rec[f'binary_omega_{model}']=float(ob); rec[f'binary_voidKE_{model}']=vb
            # gray structural reference under the same model/law
            rg=modes(z,nx,ny,model,law,k=12)
            low=rg['zeff']<=0.1
            v=rg['ke_n'][low,:].sum(axis=0)
            w_=np.flatnonzero(v<0.5)
            rec[f'gray_lowest_omega_{model}']=float(rg['omega'][0])
            rec[f'gray_struct_omega_{model}']=float(rg['omega'][w_[0]]) if w_.size else float('nan')
            rec[f'gray_struct_ordinal_{model}']=int(w_[0])+1 if w_.size else -1
        rows.append(rec)
    print(f'  {mesh:8s} stride={stride} states={len(range(1,n+1,stride)):5d} [{time.time()-t0:.0f}s]',flush=True)
with open(os.path.join(OUT,'GRAY_VS_BINARY_QUALITY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------- WP12 : projection stability --------------------------------
stab=[]
# (a) genuine double-vs-single paired states from Phase 2B
for fn,nx,ny in (('gray_full_24x4_h200_paired_states.mat',24,4),
                 ('s1_transition_96x12_h320_paired_states.mat',96,12)):
    with h5py.File(os.path.join(QR,fn),'r') as f:
        XD=f['x_double'][()]; XS=f['x_single'][()]; it=f['pairIterations'][()].ravel()
    for i in range(XD.shape[0]):
        xd=np.float64(XD[i]); xs=np.float64(XS[i])
        bd=exact_count_binary(xd); bs=exact_count_binary(xs)
        flips=int(np.count_nonzero(bd!=bs))
        gap,cutval,nties=rank_margin(xd)
        rec=dict(source=fn,state=int(it[i]),mesh=f'{nx}x{ny}',perturbation='double vs float32 storage',
                 n_elements=xd.size,binary_assignment_flips=flips,cut_gap=gap,n_tied_at_cut=nties)
        for model in ('E1','E2','E3'):
            law='linear' if model=='E1' else 'eq4a'
            od=modes(bd,nx,ny,model,law,k=3)['omega'][0]
            os_=modes(bs,nx,ny,model,law,k=3)['omega'][0]
            rec[f'binary_rel_change_{model}']=float(abs(os_-od)/od)
        stab.append(rec)
# (b) one-ULP perturbation of the production trajectories
for mesh,nx,ny,stride in (('160x20',160,20,25),('320x40',320,40,100)):
    with h5py.File(os.path.join(REPO,f'examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat'),'r') as f:
        X=f['res/rho_snapshots'][()]
    n=X.shape[0]-1
    for k in range(1,n+1,stride):
        z=np.clip(np.float64(X[k]),0,1)
        gap,cutval,nties=rank_margin(z)
        zp=z.copy()
        at=np.flatnonzero(z==cutval)                 # elements exactly at the cutoff value
        if at.size: zp[at[0]]=np.nextafter(zp[at[0]],0.0)
        b0=exact_count_binary(z); b1=exact_count_binary(zp)
        rec=dict(source=f'{mesh} trajectory',state=k,mesh=mesh,
                 perturbation='one double ULP on an element at the exact-count cutoff',
                 n_elements=z.size,binary_assignment_flips=int(np.count_nonzero(b0!=b1)),
                 cut_gap=gap,n_tied_at_cut=nties)
        for model in ('E1','E2','E3'):
            law='linear' if model=='E1' else 'eq4a'
            o0=modes(b0,nx,ny,model,law,k=3)['omega'][0]
            o1=modes(b1,nx,ny,model,law,k=3)['omega'][0]
            rec[f'binary_rel_change_{model}']=float(abs(o1-o0)/o0)
        stab.append(rec)
    print(f'  ULP test {mesh} done [{time.time()-t0:.0f}s]',flush=True)
with open(os.path.join(OUT,'BINARY_PROJECTION_STABILITY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(stab[0].keys())); w.writeheader(); w.writerows(stab)
print(f'\ntotal {time.time()-t0:.0f}s ; trajectory rows {len(rows)} ; stability rows {len(stab)}')
