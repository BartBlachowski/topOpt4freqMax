#!/usr/bin/env python3
"""Export TRAJECTORY_MODAL_SURVEY.csv and MODE_COUNT_SENSITIVITY.csv (WP4, WP18)."""
import os, csv, numpy as np
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
Z=np.load(os.path.join(OUT,'scripts','survey.npz')); TAUS=Z['TAUS']
GF=np.load(os.path.join(OUT,'scripts','gate_full.npz'))
IT=int(np.argmin(np.abs(TAUS-0.10)))
G=lambda c,f: Z[f'{c}|{f}']
CFGS=sorted({c[:-6] for c in Z.files if c.endswith('|omega')})

rows=[]
for cfg in CFGS:
    mesh,model,law=cfg.split('|')
    om=G(cfg,'omega'); kl=G(cfg,'keLow')[:,IT,:]; sl=G(cfg,'seLow')[:,IT,:]
    dw=G(cfg,'dwp'); ip=G(cfg,'ipr'); ks=G(cfg,'k'); nm=G(cfg,'nmodes')
    hk=GF[f'{mesh}|GATE2|k']; hard=GF[f'{mesh}|GATE2|hard'].astype(bool)
    nlow=GF[f'{mesh}|GATE2|n_low']; gray=GF[f'{mesh}|GATE2|grayness']
    hmap={int(a):(bool(h),int(n),float(g)) for a,h,n,g in zip(hk,hard,nlow,gray)}
    for s in range(om.shape[0]):
        v=kl[s]; fin=np.isfinite(v)
        idx=np.flatnonzero(fin&(v<0.5))
        first=int(idx[0])+1 if idx.size else -1
        hg,nl,gr=hmap.get(int(ks[s]),(None,-1,float('nan')))
        rows.append(dict(mesh=mesh,model=model,mass_law=law,state=int(ks[s]),
            hard_gate_pass=hg,n_low_density=nl,grayness=gr,modes_computed=int(nm[s]),
            lowest_omega=float(om[s,0]),lowest_voidKE=float(v[0]),
            lowest_voidSE=float(sl[s,0]),lowest_densPart=float(dw[s,0]),lowest_IPR=float(ip[s,0]),
            first_structural_ordinal=first,
            first_structural_omega=float(om[s,idx[0]]) if idx.size else float('nan'),
            n_artificial_below=int(idx[0]) if idx.size else -1,
            lowest_over_structural=float(om[s,0]/om[s,idx[0]]) if idx.size else float('nan')))
with open(os.path.join(OUT,'TRAJECTORY_MODAL_SURVEY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'TRAJECTORY_MODAL_SURVEY.csv: {len(rows)} rows over {len(CFGS)} configurations')

# ---- WP18 mode-count sensitivity -------------------------------------------
sens=[]
for nM in (3,5,10,12,20,24,40):
    for cfg in CFGS:
        mesh,model,law=cfg.split('|')
        v=G(cfg,'keLow')[:,IT,:]; om=G(cfg,'omega')
        ok=0; miss=0; wrongsel=0
        for s in range(v.shape[0]):
            fin=np.isfinite(v[s])
            idx=np.flatnonzero(fin&(v[s]<0.5))
            true_first=idx[0] if idx.size else None
            avail=min(nM,int(np.count_nonzero(fin)))
            idx_n=idx[idx<avail]
            if true_first is None: continue
            if idx_n.size==0: miss+=1
            elif idx_n[0]!=true_first: wrongsel+=1
            else: ok+=1
        sens.append(dict(nModes=nM,mesh=mesh,model=model,mass_law=law,
            n_states=int(v.shape[0]),captured=ok,missed_no_valid_mode=miss,
            wrong_selection=wrongsel,
            capture_fraction=ok/max(1,ok+miss+wrongsel)))
with open(os.path.join(OUT,'MODE_COUNT_SENSITIVITY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sens[0].keys())); w.writeheader(); w.writerows(sens)
print('\n=== WP18 mode-count sufficiency, pooled over all eq4a configurations ===')
print(f"{'nModes':>7} {'states':>8} {'captured':>9} {'missed':>8} {'capture %':>10}")
for nM in (3,5,10,12,20,24,40):
    S=[r for r in sens if r['nModes']==nM and r['mass_law']=='eq4a']
    tot=sum(r['n_states'] for r in S); cap=sum(r['captured'] for r in S); mis=sum(r['missed_no_valid_mode'] for r in S)
    print(f"{nM:7d} {tot:8d} {cap:9d} {mis:8d} {100*cap/max(1,tot):10.3f}")
print('\n  (a state is "missed" if no mode with voidKE<0.5 appears within the first nModes)')
