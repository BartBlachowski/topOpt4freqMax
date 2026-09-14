#!/usr/bin/env python3
"""
WP3/WP5/WP9/WP17 - population structure of modal localisation, mode-count
requirements, threshold sweep and plateau analysis.  Consumes survey.npz.
READ-ONLY.  No threshold is frozen anywhere.
"""
import os, csv, numpy as np
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
Z=np.load(os.path.join(OUT,'scripts','survey.npz'))
TAUS=Z['TAUS']
KEYS=sorted({k.rsplit('|',1)[0] for k in Z.files if k.count('|')==3 and not k.endswith('|k')})
def get(key,fld): return Z[f'{key}|{fld}']
def cfgs():
    seen=[]
    for k in Z.files:
        if k.endswith('|omega'):
            seen.append(k[:-len('|omega')])
    return sorted(set(seen))

IT=int(np.argmin(np.abs(TAUS-0.10)))     # index of tau=0.1 within the sweep

# =============== WP5 : population structure of the localisation measures ======
dist_rows=[]; pooled={}
for cfg in cfgs():
    mesh,model,law=cfg.split('|')
    om=get(cfg,'omega'); kl=get(cfg,'keLow'); sl=get(cfg,'seLow')
    dw=get(cfg,'dwp'); ip=get(cfg,'ipr')
    v=kl[:,IT,:]                                  # (states, modes) voidKE at tau=0.1
    fin=np.isfinite(v)
    # A mode is called clearly artificial / clearly structural only where the
    # THREE measures agree; everything else is reported as ambiguous.
    art = fin & (v>0.95) & (sl[:,IT,:]>0.95) & (dw<0.30)
    stc = fin & (v<0.05) & (sl[:,IT,:]<0.05) & (dw>0.70)
    amb = fin & ~art & ~stc
    pooled[cfg]=dict(v=v,art=art,stc=stc,amb=amb,dw=dw,ip=ip,se=sl[:,IT,:],fin=fin)
    def q(a,p): return float(np.percentile(a,p)) if a.size else float('nan')
    for pop,msk in (('artificial',art),('structural',stc),('ambiguous',amb)):
        a=v[msk]; d=dw[msk]; i=ip[msk]
        dist_rows.append(dict(mesh=mesh,model=model,mass_law=law,population=pop,
            n_modes=int(msk.sum()),
            voidKE_min=q(a,0),voidKE_p01=q(a,1),voidKE_p50=q(a,50),voidKE_p99=q(a,99),voidKE_max=q(a,100),
            densPart_min=q(d,0),densPart_p50=q(d,50),densPart_max=q(d,100),
            IPR_min=q(i,0),IPR_p50=q(i,50),IPR_max=q(i,100)))
with open(os.path.join(OUT,'MODAL_LOCALIZATION_DISTRIBUTIONS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(dist_rows[0].keys())); w.writeheader(); w.writerows(dist_rows)

print('=== WP5 population separation (voidKE at tau=0.1) ===')
print(f"{'config':28s} {'n_art':>7} {'n_str':>7} {'n_amb':>7} {'art min':>10} {'str max':>10} {'GAP':>10}")
gaps=[]
for cfg in cfgs():
    p=pooled[cfg]
    amin=p['v'][p['art']].min() if p['art'].any() else np.nan
    smax=p['v'][p['stc']].max() if p['stc'].any() else np.nan
    gap=amin-smax
    gaps.append((cfg,amin,smax,gap,int(p['amb'].sum())))
    print(f"{cfg:28s} {int(p['art'].sum()):7d} {int(p['stc'].sum()):7d} {int(p['amb'].sum()):7d} "
          f"{amin:10.6f} {smax:10.6f} {gap:10.6f}")

# ambiguous modes: what are they?
print('\n=== ambiguous modes (the three measures disagree) ===')
amb_rows=[]
for cfg in cfgs():
    p=pooled[cfg]; ks=get(cfg,'k')
    idx=np.argwhere(p['amb'])
    for s,m in idx[:2000]:
        amb_rows.append(dict(config=cfg,state=int(ks[s]),mode_ordinal=int(m)+1,
            omega=float(get(cfg,'omega')[s,m]),voidKE=float(p['v'][s,m]),
            voidSE=float(p['se'][s,m]),densPart=float(p['dw'][s,m]),IPR=float(p['ip'][s,m])))
    print(f'  {cfg:28s} ambiguous modes: {int(p["amb"].sum())} of {int(p["fin"].sum())} '
          f'({100*p["amb"].sum()/max(1,p["fin"].sum()):.3f}%)')
if amb_rows:
    with open(os.path.join(OUT,'AMBIGUOUS_MODES.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(amb_rows[0].keys())); w.writeheader(); w.writerows(amb_rows)

# =============== WP3/WP18 : how many modes must be computed ==================
mc_rows=[]
print('\n=== WP3 first-structural-mode ordinal (criterion voidKE<0.5 at tau=0.1) ===')
for cfg in cfgs():
    mesh,model,law=cfg.split('|')
    v=pooled[cfg]['v']; ks=get(cfg,'k'); nm=get(cfg,'nmodes')
    ordn=np.full(v.shape[0],np.nan)
    for s in range(v.shape[0]):
        w_=np.flatnonzero(np.isfinite(v[s]) & (v[s]<0.5))
        if w_.size: ordn[s]=w_[0]+1
    ok=np.isfinite(ordn)
    mc_rows.append(dict(mesh=mesh,model=model,mass_law=law,n_states=int(v.shape[0]),
        states_with_structural_found=int(ok.sum()),
        max_first_structural_ordinal=int(np.nanmax(ordn)) if ok.any() else -1,
        p99_ordinal=float(np.nanpercentile(ordn,99)) if ok.any() else float('nan'),
        median_ordinal=float(np.nanmedian(ordn)) if ok.any() else float('nan'),
        states_ordinal_gt_1=int(np.nansum(ordn>1)),states_ordinal_gt_3=int(np.nansum(ordn>3)),
        states_ordinal_gt_5=int(np.nansum(ordn>5)),states_ordinal_gt_10=int(np.nansum(ordn>10)),
        max_modes_requested=int(nm.max()),escalations=int((nm>12).sum())))
    print(f'  {cfg:28s} max ordinal={mc_rows[-1]["max_first_structural_ordinal"]:3d}  '
          f'>1:{mc_rows[-1]["states_ordinal_gt_1"]:5d}  >3:{mc_rows[-1]["states_ordinal_gt_3"]:5d}  '
          f'>5:{mc_rows[-1]["states_ordinal_gt_5"]:4d}  >10:{mc_rows[-1]["states_ordinal_gt_10"]:3d}  '
          f'escalations={mc_rows[-1]["escalations"]}')
with open(os.path.join(OUT,'MODE_COUNT_REQUIREMENTS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(mc_rows[0].keys())); w.writeheader(); w.writerows(mc_rows)

# =============== WP17 : threshold sweep and plateaus =========================
CUTS=np.concatenate([np.array([1e-4,3e-4,1e-3,3e-3,0.01,0.02,0.03,0.05,0.07]),
                     np.arange(0.10,0.96,0.05)])
sweep=[]; plateau=[]
print('\n=== WP17 modal-validity threshold sweep (criterion: voidKE(tau) < cut) ===')
for cfg in cfgs():
    if cfg.split('|')[2]!='eq4a': continue
    mesh,model,law=cfg.split('|')
    om=get(cfg,'omega'); kl=get(cfg,'keLow'); ks=get(cfg,'k')
    for ti,tau in enumerate(TAUS):
        v=kl[:,ti,:]
        prev=None
        for cut in CUTS:
            sel=np.full(v.shape[0],-1); w_om=np.full(v.shape[0],np.nan)
            for s in range(v.shape[0]):
                w_=np.flatnonzero(np.isfinite(v[s]) & (v[s]<cut))
                if w_.size: sel[s]=w_[0]+1; w_om[s]=om[s,w_[0]]
            nfound=int((sel>0).sum())
            chg=int((sel!=prev[0]).sum()) if prev is not None else -1
            sweep.append(dict(mesh=mesh,model=model,tau=float(tau),cut=float(cut),
                n_states=int(v.shape[0]),states_with_valid_mode=nfound,
                states_selection_changed_vs_prev_cut=chg,
                median_selected_ordinal=float(np.median(sel[sel>0])) if nfound else float('nan'),
                max_selected_ordinal=int(sel.max()),
                min_selected_omega=float(np.nanmin(w_om)) if nfound else float('nan')))
            prev=(sel.copy(),w_om.copy())
    print(f'  {cfg} swept {len(CUTS)} cuts x {len(TAUS)} taus',flush=True)
with open(os.path.join(OUT,'STRUCTURAL_MODE_THRESHOLD_SWEEP.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)

# plateau detection: contiguous cut ranges over which the SELECTED MODE is
# identical at every state, for tau = 0.1 and for the widest tau tested
print('\n=== WP17 plateaus: cut ranges giving identical selection at every state ===')
for cfg in cfgs():
    if cfg.split('|')[2]!='eq4a': continue
    mesh,model,_=cfg.split('|')
    kl=get(cfg,'keLow'); om=get(cfg,'omega')
    for ti,tau in enumerate(TAUS):
        v=kl[:,ti,:]
        sels=[]
        for cut in CUTS:
            sel=np.array([ (np.flatnonzero(np.isfinite(v[s])&(v[s]<cut))[:1]+1).tolist() or [-1]
                           for s in range(v.shape[0])]).ravel()
            sels.append(sel)
        sels=np.array(sels)
        run_lo=0
        for i in range(1,len(CUTS)+1):
            if i==len(CUTS) or not np.array_equal(sels[i],sels[run_lo]):
                if i-run_lo>=2:
                    sel0=sels[run_lo]
                    # A plateau is DEGENERATE if the criterion accepts every mode, so the
                    # selection is identically ordinal 1 and the cut does nothing.  Such a
                    # plateau is vacuous and must not be reported as robustness.
                    degenerate=bool((sel0==1).all())
                    plateau.append(dict(mesh=mesh,model=model,tau=float(tau),degenerate=degenerate,
                        cut_lo=float(CUTS[run_lo]),cut_hi=float(CUTS[i-1]),
                        n_cuts_in_plateau=i-run_lo,
                        width_decades=float(np.log10(CUTS[i-1]/CUTS[run_lo])) if CUTS[run_lo]>0 else float('nan'),
                        all_states_have_valid_mode=bool((sels[run_lo]>0).all()),
                        max_selected_ordinal=int(sels[run_lo].max())))
                run_lo=i
with open(os.path.join(OUT,'STRUCTURAL_MODE_THRESHOLD_PLATEAUS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(plateau[0].keys())); w.writeheader(); w.writerows(plateau)
best={}
for p in plateau:
    k=(p['mesh'],p['model'],p['tau'])
    if p['all_states_have_valid_mode'] and not p['degenerate'] and \
       (k not in best or p['width_decades']>best[k]['width_decades']):
        best[k]=p
ndeg=sum(1 for p in plateau if p['degenerate'])
print(f'  plateaus found: {len(plateau)}; DEGENERATE (selection identically ordinal 1): {ndeg}')
print('  widest NON-DEGENERATE full-coverage plateau per (mesh, model, tau):')
for k,p in sorted(best.items()):
    print(f"  {k[0]:8s} {k[1]} tau={k[2]:<5g} cut in [{p['cut_lo']:.4g}, {p['cut_hi']:.4g}]"
          f"  = {p['width_decades']:.2f} decades  (max selected ordinal {p['max_selected_ordinal']})")
miss=[k for k in {(p['mesh'],p['model'],p['tau']) for p in plateau} if k not in best]
if miss:
    print(f'  (mesh,model,tau) with NO non-degenerate plateau: {len(miss)}')
    for k in sorted(miss)[:12]: print(f'     {k}')
print('\ndone')
