#!/usr/bin/env python3
"""
WP22 - active falsification of candidates C and D.  Deliberately searches for
failure, not for confirmation.  Consumes survey.npz and the binary study outputs.
READ-ONLY.
"""
import os, csv, numpy as np
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
Z=np.load(os.path.join(OUT,'scripts','survey.npz')); TAUS=Z['TAUS']
IT=int(np.argmin(np.abs(TAUS-0.10)))
G=lambda c,f: Z[f'{c}|{f}']
CFGS=[c[:-6] for c in Z.files if c.endswith('|omega')]
EQ4A=[c for c in CFGS if c.endswith('eq4a')]
out=[]

def rec(cat,cand,sev,desc,**kw):
    d=dict(category=cat,candidate=cand,severity=sev,description=desc); d.update(kw); out.append(d)

# ---------------- Candidate C falsification ----------------------------------
print('=== WP22 candidate C falsification ===')
# C1: no structural mode within the computed spectrum
tot=0; miss=0; miss_states=[]
for c in EQ4A:
    v=G(c,'keLow')[:,IT,:]
    for s in range(v.shape[0]):
        fin=np.isfinite(v[s]); tot+=1
        if not (fin & (v[s]<0.5)).any():
            miss+=1; miss_states.append((c,int(G(c,'k')[s])))
print(f'  C1 no structural mode found within the computed spectrum: {miss}/{tot} states')
rec('no_structural_mode_in_spectrum','C','CRITICAL' if miss else 'NOTE',
    f'{miss} of {tot} surveyed Eq.(4a) states had no mode with voidKE<0.5 within the computed spectrum',
    n_affected=miss,n_total=tot,examples=';'.join(f'{a}@{b}' for a,b in miss_states[:10]))

# C2/C3: overlapping populations - structural modes with high voidKE, artificial with low
worst_struct=[]; worst_art=[]
for c in EQ4A:
    v=G(c,'keLow')[:,IT,:]; sv=G(c,'seLow')[:,IT,:]; dw=G(c,'dwp')
    fin=np.isfinite(v)
    art=fin&(v>0.95)&(sv>0.95)&(dw<0.30)
    stc=fin&(v<0.05)&(sv<0.05)&(dw>0.70)
    if stc.any(): worst_struct.append((c,float(v[stc].max())))
    if art.any():  worst_art.append((c,float(v[art].min())))
smax=max(x[1] for x in worst_struct) if worst_struct else np.nan
amin=min(x[1] for x in worst_art) if worst_art else np.nan
print(f'  C2 highest voidKE among clearly-structural modes : {smax:.6f}')
print(f'  C3 lowest  voidKE among clearly-artificial modes : {amin:.6f}')
print(f'  C4 separation gap                                : {amin-smax:.6f}')
rec('population_overlap','C','NOTE' if amin>smax else 'CRITICAL',
    f'structural voidKE max {smax:.6g}; artificial voidKE min {amin:.6g}; gap {amin-smax:.6g}',
    structural_max=smax,artificial_min=amin,gap=amin-smax)

# C5: ambiguous modes below the first structural mode (these are the dangerous ones)
amb_below=0; amb_tot=0; amb_ex=[]
for c in EQ4A:
    v=G(c,'keLow')[:,IT,:]; sv=G(c,'seLow')[:,IT,:]; dw=G(c,'dwp'); ks=G(c,'k')
    fin=np.isfinite(v)
    art=(v>0.95)&(sv>0.95)&(dw<0.30); stc=(v<0.05)&(sv<0.05)&(dw>0.70)
    amb=fin&~art&~stc
    amb_tot+=int(amb.sum())
    for s in range(v.shape[0]):
        idx=np.flatnonzero(fin[s]&(v[s]<0.5))
        first=idx[0] if idx.size else 10**9
        bad=np.flatnonzero(amb[s])
        bad=bad[bad<first]
        if bad.size:
            amb_below+=bad.size
            if len(amb_ex)<12: amb_ex.append(f'{c}@{int(ks[s])}:m{bad[0]+1}(voidKE={v[s,bad[0]]:.4f})')
print(f'  C5 ambiguous modes anywhere: {amb_tot}; ambiguous modes BELOW the selected mode: {amb_below}')
rec('ambiguous_modes','C','MODERATE' if amb_below else 'NOTE',
    f'{amb_tot} ambiguous modes total; {amb_below} of them lie below the selected structural mode',
    n_ambiguous=amb_tot,n_ambiguous_below_selection=amb_below,examples=';'.join(amb_ex))

# C6: near-degenerate clusters straddling the selection boundary
clus=0; clus_ex=[]
for c in EQ4A:
    v=G(c,'keLow')[:,IT,:]; om=G(c,'omega'); ks=G(c,'k')
    for s in range(v.shape[0]):
        idx=np.flatnonzero(np.isfinite(v[s])&(v[s]<0.5))
        if not idx.size: continue
        j=idx[0]
        if j>0 and np.isfinite(om[s,j-1]):
            if abs(om[s,j]-om[s,j-1])/om[s,j] < 1e-3:      # selected mode nearly degenerate with a rejected one
                clus+=1
                if len(clus_ex)<10: clus_ex.append(f'{c}@{int(ks[s])}')
print(f'  C6 selected mode within 0.1% of the rejected mode immediately below it: {clus} states')
rec('near_degenerate_selection_boundary','C','MODERATE' if clus else 'NOTE',
    f'{clus} states where the selected structural mode is within 0.1% of the rejected mode below it',
    n_affected=clus,examples=';'.join(clus_ex))

# ---------------- Candidate D falsification ----------------------------------
print('\n=== WP22 candidate D falsification ===')
gp=os.path.join(OUT,'GRAY_VS_BINARY_QUALITY.csv')
if os.path.exists(gp):
    B=list(csv.DictReader(open(gp)))
    f=lambda r,k: float(r[k])
    # D1: degenerate projections (near-mechanism binary structures)
    deg=[r for r in B if f(r,'binary_omega_E2') < 0.5*f(r,'gray_struct_omega_E2')]
    print(f'  D1 binary omega < 50% of gray structural: {len(deg)}/{len(B)} states')
    rec('degenerate_binary_projection','D','CRITICAL' if deg else 'NOTE',
        f'{len(deg)} of {len(B)} surveyed states have binary omega below half the gray structural value',
        n_affected=len(deg),n_total=len(B),
        examples=';'.join(f"{r['mesh']}@{r['state']}({f(r,'binary_omega_E2'):.3f}vs{f(r,'gray_struct_omega_E2'):.3f})" for r in deg[:10]))
    # D2: cutoff ties
    ties=[r for r in B if int(r['n_tied_at_cut'])>1]
    big=[r for r in B if int(r['n_tied_at_cut'])>10]
    print(f'  D2 states with >1 element tied at the exact-count cutoff : {len(ties)}/{len(B)}')
    print(f'     states with >10 elements tied                         : {len(big)}/{len(B)}')
    rec('projection_tie_arbitrariness','D','MAJOR' if big else 'MODERATE',
        f'{len(ties)} of {len(B)} states have >1 element tied at the cutoff; {len(big)} have >10, '
        f'where the projection is decided by the index tie-break rather than by density',
        n_ties_gt1=len(ties),n_ties_gt10=len(big),n_total=len(B),
        max_tied=max(int(r['n_tied_at_cut']) for r in B))
    # D3: discrete jumps in the binary sequence
    for mesh in sorted({r['mesh'] for r in B}):
        sub=[r for r in B if r['mesh']==mesh]
        sub.sort(key=lambda r:int(r['state']))
        q=np.array([f(r,'binary_omega_E2') for r in sub])
        d=np.abs(np.diff(q))/q[:-1]
        g=np.array([f(r,'gray_struct_omega_E2') for r in sub])
        dg=np.abs(np.diff(g))/g[:-1]
        if mesh=='160x20':
            print(f'  D3 {mesh}: binary step median {np.nanmedian(d):.3e} max {np.nanmax(d):.3e} '
                  f'steps>2%: {int(np.nansum(d>0.02))} | gray-structural steps>2%: {int(np.nansum(dg>0.02))}')
            rec('binary_discrete_jumps','D','MAJOR',
                f'{mesh}: binary sequence max step {np.nanmax(d):.3g}, {int(np.nansum(d>0.02))} steps above 2%; '
                f'gray structural sequence has {int(np.nansum(dg>0.02))} steps above 2%',
                mesh=mesh,binary_max_step=float(np.nanmax(d)),binary_steps_gt2pct=int(np.nansum(d>0.02)),
                gray_steps_gt2pct=int(np.nansum(dg>0.02)))
    # D4: disagreement with gray structural
    rel=np.array([abs(f(r,'binary_omega_E2')-f(r,'gray_struct_omega_E2'))/f(r,'gray_struct_omega_E2') for r in B])
    fin=np.isfinite(rel)
    print(f'  D4 |binary - gray structural| / gray structural: median {np.nanmedian(rel[fin]):.4f} '
          f'p95 {np.nanpercentile(rel[fin],95):.4f} max {np.nanmax(rel[fin]):.4f}')
    rec('binary_vs_gray_disagreement','D','MAJOR',
        f'median {np.nanmedian(rel[fin]):.4g}, p95 {np.nanpercentile(rel[fin],95):.4g}, max {np.nanmax(rel[fin]):.4g}',
        median=float(np.nanmedian(rel[fin])),p95=float(np.nanpercentile(rel[fin],95)),max=float(np.nanmax(rel[fin])))
    # D5: binary modal validity
    bv=np.array([f(r,'binary_voidKE_E2') for r in B])
    print(f'  D5 binary-field mode-1 void KE share: max {np.nanmax(bv):.6f} '
          f'(states >0.5: {int(np.nansum(bv>0.5))})')
    rec('binary_modal_validity','D','NOTE' if np.nanmax(bv)<0.5 else 'MAJOR',
        f'binary-field lowest mode void KE share max {np.nanmax(bv):.6g}; '
        f'{int(np.nansum(bv>0.5))} states void-localised',
        max_voidKE=float(np.nanmax(bv)),n_void_localised=int(np.nansum(bv>0.5)))
else:
    print('  (binary study not finished)')

with open(os.path.join(OUT,'FAILURE_CASES.csv'),'w',newline='') as f:
    ks_=sorted({k for r in out for k in r})
    w=csv.DictWriter(f,fieldnames=ks_,restval=''); w.writeheader(); w.writerows(out)
print(f'\n{len(out)} falsification probes recorded')
