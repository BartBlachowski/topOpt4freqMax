#!/usr/bin/env python3
"""
WP10 (mode identity / crossings, MAC), WP16 (evaluator-perspective redundancy),
WP20 (reference/persistence compatibility), WP21 (hard gate vs modal validity).
Consumes survey.npz; recomputes eigenvectors on 160x20 for the MAC study.
READ-ONLY.  Sparse solver throughout.
"""
import sys, os, csv, time, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_engine import modes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))),'iteration_efficiency_phase2d_delta_audit','scripts'))
from frozen_engines import reference_phase, scan_persistence, acceptance
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
Z=np.load(os.path.join(OUT,'scripts','survey.npz'))
GF=np.load(os.path.join(OUT,'scripts','gate_full.npz'))   # full-coverage hard gate
TAUS=Z['TAUS']
# Primary diagnostic partition: tau = 0.1, the location of the Du-Olhoff mass branch.
# NOT tau=0.02: the artificial modes are localised at rho ~ 0.05-0.10, so a partition at
# 0.02 excludes the very elements carrying them, their voidKE collapses to ~0, every mode
# is accepted, and the selection degenerates to ordinal 1 at every state.  Any "plateau"
# measured there is vacuous.  See MODAL_DIAGNOSTIC_DEFINITIONS.md.
IT=int(np.argmin(np.abs(TAUS-0.10)))
G=lambda c,f: Z[f'{c}|{f}']
CFGS=[c[:-6] for c in Z.files if c.endswith('|omega')]

def sel_struct(cfg,cut=0.5,ti=IT):
    v=G(cfg,'keLow')[:,ti,:]; om=G(cfg,'omega')
    o=np.full(v.shape[0],-1); w=np.full(v.shape[0],np.nan)
    for s in range(v.shape[0]):
        idx=np.flatnonzero(np.isfinite(v[s])&(v[s]<cut))
        if idx.size: o[s]=idx[0]+1; w[s]=om[s,idx[0]]
    return o,w,om[:,0]

# ================= WP21 : hard gate vs modal validity =========================
rows=[]
for cfg in CFGS:
    mesh,model,law=cfg.split('|')
    gk=f'{mesh}|GATE2'
    if f'{gk}|k' not in GF.files: continue
    hk=GF[f'{gk}|k']; hard=GF[f'{gk}|hard'].astype(bool)
    o,ws,low=sel_struct(cfg); ks=G(cfg,'k')
    m={int(a):(b,c,d) for a,b,c,d in zip(ks,o,ws,low)}
    common=[int(a) for a in hk if int(a) in m]
    hmask=np.array([bool(h) for a,h in zip(hk,hard) if int(a) in m])
    ordv=np.array([m[a][0] for a in common]); wsv=np.array([m[a][1] for a in common])
    lowv=np.array([m[a][2] for a in common])
    inval=ordv>1
    rows.append(dict(mesh=mesh,model=model,mass_law=law,n_states=len(common),
        hard_gate_pass=int(hmask.sum()),
        states_with_invalid_lowest_mode=int(inval.sum()),
        pct_states_invalid=100*inval.mean(),
        hardgate_pass_AND_invalid_lowest=int((hmask&inval).sum()),
        pct_of_hardgate_pass_invalid=100*(hmask&inval).sum()/max(1,hmask.sum()),
        worst_lowest_over_structural=float(np.nanmin(lowv/wsv)) if np.isfinite(wsv).any() else float('nan'),
        no_structural_mode_found=int((ordv<0).sum())))
with open(os.path.join(OUT,'HARD_GATE_VS_MODAL_VALIDITY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('=== WP21 hard gate vs modal validity ===')
for r in rows:
    if r['mass_law']!='eq4a' and r['model']!='E1': continue
    print(f"  {r['mesh']:8s} {r['model']} {r['mass_law']:6s}: invalid lowest {r['states_with_invalid_lowest_mode']:5d}"
          f"/{r['n_states']:5d} ({r['pct_states_invalid']:5.2f}%)  "
          f"hard-gate PASS & invalid {r['hardgate_pass_AND_invalid_lowest']:5d}/{r['hard_gate_pass']:5d} "
          f"({r['pct_of_hardgate_pass_invalid']:5.2f}%)  worst lowest/struct {r['worst_lowest_over_structural']:.4f}")

# ================= WP16 : evaluator-perspective redundancy ====================
red=[]
for mesh in sorted({c.split('|')[0] for c in CFGS}):
    have={c.split('|')[1]+'/'+c.split('|')[2]:c for c in CFGS if c.startswith(mesh+'|')}
    def series(tag):
        c=have.get(tag)
        if c is None: return None,None
        o,ws,low=sel_struct(c); return G(c,'k'),ws
    k1,e1=series('E1/linear'); k2,e2=series('E2/eq4a'); k3,e3=series('E3/eq4a')
    if e2 is None or e3 is None: continue
    common=sorted(set(k2.tolist())&set(k3.tolist())&(set(k1.tolist()) if k1 is not None else set(k2.tolist())))
    idx=lambda kk,ee:{int(a):b for a,b in zip(kk,ee)}
    m2,m3=idx(k2,e2),idx(k3,e3); m1=idx(k1,e1) if k1 is not None else None
    a2=np.array([m2[a] for a in common]); a3=np.array([m3[a] for a in common])
    ok=np.isfinite(a2)&np.isfinite(a3)
    d23=np.abs(a2-a3)/a2
    rec=dict(mesh=mesh,n_states=int(ok.sum()),
        E2_E3_median_rel_diff=float(np.median(d23[ok])),E2_E3_p95_rel_diff=float(np.percentile(d23[ok],95)),
        E2_E3_max_rel_diff=float(np.nanmax(d23[ok])),
        E2_E3_correlation=float(np.corrcoef(a2[ok],a3[ok])[0,1]))
    if m1 is not None:
        a1=np.array([m1[a] for a in common]); ok1=ok&np.isfinite(a1)
        d12=np.abs(a1-a2)/a1
        rec.update(E1_E2_median_rel_diff=float(np.median(d12[ok1])),
                   E1_E2_p95_rel_diff=float(np.percentile(d12[ok1],95)),
                   E1_E2_max_rel_diff=float(np.nanmax(d12[ok1])),
                   E1_E2_correlation=float(np.corrcoef(a1[ok1],a2[ok1])[0,1]),
                   binding_E1=int(np.sum((a1<=a2)&(a1<=a3))),binding_E2=int(np.sum((a2<a1)&(a2<=a3))),
                   binding_E3=int(np.sum((a3<a1)&(a3<a2))))
    red.append(rec)
with open(os.path.join(OUT,'EVALUATOR_PERSPECTIVE_REDUNDANCY.csv'),'w',newline='') as f:
    ks_=sorted({k for r in red for k in r}); w=csv.DictWriter(f,fieldnames=ks_,restval='')
    w.writeheader(); w.writerows(red)
print('\n=== WP16 evaluator redundancy after structural-mode selection ===')
for r in red:
    print(f"  {r['mesh']:8s} E2-E3 median {r['E2_E3_median_rel_diff']:.3e} p95 {r['E2_E3_p95_rel_diff']:.3e} "
          f"max {r['E2_E3_max_rel_diff']:.3e}" + (f"  |  E1-E2 median {r['E1_E2_median_rel_diff']:.3e}" if 'E1_E2_median_rel_diff' in r else ''))

# ================= WP10 : mode identity and crossings (MAC) ===================
print('\n=== WP10 mode identity / crossings, 160x20 E2 Eq.(4a) ===')
with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    X=f['res/rho_snapshots'][()]
KMAC=12; prevU=None; prevSel=None; mac_rows=[]; t0=time.time()
for k in range(1,1601):
    z=np.clip(np.float64(X[k]),0,1)
    r=modes(z,160,20,'E2','eq4a',k=KMAC,return_vectors=True)
    low=r['zeff']<=0.1; v=r['ke_n'][low,:].sum(axis=0)
    idx=np.flatnonzero(v<0.5); sel=int(idx[0]) if idx.size else -1
    U=r['U']
    if prevU is not None and sel>=0 and prevSel>=0:
        a=prevU[:,prevSel]; B=U
        num=(a@B)**2; mac=num/((a@a)*np.einsum('ij,ij->j',B,B))
        best=int(np.argmax(mac))
        mac_rows.append(dict(state=k,selected_ordinal=sel+1,prev_selected_ordinal=prevSel+1,
            omega=float(r['omega'][sel]),
            MAC_selected_vs_prev_selected=float(mac[sel]),
            best_MAC_partner_ordinal=best+1,best_MAC=float(mac[best]),
            identity_preserved=bool(best==sel),
            ordinal_changed=bool(sel!=prevSel)))
    prevU=U; prevSel=sel
    if k%400==0: print(f'  MAC {k}/1600 [{time.time()-t0:.0f}s]',flush=True)
with open(os.path.join(OUT,'MODE_IDENTITY_MAC.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(mac_rows[0].keys())); w.writeheader(); w.writerows(mac_rows)
mm=np.array([r['MAC_selected_vs_prev_selected'] for r in mac_rows])
oc=np.array([r['ordinal_changed'] for r in mac_rows])
ip=np.array([r['identity_preserved'] for r in mac_rows])
print(f'  consecutive-state MAC of the selected structural mode: median {np.median(mm):.4f} '
      f'p05 {np.percentile(mm,5):.4f} min {mm.min():.4f}')
print(f'  selection ordinal changed at {int(oc.sum())}/{len(oc)} steps; '
      f'best-MAC partner == selected at {int(ip.sum())}/{len(ip)} steps')
print(f'  steps with MAC<0.9 (genuine mode change or crossing): {int((mm<0.9).sum())}')

# ================= WP20 : reference / persistence compatibility ===============
print('\n=== WP20 scalar quality-sequence suitability (offline, 1600-horizon) ===')
gk='160x20|GATE2'; hk=GF[f'{gk}|k']; hard=GF[f'{gk}|hard'].astype(bool)
seq={}
for tag,cfg in (('A: Eq.(4) lowest','160x20|E2|eq4'),('B: Eq.(4a) lowest','160x20|E2|eq4a'),
                ('C: Eq.(4a) structural','160x20|E2|eq4a')):
    o,ws,low=sel_struct(cfg)
    seq[tag]=ws if tag.startswith('C') else low
wp20=[]
for tag,q in seq.items():
    d=np.abs(np.diff(q))/q[:-1]
    wp20.append(dict(candidate=tag,n_states=int(q.size),undefined_states=int((~np.isfinite(q)).sum()),
        median_step_rel=float(np.nanmedian(d)),p99_step_rel=float(np.nanpercentile(d,99)),
        max_step_rel=float(np.nanmax(d)),
        steps_gt_0p5pct=int(np.nansum(d>0.005)),steps_gt_1pct=int(np.nansum(d>0.01)),
        steps_gt_2pct=int(np.nansum(d>0.02))))
    print(f"  {tag:24s} undefined {wp20[-1]['undefined_states']:4d}  median step {wp20[-1]['median_step_rel']:.3e}  "
          f"max step {wp20[-1]['max_step_rel']:.3e}  steps>0.5%: {wp20[-1]['steps_gt_0p5pct']:4d}")
with open(os.path.join(OUT,'QUALITY_SEQUENCE_SUITABILITY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(wp20[0].keys())); w.writeheader(); w.writerows(wp20)
print('\ndone')
