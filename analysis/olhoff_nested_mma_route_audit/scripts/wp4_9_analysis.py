#!/usr/bin/env python3
"""WP4/5/6/7/9/11/16 - phase analysis of the BASE_mma_160x20 trajectory."""
import os, csv, numpy as np
OUT='/Users/piotrek/Programming/topOpt4freqMax/analysis/olhoff_nested_mma_route_audit'
R=list(csv.DictReader(open(os.path.join(OUT,'BASE_MMA_HISTORY.csv'))))
g=lambda k,t=float: np.array([t(r[k]) for r in R])
it=g('outer',int); w1=g('omega1'); w2=g('omega2'); w3=g('omega3'); N=g('N',int)
sb=g('sqrt_beta'); ni=g('nInner',int); cum=g('cumInner',int); mdr=g('maxdrho'); vol=g('vol')
conv=np.array([r['innerConv']=='True' for r in R])

# ---------------- WP5 : multiplicity transitions ------------------------------
gap=np.abs(w2-w1)/w1
trans=[]
for i in range(1,len(N)):
    if N[i]!=N[i-1]:
        trans.append(dict(outer=int(it[i]),N_from=int(N[i-1]),N_to=int(N[i]),
            omega1=w1[i],omega2=w2[i],abs_gap=float(w2[i]-w1[i]),rel_gap=float(gap[i]),
            tolMult=0.05))
first2=int(it[N==2][0]) if (N==2).any() else None
# longest persistent N=2 run
best=(0,None,None); cur=None
for i in range(len(N)):
    if N[i]==2:
        if cur is None: cur=i
        if i-cur+1>best[0]: best=(i-cur+1,int(it[cur]),int(it[i]))
    else: cur=None
returns=sum(1 for t in trans if t['N_from']==2 and t['N_to']==1)
with open(os.path.join(OUT,'MULTIPLICITY_TRANSITIONS.csv'),'w',newline='') as f:
    if trans:
        w=csv.DictWriter(f,fieldnames=list(trans[0].keys())); w.writeheader(); w.writerows(trans)
print('=== WP5 multiplicity ===')
print(f'  first N=2 outer iteration          : {first2}')
print(f'  total N transitions                : {len(trans)}  (N=2 -> N=1 returns: {returns})')
print(f'  longest persistent N=2 interval    : {best[1]}..{best[2]}  ({best[0]} iterations)')
print(f'  outer iterations with N=1 / N=2    : {int((N==1).sum())} / {int((N==2).sum())}')
print(f'  N values observed                  : {sorted(set(N.tolist()))}')
print('  transitions (first 12):')
for t in trans[:12]:
    print(f'    it {t["outer"]:4d}  N {t["N_from"]}->{t["N_to"]}  w1={t["omega1"]:.2f} w2={t["omega2"]:.2f} '
          f'abs gap={t["abs_gap"]:.3f}  rel gap={t["rel_gap"]*100:.3f}%')

# ---------------- WP4 / WP6 : inner work by phase -----------------------------
m1=N==1; m2=N==2
def stats(mask,label):
    x=ni[mask]; c=conv[mask]
    return dict(phase=label,n_outer=int(mask.sum()),
        inner_mean=float(x.mean()),inner_median=float(np.median(x)),
        inner_min=int(x.min()),inner_p90=float(np.percentile(x,90)),
        inner_p95=float(np.percentile(x,95)),inner_max=int(x.max()),
        inner_total=int(x.sum()),
        cap_hits=int((x>=300).sum()),cap_hit_fraction=float((x>=300).mean()),
        converged=int(c.sum()),converged_fraction=float(c.mean()),
        move_saturated_fraction=float((mdr[mask]>=0.00995).mean()))
S=[stats(m1,'N=1 (simple)'),stats(m2,'N=2 (multiple)'),stats(np.ones_like(m1,bool),'ALL')]
with open(os.path.join(OUT,'INNER_WORK_SUMMARY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(S[0].keys())); w.writeheader(); w.writerows(S)
print('\n=== WP4/WP6 inner MMA work by phase ===')
print(f"{'phase':16s} {'outers':>7} {'mean':>7} {'med':>6} {'min':>5} {'p90':>6} {'p95':>6} {'max':>5} {'total':>8} {'cap%':>7} {'conv%':>7}")
for s in S:
    print(f"{s['phase']:16s} {s['n_outer']:7d} {s['inner_mean']:7.1f} {s['inner_median']:6.0f} "
          f"{s['inner_min']:5d} {s['inner_p90']:6.0f} {s['inner_p95']:6.0f} {s['inner_max']:5d} "
          f"{s['inner_total']:8d} {100*s['cap_hit_fraction']:6.2f}% {100*s['converged_fraction']:6.2f}%")
from scipy import stats as sps
u=sps.mannwhitneyu(ni[m1],ni[m2],alternative='less')
print(f"\n  Mann-Whitney U (N=1 inner < N=2 inner): U={u.statistic:.0f}  p={u.pvalue:.3e}")
print(f"  ratio of means N=2/N=1 = {ni[m2].mean()/ni[m1].mean():.3f}")

# ---------------- WP7 : does a cap-hit inner solve harm the next outer step? ---
prev_cap=np.zeros(len(R),bool); prev_cap[1:]=(ni[:-1]>=300)
prev_conv=np.zeros(len(R),bool); prev_conv[1:]=conv[:-1]
d1=np.zeros(len(R)); d1[1:]=w1[1:]-w1[:-1]
d2=np.zeros(len(R)); d2[1:]=w2[1:]-w2[:-1]
dg=np.zeros(len(R)); dg[1:]=gap[1:]-gap[:-1]
sel=m2.copy(); sel[0]=False       # compare only inside the N=2 regime
rows7=[]
for lab,msk in (('prev inner CONVERGED',sel&prev_conv),('prev inner CAP-HIT',sel&prev_cap)):
    rows7.append(dict(group=lab,n=int(msk.sum()),
        mean_delta_omega1=float(d1[msk].mean()),median_delta_omega1=float(np.median(d1[msk])),
        std_delta_omega1=float(d1[msk].std()),
        mean_delta_omega2=float(d2[msk].mean()),
        mean_abs_delta_omega1=float(np.abs(d1[msk]).mean()),
        mean_delta_relgap=float(dg[msk].mean()),
        mean_maxdrho=float(mdr[msk].mean()),
        frac_omega1_decreased=float((d1[msk]<0).mean())))
with open(os.path.join(OUT,'INNER_STATUS_VS_OUTER_PROGRESS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows7[0].keys())); w.writeheader(); w.writerows(rows7)
print('\n=== WP7 outer progress conditioned on previous inner status (N=2 regime) ===')
for r in rows7:
    print(f"  {r['group']:22s} n={r['n']:4d}  mean dw1={r['mean_delta_omega1']:+.5f}  "
          f"median dw1={r['median_delta_omega1']:+.5f}  mean|dw1|={r['mean_abs_delta_omega1']:.5f}  "
          f"P(dw1<0)={r['frac_omega1_decreased']:.3f}  mean maxdrho={r['mean_maxdrho']:.5f}")
a=d1[sel&prev_conv]; b=d1[sel&prev_cap]
if len(b)>2:
    t=sps.mannwhitneyu(a,b,alternative='two-sided')
    print(f"  Mann-Whitney two-sided on delta omega1: U={t.statistic:.0f}  p={t.pvalue:.4f}")

# ---------------- WP9 / WP11 : spectral trajectory and endpoint ---------------
i_best=int(np.argmax(w1))
tail=slice(len(R)-100,len(R))
rows9=[dict(quantity='max omega1',value=float(w1.max()),at_outer=int(it[i_best])),
       dict(quantity='terminal omega1',value=float(w1[-1]),at_outer=int(it[-1])),
       dict(quantity='terminal omega2',value=float(w2[-1]),at_outer=int(it[-1])),
       dict(quantity='terminal omega3',value=float(w3[-1]),at_outer=int(it[-1])),
       dict(quantity='terminal abs gap',value=float(w2[-1]-w1[-1]),at_outer=int(it[-1])),
       dict(quantity='terminal rel gap',value=float(gap[-1]),at_outer=int(it[-1])),
       dict(quantity='terminal maxdrho',value=float(mdr[-1]),at_outer=int(it[-1])),
       dict(quantity='tolOuter',value=1e-3,at_outer=-1),
       dict(quantity='last-100 omega1 mean',value=float(w1[tail].mean()),at_outer=-1),
       dict(quantity='last-100 omega1 std',value=float(w1[tail].std()),at_outer=-1),
       dict(quantity='last-100 omega1 peak-to-peak',value=float(np.ptp(w1[tail])),at_outer=-1),
       dict(quantity='last-100 maxdrho min',value=float(mdr[tail].min()),at_outer=-1),
       dict(quantity='last-100 maxdrho mean',value=float(mdr[tail].mean()),at_outer=-1),
       dict(quantity='last-100 fraction move-saturated (>=0.00995)',value=float((mdr[tail]>=0.00995).mean()),at_outer=-1),
       dict(quantity='final volume',value=float(vol[-1]),at_outer=int(it[-1])),
       dict(quantity='total cumulative inner MMA iterations',value=float(cum[-1]),at_outer=int(it[-1])),
       ]
with open(os.path.join(OUT,'SPECTRAL_TRAJECTORY_ANALYSIS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['quantity','value','at_outer']); w.writeheader(); w.writerows(rows9)
print('\n=== WP9/WP11 spectral trajectory and endpoint ===')
for r in rows9: print(f"  {r['quantity']:48s} {r['value']:>14.6f}" + (f"  @ outer {r['at_outer']}" if r['at_outer']>0 else ''))
print(f"\n  omega1 monotone-increasing fraction of steps: {(np.diff(w1)>0).mean():.3f}")
print(f"  omega1 last-200 trend (linear slope per iter): {np.polyfit(it[-200:],w1[-200:],1)[0]:+.3e}")
print(f"  near-bimodal (rel gap<1%) outer iterations   : {int((gap<0.01).sum())}")
print(f"  near-bimodal (rel gap<0.5%) outer iterations : {int((gap<0.005).sum())}")
