#!/usr/bin/env python3
"""WP12/WP13 - decision-margin analysis of the frozen reference/persistence
machinery on the ONLY reference-length (B_ref = 3200) trajectory that exists in
the repository: the Phase-2B 96x12 probe.  Densities were not retained, so the
amended Eq.(4a) quality sequence cannot be recomputed; what CAN be established
independently is how large a pointwise relative perturbation of Q the frozen
decisions tolerate.  READ-ONLY."""
import sys, os, csv, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_engines import reference_phase, scan_persistence, measurement_budget, acceptance
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')
QR=os.path.join(REPO,'analysis/iteration_efficiency_phase2b_recheck/qualification_runs')
LV=[0.98,0.99,0.995]; P=100; LREF=500; EPS=1e-3; BREF=3200

with h5py.File(os.path.join(QR,'probe_96x12_H3200.mat'),'r') as f:
    QD=f['Qhi'][()].T; H0=f['H0'][()].ravel().astype(bool)     # Qhi == the double trajectory
ref=reference_phase(QD,H0); b_ref=ref['b_ref']; Qref=ref['Q_ref']
passM,rob=acceptance(QD,Qref,H0,LV); per=scan_persistence(passM,P)
print(f'nominal: b_ref={b_ref}  k_enter={per["k_enter"]}  k_cert={per["k_cert"]}')

# ---------- 1. b_ref margins, block by block --------------------------------------
brows=[]
for b1 in range(P,BREF+1,P):
    b=b1-1; bl=b1-LREF-1
    if bl<0 or not np.isfinite(ref['F'][b]).all(): continue
    g=ref['gain'][b]
    brows.append(dict(b=b1,max_gain=float(np.nanmax(g)),
                      gain_E1=g[0],gain_E2=g[1],gain_E3=g[2],
                      candidate=bool(ref['candidate'][b]),
                      distance_to_epsilon=float(EPS-np.nanmax(g)),
                      abs_distance=float(abs(EPS-np.nanmax(g)))))
with open(os.path.join(OUT,'WP12_BREF_BLOCK_MARGINS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
# binding margins: the chosen endpoint must stay a candidate; the last non-candidate
# endpoint before it must stay a non-candidate.
sel=[r for r in brows if r['b']==b_ref][0]
prev=[r for r in brows if r['b']<b_ref]
print(f'\nb_ref block margins (epsilon_ref={EPS}):')
print(f'  chosen b={b_ref}: max gain={sel["max_gain"]:.6e}  slack below eps = {sel["distance_to_epsilon"]:.6e}')
if prev:
    worst=min(prev,key=lambda r:r['abs_distance'])
    print(f'  nearest earlier non-candidate b={worst["b"]}: max gain={worst["max_gain"]:.6e}'
          f'  excess over eps = {worst["max_gain"]-EPS:.6e}')
    bref_gain_margin=min(sel['distance_to_epsilon'],worst['max_gain']-EPS)
else:
    bref_gain_margin=sel['distance_to_epsilon']
print(f'  binding gain margin = {bref_gain_margin:.6e}')

# ---------- 2. interval (worst-case) propagation of a bounded relative Q error -----
def bref_interval(delta):
    """Exact monotone bounds: F is a max of mins of Q, so scaling every Q by
    (1+-delta) scales F by (1+-delta).  gain = 1 - F(b-L)/F(b)."""
    F=ref['F']; lo=None; hi=None
    cand_lo=[]; cand_hi=[]
    for b1 in range(P,BREF+1,P):
        b=b1-1; bl=b1-LREF-1
        if bl<0 or not np.isfinite(F[b]).all() or not (F[b]>0).all(): continue
        r=F[bl]/F[b]
        gmin=1-r*(1+delta)/(1-delta)     # smallest possible gain
        gmax=1-r*(1-delta)/(1+delta)     # largest possible gain
        cand_lo.append((b1,bool((gmax<=EPS).all())))   # candidate even in the worst case
        cand_hi.append((b1,bool((gmin<=EPS).all())))   # candidate in the best case
    def first(c):
        for b1,ok in c:
            if ok and b1>=P+LREF: return b1
        return None
    return first(cand_hi), first(cand_lo)   # (earliest possible, latest possible)

def kenter_interval(delta):
    """Acceptance is rob>=q with rob=min_e Q_e/Q_ref_e.  Both numerator and the
    reference (itself built from Q) move by at most delta relative, so
    rob moves by at most factor (1+delta)/(1-delta)."""
    fac=(1+delta)/(1-delta)
    out={}
    for j,q in enumerate(LV):
        p_opt=H0&(rob*fac>=q)      # most permissive
        p_pes=H0&(rob/fac>=q)      # least permissive
        so=scan_persistence(p_opt[:,None],P); sp=scan_persistence(p_pes[:,None],P)
        out[q]=(so['k_enter'][0],sp['k_enter'][0])
    return out

def bisect(pred, lo=1e-16, hi=1e-1, it=200):
    if pred(hi) is False: return None
    for _ in range(it):
        mid=np.sqrt(lo*hi)
        if pred(mid): hi=mid
        else: lo=mid
    return hi

d_bref=bisect(lambda d: bref_interval(d)!=(b_ref,b_ref))
kk={q:(per['k_enter'][j],) for j,q in enumerate(LV)}
d_kent={q: bisect(lambda d,q=q,j=j: kenter_interval(d)[q]!=(per['k_enter'][j],per['k_enter'][j]))
        for j,q in enumerate(LV)}
print(f'\ncritical uniform relative Q perturbation (worst-case interval analysis):')
print(f'  smallest delta that can move b_ref      : {d_bref:.4e}')
for q in LV:
    print(f'  smallest delta that can move k_enter q={q}: {d_kent[q]:.4e}')

# ---------- 3. pointwise acceptance margins ---------------------------------------
arows=[]
for j,q in enumerate(LV):
    m=np.abs(rob-q)/rob
    m_valid=m[H0]
    # decisive states: those inside the nominal certification window, or before it
    ke=int(per['k_enter'][j]); kc=int(per['k_cert'][j])
    inwin=np.zeros_like(H0); inwin[ke-1:kc]=True
    arows.append(dict(q=q,
        min_margin_all_states=float(m_valid.min()),
        min_margin_in_cert_window=float(m[inwin].min()),
        margin_at_state_before_kenter=float(m[ke-2]) if ke>=2 else float('nan'),
        n_states_margin_lt_1e_6=int((m_valid<1e-6).sum()),
        n_states_margin_lt_1e_7=int((m_valid<1e-7).sum()),
        n_states_margin_lt_5p6e_8=int((m_valid<5.6e-8).sum()),
        n_states_margin_lt_2x5p6e_8=int((m_valid<1.12e-7).sum()),
        critical_delta_kenter=d_kent[q]))
with open(os.path.join(OUT,'WP12_ACCEPTANCE_MARGINS.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(arows[0].keys())); w.writeheader(); w.writerows(arows)
print('\npointwise acceptance margins (relative, on the robust ratio):')
for r in arows:
    print(f"  q={r['q']}: min over all states={r['min_margin_all_states']:.4e}  "
          f"min inside cert window={r['min_margin_in_cert_window']:.4e}  "
          f"#states within 1.12e-7={r['n_states_margin_lt_2x5p6e_8']}")

# ---------- 4. summary vs the amended perturbation bounds --------------------------
AMEND_F32=5.596e-8; AMEND_ULP=8.3e-13
srows=[]
srows.append(dict(decision='b_ref',critical_delta=d_bref,
    amended_float32_delta=AMEND_F32,safety_factor_f32=d_bref/AMEND_F32,
    amended_double_ulp_delta=AMEND_ULP,safety_factor_ulp=d_bref/AMEND_ULP))
for j,q in enumerate(LV):
    srows.append(dict(decision=f'k_enter/k_cert q={LV[j]}',critical_delta=d_kent[LV[j]],
        amended_float32_delta=AMEND_F32,safety_factor_f32=d_kent[LV[j]]/AMEND_F32,
        amended_double_ulp_delta=AMEND_ULP,safety_factor_ulp=d_kent[LV[j]]/AMEND_ULP))
with open(os.path.join(OUT,'WP12_CRITICAL_PERTURBATION.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)
print('\nsafety factors (critical delta / amended perturbation):')
for r in srows:
    print(f"  {r['decision']:24s} delta*={r['critical_delta']:.3e}  "
          f"/f32 = {r['safety_factor_f32']:.3g}x   /ulp = {r['safety_factor_ulp']:.3g}x")

# ---------- 5. what the OLD Eq.(4) perturbation did on this same trajectory --------
with h5py.File(os.path.join(QR,'probe_96x12_H3200.mat'),'r') as f:
    QS=f['Qlo'][()].T
relerr=np.abs(QS-QD)/np.abs(QD)
print(f'\nOLD Eq.(4) double-vs-single relative error on this 3200 trajectory:')
print(f'  E1 max={relerr[:,0].max():.4e}  E2 max={relerr[:,1].max():.4e}  E3 max={relerr[:,2].max():.4e}')
print(f'  fraction of states with E2 error > critical delta for k_enter(0.995) '
      f'({d_kent[0.995]:.3e}): {(relerr[:,1]>d_kent[0.995]).mean():.3f}')
