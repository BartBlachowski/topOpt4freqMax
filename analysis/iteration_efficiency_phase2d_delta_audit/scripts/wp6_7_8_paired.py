#!/usr/bin/env python3
"""WP6/WP7/WP8 - independent reproduction of the OLD Eq.(4) defect and of the
Eq.(4a) cure, from stored density evidence only.  READ-ONLY, no optimizer."""
import sys, os, csv, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_evaluator import Qvec
REPO = '/Users/piotrek/Programming/topOpt4freqMax'
OUT  = os.path.join(REPO, 'analysis/iteration_efficiency_phase2d_delta_audit')
QR   = os.path.join(REPO, 'analysis/iteration_efficiency_phase2b_precision/qualification_runs')

FILES = [('gray_full_24x4_h200_paired_states.mat', 24, 4),
         ('s1_transition_96x12_h320_paired_states.mat', 96, 12)]

# ---------------- WP8: pure arithmetic on the known problematic pair --------------
print('=== WP8  branch-crossing arithmetic (no FE involved) ===')
d = 0.099999999999999644729
s = float(np.float32(0.1))
print(f'  double value                : {d!r}')
print(f'  its float32 image           : {float(np.float32(d))!r}')
print(f'  float32(0.1) as double      : {s!r}')
print(f'  same value?                 : {np.float32(d) == np.float32(0.1)}')
g4  = lambda x: x**6 if x <= 0.1 else x
g4a = lambda x: 1e5*x**6 if x <= 0.1 else x
for nm, f in (('Eq.(4) ', g4), ('Eq.(4a)', g4a)):
    a, b = f(d), f(s)
    print(f'  {nm}: g(double)={a:.17e}  g(single)={b:.17e}  '
          f'ratio={b/a:.6e}  rel|db|={abs(b-a)/a:.6e}')
print(f'  branch of double : {"low (x^6)" if d<=0.1 else "high (x)"}')
print(f'  branch of single : {"low (x^6)" if s<=0.1 else "high (x)"}')

# ---------------- WP6/WP7 (b): float32 paired states --------------------------------
rows = []
for fn, nx, ny in FILES:
    with h5py.File(os.path.join(QR, fn), 'r') as f:
        XD = f['x_double'][()]        # (npairs, nel)
        XS = f['x_single'][()]
        it = f['pairIterations'][()].ravel()
    for i in range(XD.shape[0]):
        xd = np.float64(XD[i]); xs = np.float64(XS[i])
        ncross = int(np.count_nonzero((xd <= 0.1) != (xs <= 0.1)))
        qo_d = Qvec(xd, nx, ny, False); qo_s = Qvec(xs, nx, ny, False)
        qn_d = Qvec(xd, nx, ny, True);  qn_s = Qvec(xs, nx, ny, True)
        ro = np.abs(qo_d-qo_s)/np.abs(qo_d)
        rn = np.abs(qn_d-qn_s)/np.abs(qn_d)
        rows.append(dict(source_file=fn, iteration=int(it[i]), n_branch_crossings=ncross,
                         max_abs_dx=float(np.max(np.abs(xd-xs))),
                         old_rel_E1=ro[0], old_rel_E2=ro[1], old_rel_E3=ro[2],
                         new_rel_E1=rn[0], new_rel_E2=rn[1], new_rel_E3=rn[2]))
    print(f'  processed {fn}: {XD.shape[0]} paired states')

with open(os.path.join(OUT, 'EQ4A_INDEPENDENT_STABILITY.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
cross = [r for r in rows if r['n_branch_crossings'] > 0]
with open(os.path.join(OUT, 'OLD_DEFECT_INDEPENDENT_REPRODUCTION.csv'), 'w', newline='') as f:
    ks = ['source_file','iteration','n_branch_crossings','old_rel_E1','old_rel_E2','old_rel_E3',
          'new_rel_E1','new_rel_E2','new_rel_E3']
    w = csv.DictWriter(f, fieldnames=ks, extrasaction='ignore'); w.writeheader(); w.writerows(cross)

mx = lambda k: max(r[k] for r in rows)
print('\n=== WP6 (float32 storage, 236 genuine paired states) ===')
print(f'  states={len(rows)}  with branch crossings={len(cross)}')
print(f'  OLD Eq.(4) : max rel E1={mx("old_rel_E1"):.4e}  E2={mx("old_rel_E2"):.4e}  E3={mx("old_rel_E3"):.4e}')
print(f'  NEW Eq.(4a): max rel E1={mx("new_rel_E1"):.4e}  E2={mx("new_rel_E2"):.4e}  E3={mx("new_rel_E3"):.4e}')
print(f'  reduction factor E2 = {mx("old_rel_E2")/mx("new_rel_E2"):.3e}')

# ---------------- WP6/WP7 (a): double-ULP straddle at the branch ---------------------
with h5py.File(os.path.join(QR, 's1_transition_96x12_h320_paired_states.mat'), 'r') as f:
    XD = f['x_double'][()]; it = f['pairIterations'][()].ravel()
u = []
nb = np.nextafter(0.1, 0.0); na = np.nextafter(0.1, 1.0)
for i in range(XD.shape[0]):
    x = np.float64(XD[i]); at = np.abs(x-0.1) < 1e-9
    if not at.any(): continue
    xm = x.copy(); xm[at] = nb
    xp = x.copy(); xp[at] = na
    qom = Qvec(xm, 96, 12, False); qop = Qvec(xp, 96, 12, False)
    qnm = Qvec(xm, 96, 12, True);  qnp = Qvec(xp, 96, 12, True)
    do = np.abs(qop-qom)/np.abs(qom); dn = np.abs(qnp-qnm)/np.abs(qnm)
    u.append(dict(iteration=int(it[i]), n_elements_at_branch=int(at.sum()),
                  density_perturbation=na-nb,
                  old_rel_dE1=do[0], old_rel_dE2=do[1], old_rel_dE3=do[2],
                  new_rel_dE1=dn[0], new_rel_dE2=dn[1], new_rel_dE3=dn[2]))
with open(os.path.join(OUT, 'DOUBLE_ULP_INDEPENDENT_REPRODUCTION.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(u[0].keys())); w.writeheader(); w.writerows(u)
mu = lambda k: max(r[k] for r in u)
print('\n=== WP7 (branch straddle: nextbelow(0.1) vs nextabove(0.1), all at-branch elements) ===')
print(f'  states={len(u)}   per-element density separation = {na-nb:.6e} (= 2 double ULP)')
print(f'  OLD Eq.(4) : max rel E1={mu("old_rel_dE1"):.4e}  E2={mu("old_rel_dE2"):.4e}  E3={mu("old_rel_dE3"):.4e}')
print(f'  NEW Eq.(4a): max rel E1={mu("new_rel_dE1"):.4e}  E2={mu("new_rel_dE2"):.4e}  E3={mu("new_rel_dE3"):.4e}')
print(f'  reduction factor E2 = {mu("old_rel_dE2")/mu("new_rel_dE2"):.3e}')

# single-element variant: how much does ONE element crossing the branch move omega_1?
i = max(range(XD.shape[0]), key=lambda j: np.count_nonzero(np.abs(np.float64(XD[j])-0.1) < 1e-9))
x = np.float64(XD[i]); at = np.flatnonzero(np.abs(x-0.1) < 1e-9)
xm = x.copy(); xm[at] = nb
xp = x.copy(); xp[at[:1]] = na; xp[at[1:]] = nb
qom = Qvec(xm, 96, 12, False); qop = Qvec(xp, 96, 12, False)
qnm = Qvec(xm, 96, 12, True);  qnp = Qvec(xp, 96, 12, True)
print(f'\n  SINGLE at-branch element straddled (iteration {int(it[i])}, {len(at)} at-branch):')
print(f'    OLD rel dE1/E2/E3 = {np.abs(qop-qom)/qom}')
print(f'    NEW rel dE1/E2/E3 = {np.abs(qnp-qnm)/qnm}')
