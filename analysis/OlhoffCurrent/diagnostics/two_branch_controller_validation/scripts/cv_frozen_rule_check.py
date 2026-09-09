#!/usr/bin/env python3
"""Independently re-derive the frozen A/B rule on the recomputed F400 arm.

PREREGISTRATION section 2.6 recorded specific values obtained by running the frozen
MATLAB detector (two_branch_maturity_240/scripts/tb_branches.m) against the
original F400 trajectory.  That trajectory was lost and recomputed here.  This
script re-derives the same quantities from the frozen DEFINITIONS in Python, so
agreement is between two independent implementations rather than a re-run of one.

Frozen definitions (PREREGISTRATION section 2.1-2.4):
    d_k      = rho_k - rho_{k-1}
    cos(k)   = <d_k, d_{k-1}> / (||d_k|| ||d_{k-1}||)
    net(k)   = ||rho_k - rho_{k-10}|| / sum_{j=k-9}^{k} ||d_j||          W_np = 10
    amp(k)   = ||d_k||_2
    med20 x  = trailing median over [k-19, k], omitnan
    A(k) = med20cos < 0 and med20net < 0.5 and amp >= tol
    B(k) = amp < tol and med20cos > 0
"""
import json, os, sys
import numpy as np
import h5py

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
F400 = os.path.join(REPO, 'analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat')
STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Values recorded in PREREGISTRATION section 2.6 from the frozen MATLAB detector.
FROZEN = {
    'tol': 0.125,
    'branchA_ever': False,
    'first_B': 369,
    'nativeStop': 369,
    'med20cos_at_369': 0.9937451892796663,
    'med20net_at_369': 0.9728297014945854,
    'amp_at_369': 0.1241004791601378,
    'Mnd_at_369': 16.158892933214315,
    'omega1_at_369': 166.3649498603138,
}


def trailing_median(v, k, W=20):
    lo = max(0, k - W + 1)
    w = v[lo:k + 1]
    w = w[~np.isnan(w)]
    return np.nan if w.size == 0 else float(np.median(w))


def main():
    if not os.path.isfile(F400):
        print('F400 not present:', F400); return 1
    with h5py.File(F400, 'r') as f:
        RHO = np.array(f['RHO'])                      # (nOuter, NE)
        omega = np.array(f['hist']['omega'])   # (nOuter, nModes)
    n, NE = RHO.shape
    tol = 0.05 * np.sqrt(NE / 3200)
    rho0 = 0.5
    R = np.vstack([np.full((1, NE), rho0), RHO])      # R[0] = rho_0
    D = np.diff(R, axis=0)                            # D[k-1] = d_k, k = 1..n
    amp = np.linalg.norm(D, axis=1)
    cos = np.full(n, np.nan)
    for k in range(1, n):                             # k index -> iteration k+1
        a, b = D[k], D[k - 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos[k] = float(a @ b) / (na * nb)
    net = np.full(n, np.nan)
    for k in range(9, n):                             # needs rho_{k-9} .. anchor
        num = np.linalg.norm(R[k + 1] - R[k + 1 - 10])
        den = amp[k - 9:k + 1].sum()
        if den > 0:
            net[k] = num / den
    mc = np.array([trailing_median(cos, k) for k in range(n)])
    mn = np.array([trailing_median(net, k) for k in range(n)])
    A = (mc < 0) & (mn < 0.5) & (amp >= tol)
    B = (amp < tol) & (mc > 0)
    A = np.where(np.isnan(mc) | np.isnan(mn), False, A)
    B = np.where(np.isnan(mc), False, B)

    iA = np.where(A)[0]
    iB = np.where(B)[0]
    rho_end = RHO[-1, :]
    got = {
        'nOuter': int(n),
        'tol': float(tol),
        'branchA_ever': bool(A.any()),
        'first_B': int(iB[0] + 1) if iB.size else None,
        'med20cos_at_369': float(mc[368]),
        'med20net_at_369': float(mn[368]),
        'amp_at_369': float(amp[368]),
        'Mnd_at_369': float(100 * np.mean(4 * rho_end * (1 - rho_end))),
        'omega1_at_369': float(omega[-1, 0]) if omega.size else None,
    }
    print(f'recomputed F400: nOuter={n}  NE={NE}  tol={tol}')
    print(f"{'quantity':22s} {'frozen (MATLAB)':>24s} {'re-derived (Python)':>24s}  agree")
    ok = True
    for k, fv in FROZEN.items():
        if k == 'nativeStop':
            continue
        gv = got.get(k)
        if isinstance(fv, bool) or fv is None:
            agree = (gv == fv)
        elif isinstance(gv, (int,)) or isinstance(fv, int):
            agree = (gv == fv)
        else:
            agree = gv is not None and abs(gv - fv) <= 1e-12 * max(1.0, abs(fv))
        ok &= bool(agree)
        print(f'{k:22s} {str(fv):>24s} {str(gv):>24s}  {agree}')
    print(f'\nALL FROZEN VALUES REPRODUCED: {ok}')
    out = {'frozen': FROZEN, 'rederived': got, 'allAgree': bool(ok),
           'note': 'Two independent implementations of the frozen rule, on a '
                   'recomputed F400 trajectory.'}
    p = os.path.join(STUDY, 'evidence', 'frozen_rule_recheck.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
