#!/usr/bin/env python3
"""Per-outer-iteration telemetry (AUDIT_PREREGISTRATION sec. 7) for the three 480x60
trajectories: target C480 three-rung canary (retained), source S480 (retained hist/aux),
source M1 (this audit's one run).  Post hoc only; nothing is fed back.

Indexing: RHO[k-1] is the design AFTER outer iteration k; omega[k-1] is evaluated at the
START state of iteration k (before its update), exactly as olhoffSolve records it.
"""
import csv
import numpy as np
import h5py
from sd_common import *

RHOMIN = 1e-3


def bound_sat(rho_start, d, box):
    lo = np.maximum(RHOMIN - rho_start, -box)
    hi = np.minimum(1 - rho_start, box)
    w = np.maximum(hi - lo, 1e-300)
    on = (d <= lo + 1e-6 * w) | (d >= hi - 1e-6 * w)
    at_move = np.abs(d) >= 0.99 * box
    return float(on.mean()), float(at_move.mean())


def full_rows(RHO, DRHO, BOX, om, h, extra):
    n = om.shape[0]
    rows = []
    prevD = None
    rho_start = np.full(NE, 0.5)
    lam = om[:, :2] ** 2
    for k in range(n):
        r = RHO[k]
        d = DRHO[k]
        D = r - rho_start
        box = BOX[k]
        disc = discreteness(r)
        on, atm = bound_sat(rho_start, d, box)
        rev = float(np.mean(D * prevD < 0)) if prevD is not None else np.nan
        beta = float(h['beta'].ravel()[k])
        pred = beta - lam[k, 0]
        real = lam[k + 1, 0] - lam[k, 0] if k + 1 < n else np.nan
        row = dict(outer=k + 1, rho_sha256=sha256_double(r), omega1=om[k, 0], omega2=om[k, 1], omega3=om[k, 2],
                   lambda1=lam[k, 0], lambda2=lam[k, 1], gap12=(om[k, 1] - om[k, 0]) / om[k, 0],
                   dOff=lam[k, 1] - lam[k, 0], volume=float(h['vol'].ravel()[k]),
                   Mnd=disc['Mnd'], gray=disc['gray'], mid=disc['mid'], broad_core=disc['broad_core_fraction'],
                   void_lt_0p1=disc['void_lt_0p1'], at_rhomin=disc['at_rhomin'],
                   beta=beta, box_max=float(np.max(box)), box_mean=float(np.mean(box)),
                   box_at_floor=float(np.mean(np.asarray(box) <= 0.002 + 1e-15)) if np.ndim(box) else np.nan,
                   N=int(h['N'].ravel()[k]), multJ=int(h['multJ'].ravel()[k]),
                   max_abs_drho=float(np.max(np.abs(d))), l2_drho=float(np.linalg.norm(d)),
                   rms_drho=float(np.linalg.norm(d) / np.sqrt(NE)),
                   sign_reversal=rev, on_bound=on, at_move_box=atm,
                   step_over_box_rms=float(np.sqrt(np.mean((d / box) ** 2))),
                   pred_gain=pred, real_gain=real, gain_ratio=real / pred if (np.isfinite(real) and pred != 0) else np.nan,
                   stop_metric=float(np.linalg.norm(d)), stop_eps=EPS, stop_raw=bool(np.linalg.norm(d) < EPS),
                   nInner=int(h['nInner'].ravel()[k]), innerConv=int(h['innerConv'].ravel()[k]),
                   spike=bool(k > 0 and om[k, 0] < 0.7 * om[k - 1, 0]))
        row.update({kk: v[k] for kk, v in extra.items()})
        rows.append(row)
        prevD = D
        rho_start = r
    return rows


def write(rows, name):
    with open(EVAL / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    # ---- target C480 --------------------------------------------------------
    f = h5py.File(C480, 'r')
    h = {k: np.asarray(f['hist'][k][()]).ravel() if k != 'omega' else np.asarray(f['hist'][k][()]) for k in f['hist'].keys()}
    om = h['omega'] if h['omega'].shape[1] == 5 else h['omega'].T
    RHO = np.asarray(f['RHO'][()]); DRHO = np.asarray(f['DRHO'][()])
    BOX = [float(m) for m in h['move']]
    extra = {'stage': h['stage'], 'exA': h['exA'], 'exB': h['exB'], 'exE': h['exE'], 'exDecl': h['exDecl']}
    rowsC = full_rows(RHO, DRHO, BOX, om, h, extra)
    write(rowsC, 'trajectory_C480.csv')
    np.save(EVAL / 'C480_final_rho.npy', RHO[-1])

    # ---- source M1 ------------------------------------------------------------
    g = h5py.File(M1 / '..' / 'M1_480x60_trajectory.mat' if False else EVAL / 'm1_run' / 'M1_480x60_trajectory.mat', 'r')
    hm = {k: np.asarray(g['hist'][k][()]).ravel() if k != 'omega' else np.asarray(g['hist'][k][()]) for k in g['hist'].keys()}
    omm = hm['omega'] if hm['omega'].shape[1] == 5 else hm['omega'].T
    RHOm = np.asarray(g['RHO'][()]); DRHOm = np.asarray(g['DRHO'][()]); DVEC = np.asarray(g['DVEC'][()])
    BOXm = [DVEC[k] for k in range(omm.shape[0])]
    rowsM = full_rows(RHOm, DRHOm, BOXm, omm, hm, {'stage': np.ones(omm.shape[0])})
    write(rowsM, 'trajectory_M1.csv')
    np.save(EVAL / 'M1_final_rho.npy', RHOm[-1])

    # ---- source S480 (retained hist/aux only) --------------------------------
    s = load_s480()
    hs = {k: np.asarray(v).ravel() if k != 'omega' else v for k, v in s['hist'].items()}
    oms = s['omega']
    n = oms.shape[0]
    lam = oms[:, :2] ** 2
    rowsS = []
    for k in range(n):
        pred = hs['beta'][k] - lam[k, 0]
        real = lam[k + 1, 0] - lam[k, 0] if k + 1 < n else np.nan
        rowsS.append(dict(outer=k + 1, omega1=oms[k, 0], omega2=oms[k, 1], omega3=oms[k, 2], lambda1=lam[k, 0],
                          lambda2=lam[k, 1], gap12=(oms[k, 1] - oms[k, 0]) / oms[k, 0], dOff=lam[k, 1] - lam[k, 0],
                          volume=hs['vol'][k], Mnd=s['Mnd'][k], beta=hs['beta'][k], box_max=s['move'][k],
                          box_mean=s['moveMean'][k], N=int(hs['N'][k]), multJ=int(hs['multJ'][k]),
                          max_abs_drho=hs['dxOuter'][k], l2_drho=hs['dxNorm2'][k],
                          rms_drho=hs['dxNorm2'][k] / np.sqrt(NE), pred_gain=pred, real_gain=real,
                          gain_ratio=real / pred if (np.isfinite(real) and pred != 0) else np.nan,
                          stop_metric=hs['dxNorm2'][k], stop_eps=EPS, stop_raw=bool(hs['dxNorm2'][k] < EPS),
                          nInner=int(hs['nInner'][k]), innerConv=int(hs['innerConv'][k]),
                          spike=bool(k > 0 and oms[k, 0] < 0.7 * oms[k - 1, 0])))
    write(rowsS, 'trajectory_S480.csv')
    np.save(EVAL / 'S480_final_rho.npy', s['rho'])
    for nm, rows in [('C480', rowsC), ('M1', rowsM), ('S480', rowsS)]:
        sp = [r['outer'] for r in rows if r['spike']]
        print(nm, 'n', len(rows), 'final Mnd', rows[-1]['Mnd'], 'spikes', len(sp), sp[:12])


if __name__ == '__main__':
    main()
