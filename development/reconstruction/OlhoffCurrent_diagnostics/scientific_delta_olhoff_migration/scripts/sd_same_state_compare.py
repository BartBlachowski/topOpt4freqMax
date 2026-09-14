#!/usr/bin/env python3
"""Part 15: compare same-state evaluations.  T vs S1 = implementation identity;
S1 vs S0 = formulation operator difference.  Also validates the offline evaluator
against in-run records (C480 DRHO/omega, M1 DRHO, S480 hist)."""
import numpy as np
import scipy.io as sio
import h5py
from sd_common import *

SS = EVAL / 'same_state'
TOL = 1e-9


def load(state, ev):
    p = SS / f'{state}__{ev}.mat'
    if not p.exists():
        return None
    return sio.loadmat(p, squeeze_me=True, struct_as_record=False)['R']


def rel(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    den = np.maximum(np.abs(b), 1e-300)
    d = np.abs(a - b)
    # relative where the reference is non-negligible, absolute-over-scale otherwise
    scale = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.where(np.abs(b) > 1e-12 * scale, d / den, d / scale)))


def inner_list(R):
    inn = R.inner
    return list(inn) if isinstance(inn, np.ndarray) else [inn]


def l2rel(a, b, mask=None):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if mask is not None:
        a = a[mask]; b = b[mask]
    if not len(b):
        return None
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else None


def compare(A, B, rho=None):
    out = {}
    if rho is not None:
        lo = rho < 0.1; hi = ~lo
        out['frac_rho_lt_0p1'] = float(lo.mean())
        for nm, a, b in [('raw_f11', A.Fraw[:, 0, 0], B.Fraw[:, 0, 0]), ('filt_f11', A.Ffilt[:, 0, 0], B.Ffilt[:, 0, 0]),
                         ('raw_fJJ', A.fJJraw, B.fJJraw), ('filt_fJJ', A.fJJfilt, B.fJJfilt)]:
            out[nm + '_L2rel_all'] = l2rel(a, b)
            out[nm + '_L2rel_rho_ge_0p1'] = l2rel(a, b, hi)
            out[nm + '_L2rel_rho_lt_0p1'] = l2rel(a, b, lo)
            out[nm + '_corr_all'] = float(np.corrcoef(a, b)[0, 1])
    out['K_bitwise'] = A.K_sha256 == B.K_sha256
    out['M_bitwise'] = A.M_sha256 == B.M_sha256
    out['K_fro_rel'] = abs(A.K_fro - B.K_fro) / B.K_fro
    out['M_fro_rel'] = abs(A.M_fro - B.M_fro) / B.M_fro
    out['omega_rel'] = rel(A.omega, B.omega)
    out['lambda1_rel'] = abs(A.lam[0] - B.lam[0]) / B.lam[0]
    out['lambda2_rel'] = abs(A.lam[1] - B.lam[1]) / B.lam[1]
    out['N_equal'] = int(A.N) == int(B.N)
    out['multJ_equal'] = bool(A.multJ) == bool(B.multJ)
    out['dOff_rel'] = rel(A.dOff, B.dOff)
    # eigenvector sign alignment for the off-diagonal f_12
    sgn = np.sign(np.dot(A.Phi[:, 0], B.Phi[:, 0])) * np.sign(np.dot(A.Phi[:, 1], B.Phi[:, 1]))
    Fa = np.array(A.Fraw, dtype=float); Fb = np.array(B.Fraw, dtype=float)
    Fa[:, 0, 1] *= sgn; Fa[:, 1, 0] *= sgn
    Ga = np.array(A.Ffilt, dtype=float); Gb = np.array(B.Ffilt, dtype=float)
    Ga[:, 0, 1] *= sgn; Ga[:, 1, 0] *= sgn
    out['raw_f11_rel'] = rel(Fa[:, 0, 0], Fb[:, 0, 0]); out['raw_f22_rel'] = rel(Fa[:, 1, 1], Fb[:, 1, 1])
    out['raw_f12_rel'] = rel(Fa[:, 0, 1], Fb[:, 0, 1]); out['raw_fJJ_rel'] = rel(A.fJJraw, B.fJJraw)
    out['filt_f11_rel'] = rel(Ga[:, 0, 0], Gb[:, 0, 0]); out['filt_f22_rel'] = rel(Ga[:, 1, 1], Gb[:, 1, 1])
    out['filt_f12_rel'] = rel(Ga[:, 0, 1], Gb[:, 0, 1]); out['filt_fJJ_rel'] = rel(A.fJJfilt, B.fJJfilt)
    out['raw_bitwise'] = bool(np.array_equal(A.Fraw, B.Fraw) and np.array_equal(A.fJJraw, B.fJJraw))
    out['filt_bitwise'] = bool(np.array_equal(A.Ffilt, B.Ffilt) and np.array_equal(A.fJJfilt, B.fJJfilt))
    out['raw_f11_corr'] = float(np.corrcoef(Fa[:, 0, 0], Fb[:, 0, 0])[0, 1])
    out['filt_f11_corr'] = float(np.corrcoef(Ga[:, 0, 0], Gb[:, 0, 0])[0, 1])
    out['rows_fval_rel'] = rel(A.rows_fval, B.rows_fval)
    out['rows_dfdx_rel'] = rel(A.rows_dfdx, B.rows_dfdx)
    out['rows_bitwise'] = bool(np.array_equal(A.rows_fval, B.rows_fval) and np.array_equal(A.rows_dfdx, B.rows_dfdx))
    ia = inner_list(A)[0]; ib = inner_list(B)[0]           # common box 0.04 is always first
    assert np.all(np.asarray(ia.box) == 0.04) and np.all(np.asarray(ib.box) == 0.04)
    out['inner004_drho_maxabs'] = float(np.max(np.abs(ia.drho - ib.drho)))
    out['inner004_drho_rel_to_box'] = out['inner004_drho_maxabs'] / 0.04
    out['inner004_bitwise'] = bool(np.array_equal(ia.drho, ib.drho))
    out['inner004_nInner'] = [int(ia.nInner), int(ib.nInner)]
    out['inner004_beta_rel'] = abs(ia.beta - ib.beta) / abs(ib.beta)
    out['inner004_cos'] = float(np.dot(ia.drho, ib.drho) / (np.linalg.norm(ia.drho) * np.linalg.norm(ib.drho)))
    out['agree_all_1e-9'] = bool(max(out['K_fro_rel'], out['M_fro_rel'], out['omega_rel'], out['dOff_rel'],
                                     out['raw_f11_rel'], out['raw_f22_rel'], out['raw_f12_rel'], out['raw_fJJ_rel'],
                                     out['filt_f11_rel'], out['filt_f22_rel'], out['filt_f12_rel'], out['filt_fJJ_rel'],
                                     out['rows_fval_rel'], out['rows_dfdx_rel'], out['inner004_drho_rel_to_box']) <= TOL
                                 and out['N_equal'] and out['multJ_equal'])
    return out


def state_rho(st):
    if st == 'rho0':
        return np.full(NE, 0.5)
    if st.startswith('C480_k'):
        with h5py.File(C480, 'r') as f:
            return np.asarray(f['RHO'][int(st[-3:]) - 1])
    if st == 'S480_final':
        return load_s480()['rho']
    with h5py.File(EVAL / 'm1_run' / 'M1_480x60_trajectory.mat', 'r') as g:
        return np.asarray(g['RHO'][int(st[-3:]) - 1])


def main():
    states = ['rho0', 'C480_k010', 'C480_k020', 'C480_k100', 'C480_k386', 'S480_final', 'M1_k005', 'M1_k011', 'M1_k064']
    res = {'impl_identity_T_vs_S1': {}, 'formulation_S1_vs_S0': {}, 'validation': {}, 'omega': {}}
    for st in states:
        T, S1, S0 = load(st, 'T'), load(st, 'S1'), load(st, 'S0')
        rho = state_rho(st)
        res['impl_identity_T_vs_S1'][st] = compare(S1, T, rho)
        res['formulation_S1_vs_S0'][st] = compare(S0, S1, rho)
        res['omega'][st] = {'T': T.omega[:5], 'S1': S1.omega[:5], 'S0': S0.omega[:5]}
        print(st, 'T-S1 agree', res['impl_identity_T_vs_S1'][st]['agree_all_1e-9'],
              'K', res['impl_identity_T_vs_S1'][st]['K_bitwise'], 'raw', res['impl_identity_T_vs_S1'][st]['raw_bitwise'],
              'filt', res['impl_identity_T_vs_S1'][st]['filt_bitwise'], 'inner', res['impl_identity_T_vs_S1'][st]['inner004_bitwise'],
              '| S0-S1 lo %.4f omega_rel %.3e raw_f11 L2 %.3e (hi %.3e) filt_f11 L2 %.3e corr %.4f inner cos %.4f' % (
                  res['formulation_S1_vs_S0'][st]['frac_rho_lt_0p1'], res['formulation_S1_vs_S0'][st]['omega_rel'],
                  res['formulation_S1_vs_S0'][st]['raw_f11_L2rel_all'], res['formulation_S1_vs_S0'][st]['raw_f11_L2rel_rho_ge_0p1'] or 0,
                  res['formulation_S1_vs_S0'][st]['filt_f11_L2rel_all'], res['formulation_S1_vs_S0'][st]['filt_f11_corr_all'],
                  res['formulation_S1_vs_S0'][st]['inner004_cos']))
    # ---- validation against in-run records -----------------------------------
    f = h5py.File(C480, 'r'); DR = f['DRHO']; om = np.asarray(f['hist']['omega'][()]); om = om if om.shape[1] == 5 else om.T
    g = h5py.File(EVAL / 'm1_run' / 'M1_480x60_trajectory.mat', 'r'); DM = g['DRHO']
    s = load_s480()
    V = {}
    T0 = load('rho0', 'T'); S10 = load('rho0', 'S1'); S00 = load('rho0', 'S0')
    V['T_rho0_drho_vs_C480_iter1_bitwise'] = bool(np.array_equal(inner_list(T0)[0].drho, DR[0]))
    V['T_rho0_omega_vs_C480_iter1_bitwise'] = bool(np.array_equal(T0.omega[:5], om[0]))
    V['S1_rho0_box010_drho_vs_M1_iter1_bitwise'] = bool(np.array_equal(inner_list(S10)[1].drho, DM[0]))
    # (a numpy norm is not MATLAB's dnrm2; compare the vectors instead: S0 == S1 at rho0, S1 == M1 iter 1,
    #  and M1 hist(1) == S480 hist(1) bitwise by the P-prefix test)
    V['S0_rho0_box010_drho_equals_S1_bitwise'] = bool(np.array_equal(inner_list(S00)[1].drho, inner_list(S10)[1].drho))
    V['S0_rho0_omega_vs_S480_hist1_bitwise'] = bool(np.array_equal(S00.omega[:5], s['omega'][0]))
    for k in [10, 20, 100]:
        R = load(f'C480_k{k:03d}', 'T')
        V[f'T_C480_k{k:03d}_drho_vs_DRHO{k+1}_bitwise'] = bool(np.array_equal(inner_list(R)[0].drho, DR[k]))
        V[f'T_C480_k{k:03d}_omega_vs_hist{k+1}_bitwise'] = bool(np.array_equal(R.omega[:5], om[k]))
    for k in [5, 11]:
        R = load(f'M1_k{k:03d}', 'S1')
        V[f'S1_M1_k{k:03d}_nativebox_drho_vs_M1_DRHO{k+1}_bitwise'] = bool(np.array_equal(inner_list(R)[1].drho, DM[k]))
    res['validation'] = V
    for k, v in V.items():
        print(k, v)
    # ---- first step at rho0 under native boxes (D7) ---------------------------
    a = inner_list(S10)[1]; b = inner_list(T0)[0]
    res['rho0_native_first_step'] = dict(
        source_box=0.10, target_box=0.04,
        source_l2=float(np.linalg.norm(a.drho)), target_l2=float(np.linalg.norm(b.drho)),
        source_max=float(np.max(np.abs(a.drho))), target_max=float(np.max(np.abs(b.drho))),
        cos=float(np.dot(a.drho, b.drho) / (np.linalg.norm(a.drho) * np.linalg.norm(b.drho))),
        l2_ratio=float(np.linalg.norm(a.drho) / np.linalg.norm(b.drho)),
        source_pred_gain=float(a.beta - S10.lam[0]), target_pred_gain=float(b.beta - T0.lam[0]),
        source_nInner=int(a.nInner), target_nInner=int(b.nInner),
        source_common004_equals_target=bool(np.array_equal(inner_list(S10)[0].drho, b.drho)))
    print(res['rho0_native_first_step'])
    jdump(res, EVAL / 'same_state_comparison.json')


if __name__ == '__main__':
    main()
