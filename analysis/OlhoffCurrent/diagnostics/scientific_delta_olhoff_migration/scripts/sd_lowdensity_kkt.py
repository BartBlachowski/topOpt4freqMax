#!/usr/bin/env python3
"""Part 16 (endpoint stationarity, frozen gray-fit definition) and Part 18 (low-density
modal participation) from the same-state evaluator outputs.  Zero rho updates."""
import numpy as np
from sd_same_state_compare import load, state_rho
from sd_common import *


def modal(R, rho):
    Ek = np.asarray(R.Ekin, dtype=float); Es = np.asarray(R.Estr, dtype=float)
    out = []
    for j in range(Ek.shape[1]):
        ek = Ek[:, j]; es = Es[:, j]
        tk, ts = ek.sum(), es.sum()
        out.append(dict(mode=j + 1, omega=float(R.omega[j]),
                        kin_frac_rho_lt_0p1=float(ek[rho < .1].sum() / tk), kin_frac_rho_lt_0p3=float(ek[rho < .3].sum() / tk),
                        str_frac_rho_lt_0p1=float(es[rho < .1].sum() / ts), str_frac_rho_lt_0p3=float(es[rho < .3].sum() / ts),
                        localized=bool(ek[rho < .3].sum() / tk >= .5),
                        kin_participation_ratio=float(tk ** 2 / (NE * np.sum(ek ** 2)))))
    return out


def main():
    res = {'kkt_validation': {}, 'kkt': {}, 'modes': {}, 'element_factors': {}}
    # ---- validation of the frozen KKT definition at the C480 endpoint --------
    rC = state_rho('C480_k386'); T = load('C480_k386', 'T')
    rms, scale, mu = kkt_grayfit(T.Fraw[:, 0, 0], rC, T.lam[0])
    res['kkt_validation'] = dict(C480_gray_fit_raw_RMS=rms, prior_value=0.334039, abs_diff=abs(rms - 0.334039),
                                 pass_tol_1e4=bool(abs(rms - 0.334039) <= 1e-4))
    rmsf, _, _ = kkt_grayfit(T.Ffilt[:, 0, 0], rC, T.lam[0])
    print(res['kkt_validation'], 'filtered (own gray-fit, raw-scale not applied)', rmsf)
    # ---- endpoint stationarity under native and common formulations ----------
    for st, evs in [('C480_k386', ['T', 'S0']), ('S480_final', ['S0', 'S1']), ('M1_k064', ['S1', 'S0'])]:
        r = state_rho(st)
        for ev in evs:
            R = load(st, ev)
            raw, sc, mu = kkt_grayfit(R.Fraw[:, 0, 0], r, R.lam[0])
            # filtered residual on the raw scale, as the prior 'filtered gray RMS, best gray dual' column
            v = np.asarray(R.Ffilt[:, 0, 0], dtype=float); lam = R.lam[0]
            gray = (r > .1) & (r < .9); Vtot = .5 * NE
            muf = max(0.0, float(np.mean(v[gray] / lam)) * Vtot)
            redf = -v / lam + muf / Vtot
            filt = float(np.sqrt(np.mean((redf[gray] / sc) ** 2)))
            gap = float((R.omega[1] - R.omega[0]) / R.omega[0])
            res['kkt'][f'{st}__{ev}'] = dict(gray_fit_raw_RMS=raw, gray_fit_filtered_RMS_rawscale=filt, scale=sc, mu=mu,
                                             omega1=float(R.omega[0]), gap12=gap, n_gray=int(gray.sum()),
                                             simple_eigenvalue_valid=bool(gap > 0.05))
            print(st, ev, res['kkt'][f'{st}__{ev}'])
    # ---- low-density modal participation -----------------------------------
    for st in ['C480_k386', 'S480_final', 'M1_k064', 'C480_k100', 'M1_k011']:
        r = state_rho(st)
        res['modes'][st] = {}
        for ev in ['T', 'S1', 'S0']:
            R = load(st, ev)
            res['modes'][st][ev] = modal(R, r)
        print(st, 'S1 modes', [(m['omega'], round(m['kin_frac_rho_lt_0p3'], 3), m['localized']) for m in res['modes'][st]['S1'][:3]],
              '| S0', [(m['omega'], round(m['kin_frac_rho_lt_0p3'], 3), m['localized']) for m in res['modes'][st]['S0'][:3]])
    # ---- element factors in the low-density band -----------------------------
    rho = np.array([1e-3, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    simpK = rho ** 3
    pedK = np.where(rho < .1, rho * .1 ** 2, rho ** 3)
    c1, c2 = 6e5, -5e6
    m4b = np.where(rho <= .1, c1 * rho ** 6 + c2 * rho ** 7, rho)
    m2 = rho
    res['element_factors'] = dict(rho=rho, K_simp=simpK, K_pedersen=pedK, M_eq4b=m4b, M_eq2=m2,
                                  MK_ratio_simp_eq4b=m4b / simpK, MK_ratio_pedersen_eq2=m2 / pedK,
                                  dK_simp=3 * rho ** 2, dK_pedersen=np.where(rho < .1, .01, 3 * rho ** 2),
                                  dM_eq4b=np.where(rho <= .1, 6 * c1 * rho ** 5 + 7 * c2 * rho ** 6, 1.0), dM_eq2=np.ones_like(rho))
    for k in ['MK_ratio_simp_eq4b', 'MK_ratio_pedersen_eq2']:
        print(k, np.round(res['element_factors'][k], 6))
    jdump(res, EVAL / 'lowdensity_kkt.json')


if __name__ == '__main__':
    main()
