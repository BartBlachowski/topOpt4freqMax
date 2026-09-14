"""Part 10: physical and filtered stationarity at an endpoint (no optimization).

Mirrors gray_kkt_forensic_audit/scripts/stationarity.py main() for one mesh,
minus the finite-difference block, with the verbatim kkt()/stats() in cs_common.
The control run must reproduce that audit's retained 480 values exactly.

  python cs_stationarity.py control|treatment
"""
import sys
import scipy.io as sio
from cs_common import *


def stationarity(which):
    d = sio.loadmat(EV / f'spectral_{which}.mat', squeeze_me=True)
    r = d['rho']; ne = len(r); lam = d['lam'][0]; g = d['gRaw']; gf = d['gFiltered']; gk = d['gK']; gm = d['gM']
    _, fl = geometry_metrics(r)
    dep = fl['depth'].T.ravel()
    free = (r > .0010001) & (r < .9999999); gray = (r > .1) & (r < .9)
    classes = {'gray': gray, 'mid': (r >= .4) & (r <= .6), 'solid': r > .9, 'void': r < .1, 'broad': dep > .06}
    scale = np.sqrt(np.mean((g[free] / lam) ** 2)); scaleF = np.sqrt(np.mean((gf[free] / lam) ** 2))
    C = np.abs(g) / (np.abs(gk) + np.abs(gm) + np.finfo(float).eps)
    redraw, vr, kr = kkt(g, r, lam, free, scale); redf, vf, kf = kkt(gf, r, lam, free, scale)
    rg, vg, kg = kkt(g, r, lam, gray, scale); fg, vfg, kfg = kkt(gf, r, lam, gray, scale)
    met = {'omega': d['omega'].tolist(), 'lambda': d['lam'].tolist(),
           'gap12': float((d['omega'][1] - d['omega'][0]) / d['omega'][0]),
           'gap23': float((d['omega'][2] - d['omega'][1]) / d['omega'][1]),
           'mass_orthogonality_Fro': float(d['massOrth']), 'eig_residual': d['eigResidual'].tolist(),
           'raw_KKT': kr, 'filtered_subproblem': kf, 'gray_fit_raw': kg, 'gray_fit_filtered': kfg,
           'filtered_own_scale': float(scaleF), 'free_count': int(free.sum()), 'classes': {},
           'bound_tolerance_robustness': []}
    for tol in [1e-7, 1e-5, 1e-4, 1e-3]:
        fi = (r > .001 + tol) & (r < 1 - tol)
        if fi.sum() < 2:
            met['bound_tolerance_robustness'].append({'tol': tol, 'free_count': int(fi.sum())}); continue
        rr, _, kk = kkt(g, r, lam, fi, scale); ff, _, fk = kkt(gf, r, lam, fi, scale)
        met['bound_tolerance_robustness'].append({'tol': tol, 'free_count': int(fi.sum()), 'raw_mu': kk['mu_volume'],
            'filtered_mu': fk['mu_volume'], 'raw_gray_RMS': stats(rr[gray] / scale)['RMS'],
            'filtered_gray_RMS': stats(ff[gray] / scale)['RMS']})
    for cls, mask in classes.items():
        if not mask.any():
            met['classes'][cls] = {'n': 0}; continue
        gg = g[mask]; ff = gf[mask]; rmsraw = float(np.sqrt(np.mean(gg ** 2))); rmsf = float(np.sqrt(np.mean(ff ** 2)))
        met['classes'][cls] = {'n': int(mask.sum()), 'gK_abs': stats(gk[mask]), 'gM_abs': stats(gm[mask]),
            'gRaw_abs': stats(gg), 'gFiltered_abs': stats(ff), 'cancellation': stats(C[mask]),
            'cancellation_fraction_lt_01': float((C[mask] < .1).mean()),
            'gK_over_abs_gM_quantiles': np.quantile(gk[mask] / np.maximum(abs(gm[mask]), 1e-30), [.1, .5, .9]).tolist(),
            'raw_negative_fraction': float((gg < 0).mean()), 'filter_sign_flip_fraction': float((gg * ff < 0).mean()),
            'filter_RMS_ratio': rmsf / rmsraw, 'filter_std_ratio': float(np.std(ff) / np.std(gg)),
            'raw_reduced_normalized': stats(redraw[mask] / scale),
            'filtered_reduced_common_scale': stats(redf[mask] / scale),
            'filtered_reduced_own_scale': stats(redf[mask] / scaleF),
            'raw_grayfit_normalized': stats(rg[mask] / scale),
            'filtered_grayfit_common_scale': stats(fg[mask] / scale),
            'filtered_grayfit_gray_own_scale': stats(fg[mask] / (np.sqrt(np.mean((gf[gray] / lam) ** 2)))),
            'raw_grayfit_fraction_abs_lt_01': float((abs(rg[mask] / scale) < .1).mean())}
    np.savez_compressed(EV / f'kkt_{which}.npz', rho=r, raw_reduced=redraw, filtered_reduced=redf, raw_grayfit=rg,
                        filtered_grayfit=fg, raw_scale=scale, violation_raw=vr, violation_filtered=vf,
                        raw_grayfit_violation=vg, filtered_grayfit_violation=vfg)
    return met


def main(which):
    met = stationarity(which)
    if which == 'control':
        ref = json.loads((GRAYKKT / 'evaluations' / 'stationarity.json').read_text())['480']
        chk = {}
        for k in ['raw_KKT', 'filtered_subproblem', 'gray_fit_raw', 'gray_fit_filtered', 'classes', 'gap12']:
            chk[k] = json.loads(json.dumps(met[k], default=float)) == ref[k]
        met['reproduction_of_gray_kkt_audit_480'] = chk
        met['reproduction_exact'] = all(chk.values())
        print('reproduction', chk)
    dump(EV / f'stationarity_{which}.json', met)
    c = met['classes']
    print(which, {cl: (c[cl].get('raw_reduced_normalized', {}).get('RMS'), c[cl].get('raw_grayfit_normalized', {}).get('RMS'),
                       c[cl].get('filtered_grayfit_common_scale', {}).get('RMS')) for cl in c})
    print(' global projected raw RMS', met['raw_KKT']['global_projected_residual_normalized']['RMS'],
          'filtered', met['filtered_subproblem']['global_projected_residual_normalized']['RMS'],
          'bounds', met['raw_KKT']['lower_count'], met['raw_KKT']['upper_count'])


if __name__ == '__main__':
    main(sys.argv[1])
