"""Parts 5, 6, 8 (per iteration): identical telemetry for control and treatment.

Definitions are AUDIT_PREREGISTRATION.md sections 7-8.  Read-only on the saved
trajectories; writes evaluations/traj_<which>.csv and traj_<which>_summary.json.

  python cs_trajectory_metrics.py control
  python cs_trajectory_metrics.py treatment
"""
import csv
import sys
from cs_common import *

RHOMIN = 1e-3


def classes(r):
    void = r <= .1
    solid = r >= .9
    core = (r >= .4) & (r <= .6)
    shell = (r > .1) & (r < .9) & ~core
    return {'void': void, 'shell': shell, 'core': core, 'solid': solid}


def compute(which):
    if which == 'control':
        f = h5py.File(CONTROL_TRAJ, 'r')
        rec = json.loads((CANARY / 'runs' / 'C480x60_three_rung_record.json').read_text())
        omega_final = np.array(rec['omega'])
    else:
        f = h5py.File(TREAT_TRAJ, 'r')
        omega_final = f['omegaFinal'][()].squeeze()
    RHO = f['RHO'][()]            # (n, NE)
    DRHO = f['DRHO'][()]
    h = f['hist']
    g = lambda k: h[k][()].squeeze()
    omega = h['omega'][()]        # (n, 5)
    if omega.shape[0] != RHO.shape[0]:
        omega = omega.T
    n, NE = RHO.shape
    move, stage, beta, nInner = g('move'), g('stage'), g('beta'), g('nInner')
    exA, exB, exNA, exNB, exAmp, exCos, exNet, exMedcos, exMednet, exDecl = (g(k) for k in
        ['exA', 'exB', 'exNA', 'exNB', 'exAmp', 'exCos', 'exNet', 'exMedcos', 'exMednet', 'exDecl'])
    tEig, tGrad, tInner, tOuter = g('tEig'), g('tGrad'), g('tInner'), g('tOuter')[:n]
    vol, multJ = g('vol'), g('multJ')
    f.close()

    lam1 = omega[:, 0] ** 2
    lam1_next = np.r_[lam1[1:], omega_final[0] ** 2]
    act = lam1_next - lam1
    pred = beta - lam1
    rows = []
    rho_prev = np.full(NE, 0.5)
    rho_prev2 = None
    d_prev = None
    gray_rows = per_iteration_gray(RHO)
    for k in range(n):
        d = DRHO[k]
        r0 = rho_prev
        mv = move[k]
        lo = np.maximum(RHOMIN - r0, -mv); hi = np.minimum(1 - r0, mv); w = hi - lo
        atLo = d <= lo + 1e-6 * w; atHi = d >= hi - 1e-6 * w
        loMove = (-mv) > (RHOMIN - r0); hiMove = mv < (1 - r0)
        cl = classes(r0)
        gray = (r0 > .1) & (r0 < .9)
        row = {'outer': k + 1, 'move': mv, 'stage': stage[k], 'omega1': omega[k, 0], 'omega2': omega[k, 1],
               'omega3': omega[k, 2], 'lam1': lam1[k], 'beta': beta[k], 'pred': pred[k], 'act': act[k],
               'nInner': nInner[k], 'vol': vol[k], 'multJ': multJ[k],
               'frac_lower_density': np.mean(atLo & ~loMove), 'frac_lower_move': np.mean(atLo & loMove),
               'frac_upper_move': np.mean(atHi & hiMove), 'frac_upper_density': np.mean(atHi & ~hiMove),
               'frac_interior': np.mean(~atLo & ~atHi), 'frac_any_bound': np.mean(atLo | atHi),
               'frac_pm_move': np.mean((atLo & loMove) | (atHi & hiMove)),
               'frac_density_limited': np.mean((atLo & ~loMove) | (atHi & ~hiMove)),
               'gray_n': int(gray.sum()),
               'gray_full_move_frac': float(np.mean(((atLo & loMove) | (atHi & hiMove))[gray])) if gray.any() else np.nan,
               'max_abs_drho': np.max(np.abs(d)), 'norm2_drho': np.linalg.norm(d),
               'Mnd_percent': gray_rows[k, 0], 'gray_fraction': gray_rows[k, 1], 'mid_fraction': gray_rows[k, 2],
               'broad_core_fraction': gray_rows[k, 3],
               'exA': exA[k], 'exB': exB[k], 'exNA': exNA[k], 'exNB': exNB[k], 'exAmp': exAmp[k], 'exCos': exCos[k],
               'exNet': exNet[k], 'exMedcos': exMedcos[k], 'exMednet': exMednet[k], 'exDecl': exDecl[k],
               'tEig': tEig[k], 'tGrad': tGrad[k], 'tInner': tInner[k], 'tOuter': tOuter[k]}
        for c, m in cl.items():
            row[f'{c}_n'] = int(m.sum())
            row[f'{c}_pm_move_frac'] = float(np.mean(((atLo & loMove) | (atHi & hiMove))[m])) if m.any() else np.nan
        if d_prev is not None:
            nd, np_ = np.linalg.norm(d), np.linalg.norm(d_prev)
            row['cos_prev'] = float(d @ d_prev / (nd * np_)) if nd > 0 and np_ > 0 else np.nan
            row['sum_norm'] = float(np.linalg.norm(d + d_prev))
            row['sum_norm_ratio'] = row['sum_norm'] / (nd + np_) if nd + np_ > 0 else np.nan
            both = (np.abs(d) > 1e-9) & (np.abs(d_prev) > 1e-9)
            rev = both & (d * d_prev < 0)
            row['sign_reversal_frac'] = float(rev.sum() / both.sum()) if both.any() else np.nan
            for c, m in cl.items():
                bm = both & m
                row[f'{c}_sign_reversal_frac'] = float((rev & m).sum() / bm.sum()) if bm.any() else np.nan
        rho_now = RHO[k]
        if rho_prev2 is not None:
            den = np.linalg.norm(rho_now - rho_prev)
            row['recurrence'] = float(np.linalg.norm(rho_now - rho_prev2) / den) if den > 0 else np.nan
        rows.append(row)
        rho_prev2 = rho_prev
        rho_prev = rho_now
        d_prev = d
    keys = []
    for r_ in rows:
        for k_ in r_:
            if k_ not in keys:
                keys.append(k_)
    df = {k_: np.array([r_.get(k_, np.nan) for r_ in rows], dtype=float) for k_ in keys}
    df['eligible'] = df['pred'] > 1e-7 * df['lam1']
    el = df['eligible']
    df['r'] = np.where(el, df['act'] / np.where(el, df['pred'], 1), np.nan)
    df['model_err_abs'] = df['act'] - df['pred']
    df['model_err_rel'] = np.where(el, (df['act'] - df['pred']) / np.where(el, df['pred'], 1), np.nan)
    df['sign_agree'] = np.where(el, (df['act'] > 0).astype(float), np.nan)
    df['cum_pred'] = np.where(el, df['pred'], 0).cumsum()
    df['cum_act_eligible'] = np.where(el, df['act'], 0).cumsum()
    df['cum_act_all'] = df['act'].cumsum()
    write_csv(EV / f'traj_{which}.csv', df)
    return df, omega_final


def write_csv(path, df):
    keys = list(df)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(len(df[keys[0]])):
            w.writerow([repr(float(df[k][i])) for k in keys])


def read_csv(path):
    with open(path) as fh:
        rd = csv.reader(fh)
        keys = next(rd)
        vals = np.array([[float(x) for x in row] for row in rd])
    return {k: vals[:, i] for i, k in enumerate(keys)}


def _med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def _mean(b, mask=None):
    if mask is not None:
        b = b[mask]
    return float(np.mean(b)) if len(b) else None


def realization_summary(df):
    el = df['eligible'].astype(bool)
    r, act, pred = df['r'][el], df['act'][el], df['pred'][el]
    out = {'n_steps': int(len(el)), 'n_eligible': int(el.sum())}
    if el.any():
        out.update({'median_r': float(np.median(r)), 'q10_r': float(np.quantile(r, .1)),
                    'q25_r': float(np.quantile(r, .25)), 'q75_r': float(np.quantile(r, .75)),
                    'q90_r': float(np.quantile(r, .9)), 'frac_act_negative': float(np.mean(act < 0)),
                    'frac_r_lt_0p25': float(np.mean(r < .25)), 'frac_r_gt_4': float(np.mean(r > 4)),
                    'cum_ratio': float(act.sum() / pred.sum()), 'sum_pred': float(pred.sum()),
                    'sum_act': float(act.sum()),
                    'median_abs_rel_err': float(np.median(np.abs(df['model_err_rel'][el])))})
    cp = df['cos_prev']
    out['frac_cos_lt_m0p5'] = float(np.mean(cp[np.isfinite(cp)] < -.5))
    out['frac_lam1_decrease'] = float(np.mean(df['act'] < 0))
    sg = np.sign(df['act'])
    out['frac_act_sign_alternation'] = float(np.mean(sg[1:] * sg[:-1] < 0)) if len(sg) > 1 else None
    stages = {}
    for st in np.unique(df['stage']):
        m = df['stage'] == st
        me = m & el
        g = lambda k: df[k][m]
        cpg = g('cos_prev')
        stages[str(int(st))] = {'n': int(m.sum()), 'n_eligible': int(me.sum()), 'move': float(g('move')[0]),
            'first': int(g('outer')[0]), 'last': int(g('outer')[-1]),
            'median_r': float(np.median(df['r'][me])) if me.any() else None,
            'frac_act_negative': float(np.mean(df['act'][me] < 0)) if me.any() else None,
            'cum_ratio': float(df['act'][me].sum() / df['pred'][me].sum()) if me.any() else None,
            'frac_cos_lt_m0p5': float(np.mean(cpg[np.isfinite(cpg)] < -.5)) if np.isfinite(cpg).any() else 0.0,
            'median_cos_prev': _med(cpg), 'median_sign_reversal_frac': _med(g('sign_reversal_frac')),
            'median_recurrence': _med(g('recurrence')), 'median_frac_any_bound': _med(g('frac_any_bound')),
            'median_frac_pm_move': _med(g('frac_pm_move')), 'median_frac_interior': _med(g('frac_interior')),
            'median_gray_full_move_frac': _med(g('gray_full_move_frac')),
            'median_abs_rel_err': float(np.median(np.abs(df['model_err_rel'][me]))) if me.any() else None,
            'frac_lam1_decrease': float(np.mean(g('act') < 0))}
        idx = np.flatnonzero(m)[-20:]
        l20 = np.zeros_like(m); l20[idx] = True
        stages[str(int(st))]['last20'] = {'median_cos_prev': _med(df['cos_prev'][l20]),
            'median_sign_reversal_frac': _med(df['sign_reversal_frac'][l20]),
            'median_recurrence': _med(df['recurrence'][l20]),
            'frac_act_negative': float(np.mean(df['act'][l20] < 0)),
            'median_r': _med(df['r'][l20 & el])}
    out['stages'] = stages
    return out


def realization_verdict(s):
    """AUDIT_PREREGISTRATION.md section 7, mechanical."""
    if s['n_eligible'] < 20:
        return 'OUTER_MODEL_REALIZATION_INCONCLUSIVE', ['fewer than 20 eligible steps']
    reasons = []
    if s['median_r'] < .25: reasons.append('median r < 0.25')
    if s['frac_act_negative'] > .30: reasons.append('fraction act<0 > 0.30')
    if s['cum_ratio'] < .25: reasons.append('sum act / sum pred < 0.25')
    for st, v in s['stages'].items():
        if v['n_eligible'] >= 10 and (v['median_r'] < 0 or v['frac_act_negative'] > .5):
            reasons.append(f'stage {st}: median r < 0 or fraction act<0 > 0.5')
        if v['frac_cos_lt_m0p5'] > .5:
            reasons.append(f'stage {st}: fraction cos < -0.5 exceeds 0.5')
    if reasons:
        return 'OUTER_GLOBALIZATION_PROBLEM_EXPOSED', reasons
    healthy = (.5 <= s['median_r'] <= 2.0 and s['frac_act_negative'] <= .10 and s['cum_ratio'] >= .5 and
               all((v['n_eligible'] < 10 or (v['median_r'] >= .5 and v['frac_act_negative'] <= .25)) and
                   v['frac_cos_lt_m0p5'] <= .25 for v in s['stages'].values()))
    if healthy:
        return 'OUTER_MODEL_REALIZATION_HEALTHY', []
    fails = []
    if not (.5 <= s['median_r'] <= 2.0): fails.append('median r outside [0.5, 2]')
    if s['frac_act_negative'] > .10: fails.append('fraction act<0 > 0.10')
    if s['cum_ratio'] < .5: fails.append('cum ratio < 0.5')
    for st, v in s['stages'].items():
        if v['n_eligible'] >= 10 and (v['median_r'] < .5 or v['frac_act_negative'] > .25):
            fails.append(f'stage {st}: median r < 0.5 or fraction act<0 > 0.25')
        if v['frac_cos_lt_m0p5'] > .25:
            fails.append(f'stage {st}: fraction cos < -0.5 > 0.25')
    return 'OUTER_MODEL_REALIZATION_MARGINAL', fails


def main(which):
    df, omega_final = compute(which)
    s = realization_summary(df)
    v, why = realization_verdict(s)
    s['verdict_if_treatment_rule_applied'] = v
    s['verdict_reasons'] = why
    s['omega_final'] = omega_final.tolist()
    dump(EV / f'traj_{which}_summary.json', s)
    print(which, json.dumps({k: s[k] for k in s if k != 'stages'}, indent=1))
    for st, x in s['stages'].items():
        print(' stage', st, {k: x[k] for k in ['n', 'n_eligible', 'median_r', 'frac_act_negative', 'frac_cos_lt_m0p5',
                                               'median_frac_pm_move', 'median_gray_full_move_frac']})


if __name__ == '__main__':
    main(sys.argv[1])
