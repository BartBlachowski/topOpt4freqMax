#!/usr/bin/env python3
"""Preregistered trajectory analyses (sec. 10, 11, 12): decomposition, milestones,
grayness divergence onset, stopping and controller statistics, inner attenuation."""
import csv
import numpy as np
from sd_common import *


def read(name):
    with open(EVAL / name) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0].keys():
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(x) if x not in ('', 'nan', 'NaN') else np.nan for x in col])
        except ValueError:
            out[k] = np.array([1.0 if x == 'True' else 0.0 if x == 'False' else np.nan for x in col]) if set(col) <= {'True', 'False'} else np.array(col)
    return out


def onset(a, b, thr=0.03, persist=10):
    n = min(len(a), len(b))
    d = np.abs(a[:n] - b[:n]) >= thr
    for k in range(n - persist + 1):
        if d[k:k + persist].all():
            return k + 1
    return None


def first_cross(m, t):
    ix = np.flatnonzero(m <= t)
    return int(ix[0] + 1) if len(ix) else None


def stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(n=len(v), median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                mean=float(v.mean()), min=float(v.min()), max=float(v.max()))


def main():
    C = read('trajectory_C480.csv'); M = read('trajectory_M1.csv'); S = read('trajectory_S480.csv')
    out = {}
    # ---- endpoint and decomposition (sec. 10) ------------------------------
    MC, MM, MS = C['Mnd'][-1], M['Mnd'][-1], S['Mnd'][-1]
    dT = MC - MS; dCtrl = MC - MM; dMat = MM - MS
    share = dMat / dT
    out['decomposition'] = dict(Mnd_C480=MC, Mnd_M1=MM, Mnd_S480=MS, dTotal=dT, dController=dCtrl, dMaterial=dMat,
                                share_material=share,
                                rule=('PRIMARILY_FORMULATION' if share >= .75 else
                                      'PRIMARILY_OUTER_BOX_CONTROLLER' if share <= .25 else 'MULTICAUSAL'),
                                M1_spike_events=int(np.nansum(M['spike'])), M1_status='CONVERGED@64 (in spike state)')
    # ---- milestones (sec. 12) ----------------------------------------------
    its = [1, 5, 10, 20, 40, 60, 64, 80, 100, 112, 150, 200, 300, 386]
    ms = []
    for k in its:
        row = {'outer': k}
        for nm, T in [('C480', C), ('M1', M), ('S480', S)]:
            if k <= len(T['Mnd']):
                row[nm + '_Mnd'] = float(T['Mnd'][k - 1]); row[nm + '_omega1'] = float(T['omega1'][k - 1])
                row[nm + '_gap12'] = float(T['gap12'][k - 1]); row[nm + '_box_max'] = float(T['box_max'][k - 1])
                row[nm + '_box_mean'] = float(T['box_mean'][k - 1])
                if 'gray' in T:
                    row[nm + '_gray'] = float(T['gray'][k - 1]); row[nm + '_mid'] = float(T['mid'][k - 1])
                    row[nm + '_broad'] = float(T['broad_core'][k - 1])
        ms.append(row)
    out['milestones'] = ms
    out['first_crossings'] = {nm: {str(t): first_cross(T['Mnd'], t) for t in [.75, .5, .35, .25, .15]}
                              for nm, T in [('C480', C), ('M1', M), ('S480', S)]}
    out['grayness_divergence_onset'] = dict(S480_vs_C480=onset(S['Mnd'], C['Mnd']), S480_vs_M1=onset(S['Mnd'], M['Mnd']),
                                            M1_vs_C480=onset(M['Mnd'], C['Mnd']))
    # first iteration where any element is <= 0.1 in the START state (formulation activation)
    out['low_density_activation'] = {nm: (int(np.flatnonzero(T['void_lt_0p1'] > 0)[0] + 2) if (T['void_lt_0p1'] > 0).any() else None)
                                     for nm, T in [('C480', C), ('M1', M)]}
    # ---- spikes -------------------------------------------------------------
    out['spikes'] = {nm: [int(x) for x in T['outer'][T['spike'] > 0]] for nm, T in [('C480', C), ('M1', M), ('S480', S)]}
    # ---- stopping (sec. 11) --------------------------------------------------
    out['stopping'] = {
        'eps': EPS, 'eps_rms_per_element': EPS / np.sqrt(NE),
        'floor_fraction_at_eps': (EPS / 0.002) ** 2 / NE,
        'S480_first_raw_stop': int(np.flatnonzero(S['stop_raw'] > 0)[0] + 1),
        'S480_final_l2': float(S['l2_drho'][-1]),
        'M1_first_raw_stop': int(np.flatnonzero(M['stop_raw'] > 0)[0] + 1),
        'C480_first_raw_l2_below_eps': int(np.flatnonzero(C['stop_raw'] > 0)[0] + 1) if (C['stop_raw'] > 0).any() else None,
        'C480_stage_starts': [1] + [int(k) for k in np.flatnonzero(np.diff(C['stage']) > 0) + 2],
        'C480_final_l2': float(C['l2_drho'][-1]),
        'C480_terminal_box': float(C['box_max'][-1]),
    }
    # ---- inner attenuation / box usage --------------------------------------
    out['box_usage'] = {}
    for nm, T in [('C480', C), ('M1', M)]:
        out['box_usage'][nm] = dict(on_bound=stats(T['on_bound']), at_move_box=stats(T['at_move_box']),
                                    step_over_box_rms=stats(T['step_over_box_rms']), sign_reversal=stats(T['sign_reversal']),
                                    nInner=stats(T['nInner']), gain_ratio=stats(T['gain_ratio']),
                                    first10_at_move_box=stats(T['at_move_box'][:10]),
                                    first10_step_over_box_rms=stats(T['step_over_box_rms'][:10]))
    out['box_usage']['S480'] = dict(nInner=stats(S['nInner']), gain_ratio=stats(S['gain_ratio']),
                                    max_over_box=stats(S['max_abs_drho'] / S['box_max']))
    out['M1_box_floor_fraction'] = stats(M['box_at_floor'])
    out['M1_box_floor_final'] = float(M['box_at_floor'][-1])
    jdump(out, EVAL / 'trajectory_analysis.json')
    print(json.dumps(out['decomposition'], indent=1))
    print(out['grayness_divergence_onset'], out['low_density_activation'])
    print(json.dumps(out['stopping'], indent=1))
    for r in ms:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if 'Mnd' in k or k == 'outer'})
    print(json.dumps(out['box_usage'], indent=1)[:3000])


if __name__ == '__main__':
    main()
