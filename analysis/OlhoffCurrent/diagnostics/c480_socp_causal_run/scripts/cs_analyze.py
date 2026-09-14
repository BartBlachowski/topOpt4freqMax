"""Parts 4-18: endpoint metrics, topology comparison, SOCP certification and
coverage summaries, controller events, cost, and the MECHANICAL verdicts of
AUDIT_PREREGISTRATION.md sections 11-12 (as amended by Amendment 1, which changes
no threshold).  Read-only.  Writes evaluations/analysis.json, METRICS.json,
MASTER_METRICS.csv.
"""
import csv
from cs_common import *
from cs_trajectory_metrics import read_csv

CTRL_REC = CANARY / 'runs' / 'C480x60_three_rung_record.json'
TREAT_REC = RUN / 'C480x60_socp_record.json'
SOCP_CSV = RUN / 'C480x60_socp_socp_iterations.csv'


def events(tr):
    decl = tr['exDecl']
    rise = np.flatnonzero((decl > 0) & (np.r_[0, decl[:-1]] == 0))
    ev = []
    for k in rise:
        ev.append({'iter': int(tr['outer'][k]), 'stage': int(tr['stage'][k]), 'move': float(tr['move'][k]),
                   'branch': 'B' if tr['exNB'][k] >= 20 else ('A' if tr['exNA'][k] >= 20 else '?'),
                   'amp': float(tr['exAmp'][k]), 'medcos': float(tr['exMedcos'][k]), 'mednet': float(tr['exMednet'][k]),
                   'nA': int(tr['exNA'][k]), 'nB': int(tr['exNB'][k]), 'omega1_pre': float(tr['omega1'][k]),
                   'Mnd_percent': float(tr['Mnd_percent'][k]), 'gray_fraction': float(tr['gray_fraction'][k]),
                   'broad_core_fraction': float(tr['broad_core_fraction'][k])})
    return ev


def stage_table(tr):
    out = []
    for st in np.unique(tr['stage']):
        m = tr['stage'] == st
        idx = np.flatnonzero(m)
        out.append({'stage': int(st), 'move': float(tr['move'][idx[0]]), 'first': int(tr['outer'][idx[0]]),
                    'last': int(tr['outer'][idx[-1]]), 'length': int(m.sum()),
                    'Mnd_end': float(tr['Mnd_percent'][idx[-1]]), 'gray_end': float(tr['gray_fraction'][idx[-1]]),
                    'mid_end': float(tr['mid_fraction'][idx[-1]]), 'broad_end': float(tr['broad_core_fraction'][idx[-1]]),
                    'omega1_end_pre': float(tr['omega1'][idx[-1]]), 'sum_tInner': float(tr['tInner'][m].sum()),
                    'sum_tOuter': float(tr['tOuter'][m].sum()), 'sum_nInner': float(tr['nInner'][m].sum())})
    return out


def terminal_window(tr, omega_final):
    n = len(tr['outer'])
    k = min(20, n)
    om_post = np.sqrt(np.r_[tr['lam1'][1:], omega_final[0] ** 2])
    w = slice(n - k, n)
    return {'n': k, 'Mnd_range_pp': float(np.ptp(tr['Mnd_percent'][w])), 'gray_range': float(np.ptp(tr['gray_fraction'][w])),
            'broad_range': float(np.ptp(tr['broad_core_fraction'][w])), 'mid_range': float(np.ptp(tr['mid_fraction'][w])),
            'omega1_range_rel': float(np.ptp(om_post[w]) / omega_final[0]),
            'omega1_max_along': float(max(np.max(tr['omega1']), omega_final[0])),
            'omega1_final_over_max': float(omega_final[0] / max(np.max(tr['omega1']), omega_final[0]))}


def topology(rc, rt, fc, ft):
    d = rt - rc
    sc, st_ = rc >= .5, rt >= .5
    jac = lambda a, b: float((a & b).sum() / (a | b).sum()) if (a | b).any() else None
    gc, gt = fc['gray'].T.ravel(), ft['gray'].T.ravel()
    bc, bt = fc['broad'].T.ravel(), ft['broad'].T.ravel()
    return {'l2': float(np.linalg.norm(d)), 'linf': float(np.max(np.abs(d))), 'rms': float(np.sqrt(np.mean(d * d))),
            'l2_relative_to_control': float(np.linalg.norm(d) / np.linalg.norm(rc)),
            'pearson': float(np.corrcoef(rc, rt)[0, 1]),
            'solid_threshold_0p5_jaccard': jac(sc, st_), 'solid_threshold_0p5_agreement': float(np.mean(sc == st_)),
            'void_threshold_0p5_jaccard': jac(~sc, ~st_),
            'gray_mask_jaccard': jac(gc, gt), 'broad_core_jaccard': jac(bc, bt),
            'material_relocation_fraction': float(np.abs(d).sum() / (2 * rc.sum())),
            'frac_elements_abs_diff_gt_0p5': float(np.mean(np.abs(d) > .5)),
            'frac_elements_abs_diff_gt_0p1': float(np.mean(np.abs(d) > .1))}


def socp_summary(path):
    T = read_csv(path)
    acc = T['accepted'] == 1
    n = len(acc)
    s = {'rows': n, 'accepted': int(acc.sum()), 'rejected_rows': int((~acc).sum())}
    A = {k: v[acc] for k, v in T.items()}
    NE = 28800
    s['N_counts'] = {'N=2': n}   # E1 checked per row; any N!=2 would be a rejected row
    s['attempt1_schur'] = int(np.sum(A['acceptedAttempt'] == 1)); s['attempt2_augmented'] = int(np.sum(A['acceptedAttempt'] == 2))
    for k in ['gap', 'primalResidual', 'boxComp', 'rowComp', 'bsStat', 'statRms', 'statMax', 'rawBoxViolation',
              'eqAcc_cone_err', 'eqAcc_grad_relerr', 'eqAcc_redundancy', 'eqAcc_row_nextmode_err', 'eqAcc_row_volume_err']:
        v = A[k][np.isfinite(A[k])]
        s[k] = {'max': float(v.max()), 'median': float(np.median(v)), 'min': float(v.min())} if len(v) else None
    s['gap_abs_max'] = float(np.max(np.abs(A['gap'])))
    s['exitflag1_counts'] = {str(int(e)): int(c) for e, c in zip(*np.unique(A['exitflag1'], return_counts=True))}
    s['candidate_counts'] = {str(int(e)): int(c) for e, c in zip(*np.unique(A['candidateIndex'], return_counts=True))}
    s['candidate_index_legend'] = {'1': 'solverDuals', '2': 'complementarySlackness', '3': 'dualboundGeneral', '4': 'dualboundAligned'}
    s['apex_count'] = int(np.nansum(A['apex']))
    s['ipm_iterations'] = {'median': float(np.median(A['iters1'])), 'max': float(np.max(A['iters1'])), 'sum': float(np.nansum(A['iters1']) + np.nansum(A['iters2']))}
    fr = lambda k: A[k] / NE
    s['bound_fractions_median'] = {k: float(np.median(fr(k))) for k in ['nLowerDensity', 'nLowerMove', 'nUpperMove', 'nUpperDensity', 'nInterior']}
    anyb = 1 - fr('nInterior')
    s['frac_any_bound'] = {'median': float(np.median(anyb)), 'min': float(anyb.min()), 'max': float(anyb.max())}
    s['iterations_with_ge_99pct_bound'] = int(np.sum(anyb >= .99))
    s['iterations_with_ge_99p9pct_bound'] = int(np.sum(anyb >= .999))
    gfm = np.where(A['nGray'] > 0, A['nGrayFullMove'] / np.maximum(A['nGray'], 1), np.nan)
    s['gray_full_move_fraction'] = {'median': float(np.nanmedian(gfm)), 'min': float(np.nanmin(gfm)), 'max': float(np.nanmax(gfm))}
    s['iterations_all_gray_full_move'] = int(np.sum(A['nGrayFullMove'] == A['nGray']))
    cd = A['cross_d2']; ci = A['cross_dinf']
    ok = np.isfinite(cd)
    s['degeneracy_cross_solver'] = {'n': int(ok.sum()), 'd2_median': float(np.median(cd[ok])) if ok.any() else None,
        'd2_p90': float(np.quantile(cd[ok], .9)) if ok.any() else None, 'd2_max': float(cd[ok].max()) if ok.any() else None,
        'dinf_median': float(np.median(ci[ok])) if ok.any() else None, 'dinf_max': float(ci[ok].max()) if ok.any() else None,
        'frac_d2_gt_0p01': float(np.mean(cd[ok] > .01)) if ok.any() else None,
        'frac_dinf_gt_0p1': float(np.mean(ci[ok] > .1)) if ok.any() else None,
        'nDiff_gt_0p1move_median': float(np.median(A['cross_nDiffGt0p1Move'][ok])) if ok.any() else None,
        'nDiff_gt_0p1move_max': float(np.max(A['cross_nDiffGt0p1Move'][ok])) if ok.any() else None,
        'cross_certified_frac': float(np.mean(A['cross_certified'][ok] == 1)) if ok.any() else None,
        'dbs_max': float(np.max(A['cross_dbs'][ok])) if ok.any() else None}
    s['time'] = {k: float(np.nansum(A[k])) for k in ['tEligibility', 'tAssembly', 'tSolve', 'tCertificate', 'tTotal', 'cross_tSolve', 'cross_tCertificate']}
    s['time_per_iter_median'] = {k: float(np.nanmedian(A[k])) for k in ['tAssembly', 'tSolve', 'tCertificate', 'tTotal']}
    return s, T


def verdicts(ctrl_geo, tr_geo, ctrl_om, tr_om, twi_t, status, termination, coverage_ok, real_v, identity_ok):
    rM = tr_geo['Mnd_percent'] / ctrl_geo['Mnd_percent']
    rG = tr_geo['gray_fraction'] / ctrl_geo['gray_fraction']
    rMid = tr_geo['mid_fraction'] / ctrl_geo['mid_fraction']
    rB = tr_geo['broad_core_fraction'] / ctrl_geo['broad_core_fraction']
    Rw = tr_om / ctrl_om
    G_major = rM <= .5 and rB <= .25 and rG <= .5 and rMid <= .5
    G_material = rM <= .8 and rB <= .8
    OBJ_OK = Rw >= .99
    OBJ_COLLAPSE = Rw < .95
    TWI = twi_t['Mnd_range_pp'] > 1.0 or twi_t['gray_range'] > .01 or twi_t['broad_range'] > .01 or twi_t['omega1_range_rel'] > .005
    D4 = twi_t['omega1_final_over_max'] < .95
    cap = status == 'CAP_HIT'
    E = bool(termination) or not identity_ok
    trace = {'ratio_Mnd': rM, 'ratio_gray': rG, 'ratio_mid': rMid, 'ratio_broad': rB, 'ratio_omega1': Rw,
             'G_major': G_major, 'G_material': G_material, 'OBJ_OK': OBJ_OK, 'OBJ_COLLAPSE': OBJ_COLLAPSE,
             'TWI': TWI, 'D4_omega_lost_from_max': D4, 'CAP_HIT': cap, 'E_evidence_failure': E}
    if E:
        causal = 'C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE'
    elif cap or TWI or OBJ_COLLAPSE or D4:
        causal = 'EXACT_INNER_SOLVE_EXPOSES_OUTER_GLOBALIZATION_FAILURE'
    elif G_major and OBJ_OK:
        causal = 'INNER_SOLVER_MAJOR_CAUSE_OF_C480_GRAYNESS'
    elif G_material:
        causal = 'INNER_SOLVER_PARTIAL_CAUSE_OF_C480_GRAYNESS'
    else:
        causal = 'INNER_SOLVER_NOT_PRIMARY_CAUSE_OF_C480_GRAYNESS'
    coverage = 'C480_FULL_RUN_SOCP_COVERAGE_PASS' if coverage_ok else 'C480_FULL_RUN_SOCP_COVERAGE_FAIL'
    short = causal.split('_')[0]
    cls = {'C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE': 'E', 'EXACT_INNER_SOLVE_EXPOSES_OUTER_GLOBALIZATION_FAILURE': 'D',
           'INNER_SOLVER_MAJOR_CAUSE_OF_C480_GRAYNESS': 'A', 'INNER_SOLVER_PARTIAL_CAUSE_OF_C480_GRAYNESS': 'B',
           'INNER_SOLVER_NOT_PRIMARY_CAUSE_OF_C480_GRAYNESS': 'C'}[causal]
    if not coverage_ok:
        socp = 'SOCP_INNER_SOLVER_CANDIDATE_REJECTED'
    elif real_v in ('OUTER_MODEL_REALIZATION_HEALTHY', 'OUTER_MODEL_REALIZATION_MARGINAL') and cls in 'ABC':
        socp = 'SOCP_REMAINS_VALID_INNER_SOLVER_CANDIDATE'
    else:
        socp = 'SOCP_INNER_SOLVER_CANDIDATE_WEAKENED'
    filt = 'FILTER_FORMULATION_STUDY_NOW_JUSTIFIED' if cls in 'BC' else 'FILTER_FORMULATION_STUDY_STILL_DEFERRED'
    perf = ('PERFORMANCE_CAMPAIGN_RECONSIDERATION_JUSTIFIED'
            if cls == 'A' and real_v == 'OUTER_MODEL_REALIZATION_HEALTHY' and coverage_ok else 'PERFORMANCE_CAMPAIGN_STILL_BLOCKED')
    return {'causal': causal, 'causal_class': cls, 'coverage': coverage, 'realization': real_v, 'socp_candidate': socp,
            'filter_gate': filt, 'performance_gate': perf, 'trace': trace}


def stationarity_compare(sc, st):
    out = {}
    for label, key in [('physical_raw_all_free_dual', 'raw_reduced_normalized'), ('physical_raw_gray_fit_dual', 'raw_grayfit_normalized'),
                       ('filtered_gray_fit_dual_common_scale', 'filtered_grayfit_common_scale'),
                       ('filtered_all_free_dual_common_scale', 'filtered_reduced_common_scale')]:
        row = {}
        for cl in ['gray', 'mid', 'solid', 'void', 'broad']:
            a = sc['classes'].get(cl, {}).get(key, {}) or {}
            b = st['classes'].get(cl, {}).get(key, {}) or {}
            ra, rb = a.get('RMS'), b.get('RMS')
            row[cl] = {'control_RMS': ra, 'treatment_RMS': rb, 'n_control': sc['classes'].get(cl, {}).get('n'),
                       'n_treatment': st['classes'].get(cl, {}).get('n'),
                       'ratio': (rb / ra) if (ra and rb is not None) else None}
        out[label] = row
    for label, key in [('global_projected_raw', 'raw_KKT'), ('global_projected_filtered', 'filtered_subproblem')]:
        out[label] = {'control_RMS': sc[key]['global_projected_residual_normalized']['RMS'],
                      'treatment_RMS': st[key]['global_projected_residual_normalized']['RMS'],
                      'control_bounds': [sc[key]['lower_count'], sc[key]['upper_count']],
                      'treatment_bounds': [st[key]['lower_count'], st[key]['upper_count']],
                      'control_mu': sc[key]['mu_volume'], 'treatment_mu': st[key]['mu_volume'],
                      'control_scale': sc[key]['scale_raw_objective_RMS_interior'],
                      'treatment_scale': st[key]['scale_raw_objective_RMS_interior']}

    def judge(r):
        if r is None: return 'INCONCLUSIVE'
        return 'IMPROVED' if r <= .5 else ('WORSENED' if r >= 2.0 else 'SIMILAR')
    out['physical_verdict'] = judge(out['physical_raw_gray_fit_dual']['gray']['ratio'])
    out['filtered_verdict'] = judge(out['filtered_gray_fit_dual_common_scale']['gray']['ratio'])
    return out


def main():
    ci = json.loads((EV / 'control_identity.json').read_text())
    crec = json.loads(CTRL_REC.read_text()); trec = json.loads(TREAT_REC.read_text())
    ctr = read_csv(EV / 'traj_control.csv'); ttr = read_csv(EV / 'traj_treatment.csv')
    csum = json.loads((EV / 'traj_control_summary.json').read_text())
    tsum = json.loads((EV / 'traj_treatment_summary.json').read_text())
    with h5py.File(CONTROL_TRAJ, 'r') as f:
        rc = f['RHO'][-1]
    with h5py.File(TREAT_TRAJ, 'r') as f:
        rt = f['RHO'][-1] if f['RHO'].shape[0] else np.full(28800, .5)
    assert sha256_vec(rc) == '0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60'
    assert sha256_vec(rt) == trec['rho_sha256'], 'treatment rho hash mismatch'
    gc, fc = geometry_metrics(rc); gt, ft = geometry_metrics(rt)
    ctrl_om, tr_om = crec['omega'][0], trec['omega'][0]
    tw_c = terminal_window(ctr, np.array(crec['omega'])); tw_t = terminal_window(ttr, np.array(trec['omega']))
    socp, T = socp_summary(SOCP_CSV)
    termination = trec.get('termination') or ''
    coverage_ok = (not termination) and socp['rejected_rows'] == 0 and socp['accepted'] == trec['nOuter']
    pf = json.loads((EV / 'preflight.json').read_text())
    identity_ok = (ci['verdict'] == 'C480_CONTROL_EVIDENCE_PASS' and pf['verdict'] == 'C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS'
                   and trec['meta']['implTree_post'] == trec['meta']['implTree_pre'] == 'edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb'
                   and trec['meta']['cfgHash'] == '03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e'
                   and trec['rebuildExact'])
    V = verdicts(gc, gt, ctrl_om, tr_om, tw_t, trec['status'], termination, coverage_ok,
                 tsum['verdict_if_treatment_rule_applied'], identity_ok)
    k_t = int(trec['nOuter'])
    with h5py.File(CONTROL_TRAJ, 'r') as f:
        rc_k = f['RHO'][k_t - 1]
    gck, fck = geometry_metrics(rc_k)
    matched = {'iteration': k_t, 'control_geometry': gck, 'treatment_geometry': gt,
               'control_omega1_at_rho_k': float(ctr['omega1'][k_t]) if k_t < len(ctr['omega1']) else None,
               'treatment_omega1_at_rho_k': tr_om, 'topology': topology(rc_k, rt, fck, ft),
               'control_traj_row': {k: float(v[k_t - 1]) for k, v in ctr.items()},
               'treatment_traj_row': {k: float(v[k_t - 1]) for k, v in ttr.items()}}
    s14 = EV / 'stationarity_control14.json'
    stc = json.loads((EV / 'stationarity_control.json').read_text())
    stt = json.loads((EV / 'stationarity_treatment.json').read_text()) if (EV / 'stationarity_treatment.json').exists() else None
    stat = stationarity_compare(stc, stt) if stt else None
    stat_matched = stationarity_compare(json.loads(s14.read_text()), stt) if (stt and s14.exists()) else None

    def spectral(rec):
        om = rec['omega']
        return {'omega1': om[0], 'omega2': om[1], 'omega3': om[2], 'lambda1': om[0] ** 2, 'lambda2': om[1] ** 2,
                'gap12': (om[1] - om[0]) / om[0], 'gap23_next_mode': (om[2] - om[1]) / om[1], 'volume': rec['volume_final']}
    A = {'control': {'geometry': gc, 'spectral': spectral(crec), 'status': crec['status'], 'nOuter': crec['nOuter'],
                     'events': events(ctr), 'stages': stage_table(ctr), 'terminal_window': tw_c, 'realization': csum,
                     'inner_total': crec['innerTotal'], 'wall_s': crec['wall_s'], 't': crec['t']},
         'treatment': {'geometry': gt, 'spectral': spectral(trec), 'status': trec['status'], 'termination': termination,
                       'nOuter': trec['nOuter'], 'events': events(ttr), 'stages': stage_table(ttr),
                       'terminal_window': tw_t, 'realization': tsum, 'socp': socp, 'record_t': trec.get('t'),
                       'wall_solver_s': trec.get('wallclock_solver'), 'meta': trec['meta'],
                       'terminalDeclared': trec.get('terminalDeclared'), 'terminalBranch': trec.get('terminalBranch'),
                       'stageStarts': trec.get('stageStarts')},
         'topology': topology(rc, rt, fc, ft), 'stationarity': stat, 'stationarity_matched_iteration': stat_matched,
         'matched_iteration': matched, 'verdicts': V, 'identity_ok': identity_ok,
         'objective_improvement': {'omega1_delta': tr_om - ctrl_om, 'omega1_ratio': tr_om / ctrl_om,
                                   'lambda1_ratio': (tr_om / ctrl_om) ** 2,
                                   'control_from_initial': ctrl_om - ctr['omega1'][0],
                                   'treatment_from_initial': tr_om - ttr['omega1'][0]}}
    dump(EV / 'analysis.json', A)
    np.savez_compressed(EV / 'endpoint_fields.npz', rho_control=rc, rho_treatment=rt,
                        depth_control=fc['depth'], depth_treatment=ft['depth'], labels_control=fc['labels'],
                        labels_treatment=ft['labels'])
    # MASTER_METRICS.csv
    s14 = json.loads(s14.read_text()) if s14.exists() else {'omega': [np.nan] * 3}
    om14 = s14['omega']
    rows = [('omega1', ctrl_om, om14[0], tr_om), ('omega2', crec['omega'][1], om14[1], trec['omega'][1]),
            ('gap12', A['control']['spectral']['gap12'], (om14[1] - om14[0]) / om14[0], A['treatment']['spectral']['gap12']),
            ('gap23_next_mode', A['control']['spectral']['gap23_next_mode'], (om14[2] - om14[1]) / om14[1], A['treatment']['spectral']['gap23_next_mode']),
            ('volume', crec['volume_final'], matched['control_traj_row']['vol'], trec['volume_final']),
            ('outer_iterations_accepted', crec['nOuter'], k_t, trec['nOuter'])]
    for k in ['Mnd_percent', 'gray_fraction', 'mid_fraction', 'broad_core_fraction', 'broad_core_area', 'gray_area',
              'max_depth', 'max_depth_over_R', 'gray_components_4', 'gray_components_8', 'largest_gray_component_area',
              'rho_lt_001', 'rho_gt_099']:
        rows.append((k, gc[k], gck[k], gt[k]))
    for k in ['median_r', 'frac_act_negative', 'cum_ratio', 'frac_cos_lt_m0p5', 'frac_lam1_decrease']:
        rows.append((f'realization_{k}_all_steps', csum.get(k), None, tsum.get(k)))
    with open(STUDY / 'MASTER_METRICS.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'control_final_outer386', f'control_matched_outer{k_t}', f'treatment_exactSOCP_outer{k_t}_terminated',
                    'treatment_over_control_final', 'treatment_over_control_matched'])
        for name, a, b, c in rows:
            ok = lambda u, v: isinstance(u, (int, float)) and isinstance(v, (int, float)) and u not in (0, None)
            w.writerow([name, a, '' if b is None else b, c, (c / a) if ok(a, c) else '', (c / b) if ok(b, c) else ''])
    M = {'study': 'c480_socp_causal_run', 'verdicts': V, 'control': {k: A['control'][k] for k in ['spectral', 'status', 'nOuter']},
         'treatment': {k: A['treatment'][k] for k in ['spectral', 'status', 'nOuter', 'termination']},
         'geometry_control': {k: v for k, v in gc.items() if k != 'components'},
         'geometry_treatment': {k: v for k, v in gt.items() if k != 'components'},
         'topology': A['topology'], 'socp': socp, 'realization_treatment': tsum, 'realization_control': csum,
         'stationarity': stat, 'terminal_window_treatment': tw_t, 'terminal_window_control': tw_c,
         'events_control': A['control']['events'], 'events_treatment': A['treatment']['events'],
         'matched_iteration': {k: v for k, v in matched.items() if k not in ('control_geometry', 'treatment_geometry')},
         'matched_geometry_control': {k: v for k, v in gck.items() if k != 'components'},
         'stationarity_matched_iteration': stat_matched,
         'termination_record': json.loads((EV / 'termination_record.json').read_text()) if (EV / 'termination_record.json').exists() else None,
         'posthoc_apex_diagnostic_EXCLUDED_FROM_VERDICTS': json.loads((EV / 'posthoc_apex_diagnostic.json').read_text()) if (EV / 'posthoc_apex_diagnostic.json').exists() else None}
    dump(STUDY / 'METRICS.json', M)
    print(json.dumps(V, indent=1, default=float))


if __name__ == '__main__':
    main()
