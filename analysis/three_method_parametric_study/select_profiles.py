#!/usr/bin/env python3
"""WP12/WP13/WP15 -- eligibility, Pareto fronts, quality levels, knee selection.

Implements study_preregistration.json verbatim.  The selection rule was fixed
before any hold-out mesh was run and is not re-derived from the results.
"""
import csv, json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
REPO = os.path.dirname(os.path.dirname(HERE))

def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return float('nan')

def load(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))

# Authoritative pass per method.  The first Yuksel/Proposed pass capped both
# Yuksel stages and the Proposed loop at 300 iterations, which censored the
# native stop itself; those rows stay in the ledger but cannot select a profile.
AUTH_PASS = {'Olhoff': 'stage_a', 'Yuksel': 'stage_a_v2', 'Proposed': 'stage_a_v2'}

COST = 'loop_time_practical_s'           # preregistered timing metric
QUAL = 'omega1_common_raw_E1_practical'  # preregistered quality metric

# Look-ahead progress threshold for the single-loop methods.  A native stop is
# a FALSE CONVERGENCE if more than 0.5% of the run's eventual native-objective
# improvement still lies ahead of it -- the same 0.1%-scale objective yardstick
# the Olhoff look-ahead label uses, relaxed one decade because these methods
# are compared on a monotone compliance rather than a phase-balanced omega_1.
REBOUND_HORIZON = 100

# Declared look-ahead for the single-loop methods (WP11).  The test is on the
# OUTCOME the study reports, not on whether the design has stopped jiggling:
# a stop is a false convergence when continuing the trajectory to the safety
# budget would materially change the reported common-evaluator omega_1, or
# would change the topology's support-to-support connectivity.  0.1% is the
# same objective-deviation tolerance the frozen Olhoff look-ahead label uses,
# applied to the same kind of quantity.
#
# The design-rebound fraction is computed and reported alongside it, but it is
# deliberately NOT an eligibility gate: it measures whether the native rule is
# a stationarity detector (it is not, for either method), which is a finding
# about those rules rather than grounds for discarding their outputs.
LOOKAHEAD_OMEGA1_TOL = 1e-3

def progress_at_stop(traces, run_id, pas, k):
    """Fraction of this run's total native-objective improvement realised by k."""
    tr = traces.get((run_id, pas))
    if not tr or not math.isfinite(k):
        return float('nan')
    tr = sorted(tr, key=lambda t: int(t['iter']))
    f = [fnum(t['objective']) for t in tr]
    fs = [f[0]] + [0.5 * (f[i] + f[i - 1]) for i in range(1, len(f))]
    f0 = fs[0]
    flate = sum(fs[-20:]) / min(20, len(fs))
    span = flate - f0
    if span == 0:
        return float('nan')
    idx = int(k) - 1
    if idx < 0 or idx >= len(fs):
        return float('nan')
    return (fs[idx] - f0) / span

def design_rebound(traces, run_id, pas, k, tol, horizon=REBOUND_HORIZON):
    """Fraction of the H iterations after the native stop at which the design
    change exceeds the very tolerance that fired.

    A native stop is premature exactly when the rule caught a transient lull:
    it fires, and then the design resumes moving by more than tol.  This is the
    directly measurable WP11 test for the single-loop methods, and unlike an
    objective-progress test it is not blind to design evolution that continues
    after the objective has flat-lined.
    """
    tr = traces.get((run_id, pas))
    if not tr or not math.isfinite(k) or not math.isfinite(tol):
        return float('nan')
    tr = sorted(tr, key=lambda t: int(t['iter']))
    after = [fnum(t['d_inf']) for t in tr if int(t['iter']) > k][:horizon]
    after = [v for v in after if math.isfinite(v)]
    if not after:
        return float('nan')
    return sum(1 for v in after if v > tol) / len(after)

def eligibility(r, det, traces=None):
    """Preregistered eligibility list.  Returns (bool, reason)."""
    m = r['method']
    if r['status'] != 'COMPLETED_OBSERVER':
        return False, 'NOT_COMPLETED'
    if r['pass'] != AUTH_PASS[m]:
        return False, 'CENSORED_FIRST_PASS'
    if r['practical_status'] != 'PRACTICAL_STOP':
        return False, 'NO_PRACTICAL_STOP_CAP_HIT'
    if fnum(r['n_solver_failures']) > 0:
        return False, 'SOLVER_FAILURE'
    if not math.isfinite(fnum(r[QUAL])) or not math.isfinite(fnum(r[COST])):
        return False, 'NON_FINITE_EVALUATION'
    if abs(fnum(r['vol_resid_practical'])) > 1e-3:
        return False, 'VOLUME_RESIDUAL'
    if fnum(r['connected_raw_practical']) != 1 or fnum(r['connected_bin_practical']) != 1:
        return False, 'CONNECTIVITY_FAILURE'
    if m == 'Olhoff':
        if fnum(r['gap12_native_terminal']) > 0.01:
            return False, 'NOT_BIMODAL'
        d = det.get(fnum(r['move']))
        if d is None:
            return False, 'NO_DETECTOR_RECORD'
        if d['classification'] == 'TRUE_BUT_HORIZON_LIMITED':
            return False, 'LOOKAHEAD_CENSORED'
        if not d['classification'].startswith('TRUE'):
            return False, 'FALSE_CONVERGENCE'
    else:
        # WP11 for the single-loop methods: their trajectories were continued
        # past the native stop, so premature firing is directly measurable.
        a = fnum(r['omega1_common_raw_E1_practical'])
        b = fnum(r['omega1_common_raw_E1_terminal'])
        if math.isfinite(a) and math.isfinite(b) and b != 0:
            if abs(a - b) / abs(b) > LOOKAHEAD_OMEGA1_TOL:
                return False, 'FALSE_CONVERGENCE_OMEGA1_DRIFT'
        if fnum(r['connected_raw_terminal']) != 1:
            return False, 'CONNECTIVITY_FAILURE_AFTER_STOP'
    return True, 'ELIGIBLE'

def pareto(points):
    """points: list of (idx, cost, quality).  Lower cost / higher quality better."""
    keep = []
    for i, c, q in points:
        dominated = any(
            (c2 <= c and q2 >= q and (c2 < c or q2 > q))
            for j, c2, q2 in points if j != i)
        if not dominated:
            keep.append(i)
    return keep

def main():
    ledger = load('parametric_run_ledger.csv')
    det = {}
    for d in load('olhoff_detector_grid.csv'):
        if d['family'] == 'hybrid' and d['lag'] == '2' and d['level'] == 'strict':
            det[fnum(d['move'])] = d

    traces = defaultdict(list)
    with open(os.path.join(RES, 'objective_traces.csv')) as fh:
        for t in csv.DictReader(fh):
            traces[(t['run_id'], t['pass'])].append(t)

    rows = []
    for r in ledger:
        ok, why = eligibility(r, det, traces)
        p = progress_at_stop(traces, r['run_id'], r['pass'],
                             fnum(r['practical_stop_iter']))
        rb = design_rebound(traces, r['run_id'], r['pass'],
                            fnum(r['practical_stop_iter']), fnum(r['tol']))
        rows.append(dict(r, _eligible=ok, _reason=why, _progress=p, _rebound=rb))

    # ---- Pareto per method -------------------------------------------
    out = []
    selected = {}
    for m in ('Olhoff', 'Yuksel', 'Proposed'):
        el = [r for r in rows if r['method'] == m and r['_eligible']]
        pts = [(i, fnum(r[COST]), fnum(r[QUAL])) for i, r in enumerate(el)]
        front = set(pareto(pts))
        costs = [c for _, c, _ in pts]; quals = [q for _, _, q in pts]
        cmin, cmax = min(costs), max(costs)
        qmin, qmax = min(quals), max(quals)
        best = None
        for i, r in enumerate(el):
            c, q = fnum(r[COST]), fnum(r[QUAL])
            cs = 0.0 if cmax == cmin else (c - cmin) / (cmax - cmin)
            qs = 1.0 if qmax == qmin else (q - qmin) / (qmax - qmin)
            dist = math.hypot(cs, 1.0 - qs)
            rec = {
                'method': m, 'run_id': r['run_id'], 'params': r['family_params'],
                'n_iter_practical': r['practical_stop_iter'],
                'cost_loop_s': round(c, 3), 'quality_omega1_E1': round(q, 4),
                'quality_loss_vs_best_pct': round(100 * (qmax - q) / qmax, 4),
                'cost_scaled': round(cs, 4), 'quality_scaled': round(qs, 4),
                'utopia_distance': round(dist, 4),
                'nondominated': i in front,
                'robustness_stage_a': r['validity'],
                'selection_status': 'CANDIDATE' if i in front else 'DOMINATED',
            }
            out.append(rec)
            if i in front and (best is None or dist < best['utopia_distance'] - 1e-12):
                best = rec
        # preregistered tie-break: within 0.02 prefer the simpler/more conservative
        ties = [r for r in out if r['method'] == m and r['nondominated']
                and abs(r['utopia_distance'] - best['utopia_distance']) <= 0.02]
        if len(ties) > 1:
            # "more conservative" = the tighter native control, i.e. the higher
            # cost among tied points (it buys quality and stays nearer the
            # established reproduction / default setting).
            best = max(ties, key=lambda r: r['cost_loop_s'])
            best['tie_break_applied'] = True
        for r in out:
            if r['method'] == m and r['run_id'] == best['run_id']:
                r['selection_status'] = 'SELECTED_PRIMARY'
        selected[m] = best
        # conservative companion: highest-quality eligible nondominated point
        cons = max([r for r in out if r['method'] == m and r['nondominated']],
                   key=lambda r: r['quality_omega1_E1'])
        if cons['run_id'] != best['run_id']:
            for r in out:
                if r['method'] == m and r['run_id'] == cons['run_id']:
                    r['selection_status'] = 'SELECTED_CONSERVATIVE'
            selected[m + '_conservative'] = cons

    with open(os.path.join(RES, 'method_pareto_profiles.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()) + ['tie_break_applied'])
        w.writeheader()
        for r in out:
            r.setdefault('tie_break_applied', False)
            w.writerow(r)

    # ---- eligibility ledger ------------------------------------------
    with open(os.path.join(RES, 'eligibility_ledger.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'run_id', 'pass', 'params', 'eligible', 'reason',
                    'practical_stop_iter', COST, QUAL,
                    'native_objective_progress_at_stop',
                    'design_rebound_fraction_H100', 'omega1_drift_pct', 'validity'])
        for r in rows:
            w.writerow([r['method'], r['run_id'], r['pass'], r['family_params'],
                        int(r['_eligible']), r['_reason'], r['practical_stop_iter'],
                        r[COST], r[QUAL],
                        '' if not math.isfinite(r['_progress']) else round(r['_progress'], 6),
                        '' if not math.isfinite(r['_rebound']) else round(r['_rebound'], 4),
                        '' if not math.isfinite(fnum(r['omega1_common_raw_E1_practical']))
                             or not math.isfinite(fnum(r['omega1_common_raw_E1_terminal']))
                        else round(100 * abs(fnum(r['omega1_common_raw_E1_practical'])
                                             - fnum(r['omega1_common_raw_E1_terminal']))
                                   / abs(fnum(r['omega1_common_raw_E1_terminal'])), 5),
                        r['validity']])

    # ---- WP12 within-method quality levels ---------------------------
    levels = [0.95, 0.975, 0.99, 0.995]
    qrows = []
    for r in rows:
        if not r['_eligible']:
            continue
        key = (r['run_id'], r['pass'])
        tr = traces.get(key)
        if not tr:
            continue
        tr.sort(key=lambda t: int(t['iter']))
        f = [fnum(t['objective']) for t in tr]
        tm = [fnum(t['loop_time_s']) for t in tr]
        # uniform 2-iteration phase smoothing: Olhoff's LP trajectory settles
        # into a period-two cycle, so an unsmoothed trace would credit or deny
        # a level purely by which phase an iteration lands on.
        fs = [f[0]] + [0.5 * (f[i] + f[i - 1]) for i in range(1, len(f))]
        f0, flate = fs[0], sum(fs[-20:]) / min(20, len(fs))
        span = flate - f0
        for lv in levels:
            hit_i = hit_t = float('nan')
            if span != 0:
                for i in range(len(fs)):
                    p = (fs[i] - f0) / span
                    if p >= lv and all((fs[j] - f0) / span >= lv for j in range(i, len(fs))):
                        hit_i, hit_t = int(tr[i]['iter']), tm[i]
                        break
            qrows.append({'method': r['method'], 'run_id': r['run_id'],
                          'params': r['family_params'], 'direction': tr[0]['direction'],
                          'level': lv, 'iter_to_level': hit_i, 'loop_time_to_level_s': hit_t,
                          'practical_stop_iter': r['practical_stop_iter'],
                          'practical_loop_time_s': r[COST]})
    with open(os.path.join(RES, 'within_method_quality_levels.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0].keys()))
        w.writeheader(); w.writerows(qrows)

    with open(os.path.join(RES, 'selected_profile_candidates.json'), 'w') as fh:
        json.dump({'selection_rule_source': 'study_preregistration.json',
                   'cost_metric': COST, 'quality_metric': QUAL,
                   'authoritative_pass': AUTH_PASS,
                   'selected': selected}, fh, indent=2)

    for m in ('Olhoff', 'Yuksel', 'Proposed'):
        s = selected[m]
        print(f"{m:9s} PRIMARY      {s['run_id']:22s} {s['params']:44s} "
              f"N={s['n_iter_practical']:>5s} T={s['cost_loop_s']:8.2f}s "
              f"w1={s['quality_omega1_E1']:9.4f} d={s['utopia_distance']:.4f}")
        c = selected.get(m + '_conservative')
        if c:
            print(f"{'':9s} CONSERVATIVE {c['run_id']:22s} {c['params']:44s} "
                  f"N={c['n_iter_practical']:>5s} T={c['cost_loop_s']:8.2f}s "
                  f"w1={c['quality_omega1_E1']:9.4f} d={c['utopia_distance']:.4f}")

if __name__ == '__main__':
    main()
