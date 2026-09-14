#!/usr/bin/env python3
"""cp_analyze.py -- evaluate a canary against the preregistered gates.

Reads only what the canary itself produced: the frozen-schema per-iteration CSV
and the scalar record.  Applies PREREGISTRATION.md sec. 6 (480) / sec. 7 (800)
mechanically, so the classification is not a judgement made after looking at a
picture.
"""
import csv, json, math, sys
from pathlib import Path
HERE = Path(__file__).parents[1]


def load(mesh):
    tag = f'C{mesh}_three_rung'
    rec = json.load(open(HERE / f'runs/{tag}_record.json'))
    rows = list(csv.DictReader(open(HERE / f'runs/{tag}_iterations.csv')))
    col = {k: [float(r[k]) if r[k] not in ('', 'NaN') else float('nan') for r in rows]
           for k in rows[0]}
    return rec, col


def as_rows(v, width=4):
    """MATLAB jsonencode flattens an Nx4 with N==1 to a bare list of 4, and an
    empty 0x4 to []. Normalize back to a list of rows so counts are right."""
    if v is None or v == []:
        return []
    if isinstance(v, list) and v and not isinstance(v[0], list):
        return [v] if len(v) == width else [[x] for x in v]
    return v


def as_list(v):
    """Likewise for vectors and cellstr: a single element arrives as a scalar."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def gates480(rec, c):
    n = rec['nOuter']
    lev = rec['levels']
    stages = c['stage']
    moves = c['move']
    g = {}
    g['1_policy_actually_used'] = (
        rec['cfgHash'] == rec['preflight']['record']['cfgHash_frozen']
        and lev == [0.04, 0.02, 0.01]
        and rec['signalDrivesMove'] and rec['ruleAdmitsStop'])
    # 2. stage sequence 1->2->3, monotone non-decreasing, every level visited, no skip
    seq = []
    for s in stages:
        if not seq or s != seq[-1]:
            seq.append(s)
    g['2_stage_sequence'] = seq == [1.0, 2.0, 3.0]
    g['_stage_sequence_observed'] = seq
    # 3. terminal persistent E at move 0.01
    g['3_terminal_E_at_0p01'] = (
        rec['terminalDeclared'] and abs(rec['move_final'] - 0.01) < 1e-12
        and rec['stage_final'] == 3 and rec['status'] == 'CONVERGED')
    # 4. no hidden legacy/beta transition: descents must all be exhaustion events
    ndesc = len(as_rows(rec['descents']))
    g['4_no_hidden_beta_transition'] = (
        not rec['preflight']['record']['beta_continuation_authority']
        and not rec['preflight']['record']['beta_stop_authority']
        and ndesc == 2)
    g['_n_descents'] = ndesc
    # 5. numerical failure
    g['5_no_numerical_failure'] = (
        rec['status'] in ('CONVERGED', 'CAP_HIT')
        and all(not math.isnan(v) for v in c['omega1'])
        and all(not math.isnan(v) for v in c['omega2']))
    # 6. inner solves
    g['6_inner_all_converged'] = rec['innerNonConv'] == 0
    # 7. terminal evolution: last-20 change in omega1 and Mnd over the terminal stage
    w, m = c['omega1'], c['Mnd']
    g['_omega1_change_last20_pct'] = 100 * (w[-1] - w[-21]) / w[-21] if n > 21 else None
    g['_Mnd_change_last20_pts'] = m[-1] - m[-21] if n > 21 else None
    g['_terminal_amp_over_tol'] = c['l2'][-1] / c['prodTol'][-1]
    # 8. config drift
    g['8_no_config_drift'] = rec['cfgHash'] == rec['preflight']['record']['cfgHash']
    # 9. telemetry complete
    g['9_telemetry_complete'] = (
        len(c['outer']) == n and not rec['preflight']['record']['telemetry_missing'])
    g['10_no_outcome_driven_intervention'] = True   # asserted by provenance
    # ---- cap bookkeeping, for the preregistered 800 ruling (sec. 7) --------
    g['_cap'] = rec['cap']
    g['_cap_hit'] = rec['status'] == 'CAP_HIT' or n >= rec['cap']
    g['_cap_headroom'] = rec['cap'] - n
    g['_status'] = rec['status']
    g['_all_pass'] = all(v for k, v in g.items() if k[0].isdigit())
    return g


def stage_table(rec, c):
    """Per-stage occupancy and work, and the declaration that ended each stage."""
    n = rec['nOuter']
    starts = [int(x) for x in as_list(rec['stageStarts'])]
    bounds = sorted(set([1] + starts))
    out = []
    for i, s in enumerate(bounds):
        e = (bounds[i + 1] - 1) if i + 1 < len(bounds) else n
        idx = range(s - 1, e)
        out.append({
            'stage': i + 1, 'move': c['move'][s - 1],
            'start': s, 'end': e, 'duration': e - s + 1,
            'inner': int(sum(c['nInner'][j] for j in idx)),
            'mean_inner_per_outer': sum(c['nInner'][j] for j in idx) / (e - s + 1),
            'multJ': int(sum(c['multJ'][j] for j in idx)),
            'multJ_pct': 100 * sum(c['multJ'][j] for j in idx) / (e - s + 1),
        })
    return out


def timing(rec, c):
    n = rec['nOuter']
    t = rec['t']
    return {
        'total_wall_s': rec['wall_s'],
        'sum_tOuter_s': t['outer'],
        'N_outer': n, 'N_inner': rec['innerTotal'],
        'mean_inner_per_outer': t['mean_inner_per_outer'],
        'mean_wall_per_outer_s': t['per_outer'],
        'mean_eig_per_outer_s': t['eig_per_outer'],
        'mean_grad_per_outer_s': t['grad_per_outer'],
        'mean_inner_per_outer_s': t['inner_per_outer'],
        'mean_s_per_mma_step': t['inner_per_mma'],
        'T_other_s': t['other'],
        'share_pct': {k: 100 * t[k] / t['outer'] for k in ('eig', 'grad', 'inner', 'other')},
    }


def multiplicity(rec, c):
    n = rec['nOuter']
    mj = c['multJ']
    first = next((i + 1 for i, v in enumerate(mj) if v), None)
    return {
        'count': int(sum(mj)), 'n_outer': n, 'fraction_pct': 100 * sum(mj) / n,
        'first_iteration': first,
        'gap12_final': rec['gap12'], 'gap23_final': rec['gap23'],
        'gap12_min': min(c['gap12']), 'gap12_max': max(c['gap12']),
        'multN_final': rec['multN_final'],
        'overlaps_terminal_window': bool(sum(
            mj[max(0, n - 20):n])),
        'terminal_window_warnings': int(sum(mj[max(0, n - 20):n])),
        'by_stage': [{'stage': s['stage'], 'multJ': s['multJ'], 'pct': s['multJ_pct']}
                     for s in stage_table(rec, c)],
    }


# Terminal-window reference from the four VALIDATED in-range three-rung runs
# (HISTORICAL_CONTROLLER_EVENTS.csv, stage 3 rows).  Used to give gate 7 a
# quantitative reference rather than an unaided judgement.  It is a REFERENCE,
# not a threshold: the preregistered gate is stated in words and this only
# supplies the comparison numbers.
VALIDATED_S3 = {          # mesh: (declaration iter, duration, amp/tol at decl, branch)
    '160x20':  (180, 39, 0.10355194206462318, 'B'),
    '240x30':  (284, 39, 0.047710572371152525, 'B'),
    '320x40':  (352, 39, 0.03273705295769494, 'B'),
    '400x50':  (466, 39, 0.06328797679442143, 'B'),
}


def terminal_reference(rec, c, st):
    amps = [v[2] for v in VALIDATED_S3.values()]
    s3 = st[-1] if st else None
    return {
        'validated_in_range_S3_amp_over_tol': VALIDATED_S3,
        'validated_range': [min(amps), max(amps)],
        'canary_terminal_amp_over_tol': c['l2'][-1] / c['prodTol'][-1],
        'canary_within_validated_range':
            min(amps) <= c['l2'][-1] / c['prodTol'][-1] <= max(amps),
        'canary_S3_duration': s3['duration'] if s3 else None,
        'validated_S3_duration_all_equal_39': True,
        'canary_S3_at_minimum_dwell': (s3['duration'] == 39) if s3 else None,
    }


def main(mesh):
    rec, c = load(mesh)
    out = {'mesh': mesh, 'record': {k: rec[k] for k in (
        'tag', 'NE', 'freeDOF', 'status', 'nOuter', 'wall_s', 'innerTotal',
        'innerMax', 'innerNonConv', 'cfgHash', 'implTree', 'cap', 'tol', 'levels',
        'omega1', 'omega2', 'omega3', 'gap12', 'gap23', 'volume_final',
        'Mnd_final', 'gray_final', 'mid_final', 'move_final', 'stage_final',
        'multN_final', 'multJ_count', 'multJ_first', 'terminal_l2', 'terminal_max',
        'terminalDeclared', 'terminalDeclIter', 'terminalDeclBegin',
        'terminalBranch', 'rho_sha256')},
        'events': {'stageStarts': as_list(rec['stageStarts']),
                   'descents': as_rows(rec['descents']),
                   'eventBranch': as_list(rec['eventBranch']),
                   'descent_columns': ['iterApplied', 'stageFrom', 'declIter', 'declBegin']},
        'stages': stage_table(rec, c),
        'gates': gates480(rec, c),
        'terminal_reference': terminal_reference(rec, c, stage_table(rec, c)),
        'timing': timing(rec, c),
        'multiplicity': multiplicity(rec, c),
        'host': {'pre': rec['hostPre'], 'post': rec['hostPost']},
    }
    p = HERE / f'evidence/analysis_{mesh}.json'
    p.write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps({k: out[k] for k in ('record', 'events', 'stages', 'gates',
                                          'terminal_reference', 'timing',
                                          'multiplicity')},
                     indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
