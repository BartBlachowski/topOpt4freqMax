#!/usr/bin/env python3
"""WP18 -- robustness classification of the frozen profiles, per the
preregistered class definitions in study_preregistration.json."""
import csv, json, math, os
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return float('nan')

rows = [r for r in csv.DictReader(open(os.path.join(RES, 'cross_resolution_validation.csv')))
        if r['mode'] == 'observer']
man = json.load(open(os.path.join(RES, 'profile_freeze_manifest.json')))
key_of = {p['profile_id']: k for k, p in man['profiles'].items()}

by = defaultdict(list)
for r in rows:
    by[r['profile']].append(r)

out = []
for pid, rs in by.items():
    rs.sort(key=lambda r: int(r['nelx']))
    meshes = len(rs)
    fails = sum(1 for r in rs if r['convergence_status'] == 'SOLVER_FAILURE')
    caps = sum(1 for r in rs if r['convergence_status'] == 'CAP_HIT')
    disconn = sum(1 for r in rs if fnum(r['connected_raw_terminal']) != 1)
    notbi = sum(1 for r in rs if r['convergence_status'] == 'STATIONARY_NOT_BIMODAL')
    valid = sum(1 for r in rs if r['convergence_status'] in ('CONVERGED_BIMODAL', 'CONVERGED_NATIVE'))
    invalid = meshes - valid
    if fails > 1:
        cls = 'FAILED'
    elif invalid >= 2:
        cls = 'FAILED'
    elif invalid == 1:
        cls = 'RESOLUTION_SENSITIVE'
    elif fails or disconn:
        cls = 'RESOLUTION_SENSITIVE'
    else:
        cls = 'ROBUST'
    out.append({
        'profile_key': key_of.get(pid, ''), 'profile_id': pid, 'method': rs[0]['method'],
        'meshes_tested': meshes, 'meshes_valid': valid,
        'cap_hits': caps, 'solver_failures': fails,
        'not_bimodal': notbi, 'disconnected': disconn,
        'robustness_class': cls,
        'iterations': '/'.join('cap' if r['practical_stop_iter'] in ('NaN', '')
                               else str(int(fnum(r['practical_stop_iter']))) for r in rs),
        'ms_per_iter': '/'.join(f"{1000*fnum(r['loop_time_per_iter_s']):.0f}" for r in rs),
        'omega1_common_raw_E1_practical': '/'.join(
            'n/a' if r['omega1_common_raw_E1_practical'] in ('NaN', '')
            else f"{fnum(r['omega1_common_raw_E1_practical']):.2f}" for r in rs),
        'terminal_gap12_pct': '/'.join(f"{100*fnum(r['gap12_native']):.3f}" for r in rs),
        'connected_terminal': '/'.join(str(int(fnum(r['connected_raw_terminal']))) for r in rs),
    })

out.sort(key=lambda r: (r['method'], r['profile_key']))
with open(os.path.join(RES, 'robustness_classification.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

print(f"{'profile_key':26s} {'method':9s} {'valid':>6s} {'iters (160/320/400)':>22s} "
      f"{'ms/it':>16s} {'class':22s}")
for r in out:
    print(f"{r['profile_key']:26s} {r['method']:9s} {r['meshes_valid']}/{r['meshes_tested']:<4d} "
          f"{r['iterations']:>22s} {r['ms_per_iter']:>16s} {r['robustness_class']:22s}")
