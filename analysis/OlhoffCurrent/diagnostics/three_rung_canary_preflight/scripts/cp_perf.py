#!/usr/bin/env python3
"""cp_perf.py -- Part E/F analysis on the data that EXISTS: the legacy campaign.

No canary ran, so no canary decomposition is produced.  What is produced is
(a) the exact arithmetic of the legacy 720 -> 800 total-wall inversion, and
(b) legacy per-outer kernel scaling exponents, which are the reference the
    Part F fixed-work measurement would have been compared against.
Every number here is LEGACY (beta / four-rung / diagnostics off).
"""
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import ROOT

AUD = ROOT / 'analysis/OlhoffCurrent/diagnostics/nine_mesh_campaign_audit'
HERE = Path(__file__).parents[1]
L = {int(r['NE']): r for r in csv.DictReader(open(AUD / 'MASTER_TABLE.csv'))}
f = lambda ne, k: float(L[ne][k])


def fit(xs, ys):
    n = len(xs)
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    p = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    c = math.exp(my - p * mx)
    ssr = sum((b - (math.log(c) + p * a)) ** 2 for a, b in zip(lx, ly))
    sst = sum((b - my) ** 2 for b in ly)
    return {'C': c, 'p': p, 'R2_log': 1 - ssr / sst}


A, B = 64800, 80000     # 720x90 -> 800x100
inv = {
    'note': 'LEGACY campaign only (beta / four-rung / diagnostics OFF).',
    'from': '720x90', 'to': '800x100',
    'NE_ratio': B / A,
    'total_wall_ratio': f(B, 'runtime_total_s') / f(A, 'runtime_total_s'),
    'per_outer_ratio': f(B, 'wall_per_outer_s') / f(A, 'wall_per_outer_s'),
    'outer_count_ratio': int(L[B]['outer']) / int(L[A]['outer']),
    'identity_check': (f(B, 'wall_per_outer_s') / f(A, 'wall_per_outer_s'))
                      * (int(L[B]['outer']) / int(L[A]['outer'])),
    'components_per_outer_ratio': {
        'eigen': f(B, 'eigen_per_outer_s') / f(A, 'eigen_per_outer_s'),
        'gradient': f(B, 'gradient_per_outer_s') / f(A, 'gradient_per_outer_s'),
        'inner': f(B, 'inner_per_outer_s') / f(A, 'inner_per_outer_s'),
    },
    'inner_split': {
        'mma_steps_per_outer_ratio': f(B, 'mean_inner_per_outer') / f(A, 'mean_inner_per_outer'),
        's_per_mma_step_ratio': f(B, 'inner_s_per_MMA') / f(A, 'inner_s_per_MMA'),
    },
}
inv['local_exponents_vs_NE'] = {
    k: math.log(v) / math.log(B / A)
    for k, v in {**inv['components_per_outer_ratio'],
                 's_per_mma_step': inv['inner_split']['s_per_mma_step_ratio']}.items()
}

ne = sorted(L)
fits = {k: fit(ne, [f(x, k) for x in ne]) for k in
        ('wall_per_outer_s', 'eigen_per_outer_s', 'gradient_per_outer_s',
         'inner_per_outer_s', 'inner_s_per_MMA', 'runtime_total_s')}
fits['outer'] = fit(ne, [int(L[x]['outer']) for x in ne])

out = {
    'legacy_inversion_720_to_800': inv,
    'legacy_nine_mesh_loglog_fits': fits,
    'canary_decomposition': 'NOT_REACHED — no canary ran',
    'fixed_work_measurement': 'NOT_REACHED — no saved canary state exists to benchmark',
    'per_mesh': {L[x]['mesh']: {k: f(x, k) for k in
                 ('runtime_total_s', 'wall_per_outer_s', 'eigen_per_outer_s',
                  'gradient_per_outer_s', 'inner_per_outer_s', 'inner_s_per_MMA',
                  'mean_inner_per_outer')} | {'outer': int(L[x]['outer']),
                  'inner_MMA': int(float(L[x]['inner_MMA']))} for x in ne},
}
(HERE / 'evidence/legacy_performance.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({'inversion': inv, 'fits': fits}, indent=1))
