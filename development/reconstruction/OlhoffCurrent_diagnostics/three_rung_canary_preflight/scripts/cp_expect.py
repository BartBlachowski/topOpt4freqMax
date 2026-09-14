#!/usr/bin/env python3
"""cp_expect.py -- COST EXPECTATION for the two canaries, from retained data only.

This is a budget projection, NOT an acceptance criterion and NOT a prediction of
any scientific outcome.  The ten acceptance gates in PREREGISTRATION.md sec. 6
are the only things a canary is judged against.

Inputs, all already in the repository:
  * the four VALIDATED three-rung endpoints (160/240/320/400)
  * their per-stage structure (HISTORICAL_STAGE_WORK.csv)
  * the legacy nine-mesh per-outer timings (MASTER_TABLE.csv)
  * the diagnostics-recorder overhead, measured at 400x50 at equal policy
"""
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import ROOT

AUD = ROOT / 'analysis/OlhoffCurrent/diagnostics/nine_mesh_campaign_audit'
HERE = Path(__file__).parents[1]

# ---- the validated three-rung structure --------------------------------
# S1 is the only stage whose length is data-driven; S2 and S3 sit at the
# MINIMUM possible dwell of 39 = (W-1) + P on every validated mesh.
S1 = {3200: 102, 7200: 206, 12800: 274, 20000: 388}
THREE = {   # NE: (outer, inner, omega1, Mnd)
    3200:  (180, 4283, 169.97512028959605, 12.756141959540171),
    7200:  (284, 5506, 167.0393810208395, 12.91652409156148),
    12800: (352, 6498, 166.42630441353683, 12.940093529981493),
    20000: (466, 8848, 166.45229843313905, 15.373230197122789),
}
MIN_DWELL = 39


def loglog_fit(xs, ys):
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    p = sxy / sxx
    c = math.exp(my - p * mx)
    ss_res = sum((b - (math.log(c) + p * a)) ** 2 for a, b in zip(lx, ly))
    ss_tot = sum((b - my) ** 2 for b in ly)
    return c, p, 1 - ss_res / ss_tot


legacy = {}
for r in csv.DictReader(open(AUD / 'MASTER_TABLE.csv')):
    legacy[int(r['NE'])] = r

out = {'basis': 'retained validated three-rung endpoints + legacy nine-mesh timings',
       'not_an_acceptance_criterion': True}

# ---- 1. S1 declaration iteration vs NE ---------------------------------
c, p, r2 = loglog_fit(list(S1), list(S1.values()))
out['S1_fit'] = {'C': c, 'p': p, 'R2_log': r2,
                 'form': 'S1_declaration = C * NE^p',
                 'basis_meshes': [160, 240, 320, 400]}

# ---- 2. inner MMA per outer, validated three-rung -----------------------
ipo = {ne: v[1] / v[0] for ne, v in THREE.items()}
out['inner_per_outer_validated'] = ipo

# ---- 3. the recorder overhead, measured at EQUAL policy and mesh --------
# two_branch_controller_validation arm 'P' is the production/legacy policy WITH
# runtime.diagnostics = true.  The legacy nine-mesh campaign is the same policy
# with diagnostics = false.  They share mesh, preset and formulation, so the
# per-outer difference is the recorder plus host drift -- an UPPER bound on the
# recorder, never a clean isolation, and labelled as such.
out['recorder_overhead'] = {
    'note': ('The canaries MUST run with runtime.diagnostics = true to retain '
             'trajectories; the legacy nine-mesh campaign ran with it false. '
             'Canary wall time is therefore NOT directly comparable to legacy '
             'wall time, and PERFORMANCE_DECOMPOSITION.md states this before '
             'any number is compared.'),
    'legacy_400x50_s_per_outer': float(legacy[20000]['wall_per_outer_s']),
    'diagnostics_on_400x50_four_rung_s_per_outer': 3539.3367536666665 / 505,
    'ratio_upper_bound': (3539.3367536666665 / 505) / float(legacy[20000]['wall_per_outer_s']),
    'confounds': ['different controller (four-rung E vs legacy beta)',
                  'different stage mix, hence different inner counts per outer',
                  'different host session'],
}

# ---- 4. projected canary cost ------------------------------------------
# Per-outer wall is taken from the LEGACY campaign at the same mesh, scaled by
# the ratio of inner-MMA steps per outer, then by the recorder upper bound.
proj = {}
for nx, ny in [(480, 60), (800, 100)]:
    NE = nx * ny
    s1 = c * NE ** p
    outer = s1 + 2 * MIN_DWELL
    ci, pi, _ = loglog_fit(list(ipo), list(ipo.values()))
    inner_po = ci * NE ** pi
    legacy_row = legacy[NE]
    s_per_mma = float(legacy_row['inner_s_per_MMA'])
    eig_po = float(legacy_row['eigen_per_outer_s'])
    grad_po = float(legacy_row['gradient_per_outer_s'])
    per_outer = inner_po * s_per_mma + eig_po + grad_po
    proj[f'{nx}x{ny}'] = {
        'NE': NE,
        'projected_S1_declaration': round(s1),
        'projected_outer_total': round(outer),
        'projected_inner_per_outer': inner_po,
        'projected_inner_total': round(outer * inner_po),
        'legacy_s_per_MMA_step': s_per_mma,
        'projected_s_per_outer_recorder_off': per_outer,
        'projected_wall_s_recorder_off': per_outer * outer,
        'projected_wall_h_recorder_off': per_outer * outer / 3600,
        'projected_wall_h_recorder_on_upper': (per_outer * outer / 3600)
                                              * out['recorder_overhead']['ratio_upper_bound'],
        'cap_1600_headroom': round(1600 - outer),
        'cap_adequate': outer < 1600,
        'RHO_DRHO_bytes_at_projected_outer': int(2 * NE * 8 * round(outer)),
    }
out['projection'] = proj
out['projection_caveat'] = (
    'A four-point log-log fit extrapolated 1.4x (480) and 4x (800) beyond its '
    'range.  It sizes a time budget and checks that cap 1600 is not obviously '
    'too small.  It is not evidence about any canary outcome and no gate reads it.')

(HERE / 'evidence/cost_expectation.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
