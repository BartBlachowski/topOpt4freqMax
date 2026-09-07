#!/usr/bin/env python3
"""ma4_metrics -- assemble METRICS.json for the 400x50 study."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ma4_scaling as S

OUT = S.OUT
M = {'schema': 'olhoff_move_activity_400/1',
     'study': 'move_activity_400',
     'task': 'third-mesh measurement of design-activity scaling; NOT a controller experiment',
     'preregistration_sha256': '706b8865075f97cb0d5824658fa6e562636d464dba2611580d6c10fdff7716d4',
     'implementation': 'analysis/OlhoffCurrent',
     'production_preset': 'duOlhoffFixedPenaltySensitivityFiltered',
     'source_tree_sha256': 'c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c',
     'config_hash_400x50_production': '044d50a496cb64d43ed4bf75a6a7976a6726cca6feaf65e0b911a57e4593969c',
     'matlab': '25.2.0.2998904 (R2025b)',
     'matlab_prior_studies': '25.2.0.3042426 (R2025b) Update 1',
     'starting_commit': '4ef315b'}

P = S.load(os.path.join(OUT, 'runs/P400_400x50_iterations.csv'))
F = S.load(os.path.join(OUT, 'runs/F400_400x50_iterations.csv'))

def arm(R, name, cap, status, wall, inner, traj, tbytes):
    i, it = S.first_descent(R)
    cm = S.monotone(S.completion(R))
    d = [dict(iter=R['outer'][k], moveFrom=R['move'][k-1], moveTo=R['move'][k])
         for k in range(1, R['n']) if R['move'][k] < R['move'][k-1]]
    return dict(arm=name, mesh=[400, 50], NE=20000, cap=cap, nOuter=R['n'],
                status=status, wall_s=wall, innerTotal=inner,
                descents=d, firstDescentIter=it,
                lastIterAtMove004=(R['outer'][i] if i is not None else None),
                omega1_final=R['omega1'][-1], omega2_final=R['omega2'][-1],
                gap12_final=R['gap12'][-1], volume_final=R['volume'][-1],
                Mnd_final=R['Mnd_pct'][-1], gray_final=R['gray_frac'][-1],
                mid_final=R['mid_frac'][-1],
                trajectory=traj, trajectoryBytes=tbytes)

M['arms'] = [
    arm(P, 'P400', 400, 'CONVERGED', 541.0, int(sum(P['nInner'])),
        'analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat', 36576273),
    arm(F, 'F400', 600, 'CONVERGED', 1633.0, int(sum(F['nInner'])),
        'analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat', 102057786)]

# primary measurement point
i, it = S.first_descent(P)
cm = S.monotone(S.completion(F))
prim = dict(descentIter=it, lastIterAtMove004=P['outer'][i], NE=20000, move=0.04,
            completion_at_descent=cm[i])
for k in ['omega1', 'omega2', 'gap12', 'volume', 'Mnd_pct', 'gray_frac', 'mid_frac',
          'maxAbs', 'rms', 'l2', 'r_rho', 'Neff', 'u_min', 'u_P50', 'u_P75', 'u_P90',
          'u_P95', 'u_P975', 'u_P99', 'u_max', 'u_mean', 'u_RMS',
          'nActive_1e4', 'nActive_epsRMS', 'nActive_1e3', 'nActive_1e2']:
    prim[k] = F[k][i]
for t, c in [('1e-4', 'nActive_1e4'), ('epsRMS', 'nActive_epsRMS'),
             ('1e-3', 'nActive_1e3'), ('1e-2', 'nActive_1e2')]:
    prim[f'activeFrac_{t}'] = F[c][i] / 20000.0
prim['Mnd_fixedMoveEndpoint'] = F['Mnd_pct'][-1]
prim['remaining_absolute'] = F['Mnd_pct'][i] - F['Mnd_pct'][-1]
prim['remaining_relative'] = prim['remaining_absolute'] / F['Mnd_pct'][i]
prim['omega1_fixedMoveEndpoint'] = F['omega1'][-1]
prim['omega1_relChange_to_endpoint'] = abs(F['omega1'][-1] - F['omega1'][i]) / F['omega1'][i]
M['primary_measurement_at_production_first_descent'] = prim

M['production_vs_counterfactual'] = dict(
    P400_Mnd=P['Mnd_pct'][-1], F400_Mnd=F['Mnd_pct'][-1],
    Mnd_relative_reduction=(P['Mnd_pct'][-1]-F['Mnd_pct'][-1])/P['Mnd_pct'][-1],
    P400_omega1=P['omega1'][-1], F400_omega1=F['omega1'][-1],
    omega1_relative_change=(F['omega1'][-1]-P['omega1'][-1])/P['omega1'][-1],
    P400_gray=P['gray_frac'][-1], F400_gray=F['gray_frac'][-1],
    P400_mid=P['mid_frac'][-1], F400_mid=F['mid_frac'][-1],
    note='F400 is NOT a production candidate; it measures what the descent truncates')

M['reconstruction_checks'] = dict(
    volume_max_error='< 1e-12 (asserted at run time, both arms)',
    final_design_bitwise_equals_res_rho=True,
    clamp_inert_max_error=5.551e-17,
    clamp_elements_out_of_bounds=0,
    clamp_note=('zero elements out of 20000 x 139 ever had rho+drho outside [rho_min,1]; '
                'the 5.551e-17 residual is round-off of the subtraction (eps/2 = 1.11e-16), '
                'NOT clamp activation.  Prior studies asserted this in prose; here it is measured.'))

for f, key in [('PREFIX_GATE.json', 'counterfactual_prefix_gate'),
               ('SCALING_ANALYSIS.json', 'scaling'),
               ('SPATIAL_ANALYSIS.json', 'spatial')]:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        M[key] = json.load(open(p))

json.dump(M, open(os.path.join(OUT, 'METRICS.json'), 'w'), indent=2)
print('wrote METRICS.json')
print(f"  P400 M_nd={P['Mnd_pct'][-1]:.4f}%  F400 M_nd={F['Mnd_pct'][-1]:.4f}%  "
      f"reduction {100*M['production_vs_counterfactual']['Mnd_relative_reduction']:.1f}%")
print(f"  omega1 {P['omega1'][-1]:.4f} -> {F['omega1'][-1]:.4f} "
      f"({100*M['production_vs_counterfactual']['omega1_relative_change']:+.3f}%)")
print(f"  remaining at descent: {prim['remaining_absolute']:.3f} pts = "
      f"{100*prim['remaining_relative']:.1f}%")
