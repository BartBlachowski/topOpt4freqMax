"""Part 1 (Python half): control evidence identity.  READ-ONLY.

Verifies the retained C480 three-rung canary against its own manifests and
proves the copied geometry routine reproduces the gray/KKT audit exactly.
Combines with evaluations/control_identity_matlab.json into the verdict.
"""
from cs_common import *


def main():
    m = json.loads((EV / 'control_identity_matlab.json').read_text())
    o = {'matlab_checks': m['checks'], 'matlab_all_pass': m['all_matlab_checks_pass']}
    ev = json.loads((CANARY / 'EVIDENCE.json').read_text())
    art = {a['path']: a for a in ev['artifacts']}
    o['trajectory_sha256'] = sha256_file(CONTROL_TRAJ)
    o['trajectory_sha256_declared'] = art['C480x60_three_rung_trajectory.mat']['sha256']
    state = CONTROL_TRAJ.parent / 'C480x60_three_rung_state.mat'
    o['state_sha256'] = sha256_file(state)
    o['state_sha256_declared'] = art['C480x60_three_rung_state.mat']['sha256']

    # canary FINAL_SHA256 entries for the run files and reports
    listed = {}
    for line in (CANARY / 'FINAL_SHA256.txt').read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            listed[parts[1]] = parts[0]
    header = (CANARY / 'FINAL_SHA256.txt').read_text().splitlines()[:8]
    o['canary_final_sha256_header'] = header
    checked = {}
    for rel in ['runs/C480x60_three_rung_iterations.csv', 'runs/C480x60_three_rung_record.json',
                'runs/C480x60_three_rung_supplement.csv', 'C480_REPORT.md', 'PREREGISTRATION.md',
                'EFFECTIVE_CONFIG.json', 'EVIDENCE.json', 'evidence/analysis_480x60.json']:
        p = CANARY / rel
        checked[rel] = {'listed': listed.get(rel), 'actual': sha256_file(p) if p.exists() else None}
        checked[rel]['match'] = checked[rel]['listed'] is not None and checked[rel]['listed'] == checked[rel]['actual']
    o['canary_manifest_checks'] = checked

    # geometry: copied routine vs retained audit values
    with h5py.File(CONTROL_TRAJ, 'r') as f:
        RHO = f['RHO'][()]
        omega = f['hist']['omega'][()]
    assert RHO.shape == (386, 28800), RHO.shape
    rho = RHO[385]
    o['rho386_sha256'] = sha256_vec(rho)
    geo, _ = geometry_metrics(rho)
    ref = json.loads((GRAYKKT / 'evaluations' / 'geometry.json').read_text())['480']
    keys = ['Mnd_percent', 'gray_fraction', 'mid_fraction', 'gray_area', 'mid_area', 'broad_core_fraction',
            'broad_core_area', 'gray_components_4', 'gray_components_8', 'largest_gray_component_area',
            'max_depth', 'gray_depth_p50_p90_p95_p99', 'gray_interface_distance_p50_p90_p95_p99',
            'rho_quantiles', 'rho_lt_001', 'rho_gt_099', 'components']
    o['geometry_reproduction'] = {k: geo[k] == ref[k] for k in keys}
    o['geometry_reproduction_all_exact'] = all(o['geometry_reproduction'].values())
    o['geometry'] = geo

    # per-iteration grayness vs the audit's trajectory_480.csv
    tr = per_iteration_gray(RHO)
    old = np.loadtxt(GRAYKKT / 'evaluations' / 'trajectory_480.csv', delimiter=',', skiprows=1)
    o['per_iteration_gray_max_abs_diff'] = float(np.max(np.abs(tr - old[:, 1:5])))
    np.savetxt(EV / 'control_per_iteration_gray.csv', np.c_[np.arange(1, 387), tr], delimiter=',',
               header='outer,Mnd_percent,gray_fraction,mid_fraction,broad_core_fraction', comments='')

    mm = m
    o['summary'] = {
        'mesh': mm['mesh'], 'initial_rho_sha256': mm['rho0_sha256'], 'final_rho_sha256': mm['rho386_sha256'],
        'config_hash': mm['cfgHash_stored'], 'impl_tree': mm['meta_implTree'], 'terminal_iteration': mm['nOuter'],
        'terminal_status': mm['status_record'], 'stage_starts': mm['stageStarts'],
        'stage_lengths': mm['stage_lengths'], 'controller_events': mm['controller_events'],
        'final_move': mm['moves_by_stage'][-1], 'omega1': mm['omega1_record'], 'omega2': mm['omega2_record'],
        'gap12': mm['gap12_record'], 'volume': mm['volume'], 'Mnd_percent': geo['Mnd_percent'],
        'gray_fraction': geo['gray_fraction'], 'mid_fraction': geo['mid_fraction'],
        'broad_core_fraction': geo['broad_core_fraction'], 'broad_core_area': geo['broad_core_area'],
        'max_depth': geo['max_depth'], 'total_outer': mm['nOuter'], 'inner_mma_total': mm['innerTotal'],
        'wall_s': mm['wall_record'], 'sum_tInner': mm['sum_tInner'], 'sum_tEig': mm['sum_tEig'],
        'sum_tGrad': mm['sum_tGrad'], 'sum_tOuter': mm['sum_tOuter']}
    checks = {
        'matlab_all': bool(m['all_matlab_checks_pass']),
        'trajectory_hash': o['trajectory_sha256'] == o['trajectory_sha256_declared'] == 'a87546bc391cdc683def34a9f27678884528032f6b140e156349d2e74135ab9b',
        'state_hash': o['state_sha256'] == o['state_sha256_declared'],
        'rho386_python': o['rho386_sha256'] == '0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60',
        'canary_manifest': all(v['match'] for v in checked.values()),
        'geometry_exact': o['geometry_reproduction_all_exact'],
        'per_iteration_gray': o['per_iteration_gray_max_abs_diff'] <= 1e-12,
    }
    o['checks'] = checks
    o['verdict'] = 'C480_CONTROL_EVIDENCE_PASS' if all(checks.values()) else 'C480_CONTROL_EVIDENCE_FAIL'
    dump(EV / 'control_identity.json', o)
    print(json.dumps(checks, indent=1))
    print({k: v for k, v in checked.items() if not v['match']})
    print(o['verdict'])


if __name__ == '__main__':
    main()
