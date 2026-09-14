#!/usr/bin/env python3
"""METRICS.json, EVIDENCE.json, DATA_MANIFEST.json, FINAL_SHA256.txt (written last)."""
import json, os, hashlib
from sd_common import AUDIT, EVAL, C480, S480, OC, file_sha256, jdump

VERDICTS = ['OLHOFF_SOURCE_COMMIT_VERIFIED', 'OLHOFFCURRENT_TARGET_IDENTITY_PASS', 'SOURCE_SWEEP_EVIDENCE_PASS',
            'SOURCE_TARGET_DIFFERENT_SCIENTIFIC_FORMULATION', 'FIRST_DIVERGENCE_OUTER_BOX_CONTROLLER',
            'SOURCE_SUCCESS_PRIMARILY_FORMULATION', 'PRESERVE_OLD_OLHOFFCURRENT_PRESET',
            'SOURCE_METHOD_REQUIRES_DISTINCT_NAMED_PRESET', 'OLHOFFCURRENT_MIGRATION_READY_WITH_NAMED_FORMULATION_SPLIT']


def J(name):
    return json.loads((EVAL / name).read_text())


def main():
    A = J('trajectory_analysis.json'); S = J('same_state_comparison.json'); L = J('lowdensity_kkt.json')
    V = J('sweep_verification.json'); T = J('target_identity.json'); M1v = J('m1_run/M1_verification.json')
    F = J('file_map.json'); C = J('config_comparison.json'); FI = J('final_integrity.json')
    metrics = dict(
        verdicts=VERDICTS, supporting=['PREFIX_BITWISE_PASS', 'HEURISTIC_STOP'], next_action='C',
        preregistration_sha256='ae998b0ff10bcbd5bde552e284626ae43929cc459885577ca2dbfc7f053dc28f',
        amendment1_sha256='9b90de7be7cd54e3c19c982eda3df759e7851c01909de7a8ae121a1c45bbcb67',
        source=dict(commit='6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7', full_archive_sha256='527b56d9f708da8466bc0422b721d697a3a38f3b5c4ce1f60d864c5fad6d29c0',
                    subset_archive_sha256='0dd479ff5d2ea8580497f447fa49d86639b8b18ac6510df2a9ee20423746424a',
                    snapshot_tree_sha256=J('source_snapshot_manifest.json')['snapshot_tree_sha256'], snapshot_files=405,
                    worktree_dirty_external=True, worktree_diff_sha256=FI['source_worktree_diff_sha256'].split()[0]),
        target=dict(head=FI['target_head'], impl_tree=FI['impl_tree'], impl_manifest_ok=bool(FI['impl_manifest_ok']),
                    currentness=T['currentness_state'], production_cfgHash_480=T['production_cfgHash_480'], c480_cfgHash_match=T['c480_cfgHash_match']),
        sweep_verification=dict(runs=len(V), table_pass=sum(r['table_pass'] for r in V), natural_stop_consistent=sum(r['natural_stop_consistent'] for r in V),
                                summary_match=sum(r['summary_match'] for r in V), max_rel_diff=max(r['table_max_rel_diff'] for r in V),
                                spike_events_total=sum(r['spike_events'] for r in V)),
        m1=dict(status='CONVERGED', nOuter=64, wall_s=395.7, final_native_omega1=34.3571058017119, final_pedersen_omega1=163.554052829928,
                kstar=M1v['kstar'], prefix=M1v['prefix_verdict'], spikes=A['spikes']['M1'], box_floor_fraction_final=A['M1_box_floor_final']),
        decomposition=A['decomposition'], first_crossings=A['first_crossings'], grayness_divergence_onset=A['grayness_divergence_onset'],
        stopping=A['stopping'], milestones=A['milestones'],
        same_state=dict(states=list(S['impl_identity_T_vs_S1'].keys()), impl_identity_all_bitwise=all(v['agree_all_1e-9'] and v['K_bitwise'] and v['raw_bitwise'] and v['filt_bitwise'] and v['rows_bitwise'] and v['inner004_bitwise'] for v in S['impl_identity_T_vs_S1'].values()),
                        validations=S['validation'], rho0_first_step=S['rho0_native_first_step']),
        kkt=L['kkt'], kkt_validation=L['kkt_validation'],
        file_map_counts=F['counts'], plan_unchanged_claim_all_identical=F['plan_unchanged_claim_all_identical'],
        config=dict(leaves=C['n_leaves'], identical_all_four=C['n_identical'], differing=len(C['differing'])),
        budget=dict(source_runs=1, target_runs=0, other_mesh_runs=0, socp_runs=0, rho_updates_offline=0))
    jdump(metrics, AUDIT / 'METRICS.json')

    ev = lambda *p: {str(x): file_sha256(AUDIT / x) for x in p}
    evidence = {
        'OLHOFF_SOURCE_COMMIT_VERIFIED': ev('SOURCE_IDENTITY.md', 'evaluations/source_snapshot_manifest.json', 'scripts/sd_snapshot_manifest.py'),
        'OLHOFFCURRENT_TARGET_IDENTITY_PASS': ev('TARGET_IDENTITY.md', 'evaluations/target_identity.json', 'evaluations/final_integrity.json'),
        'SOURCE_SWEEP_EVIDENCE_PASS': ev('evaluations/sweep_verification.json', 'scripts/sd_verify_sweeps.m'),
        'SOURCE_TARGET_DIFFERENT_SCIENTIFIC_FORMULATION': ev('FORMULATION_COMPARISON.md', 'evaluations/m1_run/M1_verification.json', 'evaluations/same_state_comparison.json', 'evaluations/effective_configs_480.json'),
        'FIRST_DIVERGENCE_OUTER_BOX_CONTROLLER': ev('FIRST_DIVERGENCE.md', 'evaluations/same_state_comparison.json', 'evaluations/config_comparison.json'),
        'SOURCE_SUCCESS_PRIMARILY_FORMULATION': ev('CAUSAL_ATTRIBUTION.md', 'evaluations/trajectory_analysis.json', 'evaluations/lowdensity_kkt.json', 'evaluations/retained_pairs.json', 'evaluations/m1_run/M1_480x60_res.mat'),
        'PRESERVE_OLD_OLHOFFCURRENT_PRESET': ev('MIGRATION_CLASSIFICATION.md'),
        'SOURCE_METHOD_REQUIRES_DISTINCT_NAMED_PRESET': ev('MIGRATION_CLASSIFICATION.md', 'FORMULATION_COMPARISON.md'),
        'OLHOFFCURRENT_MIGRATION_READY_WITH_NAMED_FORMULATION_SPLIT': ev('MIGRATION_GATE.md'),
        'PREFIX_BITWISE_PASS': ev('evaluations/m1_run/M1_verification.json', 'scripts/sd_m1_post.m'),
        'HEURISTIC_STOP': ev('STOPPING_COMPARISON.md', 'evaluations/lowdensity_kkt.json'),
        'retained_inputs': {'C480_trajectory': {'path': str(C480), 'sha256': file_sha256(C480)},
                            'S480_res': {'path': str(S480), 'sha256': file_sha256(S480)},
                            'prior_stationarity_definition': {'path': str(OC / 'diagnostics/gray_kkt_forensic_audit/scripts/stationarity.py'), 'sha256': file_sha256(OC / 'diagnostics/gray_kkt_forensic_audit/scripts/stationarity.py')},
                            'prior_geometry_definition': {'path': str(OC / 'diagnostics/gray_kkt_forensic_audit/scripts/geometry.py'), 'sha256': file_sha256(OC / 'diagnostics/gray_kkt_forensic_audit/scripts/geometry.py')},
                            'prior_telemetry_definition': {'path': str(OC / 'diagnostics/dynamical_regime/scripts/dr_telemetry.m'), 'sha256': file_sha256(OC / 'diagnostics/dynamical_regime/scripts/dr_telemetry.m')}},
    }
    jdump(evidence, AUDIT / 'EVIDENCE.json')

    files = []
    for root, dirs, names in os.walk(AUDIT):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for n in names:
            p = os.path.join(root, n); rel = os.path.relpath(p, AUDIT)
            if rel in ('FINAL_SHA256.txt', 'DATA_MANIFEST.json') or n == '.DS_Store':
                continue
            files.append(dict(path=rel, bytes=os.path.getsize(p), sha256=file_sha256(p)))
    files.sort(key=lambda f: f['path'])
    jdump(dict(generated_by='scripts/sd_finalize.py', n_files=len(files),
               large_binaries=[f for f in files if f['bytes'] >= 5_000_000],
               external_inputs=evidence['retained_inputs'], files=files), AUDIT / 'DATA_MANIFEST.json')

    lines = []
    for root, dirs, names in os.walk(AUDIT):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for n in names:
            p = os.path.join(root, n); rel = os.path.relpath(p, AUDIT)
            if rel == 'FINAL_SHA256.txt' or n == '.DS_Store':
                continue
            lines.append(f'{file_sha256(p)}  {rel}')
    lines.sort(key=lambda s: s.split('  ', 1)[1])
    (AUDIT / 'FINAL_SHA256.txt').write_text('\n'.join(lines) + '\n')
    print('finalized', len(files), 'files; FINAL lines', len(lines))


if __name__ == '__main__':
    main()
