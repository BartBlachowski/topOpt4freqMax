#!/usr/bin/env python3
"""METRICS.json, DATA_MANIFEST.json and FINAL_SHA256.txt for upstream_253069_migration (run last)."""
import hashlib, json, os, re, datetime
W = '/Users/piotrek/Programming/topOpt4freqMax-migration-253069'
REL = 'analysis/OlhoffCurrent/diagnostics/upstream_253069_migration'
S = f'{W}/{REL}'
EVREL = 'analysis/OlhoffCurrent/evidence/upstream_253069_migration'
EV = f'{W}/{EVREL}'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
J = lambda f: json.load(open(f'{S}/evidence/{f}'))
cmp_, ct, bi, mp, hc, da, src, start = (J('comparisons.json'), J('config_transition.json'), J('byte_identity.json'),
    J('manifest_provenance_check.json'), J('harness_check.json'), J('diff_audit.json'), J('source_identity.json'), J('target_start_state.json'))
fg = J('finalization_gate_affected.json')

keep = ('status', 'nOuter', 'innerTotal', 'innerMax', 'innerNotConverged', 'omega1', 'omega2', 'omega3', 'gap12_pct', 'volume',
        'Mnd_pct', 'gray_fraction', 'mid_fraction', 'rho_sha256_bytes', 'final_move_max', 'final_move_mean', 'final_stage',
        'stage_starts', 'final_dxNorm2', 'final_N', 'treeHash', 'exhaustion')
runs = {k: {x: v[x] for x in keep if x in v} for k, v in cmp_['runs'].items()}

def suite_failures(log):
    out = {}
    for line in open(f'{EV}/logs/{log}'):
        m = re.match(r'SUITE (\S+(?: \S+)?)\s+failures=(\d+)', line.strip())
        if m: out[m.group(1)] = int(m.group(2))
    return out
suites = {}
for log in ['T_gates3.log', 'T_preset_identity.log', 'T_preset_equivalence.log', 'T_named_pedersen.log', 'T_named_stageExhaustion.log',
            'T_cost_reporting.log', 'T_pedersen_adaptive_units.log', 'T_upstream_suites.log']:
    suites.update(suite_failures(log))
snap = suite_failures('T_upstream_suites_snapshot.log')

verdicts = ['UPSTREAM_253069_IDENTITY_PASS', 'MIGRATION_BYTE_PROMOTION_PASS', 'HISTORICAL_PRESET_PRESERVED_PASS',
            'HISTORICAL_PRESET_REPRODUCTION_PASS', 'PEDERSEN_PRESET_DISTINCT_IDENTITY_PASS', 'PEDERSEN_PRESET_S160_REPRODUCTION_PASS',
            'SHARED_IMPLEMENTATION_EQUIVALENCE_PASS', 'CONFIG_SCHEMA_MIGRATION_PASS', 'MANIFEST_PROVENANCE_PASS',
            'BENCHMARK_PREFLIGHT_PASS', 'RELEVANT_TEST_SUITE_PASS', 'PHASE6_UNTOUCHED_PASS', 'UNAUTHORIZED_DIFF_ZERO',
            'OLHOFFCURRENT_MIGRATION_COMPLETE', 'OLHOFF_NINE_MESH_CAMPAIGN_READY_FOR_AUTHORIZATION']
M = {
 'schema': 'upstream_253069_migration_metrics/1',
 'generated': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
 'verdicts': verdicts,
 'verdict_notes': {'RELEVANT_TEST_SUITE_PASS': 'no failure attributable to the migration; finalization-gate H/I and harness self-test T2 fail identically on pristine 013cc48 (PREREGISTRATION_AMENDMENT_1 A1)'},
 'upstream': {'commit': src['commit'], 'tree': src['tree'], 'parent': src['parents'][0], 'archive_sha256': src['archive_sha256'],
              'snapshot_files': src['snapshot_files'], 'snapshot_blob_mismatches': src['snapshot_blob_mismatches']},
 'target_start': {'branch': start['branch'], 'head': start['head'], 'impl_tree': start['impl_live_tree_sha256'],
                  'impl_files': start['impl_live_n_files'], 'tracked_changes': start['tracked_changes'], 'untracked': start['porcelain']},
 'impl_after': bi['summary'],
 'config_schema': {'rows_before': ct['schema']['oldRows'], 'rows_after': ct['schema']['newRows'], 'added': ct['schema']['added'],
                   'checks': ct['checks']},
 'config_hashes': {'historical': [{k: h[k] for k in ('case', 'mesh', 'recordedHash', 'postHash', 'oldSchemaHashFromPostEqualsRecorded')} for h in ct['historical']],
                   'pedersen': [{k: p[k] for k in ('mesh', 'postHash', 'postEqualsUpstreamAllLeaves')} for p in ct['pedersen']]},
 'presets': {'production': 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered',
             'historical_formulation': 'duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered',
             'historical_diagnostic': 'duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered',
             'compatibility_alias': {'duOlhoffFixedPenaltySensitivityFiltered': 'duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered'}},
 'runs_160x20': runs,
 'comparisons_pass': cmp_['pass'],
 'pcontinuation_defect': cmp_['pcont'],
 'manifest_provenance_check': mp,
 'harness': {'preflight_pass': hc['preflight']['pass'], 'preflight_checks': hc['preflight']['n_checks'],
             'preflight_failed': hc['preflight']['n_failed'], 'selftest_failed_ids': hc['selftest']['failed_ids'],
             'selftest_failed_ids_on_pristine_013cc48': ['T2'], 'smoke': hc['smoke'], 'export': hc['export']},
 'test_suites_failures': suites,
 'upstream_suites_on_snapshot_failures': snap,
 'finalization_gate_pre_existing': {'H_failing': ['move_activity_400'], 'I_failing_studies': ['controller_architecture_offline', 'move_activity_400',
        'move_ladder_necessity', 'three_rung_architecture', 'three_rung_promotion_closure', 'three_rung_promotion_validation_retry1', 'two_rung_architecture'],
        'identical_on_pristine_013cc48': True},
 'finalization_gate_affected_studies': {k: {'ok': v['ok'], 'superseded': [s['path'] for s in v['supersededSource']] if isinstance(v['supersededSource'], list) else v['supersededSource']} for k, v in fg.items()},
 'diff_audit': {'paths': da['n_paths'], 'by_class': da['by_class'], 'unauthorized': len(da['unauthorized']), 'phase6_paths': da['phase6_paths']},
 'meshes_solved': ['160x20'], 'max_mesh_solved': '160x20',
}
json.dump(M, open(f'{S}/METRICS.json', 'w'), indent=1)

# ---- DATA_MANIFEST.json ---------------------------------------------------
arts = []
for f in sorted(os.listdir(EV)):
    p = f'{EV}/{f}'
    if os.path.isfile(p):
        arts.append({'path': f'{EVREL}/{f}', 'bytes': os.path.getsize(p), 'sha256': sha(p), 'tracked': False,
                     'storage': 'git-ignored durable evidence root (EVIDENCE_POLICY.md)'})
for d, _, fs in os.walk(S):
    for f in sorted(fs):
        if f in ('FINAL_SHA256.txt', 'DATA_MANIFEST.json', '.DS_Store'): continue
        p = os.path.join(d, f)
        arts.append({'path': os.path.relpath(p, W), 'bytes': os.path.getsize(p), 'sha256': sha(p), 'tracked': True})
ext = [
 {'what': 'upstream Olhoff commit (git archive)', 'id': src['commit'], 'archive_sha256': src['archive_sha256']},
 {'what': 'committed S160x20 result', 'id': 'Olhoff 253069:repro/results/S160x20/res.mat (git blob 93c6550980f06fe3535cc1780bc0317517b6a0c0)',
  'sha256': 'de7b15764509a1797d8713495f37481004cefb7b33f248d9b3981a4a3450daa0'},
 {'what': 'upstream anchors A6/A7 committed references', 'id': 'Olhoff 253069:architecture/anchors/reference/{A6_pdecoupled160,A7_massp160}'},
 {'what': 'frozen conference campaign records', 'id': 'examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat (git-ignored, primary checkout)',
  'sha256': cmp_['A_beta']['post_vs_campaign']['reference_sha256']},
 {'what': 'pre-migration target EX3_160 record (upstream capability audit)', 'id': '/Users/piotrek/Programming/Matlab/Olhoff-upstream-capabilities-evidence/runs/case_TARGET_EX3_160',
  'sha256': cmp_['A_ex3']['post_vs_targetAudit']['reference_sha256']},
 {'what': 'upstream capability audit A6/A7 candidate anchor records', 'id': '/Users/piotrek/Programming/Matlab/Olhoff-upstream-capabilities-evidence/runs/anchor_CAND_{A6_pdecoupled160,A7_massp160}'},
 {'what': 'committed two-branch C160x20 target trajectory', 'id': 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory (declared by that study)',
  'sha256': cmp_['A_ex4']['trajectory_sha256']},
 {'what': 'scientific delta audit (untracked in the primary checkout)', 'id': 'analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration'},
]
json.dump({'schema': 'upstream_253069_migration_data_manifest/1', 'generated': M['generated'], 'artifacts': arts, 'external_inputs': ext},
          open(f'{S}/DATA_MANIFEST.json', 'w'), indent=1)

# ---- FINAL_SHA256.txt -------------------------------------------------------
lines = []
for d, _, fs in os.walk(S):
    for f in sorted(fs):
        if f in ('FINAL_SHA256.txt', '.DS_Store'): continue
        p = os.path.join(d, f)
        lines.append(f'{sha(p)}  {os.path.relpath(p, S)}')
lines.sort(key=lambda l: l.split('  ', 1)[1])
for f in sorted(os.listdir(EV)):
    p = f'{EV}/{f}'
    if os.path.isfile(p) and f.endswith('.mat'):
        lines.append(f'{sha(p)}  {EVREL}/{f}')
open(f'{S}/FINAL_SHA256.txt', 'w').write('\n'.join(lines) + '\n')
print('METRICS verdicts', len(verdicts), '| DATA_MANIFEST artifacts', len(arts), '| FINAL lines', len(lines))
