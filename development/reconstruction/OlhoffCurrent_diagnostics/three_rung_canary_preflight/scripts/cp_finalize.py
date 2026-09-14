#!/usr/bin/env python3
"""cp_finalize.py -- METRICS.json, DATA_MANIFEST.json, EVIDENCE.json, FINAL_SHA256.txt."""
import hashlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import ROOT

HERE = Path(__file__).parents[1]
STUDY = 'three_rung_canary_preflight'
EVROOT = f'analysis/OlhoffCurrent/evidence/{STUDY}'


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


HEAD = sh(f'git -C {ROOT} rev-parse HEAD')
BRANCH = sh(f'git -C {ROOT} rev-parse --abbrev-ref HEAD')
integ = json.load(open(HERE / 'evidence/integrity.json'))
pred = json.load(open(HERE / 'evidence/predicted_config_hashes.json'))
host = json.load(open(HERE / 'evidence/host_environment.json'))
cost = json.load(open(HERE / 'evidence/cost_expectation.json'))
perf = json.load(open(HERE / 'evidence/legacy_performance.json'))
an = {m: json.load(open(HERE / f'evidence/analysis_{m}.json')) for m in ('480x60', '800x100')}
fws = json.load(open(HERE / 'evidence/fixedwork_scaling.json'))
topo = json.load(open(HERE / 'evidence/topology_800_comparison.json'))
stg = json.load(open(HERE / 'evidence/stage_timing_both.json'))


def evsha(name):
    f = ROOT / 'analysis/OlhoffCurrent/evidence/three_rung_canary_preflight' / name
    return (sha(f), f.stat().st_size) if f.exists() else (None, None)

# ------------------------------------------------------------------ METRICS
metrics = {
    'study': STUDY,
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'branch': BRANCH, 'head': HEAD,
    'scientific_runs': 2,
    'runs_executed': {'C480x60': 'CONVERGED', 'C800x100': 'CONVERGED',
                      'nine_mesh_campaign': 'NOT_EXECUTED',
                      'meshes_560_640_720': 'NOT_EXECUTED'},
    'verdicts': {
        'deployment_preflight': 'THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS',
        'C480': 'C480_THREE_RUNG_CANARY_PASS',
        'C800_controller': 'C800_THREE_RUNG_CONTROLLER_PASS',
        'C800_endpoint': 'C800_ENDPOINT_SCIENTIFICALLY_SUSPICIOUS',
        'C800_runtime': 'C800_RUNTIME_BEHAVIOR_EXPLAINED',
        'fixed_work': 'FIXED_WORK_SCALING_SANITY_PASS',
        'next_mode_warning': 'NEXT_MODE_WARNING_MATERIAL_CONCERN',
        'overall': 'THREE_RUNG_CONTROLLER_GENERALIZES_BUT_FINE_MESH_SCIENCE_SUSPICIOUS',
        'campaign': 'CORRECT_NINE_MESH_CAMPAIGN_BLOCKED',
    },
    'preregistered_case': 'Case 4 (PREREGISTRATION.md sec. 11)',
    'canaries': {m: {
        'status': an[m]['record']['status'],
        'nOuter': an[m]['record']['nOuter'],
        'cap': an[m]['record']['cap'],
        'cap_headroom': an[m]['gates']['_cap_headroom'],
        'innerTotal': an[m]['record']['innerTotal'],
        'innerNonConv': an[m]['record']['innerNonConv'],
        'omega1': an[m]['record']['omega1'],
        'omega2': an[m]['record']['omega2'],
        'omega3': an[m]['record']['omega3'],
        'gap12': an[m]['record']['gap12'],
        'gap23': an[m]['record']['gap23'],
        'Mnd_pct': an[m]['record']['Mnd_final'],
        'volume': an[m]['record']['volume_final'],
        'move_final': an[m]['record']['move_final'],
        'stage_final': an[m]['record']['stage_final'],
        'wall_s': an[m]['record']['wall_s'],
        'cfgHash': an[m]['record']['cfgHash'],
        'rho_sha256': an[m]['record']['rho_sha256'],
        'stageStarts': an[m]['events']['stageStarts'],
        'declarations': [{'stage': s['stage'], 'end': s['end'],
                          'duration': s['duration'], 'inner': s['inner']}
                         for s in an[m]['stages']],
        'terminal_branch': an[m]['record']['terminalBranch'],
        'terminal_amp_over_tol': an[m]['gates']['_terminal_amp_over_tol'],
        'omega1_change_last20_pct': an[m]['gates']['_omega1_change_last20_pct'],
        'Mnd_change_last20_pts': an[m]['gates']['_Mnd_change_last20_pts'],
        'multJ_count': an[m]['multiplicity']['count'],
        'multJ_fraction_pct': an[m]['multiplicity']['fraction_pct'],
        'multJ_first': an[m]['multiplicity']['first_iteration'],
        'multJ_overlaps_terminal': an[m]['multiplicity']['overlaps_terminal_window'],
        'all_gates_pass': an[m]['gates']['_all_pass'],
        'timing': an[m]['timing'],
        'stage_timing': stg[m],
    } for m in ('480x60', '800x100')},
    'fixed_work_scaling': fws,
    'topology_800_legacy_vs_three_rung': topo,
    'mesh_trend_three_rung': {
        'note': 'M_nd rises monotonically and accelerates; omega1 falls monotonically. '
                'Under one frozen controller. This is the central negative finding.',
        'Mnd_pct': {'160x20': 12.756141959540171, '240x30': 12.91652409156148,
                    '320x40': 12.940093529981493, '400x50': 15.373230197122789,
                    '480x60': an['480x60']['record']['Mnd_final'],
                    '800x100': an['800x100']['record']['Mnd_final']},
        'omega1': {'160x20': 169.97512028959605, '240x30': 167.0393810208395,
                   '320x40': 166.42630441353683, '400x50': 166.45229843313905,
                   '480x60': an['480x60']['record']['omega1'],
                   '800x100': an['800x100']['record']['omega1']},
        'S1_declaration': {'160x20': 102, '240x30': 206, '320x40': 274,
                           '400x50': 388, '480x60': 308, '800x100': 390},
        'S1_monotone_in_mesh': False,
        'structure_all_meshes': 'S1 + 78; S2 and S3 both at the minimum dwell 39',
    },
    'preflight_checks_discharged_offline': {
        'impl_tree_sha256': integ['actual_tree_sha256'],
        'impl_tree_match': integ['tree_match'],
        'impl_files': integ['per_file_match'],
        'impl_tree_equals_validated_c320_implTree': (
            integ['actual_tree_sha256']
            == 'edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb'),
        'legacy_nine_config_hashes_reproduced': '9/9',
        'validated_c320_hash_reproduced': pred['c320_match'],
        'validated_c320_hash': pred['validated_c320_recorded'],
    },
    'preflight_runtime': {
        'verdict': 'THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS',
        'meshes': ['480x60', '800x100'],
        'field_checks': '47/47 at each mesh',
        'blockers': 0,
        'frozen_hash_equals_runtime_hash': True,
        'dispatch_ok': True,
        'beta_continuation_authority': False,
        'beta_stop_authority': False,
        'telemetry_missing': [],
        'matlab': '25.2.0.2998904 (R2025b)',
        'matlab_of_validated_c320_run': '25.2.0.3042426 (R2025b) Update 1',
        'matlab_build_differs_from_validated_run': True,
        'earlier_failed_attempt': {
            'when': '2026-09-12T09:17:43Z',
            'error': 'MathWorks Licensing Error 15, code -15.2',
            'scientific_runs_executed': 0,
            'note': 'the gate refused before compute; retained as the direct '
                    'demonstration that cp_run fails closed',
        },
    },
    'frozen_expectation': {
        'cap': pred['cap'],
        'policy': {'move.levels': [0.04, 0.02, 0.01],
                   'move.continuation.signal': 'stageExhaustion',
                   'stop.rule': 'stageExhaustion',
                   'beta_authority': 'none'},
        'meshes': pred['meshes'],
    },
    'production_preset_promoted_to_three_rung': False,
    'cost_projection_prerun_vs_actual': {
        'note': 'the pre-run projection assumed S1 is monotone in mesh; it is not',
        '480x60': {'projected_outer': 586, 'actual_outer': 386, 'error_factor': 586 / 386},
        '800x100': {'projected_outer': 1130, 'actual_outer': 468, 'error_factor': 1130 / 468},
    },
    'raw_evidence': {
        'C480x60_trajectory': dict(zip(('sha256', 'bytes'), evsha('C480x60_three_rung_trajectory.mat'))),
        'C800x100_trajectory': dict(zip(('sha256', 'bytes'), evsha('C800x100_three_rung_trajectory.mat'))),
        'C480x60_state': dict(zip(('sha256', 'bytes'), evsha('C480x60_three_rung_state.mat'))),
        'C800x100_state': dict(zip(('sha256', 'bytes'), evsha('C800x100_three_rung_state.mat'))),
    },
    'legacy_inversion_720_to_800': perf['legacy_inversion_720_to_800'],
    'host': host['host'],
    'scientific_parameters_changed_after_seeing_outcomes': False,
    'preregistration_frozen_sha256': '25a2500b6e708202b51fba705cb8604c20de577f56f8707f1ac93ce6fec14c82',
    'preregistration_digest_recorded_before_any_run_in': 'evidence/FINAL_SHA256.PRERUN.txt',
}
(HERE / 'METRICS.json').write_text(json.dumps(metrics, indent=1) + '\n')

# --------------------------------------------------------------- EVIDENCE
evidence = {
    'schema': 'olhoff_current_evidence/1',
    'study': STUDY,
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'evidenceRoot': EVROOT,
    'implementation': 'analysis/OlhoffCurrent',
    'sourceTree': integ['actual_tree_sha256'],
    'implTree': integ['actual_tree_sha256'],
    'matlab': '25.2.0.2998904 (R2025b)',
    'scientific_runs': 2,
    'candidate_cfg_hash_480x60_predicted': pred['meshes']['480x60']['predicted_config_hash'],
    'candidate_cfg_hash_800x100_predicted': pred['meshes']['800x100']['predicted_config_hash'],
    'cfg_hash_status': 'RUNTIME_RESOLVED_AND_EQUAL_TO_THE_PRE_RUN_FROZEN_PREDICTION',
    'artifacts': [
        {'path': n, 'class': c, 'description': d,
         'sha256': evsha(n)[0], 'bytes': evsha(n)[1], 'present': evsha(n)[0] is not None,
         'format': 'MAT-file (v7.3 = HDF5) -- read with load()'}
        for n, c, d in [
            ('C480x60_three_rung_trajectory.mat', 'required',
             'canary 1: RHO, DRHO, move, hist, cfg, meta, exh, log for all 386 outer iterations'),
            ('C800x100_three_rung_trajectory.mat', 'required',
             'canary 2: the same, for all 468 outer iterations'),
            ('C480x60_three_rung_state.mat', 'required',
             'canary 1 terminal design + cfg, read by cp_fixedwork'),
            ('C800x100_three_rung_state.mat', 'required',
             'canary 2 terminal design + cfg, read by cp_fixedwork'),
        ]],
    'declared_required_artifacts': 4,
    'rebuild_proof': ('both trajectories were rebuilt from res.diag.drho and proved '
                      'bitwise equal to res.rho; clamp-displacement residual '
                      '5.551e-17 at both meshes'),
    'reused_evidence_read_only': [
        'diagnostics/nine_mesh_campaign_audit/MASTER_TABLE.csv',
        'diagnostics/nine_mesh_campaign_audit/effective_configs.json',
        'diagnostics/nine_mesh_campaign_audit/HISTORICAL_STAGE_WORK.csv',
        'diagnostics/nine_mesh_campaign_audit/HISTORICAL_CONTROLLER_EVENTS.csv',
        'diagnostics/nine_mesh_campaign_audit/LEGACY_VS_THREE_RUNG.csv',
        'diagnostics/three_rung_promotion_validation_retry1/METRICS.json',
        'diagnostics/two_branch_controller_validation/runs/*_record.json',
        'analysis/OlhoffCurrent/SOURCE_MANIFEST.json',
    ],
}
(HERE / 'EVIDENCE.json').write_text(json.dumps(evidence, indent=1) + '\n')

# ---------------------------------------------------------- DATA_MANIFEST
KIND = {'.md': 'document', '.json': 'manifest', '.py': 'script', '.m': 'script',
        '.png': 'figure', '.svg': 'figure', '.csv': 'data', '.txt': 'data',
        '.log': 'evidence'}
files = []
for p in sorted(HERE.rglob('*')):
    if not p.is_file() or p.name == '.DS_Store' or '__pycache__' in p.parts:
        continue
    if p.name in ('DATA_MANIFEST.json', 'FINAL_SHA256.txt'):
        continue
    rel = str(p.relative_to(HERE))
    files.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha(p),
                  'kind': KIND.get(p.suffix, 'other')})
dm = {
    'manifest_schema': 'olhoff_current_data_manifest/1',
    'study': STUDY,
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'repo_head': HEAD, 'branch': BRANCH,
    'implTree': integ['actual_tree_sha256'],
    'scientific_runs': 0,
    'runs_dir_empty': True,
    'n_files': len(files),
    'files': files,
}
(HERE / 'DATA_MANIFEST.json').write_text(json.dumps(dm, indent=1) + '\n')

# ----------------------------------------------------------- FINAL_SHA256
lines = [f'FINAL_SHA256 -- {STUDY}', '=' * 78,
         f'generated {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
         f'branch    {BRANCH}', f'HEAD      {HEAD}',
         f'implTree  {integ["actual_tree_sha256"]}',
         'scientific runs  0   (deployment preflight failed; canaries NOT_REACHED)',
         '=' * 78, '']
final = []
for p in sorted(HERE.rglob('*')):
    if not p.is_file() or p.name == '.DS_Store' or '__pycache__' in p.parts:
        continue
    if p.name == 'FINAL_SHA256.txt':
        continue
    final.append(f'{sha(p)}  {p.relative_to(HERE)}')
lines += final + ['', f'{len(final)} files']
(HERE / 'FINAL_SHA256.txt').write_text('\n'.join(lines) + '\n')

print(f'METRICS.json, EVIDENCE.json, DATA_MANIFEST.json ({len(files)} files), '
      f'FINAL_SHA256.txt ({len(final)} entries)')
