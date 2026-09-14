#!/usr/bin/env python3
"""fi_finalize.py -- METRICS.json, DATA_MANIFEST.json, EVIDENCE.json, FINAL_SHA256.txt"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).parents[1]
ROOT = HERE.parents[3]
STUDY = 'filtered_subproblem_integrability_audit'
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
J = lambda n: json.load(open(HERE / 'evaluations' / n))

st = J('state_identity.json'); kkt = J('inner_kkt.json'); kr = J('inner_kkt_refined.json')
conv = J('inner_convergence.json'); sym = J('jacobian_symmetry.json')
loops = J('closed_loops.json'); mixed = J('mixed_partials.json')
filt = J('filter_operator.json'); dec = J('asymmetry_decomposition.json')
av = J('analytic_verification.json'); cf = J('counterfactual_operators.json')
byc = J('residual_by_class.json'); adm = J('loop_admissibility.json')

metrics = {
    'study': STUDY,
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'branch': sh(f'git -C {ROOT} rev-parse --abbrev-ref HEAD'),
    'head': sh(f'git -C {ROOT} rev-parse HEAD'),
    'preregistration_sha256': sha(HERE / 'AUDIT_PREREGISTRATION.md'),
    'locks': {'optimization_runs': 0, 'accepted_density_updates': 0,
              'continuation': 0, 'controller_transitions': 0,
              'production_files_modified': 0,
              'other_mesh_evaluations': 0,
              'projection_runs': 0, 'p_continuation_runs': 0},
    'verdicts': {
        'state_identity': st['verdict'],
        'inner_mma_kkt': kkt['verdict'],
        'filtered_field': 'FILTERED_FIELD_LOCALLY_NONCONSERVATIVE',
        'effective_objective': 'EFFECTIVE_SCALAR_OBJECTIVE_NOT_IDENTIFIED',
        'surrogate_vs_physical': 'SURROGATE_PHYSICAL_KKT_MISMATCH',
        'grayness': 'SURROGATE_MISMATCH_PARTIALLY_EXPLAINS_NONSTATIONARY_GRAYNESS',
        'next_action': 'INNER_MMA_CERTIFICATION_FAILURE_REQUIRES_RESOLUTION',
        'performance_campaign': 'PERFORMANCE_CAMPAIGN_STILL_BLOCKED',
    },
    'state': {k: st[k] for k in ('rho_sha256', 'rho385_sha256', 'drho386_sha256',
                                 'cfgHash', 'implTree', 'nOuter', 'stage', 'move',
                                 'N', 'multJ', 'nInner_final', 'gap12_at_rho386',
                                 'Mnd_386', 'volume_386')},
    'subproblem_reproduction': kkt['reproduction'],
    'inner_kkt': {
        'retained_exact_dual': kkt['retained_exact_dual'],
        'dual_class': kkt['dual_class'],
        'lam': kkt['kkt_production']['dual']['lam'],
        'fval': kkt['kkt_production']['primal']['fval'],
        'primal_max_fval': kkt['kkt_production']['primal']['max_fval'],
        'max_abs_lam_f': kkt['kkt_production']['complementarity']['max_abs_lam_f'],
        'preregistered_projected_normRMS': {
            'production': kr['production']['without_box_multipliers']['drho_norm_rms'],
            'certification_500': kr['certified']['without_box_multipliers']['drho_norm_rms'],
            'extended_5000': conv['kkt_extended']['nobox_norm_rms']},
        'exact_with_box_multipliers_normRMS': {
            'production': kr['production']['exact_mma_stationarity']['drho_norm_rms'],
            'certification_500': kr['certified']['exact_mma_stationarity']['drho_norm_rms'],
            'extended_5000': conv['kkt_extended']['exact_norm_rms']},
        'preregistration_error_disclosed': (
            'bound classification tolerance 1e-12 finds 0 active bounds against an '
            'interior-point solver, so the preregistered statistic omits -xsi+eta; '
            'both statistics cross the FAIL bar of 0.1'),
        'residual_by_class': {c['class']: c['kkt_norm_rms'] for c in byc},
    },
    'inner_convergence': {
        'converged': conv['converged'], 'iterations': conv['nInner'],
        'relStep': conv['relStep'],
        'maxAbsDrho_over_move': {
            'production_19': kr['production']['max_abs_drho'] / 0.01,
            'certification_500': kr['certified']['max_abs_drho'] / 0.01,
            'extended_5000': conv['maxAbsDrho']['final_over_move']},
        'extended_over_production': conv['maxAbsDrho']['final_over_production'],
    },
    'integrability': {
        'jacobian_median_by_delta': sym['median_by_delta'],
        'jacobian_median_cols': sym['median_cols'],
        'loop_median_by_amp': loops['median_by_amp'],
        'loop_median_cols': loops['median_cols'],
        'loop_exponent_filtered': loops['median_exponent_filt'],
        'loop_exponent_physical': loops['median_exponent_phys'],
        'loop_reversal_relative_sum': loops['reversal']['relative_sum'],
        'mixed_fro_skew_ratio': {'filtered': mixed['filtered']['fro_skew_ratio'],
                                 'physical': mixed['physical']['fro_skew_ratio']},
        'multiplicity_screen': sym['screen'],
    },
    'filter_operator': {
        'A_skew_ratio': filt['A']['skew_ratio'],
        'A_rowsum_range': [filt['A']['rowsum_min'], filt['A']['rowsum_max']],
        'H_symmetric': filt['H_symmetric'],
        'max_guard_active_elements': filt['max_guard_active'],
        'symmetry_condition': filt['symmetry_condition'],
        'amplification_by_class': {c['class']: c['gFilt_over_gPhys'] for c in byc},
    },
    'mechanism': {
        'decomposition': av['claim'],
        'verified_max_relative_error': av['max_relerr'],
        'median_S1_share_rho_weighting': dec['median_S1_share'],
        'median_S2_share_A_times_Hess': dec['median_S2_share'],
        'counterfactual_operators': cf['summary'],
    },
    'disclosed_imperfections': [
        'preregistered bound tolerance mis-specified for an interior-point solver (both statistics reported)',
        f"one loop corner leaves the box by {adm['max_below_by']:.2e} "
        f"({100*adm['max_relative_excursion']:.2f}% of rhomin) on the lowest-asymmetry pair",
        'the 30x30 submatrix test cannot separate the two skew terms; two other tests do',
        'a first extended-certification attempt was aborted for memory and produced no result',
    ],
    'host': {'matlab': '25.2.0.2998904 (R2025b)', 'cpu': 'Apple M1 Max', 'threads': 1},
}
(HERE / 'METRICS.json').write_text(json.dumps(metrics, indent=1) + '\n')

evidence = {
    'schema': 'olhoff_current_evidence/1',
    'study': STUDY,
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'evidenceRoot': f'analysis/OlhoffCurrent/diagnostics/{STUDY}/evaluations',
    'implementation': 'analysis/OlhoffCurrent',
    'implTree': st['implTree'],
    'matlab': '25.2.0.2998904 (R2025b)',
    'scientific_runs': 0,
    'density_updates': 0,
    'source_state': {
        'study': 'three_rung_canary_preflight',
        'trajectory': 'analysis/OlhoffCurrent/evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat',
        'rho386_sha256': st['rho_sha256'],
        'note': 'read-only input; this audit produced no trajectory of its own'},
    'artifacts': [],
}
for p in sorted((HERE / 'evaluations').glob('*')):
    if p.is_file():
        evidence['artifacts'].append({
            'path': p.name, 'class': 'required' if p.suffix == '.json' else 'optional',
            'bytes': p.stat().st_size, 'sha256': sha(p), 'present': True,
            'description': 'audit evaluation output (derived, regenerable from scripts/)'})
evidence['declared_required_artifacts'] = sum(
    1 for a in evidence['artifacts'] if a['class'] == 'required')
(HERE / 'EVIDENCE.json').write_text(json.dumps(evidence, indent=1) + '\n')

KIND = {'.md': 'document', '.json': 'manifest', '.py': 'script', '.m': 'script',
        '.png': 'figure', '.svg': 'figure', '.mat': 'evaluation', '.txt': 'data'}
files = []
for p in sorted(HERE.rglob('*')):
    if not p.is_file() or p.name == '.DS_Store' or '__pycache__' in p.parts:
        continue
    if p.name in ('DATA_MANIFEST.json', 'FINAL_SHA256.txt'):
        continue
    files.append({'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size,
                  'sha256': sha(p), 'kind': KIND.get(p.suffix, 'other')})
(HERE / 'DATA_MANIFEST.json').write_text(json.dumps({
    'manifest_schema': 'olhoff_current_data_manifest/1', 'study': STUDY,
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'repo_head': metrics['head'], 'branch': metrics['branch'],
    'implTree': st['implTree'], 'optimization_runs': 0, 'density_updates': 0,
    'n_files': len(files), 'files': files}, indent=1) + '\n')

lines = [f'FINAL_SHA256 -- {STUDY}', '=' * 78,
         f'generated {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
         f'branch    {metrics["branch"]}', f'HEAD      {metrics["head"]}',
         f'implTree  {st["implTree"]}',
         f'prereg    {metrics["preregistration_sha256"]}',
         'optimization runs 0   density updates 0', '=' * 78, '']
final = [f'{sha(p)}  {p.relative_to(HERE)}' for p in sorted(HERE.rglob('*'))
         if p.is_file() and p.name != '.DS_Store' and '__pycache__' not in p.parts
         and p.name != 'FINAL_SHA256.txt']
(HERE / 'FINAL_SHA256.txt').write_text('\n'.join(lines + final + ['', f'{len(final)} files']) + '\n')
print(f'METRICS.json, EVIDENCE.json ({evidence["declared_required_artifacts"]} required), '
      f'DATA_MANIFEST.json ({len(files)} files), FINAL_SHA256.txt ({len(final)})')
