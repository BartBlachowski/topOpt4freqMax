#!/usr/bin/env python3
"""fp_finalize.py -- METRICS.json, DATA_MANIFEST.json, EVIDENCE.json, FINAL_SHA256.txt"""
import hashlib, json, subprocess, time
from pathlib import Path
HERE = Path(__file__).parents[1]
ROOT = HERE.parents[3]
STUDY = 'frozen_problem25_reference'
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
J = lambda n: json.load(open(HERE / 'evaluations' / n))

st = J('state_identity.json'); eq = J('socp_equivalence.json'); ref = J('reference_solution.json')
sw = J('coneprog_sweep.json'); cmpj = J('mma_comparison.json'); gc = J('gradient_check.json')
fmE = J('fmincon_main_exact.json'); fmS3 = J('fmincon_S3.json')
fmL = J('fmincon_main.json') if (HERE / 'evaluations' / 'fmincon_main.json').exists() else {'runs': []}
sqp = J('fmincon_sqp.json') if (HERE / 'evaluations' / 'fmincon_sqp.json').exists() else {}
rep = J('mma_replay_summary.json')
runs = lambda d: d['runs'] if isinstance(d['runs'], list) else [d['runs']]
allruns = runs(fmE) + runs(fmS3) + [r for r in runs(fmL) if r['mode'] == 'lbfgs']
bsRef = ref['bs']

def fm_summary(r):
    k = r.get('kkt', {})
    return {'start': r['start'], 'mode': r['mode'], 'exitflag': r['exitflag'], 'iterations': r['iterations'],
            'bs': r['bs'], 'bs_minus_ref': r['bs'] - bsRef, 'firstorderopt': r['firstorderopt'],
            'constrviolation': r['constrviolation'], 'min_separation': r['min_sep'], 'wall_s': r['wall_s'],
            'kkt_verdict': k.get('verdict'), 'stat_norm_rms': k.get('stationarity', {}).get('norm_rms'),
            'box_comp_normalized': k.get('complementarity', {}).get('max_box_comp_normalized'),
            'image_abc': r.get('sol', {}).get('image_abc'), 'sum_drho': r.get('sol', {}).get('sum_drho'),
            'fJJ_drho': r.get('sol', {}).get('fJJ_drho')}
fm = [fm_summary(r) for r in allruns if r['exitflag'] != -99]
fm_bs = [r['bs'] for r in fm]
agree = max(abs(b - bsRef) for b in fm_bs) <= 1e-6 if fm_bs else None

dec = cmpj['decision']; traj = cmpj['trajectory']
relationship = dec['relationship_verdict']
final = {'REPEATED_MMA_REALIZATION_FAILS_PROBLEM25_REFERENCE': 'FROZEN_INNER_SOLVER_STUDY_JUSTIFIED',
         'PRODUCTION_INNER_TRUNCATION_MATERIALLY_PREMATURE': 'FROZEN_INNER_SOLVER_STUDY_JUSTIFIED',
         'PROBLEM25_MULTIPLE_LOCAL_SOLUTIONS_MATERIAL': 'FROZEN_INNER_SOLVER_STUDY_JUSTIFIED',
         'PRODUCTION_TRUNCATION_NEAR_REFERENCE': 'FROZEN_INNER_SOLVER_STUDY_NOT_JUSTIFIED',
         'PROBLEM25_REFERENCE_INCONCLUSIVE': 'FROZEN_INNER_SOLVER_STUDY_PREMATURE'}[relationship]

metrics = {
    'study': STUDY, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'branch': sh(f'git -C {ROOT} rev-parse --abbrev-ref HEAD'), 'head': sh(f'git -C {ROOT} rev-parse HEAD'),
    'preregistration_sha256': sha(HERE / 'AUDIT_PREREGISTRATION.md'),
    'locks': {'topology_optimization_runs': 0, 'outer_iterations': 0, 'accepted_density_updates': 0,
              'controller_transitions': 0, 'production_files_modified': 0, 'move_limit_changed': False,
              'packages_installed': 0},
    'verdicts': {
        'state': st['verdict'], 'cluster_reduction': eq['cluster_verdict'], 'socp_equivalence': eq['socp_verdict'],
        'convexity': eq['convexity_verdict'], 'reference_kkt': ref['kkt_verdict'], 'reference_kkt_with_solver_duals': ref['kkt_verdict_solver_duals'],
        'global': ref['global_verdict'], 'relationship': relationship, 'final': final},
    'state': {k: st[k] for k in ('rho386_sha256', 'rho385_sha256', 'drho386_sha256', 'cfgHash', 'implTree', 'nOuter', 'stage', 'move', 'N', 'J', 'multJ', 'nInner_final', 'lamref', 'lamJ', 'dOff', 'dOff_present', 'offDiag', 'gap12_fresh')},
    'equivalence': eq['summary'] | {'toy_convention_confirmed': eq['toy']['documented_convention_confirmed'], 'hessian_check': eq['hessian_check'], 'smooth_margin': eq['smoothMargin'], 'n_points': eq['n_points']},
    'reference': {
        'solver': 'coneprog', 'config': ref['config'], 'exitflag': ref['exitflag'], 'iterations': ref['output']['iterations'],
        'bs': bsRef, 'beta': ref['beta'], 'omega_pred': ref['omega_pred'], 'omega1_current': ref['omega1_current'], 'bs_minus_1': ref['bs_minus_1'],
        'duality_gap_best': ref['certificate']['gap'], 'duality_gap_aligned': ref['certificate_aligned']['gap'],
        'certificate_multipliers': {'mu': ref['certificate_aligned']['mu'], 'nu': ref['certificate_aligned']['nu'], 'w': ref['certificate_aligned']['w']},
        'production_fval': ref['production_fval'], 'solution': ref['solution'], 'active': ref['active'], 'active_sweep': ref['active_sweep'],
        'by_class': ref['by_class'], 'free_coordinates': ref['free_coordinates'], 'move_bound_dominated_preregistered_rule': ref['move_bound_dominated'],
        'kkt_certificate': {'stationarity': ref['kkt_certificate_duals']['stationarity'], 'complementarity': ref['kkt_certificate_duals']['complementarity'], 'primal': ref['kkt_certificate_duals']['primal']},
        'coneprog_sweep': [{k: c[k] for k in ('form', 'solver', 'tol', 'exitflag', 'iterations', 'bs', 'max_fval', 'gap')} for c in sw['candidates']]},
    'fmincon': {'gradient_check': gc, 'runs': fm, 'all_agree_in_objective_1e-6': agree,
                'max_abs_bs_minus_ref': max(abs(b - bsRef) for b in fm_bs) if fm_bs else None,
                'sqp': {k: sqp.get(k) for k in ('exitflag', 'iterations', 'wall_s', 'viable', 'bs', 'message')}},
    'replay': rep,
    'comparison': {'P19': {k: cmpj['P19'][k] for k in ('bs', 'beta', 'obj_gap', 'G', 'dist2', 'distInf', 'dist2_rel', 'cosine', 'max_util', 'viol', 'kkt_mma_norm_rms', 'kkt_refdual_proj_norm_rms', 'jaccard', 'counts', 'frac_move_bound', 'sign_agreement')},
                   'M500': {k: cmpj['M500'][k] for k in ('bs', 'beta', 'obj_gap', 'G', 'dist2', 'distInf', 'dist2_rel', 'cosine', 'max_util', 'viol', 'kkt_mma_norm_rms', 'kkt_refdual_proj_norm_rms', 'jaccard', 'counts', 'frac_move_bound', 'sign_agreement')},
                   'M5000': {k: cmpj['M5000'][k] for k in ('bs', 'beta', 'obj_gap', 'G', 'dist2', 'distInf', 'dist2_rel', 'cosine', 'max_util', 'viol', 'kkt_mma_norm_rms', 'kkt_refdual_proj_norm_rms', 'jaccard', 'counts', 'frac_move_bound', 'sign_agreement')},
                   'trajectory': traj, 'decision': dec, 'P19_equals_replay19': cmpj['P19_equals_replay19'], 'M500_equals_replay500': cmpj['M500_equals_replay500']},
}
json.dump(metrics, open(HERE / 'METRICS.json', 'w'), indent=1)

# ---- DATA_MANIFEST ----
kinds = {'.md': 'document', '.json': 'manifest', '.mat': 'evaluation', '.png': 'figure', '.svg': 'figure', '.m': 'script', '.py': 'script', '.txt': 'log'}
files = []
for p in sorted(HERE.rglob('*')):
    if p.is_file() and p.name not in ('DATA_MANIFEST.json', 'FINAL_SHA256.txt', 'EVIDENCE.json'):
        files.append({'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': sha(p), 'kind': kinds.get(p.suffix, 'other')})
man = {'manifest_schema': 'olhoff_current_data_manifest/1', 'study': STUDY, 'generated': metrics['generated'],
       'repo_head': metrics['head'], 'branch': metrics['branch'], 'implTree': st['implTree'],
       'optimization_runs': 0, 'density_updates': 0, 'n_files': len(files), 'files': files}
json.dump(man, open(HERE / 'DATA_MANIFEST.json', 'w'), indent=1)

# ---- EVIDENCE ----
traj = ROOT / 'analysis/OlhoffCurrent/evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat'
prior = ROOT / 'analysis/OlhoffCurrent/diagnostics/filtered_subproblem_integrability_audit/evaluations/inner_kkt_state.mat'
ev = {'study': STUDY, 'generated': metrics['generated'], 'head': metrics['head'],
      'authoritative_state': {'trajectory': str(traj.relative_to(ROOT)), 'trajectory_sha256': sha(traj), 'trajectory_bytes': traj.stat().st_size,
                              'rho386_sha256': st['rho386_sha256'], 'rho385_sha256': st['rho385_sha256'], 'drho386_sha256': st['drho386_sha256'],
                              'cfgHash': st['cfgHash'], 'implTree': st['implTree']},
      'prior_retained_state': {'path': str(prior.relative_to(ROOT)), 'bytes': prior.stat().st_size, 'role': 'ctx (reproduced bitwise here), P19 and M500 iterates',
                               'sha256_recorded_by_prior_audit': '2952b301c19cb60f4f7ef71da25feb533669415685458d5c64d302ebc48518b6'},
      'preregistration': {'path': 'AUDIT_PREREGISTRATION.md', 'sha256': metrics['preregistration_sha256']},
      'derived_here': [{'path': f['path'], 'sha256': f['sha256'], 'bytes': f['bytes']} for f in files if f['path'].startswith('evaluations/') and f['path'].endswith('.mat')],
      'locks': metrics['locks'], 'verdicts': metrics['verdicts']}
json.dump(ev, open(HERE / 'EVIDENCE.json', 'w'), indent=1)

# ---- FINAL_SHA256 ----
lines = [f"{sha(HERE / f['path'])}  {f['path']}" for f in files]
for extra in ('DATA_MANIFEST.json', 'EVIDENCE.json'):
    lines.append(f"{sha(HERE / extra)}  {extra}")
(HERE / 'FINAL_SHA256.txt').write_text('\n'.join(lines) + '\n')
print('METRICS/DATA_MANIFEST/EVIDENCE/FINAL_SHA256 written;', len(files), 'files')
print('verdicts:', json.dumps(metrics['verdicts'], indent=1))
