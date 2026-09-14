"""Finalization: integrity re-verification, DATA_MANIFEST.json, FINAL_SHA256.txt."""
import datetime
import subprocess
from cs_common import *


def main():
    now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    pf = json.loads((EV / 'preflight.json').read_text())
    integ = {'preregistration_sha256': sha256_file(STUDY / 'AUDIT_PREREGISTRATION.md'),
             'preregistration_expected': '5b6186f2f2b438f73ce4cfeae8e4565326dc91d99b19e207aa47604c32286ddd',
             'amendment1_sha256': sha256_file(STUDY / 'PREREGISTRATION_AMENDMENT_1.md'),
             'amendment1_expected': '298efd23763f678f7519d37dc7c40f6e693bad2bec8a1933e4d853e0f09fa340',
             'treatment_code_unchanged_since_launch': {}}
    for name, h in pf['treatment_code_sha256_at_launch'].items():
        integ['treatment_code_unchanged_since_launch'][name] = sha256_file(HERE / name) == h
    integ['all_treatment_code_unchanged'] = all(integ['treatment_code_unchanged_since_launch'].values())
    integ['prereg_ok'] = integ['preregistration_sha256'] == integ['preregistration_expected']
    integ['amendment_ok'] = integ['amendment1_sha256'] == integ['amendment1_expected']
    integ['production_olhoffSolve_sha256'] = sha256_file(ROOT / '+impl' / 'architecture' / 'olhoffSolve.m')
    integ['production_olhoffSolve_expected'] = '1e5a114cbf91717e01e5592e86203f401ed5cdb3e5f5ce9a727b8163cdf2fba3'
    integ['control_trajectory_sha256'] = sha256_file(CONTROL_TRAJ)
    integ['control_trajectory_ok'] = integ['control_trajectory_sha256'] == 'a87546bc391cdc683def34a9f27678884528032f6b140e156349d2e74135ab9b'
    git = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'analysis/OlhoffCurrent/+impl',
                          'analysis/OlhoffCurrent/diagnostics/three_rung_canary_preflight',
                          'analysis/OlhoffCurrent/diagnostics/frozen_problem25_reference',
                          'analysis/OlhoffCurrent/diagnostics/frozen_inner_solver_study'],
                         capture_output=True, text=True).stdout
    integ['git_status_protected_paths'] = git
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dump(EV / 'integrity_final.json', integ)
    files = []
    skip = {'FINAL_SHA256.txt'}
    for p in sorted(STUDY.rglob('*')):
        if p.is_file() and p.name not in skip and '__pycache__' not in p.parts and p.name != '.DS_Store':
            rel = p.relative_to(STUDY).as_posix()
            kind = ('script' if rel.startswith('scripts/') else 'figure' if rel.startswith('figures/') else
                    'run' if rel.startswith('run/') else 'evaluation' if rel.startswith('evaluations/') else 'document')
            files.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha256_file(p), 'kind': kind})
    ev = json.loads((STUDY / 'EVIDENCE.json').read_text()) if (STUDY / 'EVIDENCE.json').exists() else {}
    man = {'manifest_schema': 'c480_socp_causal_run/1', 'study': 'c480_socp_causal_run', 'generated': now,
           'repo_head': head, 'branch': 'benchmark-methodology-r2',
           'implTree': 'edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb',
           'treatment_runs': 1, 'control_reruns': 0, 'n_files': len(files), 'files': files,
           'external_raw_evidence': [{k: a.get(k) for k in ['path', 'class', 'bytes', 'sha256']} for a in ev.get('artifacts', [])]}
    dump(STUDY / 'DATA_MANIFEST.json', man)
    lines = ['FINAL_SHA256 -- c480_socp_causal_run', '=' * 78, f'generated {now}', f'HEAD      {head}',
             'implTree  edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb',
             'treatment runs 1 (C480 exact SOCP); control reruns 0', '=' * 78, '']
    for f in files + [{'path': 'DATA_MANIFEST.json', 'sha256': sha256_file(STUDY / 'DATA_MANIFEST.json')}]:
        if f['path'] == 'DATA_MANIFEST.json' and f is not files[-1] and any(x['path'] == 'DATA_MANIFEST.json' for x in files):
            continue
        lines.append(f"{f['sha256']}  {f['path']}")
    for a in ev.get('artifacts', []):
        if a.get('present'):
            lines.append(f"{a['sha256']}  analysis/OlhoffCurrent/evidence/c480_socp_causal_run/{a['path']}")
    (STUDY / 'FINAL_SHA256.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in integ.items() if k != 'treatment_code_unchanged_since_launch'}, indent=1))
    print('files', len(files))


if __name__ == '__main__':
    main()
