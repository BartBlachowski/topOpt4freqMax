#!/usr/bin/env python3
"""Read-only input verification; writes only this audit's inventory evidence.

Run with repository .venv/bin/python. Does not import/call any optimizer,
execute MATLAB, replay trajectories, update prior manifests, or certify P15.
"""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
REPO = ROOT.parents[1]
CV = ROOT / 'diagnostics/two_branch_controller_validation'
TB = ROOT / 'diagnostics/two_branch_maturity_240'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rel(path):
    return str(path.relative_to(REPO))


def read(path):
    return json.loads(path.read_text())


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def artifact(p):
    # Exact port of olhoffcurrent_is_artifact.m, whose hash is also recorded.
    ext = p.suffix.lower()
    if ext in {'.m', '.mlx', '.mlapp', '.p', '.slx', '.mdl', '.fig',
               '.mat', '.json', '.md', '.txt', '.csv'} or ext.startswith('.mex'):
        return False
    return (p.name.lower() in {'.ds_store', 'thumbs.db', 'desktop.ini', '.localized'}
            or p.name.startswith('._')
            or ext in {'.asv', '.m~', '.autosave', '.orig', '.rej', '.swp', '.swo', '.bak'})


def main():
    generated = datetime.now(timezone.utc).isoformat()
    source = read(ROOT / 'SOURCE_MANIFEST.json')
    core = ROOT / '+impl'
    actual = {str(p.relative_to(core)): sha(p) for p in sorted(core.rglob('*'))
              if p.is_file() and not artifact(p)}
    expected = {x['path']: x['sha256'] for x in source['files']}
    tree = hashlib.sha256('\n'.join(f'{p}  {actual[p]}' for p in sorted(actual)).encode()).hexdigest()
    source_ok = actual == expected and tree == source['tree_sha256']
    source_check = {'pass': source_ok, 'tree_sha256': tree,
                    'recorded_tree_sha256': source['tree_sha256'],
                    'n_files': len(actual), 'files': actual,
                    'manifest_sha256': sha(ROOT / 'SOURCE_MANIFEST.json'),
                    'artifact_policy_sha256': sha(ROOT / 'olhoffcurrent_is_artifact.m')}
    dump(STUDY / 'evidence/source_integrity.json', source_check)

    manifests = [CV / 'DATA_MANIFEST.json', TB / 'DATA_MANIFEST.json']
    records = []
    for manifest in manifests:
        m = read(manifest)
        for section in ['tracked', 'raw', 'inputs', 'outputs']:
            for x in m.get(section, []):
                p = REPO / x['path']
                observed = sha(p) if p.is_file() else None
                exp = x.get('sha256')
                status = ('MISSING' if observed is None else
                          'UNHASHED_IN_PRIOR_MANIFEST' if exp is None else
                          'PASS' if observed == exp else 'HASH_MISMATCH')
                records.append({'path': x['path'], 'manifest': rel(manifest),
                                'section': section, 'expected_sha256': exp,
                                'observed_sha256': observed,
                                'bytes': p.stat().st_size if p.is_file() else None,
                                'status': status})
    prior = read(CV / 'evidence/provenance_resume_20260909.json')
    frozen = []
    for x in prior['frozen']:
        p = ROOT / x['path']
        frozen.append({'path': rel(p), 'expected': x['sha256'],
                       'observed': sha(p), 'pass': sha(p) == x['sha256']})
    p = CV / 'PREREGISTRATION.md'
    h = sha(p)
    expected_cv = next(x['sha256'] for x in read(manifests[0])['tracked']
                       if x['path'] == rel(p))
    frozen.append({'path': rel(p), 'expected': expected_cv, 'observed': h,
                   'pass': h == expected_cv and h == sha(CV / 'evidence/PREREGISTRATION.frozen')})

    matlab_path = Path('/Applications/MATLAB_R2025b.app/VersionInfo.xml')
    xml = ET.parse(matlab_path).getroot()
    matlab = {k: xml.findtext(k) for k in ['version', 'release', 'description', 'date']}
    matlab['method'] = 'Installed VersionInfo.xml, corroborated by existing C320 log; no MATLAB launched'

    tests = []
    for filename in ['software_tests_rerun_20260909_after_F400_restore.json',
                     'suite_tests_20260909.json']:
        p = CV / 'evidence' / filename
        d = read(p)
        tests.append({'path': rel(p), 'sha256': sha(p), 'n_tests': d['nTests'],
                      'n_fail': d['nFail'], 'provenance': 'PRIOR_RETAINED_RESULT_NOT_RERUN'})
    c320 = ROOT / 'evidence/two_branch_controller_validation/C320x40_trajectory.mat'
    current_status = subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True)
    data = {'study': 'POST-HOC OFFLINE ARCHITECTURE AUDIT', 'generated_utc': generated,
            'phase': 'PHASE_0_PENDING_C320' if not c320.exists() else 'PHASE_0_REQUIRES_C320_COMPLETION_VERIFICATION',
            'evidence_gate_pass': False,
            'note': 'Inventory only. Presence/hash alone cannot close the C320 numerical-equivalence/P15 gate.',
            'scientific_optimization_runs_launched_by_this_audit': 0,
            'source_integrity_pass': source_ok, 'impl_tree_sha256': tree,
            'original_controller_sha256': sha(core / 'architecture/+olh/+move/exhaustion.m'),
            'causal_preregistration_sha256': h,
            'mechanism_preregistration_sha256': sha(TB / 'PREREGISTRATION.md'),
            'offline_preregistration_sha256': sha(STUDY / 'PREREGISTRATION.md'),
            'prior_frozen_hashes_unchanged': all(x['pass'] for x in frozen),
            'frozen_checks': frozen, 'matlab': matlab, 'retained_tests': tests,
            'input_manifests': [{'path': rel(p), 'sha256': sha(p)} for p in manifests],
            'input_checks': records,
            'check_counts': {s: sum(x['status'] == s for x in records)
                             for s in ['PASS', 'MISSING', 'HASH_MISMATCH', 'UNHASHED_IN_PRIOR_MANIFEST']},
            'snapshot_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'snapshot_status': current_status}
    dump(STUDY / 'DATA_MANIFEST.json', data)
    dump(STUDY / 'evidence/inventory_tests.json', {
        'generated_utc': generated, 'scope': 'READ_ONLY_PROVENANCE_CHECKS_ONLY',
        'source_integrity': source_ok,
        'frozen_preregistrations_and_definitions': all(x['pass'] for x in frozen),
        'prior_manifest_hashes_for_present_inputs': all(x['status'] == 'PASS' for x in records if x['status'] != 'MISSING'),
        'missing_inputs_do_not_count_as_pass': True,
        'complete_evidence_gate': False,
        'scientific_optimization_runs': 0})
    print(json.dumps({k: data[k] for k in ['phase', 'source_integrity_pass', 'impl_tree_sha256',
                                         'prior_frozen_hashes_unchanged', 'check_counts']}, indent=2))
    for x in records:
        if x['status'] != 'PASS':
            print(x['status'], x['path'])


if __name__ == '__main__':
    main()
