#!/usr/bin/env python3
"""WP0 - independent integrity hashing for the Phase-2D delta audit. READ-ONLY."""
import hashlib, json, os, subprocess, sys, datetime

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
OUT  = os.path.join(REPO, 'analysis/iteration_efficiency_phase2d_delta_audit')

def sha(p):
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()
    except FileNotFoundError:
        return 'MISSING'

def run(c):
    return subprocess.run(c, shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()

# --- protected numerical sources, from the Phase-2A record (authority) ---
p2a = json.load(open(os.path.join(REPO, 'analysis/iteration_efficiency_phase2a/implementation_provenance.json')))
protected = {e['path']: e['sha256'] for e in p2a['protected_numerical_sources']}
profiles  = {e['path']: e['sha256'] for e in p2a.get('profile_sources', [])}
audits    = {e['path']: e['sha256'] for e in p2a.get('audit_records', [])}

# --- Eq.(4)/(4a) implementations and other files of interest ---
extra = [
 'Matlab/reproduction2007/fem/massScale.m',
 'Matlab/reproduction2007/algo/defaultCfg.m',
 'analysis/OlhoffApproachExact/Matlab/mass_interp.m',
 'analysis/iteration_efficiency_phase2a/iteration_efficiency_contract.json',
 'analysis/iteration_efficiency_phase2a/implementation_provenance.json',
 'analysis/iteration_efficiency_phase2a/+ie2a/production_preflight.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/frozen_contract_sha256.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/reference_phase.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/scan_persistence.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/measurement_budget.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/topology_metrics.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/quality_effort.m',
 'analysis/iteration_efficiency_phase2a/+ie2a/evaluate_common.m',
 'analysis/iteration_efficiency_phase2d_evaluator_amendment/+ie2d/study_evaluate_design_eq4a.m',
 'analysis/iteration_efficiency_study_design/QUALITY_EFFORT_SPEC.md',
 'analysis/iteration_efficiency_study_design/ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md',
 'analysis/iteration_efficiency_methodology_final_recheck/METHODOLOGY_FREEZE_RECORD.md',
 'references/Du2007_Topological.pdf',
]

rec = {
 'phase': '2E delta audit (independent)',
 'classification': 'READ_ONLY_INDEPENDENT_DELTA_AUDIT / NO_OPTIMIZATION / NO_REFREEZE',
 'captured_at': datetime.datetime.now().astimezone().isoformat(),
 'branch': run('git branch --show-current'),
 'head': run('git rev-parse HEAD'),
 'git_status': run('git status --porcelain').splitlines(),
}

def check(group, table):
    out = {}
    for p, expect in table.items():
        got = sha(os.path.join(REPO, p))
        out[p] = {'expected_phase2a': expect, 'observed': got, 'match': got == expect}
    rec[group] = out
    bad = [p for p, v in out.items() if not v['match']]
    return bad

bad = []
bad += check('protected_numerical_sources', protected)
bad += check('profile_sources', profiles)
bad += check('audit_records', audits)
rec['protected_mismatches'] = bad

# Phase-2D declared unchanged hashes, checked against live files independently
amd = json.load(open(os.path.join(REPO, 'analysis/iteration_efficiency_phase2d_evaluator_amendment/amendment_provenance.json')))
rec['phase2d_declared_unchanged'] = {
  p: {'declared': h, 'observed': sha(os.path.join(REPO, p)), 'match': sha(os.path.join(REPO, p)) == h}
  for p, h in amd['unchanged_file_hashes'].items()}
rec['phase2d_declared_mismatches'] = [p for p, v in rec['phase2d_declared_unchanged'].items() if not v['match']]

rec['other_files'] = {p: sha(os.path.join(REPO, p)) for p in extra}

# Full Phase-2D package hash, and verification of its own SHA256SUMS.txt
p2d = os.path.join(REPO, 'analysis/iteration_efficiency_phase2d_evaluator_amendment')
pkg = {}
for root, dirs, files in os.walk(p2d):
    for fn in sorted(files):
        fp = os.path.join(root, fn)
        pkg[os.path.relpath(fp, REPO)] = sha(fp)
rec['phase2d_package'] = pkg

# verify Phase-2D's own SHA256SUMS.txt
sums = {}
sp = os.path.join(p2d, 'SHA256SUMS.txt')
if os.path.exists(sp):
    for line in open(sp):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].lstrip('*').strip()] = parts[0]
selfcheck = {}
for name, h in sums.items():
    cand = os.path.join(p2d, name) if not os.path.isabs(name) else name
    if not os.path.exists(cand):
        cand2 = os.path.join(REPO, name)
        cand = cand2 if os.path.exists(cand2) else cand
    got = sha(cand)
    selfcheck[name] = {'declared': h, 'observed': got, 'match': got == h}
rec['phase2d_sha256sums_selfcheck'] = selfcheck
rec['phase2d_sha256sums_mismatches'] = [k for k, v in selfcheck.items() if not v['match']]

tag = sys.argv[1] if len(sys.argv) > 1 else 'pre'
with open(os.path.join(OUT, f'WP0_INTEGRITY_{tag}.json'), 'w') as f:
    json.dump(rec, f, indent=1)

print('branch:', rec['branch'], 'head:', rec['head'])
print('protected mismatches       :', rec['protected_mismatches'] or 'NONE')
print('phase2d-declared mismatches:', rec['phase2d_declared_mismatches'] or 'NONE')
print('phase2d SHA256SUMS mismatch:', rec['phase2d_sha256sums_mismatches'] or 'NONE')
print('files in phase2d package   :', len(pkg))
