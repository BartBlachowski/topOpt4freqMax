#!/usr/bin/env python3
"""Capture immutable Phase-2I starting provenance before numerical work."""
from __future__ import annotations
import hashlib, json, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def git(*args: str) -> str:
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

contract_path=REPO/'analysis/iteration_efficiency_phase2a/iteration_efficiency_contract.json'
contract=json.loads(contract_path.read_text())
norm_lines=''.join(f"{d['sha256']}  {d['path']}\n" for d in contract['normative_documents'])
norm_hash=hashlib.sha256(norm_lines.encode()).hexdigest()
files=[
 'analysis/iteration_efficiency_phase2a/iteration_efficiency_contract.json',
 'analysis/three_method_parametric_study/study_evaluate_design.m',
 'analysis/iteration_efficiency_phase2h_c_refreeze/PHASE2H_FREEZE_RECORD.md',
 'analysis/iteration_efficiency_phase2h_c_refreeze/implementation_provenance.json',
 'analysis/olhoff_stabilization_audit/olhoffOptStabilized.m',
 'Matlab/reproduction2007/algo/innerLoopLP.m',
 'analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m',
 'analysis/iteration_efficiency_phase2b_recheck/PHASE2B_RECHECK_REPORT.md',
 'analysis/iteration_efficiency_phase2b_recheck/qualification_manifest.json',
 'analysis/iteration_efficiency_phase2b_recheck/qualification_runs/probe_96x12_H3200.mat',
 'analysis/iteration_efficiency_phase2b_recheck/qualification_runs/resolve_96x12.mat',
 'analysis/iteration_efficiency_phase2f_evaluator_redesign/scripts/survey.npz',
 'analysis/iteration_efficiency_phase2g_evaluator_selection_audit/PRECISION_PAIR_AUDIT.csv',
 'analysis/three_method_parametric_study/results/profile_freeze_manifest.json',
]
prov={
 'schema_version':'phase2i_initial_provenance_v1',
 'captured_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
 'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),
 'starting_git_status':git('status','--short').splitlines(),
 'environment':{
   'matlab':'25.2.0.2998904 (R2025b)','matlab_arch':'MACA64',
   'matlab_threads':1,'blas':'Apple Accelerate BLAS (ILP64)',
   'lapack':'NAG Performance Components 1.2.1 / LAPACK 3.9.1',
   'python':platform.python_version(),'platform':platform.platform(),
 },
 'binding_identities':{
   'contract_sha256':sha(contract_path),
   'evaluator_sha256':sha(REPO/contract['quality']['source']),
   'normative_manifest_sha256':norm_hash,
   'freeze_record_sha256':sha(REPO/contract['phase2h_refreeze']['freeze_record']),
 },
 'starting_hashes':{p:sha(REPO/p) for p in files},
 'native_optimizer_modified_by_phase2i':False,
 'frozen_methodology_modified_by_phase2i':False,
 'production_campaign_run':False,
}
(HERE/'initial_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
md=f"""# Phase 2I initial provenance

Captured before numerical qualification at `{prov['captured_at_utc']}`.

- Branch: `{prov['branch']}`
- HEAD: `{prov['head']}`
- Repository state: pre-existing dirty worktree, recorded verbatim in `initial_provenance.json`
- MATLAB: `{prov['environment']['matlab']}`, `{prov['environment']['matlab_arch']}`
- BLAS: `{prov['environment']['blas']}`
- LAPACK/eigensolver: `{prov['environment']['lapack']}` / MATLAB `eigs`
- Thread count: `1`
- Platform: `{prov['environment']['platform']}`
- Contract: `{prov['binding_identities']['contract_sha256']}`
- Evaluator: `{prov['binding_identities']['evaluator_sha256']}`
- Normative manifest: `{prov['binding_identities']['normative_manifest_sha256']}`
- Freeze record: `{prov['binding_identities']['freeze_record_sha256']}`
- Native optimizer modified: **NO**
- Frozen methodology modified: **NO**
- Production campaign run: **NO**

The full starting status and input hashes are in `initial_provenance.json`.
"""
(HERE/'INITIAL_PROVENANCE.md').write_text(md)
print(json.dumps(prov['binding_identities'],indent=2))
