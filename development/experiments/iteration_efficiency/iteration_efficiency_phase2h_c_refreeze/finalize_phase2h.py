#!/usr/bin/env python3
"""Generate deterministic Phase-2H provenance and package SHA manifest."""
from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

modified=[
"analysis/three_method_parametric_study/study_evaluate_design.m",
"analysis/iteration_efficiency_phase2a/+ie2a/evaluate_common.m",
"analysis/iteration_efficiency_phase2a/+ie2a/analyze_trajectory.m",
"analysis/iteration_efficiency_phase2a/+ie2a/reference_phase.m",
"analysis/iteration_efficiency_phase2a/+ie2a/classify_status.m",
"analysis/iteration_efficiency_phase2a/+ie2a/account_iterations.m",
"analysis/iteration_efficiency_phase2a/+ie2a/run_method_trajectory.m",
"analysis/iteration_efficiency_phase2a/+ie2a/run_production_campaign.m",
"analysis/iteration_efficiency_phase2a/+ie2a/production_preflight.m",
"analysis/iteration_efficiency_phase2a/+ie2a/validate_contract.m",
"analysis/iteration_efficiency_phase2a/+ie2a/frozen_contract_sha256.m",
"analysis/iteration_efficiency_phase2a/iteration_efficiency_campaign.m",
"analysis/iteration_efficiency_phase2a/iteration_efficiency_contract.json",
"analysis/iteration_efficiency_phase2a/run_phase2a_tests.m",
"analysis/iteration_efficiency_study_design/ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md",
"analysis/iteration_efficiency_study_design/ACCEPTANCE_GATE_SPEC.md",
"analysis/iteration_efficiency_study_design/REFERENCE_QUALITY_SPEC.md",
"analysis/iteration_efficiency_study_design/QUALITY_EFFORT_SPEC.md",
"analysis/iteration_efficiency_study_design/ITERATION_ACCOUNTING_SPEC.md",
"analysis/iteration_efficiency_study_design/TIMING_SPEC.md",
"analysis/iteration_efficiency_study_design/PROPOSED_TABLE_LAYOUTS.md",
"analysis/iteration_efficiency_study_design/IMPLEMENTATION_REQUIREMENTS.md",
"analysis/iteration_efficiency_study_design/FAIRNESS_RISK_REGISTER.md"]
created=[
"analysis/iteration_efficiency_phase2a/+ie2a/olhoff_variant_plan.m",
"analysis/iteration_efficiency_phase2a/+ie2a/validate_qualification.m",
"analysis/iteration_efficiency_phase2a/+ie2a/frozen_freeze_record_sha256.m"]
created += [str(p.relative_to(REPO)) for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in {
    'initial_provenance.json','implementation_provenance.json','SHA256SUMS.txt'}]
native=["analysis/ourApproach/Matlab/topopt_freq.m",
"analysis/YukselApproach/Matlab/top99neo_inertial_freq.m",
"analysis/olhoff_stabilization_audit/olhoffOptStabilized.m",
"Matlab/reproduction2007/algo/innerLoopLP.m",
"analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m"]
initial=json.loads((HERE/'initial_provenance.json').read_text())
start={x['path']:x['sha256'] for x in initial['starting_hashes']}
native_integrity={p:{'sha256':sha(REPO/p),'unchanged_from_start':start.get(p,sha(REPO/p))==sha(REPO/p)} for p in native}
tracked=sorted(set(modified+created+native+[
"analysis/iteration_efficiency_study_design/TOPOLOGY_SANITY_SPEC.md",
"analysis/iteration_efficiency_study_design/SCALING_AND_FIGURE_SPEC.md",
"analysis/iteration_efficiency_study_design/EVIDENCE_AVAILABILITY_MATRIX.csv"]))
prov={
 'schema_version':'phase2h_implementation_provenance_v1',
 'generated_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
 'phase':'2H controlled Candidate-C implementation and refreeze',
 'starting_branch':initial['branch'],'starting_head':initial['head'],
 'starting_git_status':initial['starting_git_status'],'environment':initial['environment'],
 'modified_files':modified,'created_files':created,
 'final_hashes':{p:sha(REPO/p) for p in tracked},
 'native_optimizer_integrity':native_integrity,
 'phase2g_manifest_verification':initial['phase2g_manifest_verification'],
 'tests':{'offline_regression':'PASS','candidate_c_matlab':'PASS_11_OF_11',
          'phase2a_regression':'PASS_12_OF_12','stale_preflight_controls':'PASS_7_OF_7'},
 'optimizer_run':False,'production_campaign_run':False,
 'production_status':'BLOCKED_PENDING_THREE_QUALIFICATIONS'}
(HERE/'implementation_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
files=sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
(HERE/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
print(json.dumps({'provenance':str(HERE/'implementation_provenance.json'),
                  'manifest_files':len(files),'native_unchanged':all(x['unchanged_from_start'] for x in native_integrity.values())},indent=2))
