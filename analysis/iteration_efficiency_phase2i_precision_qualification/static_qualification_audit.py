#!/usr/bin/env python3
"""Fail-closed static audit of the prepared Phase-2I comparison path."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
def sha(p: Path) -> str:return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
protected=(REPO/'analysis/olhoff_stabilization_audit/olhoffOptStabilized.m').read_text()
capture=(HERE/'olhoffOptStabilizedDoubleCapture.m').read_text()
evaluator=(REPO/'analysis/three_method_parametric_study/study_evaluate_design.m').read_text()
pairer=(REPO/'analysis/iteration_efficiency_phase2b_precision/+ie2b/capture_prefix_case.m').read_text()
common=(REPO/'analysis/iteration_efficiency_phase2a/+ie2a/evaluate_common.m').read_text()
contract=REPO/'analysis/iteration_efficiency_phase2a/iteration_efficiency_contract.json'
freeze=REPO/'analysis/iteration_efficiency_phase2h_c_refreeze/PHASE2H_FREEZE_RECORD.md'

checks['contract_hash']=sha(contract)=='cc900b4ad4cae18b0bcd9b7a559f51e04e5167db587f64180b371d3c399bf95b'
checks['evaluator_hash']=sha(REPO/'analysis/three_method_parametric_study/study_evaluate_design.m')=='e14a21efe0bb2d9b9d7f3187b4c3f671ec089f6ff96773074b8f3b56cacd79e9'
checks['freeze_hash']=sha(freeze)=='b05d71b716a78f55f1bcd5d39fc76694d712a067bca5633dbafe4dd99bb84119'
checks['snapshot_is_2d']= "snapshots(:,outer+1)=single(rho)" in protected and "rho_snapshots',snapshots" in protected
checks['initial_and_postupdate_indexing']= "snapshots(:,1)=single(rho)" in protected and "baseline.rho_snapshots(:,k+1)" in pairer
checks['res_rho_uncast_double']= "'rho',rho" in protected and "rho=cfg.rho0*ones" in protected
checks['same_run_pair']= "xd=r.rho;xs=r.rho_snapshots(:,end)" in pairer
checks['prefix_only_for_identity']= "prefix_single_identical" in pairer and "cast_identity" in pairer
checks['candidate_c_invoked']= "study_evaluate_design" in common and "evaluator_candidate='C'" in common
checks['eq4a_invoked']= evaluator.count("g(low)=1e5*") == 2 and "g(low)=z(low).^6" not in evaluator
checks['adaptive_search_invoked']= "requested = min(3,technicalLimit)" in evaluator and "requested = min(2*requested,technicalLimit)" in evaluator
checks['unanimous_classifier_invoked']= "valid=eigenpairValid&diagnosticFinite&all(margins>0,2)" in evaluator
checks['strict_thresholds']= "margins=[0.5-voidKE,0.5-voidSE,dwp-0.5]" in evaluator
checks['old_eq4_excluded']= "g(low)=z(low).^6" not in evaluator and "Candidate-C" in evaluator

# Prove the qualification mirror changes only its identity/comments and the
# observational snapshot representation.
p_lines=protected.splitlines(); c_lines=capture.splitlines()
checks['capture_line_count_equal']=len(p_lines)==len(c_lines)
allowed={0,1,2,3,4,20,64}
diff=[i for i,(a,b) in enumerate(zip(p_lines,c_lines)) if a!=b]
checks['capture_diff_only_declared_lines']=set(diff)==allowed

with (REPO/'analysis/iteration_efficiency_phase2g_evaluator_selection_audit/PRECISION_PAIR_AUDIT.csv').open() as f:
    n_pairs=sum(1 for _ in csv.DictReader(f))
checks['historical_candidate_c_pair_records']=n_pairs==708

result={'schema_version':'phase2i_static_audit_v1','checks':checks,'pass':all(checks.values()),
        'capture_differing_zero_based_lines':diff,'historical_candidate_c_pair_records':n_pairs}
(HERE/'raw').mkdir(exist_ok=True)
(HERE/'raw/static_audit.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['pass']:
    raise SystemExit(json.dumps({k:v for k,v in checks.items() if not v},indent=2))
print(json.dumps(result,indent=2))
