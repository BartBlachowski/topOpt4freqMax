#!/usr/bin/env python3
"""Fail closed if the Phase-2I negative qualification package is incomplete."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name:str):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))

required={
 'PHASE2I_PRECISION_QUALIFICATION_REPORT.md','QUALIFICATION_IMPLEMENTATION_AUDIT.md',
 'INITIAL_PROVENANCE.md','PRECISION_ERROR_SUMMARY.csv','MODAL_SELECTION_EQUIVALENCE.csv',
 'CLASSIFIER_MARGIN_SUMMARY.csv','HARD_GATE_EQUIVALENCE.csv','QUALITY_EQUIVALENCE.csv',
 'REFERENCE_EQUIVALENCE.csv','PERSISTENCE_EQUIVALENCE.csv','DECISION_EQUIVALENCE.csv',
 'PRODUCTION_SCALE_RISK_CHECK.csv','HISTORICAL_PHASE2B_COMPARISON.md',
 'INDEPENDENT_REPLAY_REPORT.md','PREFLIGHT_AFTER_PHASE2I.md','qualification_provenance.json',
 'negative_precision_qualification.json','SHA256SUMS.txt','raw/REPRESENTATION_ERROR.csv',
 'raw/MODAL_DIAGNOSTICS.csv','raw/capture_96x12_H3200.mat','raw/reference_evaluation.mat',
 'raw/independent_replay.json','raw/preflight_after.json','raw/static_audit.json'}
checks={}
checks['required_files']=all((HERE/p).is_file() and (HERE/p).stat().st_size>0 for p in required)
static=json.loads((HERE/'raw/static_audit.json').read_text())
checks['binding_integrity']=static['pass'] and all(static['checks'].values())
checks['binding_rows']=len(rows('MODAL_SELECTION_EQUIVALENCE.csv'))==9600 and len(rows('HARD_GATE_EQUIVALENCE.csv'))==3200
checks['strategic_prefixes']=len(rows('PREFIX_DETERMINISM.csv'))>=7 and all(r['double_density_identical']=='1' for r in rows('PREFIX_DETERMINISM.csv'))
difficult=rows('DIFFICULT_CASE_MODAL_EQUIVALENCE.csv')
checks['adaptive_difficult_cases']=max(int(float(r['selected_ordinal_double'])) for r in difficult)>=18 and all(r['ordinal_identical']=='1' for r in difficult)
decision={r['criterion']:r['disposition'] for r in rows('DECISION_EQUIVALENCE.csv')}
checks['decision_table']=len(decision)==16 and decision.get('Q7')=='FAIL' and all(decision.get(f'Q{i}')=='PASS' for i in range(1,17) if i!=7)
negative=json.loads((HERE/'negative_precision_qualification.json').read_text())
checks['negative_artifact']=negative.get('pass') is False and negative.get('failure_reason')=='COMPLETE_HARD_GATE_DECISION_IDENTITY_FAILED_Q7'
report=(HERE/'PHASE2I_PRECISION_QUALIFICATION_REPORT.md').read_text().rstrip()
checks['exact_verdict']=report.endswith('PHASE 2I FAILED —\nOLHOFF SINGLE-PRECISION TRAJECTORY NOT QUALIFIED UNDER CANDIDATE C')
manifest={}
for line in (HERE/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split('  ',1);manifest[name]=digest
checks['checksum_manifest']=all((HERE/name).is_file() and sha(HERE/name)==digest for name,digest in manifest.items())
checks['checksum_coverage']=all(p=='SHA256SUMS.txt' or p in manifest for p in required)
result={'schema_version':'phase2i_package_verification_v1','checks':checks,'pass':all(checks.values())}
print(json.dumps(result,indent=2))
if not result['pass']:raise SystemExit(1)
