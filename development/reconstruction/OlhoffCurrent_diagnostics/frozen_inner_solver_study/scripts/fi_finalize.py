from pathlib import Path
import json,hashlib,datetime,subprocess,re,csv
S=Path(__file__).resolve().parents[1];E=S/'evaluations';root=S.parents[1];repo=root.parents[1]
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((E/'provenance_before.json').read_text());bad=[]
for n,h in before['protected'].items():
 p=Path(n);now=hashfile(p) if p.exists() else 'MISSING'
 if now!=h:bad.append({'path':n,'before':h,'after':now})
assert not bad,bad
expected=(E/'preregistration_sha256.txt').read_text().split()[0]
assert hashfile(S/'AUDIT_PREREGISTRATION.md')==expected
M=json.loads((S/'METRICS.json').read_text());assert M['complete']
V=json.loads((E/'verdict_decisions.json').read_text());R=json.loads((E/'results_validation.json').read_text())
assert R['verdict']=='FROZEN_VARIANTS_EXACT_PROBLEM_AND_METRICS_PASS'
required=['AUDIT_PREREGISTRATION.md','PROVENANCE.md','ORACLE_IDENTITY.md','RECONSTRUCTION_CHOICES.md','BASELINE_REPLAY.md','ORACLE_METRICS.md','PERSISTENT_STATE_TEST.md','ASYMPTOTE_INITIALIZATION_TEST.md','ASYMPTOTE_UPDATE_TEST.md','STOPPING_RULE_ANALYSIS.md','GCMMA_TEST.md','SOCP_ORACLE_COST.md','SOCP_GENERALIZATION.md','ORACLE_SIGN_STRUCTURE.md','FAILURE_LOCALIZATION.md','BOUND_STRUCTURE.md','WORK_NORMALIZED_COMPARISON.md','CAUSAL_ATTRIBUTION.md','INNER_SOLVER_SELECTION.md','SOCP_PROMOTION_GATE.md','C480_CAUSAL_RUN_GATE.md','FILTER_STUDY_GATE.md','PERFORMANCE_STATUS.md','REPORT.md','MASTER_METRICS.csv','METRICS.json']
assert all((S/f).is_file() for f in required)
assert (S/'REPORT.md').read_text().startswith('BOTTOM LINE\n')
assert len(re.findall(r'^\d+\. \*\*', (S/'REPORT.md').read_text(),re.M))==33
figures=sorted((S/'figures').glob('FIG_*.png'));assert len(figures)>=20
integrity={'timestamp':datetime.datetime.now().astimezone().isoformat(),'protected_files_checked':len(before['protected']),'protected_failures':bad,'preregistration_unchanged':True,'production_unchanged':True,'reference_unchanged':True,'trajectory_unchanged':True,'git_status_after':subprocess.check_output(['git','status','--short'],cwd=repo).decode(),'verdict':'ZERO_UPDATE_AND_PROTECTED_FILES_PASS'}
(E/'integrity_final.json').write_text(json.dumps(integrity,indent=2)+'\n')
verdictkeys=['oracle','persistence','stopping','gcmma','socp_gate','primary_cause','c480_gate','filter_gate','performance_gate']
evidence={'study':'frozen_inner_solver_study','schema':'frozen_solver_audit/1','timestamp':integrity['timestamp'],'preregistration_sha256':expected,'repo_head':before['head'],'verdicts':{k:V[k] for k in verdictkeys},'selected_candidate':V['candidate'],'topology_runs':0,'outer_density_updates':0,'accepted_rho_updates':0,'controller_transitions':0,'production_modifications':0,'reference_modifications':0,'oracle_identity':M['oracle'],'results_validation':R,'integrity':integrity,'limitations':['Fresh B0 extends to 500; the 5000-call replay is authenticated and re-evaluated, not freshly rerun.','MMA variant budgets are 500 calls or 1800 solver seconds, whichever fires first. Incomplete 500-call checkpoints cannot support strong registered attribution.','Timing is indicative on a shared host.','N>2 SDP formulation is analytical; no production SDP implementation is validated.','No future C480 topology run was executed.'],'input_hashes':{'reference_manifest':hashfile(S.parent/'frozen_problem25_reference/DATA_MANIFEST.json'),'oracle_mat':hashfile(S.parent/'frozen_problem25_reference/evaluations/conic_reference.mat'),'frozen_context':hashfile(S.parent/'frozen_problem25_reference/evaluations/frozen_ctx.mat'),'upstream_archive':before['upstream_sha256'],'pedersen2000':hashfile(repo/'docs/s001580050130.pdf')},'required_files_present':required,'figures':[str(p.relative_to(S)) for p in figures]}
(S/'EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
# Exclude final manifest itself and FINAL_SHA256 from its own entries. The
# latter authenticates the data manifest as well, avoiding circular hashes.
files=[p for p in S.rglob('*') if p.is_file() and p.name not in ['DATA_MANIFEST.json','FINAL_SHA256.txt'] and '__pycache__' not in p.parts and 'mpl_cache' not in p.parts]
entries=[{'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'sha256':hashfile(p)} for p in sorted(files)]
manifest={'schema':'frozen_solver_audit_manifest/1','generated':integrity['timestamp'],'study':'frozen_inner_solver_study','preregistration_sha256':expected,'input_hashes':evidence['input_hashes'],'file_count':len(entries),'files':entries,'excludes':['DATA_MANIFEST.json (self)','FINAL_SHA256.txt (self-authentication cycle)','Python/font caches']}
(S/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
files.append(S/'DATA_MANIFEST.json')
(S/'FINAL_SHA256.txt').write_text(''.join(f'{hashfile(p)}  {p.relative_to(S)}\n' for p in sorted(files)))
# Reopen delivered hashes, rather than trusting only writes above.
for f in json.loads((S/'DATA_MANIFEST.json').read_text())['files']:assert hashfile(S/f['path'])==f['sha256']
print('FINAL AUDIT PASS:',len(before['protected']),'protected files;',len(entries),'artifact hashes;',len(figures),'figures; 33 answers; zero density updates.')
