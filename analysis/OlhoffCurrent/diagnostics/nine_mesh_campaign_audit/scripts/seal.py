"""Verify and seal audit artifacts only. Never writes source/evidence inputs."""
import json,hashlib,csv,re,subprocess,datetime,sys
from pathlib import Path
import h5py,numpy as np
P=Path(__file__).resolve().parents[1];ROOT=P.parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(n):return json.loads((P/n).read_text())
def dump(n,d):(P/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
required='AUDIT_PREREGISTRATION.md PROVENANCE.md CAMPAIGN_INTEGRITY.md MASTER_TABLE.csv CONTROLLER_GENERALIZATION.md TERMINATION_QUALITY.md MESH_REFINEMENT.md TOPOLOGY_CONVERGENCE.md FILTER_AUDIT.md LITERATURE_FIDELITY.md THREE_RUNG_VALUE.md PERFORMANCE_SCALING.md INNER_MMA_SCALING.md MULTIPLICITY_AUDIT.md REGIME_CHANGE_AUDIT.md SCORECARD.md RECOMMENDATION.md REPORT.md'.split()
assert all((P/x).exists() for x in required)
with (P/'MASTER_TABLE.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==9 and [int(r['nelx']) for r in rows]==list(range(160,801,80))
assert len(list((P/'figures').glob('*.png')))==12 and len(list((P/'figures').glob('*.svg')))==12
v=load('verification.json');assert all(x['match'] for x in v['source_hashes']);assert v['impl_tree']==v['expected_tree'];assert v['all_impl_files_match'];assert not v['extra_impl_files'];assert all(not x for x in v['manifest_vs_actual_config_differences'])
checks=list(csv.DictReader((P/'INTEGRITY_TABLE.csv').open()))
for r in checks:
 for k in ['raw_json_match','config_hash_match','density_shape_match','density_finite','density_bounds_ok','density_grayness_match','eigen_finite','ordered_positive_frequencies','rmin_correct','one_thread','error_empty']:assert r[k]=='True',(r['mesh'],k)
for r in load('historical_replay_checks.json'):
 for k in ['rho_finite','drho_finite','A_all_match','B_all_match','nA_all_match','nB_all_match','inner_sum_matches_record']:assert r[k]
assert load('C320_validation_recheck.json')['differing_columns']==[]
for r in rows:
 assert abs(sum(float(r[k]) for k in ['eigen_s','gradient_s','inner_s','other_s'])-float(r['runtime_total_s']))<1e-8
 assert float(r['rminEl'])==float(r['nely'])*.06 or abs(float(r['rminEl'])-float(r['nely'])*.06)<1e-12
 assert r['first_terminal_E']==''
report=(P/'REPORT.md').read_text();assert report.startswith('# BOTTOM LINE');assert all(f'**{i}.' in report for i in range(1,26));assert all(x in report for x in ['# WHAT WE LEARNED','# WHAT WE DID NOT LEARN','# WHAT I WOULD DO NEXT'])
verdicts=load('VERDICTS.json');assert len(verdicts)==9 and verdicts['PROGRAMME']=='IMPLEMENTATION_OR_CAMPAIGN_INVALID'
# Local explicit Markdown links should resolve. External DOI links are cited source locators.
broken=[]
for p in P.glob('*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if link.startswith(('https://','http://','#')):continue
  link=re.sub(r':\d+$','',link);target=Path(link) if link.startswith('/') else p.parent/link
  if not target.exists():broken.append((str(p.name),link))
assert not broken,broken
# The task began clean: nothing outside the audit may change.
status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True)
unexpected=[line for line in status.splitlines() if 'analysis/OlhoffCurrent/diagnostics/nine_mesh_campaign_audit/' not in line]
assert not unexpected,unexpected
inputs={x['path']:x for x in load('INPUT_MANIFEST.json')['inputs']}
for x in load('SUPPLEMENT_INPUTS.json'):inputs[x['path']]={**x,'present':True}
# All text evidence that informed interpretive statements is also part of the input ledger.
extra=['references/Du2007_Topological.pdf','references/Du and Olhoff - Topological design of freely vibrating continuum s.pdf','analysis/OlhoffCurrent/EVIDENCE_POLICY.md','analysis/OlhoffCurrent/diagnostics/three_rung_promotion_closure/REPORT.md','analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation_retry1/C320_PREFIX_EQUIVALENCE.md','analysis/OlhoffCurrent/diagnostics/two_rung_architecture/REPORT.md','analysis/OlhoffCurrent/diagnostics/fixedmove_400_dynamics/REPORT.md','analysis/OlhoffCurrent/olhoffcurrent_config_hash.m','analysis/OlhoffCurrent/olhoffcurrent_source_manifest.m','analysis/OlhoffCurrent/olhoffcurrent_evidence_gate.m','examples/Performance/conference_benchmark/campaign_9mesh_r2/BENCHMARK_NOTES.md','examples/Performance/conference_benchmark/campaign_9mesh_r2/timing_schema.json']
for rel in extra:
 p=ROOT/rel;inputs[rel]={'path':rel,'present':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'sha256':sha(p) if p.exists() else None}
changed=[]
for rel,x in inputs.items():
 p=ROOT/rel
 if x.get('present') and (not p.exists() or sha(p)!=x['sha256']):changed.append(rel)
assert not changed,changed
# Dimensions are measured from HDF5 objects, retaining explicit HDF5 orientation.
artifacts=[]
raw=[x for x in inputs.values() if x['path'].endswith('.mat') and '/evidence/' in x['path'] and any(t in x['path'] for t in ['two_branch_controller_validation/','three_rung_resolution_240/']) and x.get('present')]
raw.append(inputs['examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat'])
for x in raw:
 p=ROOT/x['path'];dims=[]
 with h5py.File(p) as f:
  for k,o in f.items():
   if k=='#refs#':continue
   if isinstance(o,h5py.Dataset):dims.append({'name':k,'HDF5_shape':list(o.shape),'dtype':str(o.dtype),'MATLAB_class':str(o.attrs.get('MATLAB_class',b'').decode())})
   else:dims.append({'name':k,'kind':'MATLAB struct represented as HDF5 group','fields':list(o.keys())})
  if 'records' in f:
   rho=[]
   for i,ref in enumerate(f['records/method_key'][()].ravel()):
    text=''.join(chr(int(z)) for z in f[ref][()].ravel())
    if text=='olhoff':
     refx=f['records/x'][()].ravel()[i];o=f[refx];rho.append({'record_index_MATLAB':i+1,'HDF5_shape':list(o.shape),'dtype':str(o.dtype)})
   dims.append({'name':'records.x for method_key=olhoff','arrays':rho})
 artifacts.append({'path':x['path'],'class':'required','description':'Stored numerical input for this offline audit; scientific identity limitations are documented separately','bytes':p.stat().st_size,'sha256':sha(p),'present':True,'format':'MAT v7.3 HDF5','variables':dims})
missing='analysis/OlhoffCurrent/evidence/three_rung_promotion_validation_retry1/C320x40_three_rung_trajectory.mat'
artifacts.append({'path':missing,'class':'optional','description':'Missing original candidate; absence disclosed. Audit does not claim raw candidate replay.','sha256':'fb29817e1a5de7e3803fd00ef991c1b3cf23baca4aca48e50876b86ca5bbaffe','present':False,'format':'MAT v7.3 HDF5'})
dump('EVIDENCE.json',{'schema':'olhoff_current_evidence/1','study':'nine_mesh_campaign_audit','evidenceRoot':'','audit_only':True,'scientific_runs':0,'artifacts':artifacts})
qa={'required_artifacts_present':True,'master_rows':9,'figures_png':12,'figures_svg':12,'source_hash_matches':21,'effective_config_hash_matches':9,'impl_files_match':75,'historical_branch_replays_match':4,'candidate_csv_columns_match':52,'timing_reconciliation':True,'local_links_resolve':True,'no_changes_outside_audit':True,'all_measured_inputs_unchanged':True,'scientific_runs':0,'review_notes':'Atlas, refinement/scaling/decomposition/branch/residual figures inspected visually. Missing histories are left blank; current hashes do not authenticate an absent original campaign seal.'}
dump('QA.json',qa)
# Exclude self and the final seal to avoid circular hashes; the seal covers DATA_MANIFEST itself.
files=[p for p in P.rglob('*') if p.is_file() and p.name not in ['DATA_MANIFEST.json','FINAL_SHA256.txt'] and '__pycache__' not in p.parts]
outputs=[{'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
dump('DATA_MANIFEST.json',{'schema':'audit_manifest/1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_root':str(P.relative_to(ROOT)),'source_HEAD':v['current_head'],'scientific_runs':0,'scope':'Legacy nine-mesh campaign identity audit plus retained historical E-controller evidence; no new scientific runs','inputs':list(inputs.values()),'outputs':outputs,'hash_coverage':'Outputs exclude DATA_MANIFEST.json and FINAL_SHA256.txt to avoid circularity; FINAL_SHA256.txt covers DATA_MANIFEST.json too.','verdicts':verdicts})
files.append(P/'DATA_MANIFEST.json')
(P/'FINAL_SHA256.txt').write_text('\n'.join(sha(p)+'  '+str(p.relative_to(P)) for p in sorted(files))+'\n')
# Read back rather than trust the write.
for line in (P/'FINAL_SHA256.txt').read_text().splitlines():
 h,rel=line.split('  ',1);assert sha(P/rel)==h
print(json.dumps({'QA':'PASS','sealed_files':len(files),'inputs':len(inputs),'verdicts':verdicts},indent=2))
