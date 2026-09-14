"""Read-only integrity verification plus audit manifests. Never calls MATLAB."""
from identity import *
import scipy.io as sio
import datetime,re,csv

def desc(p,klass='required'):
 a={'path':str(p.relative_to(ROOT)),'class':klass,'bytes':p.stat().st_size,'sha256':sha(p),'present':True}
 ext=p.suffix
 if ext=='.mat':
  if h5py.is_hdf5(p):
   a['format']='MATLAB v7.3 HDF5'
   with h5py.File(p) as f:a['variables']=[{'name':k,'size_matlab':list(v.shape[::-1]),'precision':str(v.dtype)} for k,v in f.items() if isinstance(v,h5py.Dataset)]
  else:a['format']='MATLAB v5/v7';a['variables']=[{'name':n,'size':list(sz),'class':cl} for n,sz,cl in sio.whosmat(p)]
 elif ext=='.npz':
  a['format']='NumPy compressed archive'
  with np.load(p) as f:a['variables']=[{'name':n,'size':list(f[n].shape),'precision':str(f[n].dtype)} for n in f.files]
 else:a['format']={'.json':'JSON','.csv':'CSV','.md':'Markdown','.m':'MATLAB source','.py':'Python source','.pdf':'PDF'}.get(ext,'text or binary')
 return a

def main():
 ident=json.loads((OUT/'evaluations/identity.json').read_text());m=json.loads((OUT/'METRICS.json').read_text())
 assert sha(OUT/'AUDIT_PREREGISTRATION.md')==ident['preregistration_sha256']
 for a in ident['inputs']:assert sha(ROOT/a['path'])==a['sha256']
 for a in ident['production_file_checks']:assert sha(Path(a['path']))==a['expected']
 # Verify exact final states and valid FD float serialization once more.
 for c in ident['cases']:
  n=c['nelx'];d=sio.loadmat(OUT/'evaluations'/f'spectral_{n}.mat',squeeze_me=True);r=d['rho']
  assert hashlib.sha256(r.astype('<f8').tobytes()).hexdigest()==c['rho_sha256']
  fd=sio.loadmat(OUT/'evaluations'/f'fd_{n}.mat',squeeze_me=True)['fd'];assert fd.dtype.kind=='f'
  assert np.array_equal(np.unique(fd[:,2]),[1e-4,3e-4,1e-3]);assert np.isfinite(fd).all()
  assert np.all((fd[:,1]>=.001)&(fd[:,1]<=1));assert m['stationarity'][str(n)]['FD']['validated_raw']
  assert np.max(abs(d['Fraw'][:,0,0]-d['gRaw']))<1e-10
  assert np.max(abs(d['gK']+d['gM']-d['gRaw']))<1e-10
  assert abs(float(400*np.mean(r*(1-r)))-m['geometry'][str(n)]['Mnd_percent'])<1e-10
  assert len(np.loadtxt(OUT/'evaluations'/f'trajectory_{n}.csv',delimiter=',',skiprows=1))==c['iteration']
 req=['AUDIT_PREREGISTRATION.md','PROVENANCE.md','FORMULATION_RECOVERY.md','EVIDENCE_IDENTITY.md','GRAYNESS_GEOMETRY.md','SENSITIVITY_DECOMPOSITION.md','MASS_STIFFNESS_BALANCE.md','FILTER_GRADIENT_AUDIT.md','KKT_STATIONARITY.md','FINITE_DIFFERENCE_VALIDATION.md','MULTIPLICITY_RELATION.md','GRAYNESS_TRAJECTORY.md','CROSS_MESH_REGIME_CHANGE.md','CAUSAL_RANKING.md','PROJECTION_DECISION.md','P_CONTINUATION_DECISION.md','PERFORMANCE_IMPLICATIONS.md','REPORT.md','MASTER_METRICS.csv','METRICS.json']
 assert all((OUT/p).is_file() for p in req);assert (OUT/'REPORT.md').read_text().startswith('BOTTOM LINE\n')
 assert not (OUT/'runs').exists()
 # Static call guard: code lines (comments stripped) must not call optimizers.
 code='\n'.join(line.split('%')[0] for line in (OUT/'scripts/frozen_evaluate.m').read_text().splitlines())
 assert not re.search(r'\b(olhoffSolve|olhoffOpt|innerLoop\w*|mmasub|subsolv|cp_run|cp_fixedwork)\s*\(',code)
 figlist=sorted((OUT/'figures').glob('*.png'));assert len(figlist)>=16
 for p in OUT.rglob('*.json'):
  if p.name not in ['EVIDENCE.json','DATA_MANIFEST.json']:json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 validation={'generated_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_containers_verified':len(ident['inputs']),'production_files_verified':len(ident['production_file_checks']),'endpoint_rho_hashes_verified':3,'float_FD_tables_verified':3,'required_documents_verified':len(req),'figures':len(figlist),'forbidden_optimizer_calls_in_evaluator':0,'optimization_runs':0,'rho_updates':0,'no_runs_directory':True,'preregistration_unchanged':True,'production_unchanged':True,'verdict':'AUDIT_ARTIFACT_INTEGRITY_PASS'}
 (OUT/'evaluations/FINAL_VALIDATION.json').write_text(json.dumps(validation,indent=2))
 inputs=[desc(ROOT/a['path']) for a in ident['inputs']]
 prior=[BASE/'SOURCE_MANIFEST.json',BASE/'diagnostics/three_rung_architecture/EVIDENCE.json',BASE/'diagnostics/three_rung_architecture/METRICS.json',BASE/'diagnostics/three_rung_architecture/COUNTERFACTUAL_VALIDITY.md',BASE/'diagnostics/three_rung_architecture/PROVENANCE.md',BASE/'diagnostics/two_branch_controller_validation/runs/C400x50_record.json',BASE/'diagnostics/two_branch_controller_validation/PROVENANCE.md',BASE/'diagnostics/three_rung_canary_preflight/EVIDENCE.json',BASE/'diagnostics/three_rung_canary_preflight/PROVENANCE.md',BASE/'diagnostics/three_rung_canary_preflight/scripts/cp_confighash.py',BASE/'diagnostics/three_rung_canary_preflight/runs/C480x60_three_rung_record.json',BASE/'diagnostics/three_rung_canary_preflight/runs/C800x100_three_rung_record.json',ROOT/'references/Du2007_Topological.pdf',ROOT/'docs/olhoff_penalty_continuation_experiment.md']
 inputs.extend(desc(p) for p in prior)
 evaluations=[desc(p,'scratch' if 'INVALID' in p.name else 'required') for p in sorted((OUT/'evaluations').iterdir()) if p.is_file()]
 ev={'schema':'olhoff_current_evidence/1','study':'gray_kkt_forensic_audit','implementation':'analysis/OlhoffCurrent','evidenceRoot':str((OUT/'evaluations').relative_to(ROOT)),'sourceTree':ident['impl_tree'],'implTree':ident['impl_tree'],'generated':validation['generated_UTC'],'optimization_runs':0,'rho_updates':0,'analysis_label':'OFFLINE FROZEN-STATE ANALYSIS — NOT OPTIMIZATION','artifacts':inputs+evaluations,'verdicts':m['verdicts'],'note':'All paths are repository-relative. INVALID_INTEGER_SERIALIZATION files are declared scratch and never support a conclusion. Prior authoritative evidence is reused read-only. No original optimization result is generated in this audit.'}
 (OUT/'EVIDENCE.json').write_text(json.dumps(ev,indent=2))
 files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['FINAL_SHA256.txt','DATA_MANIFEST.json']]
 manifest={'schema':'gray_forensic_data_manifest/1','generated':validation['generated_UTC'],'HEAD':ident['HEAD'],'branch':ident['branch'],'impl_tree':ident['impl_tree'],'optimization_runs':0,'rho_updates':0,'files':[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'inputs':inputs,'exclusions':['DATA_MANIFEST.json (self)','FINAL_SHA256.txt (self-referential cycle)','__pycache__ (disposable interpreter cache)']}
 (OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 files.append(OUT/'DATA_MANIFEST.json');files.sort()
 (OUT/'FINAL_SHA256.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in files))
 for line in (OUT/'FINAL_SHA256.txt').read_text().splitlines():h,p=line.split('  ',1);assert sha(OUT/p)==h
 print(json.dumps(validation,indent=2));print(f'FINAL_SHA256 verified: {len(files)} files')
if __name__=='__main__':main()
