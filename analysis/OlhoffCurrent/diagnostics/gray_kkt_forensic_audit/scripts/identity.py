from pathlib import Path
import json,hashlib,h5py,numpy as np,subprocess,sys,copy
ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'analysis/OlhoffCurrent'; OUT=BASE/'diagnostics/gray_kkt_forensic_audit'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def decode(o,f):
 if isinstance(o,h5py.Group):return {k:decode(v,f) for k,v in o.items()}
 a=o[()]
 if o.attrs.get('MATLAB_empty',0):return '' if o.attrs.get('MATLAB_class')==b'char' else []
 if o.attrs.get('MATLAB_class')==b'char':return ''.join(chr(int(v)) for v in a.ravel())
 if h5py.check_dtype(ref=o.dtype):return [decode(f[v],f) for v in a.ravel()]
 if a.size==1:return bool(a.item()) if o.attrs.get('MATLAB_class')==b'logical' else a.item()
 return a.ravel().tolist() if 1 in a.shape else a.T.tolist()
sys.path.insert(0,str(BASE/'diagnostics/three_rung_canary_preflight/scripts'))
from cp_confighash import config_hash
def main():
 man=json.loads((BASE/'SOURCE_MANIFEST.json').read_text())
 checks=[{'path':str(BASE/'+impl'/a['path']), 'expected':a['sha256'],'actual':sha(BASE/'+impl'/a['path'])} for a in man['files']]
 assert all(a['expected']==a['actual'] for a in checks)
 inputs=[]; cases=[]
 for nx,ny,k,study in [(400,50,466,'three_rung_architecture'),(480,60,386,'three_rung_canary_preflight'),(800,100,468,'three_rung_canary_preflight')]:
  ev=json.loads((BASE/'diagnostics'/study/'EVIDENCE.json').read_text())
  arts=[a for a in ev['artifacts'] if f'C{nx}x{ny}' in a['path'] and a['path'].endswith('.mat')]
  for a in arts:
   p=ROOT/a['path']
   if not p.exists():p=ROOT/ev['evidenceRoot']/a['path']
   actual=sha(p);assert actual==a['sha256']
   with h5py.File(p) as f: shapes={key:{'matlab_shape':list(v.shape[::-1]),'dtype':str(v.dtype)} for key,v in f.items() if isinstance(v,h5py.Dataset)}
   inputs.append({'path':str(p.relative_to(ROOT)),'class':'required','sha256':actual,'bytes':p.stat().st_size,'format':'MATLAB v7.3 HDF5','datasets':shapes})
   if 'trajectory' in p.name:tp=p
  with h5py.File(tp) as f:
   rho=f['RHO'][k-1]; cfg=decode(f['cfg'],f); meta=decode(f['meta'],f)
   assert rho.size==nx*ny and meta['implTree']==man['tree_sha256']
   ch,lines=config_hash(cfg);assert ch==meta['cfgHash'],(nx,ch,meta['cfgHash'])
   effective=copy.deepcopy(cfg);effective['move']['levels']=[.04,.02,.01]
   effective_hash,_=config_hash(effective)
   assert float(f['hist/move'][()].ravel()[k-1])==0.01
   assert float(f['hist/stage'][()].ravel()[k-1])==3
   assert float(f['hist/exDecl'][()].ravel()[k-1])==1
   rh=hashlib.sha256(rho.astype('<f8').tobytes()).hexdigest()
   if nx==400:
    old=json.loads((BASE/'diagnostics/three_rung_architecture/METRICS.json').read_text())
    assert rh in json.dumps(old)
   else:
    rec=json.loads((BASE/f'diagnostics/{study}/runs/C{nx}x{ny}_three_rung_record.json').read_text());assert rh==rec['rho_sha256']
    sp=tp.with_name(tp.name.replace('trajectory','state'))
    with h5py.File(sp) as sf:assert np.array_equal(sf['state/rho'][()].ravel(),rho)
   assert cfg['move']['continuation']['signal']=='stageExhaustion' and cfg['stop']['rule']=='stageExhaustion'
   assert cfg['material']['stiffness']['p']==3 and cfg['material']['mass']['q']==1
   (OUT/'evaluations'/f'config_{nx}.json').write_text(json.dumps(cfg,indent=2))
   case={'mesh':f'{nx}x{ny}','nelx':nx,'nely':ny,'iteration':k,'source_study':study,'trajectory':str(tp.relative_to(ROOT)),'rho_sha256':rh,'impl_tree':meta['implTree'],'saved_config_hash':meta['cfgHash'],'effective_three_rung_config_hash':effective_hash,'status':'CONVERGED_EXACT_S3_COUNTERFACTUAL' if nx==400 else rec['status'],'saved_levels':cfg['move']['levels'],'effective_levels':[.04,.02,.01],'move':.01,'stage':3,'Mnd_percent_identity':float(400*np.mean(rho*(1-rho))),'meta':meta}
   cases.append(case)
 result={'verdict':'GRAY_FORENSICS_EVIDENCE_PASS','branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'impl_tree':man['tree_sha256'],'production_file_checks':checks,'cases':cases,'inputs':inputs,'optimization_runs':0,'density_updates':0,'preregistration_sha256':sha(OUT/'AUDIT_PREREGISTRATION.md')}
 (OUT/'evaluations/identity.json').write_text(json.dumps(result,indent=2))
 print(result['verdict']);print([(c['mesh'],c['rho_sha256']) for c in cases])
if __name__=='__main__':main()
