#!/usr/bin/env python3
"""Audit only: read stored arrays/configuration; no MATLAB, solver, or optimization calls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/olhoff_audit_mpl')
from pathlib import Path
import json,csv,re,hashlib,subprocess,datetime
import numpy as np
import h5py
from scipy import stats,ndimage
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
CAMP=ROOT/'examples/Performance/conference_benchmark/campaign_9mesh_r2'
DIAG=ROOT/'analysis/OlhoffCurrent/diagnostics'
INPUTS={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def track(p):
 p=Path(p);rel=str(p.relative_to(ROOT))
 if rel not in INPUTS:INPUTS[rel]={'path':rel,'present':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'sha256':sha(p) if p.exists() else None}
 return p
def readj(p):return json.loads(track(p).read_text())
def clean(v):
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def writej(name,v): (OUT/name).write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def writecsv(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:clean(v) for k,v in r.items()} for r in rows])
def decode(f,o):
 if isinstance(o,h5py.Group):return {k:decode(f,v) for k,v in o.items()}
 if o.attrs.get('MATLAB_empty',False):return '' if o.attrs.get('MATLAB_class',b'')==b'char' else []
 a=o[()];cls=o.attrs.get('MATLAB_class',b'')
 if cls==b'char':return ''.join(chr(int(x)) for x in np.asarray(a).ravel() if x)
 if h5py.check_dtype(ref=o.dtype):
  v=[decode(f,f[x]) for x in a.ravel()];return v[0] if len(v)==1 else v
 if cls==b'logical':a=a.astype(bool)
 if a.size==1:return a.item()
 return a.T.squeeze().tolist()
def flat(d,p=''):
 z={}
 for k,v in d.items():
  q=p+'.'+k if p else k
  if isinstance(v,dict):z.update(flat(v,q))
  else:z[q]=v
 return z
def matlab_show(v):
 if isinstance(v,str):return v
 if isinstance(v,bool):return str(v).lower()
 if isinstance(v,list):return '['+' '.join(matlab_show(x) for x in v)+']'
 if isinstance(v,(float,int)):return format(v,'.17g')
 raise ValueError(v)
def config_hash(c):
 schema=track(ROOT/'analysis/OlhoffCurrent/+impl/architecture/+olh/+config/schema.m').read_text()
 paths=re.findall(r"^'([^']+)'\s*,",schema,re.M);f=flat(c)
 lines=[p+'='+('<excluded>' if p=='runtime.name' else matlab_show(f[p])) for p in paths]
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest(),lines
manifest=readj(CAMP/'benchmark_manifest.json');results=readj(CAMP/'benchmark_results.json')
jruns=[r for r in results['runs'] if r['method_key']=='olhoff']
source=[]
for v in manifest['source_hashes'].values():
 p=track(ROOT/v['path']);source.append({**v,'actual_sha256':sha(p),'match':sha(p)==v['sha256']})
sm=readj(ROOT/'analysis/OlhoffCurrent/SOURCE_MANIFEST.json');lines=[];implchecks=[]
for r in sm['files']:
 p=track(ROOT/sm['root']/r['path']);h=sha(p);lines.append(r['path']+'  '+h);implchecks.append(h==r['sha256'])
tree=hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()
# Record all actual files as well; ignore only known filesystem/editor artifacts.
actual=[str(p.relative_to(ROOT/sm['root'])) for p in (ROOT/sm['root']).rglob('*') if p.is_file() and p.name!='.DS_Store' and not p.name.endswith(('.asv','~'))]
extras=sorted(set(actual)-{r['path'] for r in sm['files']})
rows=[];integrity=[];configs=[];dens=[];logs=[];rawcompare=[]
with h5py.File(track(CAMP/'benchmark_records.mat')) as f:
 g=f['records'];nr=g['mesh'].size
 for i in range(nr):
  r={k:decode(f,f[v[()].ravel()[i]]) for k,v in g.items() if k not in ['evaluator','implementation_provenance','resolved_implementation']}
  if r['method_key']!='olhoff':continue
  j=next(x for x in jruns if list(x['mesh'])==list(r['mesh']));x=np.asarray(r['x']);c=r['effective_config'];configs.append(c);nx,ny=map(int,r['mesh']);ne=nx*ny;s=r['stopping'];n=int(r['counts']['outer_iterations']);t=r['times'];w=np.array(r['omega']);dens.append(x.reshape((ny,nx),order='F'))
  ch,cl=config_hash(c)
  (OUT/f'config_{nx}x{ny}.txt').write_text('\n'.join(cl)+'\n')
  same={k:clean(r[k])==clean(j[k]) for k in ['mesh','counts','times','stopping','solver_log','omega','status','error','effective_config_hash','ok']}
  rawcompare.append({'mesh':f'{nx}x{ny}',**same})
  changes=[z for z in r['solver_log'] if 'move limit just changed' in z];lastchange=int(re.search(r'iter (\d+)',changes[-1])[1]) if changes else None
  warnings=[z for z in r['solver_log'] if '(25b) undefined' in z]
  R=c['filter']['radiusPhysical'];radius=R/(c['domain']['b']/ny)
  row={'mesh':f'{nx}x{ny}','policy':'legacy_beta_four_rung','config_resolved_local':c['provenance']['resolvedAt'],'exact_solver_start':None,'exact_solver_end':None,'nelx':nx,'nely':ny,'NE':ne,'free_DOF':2*(nx+1)*(ny+1)-4,'h_over_b':1/ny,'rminEl':radius,'status':r['status'],'outer':n,'inner_MMA':r['counts']['inner_iterations_total'],'mean_inner_per_outer':r['counts']['inner_iterations_per_outer_mean'],'max_inner_per_outer':None,'omega1':w[0],'omega2':w[1],'omega3':w[2],'gap12':(w[1]-w[0])/w[0],'multiplicity_N':s['final_multiplicity'],'volume':float(x.mean()),'volume_error':float(x.mean()-.5),'M_nd':float(100*np.mean(4*x*(1-x))),'grayness':float(np.mean(4*x*(1-x))),'intermediate_fraction_01_09':float(np.mean((x>.1)&(x<.9))),'mid_fraction_04_06':float(np.mean((x>.4)&(x<.6))),'final_move':s['final_move_limit'],'final_stage':s['final_ladder_stage'],'final_event':'designChange_with_settledMove','S1_E_iteration':None,'S1_branch':None,'S2_E_iteration':None,'S2_branch':None,'S3_E_iteration':None,'S3_branch':None,'first_terminal_E':None,'last_move_change_iteration':lastchange,'stage_terminal_observed_duration':n-lastchange+1,'terminal_max_abs_drho':s['final_max_density_change'],'terminal_l2_drho':s['final_l2_density_change'],'terminal_rms_drho':s['final_rms_density_change'],'eps_l2':s['eps_l2'],'amp_over_tol':s['final_l2_density_change']/s['eps_l2'],'runtime_total_s':t['total_wall_time_s'],'eigen_s':t['eigen_time_s'],'gradient_s':t['gradient_time_s'],'inner_s':t['inner_time_total_s'],'other_s':t['outer_bookkeeping_time_s']+t['overhead_time_s'],'inner_share_pct':t['inner_time_share_pct'],'eigen_share_pct':100*t['eigen_time_s']/t['total_wall_time_s'],'peak_memory':None,'inner_nonconverged':s['n_inner_not_converged'],'multJ_warning_count':len(warnings),'effective_config_hash':r['effective_config_hash'],'recomputed_config_hash':ch}
  for q in [20,50,100]:
   for metric in ['omega1','M_nd']:row[f'{metric}_change_last_{q}']=None
  rows.append(row);logs.append({'mesh':row['mesh'],'log':r['solver_log']})
  integrity.append({'mesh':row['mesh'],'classification_intended_campaign':'INVALID','classification_legacy_endpoint':'VALID_WITH_CAVEAT','raw_json_match':all(same.values()),'config_hash_match':ch==r['effective_config_hash'],'density_shape_match':x.size==ne,'density_finite':bool(np.isfinite(x).all()),'density_bounds_ok':bool(np.min(x)>=.001 and np.max(x)<=1),'density_grayness_match':abs(row['grayness']-s['final_grayness'])<1e-12,'eigen_finite':bool(np.isfinite(w).all()),'ordered_positive_frequencies':bool(np.all(np.diff(w)>0) and w[0]>0),'rmin_correct':abs(radius-.06*ny)<1e-12,'one_thread':c['runtime']['singleThread'],'cap':c['runtime']['maxOuter'],'error_empty':r['error']=='','inner_failures':s['n_inner_not_converged'],'full_trajectory_retained':False,'hist_field_in_archive':('hist' in r),'telemetry_empty':r['telemetry']==[],'status':r['status'],'reason':'Wrong controller; no iteration/density history; no original output digest seal'})
writecsv('MASTER_TABLE.csv',rows);writecsv('INTEGRITY_TABLE.csv',integrity);writej('effective_configs.json',configs);writej('solver_logs.json',logs)
# Verify agreement among every manifest declared canonical config and actual stored config (timestamp/metadata may differ).
mc=[v['canonical'] for v in manifest['method_configurations'] if 'canonical' in v]
config_diffs=[]
for c,d in zip(configs,mc):
 a,b=flat(c),flat(d);config_diffs.append({k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k) and not k.startswith('provenance.')})
base=flat(configs[0]);cross=[]
for c in configs:
 cc=flat(c);cross.append({k:[base.get(k),cc.get(k)] for k in set(base)|set(cc) if base.get(k)!=cc.get(k) and not k.startswith('provenance.')})
writej('verification.json',{'source_hashes':source,'impl_tree':tree,'expected_tree':sm['tree_sha256'],'all_impl_files_match':all(implchecks),'extra_impl_files':extras,'raw_json_comparison':rawcompare,'manifest_vs_actual_config_differences':config_diffs,'cross_mesh_config_differences':cross,'environment':manifest['environment'],'campaign_repository':manifest['repository'],'campaign_generated':manifest['generated_datetime'],'current_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
# Physical-coordinate comparison, common fine cell-centred grid, no symmetry alignment.
def resample(a,ny=400):
 h,w=a.shape;yy=(np.arange(ny)+.5)/ny;xx=(np.arange(8*ny)+.5)/ny
 y=np.clip(yy,.5/h,1-.5/h);x=np.clip(xx,4/w,8-4/w)
 yy,xx=np.meshgrid(y,x,indexing='ij')
 return RegularGridInterpolator(((np.arange(h)+.5)/h,(np.arange(w)+.5)*8/w),a,bounds_error=True)(np.stack([yy,xx],axis=-1))
def compare(a,b,ny):
 aa=resample(a,ny);bb=resample(b,ny);ta=aa>=.5;tb=bb>=.5
 ba=ta^ndimage.binary_erosion(ta,border_value=0);bc=tb^ndimage.binary_erosion(tb,border_value=0)
 da=ndimage.distance_transform_edt(~ba,sampling=1/ny);db=ndimage.distance_transform_edt(~bc,sampling=1/ny);bd=np.r_[db[ba],da[bc]]
 return {'L1':np.mean(np.abs(aa-bb)),'L2_RMS':np.sqrt(np.mean((aa-bb)**2)),'correlation':np.corrcoef(aa.ravel(),bb.ravel())[0,1],'IoU_05':np.sum(ta&tb)/np.sum(ta|tb),'boundary_mean_over_b':bd.mean(),'boundary_95pct_over_b':np.percentile(bd,95),'boundary_Hausdorff_over_b':bd.max()}
top=[]
for i in range(1,9):
 for ny in [200,400]:top.append({'pair':rows[i-1]['mesh']+' -> '+rows[i]['mesh'],'fine_NE':rows[i]['NE'],'grid_nely':ny,**compare(dens[i-1],dens[i],ny),'delta_M_nd':rows[i]['M_nd']-rows[i-1]['M_nd']})
writecsv('TOPOLOGY_METRICS.csv',top)
filterrows=[]
for a,r in zip(dens,rows):
 rad=r['rminEl'];k=np.arange(-int(np.ceil(rad))+1,int(np.ceil(rad)));yy,xx=np.meshgrid(k,k);weights=np.maximum(0,rad-np.hypot(xx,yy));mask=weights>0
 filterrows.append({'mesh':r['mesh'],'rminEl':rad,'positive_interior_stencil':int(mask.sum()),'center_weight_fraction':rad/weights.sum(),'weighted_rms_radius_over_b':np.sqrt(np.sum(weights*(xx**2+yy**2))/weights.sum())/r['nely'],'symmetry_x_L1':np.mean(np.abs(a-a[:,::-1])),'symmetry_y_L1':np.mean(np.abs(a-a[::-1,:])),'components_05_4connected':ndimage.label(a>=.5)[1]})
writecsv('FILTER_METRICS.csv',filterrows)
# Fits: independent variable is number of ELEMENTS; costs in seconds; log OLS intervals conditional on iid log residuals.
fits=[];residuals=[]
metrics=['runtime_total_s','eigen_s','gradient_s','inner_s','outer','inner_MMA','wall_per_outer_s','eigen_per_outer_s','gradient_per_outer_s','inner_per_outer_s','inner_s_per_MMA']
for r in rows:
 r['wall_per_outer_s']=r['runtime_total_s']/r['outer'];r['eigen_per_outer_s']=r['eigen_s']/r['outer'];r['gradient_per_outer_s']=r['gradient_s']/r['outer'];r['inner_per_outer_s']=r['inner_s']/r['outer'];r['inner_s_per_MMA']=r['inner_s']/r['inner_MMA']
for metric in metrics:
 for name,inds in [('all9',range(9)),('fine5',range(4,9)),('fine4',range(5,9))]:
  inds=list(inds);N=np.array([rows[i]['NE'] for i in inds]);y=np.array([rows[i][metric] for i in inds]);u=np.log(N);v=np.log(y);lr=stats.linregress(u,v);p=lr.slope;C=np.exp(lr.intercept);q=stats.t.ppf(.975,len(inds)-2);pred=C*N**p
  fits.append({'metric':metric,'subset':name,'n':len(inds),'C':C,'p':p,'R2_log':lr.rvalue**2,'C_CI95_lo':np.exp(lr.intercept-q*lr.intercept_stderr),'C_CI95_hi':np.exp(lr.intercept+q*lr.intercept_stderr),'p_CI95_lo':p-q*lr.stderr,'p_CI95_hi':p+q*lr.stderr,'RMSE_log':np.sqrt(np.mean((v-np.log(pred))**2))})
  residuals.extend({'metric':metric,'subset':name,'mesh':rows[i]['mesh'],'observed':yy,'predicted':pp,'residual':yy-pp,'relative_residual_pct':100*(yy-pp)/pp,'log_residual':np.log(yy/pp)} for i,yy,pp in zip(inds,y,pred))
writecsv('SCALING_FITS.csv',fits);writecsv('SCALING_RESIDUALS.csv',residuals);writecsv('MASTER_TABLE.csv',rows)
# Retained common-mesh counterfactuals. Recompute branch counters and endpoint morphology directly from stored old four-rung trajectories.
old=[];events=[];tails=[];raw_history_checks=[];stagework=[];countertop=[]
for nx,ny,study in [(160,20,'two_branch_controller_validation'),(240,30,'three_rung_resolution_240'),(320,40,'two_branch_controller_validation'),(400,50,'two_branch_controller_validation')]:
 tag=f'C{nx}x{ny}';rec=readj(DIAG/study/'runs'/f'{tag}_record.json');p=track(ROOT/rec['trajectory']);csvp=track(DIAG/study/'runs'/f'{tag}_iterations.csv');arr=np.genfromtxt(csvp,delimiter=',',names=True)
 with h5py.File(p) as f:
  H={k:np.asarray(v[()]).squeeze() for k,v in f['hist'].items() if isinstance(v,h5py.Dataset) and not h5py.check_dtype(ref=v.dtype)}
  X=np.array(f['RHO']);D=np.array(f['DRHO']);st=np.asarray(H['stage']).ravel().astype(int);mv=np.asarray(H['move']).ravel();tol=rec['tol'];ne=nx*ny;cfg=decode(f,f['cfg'])
  # Stored RHO arrays are time x elements. Validate post-update and inner increments separately.
  dx=np.diff(np.vstack([np.full((1,ne),.5),X]),axis=0);amp=np.linalg.norm(D,axis=1);dn=np.linalg.norm(dx,axis=1);cos=np.full(len(st),np.nan);net=np.full(len(st),np.nan);mc=cos.copy();mn=cos.copy();A=np.zeros(len(st),bool);B=A.copy();nA=np.zeros(len(st),int);nB=nA.copy();ev=[]
  for stage in np.unique(st):
   ii=np.flatnonzero(st==stage);s0=int(ii[0]);cnta=cntb=0;declared=False
   for k in ii:
    if k>s0 and dn[k]*dn[k-1]>0:cos[k]=dx[k]@dx[k-1]/(dn[k]*dn[k-1])
    if k>=s0+9:
     anchor=np.full(ne,.5) if k==9 else X[k-10]
     net[k]=np.linalg.norm(X[k]-anchor)/dn[k-9:k+1].sum()
    if k>=s0+19:
     mc[k]=np.nanmedian(cos[k-19:k+1]);mn[k]=np.nanmedian(net[k-19:k+1]);A[k]=(mc[k]<0 and mn[k]<.5 and amp[k]>=tol);B[k]=(mc[k]>0 and amp[k]<tol)
    cnta=cnta+1 if A[k] else 0;cntb=cntb+1 if B[k] else 0;nA[k]=cnta;nB[k]=cntb
    if not declared and max(cnta,cntb)>=20:
     ev.append({'mesh':f'{nx}x{ny}','stage':int(stage),'start':s0+1,'declaration':int(k+1),'duration':int(k-s0+1),'branch':'A' if cnta>=20 else 'B','amp_over_tol':amp[k]/tol,'medcos':mc[k],'mednet':mn[k]});declared=True
   ni=np.asarray(H['nInner']).ravel()[ii];stagework.append({'mesh':f'{nx}x{ny}','stage':int(stage),'move':mv[s0],'outer':len(ii),'inner':ni.sum(),'mean':ni.mean(),'max':ni.max(),'p50':np.median(ni),'p90':np.percentile(ni,90),'nonconverged':int(np.sum(np.asarray(H['innerConv']).ravel()[ii]==0))})
  e3=next(e for e in ev if e['stage']==3);k=e3['declaration'];rho3=X[k-1];rhof=X[-1];events.extend(ev)
  # Hist frequencies refer to pre-update designs; endpoint frequency uses stored independently evaluated state in prior audit.
  ana=readj(DIAG/('three_rung_architecture' if nx!=240 else 'three_rung_resolution_240')/'evidence/analysis.json')
  if nx!=240:states=ana['mesh'][f'm{nx}']
  else:states=ana
  # Persist keys for traceable selection; expected S3 and F based on existing frozen audit schema.
  if 'S3' not in states:raise ValueError(states.keys())
  endpoint=states['S3'];omega3=float(np.asarray(H['omega'])[k,0]);mnd3=100*np.mean(4*rho3*(1-rho3));inner3=np.asarray(H['nInner']).ravel()[:k].sum()
  old.append({'mesh':f'{nx}x{ny}','old_policy':'four_rung_E','old_status':rec['status'],'old_outer':len(st),'old_inner':rec['innerTotal'],'old_omega1':rec['omega1'],'old_M_nd':100*np.mean(4*rhof*(1-rhof)),'omega_endpoint_convention':'post-update rho_S3 from hist.omega(S3+1)','three_status':'CONVERGED_counterfactual_at_S3' if nx!=320 else 'CONVERGED_validated_CSV_raw_missing','three_outer':k,'three_inner':int(inner3),'three_omega1':omega3,'three_M_nd':mnd3,'delta_outer':k-len(st),'delta_inner':int(inner3)-rec['innerTotal'],'delta_omega1':omega3-rec['omega1'],'delta_M_nd':mnd3-100*np.mean(4*rhof*(1-rhof)),'terminal_branch':e3['branch'],'density_L1_S3_vs_F':np.mean(np.abs(rho3-rhof)),'threshold_flip_fraction':np.mean((rho3>=.5)!=(rhof>=.5)),'source_record':str((DIAG/study/'runs'/f'{tag}_record.json').relative_to(ROOT))})
  for q in [20,50,100]:
   z=k-1;ear=z-q
   if ear<0:continue
   w1=np.asarray(H['omega'])[:,0] if np.asarray(H['omega']).shape[0]==len(st) else np.asarray(H['omega'])[0]
   tails.append({'mesh':f'{nx}x{ny}','endpoint':'historical_S3','window_steps':q,'omega1_change_postupdate':w1[k]-w1[k-q],'omega1_change_pct_postupdate':100*(w1[k]/w1[k-q]-1),'omega1_change_preupdate':w1[z]-w1[ear],'omega1_change_pct_preupdate':100*(w1[z]/w1[ear]-1),'M_nd_change':100*(np.mean(4*X[z]*(1-X[z]))-np.mean(4*X[ear]*(1-X[ear]))),'density_net_L1':np.mean(np.abs(X[z]-X[ear])),'amp_over_tol':amp[z]/tol,'max_abs_drho':np.max(np.abs(D[z])),'medcos':mc[z],'mednet':mn[z],'stage_start':int(np.flatnonzero(st==3)[0]+1),'window_crosses_stage':bool(st[ear]!=3)})
  raw_history_checks.append({'mesh':f'{nx}x{ny}','n_history':len(st),'rho_shape':list(X.shape),'drho_shape':list(D.shape),'rho_finite':bool(np.isfinite(X).all()),'drho_finite':bool(np.isfinite(D).all()),'max_csv_amp_error':np.max(np.abs(amp-arr['exAmp'])),'A_all_match':bool(np.array_equal(A,arr['exA'].astype(bool))),'B_all_match':bool(np.array_equal(B,arr['exB'].astype(bool))),'nA_all_match':bool(np.array_equal(nA,arr['exNA'])),'nB_all_match':bool(np.array_equal(nB,arr['exNB'])),'max_Mnd_S3_error':abs(mnd3-endpoint['Mnd']),'inner_sum_matches_record':bool(H['nInner'].sum()==rec['innerTotal'])})
  current=rows[(nx-160)//80];a=dens[(nx-160)//80];b=rho3.reshape((ny,nx),order='F')
  countertop.append({'mesh':current['mesh'],'legacy_outer':current['outer'],'legacy_inner':current['inner_MMA'],'legacy_omega1':current['omega1'],'legacy_M_nd':current['M_nd'],'three_outer':k,'three_inner':int(inner3),'three_omega1':omega3,'three_M_nd':mnd3,'delta_outer':k-current['outer'],'delta_inner':int(inner3)-current['inner_MMA'],'delta_omega1':omega3-current['omega1'],'delta_M_nd':mnd3-current['M_nd'],'density_L1':np.mean(np.abs(a-b)),'threshold_flip_fraction':np.mean((a>=.5)!=(b>=.5))})
 del X,D,dx
writecsv('COMMON_MESH_COUNTERFACTUALS.csv',old);writecsv('LEGACY_VS_THREE_RUNG.csv',countertop);writecsv('HISTORICAL_CONTROLLER_EVENTS.csv',events);writecsv('HISTORICAL_TERMINAL_WINDOWS.csv',tails);writecsv('HISTORICAL_STAGE_WORK.csv',stagework);writej('historical_replay_checks.json',raw_history_checks)
# Declared evidence audit: do not repair stale hashes or missing files.
echecks=[]
for study in ['two_branch_controller_validation','three_rung_resolution_240','three_rung_architecture','three_rung_promotion_validation_retry1','three_rung_promotion_closure','move_activity_400']:
 e=readj(DIAG/study/'EVIDENCE.json')
 def visit(v):
  if isinstance(v,dict):
   if 'path' in v and 'sha256' in v:
    p=Path(v['path']); candidates=([ROOT/p] if str(p).startswith('analysis/') else [ROOT/e.get('evidenceRoot','')/p,DIAG/study/p,ROOT/p]) if not p.is_absolute() else [p];p=next((x for x in candidates if x.exists()),candidates[0])
    if not p.exists() and '/topOpt4freqMax/' in str(p):p=ROOT/str(p).split('/topOpt4freqMax/')[-1]
    if str(p).startswith(str(ROOT)):
     track(p);echecks.append({'study':study,'path':str(p.relative_to(ROOT)),'class':v.get('class'),'exists':p.exists(),'expected':v['sha256'],'actual':sha(p) if p.exists() else None,'match':p.exists() and sha(p)==v['sha256']})
   for x in v.values():visit(x)
  elif isinstance(v,list):
   for x in v:visit(x)
 visit(e)
writej('historical_evidence_checks.json',echecks)
seals=[]
for study in ['two_branch_controller_validation','three_rung_resolution_240','three_rung_architecture','three_rung_promotion_validation_retry1','three_rung_promotion_closure']:
 seal=track(DIAG/study/'FINAL_SHA256.txt')
 for line in seal.read_text().splitlines():
  m=re.match(r'^([0-9a-f]{64})  (.+)$',line)
  if not m:continue
  rel=re.sub(r'\s+\(\d+ bytes\)$','',m[2]);candidates=([ROOT/rel] if rel.startswith('analysis/') else [DIAG/study/rel,ROOT/rel,ROOT/'analysis/OlhoffCurrent/evidence'/study/rel])
  p=next((x for x in candidates if x.exists()),candidates[0]);track(p)
  seals.append({'study':study,'entry':rel,'path':str(p.relative_to(ROOT)),'exists':p.exists(),'match':p.exists() and sha(p)==m[1],'expected':m[1],'actual':sha(p) if p.exists() else None})
writej('historical_seal_checks.json',seals)
# Recheck candidate CSV scientific prefix against the currently retained oracle.
candrec=readj(DIAG/'three_rung_promotion_validation_retry1/runs/C320x40_three_rung_record.json')
candidate=np.genfromtxt(track(DIAG/'three_rung_promotion_validation_retry1/runs/C320x40_three_rung_iterations.csv'),delimiter=',',names=True)
oracle=np.genfromtxt(track(DIAG/'two_branch_controller_validation/runs/C320x40_iterations.csv'),delimiter=',',names=True)[:352]
cols=[k for k in candidate.dtype.names if k not in ['tOuter','prodStageShadow','prodMoveShadow']]
writej('C320_validation_recheck.json',{'raw_candidate_path':candrec['trajectory'],'raw_candidate_present':track(ROOT/candrec['trajectory']).exists(),'record_omega1':candrec['omega1'],'postupdate_oracle_omega1':next(r['three_omega1'] for r in old if r['mesh']=='320x40'),'candidate_rows':len(candidate),'compared_columns':cols,'differing_columns':[k for k in cols if not np.array_equal(candidate[k],oracle[k],equal_nan=True)],'inner_total':candidate['nInner'].sum()})

# Publication-quality plots: honest titles explicitly identify the legacy campaign.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
N=np.array([r['NE'] for r in rows]);ny=np.array([r['nely'] for r in rows]);h=1/ny
figdir=OUT/'figures'
def save(name):
 plt.savefig(figdir/(name+'.png'),bbox_inches='tight');plt.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close()
def line(name,fields,ylabel):
 fig,ax=plt.subplots(figsize=(7,4))
 for field,label in fields:ax.plot(N,[r[field] for r in rows],'o-',label=label)
 ax.set(xlabel='Number of elements, NE',ylabel=ylabel,title='Observed legacy beta campaign');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
 if len(fields)>1:ax.legend()
 save(name)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for a,x,l in zip(ax,[N,h],['Number of elements, NE','h/b = 1/nely']):a.plot(x,[r['omega1'] for r in rows],'o-');a.set(xlabel=l,ylabel=r'$\omega_1$ [rad/s]');a.grid(alpha=.2)
fig.suptitle('Legacy endpoints: refinement does not reach a plateau');fig.tight_layout();save('F01_omega1_refinement')
fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot(ny,[r['omega1'] for r in rows],'o-',label=r'$\omega_1$');ax[0].plot(ny,[r['omega2'] for r in rows],'s-',label=r'$\omega_2$');ax[0].legend();ax[0].set(ylabel='Frequency [rad/s]');ax[1].plot(ny,[100*r['gap12'] for r in rows],'o-');ax[1].set(ylabel='Relative gap [%]')
for a in ax:a.set(xlabel='Elements through height, nely');a.grid(alpha=.2)
fig.suptitle('Legacy endpoints: subspace size 2 does not imply degeneracy');fig.tight_layout();save('F02_spectrum')
line('F03_outer_iterations',[('outer','outer')],'Outer iterations');line('F04_inner_work',[('inner_MMA','inner')],'Cumulative inner MMA iterations')
fig,ax=plt.subplots(figsize=(7,4));ax.loglog(N,[r['runtime_total_s'] for r in rows],'o-',label='Observed legacy solves')
for sub in ['all9','fine5']:
 ft=next(z for z in fits if z['metric']=='runtime_total_s' and z['subset']==sub);ax.loglog(N,ft['C']*N**ft['p'],'--',label=f"{sub}: C={ft['C']:.4g}, p={ft['p']:.3f}")
ax.set(xlabel='Number of elements, NE',ylabel='Solver wall time [s]');ax.legend();ax.grid(alpha=.2,which='both');save('F05_total_runtime')
fig,ax=plt.subplots(figsize=(8,4));base=np.zeros(9)
for field,label in [('inner_s','Nested MMA'),('eigen_s','Assembly + eigensolve'),('gradient_s','Gradients + filtering'),('other_s','Other')]:
 v=np.array([r[field] for r in rows]);ax.bar(np.arange(9),v,bottom=base,label=label);base+=v
ax.set_xticks(np.arange(9),[r['mesh'] for r in rows],rotation=40,ha='right');ax.set(ylabel='Wall time [s]',title='Legacy campaign timing decomposition');ax.legend();save('F06_runtime_decomposition')
line('F07_grayness',[('grayness','Mean 4ρ(1−ρ)'),('intermediate_fraction_01_09','Fraction 0.1 < ρ < 0.9')],'Fraction / grayness')
fig,ax=plt.subplots(figsize=(7,4))
for stage in [1,2,3]:
 es=[e for e in events if e['stage']==stage];ax.plot([int(e['mesh'].split('x')[1]) for e in es],[e['declaration'] for e in es],'o-',label=f'S{stage}')
ax.set(xlabel='nely',ylabel='Declaration iteration',title='Historical E-controller only; fine-mesh observations absent');ax.legend();ax.grid(alpha=.2);save('F08_historical_E_declarations')
fig,ax=plt.subplots(figsize=(8,3));grid=np.full((3,9),np.nan)
for e in events:
 if e['stage']<=3:grid[e['stage']-1,(int(e['mesh'].split('x')[0])-160)//80]=0 if e['branch']=='A' else 1
ax.imshow(np.ma.masked_invalid(grid),aspect='auto',vmin=0,vmax=1,cmap='coolwarm')
for i in range(3):
 for j in range(9):ax.text(j,i,'NO E DATA' if np.isnan(grid[i,j]) else ['A','B'][int(grid[i,j])],ha='center',va='center',fontsize=8,color='black' if np.isnan(grid[i,j]) else 'white')
ax.set_xticks(range(9),[r['nely'] for r in rows]);ax.set_yticks(range(3),['S1','S2','S3']);ax.set(xlabel='nely',title='Prior historical branch map; NOT the nine-mesh campaign');save('F09_historical_branch_map')
fig,ax=plt.subplots(3,3,figsize=(15,5.6))
for a,x,r in zip(ax.ravel(),dens,rows):a.imshow(x,cmap='gray_r',vmin=0,vmax=1,extent=[0,8,0,1],interpolation='nearest');a.set_title(f"{r['mesh']}   ω₁={r['omega1']:.2f}   Mnd={r['M_nd']:.1f}%",fontsize=10);a.set_xticks([0,4,8]);a.set_yticks([0,1]);a.set_xlabel('x/b',labelpad=0)
fig.suptitle('Actual legacy final density fields — identical physical scale and density range');fig.tight_layout();save('F10_topology_atlas')
top400=[r for r in top if r['grid_nely']==400];fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot([r['fine_NE'] for r in top400],[r['L1'] for r in top400],'o-',label='L1');ax[0].plot([r['fine_NE'] for r in top400],[r['L2_RMS'] for r in top400],'s-',label='L2 RMS');ax[0].legend();ax[1].plot([r['fine_NE'] for r in top400],[r['IoU_05'] for r in top400],'o-');ax[1].set_ylabel('Thresholded IoU (ρ ≥ 0.5)')
for a in ax:a.set_xlabel('NE of finer mesh in adjacent pair');a.grid(alpha=.2)
fig.suptitle('Adjacent legacy designs, common physical grid');fig.tight_layout();save('F11_topology_differences')
fig,axs=plt.subplots(2,3,figsize=(13,7))
for a,metric in zip(axs.ravel(),metrics[:6]):
 for sub in ['all9','fine5']:
  z=[r for r in residuals if r['metric']==metric and r['subset']==sub];a.plot([int(r['mesh'].split('x')[0])*int(r['mesh'].split('x')[1]) for r in z],[r['relative_residual_pct'] for r in z],'o-',label=sub)
 a.axhline(0,color='gray',lw=.8);a.set(title=metric,ylabel='Residual / fitted [%]',xlabel='NE');a.grid(alpha=.2)
axs[0,0].legend();fig.tight_layout();save('F12_scaling_residuals')
# Final small machine-readable summaries.
writej('summary.json',{'master':rows,'counterfactuals':old,'legacy_comparison':countertop,'topology':top400,'filter':filterrows,'fits':fits,'input_count':len(INPUTS)})
writej('INPUT_MANIFEST.json',{'inputs':list(INPUTS.values())})
print(json.dumps({'rows':len(rows),'source_matches':sum(v['match'] for v in source),'source_total':len(source),'impl_match':tree==sm['tree_sha256'],'config_hashes_match':sum(r['config_hash_match'] for r in integrity),'replay':raw_history_checks,'fits_total':[z for z in fits if z['metric']=='runtime_total_s'],'counterfactuals':old},default=clean,indent=2))
