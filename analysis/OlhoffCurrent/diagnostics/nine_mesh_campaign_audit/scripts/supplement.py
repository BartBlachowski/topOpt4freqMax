"""Additional read-only historical CSV checks; no optimization."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[3];D=ROOT/'analysis/OlhoffCurrent/diagnostics'
def load(p):return json.loads(p.read_text())
def wr(n,x):
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
rows=[];checks=[];inputs=[]
master=list(csv.DictReader((OUT/'MASTER_TABLE.csv').open()))
for nx,ny,p in [(160,20,D/'move_stop/runs/baseline_160x20_iterations.csv'),(320,40,D/'move_stop/runs/baseline_320x40_iterations.csv'),(400,50,D/'move_activity_400/runs/P400_400x50_iterations.csv')]:
 inputs.append(p);a=np.genfromtxt(p,delimiter=',',names=True);r=next(r for r in master if r['mesh']==f'{nx}x{ny}');
 checks.append({'mesh':r['mesh'],'outer_match':len(a)==int(r['outer']),'inner_match':a['nInner'].sum()==float(r['inner_MMA']),'Mnd_error':a['Mnd_pct'][-1]-float(r['M_nd']),'l2_error':a['l2'][-1]-float(r['terminal_l2_drho']),'scope':'historical endpoint agreement; NOT campaign trajectory identity'})
 for q in [20,50,100]:
  if len(a)<=q:continue
  rows.append({'mesh':r['mesh'],'window':q,'omega_change_preupdate':a['omega1'][-1]-a['omega1'][-q-1],'omega_pct_preupdate':100*(a['omega1'][-1]/a['omega1'][-q-1]-1),'Mnd_change':a['Mnd_pct'][-1]-a['Mnd_pct'][-q-1],'inner_max_over_trajectory':a['nInner'].max(),'same_move_whole_window':bool(np.all(a['move'][-q-1:]==a['move'][-1])),'scope':'historical matched endpoint, not retained campaign history'})
wr('HISTORICAL_LEGACY_WINDOWS.csv',rows);wr('HISTORICAL_LEGACY_MATCH.csv',checks)
a=load(D/'three_rung_architecture/evidence/analysis.json');b=load(D/'three_rung_resolution_240/evidence/analysis.json');simp=[]
for nx in [160,240,320,400]:
 s=b if nx==240 else a['mesh'][f'm{nx}'];x=s['S2'];y=s['S3'];simp.append({'mesh':f'{nx}x{nx//8}','S2_iter':x['iteration'],'S3_iter':y['iteration'],'S2_omega_preupdate':x['omega1'],'S3_omega_preupdate':y['omega1'],'gain_to_S3_pct_preupdate':100*(y['omega1']/x['omega1']-1),'S2_Mnd':x['Mnd'],'S3_Mnd':y['Mnd'],'scope':'pre-update frequency convention inherited; morphology post-update'})
wr('SIMPLER_RULE_EVIDENCE.csv',simp)
fixed=[]
p=D/'move_stop/METRICS.json';inputs.append(p)
for r in load(p)['runs']:
 if 'fixedmove' in r['key']:fixed.append({'mesh':f"{r['mesh'][0]}x{r['mesh'][1]}",'policy':'fixed_0.04','status':r['status'],'outer':r['nOuter'],'inner':r['innerTotal'],'omega1':r['omega1'],'Mnd':r['Mnd_pct'],'native_stop_iter':r['nOuter'] if r['converged'] else None,'evidence':'historical summary; raw trajectory missing'})
p=D/'two_branch_maturity_240/METRICS.json';inputs.append(p);r=load(p)['runD'];fixed.append({'mesh':'240x30','policy':'fixed_0.04_extended','status':r['status'],'outer':r['nOuter'],'inner':r['innerTotal'],'omega1':r['omega1'],'Mnd':r['Mnd_final'],'native_stop_iter':r['nativeStopIter'],'evidence':'historical summary; raw trajectory missing'})
p=D/'fixedmove_400_dynamics/METRICS.json';inputs.append(p);r=load(p)['runC'];print('runC keys',r.keys());fixed.append({'mesh':'400x50','policy':'fixed_0.04_extended','status':r.get('status'),'outer':r.get('nOuter'),'inner':r.get('innerTotal'),'omega1':r.get('omega1'),'Mnd':r.get('Mnd_final'),'native_stop_iter':r.get('nativeStopIter'),'evidence':'historical summary; extended raw trajectory missing'})
wr('FIXED_MOVE_EVIDENCE.csv',fixed)
(OUT/'SUPPLEMENT_INPUTS.json').write_text(json.dumps([{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],indent=2)+'\n')
