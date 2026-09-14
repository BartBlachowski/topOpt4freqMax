from pathlib import Path
import os,json,csv,numpy as np,h5py
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'evaluations/mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
S=Path(__file__).resolve().parents[1];E=S/'evaluations';F=S/'figures';F.mkdir(exist_ok=True)
NAMES=['B0_CURRENT_REPEATED_MMA','S1_PERSISTENT','S2_ASYINIT_001','S3_CANONICAL_CLAMP','S4_SUBSOLV_ACCURACY','S5_UNIT_BOX','G0_UNSAFE','G1_GCMMA','G2_GCMMA_ACCURATE','S34_CLAMP_ACCURACY','B0_RETAINED_5000']
LABEL={'B0_CURRENT_REPEATED_MMA':'B0 current','S1_PERSISTENT':'S1 persistence identity','S2_ASYINIT_001':'S2 init .01','S3_CANONICAL_CLAMP':'S3 cap 10','S4_SUBSOLV_ACCURACY':'S4 accuracy 1e-12','S5_UNIT_BOX':'S5 unit box','G0_UNSAFE':'G0 no safeguard','G1_GCMMA':'G1 GCMMA','G2_GCMMA_ACCURATE':'G2 GCMMA, accurate','S34_CLAMP_ACCURACY':'S34 cap + accuracy','B0_RETAINED_5000':'B0 retained 5000'}
D={n:json.loads((E/(n+'.json')).read_text()) for n in NAMES if (E/(n+'.json')).exists()}
O=json.loads((E/'oracle_identity.json').read_text());C=json.loads((E/'SOCP_COST.json').read_text());T=json.loads((E/'structure.json').read_text())
with h5py.File(E/'structure.mat') as f:
 rr=np.asarray(f['rho']).ravel();bb=np.asarray(f['boundCategory']).ravel();dd=np.asarray(f['oracle']).ravel()
 T['gray_bound_check']={name:{'count':int(mask.sum()),'interior':int(np.sum(bb[mask]==0)),'min_abs_oracle_increment':float(np.min(abs(dd[mask])))} for name,mask in [('gray',(rr>.1)&(rr<.9)),('mid',(rr>=.4)&(rr<=.6))]}
rows=[]
for n,d in D.items():
 if 'RETAINED' not in n:
  for h in d['history']:
   h['auditEvaluations']=(3 if n.startswith('G') else 4)*h['iter']
   h['algorithmGradientsActuallyComputed']=h['nonlinearEvaluations']
 for h in d['history']:
  rows.append(dict(method=n,evidence='retained' if 'RETAINED' in n else 'fresh',**h))
for c in C:rows.append(dict(method='SOCP',replicate=c['replicate'],evidence='fresh',**c['metric']))
keys=['method','evidence']+sorted(set().union(*(r.keys() for r in rows))-{'method','evidence'})
with (S/'MASTER_METRICS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def arr(n,key):return np.array([h.get(key,np.nan) if h.get(key) is not None else np.nan for h in D[n]['history']],float)
def at(n,calls):
 h=[h for h in D[n]['history'] if h['calls']<=calls]
 return h[-1] if h else None

def metrics_pass(h):return bool(h['fidelity'])
comparisons=[]
for a,b,label in [('B0_CURRENT_REPEATED_MMA',n,LABEL.get(n,n)) for n in NAMES[2:6]+['S34_CLAMP_ACCURACY']]+[('G0_UNSAFE','G1_GCMMA','safeguard'),('G1_GCMMA','G2_GCMMA_ACCURATE','GCMMA accuracy'),('S3_CANONICAL_CLAMP','S34_CLAMP_ACCURACY','accuracy at canonical clamp'),('S4_SUBSOLV_ACCURACY','S34_CLAMP_ACCURACY','clamp at tight accuracy')]:
 if a not in D or b not in D:continue
 vals=[]
 for budget in [100,500]:
  common=min(budget,D[a]['history'][-1]['calls'],D[b]['history'][-1]['calls'])
  x=at(a,common);y=at(b,common)
  if x is None or y is None:continue
  gain=y['gainRecovery']-x['gainRecovery'];reduction=1-y['d2']/x['d2'];feas=y['constraintResidual']<=max(1e-8,x['constraintResidual']+1e-8)
  vals.append(dict(calls=budget,registered_checkpoint_complete=(x['calls']==budget and y['calls']==budget),actual_calls=[x['calls'],y['calls']],recovery_change=gain,d2_reduction=reduction,feasibility_ok=feas,material=gain>=.10 and reduction>=.20 and feas))
 strong=len(vals)==2 and all(v['material'] and v['registered_checkpoint_complete'] for v in vals)
 moderate=any(v['material'] for v in vals)
 cls='STRONG CAUSAL EVIDENCE' if strong else 'MODERATE CAUSAL EVIDENCE' if moderate else 'WEAK ASSOCIATION'
 single_factor=not (a=='B0_CURRENT_REPEATED_MMA' and b=='S34_CLAMP_ACCURACY')
 if not single_factor:cls='JOINT INTERVENTION; not a single-factor cause'
 if label=='safeguard':cls='EVIDENCE AGAINST' if all(abs(v['recovery_change'])<1e-14 and abs(v['d2_reduction'])<1e-14 for v in vals) else cls
 comparisons.append(dict(control=a,treatment=b,factor=label,single_factor=single_factor,classification=cls,checkpoints=vals))
stop={}
for n,d in D.items():
 h=d['history'];x=arr(n,'relStep');y=arr(n,'d2');mask=np.isfinite(x)&np.isfinite(y)
 def first(fun):return next((v['iter'] for v in h if fun(v)),None)
 r=dict(first_production=first(lambda v:v['iter']>=5 and v.get('relStep',1)<.05),first_objective=first(lambda v:abs(v['scaled_gap'])<=1e-8),first_kkt=first(lambda v:v['kkt']<=1e-6 and v['kktMax']<=1e-5),first_distance=first(lambda v:v['d2']<=.01 and v['dinf']<=.1),first_feasible=first(lambda v:v['constraintResidual']<=1e-8 and v['boxViolation']<=1e-10),first_active_stable=first(lambda v:v.get('activeStableCount',0)>=20),first_fidelity=first(metrics_pass),false_stop_fraction=None)
 if np.count_nonzero(mask)>2:
  r['pearson_relStep_d2']=float(pearsonr(x[mask],y[mask]).statistic);r['spearman_relStep_d2']=float(spearmanr(x[mask],y[mask]).statistic)
 gap=np.abs(arr(n,'scaled_gap'));gapmask=np.isfinite(x)&np.isfinite(gap)
 if np.count_nonzero(gapmask)>2:
  r['pearson_relStep_absGap']=float(pearsonr(x[gapmask],gap[gapmask]).statistic);r['spearman_relStep_absGap']=float(spearmanr(x[gapmask],gap[gapmask]).statistic)
 diagnostics={
  'production':lambda v:v['iter']>=5 and v.get('relStep',1)<.05,
  'objective':lambda v:abs(v['scaled_gap'])<=1e-8,
  'kkt':lambda v:v['kkt']<=1e-6 and v['kktMax']<=1e-5,
  'distance':lambda v:v['d2']<=.01 and v['dinf']<=.1,
  'feasible':lambda v:v['constraintResidual']<=1e-8 and v['boxViolation']<=1e-10,
  'active_stable':lambda v:v.get('activeStableCount',0)>=20,
  'fidelity':metrics_pass}
 r['persistence']={}
 for criterion,predicate in diagnostics.items():
  flags=[bool(predicate(v)) for v in h];streak=longest=0
  for flag in flags:
   streak=streak+1 if flag else 0;longest=max(longest,streak)
  r['persistence'][criterion]={'observed_hits':sum(flags),'observed_checkpoints':len(flags),'terminal_pass':flags[-1],'longest_consecutive_iterations':None if 'RETAINED' in n else longest,'sparse_history':'RETAINED' in n}
 hstop=[v for v in h if v['iter']>=5 and v.get('relStep',1)<.05]
 if hstop:r['false_stop_fraction']=float(np.mean([not v['fidelity'] for v in hstop]))
 r['gain_decreases']=int(np.sum(np.diff(arr(n,'gainRecovery'))<0));r['d2_increases']=int(np.sum(np.diff(arr(n,'d2'))>0))
 r['sparse']=('RETAINED' in n);stop[n]=r

summary={'oracle':O,'methods':{n:{'out':d['out'],'last':d['history'][-1]} for n,d in D.items()},'comparisons':comparisons,'stopping':stop,'socp':C,'structure':T,'preregistration':(E/'preregistration_sha256.txt').read_text().strip(),'complete':all(n in D for n in NAMES)}
(S/'METRICS.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.2})
colors=plt.cm.tab10(np.linspace(0,1,10));color={n:colors[i%10] for i,n in enumerate(NAMES)};color['B0_CURRENT_REPEATED_MMA']='#202e45';color['B0_RETAINED_5000']='#202e45'
def save(fig,num,name):
 fig.savefig(F/f'FIG_{num:02d}_{name}.png',bbox_inches='tight');plt.close(fig)
with h5py.File(E/'structure.mat') as f:
 data={k:np.asarray(f[k]).reshape(-1) for k in ['rho','oracle','reducedCost','filteredSensitivity','rawSensitivity','effectiveSensitivity','boundCategory','densityClass']}
 maps={int(np.asarray(f[r]).ravel()[0]):np.asarray(f[f['maps']['drho'][()].ravel()[i]]).ravel() for i,r in enumerate(f['maps']['iter'][()].ravel())}

def mapfig(a,title,num,name,v=None,cmap='RdBu_r'):
 fig,ax=plt.subplots(figsize=(13,2.5));a=np.asarray(a).reshape((60,480),order='F');v=np.max(np.abs(a)) if v is None else v
 im=ax.imshow(a,origin='upper',cmap=cmap,vmin=-v,vmax=v,aspect='equal');ax.set(title=title,xlabel='Element column',ylabel='Row');ax.grid(False);fig.colorbar(im,ax=ax,pad=.015,shrink=.75);save(fig,num,name)
mapfig(data['oracle'],'Certified oracle increment (never applied)',1,'oracle_drho',.01)
mapfig(maps[19],'B0 production iteration 19 increment',2,'B0_19_drho',.01)
mapfig(maps[5000],'B0 retained iteration 5000 increment',3,'B0_5000_drho',.01)
mapfig(data['oracle']-maps[19],'Oracle minus B0 production increment',4,'oracle_minus_19',.02)
mapfig(data['oracle']-maps[5000],'Oracle minus B0 retained 5000 increment',5,'oracle_minus_5000',.02)

def lineplot(key,title,num,name,methods=None,xkey='iter',logy=False,target=None):
 fig,ax=plt.subplots(figsize=(10,5));methods=methods or [n for n in NAMES if n not in ['S1_PERSISTENT','B0_RETAINED_5000']]
 for n in methods:
  if n not in D:continue
  x=arr(n,xkey);y=arr(n,key);valid=np.isfinite(x)&np.isfinite(y)
  if logy:y=np.maximum(y,1e-16)
  ax.plot(x[valid],y[valid],label=LABEL[n],color=color[n],lw=1.4,ls='--' if 'RETAINED' in n else '-')
 if target is not None:ax.axhline(target,color='#33865b',ls=':',label='Oracle / fidelity threshold')
 ax.set(xlabel={'iter':'Accepted approximation iteration','calls':'Approximation solves (including rejected trials)','nonlinearEvaluations':'Algorithm constraint evaluations'}[xkey],ylabel=key,title=title)
 ax.set_xscale('log');
 if logy:ax.set_yscale('log')
 ax.legend(fontsize=8,ncol=2);save(fig,num,name)
allb=[n for n in NAMES if n!='S1_PERSISTENT']
lineplot('gainRecovery','Recovery of certified beta gain',6,'gain_vs_iteration',allb,target=1)
lineplot('d2','Normalized distance to oracle',7,'d2_vs_iteration',allb,logy=True,target=.01)
lineplot('dinf','Maximum increment disagreement / move',8,'dinf_vs_iteration',allb,logy=True,target=.1)
lineplot('kkt','Exact-problem KKT residual (feasibility checked separately)',9,'kkt_vs_iteration',allb,logy=True,target=1e-6)
fig,ax=plt.subplots(figsize=(9,5))
for n in ['B0_CURRENT_REPEATED_MMA','B0_RETAINED_5000']:
 if n in D:ax.scatter(arr(n,'relStep'),arr(n,'d2'),s=10,label=LABEL[n],alpha=.6)
ax.axvline(.05,ls=':',color='red',label='Production threshold');ax.axhline(.01,ls=':',color='green',label='d2 fidelity bar');ax.set(xscale='log',xlabel='Relative step',ylabel='d2',title='A small relative step does not certify proximity');ax.legend();save(fig,10,'step_vs_distance')
lineplot('signAgreement','Sign agreement on |oracle increment| >= .9 move',11,'sign_vs_iteration',allb,target=.99)
lineplot('boundAgreement','Same signed bound, conditional on oracle-bound entries',12,'bounds_vs_iteration',allb,target=.99)
lineplot('d2','Persistence already exists: chunk-boundary identity test',13,'persistence_vs_B0',['B0_CURRENT_REPEATED_MMA','S1_PERSISTENT'])
lineplot('gainRecovery','Conservative safeguard and approximate-solve accuracy',14,'gcmma_vs_B0',['B0_CURRENT_REPEATED_MMA','G0_UNSAFE','G1_GCMMA','G2_GCMMA_ACCURATE'],target=1)
lineplot('gainRecovery','Objective recovery versus algorithm evaluation work',15,'gain_vs_work',xkey='nonlinearEvaluations',target=1)
lineplot('kkt','Exact KKT versus approximation-solve work',16,'kkt_vs_work',xkey='calls',logy=True,target=1e-6)
fig,axs=plt.subplots(2,1,figsize=(13,5))
for ax,it in zip(axs,[19,5000]):
 err=(np.sign(maps[it])!=np.sign(data['oracle']));a=np.where(err,data['densityClass'],0).reshape((60,480),order='F');im=ax.imshow(a,cmap=matplotlib.colormaps['viridis'].resampled(5),vmin=0,vmax=4);ax.set_title(f'B0 {it}: sign disagreement by frozen density class');ax.grid(False)
fig.colorbar(im,ax=list(axs),ticks=range(5),shrink=.8).ax.set_yticklabels(['agree','void','gray shell','gray core','solid']);save(fig,17,'density_class_disagreement')
fig,ax=plt.subplots(figsize=(13,3));im=ax.imshow(data['boundCategory'].reshape((60,480),order='F'),cmap=matplotlib.colormaps['tab10'].resampled(6),vmin=-.5,vmax=5.5);ax.set_title('Oracle active bound categories');ax.grid(False);fig.colorbar(im,ax=ax,ticks=range(6),shrink=.75).ax.set_yticklabels(T['bound']['names']);save(fig,18,'oracle_bound_categories')
fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].scatter(data['reducedCost'],data['oracle']/.01,c=data['densityClass'],s=2,alpha=.4,cmap='viridis');axs[0].set(xlabel='Exact effective reduced cost q',ylabel='Oracle increment / move',title='Reduced-cost sign selects the bound');axs[0].set_xscale('symlog',linthresh=1e-6)
axs[1].scatter(data['filteredSensitivity'],data['effectiveSensitivity'],s=2,alpha=.3);axs[1].axvline(T['threshold']['volume_threshold'],ls=':',color='red');axs[1].set(xlabel='Filtered F11 / lamref',ylabel='Coupled effective sensitivity',title='SOC coupling retained in effective sensitivity');axs[1].set_xscale('symlog',linthresh=1e-5);axs[1].set_yscale('symlog',linthresh=1e-5);fig.tight_layout();save(fig,19,'threshold_structure')
fig,axs=plt.subplots(1,3,figsize=(13,5));ns=[n for n in NAMES if n in D and n not in ['S1_PERSISTENT','B0_RETAINED_5000']];labs=[LABEL[n] for n in ns]
for ax,key,title in zip(axs,['gainRecovery','d2','kkt'],['Gain recovery','d2','Exact KKT']):
 ax.barh(labs,[D[n]['history'][-1][key] for n in ns],color=[color[n] for n in ns]);ax.invert_yaxis();ax.set_title(title)
 if key=='kkt':
  ax.set_xscale('log');ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle('Preregistered terminal comparison (up to 500 calls; actual work in CSV)');fig.tight_layout();save(fig,20,'causal_comparison')
lineplot('d2','Design distance versus approximation solves',21,'d2_vs_calls',xkey='calls',logy=True,target=.01)
print('Generated',len(rows),'metric rows;',len(list(F.glob('FIG_*.png'))),'figures; complete=',summary['complete'])
