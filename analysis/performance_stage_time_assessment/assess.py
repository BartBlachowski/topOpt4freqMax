import json
import numpy as np
from scipy.optimize import minimize_scalar
from pathlib import Path
# Run from the repository root. No solver calls or benchmark mutations.
OUTPUT = Path('analysis/performance_stage_time_assessment')
OUTPUT.mkdir(exist_ok=True)
R=json.loads(Path('examples/Performance/conference_benchmark/nine_mesh_comparison_pedersen_b21483b/benchmark_results.json').read_text())['runs']
def fit(x,y,model):
 z=np.log(x/3200.)
 if model=='power_log':
  p,a=np.polyfit(z,np.log(y),1);return lambda u: np.exp(a)*(u/3200.)**p, float(p)
 if model=='fixed_log':
  a=np.mean(np.log(y)-1.5*z);return lambda u:np.exp(a)*(u/3200.)**1.5,1.5
 if model in ['power_linear','fixed_linear']:
  def solve(p):
   b=(x/3200.)**p;c=b@y/(b@b);return c,np.sum((y-c*b)**2)
  p=minimize_scalar(lambda p:solve(p)[1],bounds=(0,6),method='bounded',options={'xatol':1e-12}).x if model=='power_linear' else 1.5
  c,_=solve(p);return lambda u:c*(u/3200.)**p,float(p)
 if model=='log_quadratic':
  b=np.polyfit(z,np.log(y),2);return lambda u:np.exp(np.polyval(b,np.log(u/3200.))), b.tolist()
 if model=='power_step':
  A=np.column_stack([np.ones(len(x)),z,x>=64800]);b=np.linalg.lstsq(A,np.log(y),rcond=None)[0]
  return lambda u:np.exp(b[0]+b[1]*np.log(u/3200.)+b[2]*(u>=64800)),b.tolist()
def metrics(pred,y):
 return {'RMSE_s':float(np.sqrt(np.mean((pred-y)**2))),'MAPE_pct':float(100*np.mean(abs(pred/y-1))),'log_RMSE':float(np.sqrt(np.mean(np.log(pred/y)**2))),'max_abs_relative_pct':float(100*max(abs(pred/y-1)))}
out={}
for key in ['proposed','yuksel','olhoff']:
 rs=[r for r in R if r['method_key']==key];x=np.array([np.prod(r['mesh']) for r in rs],float);y=np.array([r['times']['time1']+r['times']['time2'] for r in rs]);res={}
 for model in ['power_log','fixed_log','power_linear','fixed_linear','log_quadratic','power_step']:
  fn,params=fit(x,y,model);pred=fn(x); loo=[]
  for i in range(len(x)):
   mask=np.arange(len(x))!=i; f,_=fit(x[mask],y[mask],model);loo.append(f(x[i]))
  f,_=fit(x[:7],y[:7],model)
  res[model]={'params':params,'in_sample':metrics(pred,y),'LOOCV':metrics(np.array(loo),y),'last2_forecast':metrics(f(x[7:]),y[7:]),'residual_pct':(100*(pred/y-1)).tolist()}
  print(key,model,'p',params,'MAPE fit/CV',round(res[model]['in_sample']['MAPE_pct'],1),round(res[model]['LOOCV']['MAPE_pct'],1),'CV logRMSE',round(res[model]['LOOCV']['log_RMSE'],3),'forecast',round(res[model]['last2_forecast']['MAPE_pct'],1))
 print('local slopes',np.round(np.diff(np.log(y))/np.diff(np.log(x)),2))
 print('power exponents all/pre7/last5',*[round(fit(x[s],y[s],'power_log')[1],3) for s in [slice(None),slice(7),slice(4,None)]])
 qs={}
 if key=='proposed':qs={'SIMP_it':np.array([r['times']['time2']/r['counts']['count2'] for r in rs])}
 if key=='yuksel':qs={f'stage{i}_it':np.array([r['times'][f'time{i}']/r['counts'][f'count{i}'] for r in rs]) for i in [1,2]}
 if key=='olhoff':qs={'outer_exclusive':np.array([r['times']['time1']/r['counts']['count1'] for r in rs]),'inner_it':np.array([r['times']['time2']/r['counts']['count2'] for r in rs]),'stage_per_outer':np.array([sum(r['times'][t] for t in ['time1','time2'])/r['counts']['count1'] for r in rs])}
 per={}
 for name,q in qs.items():
  fn,p=fit(x,q,'power_log'); fn7,p7=fit(x[:7],q[:7],'power_log'); fstep,params=fit(x,q,'power_step')
  per[name]={'p':p,'p_pre7':p7,'step_factor':float(np.exp(params[2])),'raw_640_to_720_ratio':float(q[7]/q[6]),'values':q.tolist(),'pre7_forecast_ratio':(q[7:]/fn7(x[7:])).tolist()}
  print('per',name,per[name])
 out[key]={'models':res,'per_iteration':per,'Ne':x.tolist(),'stage_time':y.tolist()}
 print('fixed small residual',res['fixed_linear']['residual_pct'][0],'free large residual',res['power_log']['residual_pct'][-1])
(OUTPUT / 'assessment.json').write_text(json.dumps(out,indent=2))
print('CONDITIONAL COMPONENT MODELS: observed iteration counts are inputs')
for key in ['proposed','yuksel','olhoff']:
 rs=[r for r in R if r['method_key']==key];x=np.array([np.prod(r['mesh']) for r in rs],float);y=np.array([r['times']['time1']+r['times']['time2'] for r in rs]); comps=[]
 for j in [1,2]:
  counts=np.array([r['counts'][f'count{j}'] for r in rs],float)
  times=np.array([r['times'][f'time{j}'] for r in rs],float)
  comps.append((counts,times/counts))
 for model in ['power_log','power_step']:
  preds=[]
  for i in range(len(x)):
   mask=np.arange(len(x))!=i;p=0
   for counts,q in comps:
    f,_=fit(x[mask],q[mask],model);p+=counts[i]*f(x[i])
   preds.append(p)
  result=metrics(np.array(preds),y)
  out[key].setdefault('conditional_components',{})[model]=result
  print(key,model,result)

(OUTPUT / 'assessment.json').write_text(json.dumps(out,indent=2))
