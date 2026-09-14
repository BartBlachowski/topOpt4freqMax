from geometry import *
import csv

def stats(v):
 a=np.abs(np.asarray(v));return {'n':len(a),'max':float(a.max()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'RMS':float(np.sqrt(np.mean(a*a))) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'p95':float(np.quantile(a,.95)) if len(a) else None,'p99':float(np.quantile(a,.99)) if len(a) else None}
def kkt(v,r,lam,fitmask,scale):
 # objective -lambda/lambda_ref, volume <=0, nonnegative multiplier
 Vtot=.5*len(r);mu=max(0,float(np.mean(v[fitmask]/lam))*Vtot);red=-v/lam+mu/Vtot
 low=r<=.0010001;high=r>=.9999999;inter=~(low|high)
 violation=red.copy();violation[low]=np.minimum(red[low],0);violation[high]=np.maximum(red[high],0)
 return red,violation,{'mu_volume':mu,'volume_constraint':float((r.sum()-Vtot)/Vtot),'volume_complementarity':float(abs(mu*(r.sum()-Vtot)/Vtot)),'scale_raw_objective_RMS_interior':scale,'global_projected_residual_normalized':stats(violation/scale),'lower_count':int(low.sum()),'upper_count':int(high.sum()),'interior_count':int(inter.sum()),'lower_sign_violation':stats(violation[low]/scale),'upper_sign_violation':stats(violation[high]/scale)}
def main():
 geo=json.loads((OUT/'evaluations/geometry.json').read_text());samples=json.loads((OUT/'evaluations/FD_SAMPLE_PREREGISTERED.json').read_text());results={};maps={key:{} for key in ['raw','gK','gM','cancel','filtered','redraw','redfiltered']};scat={}
 for nx in [400,480,800]:
  d=sio.loadmat(OUT/'evaluations'/f'spectral_{nx}.mat',squeeze_me=True);r=d['rho'];ne=len(r);ny=nx//8;lam=d['lam'][0];g=d['gRaw'];gf=d['gFiltered'];gk=d['gK'];gm=d['gM'];dep=np.load(OUT/'evaluations'/f'geometry_{nx}.npz')['depth'].T.ravel()
  free=(r>.0010001)&(r<.9999999);gray=(r>.1)&(r<.9);classes={'gray':gray,'mid':(r>=.4)&(r<=.6),'solid':r>.9,'void':r<.1,'broad':dep>.06}
  scale=np.sqrt(np.mean((g[free]/lam)**2));scaleF=np.sqrt(np.mean((gf[free]/lam)**2));C=np.abs(g)/(np.abs(gk)+np.abs(gm)+np.finfo(float).eps)
  redraw,vr,kr=kkt(g,r,lam,free,scale);redf,vf,kf=kkt(gf,r,lam,free,scale);rg,vg,kg=kkt(g,r,lam,gray,scale);fg,vfg,kfg=kkt(gf,r,lam,gray,scale)
  met={'omega':d['omega'].tolist(),'lambda':d['lam'].tolist(),'gap12':float((d['omega'][1]-d['omega'][0])/d['omega'][0]),'gap23':float((d['omega'][2]-d['omega'][1])/d['omega'][1]),'mass_orthogonality_Fro':float(d['massOrth']),'eig_residual':d['eigResidual'].tolist(),'spectral_resolution_lambda_gap_over_residual_lambda':float((d['lam'][1]-lam)/(lam*np.max(d['eigResidual']))),'raw_KKT':kr,'filtered_subproblem':kf,'gray_fit_raw':kg,'gray_fit_filtered':kfg,'filtered_own_scale':float(scaleF),'classes':{},'bound_tolerance_robustness':[]}
  for tol in [1e-7,1e-5,1e-4,1e-3]:
   fi=(r>.001+tol)&(r<1-tol)
   rr,_,kk=kkt(g,r,lam,fi,scale);ff,_,fk=kkt(gf,r,lam,fi,scale)
   met['bound_tolerance_robustness'].append({'tol':tol,'free_count':int(fi.sum()),'raw_mu':kk['mu_volume'],'filtered_mu':fk['mu_volume'],'raw_gray_RMS':stats(rr[gray]/scale)['RMS'],'filtered_gray_RMS':stats(ff[gray]/scale)['RMS']})
  for cls,mask in classes.items():
   if not mask.any():met['classes'][cls]={'n':0};continue
   gg=g[mask];ff=gf[mask];rmsraw=float(np.sqrt(np.mean(gg**2)));rmsf=float(np.sqrt(np.mean(ff**2)))
   met['classes'][cls]={'n':int(mask.sum()),'gK_abs':stats(gk[mask]),'gM_abs':stats(gm[mask]),'gRaw_abs':stats(gg),'gFiltered_abs':stats(ff),'cancellation':stats(C[mask]),'cancellation_fraction_lt_01':float((C[mask]<.1).mean()),'gK_over_abs_gM_quantiles':np.quantile(gk[mask]/np.maximum(abs(gm[mask]),1e-30),[.1,.5,.9]).tolist(),'raw_negative_fraction':float((gg<0).mean()),'filter_sign_flip_fraction':float((gg*ff<0).mean()),'filter_RMS_ratio':rmsf/rmsraw,'filter_std_ratio':float(np.std(ff)/np.std(gg)),'raw_reduced_normalized':stats(redraw[mask]/scale),'filtered_reduced_common_scale':stats(redf[mask]/scale),'filtered_reduced_own_scale':stats(redf[mask]/scaleF),'raw_grayfit_normalized':stats(rg[mask]/scale),'filtered_grayfit_common_scale':stats(fg[mask]/scale),'filtered_grayfit_gray_own_scale':stats(fg[mask]/(np.sqrt(np.mean((gf[gray]/lam)**2)))),'raw_grayfit_fraction_abs_lt_01':float((abs(rg[mask]/scale)<.1).mean())}
  fd=sio.loadmat(OUT/'evaluations'/f'fd_{nx}.mat',squeeze_me=True)['fd'];sampleids=fd[:,0].astype(int)-1;err=abs(fd[:,5]-fd[:,4]);rel=err/np.maximum(abs(fd[:,4]),1e-12);errscale=err/(scale*lam)
  fdextra=np.c_[fd,err,rel,errscale]
  header='element,rho,delta,scheme,raw_analytic,FE_FD,filtered_analytic,raw_subspace_FD,filtered_subspace_FD,abs_error,relative_error,error_over_raw_interior_RMS'
  np.savetxt(OUT/'evaluations'/f'FD_RESULTS_{nx}.csv',fdextra,delimiter=',',header=header,comments='')
  met['FD']={'n_evaluations':int(sum(2 for _ in fd)),'n_samples':len(set(fd[:,0])),'n_rows':len(fd),'by_delta':[],'all_rows_error_over_raw_interior_RMS':stats(errscale),'all_rows_relative_error':stats(rel),'subspace_raw_error_over_scale':stats((fd[:,7]-fd[:,4])/(scale*lam)),'subspace_filtered_error_over_scale':stats((fd[:,8]-fd[:,6])/(scale*lam)),'raw_vs_filtered_FE_error_over_scale':stats((fd[:,5]-fd[:,6])/(scale*lam))}
  for h in sorted(set(fd[:,2]),reverse=True):
   mask=fd[:,2]==h;met['FD']['by_delta'].append({'delta':h,'relative_error':stats(rel[mask]),'error_over_raw_interior_RMS':stats(errscale[mask])})
  # preregistered all rows, no best-delta selection or post-hoc element replacement
  met['FD']['validated_raw']=bool(np.max(errscale)<.01)
  for name,v in [('raw',g*ne/lam),('gK',gk*ne/lam),('gM',gm*ne/lam),('cancel',C),('filtered',gf*ne/lam),('redraw',redraw/scale),('redfiltered',fg/scale)]:maps[name][nx]=v.reshape(nx,ny).T
  np.savez_compressed(OUT/'evaluations'/f'kkt_{nx}.npz',rho=r,raw_reduced=redraw,filtered_reduced=redf,raw_grayfit=rg,filtered_grayfit=fg,raw_scale=scale,cancellation=C)
  scat[nx]=(r,redraw/scale,fg/scale,classes)
  results[str(nx)]=met
 for name,title,file in [('raw','Raw active spectral derivative × NE / lambda1','F05_raw_sensitivity.png'),('gK','Stiffness contribution × NE / lambda1','F06_stiffness.png'),('gM','Signed mass contribution × NE / lambda1','F07_mass.png'),('cancel','Cancellation |gK+gM|/(|gK|+|gM|)','F08_cancellation.png'),('filtered','Filtered active subspace derivative × NE / lambda1','F09_filtered_gradient.png'),('redraw','Physical-problem reduced gradient / raw interior RMS','F10_reduced_KKT.png'),('redfiltered','Filtered subproblem residual, gray-fitted dual / raw interior RMS','F18_filtered_reduced.png')]:
  vals=np.concatenate([a.ravel() for a in maps[name].values()]);lim=float(np.quantile(abs(vals),.995));panel(maps[name],title+' (color clipped at pooled 99.5%)' if name!='cancel' else title,file,'coolwarm' if name!='cancel' else 'viridis',-lim if name!='cancel' else 0,lim if name!='cancel' else 1)
 fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained',sharey=True)
 for ax,(nx,(r,red,rf,cl)) in zip(axs,scat.items()):ax.scatter(r,red,s=1,alpha=.15,rasterized=True);ax.axhline(0,c='k',lw=.5);ax.set_title(str(nx));ax.set_xlabel('rho');ax.set_yscale('symlog',linthresh=.1)
 axs[0].set_ylabel('raw KKT residual / raw interior RMS');fig.savefig(OUTFIG/'F11_KKT_vs_rho.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True)
 for ax,(nx,(r,red,rf,cl)) in zip(axs,scat.items()):
  labels=[k for k,m in cl.items() if m.any()];ax.boxplot([abs(red[cl[k]]) for k in labels],tick_labels=labels,showfliers=False);ax.set_title(str(nx));ax.set_yscale('log')
 axs[0].set_ylabel('|raw reduced gradient| / raw interior RMS');fig.savefig(OUTFIG/'F12_KKT_density_classes.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');xx=[400,480,800]
 for key,label in [('raw_reduced_normalized','physical, all-free dual'),('raw_grayfit_normalized','physical, gray-fit dual'),('filtered_grayfit_common_scale','filtered, gray-fit dual')]:ax.plot(xx,[results[str(n)]['classes']['gray'][key]['RMS'] for n in xx],'o-',label=label)
 ax.set_ylabel('gray residual RMS / raw interior RMS');ax.set_xlabel('nelx');ax.legend();fig.savefig(OUTFIG/'F16_comparative_stationarity.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
 for ax,nx in zip(axs,xx):
  fd=np.loadtxt(OUT/'evaluations'/f'FD_RESULTS_{nx}.csv',delimiter=',',skiprows=1)
  for e in np.unique(fd[:,0]):a=fd[fd[:,0]==e];ax.loglog(a[:,2],a[:,-1],'.-',alpha=.5)
  ax.set_title(str(nx));ax.set_xlabel('delta')
 axs[0].set_ylabel('FD absolute error / raw interior RMS');fig.savefig(OUTFIG/'F19_FD_convergence.png',dpi=170);plt.close(fig)
 (OUT/'evaluations/stationarity.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
 print({n:{'gray_raw_rms':v['classes']['gray']['raw_reduced_normalized']['RMS'],'gray_rawfit_rms':v['classes']['gray']['raw_grayfit_normalized']['RMS'],'gray_filteredfit_rms':v['classes']['gray']['filtered_grayfit_common_scale']['RMS'],'Cmedian':v['classes']['gray']['cancellation']['median'],'FDpass':v['FD']['validated_raw']} for n,v in results.items()})
if __name__=='__main__':main()
