from identity import *
from scipy import ndimage as ndi
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUTFIG=OUT/'figures'
def fields(rho,nx,ny):
 im=rho.reshape(nx,ny).T; gray=(im>.1)&(im<.9); mid=(im>=.4)&(im<=.6); h=1/ny
 depth=ndi.distance_transform_edt(gray,sampling=(h,h)); depth[~gray]=0
 lab,n=ndi.label(gray); sizes=np.bincount(lab.ravel())[1:]; lab8,n8=ndi.label(gray,np.ones((3,3)))
 binary=im>=.5; interface=np.zeros_like(gray)
 interface[1:]|=binary[1:]!=binary[:-1];interface[:-1]|=binary[1:]!=binary[:-1]
 interface[:,1:]|=binary[:,1:]!=binary[:,:-1];interface[:,:-1]|=binary[:,1:]!=binary[:,:-1]
 di=ndi.distance_transform_edt(~interface,sampling=(h,h))
 return im,gray,mid,depth,lab,sizes,n,n8,di
def panel(arrays,title,name,cmap='viridis',vmin=None,vmax=None,label=''):
 fig,axs=plt.subplots(3,1,figsize=(12,5.6),layout='constrained')
 for ax,(nx,a) in zip(axs,arrays.items()):
  m=ax.imshow(a,origin='lower',extent=[0,8,0,1],aspect='equal',cmap=cmap,vmin=vmin,vmax=vmax)
  ax.set_title(f'{nx}×{nx//8}',loc='left',fontsize=10);ax.set_ylabel('y');ax.set_xlabel('x')
 fig.suptitle(title);fig.colorbar(m,ax=axs,label=label,shrink=.8);fig.savefig(OUTFIG/name,dpi=170);plt.close(fig)
def main():
 ident=json.loads((OUT/'evaluations/identity.json').read_text());results={}; ims={};depths={};samples={}
 for c in ident['cases']:
  nx,ny,k=c['nelx'],c['nely'],c['iteration']
  with h5py.File(ROOT/c['trajectory']) as f:
   RHO=f['RHO'][()];rho=RHO[k-1]; im,gr,mid,depth,lab,sizes,n,n8,di=fields(rho,nx,ny)
   ims[nx]=im;depths[nx]=depth/.06
   broad=gr&(depth>.06)
   comps=[{'id':int(j),'elements':int(size),'area':float(size/(nx*ny)*8),'max_depth':float(depth[lab==j].max()),'bbox_x_span':float((np.where(lab==j)[1].max()-np.where(lab==j)[1].min()+1)*8/nx),'bbox_y_span':float((np.where(lab==j)[0].max()-np.where(lab==j)[0].min()+1)/ny)} for j,size in enumerate(sizes,1)]
   comps.sort(key=lambda a:a['elements'],reverse=True)
   m={'Mnd_percent':float(400*np.mean(rho*(1-rho))),'gray_fraction':float(gr.mean()),'mid_fraction':float(mid.mean()),'gray_area':float(gr.mean()*8),'mid_area':float(mid.mean()*8),'broad_core_fraction':float(broad.mean()),'broad_core_area':float(broad.mean()*8),'gray_components_4':n,'gray_components_8':n8,'largest_gray_component_area':comps[0]['area'],'max_depth':float(depth.max()),'gray_depth_p50_p90_p95_p99':np.quantile(depth[gr],[.5,.9,.95,.99]).tolist(),'gray_interface_distance_p50_p90_p95_p99':np.quantile(di[gr],[.5,.9,.95,.99]).tolist(),'rho_quantiles':dict(zip(map(str,[0,1,5,10,25,50,75,90,95,99,100]),np.quantile(rho,np.array([0,1,5,10,25,50,75,90,95,99,100])/100).tolist())),'rho_lt_001':float((rho<.01).mean()),'rho_gt_099':float((rho>.99).mean()),'components':comps}
   ids={}; flatdepth=depth.T.ravel()
   classes={'solid':rho>.9,'void':rho<.1,'gray':(rho>.1)&(rho<.9),'mid':(rho>=.4)&(rho<=.6),'broad':flatdepth>.06}
   for cls,mask in classes.items():
    eligible=np.flatnonzero(mask);ids[cls]=[int(eligible[int(np.floor(q*(len(eligible)-1)))])+1 for q in [.2,.5,.8]] if len(eligible) else []
   samples[str(nx)]={'selection':'nearest rank 20/50/80 percent of eligible column-major element IDs, before gradients','by_class':ids,'unique_ids':sorted(set(sum(ids.values(),[])))}
   sio.savemat(OUT/'evaluations'/f'sample_{nx}.mat',{'ids':np.array(samples[str(nx)]['unique_ids']),'deltas':np.array([1e-3,3e-4,1e-4])})
   hist={key:f['hist'][key][()].squeeze()[:k] for key in ['omega','move','stage','exA','exB','exE','multJ','N','gap12','beta']}
   rows=[]
   for i in range(k):
    r=RHO[i];a=r.reshape(nx,ny).T;g=(a>.1)&(a<.9);mi=(a>=.4)&(a<=.6);de=ndi.distance_transform_edt(g,sampling=(1/ny,1/ny)) if not g.all() else np.full_like(a,np.inf)
    w=hist['omega'][i]
    rows.append([i+1,400*np.mean(r*(1-r)),g.mean(),mi.mean(),(de>.06).mean(),*w[:3],hist['move'][i],hist['stage'][i],hist['exA'][i],hist['exB'][i],hist['exE'][i],hist['gap12'][i],(w[2]-w[1])/w[1],hist['multJ'][i]])
   tr=np.asarray(rows); np.savetxt(OUT/'evaluations'/f'trajectory_{nx}.csv',tr,delimiter=',',header='iteration,Mnd_percent,gray_fraction,mid_fraction,broad_core_fraction,omega1_preupdate,omega2_preupdate,omega3_preupdate,move,stage,exA,exB,exE,gap12_preupdate,gap23_preupdate,warning',comments='')
   m['stages']=[]
   for s in [1,2,3]:
    ix=np.where(tr[:,9]==s)[0];a,b=ix[0],ix[-1]
    m['stages'].append({'stage':s,'start':int(a+1),'end':int(b+1),'Mnd_start':tr[a,1],'Mnd_end':tr[b,1],'gray_end':tr[b,2],'mid_end':tr[b,3],'broad_end':tr[b,4]})
   warn=tr[:,15]>0;m['warnings']={'count':int(warn.sum()),'iterations':tr[warn,0].astype(int).tolist(),'last':int(tr[warn,0][-1]) if warn.any() else None,'Mnd_at_last':float(tr[warn,1][-1]) if warn.any() else None,'broad_at_last':float(tr[warn,4][-1]) if warn.any() else None}
   m['first_broad_core_fraction_gt_001']=int(tr[np.where(tr[:,4]>.01)[0][0],0]) if (tr[:,4]>.01).any() else None
   m['terminal20']={'Mnd_range_pp':float(np.ptp(tr[-20:,1])),'omega1_range_percent':float(100*np.ptp(tr[-20:,5])/tr[-1,5]),'gray_range':float(np.ptp(tr[-20:,2])),'broad_range':float(np.ptp(tr[-20:,4]))}
   np.savez_compressed(OUT/'evaluations'/f'geometry_{nx}.npz',rho=rho,depth=depth,interface_distance=di,labels=lab)
   fig,axs=plt.subplots(5,1,figsize=(11,10),sharex=True,layout='constrained');xx=tr[:,0]
   axs[0].plot(xx,tr[:,5]);axs[0].set_ylabel('ω₁ (pre-update)')
   for j,label in [(1,'Mnd %'),(2,'gray %'),(3,'mid %'),(4,'broad core %')]:axs[1].plot(xx,tr[:,j]*(1 if j==1 else 100),label=label)
   axs[1].legend(ncol=4);axs[1].set_ylabel('% domain')
   axs[2].step(xx,tr[:,8],label='move');axs[2].set_ylabel('move')
   for j,label in [(10,'A'),(11,'B'),(12,'E')]:axs[3].plot(xx,tr[:,j],label=label,alpha=.7)
   axs[3].legend(ncol=3);axs[3].set_ylabel('exhaustion')
   axs[4].semilogy(xx,tr[:,13],label='gap12');axs[4].semilogy(xx,tr[:,14],label='gap23');axs[4].legend();axs[4].set_ylabel('relative ω gap');axs[4].set_xlabel('outer iteration')
   for ax in axs:
    for stage in m['stages'][1:]:ax.axvline(stage['start'],color='k',ls=':',lw=.7)
    ax.fill_between(xx,0,1,where=warn,transform=ax.get_xaxis_transform(),color='red',alpha=.1)
   fig.suptitle(f'{nx}×{ny}: saved trajectory; red = next-mode warning, dotted = stage start');fig.savefig(OUTFIG/f'F{13 if nx==480 else 14 if nx==800 else 17}_trajectory_{nx}.png',dpi=150);plt.close(fig)
   results[str(nx)]=m
 panel(ims,'Authoritative final densities, identical physical coordinates','F01_final_rho.png','gray_r',0,1,'rho')
 panel(depths,'Gray-region depth / physical filter radius R=0.06','F04_gray_depth.png',vmin=0,vmax=max(a.max() for a in depths.values()),label='depth/R')
 fig,axs=plt.subplots(1,3,figsize=(12,3),layout='constrained',sharey=True)
 for ax,(nx,a) in zip(axs,ims.items()):ax.hist(a.ravel(),bins=np.linspace(0,1,51),weights=np.ones(a.size)/a.size*100);ax.set_title(str(nx));ax.set_xlabel('rho')
 axs[0].set_ylabel('% elements per bin');fig.savefig(OUTFIG/'F02_histograms.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');xx=np.arange(3)
 for j,(key,label) in enumerate([('gray_fraction','gray'),('mid_fraction','mid'),('broad_core_fraction','broad core')]):ax.bar(xx+(j-1)*.25,[100*m[key] for m in results.values()],width=.25,label=label)
 ax.set_xticks(xx,results.keys());ax.set_ylabel('% physical domain');ax.legend();fig.savefig(OUTFIG/'F03_gray_fractions.png',dpi=170);plt.close(fig)
 (OUT/'evaluations/geometry.json').write_text(json.dumps(results,indent=2));(OUT/'evaluations/FD_SAMPLE_PREREGISTERED.json').write_text(json.dumps(samples,indent=2))
 print({nx:{k:m[k] for k in ['Mnd_percent','gray_fraction','mid_fraction','broad_core_fraction','max_depth']} for nx,m in results.items()})
if __name__=='__main__':main()
