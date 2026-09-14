from geometry import *
def main():
 ids=json.loads((OUT/'evaluations/identity.json').read_text());geo=json.loads((OUT/'evaluations/geometry.json').read_text());out={}
 fig,axs=plt.subplots(2,1,figsize=(11,6),layout='constrained')
 for ax,n in zip(axs,[480,800]):
  tr=np.loadtxt(OUT/'evaluations'/f'trajectory_{n}.csv',delimiter=',',skiprows=1);xx=tr[:,0]
  for j,l in [(1,'Mnd'),(2,'gray'),(4,'broad core')]:ax.plot(xx,tr[:,j]*(1 if j==1 else 100),label=l)
  for s in geo[str(n)]['stages']:ax.axvline(s['start'],c='k',ls=':',lw=.6);ax.text(s['start'],92,f"stage {s['stage']}",fontsize=8)
  ax.fill_between(xx,0,100,where=tr[:,-1]>0,color='red',alpha=.13,label='next-mode warning');ax.set_title(str(n));ax.set_ylabel('% domain');ax.set_xlabel('outer iteration');ax.legend(ncol=4)
 fig.savefig(OUTFIG/'F15_warning_stage_grayness.png',dpi=170);plt.close(fig)
 for case in ids['cases']:
  n=case['nelx'];ny=case['nely'];k=case['iteration'];m=geo[str(n)];dep=np.load(OUT/'evaluations'/f'geometry_{n}.npz')['depth'].T.ravel();finalcore=dep>.06
  with h5py.File(ROOT/case['trajectory']) as f:
   R=f['RHO'][()][:k];final=R[-1]
   gray=(R>.1)&(R<.9);mid=(R>=.4)&(R<=.6)
   rows=[]
   for i in range(k):
    rows.append([i+1,float(gray[i,finalcore].mean()) if finalcore.any() else 0,float(mid[i,finalcore].mean()) if finalcore.any() else 0,float(np.mean(abs(R[i]-final))),float(np.mean(abs(R[i,finalcore]-final[finalcore]))) if finalcore.any() else 0,float(np.mean((R[i]>=.5)!=(final>=.5)))])
   rows=np.asarray(rows);np.savetxt(OUT/'evaluations'/f'final_patch_history_{n}.csv',rows,delimiter=',',header='iteration,final_core_gray_fraction,final_core_mid_fraction,domain_L1_to_final,final_core_L1_to_final,threshold_flip_to_final',comments='')
   w=m['warnings']['last'];points=sorted(set([1,w]+[a['end'] for a in m['stages']]))
   out[str(n)]={'warning_end_state_vs_final':dict(zip(['iteration','final_core_gray_fraction','final_core_mid_fraction','domain_L1_to_final','final_core_L1_to_final','threshold_flip_to_final'],rows[w-1].tolist())),'stays_within_Mnd_01pp_of_final_after':int(np.where(np.abs(np.loadtxt(OUT/'evaluations'/f'trajectory_{n}.csv',delimiter=',',skiprows=1)[:,1]-m['Mnd_percent'])>.1)[0][-1]+2)}
   if n!=400:
    fig,aa=plt.subplots(len(points),1,figsize=(11,7),layout='constrained')
    for ax,it in zip(aa,points):ax.imshow(R[it-1].reshape(n,ny).T,origin='lower',extent=[0,8,0,1],cmap='gray_r',vmin=0,vmax=1);ax.set_title(f'{n}: post-update rho at iteration {it}',fontsize=9,loc='left')
    fig.savefig(OUTFIG/f'F20_saved_snapshots_{n}.png',dpi=150);plt.close(fig)
 (OUT/'evaluations/timeline.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
