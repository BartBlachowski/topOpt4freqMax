#!/usr/bin/env python3
"""Independent SciPy/Python replay of representative Phase-2I decisions."""
from __future__ import annotations
import csv, json, sys
from collections import deque
from pathlib import Path
import h5py, numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'analysis/iteration_efficiency_phase2f_evaluator_redesign/scripts'))
from modal_engine import exact_count_binary, modes  # noqa:E402

def read_csv(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))

def topology(x,nx=96,ny=12):
    x=np.asarray(x,float).ravel();xb=exact_count_binary(x,.5);solid=xb.reshape(nx,ny).T.astype(bool)
    labels=np.zeros_like(solid,dtype=np.int32);sizes=[];cid=0
    for c in range(nx):
      for r in range(ny):
       if solid[r,c] or labels[r,c]:
        if not solid[r,c] or labels[r,c]:continue
       else:continue
       cid+=1;todo=deque([(r,c)]);labels[r,c]=cid;count=0
       while todo:
        rr,cc=todo.popleft();count+=1
        for rn,cn in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
         if 0<=rn<ny and 0<=cn<nx and solid[rn,cn] and labels[rn,cn]==0:
          labels[rn,cn]=cid;todo.append((rn,cn))
       sizes.append(count)
    sr=np.array([ny//2-1,ny//2]);left=set(labels[sr,0])-{0};right=set(labels[sr,-1])-{0};span=left&right
    required=next(iter(span)) if len(span)==1 else 0;det=[s for i,s in enumerate(sizes,1) if i!=required]
    vp=abs(x.mean()-.5)/.5<=.001;tp=len(span)==1 and all(s*8/(nx*ny)<.01 for s in det)
    return {'binary':xb,'volume_pass':vp,'topology_pass':tp,'hard_gate_pass':vp and tp}

def selected(x,nx,ny,model):
    law='linear' if model=='E1' else 'eq4a'
    # 24 covers the maximum observed Phase-2G ordinal 18.
    r=modes(x,nx,ny,model,law,k=24)
    low=r['zeff']<=.1;vke=r['ke_n'][low].sum(axis=0);vse=r['se_n'][low].sum(axis=0)
    dwp=r['ke_n'].T@r['zeff'];ipr=(r['ke_n']**2).sum(axis=0)
    valid=(vke<.5)&(vse<.5)&(dwp>.5);ix=np.flatnonzero(valid)
    if not ix.size:raise RuntimeError('No structural mode in independent 24-mode replay')
    j=int(ix[0]);return j+1,float(r['omega'][j]),float(vke[j]),float(vse[j]),float(dwp[j]),float(ipr[j])

def reference(Q,H,P=100,L=500,eps=.001,B=3200):
    Q=np.asarray(Q,float)[:B];H=np.asarray(H,bool)[:B];n=len(H);F=np.full((n,3),np.nan);best=None
    for b in range(n):
      if b+1>=P and H[b-P+1:b+1].all() and np.isfinite(Q[b-P+1:b+1]).all():
       floor=Q[b-P+1:b+1].min(axis=0);best=floor if best is None else np.maximum(best,floor)
      if best is not None:F[b]=best
    bref=None
    for b1 in range(P,n+1,P):
      if b1-L>=1 and np.isfinite(F[b1-1]).all() and np.isfinite(F[b1-L-1]).all():
       gain=(F[b1-1]-F[b1-L-1])/F[b1-1]
       if (gain<=eps).all():bref=b1;break
    if bref is None:raise RuntimeError('reference not established')
    return bref,F[bref-1]

def persistence(passv,P):
    out=[]
    for col in passv.T:
      run=0;enter=cert=None
      for i,v in enumerate(col,1):
       run=run+1 if v else 0
       if run==P:enter=i-P+1;cert=i;break
      out.append((enter,cert))
    return out

with h5py.File(HERE/'raw/capture_96x12_H3200.mat','r') as f:
    Xd=np.asarray(f['Xd']);Xs=np.float64(f['Xs'])
with h5py.File(HERE/'raw/reference_evaluation.mat','r') as f:
    Qd=np.asarray(f['Qd']).T;Qs=np.asarray(f['Qs']).T
    matlab_ord_d=np.asarray(f['ordD']).T.astype(int);matlab_ord_s=np.asarray(f['ordS']).T.astype(int)
    matlab_hard_d=np.asarray(f['hardD']).ravel().astype(bool);matlab_hard_s=np.asarray(f['hardS']).ravel().astype(bool)
    robD=np.asarray(f['robD']).ravel();robS=np.asarray(f['robS']).ravel()

# Full independent hard-gate replay from densities (post-update states).
py_hd=[];py_hs=[];bin_diff=[]
for k in range(1,3201):
    td=topology(Xd[k]);ts=topology(Xs[k]);py_hd.append(td['hard_gate_pass']);py_hs.append(ts['hard_gate_pass'])
    bin_diff.append(int(np.count_nonzero(td['binary']!=ts['binary'])))
py_hd=np.asarray(py_hd);py_hs=np.asarray(py_hs)

refd=reference(Qd,py_hd);refs=reference(Qs,py_hs)
levels=np.array([.98,.99,.995]);ratio_d=Qd/refd[1];ratio_s=Qs/refs[1]
rd=ratio_d.min(axis=1);rs=ratio_s.min(axis=1)
pers={}
for P in (50,100,200):
    pd=py_hd[:,None]&(rd[:,None]>=levels);ps=py_hs[:,None]&(rs[:,None]>=levels)
    pers[str(P)]={'double':persistence(pd,P),'single':persistence(ps,P),
                  'crossing_differences':[int(np.count_nonzero(pd[:,j]!=ps[:,j])) for j in range(3)]}

# Representative independent spectra: ordinary, rho=.1 heavy, max ordinal,
# near each q, endpoint critical, and all four hard-gate mismatch states.
atrisk=np.sum(Xs==np.float32(.1),axis=1);maxrisk=int(np.argmax(atrisk[1:])+1)
maxord=int(np.argmax(matlab_ord_d.max(axis=1))+1)
near=[int(np.argmin(abs(rd-q))+1) for q in levels]
states=sorted(set([1,41,45,48,99,252,maxrisk,maxord,*near,2100,2200,3200]))
rows=[];max_rel=0.0;ordinal_mismatch=0
for k in states:
 for rep,X,mo in [('double',Xd,matlab_ord_d),('single',Xs,matlab_ord_s)]:
  for j,model in enumerate(('E1','E2','E3')):
   o,w,vk,vs,dp,ipr=selected(X[k],96,12,model);mw=(Qd if rep=='double' else Qs)[k-1,j]
   rel=abs(w-mw)/abs(mw);max_rel=max(max_rel,rel);ordinal_mismatch+=o!=mo[k-1,j]
   rows.append({'k':k,'role':'representative','representation':rep,'evaluator':model,
     'python_selected_ordinal':o,'matlab_selected_ordinal':int(mo[k-1,j]),'ordinal_identical':o==mo[k-1,j],
     'python_selected_omega':w,'matlab_selected_omega':mw,'relative_omega_difference':rel,
     'voidKE':vk,'voidSE':vs,'densityParticipation':dp,'IPR':ipr})
with (HERE/'raw/INDEPENDENT_SPECTRAL_REPLAY.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

result={'schema_version':'phase2i_independent_replay_v1','representative_states':[int(x) for x in states],
 'spectral_rows':len(rows),'spectral_ordinal_mismatches':int(ordinal_mismatch),
 'maximum_python_matlab_relative_omega_difference':float(max_rel),
 'hard_gate_matlab_identity_double':bool(np.array_equal(py_hd,matlab_hard_d)),
 'hard_gate_matlab_identity_single':bool(np.array_equal(py_hs,matlab_hard_s)),
 'hard_gate_representation_mismatch_states':np.flatnonzero(py_hd!=py_hs).astype(int).add(1).tolist() if False else (np.flatnonzero(py_hd!=py_hs)+1).tolist(),
 'binary_difference_states':int(np.count_nonzero(bin_diff)),'binary_differing_elements':int(np.sum(bin_diff)),
 'reference':{'b_ref_double':int(refd[0]),'b_ref_single':int(refs[0]),'Q_ref_double':refd[1].tolist(),'Q_ref_single':refs[1].tolist()},
 'persistence':pers}
result['pass']=bool(ordinal_mismatch==0 and max_rel<1e-6 and result['hard_gate_matlab_identity_double'] and result['hard_gate_matlab_identity_single'])
(HERE/'raw/independent_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
