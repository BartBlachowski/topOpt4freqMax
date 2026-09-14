#!/usr/bin/env python3
"""
Supplementary hard-gate pass.  The main survey's gate block overwrote its own
result when a mesh appeared in the plan more than once, leaving coarser coverage
on 240x30 / 320x40 / 480x60 / 560x70.  This recomputes the frozen hard gate and
the field descriptors over the FULL union of states each mesh needs, so WP21 has
gate coverage at least as dense as the modal survey.  Density-only, no eigensolve.
READ-ONLY.
"""
import os, h5py, numpy as np
from collections import deque
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')

def exact_count_binary(x, volfrac=0.5):
    x=np.asarray(x,float).ravel(); n=x.size; nS=int(round(volfrac*n))
    order=np.lexsort((np.arange(n),-x)); xb=np.zeros(n); xb[order[:nS]]=1.0
    return xb

def gate(x,nelx,nely,volfrac=0.5,L=8.0,H=1.0,A_sig=0.01,vtol=1e-3):
    x=np.asarray(x,float).ravel()
    xb=exact_count_binary(x,volfrac)
    solid=xb.reshape(nelx,nely).T.astype(bool); nr,nc=solid.shape
    labels=np.zeros((nr,nc),int); sizes=[]; cid=0
    for c in range(nc):
        for r in range(nr):
            if not solid[r,c] or labels[r,c]: continue
            cid+=1; q=deque([(r,c)]); labels[r,c]=cid; cnt=0
            while q:
                rr,cc=q.popleft(); cnt+=1
                for rn,cn in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0<=rn<nr and 0<=cn<nc and solid[rn,cn] and labels[rn,cn]==0:
                        labels[rn,cn]=cid; q.append((rn,cn))
            sizes.append(cnt)
    sizes=np.array(sizes); mid=nely//2; rws=np.unique([mid-1,mid])
    left=set(labels[rws,0])-{0}; right=set(labels[rws,-1])-{0}; span=left&right
    conn=len(span)==1; req=list(span)[0] if conn else 0
    det=np.array([s for i,s in enumerate(sizes,1) if i!=req])
    eA=L*H/(nelx*nely)
    strict=det.size==0 or bool(np.all(det*eA<A_sig))
    volp=abs(x.mean()-volfrac)/volfrac<=vtol
    return bool(volp and conn and strict), bool(volp), bool(conn and strict)

# densest stride the modal survey used per mesh
STRIDE={'160x20':1,'240x30':1,'320x40':1,'400x50':4,'480x60':1,'560x70':2,'640x80':8,'720x90':10}
DIM={'160x20':(160,20),'240x30':(240,30),'320x40':(320,40),'400x50':(400,50),
     '480x60':(480,60),'560x70':(560,70),'640x80':(640,80),'720x90':(720,90)}
store={}
import time; t0=time.time()
for mesh,(nx,ny) in DIM.items():
    with h5py.File(os.path.join(REPO,f'examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat'),'r') as f:
        X=f['res/rho_snapshots'][()]
    n=X.shape[0]-1
    ks=list(range(1,n+1,STRIDE[mesh]))
    hg=[];vp=[];tp=[];nlow=[];gray=[];vol=[]
    for k in ks:
        z=np.clip(np.float64(X[k]),0,1)
        a,b,c=gate(z,nx,ny); hg.append(a);vp.append(b);tp.append(c)
        nlow.append(int((z<=0.1).sum())); gray.append(float(np.mean(4*z*(1-z)))); vol.append(float(z.mean()))
    p=f'{mesh}|GATE2'
    store[p+'|k']=np.array(ks); store[p+'|hard']=np.array(hg)
    store[p+'|vol_pass']=np.array(vp); store[p+'|topo_pass']=np.array(tp)
    store[p+'|n_low']=np.array(nlow); store[p+'|grayness']=np.array(gray); store[p+'|volume']=np.array(vol)
    print(f'  {mesh:8s} stride={STRIDE[mesh]:2d} states={len(ks):5d} hard_gate_pass={sum(hg):5d} '
          f'[{time.time()-t0:.0f}s]',flush=True)
np.savez_compressed(os.path.join(OUT,'scripts','gate_full.npz'),**store)
print('total %.0fs'%(time.time()-t0))
