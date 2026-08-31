#!/usr/bin/env python3
"""
WP2/WP3/WP4/WP5 - systematic modal survey of every stored Olhoff density
trajectory.  Computes eigenpairs and per-mode localisation diagnostics, sweeping
the diagnostic density partition tau over a range rather than freezing one.
Adaptive escalation: if no structural mode is found in the first batch, more
modes are requested.  READ-ONLY.  No optimizer.
"""
import sys, os, csv, time, h5py, numpy as np
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_engine import modes, mesh_data, exact_count_binary
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')

TAUS=np.array([0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50])   # diagnostic sweep, nothing frozen
KBASE=12; KESC=[24,48]
CONFIGS=[('E1','linear'),('E2','eq4'),('E3','eq4'),('E2','eq4a'),('E3','eq4a')]

# meshes: (name, nelx, nely, stride, configs)  -- exhaustive where affordable
PLAN=[('160x20',160,20,1,CONFIGS),
      ('240x30',240,30,1,[('E1','linear'),('E2','eq4a'),('E3','eq4a')]),
      ('240x30',240,30,4,[('E2','eq4'),('E3','eq4')]),
      ('320x40',320,40,1,[('E2','eq4a'),('E3','eq4a')]),
      ('320x40',320,40,4,[('E1','linear'),('E2','eq4')]),
      ('400x50',400,50,4,[('E1','linear'),('E2','eq4'),('E2','eq4a'),('E3','eq4a')]),
      ('480x60',480,60,1,[('E2','eq4a'),('E3','eq4a')]),
      ('480x60',480,60,4,[('E1','linear'),('E2','eq4')]),
      ('560x70',560,70,2,[('E2','eq4a'),('E3','eq4a')]),
      ('560x70',560,70,8,[('E1','linear'),('E2','eq4')]),
      ('640x80',640,80,8,[('E1','linear'),('E2','eq4'),('E2','eq4a'),('E3','eq4a')]),
      ('720x90',720,90,10,[('E1','linear'),('E2','eq4'),('E2','eq4a'),('E3','eq4a')])]

def topology_gate(x,nelx,nely,volfrac=0.5,L=8.0,H=1.0,A_sig=0.01,vtol=1e-3):
    """Frozen hard gate, re-implemented (identical to +ie2a/topology_metrics.m)."""
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

def diagnostics(r):
    """Per-mode localisation measures at every tau in the sweep, plus two
    threshold-free measures."""
    nm=r['omega'].size
    keLow=np.empty((len(TAUS),nm)); seLow=np.empty((len(TAUS),nm))
    for i,t in enumerate(TAUS):
        low=r['zeff']<=t
        keLow[i]=r['ke_n'][low,:].sum(axis=0); seLow[i]=r['se_n'][low,:].sum(axis=0)
    dwp=r['ke_n'].T@r['zeff']
    ipr=(r['ke_n']**2).sum(axis=0)
    return keLow,seLow,dwp,ipr

def first_structural(keLow_at_tau, cut):
    w=np.flatnonzero(keLow_at_tau<cut)
    return int(w[0])+1 if w.size else None

def run():
    store={}; summary=[]; t0=time.time()
    done=set()
    for mesh,nx,ny,stride,cfgs in PLAN:
        with h5py.File(os.path.join(REPO,f'examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat'),'r') as f:
            X=f['res/rho_snapshots'][()]
        n=X.shape[0]-1
        ks=list(range(1,n+1,stride))
        # hard gate + field descriptors, computed once per (mesh,state)
        for model,law in cfgs:
            key=f'{mesh}|{model}|{law}'
            om=[]; kl=[]; sl=[]; dw=[]; ip=[]; kk=[]; esc=[]
            for k in ks:
                z=np.clip(np.float64(X[k]),0,1)
                nm=KBASE; r=modes(z,nx,ny,model,law,k=nm)
                keL,seL,dwp,ipr=diagnostics(r)
                # adaptive escalation: no mode with tau=0.1 void-KE below 0.5
                it=int(np.argmin(np.abs(TAUS-0.10)))
                nesc=0
                while first_structural(keL[it],0.5) is None and nesc<len(KESC):
                    nm=KESC[nesc]; nesc+=1
                    r=modes(z,nx,ny,model,law,k=nm); keL,seL,dwp,ipr=diagnostics(r)
                om.append(r['omega']); kl.append(keL); sl.append(seL); dw.append(dwp); ip.append(ipr)
                kk.append(nm); esc.append(nesc)
            L=max(a.size for a in om)
            pad=lambda a,shape: np.pad(a.astype(np.float32),
                    [(0,s-d) for s,d in zip(shape,a.shape)],constant_values=np.nan)
            store[key+'|omega']=np.stack([pad(a,(L,)) for a in om])
            store[key+'|keLow']=np.stack([pad(a,(len(TAUS),L)) for a in kl])
            store[key+'|seLow']=np.stack([pad(a,(len(TAUS),L)) for a in sl])
            store[key+'|dwp']  =np.stack([pad(a,(L,)) for a in dw])
            store[key+'|ipr']  =np.stack([pad(a,(L,)) for a in ip])
            store[key+'|k']    =np.array(ks)
            store[key+'|nmodes']=np.array(kk)
            print(f'  {mesh:8s} {model}/{law:6s} stride={stride} states={len(ks):5d} '
                  f'maxmodes={max(kk)} escalations={sum(1 for e in esc if e)} '
                  f'[{time.time()-t0:.0f}s]',flush=True)
        # field descriptors + hard gate once per mesh over the union of sampled states
        allks=sorted({k for _,_,_,s,c in PLAN if _==mesh for k in range(1,n+1,s)}) if False else sorted(set(ks))
        gkey=f'{mesh}|GATE'
        if gkey not in store:
            hg=[];vp=[];tp=[];nlow=[];gray=[];vol=[]
            for k in sorted(set(range(1,n+1,1))) if mesh=='160x20' else ks:
                z=np.clip(np.float64(X[k]),0,1)
                a,b,c=topology_gate(z,nx,ny)
                hg.append(a);vp.append(b);tp.append(c)
                nlow.append(int((z<=0.1).sum())); gray.append(float(np.mean(4*z*(1-z)))); vol.append(float(z.mean()))
            store[gkey+'|k']=np.array(sorted(set(range(1,n+1,1))) if mesh=='160x20' else ks)
            store[gkey+'|hard']=np.array(hg); store[gkey+'|vol_pass']=np.array(vp)
            store[gkey+'|topo_pass']=np.array(tp); store[gkey+'|n_low']=np.array(nlow)
            store[gkey+'|grayness']=np.array(gray); store[gkey+'|volume']=np.array(vol)
            print(f'  {mesh:8s} gate+descriptors  states={len(hg)} [{time.time()-t0:.0f}s]',flush=True)
    store['TAUS']=TAUS
    np.savez_compressed(os.path.join(OUT,'scripts','survey.npz'),**store)
    print(f'total {time.time()-t0:.0f}s ; arrays {len(store)}')

if __name__=='__main__':
    run()
