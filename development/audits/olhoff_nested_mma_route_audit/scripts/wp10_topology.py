#!/usr/bin/env python3
"""WP10 - topology audit of the MMA route against the LP reproduction.
Uses the frozen exact-count projection + four-neighbour connectivity + A_sig gate.
READ-ONLY."""
import os, csv, h5py, numpy as np
from collections import deque
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/olhoff_nested_mma_route_audit')
D='/Volumes/HP911Pro/Combobulating/Olhoff/results'

def exact_count_binary(x,volfrac=0.5):
    x=np.asarray(x,float).ravel(); n=x.size; nS=int(round(volfrac*n))
    order=np.lexsort((np.arange(n),-x)); xb=np.zeros(n); xb[order[:nS]]=1.0
    return xb

def topo(x,nelx,nely,volfrac=0.5,L=8.0,H=1.0,A_sig=0.01,vtol=1e-3):
    x=np.asarray(x,float).ravel(); xb=exact_count_binary(x,volfrac)
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
    return dict(volume=float(x.mean()),volume_pass=bool(volp),
        grayness=float(np.mean(4*x*(1-x))),
        gray_fraction_01_09=float(np.mean((x>0.1)&(x<0.9))),
        n_components=int(len(sizes)),required_connected=bool(conn),
        n_detached=int(det.size),max_detached_elements=int(det.max()) if det.size else 0,
        max_detached_area=float(det.max()*eA) if det.size else 0.0,
        aggregate_detached_elements=int(det.sum()) if det.size else 0,
        a_sig_elements=A_sig/eA,
        topology_pass=bool(conn and strict),hard_gate_pass=bool(volp and conn and strict))

def load_rho(f):
    with h5py.File(os.path.join(D,f),'r') as h:
        return np.array(h['res/rho'][()]).ravel(), np.array(h['res/omega'][()]).ravel(), \
               int(np.array(h['res/cfg/nelx'][()]).ravel()[0]), int(np.array(h['res/cfg/nely'][()]).ravel()[0]), \
               int(np.array(h['res/nOuter'][()]).ravel()[0])

cases=[('fm_mma_diag.mat','MMA route (= BASE_mma config, outer 400)'),
       ('lprmin1.2.mat','LP route, 160x20, rmin=1.2'),
       ('lprmin1.1.mat','LP route, 160x20, rmin=1.1'),
       ('lprmin2.5.mat','LP route, 160x20, rmin=2.5 (legacy-like filter)'),
       ('lp240_rmin1.3.mat','LP route, 240x30, rmin=1.3 (clean BEST)'),
       ('FIG4_definitive.mat','LP route, 240x30, Fig.4 trace')]
rows=[]; fields={}
for f,lab in cases:
    p=os.path.join(D,f)
    if not os.path.exists(p): print(f'{f} MISSING'); continue
    rho,om,nx,ny,no=load_rho(f)
    t=topo(rho,nx,ny)
    t.update(artifact=f,label=lab,mesh=f'{nx}x{ny}',nOuter=no,
             omega1=float(om[0]),omega2=float(om[1]),omega3=float(om[2]),
             rel_gap_pct=float(100*(om[1]-om[0])/om[0]))
    rows.append(t); fields[f]=(rho,nx,ny,lab,om)
ks=['artifact','label','mesh','nOuter','omega1','omega2','omega3','rel_gap_pct','volume','volume_pass',
    'grayness','gray_fraction_01_09','n_components','required_connected','n_detached',
    'max_detached_elements','max_detached_area','aggregate_detached_elements','a_sig_elements',
    'topology_pass','hard_gate_pass']
with open(os.path.join(OUT,'TOPOLOGY_AUDIT.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=ks,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
print(f"{'artifact':22s} {'mesh':8s} {'w1':>8s} {'gap%':>7s} {'gray':>7s} {'gray01_09':>10s} "
      f"{'comp':>5s} {'conn':>5s} {'ndet':>5s} {'maxdet':>7s} {'HARD':>6s}")
for r in rows:
    print(f"{r['artifact']:22s} {r['mesh']:8s} {r['omega1']:8.3f} {r['rel_gap_pct']:7.3f} "
          f"{r['grayness']:7.4f} {r['gray_fraction_01_09']:10.4f} {r['n_components']:5d} "
          f"{str(r['required_connected']):>5s} {r['n_detached']:5d} {r['max_detached_elements']:7d} "
          f"{str(r['hard_gate_pass']):>6s}")

# ---- topology images, identical convention -----------------------------------
fig,axes=plt.subplots(len(fields),1,figsize=(11,1.5*len(fields)))
for ax,(f,(rho,nx,ny,lab,om)) in zip(np.atleast_1d(axes),fields.items()):
    ax.imshow(rho.reshape(nx,ny).T,origin='lower',aspect='auto',cmap='gray_r',
              vmin=0,vmax=1,interpolation='nearest')
    ax.set_title(f'{lab}   omega1={om[0]:.3f}  gap={100*(om[1]-om[0])/om[0]:.3f}%',fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Topology comparison: MMA route vs LP reproduction (density field, identical convention)',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(os.path.join(OUT,'figures','topology_comparison.png'),dpi=130); plt.close(fig)
print('\nwrote figures/topology_comparison.png')
