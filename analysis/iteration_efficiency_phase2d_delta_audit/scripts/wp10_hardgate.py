#!/usr/bin/env python3
"""WP10 - independent re-implementation of the frozen hard gate (exact-count
projection + four-neighbour connectivity + A_sig) and reproduction of the
1600/1600 invariance claim.  READ-ONLY."""
import sys, os, csv, h5py, numpy as np
from collections import deque
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')

def topology_metrics(x, nelx, nely, volfrac=0.5, L=8.0, H=1.0, A_sig=0.01, vtol=1e-3):
    x=np.asarray(x,float).ravel(); n=x.size
    nS=int(round(volfrac*n))
    order=np.lexsort((np.arange(n), -x))          # x descending, ties by increasing index
    xb=np.zeros(n); xb[order[:nS]]=1.0
    solid=xb.reshape(nelx,nely).T.astype(bool)    # MATLAB reshape(x,nely,nelx) column-major
    nr,nc=solid.shape
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
    sizes=np.array(sizes)
    mid=nely//2                                    # MATLAB mid = nely/2 ; rows [mid, mid+1] 1-based
    rowsIdx=np.unique([mid-1, mid])                # 0-based
    left=set(labels[rowsIdx,0])-{0}; right=set(labels[rowsIdx,-1])-{0}
    spanning=left & right
    req_conn = len(spanning)==1
    reqLabel = list(spanning)[0] if req_conn else 0
    det=np.array([s for i,s in enumerate(sizes,1) if i!=reqLabel])
    eA=L*H/(nelx*nely)
    strict = det.size==0 or bool(np.all(det*eA < A_sig))
    volpass = abs(x.mean()-volfrac)/volfrac <= vtol
    return dict(volume_pass=volpass, topology_pass=bool(req_conn and strict),
                hard_gate_pass=bool(volpass and req_conn and strict))

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    X=f['res/rho_snapshots'][()]
P=list(csv.DictReader(open(os.path.join(REPO,'analysis/iteration_efficiency_phase2d_evaluator_amendment/AMENDED_OLHOFF_TRAJECTORY_EVALUATION.csv'))))
hgP=np.array([int(float(r['hard_gate'])) for r in P],bool)
rows=[]; mine=np.zeros(1600,bool)
for k in range(1,1601):
    t=topology_metrics(np.float64(X[k]),160,20)
    mine[k-1]=t['hard_gate_pass']
    rows.append(dict(k=k,volume_pass=int(t['volume_pass']),topology_pass=int(t['topology_pass']),
                     hard_gate_pass=int(t['hard_gate_pass']),phase2d_hard_gate=int(hgP[k-1]),
                     agree=int(t['hard_gate_pass']==hgP[k-1])))
with open(os.path.join(OUT,'HARD_GATE_RECHECK.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'independent hard gate vs Phase-2D column: {int((mine==hgP).sum())}/1600 identical')
print(f'  hard_gate_pass count: mine {int(mine.sum())}, Phase-2D {int(hgP.sum())}')
print('  the amended evaluator cannot change any of these values: topology_metrics()')
print('  consumes only the density field (no evaluator argument, no global, no call into')
print('  study_evaluate_design); the dependency is absent by construction, not by measurement.')
