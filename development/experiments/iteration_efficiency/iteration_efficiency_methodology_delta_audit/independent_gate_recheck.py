"""INDEPENDENT delta-audit verification of the Phase-1C repaired topology gate.
Deliberately does NOT reuse the author's scipy.ndimage implementation: uses an
explicit union-find over 4-neighbours. Read-only; no optimizer is invoked."""
import sys, h5py, numpy as np

VF=0.5; AE0=8.0/(160*20); ASIG=4*AE0

def binarize(rho,nely,nelx):
    ne=rho.size; ns=int(round(VF*ne))
    order=np.lexsort((np.arange(ne),-rho))
    b=np.zeros(ne,bool); b[order[:ns]]=True
    return b.reshape((nely,nelx),order='F')

class UF:
    def __init__(s,n): s.p=list(range(n))
    def find(s,a):
        while s.p[a]!=a: s.p[a]=s.p[s.p[a]]; a=s.p[a]
        return a
    def union(s,a,b):
        ra,rb=s.find(a),s.find(b)
        if ra!=rb: s.p[rb]=ra

def components(b):
    nely,nelx=b.shape; uf=UF(b.size)
    idx=lambda y,x: y*nelx+x
    for y in range(nely):
        for x in range(nelx):
            if not b[y,x]: continue
            if x+1<nelx and b[y,x+1]: uf.union(idx(y,x),idx(y,x+1))
            if y+1<nely and b[y+1,x]: uf.union(idx(y,x),idx(y+1,x))
    lab=-np.ones(b.shape,int)
    for y in range(nely):
        for x in range(nelx):
            if b[y,x]: lab[y,x]=uf.find(idx(y,x))
    return lab

def gate(b,a_sig):
    nely,nelx=b.shape
    lab=components(b)
    jy=nely//2                      # matches study_evaluate_design jMid=round(nely/2)
    left =[lab[jy-1,0],lab[jy,0]]
    right=[lab[jy-1,nelx-1],lab[jy,nelx-1]]
    ls={v for v in left  if v>=0}; rs={v for v in right if v>=0}
    common=ls&rs
    ids,counts=np.unique(lab[lab>=0],return_counts=True)
    det=[(i,c) for i,c in zip(ids,counts) if i not in common]
    dmax=max([c for _,c in det],default=0)
    dtot=sum(c for _,c in det)
    return bool(common), int(dmax), int(dtot), (bool(common) and dmax<a_sig)

def longest(mask):
    b=c=0
    for v in mask:
        c=c+1 if v else 0; b=max(b,c)
    return b

def run(nelx,nely,stride=1,limit=None):
    f=h5py.File(f"examples/Performance/final_campaign/raw/olhoff/s1_{nelx}x{nely}.mat",'r')
    sn=f['res/rho_snapshots']; n=sn.shape[0]
    a_sig=int(np.ceil(ASIG/(8.0/(nelx*nely))))
    sel=range(0,n if limit is None else min(n,limit),stride)
    ok=[];sup=[];dm=[];dt=[]
    for i in sel:
        b=binarize(np.asarray(sn[i],dtype=np.float64),nely,nelx)
        c,mx,tt,g=gate(b,a_sig)
        ok.append(g);sup.append(c);dm.append(mx);dt.append(tt)
    ok=np.array(ok)
    print(f"{nelx}x{nely} n={len(ok)}(stride {stride}) a_sig={a_sig} "
          f"support={100*np.mean(sup):.2f}% repaired={100*ok.mean():.2f}% "
          f"longest={longest(ok)} final_dmax={dm[-1]} final_dtot={dt[-1]} "
          f"T1_agg_pass={100*np.mean([(s and x<5 and t<5) for s,x,t in zip(sup,dm,dt)]):.2f}% "
          f"T1_percomp_pass={100*np.mean([(s and x<5) for s,x in zip(sup,dm)]):.2f}%")
    return ok,dm,dt

if __name__=='__main__':
    a=sys.argv
    run(int(a[1]),int(a[2]),int(a[3]) if len(a)>3 else 1)

# Delta-audit independent recheck (union-find; deliberately NOT the author's scipy path).
# Read-only. Run from the repository root. Requires h5py, numpy.
#
#   python3 independent_gate_recheck.py 640 80
#     640x80 n=1067 a_sig=64 support=98.50% repaired=95.03% longest=925
#            final_dmax=4 final_dtot=20 T1_agg_pass=0.56% T1_percomp_pass=45.74%
#
# final_dmax=4 / final_dtot=20 reproduces the ORIGINAL audit's diagnostic anchor
# (largest detached ~4 elements vs aggregate ~20) that motivated C1.
# T1_agg_pass 0.56% vs T1_percomp_pass 45.74% confirms the aggregate clause, not the
# per-component clause, was the binding constraint.
