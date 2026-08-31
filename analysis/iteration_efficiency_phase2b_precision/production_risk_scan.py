#!/usr/bin/env python3
"""Read-only all-state precision-risk scan of frozen Olhoff single snapshots."""
import csv, json, pathlib
import h5py
import numpy as np
from scipy.ndimage import label

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parent
MESHES=[(160,20),(240,30),(320,40),(400,50),(480,60),(560,70),(640,80),(720,90),(800,100)]
S4=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)

def project_and_gate(x,nx,ny):
    ne=x.size;ns=round(.5*ne);order=np.lexsort((np.arange(ne),-x));chosen=order[:ns]
    b=np.zeros(ne,bool);b[chosen]=1;b=b.reshape((ny,nx),order='F')
    lo=float(x[order[ns-1]]);hi=float(x[order[ns]]);gap=lo-hi
    ulp=float(max(abs(np.spacing(np.float32(lo))),abs(np.spacing(np.float32(hi)))))
    labs,n=label(b,S4);mid=ny//2
    left={labs[y,0] for y in (mid-1,mid) if labs[y,0]>0};right={labs[y,-1] for y in (mid-1,mid) if labs[y,-1]>0};common=left&right
    sizes=np.bincount(labs.ravel());sizes[0]=0;det=np.delete(sizes,list(common)+[0]) if common else sizes[1:];det=det[det>0]
    threshold=.01/(8/(nx*ny));top=(len(common)==1 and (not det.size or det.max()<threshold))
    return gap,gap==0,gap<=ulp,top,int(det.max()) if det.size else 0,int(det.sum())

def runs(a):
    out=[];cur=0
    for v in a:
        if v:cur+=1
        elif cur:out.append(cur);cur=0
    if cur:out.append(cur)
    return out

def main():
    rows=[]
    for nx,ny in MESHES:
        mesh=f'{nx}x{ny}';path=ROOT/'examples/Performance/final_campaign/raw/olhoff'/f's1_{mesh}.mat'
        if not path.exists() or path.stat().st_size==0:
            rows.append(dict(mesh=mesh,status='RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT',states=0));continue
        gaps=[];ties=[];risks=[];tops=[];dmax=[];dtot=[]
        with h5py.File(path,'r') as f:
            ds=f['res/rho_snapshots']
            for start in range(0,ds.shape[0],64):
                block=np.asarray(ds[start:min(start+64,ds.shape[0])])
                for x in block:
                    a=project_and_gate(x,nx,ny);gaps.append(a[0]);ties.append(a[1]);risks.append(a[2]);tops.append(a[3]);dmax.append(a[4]);dtot.append(a[5])
        rr=runs(tops)
        rows.append(dict(mesh=mesh,status='AVAILABLE_SINGLE_ONLY_INTERMEDIATE',states=len(gaps),minimum_cutoff_gap=min(gaps),
            median_cutoff_gap=float(np.median(gaps)),cutoff_tie_states=int(sum(ties)),rounding_interval_risk_states=int(sum(risks)),
            topology_pass_states=int(sum(tops)),topology_transition_count=int(np.count_nonzero(np.diff(tops))),
            topology_runs_near_P100=sum(95<=v<=105 for v in rr),maximum_topology_pass_run=max(rr) if rr else 0,
            maximum_detached_component_elements=max(dmax),maximum_aggregate_detached_elements=max(dtot),
            quality_threshold_distance='UNAVAILABLE_NO_FROZEN_PHASE2_REFERENCE_Q_TRAJECTORY'))
        print(mesh,len(gaps),sum(ties),sum(risks),sum(tops))
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with open(OUT/'production_scale_risk_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'outputs'/'production_scale_risk_metrics.json').parent.mkdir(exist_ok=True)
    (OUT/'outputs'/'production_scale_risk_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
