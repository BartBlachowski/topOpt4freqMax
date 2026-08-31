#!/usr/bin/env python3
"""Read-only independent replay of the frozen Olhoff topology evidence."""
import json, pathlib, sys
import h5py
import numpy as np
from scipy.ndimage import label

ROOT=pathlib.Path(__file__).resolve().parents[2]
MESHES=[(160,20),(240,30),(320,40),(400,50),(480,60),(560,70),(640,80),(720,90),(800,100)]
STRUCT4=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)

def projection(rho):
    order=np.lexsort((np.arange(rho.size),-rho)); b=np.zeros(rho.size,bool);b[order[:round(.5*rho.size)]]=1;return b

def metrics(rho,nx,ny):
    b=projection(rho).reshape((ny,nx),order='F'); lab,n=label(b,STRUCT4); mid=ny//2
    left={lab[y,0] for y in (mid-1,mid) if lab[y,0]>0};right={lab[y,-1] for y in (mid-1,mid) if lab[y,-1]>0};common=left&right
    sizes=np.bincount(lab.ravel()); sizes[0]=0
    detached=np.delete(sizes,list(common)+[0]) if common else sizes[1:]
    detached=detached[detached>0]; threshold=.01/(8/(nx*ny))
    max_det=int(detached.max()) if detached.size else 0; total=int(detached.sum())
    return len(common)==1 and max_det<threshold,max_det,total

def main(out):
    result={'classification':'IMPLEMENTATION_VALIDATION_ONLY_NOT_NEW_SCIENTIFIC_RESULTS','meshes':[]}
    for nx,ny in MESHES:
        path=ROOT/'examples/Performance/final_campaign/raw/olhoff'/f's1_{nx}x{ny}.mat'
        if not path.exists() or path.stat().st_size==0:
            result['meshes'].append({'mesh':f'{nx}x{ny}','status':'RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT'});continue
        records=[]
        # Replay every state at the F8 anchor mesh; endpoint-only checks on
        # other meshes avoid manufacturing a second expensive audit campaign.
        with h5py.File(path,'r') as f:
            snapshots=f['res/rho_snapshots'];
            if (nx,ny)==(640,80):
                data=np.asarray(snapshots)  # one HDF5 read; MATLAB storage appears transposed
                for i in range(data.shape[0]): records.append(metrics(np.asarray(data[i],float),nx,ny))
            else:
                records.append(metrics(np.asarray(snapshots[-1],float),nx,ny))
        passed=np.array([r[0] for r in records]); totals=np.array([r[2] for r in records])[passed]
        row={'mesh':f'{nx}x{ny}','replay_scope':'exhaustive' if (nx,ny)==(640,80) else 'final_state_only',
             'states_replayed':len(records),'passing_states':int(passed.sum()),'final_pass':bool(passed[-1])}
        if (nx,ny)==(640,80):
            row.update({'aggregate_detached_median_elements':int(np.median(totals)),
                        'aggregate_detached_p95_elements':int(np.percentile(totals,95)),
                        'aggregate_detached_max_elements':int(totals.max()),
                        'aggregate_detached_max_solid_percent':round(100*totals.max()/(.5*nx*ny),3)})
            assert (row['passing_states'],row['aggregate_detached_median_elements'],row['aggregate_detached_p95_elements'],row['aggregate_detached_max_elements'])==(1014,64,147,674)
        result['meshes'].append(row)
    pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main(sys.argv[1])
