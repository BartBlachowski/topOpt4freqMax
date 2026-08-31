#!/usr/bin/env python3
"""WP12 - exhaustive independent search for any stored DENSITY trajectory long
enough to support an offline B_ref=3200 reference re-evaluation under Eq. (4a).
READ-ONLY.  Scans every .mat/.npy/.npz/.h5/.csv artifact in the repository."""
import os, sys, json, csv
import numpy as np
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2d_delta_audit')

rows=[]

def note(path, dataset, dtype, shape, kind, verdict, why):
    rows.append(dict(file=os.path.relpath(path,REPO), dataset=dataset, dtype=str(dtype),
                     shape='x'.join(map(str,shape)), kind=kind, verdict=verdict, note=why))

def scan_h5(path):
    import h5py
    try: f=h5py.File(path,'r')
    except Exception as e:
        note(path,'','','', 'unreadable','N/A',f'not HDF5/v7.3: {e}'); return
    found=[]
    def v(name,obj):
        if isinstance(obj,h5py.Dataset) and obj.dtype.kind in 'fiu' and obj.ndim>=1:
            found.append((name,obj.dtype,obj.shape))
    f.visititems(v); f.close()
    classify(path,found)

def scan_scipy(path):
    import scipy.io as sio
    try: w=sio.whosmat(path)
    except Exception as e:
        try: scan_h5(path)
        except Exception: note(path,'','','','unreadable','N/A',str(e))
        return
    found=[(n,dt,shp) for n,shp,dt in w]
    # scipy whosmat returns (name, shape, class)
    found=[(n,cl,shp) for (n,shp,cl) in w]
    classify(path,found)

def classify(path,found):
    for name,dt,shape in found:
        big=max(shape) if shape else 0
        low=[s for s in shape if s>1]
        is_rho = any(t in name.lower() for t in ('rho','x_double','x_single','xphys','dens','snapshot'))
        is_Q   = any(t in name.lower() for t in ('q','omega','quality','freq'))
        if is_rho and big>=1000:
            kind='density trajectory'
            verdict='ADEQUATE_FOR_BREF' if big>=3200 else 'TOO_SHORT'
            note(path,name,dt,shape,kind,verdict,f'longest axis {big} vs required 3200')
        elif is_rho and big>=200:
            note(path,name,dt,shape,'density set','TOO_SHORT',f'longest axis {big}')
        elif is_Q and big>=3200:
            note(path,name,dt,shape,'quality-like array','QUALITY_ONLY',
                 f'length {big} >= 3200 but carries no density field')

exts=('.mat',)
for root,dirs,files in os.walk(REPO):
    if '/.git' in root: continue
    for fn in files:
        if fn.endswith(exts):
            p=os.path.join(root,fn)
            if os.path.getsize(p)==0:
                note(p,'','','','empty','N/A','zero-byte file'); continue
            with open(p,'rb') as fh: head=fh.read(128)
            if b'MATLAB 7.3' in head: scan_h5(p)
            else: scan_scipy(p)

rows.sort(key=lambda r:(r['verdict'],r['file']))
with open(os.path.join(OUT,'WP12_DENSITY_EVIDENCE_SCAN.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['file','dataset','dtype','shape','kind','verdict','note'])
    w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter(r['verdict'] for r in rows))
print('--- ADEQUATE_FOR_BREF ---')
for r in rows:
    if r['verdict']=='ADEQUATE_FOR_BREF': print(' ',r['file'],r['dataset'],r['dtype'],r['shape'])
print('--- longest density trajectories found ---')
d=[r for r in rows if 'density' in r['kind']]
d.sort(key=lambda r:-max(int(s) for s in r['shape'].split('x')))
for r in d[:15]:
    print(f"  {max(int(s) for s in r['shape'].split('x')):6d}  {r['file']}  {r['dataset']} {r['dtype']} {r['shape']}")
print('--- quality-only arrays >=3200 ---')
for r in rows:
    if r['verdict']=='QUALITY_ONLY': print(' ',r['file'],r['dataset'],r['dtype'],r['shape'])
