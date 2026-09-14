#!/usr/bin/env python3
"""Exploratory (not a preregistered verdict input): fraction of 0.1<rho<=0.3 elements with no
rho>0.5 element within one element (3x3 neighbourhood), per iteration, C480 vs M1, plus spike flags."""
import numpy as np, h5py, csv
from scipy import ndimage as ndi
from sd_common import *
out = {}
for nm, path, csvn in [('C480', C480, 'trajectory_C480.csv'), ('M1', EVAL / 'm1_run' / 'M1_480x60_trajectory.mat', 'trajectory_M1.csv')]:
    with h5py.File(path, 'r') as f:
        R = np.asarray(f['RHO'][()])
    det = []; nlo = []
    for r in R:
        im = r.reshape(NELX, NELY).T
        lo = (im > 0.1) & (im <= 0.3)
        near = ndi.binary_dilation(im > 0.5, structure=np.ones((3, 3)))
        det.append(float((lo & ~near).sum() / max(lo.sum(), 1))); nlo.append(int(lo.sum()))
    rows = list(csv.DictReader(open(EVAL / csvn)))
    spike = np.array([rr['spike'] == 'True' for rr in rows])
    det = np.array(det)
    # spike at iteration k is measured at the START state of k = RHO[k-2]
    ds = det[np.flatnonzero(spike) - 1] if spike.any() else np.array([])
    out[nm] = dict(detached_fraction=det, n_lo=nlo, detached_at_spike_start=ds.tolist(),
                   detached_median_all=float(np.median(det)), detached_final=float(det[-1]))
    print(nm, 'detached frac median', round(float(np.median(det)), 4), 'max', round(float(det.max()), 4),
          'final', round(float(det[-1]), 4), 'at spike starts', np.round(ds, 3).tolist()[:11])
jdump(out, EVAL / 'island_detachment_exploratory.json')
