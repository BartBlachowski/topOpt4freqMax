#!/usr/bin/env python3
"""Part 18 mechanism detail: occupancy of the eq.(4b) polynomial band rho_min < rho <= 0.1 along the
C480 and M1 trajectories, and the density distribution of kinetic energy of localized modes."""
import numpy as np, h5py
from sd_same_state_compare import load, state_rho
from sd_common import *
out = {}
bands = [(0.0010001, 0.01), (0.01, 0.05), (0.05, 0.1000001), (0.1000001, 0.3), (0.3, 1.01)]
import csv
for nm, path in [('C480', C480), ('M1', EVAL / 'm1_run' / 'M1_480x60_trajectory.mat')]:
    with h5py.File(path, 'r') as f:
        R = np.asarray(f['RHO'][()])
    occ = np.array([[np.mean((r > a) & (r <= b)) for a, b in bands[:4]] + [np.mean(r <= 0.0010001)] for r in R])
    out[nm + '_band_occupancy'] = dict(columns=['(rhomin,0.01]', '(0.01,0.05]', '(0.05,0.1]', '(0.1,0.3]', 'at_rhomin'], rows=occ)
    print(nm, 'band (0.1,0.3]: iter 10/20/40/60/final', [round(float(occ[k-1,3]),4) for k in [10,20,40,60,len(occ)]])
    print(nm, 'max frac in (rhomin,0.1]:', float(occ[:, :3].sum(1).max()), 'at iter', int(occ[:, :3].sum(1).argmax() + 1),
          'final', np.round(occ[-1], 4))
S1 = load('M1_k064', 'S1'); r = state_rho('M1_k064')
ek = np.asarray(S1.Ekin, dtype=float)
dist = {}
for j in range(3):
    e = ek[:, j]; t = e.sum()
    dist[f'mode{j+1}'] = {f'{a:g}-{b:g}': float(e[(r > a) & (r <= b)].sum() / t) for a, b in bands} | {'at_rhomin': float(e[r <= 0.0010001].sum() / t)}
out['M1_k064_S1_kinetic_energy_by_band'] = dist
print(dist)
occ = out['M1_band_occupancy']['rows']
rows = list(csv.DictReader(open(EVAL / 'trajectory_M1.csv')))
sp = [i for i, rr in enumerate(rows) if rr['spike'] == 'True']
print('M1 (0.1,0.3] occupancy at spike iterations (start state):', [(int(k + 1), round(float(occ[k - 1, 3]), 4)) for k in sp])
out['M1_spike_iterations'] = [int(k + 1) for k in sp]
jdump(out, EVAL / 'band_occupancy.json')
