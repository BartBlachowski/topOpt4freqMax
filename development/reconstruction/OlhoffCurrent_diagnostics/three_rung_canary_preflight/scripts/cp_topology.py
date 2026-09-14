#!/usr/bin/env python3
"""cp_topology.py -- figures 11 and 12: final density fields.

11  480x60 vs 800x100, both three-rung canaries
12  800x100 legacy (beta / four-rung) vs 800x100 three-rung canary

Canary densities come from the canary's own trajectory .mat (last column of
RHO, which cp_run proved equals res.rho).  The legacy density comes from the
September 11 campaign record, read directly -- not re-simulated.
"""
import json, sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import ROOT

HERE = Path(__file__).parents[1]
FIG = HERE / 'figures'
EV = ROOT / 'analysis/OlhoffCurrent/evidence/three_rung_canary_preflight'
CAMP = ROOT / 'examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat'


def canary_rho(mesh):
    nx, ny = (int(v) for v in mesh.split('x'))
    f = EV / f'C{mesh}_three_rung_trajectory.mat'
    if not f.exists():
        return None, None
    with h5py.File(f, 'r') as h:
        RHO = h['RHO']                      # stored (nOuter, NE) in HDF5 order
        r = np.array(RHO[-1, :]) if RHO.shape[0] < RHO.shape[1] else np.array(RHO[:, -1])
    return r.reshape((ny, nx), order='F'), (nx, ny)


def legacy_rho(mesh):
    nx, ny = (int(v) for v in mesh.split('x'))
    with h5py.File(CAMP, 'r') as f:
        g = f['records']
        for i in range(g['mesh'].size):
            m = np.array(f[g['mesh'][()].ravel()[i]]).ravel().astype(int)
            key = ''.join(chr(int(x)) for x in
                          np.array(f[g['method_key'][()].ravel()[i]]).ravel() if x)
            if list(m) == [nx, ny] and key == 'olhoff':
                x = np.array(f[g['x'][()].ravel()[i]]).ravel()
                return x.reshape((ny, nx), order='F')
    return None


def show(ax, r, title):
    ax.imshow(1 - r, cmap='gray', vmin=0, vmax=1, aspect='equal',
              interpolation='nearest')
    Mnd = 100 * np.mean(4 * r * (1 - r))
    ax.set_title(f'{title}\nM_nd = {Mnd:.2f} %   vol = {r.mean():.4f}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def main():
    made = []
    r480, _ = canary_rho('480x60')
    r800, _ = canary_rho('800x100')

    if r480 is not None and r800 is not None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 4.6))
        show(ax[0], r480, '480×60 three-rung canary')
        show(ax[1], r800, '800×100 three-rung canary')
        fig.suptitle('FIG 11 — final topology, three-rung canaries', fontsize=10)
        save(fig, 'FIG_11_topology_480_vs_800'); made.append(11)
    elif r480 is not None:
        fig, ax = plt.subplots(figsize=(10, 2.4))
        show(ax, r480, '480×60 three-rung canary')
        save(fig, 'FIG_11_topology_480_only'); made.append('11-partial')

    if r800 is not None:
        L = legacy_rho('800x100')
        fig, ax = plt.subplots(2, 1, figsize=(10, 4.6))
        show(ax[0], L, '800×100 LEGACY (beta, four-rung ladder, designChange stop)')
        show(ax[1], r800, '800×100 three-rung canary (stageExhaustion, terminal E at move 0.01)')
        fig.suptitle('FIG 12 — legacy vs three-rung final topology at 800×100', fontsize=10)
        save(fig, 'FIG_12_topology_legacy_vs_three_rung_800'); made.append(12)
        iou = float(np.sum((L > .5) & (r800 > .5)) / np.sum((L > .5) | (r800 > .5)))
        flip = float(np.mean((L > .5) != (r800 > .5)))
        d = {'IoU_threshold_0p5': iou, 'threshold_flip_fraction': flip,
             'density_L1': float(np.mean(np.abs(L - r800))),
             'Mnd_legacy_pct': float(100 * np.mean(4 * L * (1 - L))),
             'Mnd_three_rung_pct': float(100 * np.mean(4 * r800 * (1 - r800)))}
        (HERE / 'evidence/topology_800_comparison.json').write_text(json.dumps(d, indent=1) + '\n')
        print(json.dumps(d, indent=1))

    print('figures made:', made or 'none — canary trajectories not present')


if __name__ == '__main__':
    main()
