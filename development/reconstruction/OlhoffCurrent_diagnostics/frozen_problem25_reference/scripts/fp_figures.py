#!/usr/bin/env python3
"""fp_figures.py -- the twelve required figures, from this study's evaluations only."""
import json
from pathlib import Path
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parents[1]
EV = HERE / 'evaluations'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)

def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig); print('wrote', name)

with h5py.File(EV / 'compare_fields.mat', 'r') as f:
    F = {k: np.array(f[k]).ravel() for k in f.keys()}
nelx, nely = int(F['nelx'][0]), int(F['nely'][0]); move = float(F['move'][0]); bsRef = float(F['bsRef'][0])
R = lambda v: v.reshape((nely, nelx), order='F')
cmp = json.load(open(EV / 'mma_comparison.json'))

def field(ax, v, title, lim=None, cmap='RdBu_r'):
    lim = lim or np.max(np.abs(v))
    im = ax.imshow(R(v), cmap=cmap, vmin=-lim, vmax=lim, aspect='equal')
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.012, pad=0.01)

# 1-3: drho fields (reference, P19, M5000) + rho385 for orientation
for name, key, num in (('FIG_01_reference_drho', 'drho_ref', 1), ('FIG_02_P19_drho', 'drho_P19', 2), ('FIG_03_M5000_drho', 'drho_M5000', 3)):
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.4))
    im = ax[0].imshow(1 - R(F['rho385']), cmap='gray', vmin=0, vmax=1, aspect='equal'); ax[0].set_title('ρ₃₈₅ (density the subproblem is built at)', fontsize=9)
    ax[0].set_xticks([]); ax[0].set_yticks([]); plt.colorbar(im, ax=ax[0], fraction=0.012, pad=0.01)
    v = F[key]; util = np.max(np.abs(v)) / move
    field(ax[1], v, f'{key}  (max|drho|/move = {util:.3f})', lim=move)
    fig.suptitle(f'FIG {num} — {key} at the frozen 480×60 state, colour scale ±move', fontsize=10)
    save(fig, name)

# 4-5: differences
for name, key, num in (('FIG_04_reference_minus_P19', 'drho_P19', 4), ('FIG_05_reference_minus_M5000', 'drho_M5000', 5)):
    fig, ax = plt.subplots(1, 1, figsize=(11, 2.4))
    d = F['drho_ref'] - F[key]
    field(ax, d, f'drho_ref − {key}   ‖·‖₂ = {np.linalg.norm(d):.4f}, ‖·‖∞ = {np.max(np.abs(d)):.2e}', lim=2 * move)
    fig.suptitle(f'FIG {num} — difference field, colour scale ±2·move', fontsize=10)
    save(fig, name)

# 6-9: histories
it = F['h_iter']; ck = F['ck_iter']
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(it, F['h_max_util'], lw=1, label='repeated MMA, every iteration')
ax.axhline(1.0, color='k', ls='--', lw=0.8, label='move limit')
ax.axhline(float(np.max(np.abs(F['drho_ref']))) / move, color='C3', ls=':', lw=1.2, label='reference max|drho|/move')
ax.axvline(19, color='gray', lw=0.8); ax.text(19, 0.05, ' production stops (19)', fontsize=8)
ax.set_xlabel('inner sub-iteration'); ax.set_ylabel('max|drho| / move'); ax.legend(fontsize=8)
ax.set_title('FIG 6 — MMA max|drho| versus sub-iteration (frozen replay)', fontsize=10); save(fig, 'FIG_06_mma_maxdrho_vs_iter')

fig, ax = plt.subplots(figsize=(8, 4))
G = F['h_G']; ax.semilogx(it, G, lw=1, label='G = (bs_ref − bs_k)/(bs_ref − 1), every iteration')
ax.semilogx(ck, F['ck_G'], 'o', ms=3, label='checkpoints')
ax.axhline(0, color='k', lw=0.8); ax.axvline(19, color='gray', lw=0.8)
ax.set_xlabel('inner sub-iteration'); ax.set_ylabel('normalized objective gap to reference'); ax.legend(fontsize=8)
ax.set_title('FIG 7 — objective gap to the certified reference versus sub-iteration', fontsize=10); save(fig, 'FIG_07_objective_gap_vs_iter')

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].semilogx(ck, F['ck_dist2_rel'], 'o-', ms=3, lw=1); ax[0].set_ylabel('‖drho_k − drho_ref‖₂ / ‖drho_ref‖₂'); ax[0].set_xlabel('sub-iteration')
ax[0].axvline(19, color='gray', lw=0.8); ax[0].set_title('relative distance to reference', fontsize=9)
ax[1].semilogx(ck, F['ck_cosine'], 'o-', ms=3, lw=1, color='C2'); ax[1].set_ylabel('cosine similarity'); ax[1].set_xlabel('sub-iteration')
ax[1].axvline(19, color='gray', lw=0.8); ax[1].set_title('cosine to reference', fontsize=9)
fig.suptitle('FIG 8 — distance to the reference versus sub-iteration', fontsize=10); save(fig, 'FIG_08_distance_vs_iter')

fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(ck, F['ck_kkt_mma'], 'o-', ms=3, lw=1, label='exact-MMA-dual KKT residual (normalized RMS)')
ax.loglog(ck, F['ck_kkt_refdual'], 's-', ms=3, lw=1, label='projected residual under reference multipliers')
ax.axvline(19, color='gray', lw=0.8); ax.set_xlabel('sub-iteration'); ax.set_ylabel('residual / sRow0'); ax.legend(fontsize=8)
ax.set_title('FIG 9 — KKT residual of the true problem along the MMA replay', fontsize=10); save(fig, 'FIG_09_kkt_vs_iter')

# 10: active-bound map of the reference
fig, ax = plt.subplots(1, 1, figsize=(11, 2.6))
kind = F['ref_kind']  # -2 -move, -1 floor, 0 interior, +1 ceiling, +2 +move
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['#08306b', '#6baed6', '#ffffff', '#fdae6b', '#a63603'])
norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
im = ax.imshow(R(kind), cmap=cmap, norm=norm, aspect='equal'); ax.set_xticks([]); ax.set_yticks([])
cb = plt.colorbar(im, ax=ax, fraction=0.012, pad=0.01, ticks=[-2, -1, 0, 1, 2]); cb.ax.set_yticklabels(['−move', 'floor', 'interior', 'ceiling', '+move'], fontsize=8)
c = cmp['reference']['counts']
ax.set_title(f"reference active bounds: −move {c['minus_move']}, floor {c['floor']}, +move {c['plus_move']}, ceiling {c['ceiling']}, interior {c['interior']}", fontsize=9)
fig.suptitle('FIG 10 — active-bound map of the certified reference', fontsize=10); save(fig, 'FIG_10_reference_active_bounds')

# 11-12: scatters
for name, key, num in (('FIG_11_scatter_P19_vs_reference', 'drho_P19', 11), ('FIG_12_scatter_M5000_vs_reference', 'drho_M5000', 12)):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(F['drho_ref'] / move, F[key] / move, '.', ms=1.5, alpha=0.4)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('drho_ref / move'); ax.set_ylabel(f'{key} / move'); ax.set_aspect('equal')
    m = cmp['P19'] if 'P19' in key else cmp['M5000']
    ax.set_title(f"cosine {m['cosine']:.3f}, sign agreement {m['sign_agreement']:.3f}, G = {m['G']:.3f}", fontsize=9)
    fig.suptitle(f'FIG {num} — {key} versus the reference, element by element', fontsize=10); save(fig, name)

# 13 supplementary: q (reduced gradient) map — which elements are "free"
fig, ax = plt.subplots(1, 1, figsize=(11, 2.4))
q = F['q_ref']; lim = np.quantile(np.abs(q), 0.99)
field(ax, q, 'reduced gradient q_e = ∂L/∂drho_e at the reference (q>0: at lower bound, q<0: at upper bound, q≈0: free)', lim=lim)
fig.suptitle('FIG 13 (supplementary) — the certificate multipliers as a field', fontsize=10); save(fig, 'FIG_13_reduced_gradient_map')
