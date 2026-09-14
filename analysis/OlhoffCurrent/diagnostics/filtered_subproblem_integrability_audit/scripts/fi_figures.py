#!/usr/bin/env python3
"""fi_figures.py -- the twelve required figures, from audit evaluations only."""
import json, sys
from pathlib import Path
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parents[1]
EV = HERE / 'evaluations'
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
J = lambda n: json.load(open(EV / n))


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig); print('wrote', name)


with h5py.File(EV / 'fields.mat', 'r') as f:
    F = {k: np.array(f[k]).ravel() for k in
         ('rho', 'gPhys', 'gFilt', 'gDiff', 'kkt_resid', 'kkt_resid_nobox',
          'xsi', 'eta', 'drho', 'gray', 'core', 'depth')}
    nelx = int(np.array(f['nelx']).ravel()[0]); nely = int(np.array(f['nely']).ravel()[0])
    sRow = float(np.array(f['sRow']).ravel()[0])
R = lambda v: v.reshape((nely, nelx), order='F')

sym = J('jacobian_symmetry.json'); loops = J('closed_loops.json')
mixed = J('mixed_partials.json'); kkt = J('inner_kkt.json')
kktr = J('inner_kkt_refined.json'); dec = J('asymmetry_decomposition.json')
byc = J('residual_by_class.json'); cf = J('counterfactual_operators.json')
filt = J('filter_operator.json')

# ---- 1. physical vs filtered gradient map -------------------------------
fig, ax = plt.subplots(3, 1, figsize=(11, 6.2))
for a, v, t in zip(ax, (F['gPhys'], F['gFilt'], F['rho']),
                   ('physical  g_phys = dλ₁/dρ', 'filtered  g_filt = A(ρ)·g_phys', 'density ρ')):
    if t.startswith('density'):
        im = a.imshow(1 - R(v), cmap='gray', vmin=0, vmax=1, aspect='equal')
    else:
        lim = np.quantile(np.abs(v), 0.99)
        im = a.imshow(R(v), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
    a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=a, fraction=0.012, pad=0.01)
fig.suptitle('FIG 1 — physical and filtered gradient fields at the frozen 480×60 endpoint', fontsize=10)
save(fig, 'FIG_01_gradient_maps')

# ---- 2. filtered minus physical -----------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(11, 4.4))
lim = np.quantile(np.abs(F['gDiff']), 0.99)
im = ax[0].imshow(R(F['gDiff']), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
ax[0].set_title('g_filt − g_phys', fontsize=9); plt.colorbar(im, ax=ax[0], fraction=0.012)
ratio = np.abs(F['gFilt']) / np.maximum(np.abs(F['gPhys']), 1e-300)
im = ax[1].imshow(np.log10(np.clip(R(ratio), 1e-2, 1e3)), cmap='viridis', aspect='equal')
ax[1].set_title('log₁₀ |g_filt| / |g_phys|  — the filter amplifies void regions up to 245×', fontsize=9)
plt.colorbar(im, ax=ax[1], fraction=0.012)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle('FIG 2 — what the sensitivity filter does to the gradient', fontsize=10)
save(fig, 'FIG_02_filtered_minus_physical')

# ---- 3. Jacobian asymmetry distribution ---------------------------------
res = sym['results']
rf = np.array([r['r_filt'] for r in res]); rp = np.array([r['r_phys'] for r in res])
fig, ax = plt.subplots(figsize=(7.6, 4.2))
bins = np.logspace(-9, 1, 45)
ax.hist(rf, bins=bins, alpha=.75, label=f'filtered  (median {np.median(rf):.3f})', color='firebrick')
ax.hist(rp, bins=bins, alpha=.75, label=f'physical control  (median {np.median(rp):.2e})', color='#4C72B0')
ax.set_xscale('log'); ax.set_xlabel('relative Jacobian asymmetry  r = |uᵀJv − vᵀJu| / max(|uᵀJv|,|vᵀJu|)')
ax.set_ylabel('count'); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title('FIG 3 — Jacobian asymmetry, 10 direction pairs × 4 FD steps', fontsize=10)
save(fig, 'FIG_03_asymmetry_distribution')

# ---- 4. uJv vs vJu scatter ----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
for a, key, t, c in ((ax[0], 'filt', 'filtered field', 'firebrick'),
                     (ax[1], 'phys', 'physical field (control)', '#4C72B0')):
    x = np.array([r[f'uJv_{key}'] for r in res]); y = np.array([r[f'vJu_{key}'] for r in res])
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.15
    a.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='symmetric (uᵀJv = vᵀJu)')
    a.scatter(x, y, s=26, c=c, alpha=.8, edgecolor='k', linewidth=.3)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel('uᵀJv'); a.set_ylabel('vᵀJu'); a.set_title(t, fontsize=9)
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_aspect('equal')
fig.suptitle('FIG 4 — a conservative field must lie on the diagonal', fontsize=10)
save(fig, 'FIG_04_uJv_vs_vJu')

# ---- 5. symmetry error vs FD step ---------------------------------------
med = np.array(sym['median_by_delta'])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
pairs = sorted({r['pair'] for r in res})
for p in pairs:
    sel = [r for r in res if r['pair'] == p]
    d = [s['delta'] for s in sel]; y = [s['r_filt'] for s in sel]
    ax.plot(d, y, 'o-', lw=.9, ms=3.5, alpha=.6)
ax.plot(med[:, 0], med[:, 1], 'o-', color='firebrick', lw=2.5, ms=7, label='filtered, median')
ax.plot(med[:, 0], med[:, 2], 's-', color='#4C72B0', lw=2.5, ms=7, label='physical control, median')
ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
ax.set_xlabel('central-difference step δ'); ax.set_ylabel('relative asymmetry r')
ax.set_title('FIG 5 — asymmetry vs FD step: filtered is δ-independent, control is FD noise (∝1/δ)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')
save(fig, 'FIG_05_symmetry_vs_step')

# ---- 6 & 7. loop integrals ----------------------------------------------
lr = loops['results']
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for p in pairs:
    sel = [r for r in lr if r['pair'] == p]
    a = [s['amp'] for s in sel]
    ax[0].plot(a, [abs(s['loop_filt']) for s in sel], 'o-', lw=.9, ms=3.5, alpha=.65)
    ax[1].plot(a, [abs(s['loop_phys']) for s in sel], 'o-', lw=.9, ms=3.5, alpha=.65)
aa = np.array([1e-4, 1e-3])
ax[0].plot(aa, 3e-6 * (aa / 1e-3) ** 2, 'k--', lw=1.4, label='slope 2 (a curl)')
ax[1].plot(aa, 3e-9 * (aa / 1e-3) ** 1, 'k--', lw=1.4, label='slope 1 (FD noise)')
for a, t in ((ax[0], f"FIG 6 — filtered field, |∮g·dρ|\nmedian exponent {loops['median_exponent_filt']:.3f}"),
             (a := ax[1], f"FIG 7 — physical control, |∮g·dρ|\nmedian exponent {loops['median_exponent_phys']:.3f}")):
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('loop amplitude a = b')
    a.set_ylabel('|closed-loop integral|'); a.set_title(t, fontsize=9)
    a.legend(fontsize=8); a.grid(alpha=.3, which='both')
save(fig, 'FIG_06_07_loop_integrals')

# ---- 8. mixed-partial asymmetry -----------------------------------------
with h5py.File(EV / 'mixed_partials.mat', 'r') as f:
    relF = np.array(f['relF']).T; relP = np.array(f['relP']).T
    Hsub = np.array(f['Hsub']).T
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, M, t in ((ax[0], relF, 'filtered  |J_ij−J_ji|/max'), (ax[1], relP, 'physical control')):
    im = a.imshow(M, cmap='magma', vmin=0, vmax=2, aspect='equal')
    a.set_title(t, fontsize=9); plt.colorbar(im, ax=a, fraction=0.046)
    a.set_xlabel('element index in S'); a.set_ylabel('element index in S')
fig.suptitle('FIG 8 — element-pair mixed-partial asymmetry (30 selected elements)\n'
             'physical Frobenius skew ratio %.2e vs filtered %.2e'
             % (mixed['physical']['fro_skew_ratio'], mixed['filtered']['fro_skew_ratio']), fontsize=9)
save(fig, 'FIG_08_mixed_partial_asymmetry')

# ---- 9. inner KKT residual breakdown ------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
lim = np.quantile(np.abs(F['kkt_resid'] / sRow), 0.99)
im = ax[0].imshow(R(F['kkt_resid'] / sRow), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
ax[0].set_title('normalized inner-KKT residual (with box multipliers)', fontsize=9)
ax[0].set_xticks([]); ax[0].set_yticks([]); plt.colorbar(im, ax=ax[0], fraction=0.012)
names = [c['class'] for c in byc]; vals = [c['kkt_norm_rms'] for c in byc]
b = ax[1].bar(names, vals, color=['#B0B0B0', '#DD8452', '#55A868', '#4C72B0'])
ax[1].bar_label(b, fmt='%.3f', fontsize=8)
ax[1].set_ylabel('normalized KKT residual RMS'); ax[1].grid(alpha=.3, axis='y')
ax[1].set_title('by density class — the residual lives in VOID', fontsize=9)
fig.suptitle('FIG 9 — final inner MMA subproblem: where its own KKT fails', fontsize=10)
save(fig, 'FIG_09_inner_kkt_breakdown')

# ---- 10. complementarity ------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lam = np.array(kkt['kkt_production']['dual']['lam'])
fv = np.array(kkt['kkt_production']['primal']['fval'])
lbl = ['spectral mode 1', 'spectral mode 2', 'next mode J=3', 'volume']
xp = np.arange(4)
ax[0].bar(xp - .2, lam, .4, label='dual λᵢ', color='#4C72B0')
ax[0].bar(xp + .2, -fv, .4, label='−fᵢ (slack)', color='#DD8452')
ax[0].set_yscale('symlog', linthresh=1e-8); ax[0].set_xticks(xp)
ax[0].set_xticklabels(lbl, fontsize=8, rotation=15)
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')
ax[0].set_title('active set: mode 1 and volume; λ·f ≤ 4.7e−6', fontsize=9)
ax[1].hist(np.log10(np.maximum(F['xsi'] * np.maximum(F['drho'] - (-0.01), 1e-30), 1e-20)), bins=60,
           color='#55A868', alpha=.8)
ax[1].set_xlabel('log₁₀ ξ·(x−xmin)  (bound complementarity)'); ax[1].set_ylabel('count')
ax[1].grid(alpha=.3); ax[1].set_title('box complementarity residual', fontsize=9)
fig.suptitle('FIG 10 — complementarity of the final inner subproblem', fontsize=10)
save(fig, 'FIG_10_complementarity')

# ---- 11. residuals by density class -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
cl = {'void': F['rho'] < 0.1, 'gray-shell': (F['gray'] > 0) & (F['core'] == 0),
      'gray-core': F['core'] > 0, 'solid': F['rho'] > 0.9}
data = [np.abs(F['kkt_resid'][m]) / sRow for m in cl.values()]
ax[0].boxplot(data, tick_labels=list(cl), showfliers=False)
ax[0].set_yscale('log'); ax[0].set_ylabel('|normalized inner-KKT residual|')
ax[0].grid(alpha=.3); ax[0].set_title('inner-subproblem residual by class', fontsize=9)
rat = [c['gFilt_over_gPhys'] for c in byc]
b = ax[1].bar(names, rat, color=['#B0B0B0', '#DD8452', '#55A868', '#4C72B0'])
ax[1].bar_label(b, fmt='%.2f', fontsize=8); ax[1].set_yscale('log')
ax[1].axhline(1, color='k', ls=':', lw=1)
ax[1].set_ylabel('RMS |g_filt| / RMS |g_phys|'); ax[1].grid(alpha=.3, axis='y')
ax[1].set_title('filter amplification by class', fontsize=9)
fig.suptitle('FIG 11 — residuals and filter action by density class', fontsize=10)
save(fig, 'FIG_11_residual_by_class')

# ---- 12. gray-core vs surrounding gray + mechanism decomposition --------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
im = ax[0].imshow(R(F['core'] * 2 + F['gray']), cmap='viridis', aspect='equal')
ax[0].set_title('0 solid/void · 1 gray shell · 3 broad gray core', fontsize=9)
ax[0].set_xticks([]); ax[0].set_yticks([])
dr = dec['rows']
s1 = np.array([abs(r['S1']) for r in dr]); s2 = np.array([abs(r['S2']) for r in dr])
idx = np.arange(len(dr))
ax[1].bar(idx, s1 / (s1 + s2), label='S1: ρ-weighting term  A·D_{g/ρ}', color='#DD8452')
ax[1].bar(idx, s2 / (s1 + s2), bottom=s1 / (s1 + s2), label='S2: A·Hess term', color='#4C72B0')
ax[1].set_xticks(idx); ax[1].set_xticklabels([r['pair'] for r in dr], rotation=65, fontsize=6.5)
ax[1].set_ylabel('share of the measured antisymmetry'); ax[1].legend(fontsize=8)
ax[1].set_title('which mechanism makes the field non-conservative', fontsize=9)
fig.suptitle('FIG 12 — gray-core geometry and the mechanism split (median S2 share %.3f)'
             % dec['median_S2_share'], fontsize=10)
save(fig, 'FIG_12_core_and_mechanism')

# ---- 13. inner-iteration convergence history (supplementary) ------------
conv = J('inner_convergence.json')
it = np.array(conv['hist_iter']); rs = np.array(conv['hist_relStep'])
dx = np.array(conv['hist_maxAbsDrho'])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(it, rs, color='firebrick', lw=1.1)
ax[0].axhline(0.05, color='k', ls='--', lw=1.2, label='production tolInner = 0.05')
ax[0].axvline(19, color='#4C72B0', ls=':', lw=1.4, label='production stops at 19')
ax[0].set_xlabel('inner iteration'); ax[0].set_ylabel('relStep')
ax[0].set_title('relative step never reaches a fixed point\n(min %.2e, oscillates, non-monotone)'
                % conv['relStep']['min'], fontsize=9)
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which='both')
ax[1].semilogx(it, dx / conv['maxAbsDrho']['moveLimit'], color='#55A868', lw=1.3)
ax[1].axhline(1, color='k', ls='--', lw=1.2, label='move limit 0.01')
ax[1].axvline(19, color='#4C72B0', ls=':', lw=1.4, label='production stops at 19')
ax[1].set_xlabel('inner iteration'); ax[1].set_ylabel('max|Δρ| / move limit')
ax[1].set_title('the surrogate drives the increment to the move limit\n'
                '6.2% at production, 97.9% at 5000', fontsize=9)
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which='both')
fig.suptitle('FIG 13 — FROZEN-SUBPROBLEM CERTIFICATION: the final inner MMA problem does not converge',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'FIG_13_inner_convergence')
