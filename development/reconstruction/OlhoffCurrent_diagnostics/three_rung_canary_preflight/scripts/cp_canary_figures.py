#!/usr/bin/env python3
"""cp_canary_figures.py -- the required figures, from canary data only.

Figures 1-10 are drawn per canary; 11 and 12 need both canaries / the legacy
density field and are drawn only when their inputs exist.  Nothing is drawn
from substituted data.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parents[1]
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
C = {1: '#4C72B0', 2: '#DD8452', 3: '#55A868'}


def _cols(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) if r[k] not in ('', 'NaN') else np.nan
                         for r in rows]) for k in rows[0]}


def load(mesh):
    """Frozen cv_export schema, merged with the Part-B supplement columns
    (omega3..5, gap23, tEig/tGrad/tInner/tOther) that the frozen schema does
    not carry.  The supplement never overwrites a frozen column."""
    tag = f'C{mesh}_three_rung'
    d = _cols(HERE / f'runs/{tag}_iterations.csv')
    sup = HERE / f'runs/{tag}_supplement.csv'
    if sup.exists():
        for k, v in _cols(sup).items():
            d.setdefault(k, v)
    rec = json.load(open(HERE / f'runs/{tag}_record.json'))
    return d, rec


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def stage_bands(ax, d):
    """Shade the three move stages and mark the declarations that ended them."""
    st = d['stage']
    for s in (1, 2, 3):
        idx = np.where(st == s)[0]
        if idx.size:
            ax.axvspan(idx[0] + 1, idx[-1] + 1, color=C[s], alpha=.07, lw=0)
    ch = np.where(np.diff(st) > 0)[0] + 2
    for x in ch:
        ax.axvline(x, color='k', ls='--', lw=.8, alpha=.6)
    return ch


def per_canary(mesh):
    d, rec = load(mesh)
    n = len(d['outer'])
    x = d['outer']
    m = mesh.replace('x', '×')
    dec = rec['terminalDeclIter']

    # --- 1/2. omega trajectory ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stage_bands(ax, d)
    ax.plot(x, d['omega1'], label='ω₁', color='#4C72B0')
    ax.plot(x, d['omega2'], label='ω₂', color='#DD8452', lw=1)
    if 'omega3' in d and np.isfinite(d.get('omega3', np.array([np.nan]))).any():
        ax.plot(x, d['omega3'], label='ω₃', color='#55A868', lw=1)
    ax.axvline(dec, color='firebrick', lw=1.4)
    ax.annotate(f'terminal E declared\nit {dec} (branch {rec["terminalBranch"]})',
                (dec, ax.get_ylim()[0]), textcoords='offset points',
                xytext=(-96, 26), color='firebrick', fontsize=8)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('ω')
    ax.set_title(f'{m} three-rung — eigenfrequency trajectory', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f'FIG_{"1" if mesh.startswith("480") else "2"}_omega_{mesh}')

    # --- 3. Mnd / grayness ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stage_bands(ax, d)
    ax.plot(x, d['Mnd'], label='M_nd [%]', color='#4C72B0')
    ax.plot(x, 100 * d['gray'], label='grayness ×100', color='#DD8452', lw=1)
    ax.plot(x, 100 * d['mid'], label='mid fraction 0.4–0.6 ×100', color='#55A868', lw=1)
    ax.axvline(dec, color='firebrick', lw=1.4)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('[%]')
    ax.set_title(f'{m} three-rung — discreteness', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f'FIG_3_discreteness_{mesh}')

    # --- 4. move / stage timeline ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.step(x, d['move'], where='post', color='#4C72B0')
    ax.set_yscale('log'); ax.set_yticks([0.01, 0.02, 0.04])
    ax.set_yticklabels(['0.01', '0.02', '0.04'])
    for s in (1, 2, 3):
        idx = np.where(d['stage'] == s)[0]
        if idx.size:
            ax.annotate(f'stage {s}: {idx.size} it', ((idx[0] + idx[-1]) / 2, 0.045),
                        ha='center', fontsize=8, color=C[s])
    stage_bands(ax, d)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('move limit')
    ax.set_title(f'{m} three-rung — move / stage timeline', fontsize=10)
    ax.grid(alpha=.3)
    save(fig, f'FIG_4_move_stage_{mesh}')

    # --- 5. A/B/E timeline ----------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1.3]})
    stage_bands(ax[0], d); stage_bands(ax[1], d)
    ax[0].fill_between(x, 0, d['exA'], step='mid', color='#4C72B0', alpha=.75, label='A')
    ax[0].fill_between(x, 0, -d['exB'], step='mid', color='#DD8452', alpha=.75, label='B')
    ax[0].set_yticks([-1, 0, 1]); ax[0].set_yticklabels(['B', '', 'A'])
    ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=.3)
    ax[0].set_title(f'{m} three-rung — branch activity and persistence (P = 20)', fontsize=10)
    ax[1].plot(x, d['exNA'], label='persistence counter A', color='#4C72B0')
    ax[1].plot(x, d['exNB'], label='persistence counter B', color='#DD8452')
    ax[1].axhline(20, color='firebrick', ls=':', lw=1.2)
    ax[1].annotate('P = 20 ⇒ declare', (x[0], 21), fontsize=8, color='firebrick')
    ax[1].set_xlabel('outer iteration'); ax[1].set_ylabel('consecutive iterations')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    save(fig, f'FIG_5_branch_timeline_{mesh}')

    # --- 6. gaps ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stage_bands(ax, d)
    ax.semilogy(x, np.maximum(d['gap12'], 1e-8), label='gap12 = (ω₂−ω₁)/ω₁', color='#4C72B0')
    if np.isfinite(d.get('omega3', np.array([np.nan]))).any():
        g23 = (d['omega3'] - d['omega2']) / d['omega2']
        ax.semilogy(x, np.maximum(g23, 1e-8), label='gap23 = (ω₃−ω₂)/ω₂', color='#55A868')
    ax.axhline(0.05, color='firebrick', ls=':', lw=1.2)
    ax.annotate('multiplicity tolerance 0.05', (x[0], 0.055), fontsize=8, color='firebrick')
    ax.axvline(dec, color='firebrick', lw=1.4)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('relative gap')
    ax.set_title(f'{m} three-rung — spectral gaps', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')
    save(fig, f'FIG_6_gaps_{mesh}')

    # --- 7. multiple-J warning timeline ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.2))
    stage_bands(ax, d)
    w = d['multJ'] > 0
    ax.fill_between(x, 0, w.astype(float), step='mid', color='firebrick', alpha=.8)
    ax.plot(x, np.cumsum(w) / np.arange(1, n + 1), color='k', lw=1,
            label='cumulative fraction of iterations flagged')
    ax.axvline(dec, color='firebrick', lw=1.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('warning / fraction')
    ax.set_title(f'{m} three-rung — next-mode (ω₄ within 5 % of ω₃) warning, '
                 f'{int(w.sum())}/{n} iterations', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f'FIG_7_multiJ_{mesh}')

    # --- 8. inner MMA per outer ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stage_bands(ax, d)
    ax.plot(x, d['nInner'], lw=.9, color='#4C72B0', label='inner MMA iterations')
    ax.plot(x, d['cumInner'] / np.arange(1, n + 1), color='k', lw=1.2,
            label='running mean')
    ax.set_xlabel('outer iteration'); ax.set_ylabel('inner MMA iterations')
    ax.set_title(f'{m} three-rung — inner MMA work per outer '
                 f'(total {int(d["cumInner"][-1])})', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f'FIG_8_inner_{mesh}')

    # --- 9. per-outer timing decomposition -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.4))
    stage_bands(ax, d)
    if 'tEig' in d:
        ax.stackplot(x, d['tInner'], d['tEig'], d['tGrad'], np.maximum(d['tOther'], 0),
                     labels=['inner MMA', 'assembly+eigensolve', 'gradients', 'other'],
                     colors=['#55A868', '#4C72B0', '#DD8452', '#B0B0B0'], alpha=.85)
        ax.plot(x, d['tOuter'], lw=.7, color='k', label='tOuter (total)')
    else:
        ax.plot(x, d['tOuter'], lw=.8, color='k', label='tOuter')
    ax.set_xlabel('outer iteration'); ax.set_ylabel('seconds')
    ax.set_title(f'{m} three-rung — per-outer timing decomposition '
                 f'(nondeterministic telemetry)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)
    save(fig, f'FIG_9_timing_{mesh}')

    # --- 10. cumulative wall ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stage_bands(ax, d)
    ax.plot(x, np.cumsum(d['tOuter']) / 60, color='#4C72B0')
    ax.set_xlabel('outer iteration'); ax.set_ylabel('cumulative wall [min]')
    ax.set_title(f'{m} three-rung — cumulative wall time '
                 f'({np.sum(d["tOuter"])/60:.1f} min)', fontsize=10)
    ax.grid(alpha=.3)
    save(fig, f'FIG_10_cumwall_{mesh}')


if __name__ == '__main__':
    for mesh in sys.argv[1:]:
        per_canary(mesh)
