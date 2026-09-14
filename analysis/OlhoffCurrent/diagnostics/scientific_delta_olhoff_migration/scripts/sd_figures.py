#!/usr/bin/env python3
"""Data figures 1-12 and 15-19.  Palette: reference categorical slots in fixed order
(1 blue = target C480, 2 orange = source M1, 3 aqua = source S480, 4 yellow = target
production endpoint).  Density maps use one-hue light->dark; differences a blue<->red
diverging map with a neutral gray midpoint.  One y-axis per panel, always."""
import csv
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sd_common import *
from sd_same_state_compare import load, state_rho, inner_list

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C_T, C_M1, C_S, C_P = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
LBL = {'C480': 'target C480 (SIMP+eq.4b, three-rung ladder)', 'M1': 'source M1 (SIMP+eq.4b, adaptive box)',
       'S480': 'source S480 (Pedersen+eq.2, adaptive box)'}
COL = {'C480': C_T, 'M1': C_M1, 'S480': C_S}
DIV = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#0b0b0b'])
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'lines.linewidth': 1.6,
                     'legend.frameon': False})


def read(name):
    rows = list(csv.DictReader(open(EVAL / name)))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if set(vals) <= {'True', 'False'}:
            out[k] = np.array([v == 'True' for v in vals])
        else:
            try:
                out[k] = np.array([float(v) if v not in ('', 'nan') else np.nan for v in vals])
            except ValueError:
                out[k] = np.array(vals)
    return out


T = {'C480': read('trajectory_C480.csv'), 'M1': read('trajectory_M1.csv'), 'S480': read('trajectory_S480.csv')}
RHOF = {'C480': np.load(EVAL / 'C480_final_rho.npy'), 'M1': np.load(EVAL / 'M1_final_rho.npy'), 'S480': np.load(EVAL / 'S480_final_rho.npy')}
PROD = dict(outer=164, omega1=161.906, Mnd=0.346717)   # Sept-11 canonical production endpoint (nine_mesh_campaign_audit)


def img(ax, rho, title, cmap=SEQ, vmin=0, vmax=1):
    m = ax.imshow(np.asarray(rho).reshape(NELX, NELY).T, origin='lower', extent=[0, 8, 0, 1], cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, loc='left', fontsize=9); ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([0, 1]); ax.grid(False)
    return m


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches='tight'); plt.close(fig)


def traj(key, ylabel, name, title, sides=('C480', 'M1', 'S480'), logy=False, prod=None, hline=None, extra=None):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for s in sides:
        if key in T[s]:
            ax.plot(T[s]['outer'], T[s][key], color=COL[s], label=LBL[s], lw=1.4)
    if prod is not None:
        ax.plot([PROD['outer']], [prod], 'o', ms=8, color=C_P, mec=SURF, mew=2, label='target canonical production endpoint (no trajectory retained)')
    if hline is not None:
        ax.axhline(hline[0], color=INK2, lw=1, ls='--'); ax.text(ax.get_xlim()[1], hline[0], hline[1], ha='right', va='bottom', color=INK2, fontsize=8)
    if extra:
        extra(ax)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('outer iteration'); ax.set_ylabel(ylabel); ax.set_title(title, loc='left')
    ax.legend(fontsize=7.5, loc='best')
    save(fig, name)


def main():
    FIG.mkdir(exist_ok=True)
    # 1 final rho
    fig, axs = plt.subplots(3, 1, figsize=(9, 4.6), layout='constrained')
    for ax, s in zip(axs, ['S480', 'C480', 'M1']):
        d = discreteness(RHOF[s])
        m = img(ax, RHOF[s], f"{LBL[s]} — final, M_nd {d['Mnd']:.3f}, gray {d['gray']:.3f}")
    fig.colorbar(m, ax=axs, label='ρ', shrink=.8)
    save(fig, 'fig01_final_rho_480.png')
    # 2 difference
    fig, ax = plt.subplots(figsize=(9, 1.9), layout='constrained')
    m = img(ax, RHOF['S480'] - RHOF['C480'], 'ρ(source S480) − ρ(target C480), final designs', cmap=DIV, vmin=-1, vmax=1)
    fig.colorbar(m, ax=ax, label='Δρ', shrink=.9)
    save(fig, 'fig02_rho_difference_S480_minus_C480.png')
    # 3 histograms (small multiples, one axis each)
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True, layout='constrained')
    bins = np.linspace(0, 1, 51)
    for ax, s in zip(axs, ['S480', 'C480', 'M1']):
        ax.hist(RHOF[s], bins=bins, color=COL[s], edgecolor=SURF, linewidth=0.5)
        ax.set_yscale('log'); ax.set_title(LBL[s].split(' (')[0], loc='left'); ax.set_xlabel('ρ')
    axs[0].set_ylabel('elements (log)')
    save(fig, 'fig03_density_histograms.png')
    # 4-7, 9-12 trajectories
    traj('Mnd', 'M_nd = 4·mean(ρ(1−ρ))', 'fig04_Mnd_trajectories.png', 'M_nd per outer iteration (480×60)', prod=PROD['Mnd'])
    spk = lambda ax: [ax.axvline(k, color=C_M1, lw=0.5, alpha=.35) for k in T['M1']['outer'][T['M1']['spike']]]
    traj('omega1', 'ω₁ at iteration start [rad/s]', 'fig05_omega1_trajectories.png',
         'ω₁ (native model); faint orange lines mark M1 spike events', prod=PROD['omega1'], extra=spk)
    traj('omega2', 'ω₂ at iteration start [rad/s]', 'fig06_omega2_trajectories.png', 'ω₂ (native model)')
    traj('gap12', '(ω₂−ω₁)/ω₁', 'fig07_gap12_trajectories.png', 'Relative gap ω₁–ω₂', logy=True)
    traj('max_abs_drho', 'max |Δρ|', 'fig09_max_drho_trajectories.png', 'Largest element increment per outer iteration', logy=True)
    traj('nInner', 'inner MMA sub-iterations', 'fig10_inner_iterations.png', 'Inner sub-iterations per outer iteration (same innerLoop code)')
    traj('gray', 'fraction 0.1<ρ<0.9', 'fig11_gray_fraction.png', 'Gray fraction (S480 retains only the final state: point)', sides=('C480', 'M1'),
         extra=lambda ax: ax.plot([112], [discreteness(RHOF['S480'])['gray']], 'o', ms=8, color=C_S, mec=SURF, mew=2, label=LBL['S480'] + ' final'))
    traj('mid', 'fraction 0.4≤ρ≤0.6', 'fig12_mid_fraction.png', 'Mid-density fraction (S480: final point only)', sides=('C480', 'M1'),
         extra=lambda ax: ax.plot([112], [discreteness(RHOF['S480'])['mid']], 'o', ms=8, color=C_S, mec=SURF, mew=2, label=LBL['S480'] + ' final'))
    # 8 move/box: two panels (max and mean) on the same scale, plus M1 floor fraction panel
    fig, axs = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True, layout='constrained')
    for s in ['C480', 'M1', 'S480']:
        axs[0].plot(T[s]['outer'], T[s]['box_max'], color=COL[s], label=LBL[s])
        axs[1].plot(T[s]['outer'], T[s]['box_mean'], color=COL[s], label=LBL[s])
    axs[2].plot(T['M1']['outer'], T['M1']['box_at_floor'], color=C_M1, label='M1: fraction of elements at box floor 0.002')
    axs[0].set_ylabel('box: max over elements'); axs[1].set_ylabel('box: mean over elements'); axs[2].set_ylabel('fraction at floor')
    for ax in axs[:2]:
        ax.set_yscale('log')
    axs[0].legend(fontsize=7.5); axs[2].legend(fontsize=7.5); axs[2].set_xlabel('outer iteration')
    axs[0].set_title('Move limit / per-element box (target: one global value per rung)', loc='left')
    save(fig, 'fig08_move_box_trajectories.png')
    # 15 same-rho eigenvalues
    states = ['rho0', 'C480_k020', 'C480_k100', 'C480_k386', 'S480_final', 'M1_k005', 'M1_k011', 'M1_k064']
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained')
    x = np.arange(len(states))
    for off, ev, c, lab in [(-0.2, 'T', C_T, 'target +impl (SIMP+eq.4b)'), (0.0, 'S1', C_M1, 'source code, SIMP+eq.4b'), (0.2, 'S0', C_S, 'source code, Pedersen+eq.2')]:
        w = np.array([load(st, ev).omega[0] for st in states])
        axs[0].plot(x + off, w, 'o', ms=8, color=c, mec=SURF, mew=2, label=lab, ls='none')
    axs[0].set_xticks(x, states, rotation=35, ha='right', fontsize=7.5); axs[0].set_ylabel('ω₁ [rad/s]')
    axs[0].set_title('ω₁ at identical frozen ρ (T and S1 markers coincide: bitwise)', loc='left'); axs[0].legend(fontsize=7.5)
    rel = [abs(load(st, 'S0').omega[0] - load(st, 'S1').omega[0]) / load(st, 'S1').omega[0] for st in states]
    axs[1].bar(x, np.maximum(rel, 1e-17), color=C_S, edgecolor=SURF)
    axs[1].set_yscale('log'); axs[1].set_xticks(x, states, rotation=35, ha='right', fontsize=7.5)
    axs[1].set_ylabel('|ω₁(S0) − ω₁(S1)| / ω₁(S1)'); axs[1].set_title('Formulation effect on ω₁ (zero bars = exactly 0, drawn at 1e-17)', loc='left')
    save(fig, 'fig15_same_rho_eigenvalues.png')
    # 16/17 gradients at S480_final
    r = state_rho('S480_final'); A0 = load('S480_final', 'S0'); A1 = load('S480_final', 'S1'); AT = load('S480_final', 'T')
    for which, fname, ttl in [('Fraw', 'fig16_same_rho_raw_gradient.png', 'raw f₁₁'), ('Ffilt', 'fig17_same_rho_filtered_gradient.png', 'filtered f₁₁')]:
        a0 = np.asarray(getattr(A0, which))[:, 0, 0]; a1 = np.asarray(getattr(A1, which))[:, 0, 0]; at = np.asarray(getattr(AT, which))[:, 0, 0]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
        axs[0].plot(at, a1, '.', ms=2, color=C_M1)
        lim = [min(at.min(), a1.min()), max(at.max(), a1.max())]; axs[0].plot(lim, lim, color=INK2, lw=1)
        axs[0].set_xlabel(f'{ttl}, target +impl'); axs[0].set_ylabel(f'{ttl}, source SIMP+eq.4b')
        axs[0].set_title(f'Implementation identity at S480 final ρ: bitwise = {bool(np.array_equal(at, a1))}', loc='left')
        lo = r < 0.1
        axs[1].plot(a1[~lo], a0[~lo], '.', ms=2, color=C_T, label='ρ ≥ 0.1')
        axs[1].plot(a1[lo], a0[lo], '.', ms=2, color=C_P, label='ρ < 0.1')
        lim = [min(a1.min(), a0.min()), max(a1.max(), a0.max())]; axs[1].plot(lim, lim, color=INK2, lw=1)
        axs[1].set_xlabel(f'{ttl}, SIMP+eq.4b'); axs[1].set_ylabel(f'{ttl}, Pedersen+eq.2'); axs[1].legend(fontsize=7.5, markerscale=5)
        axs[1].set_title('Formulation operator at the same ρ (source code)', loc='left')
        save(fig, fname)
    # 18 first step at rho0
    T0 = load('rho0', 'T'); S10 = load('rho0', 'S1')
    d4 = inner_list(T0)[0].drho; d10 = inner_list(S10)[1].drho; d4s = inner_list(S10)[0].drho
    fig, axs = plt.subplots(3, 1, figsize=(9, 4.8), layout='constrained')
    img(axs[0], d4, f'target, box 0.04: ‖Δρ‖₂={np.linalg.norm(d4):.2f}', cmap=DIV, vmin=-.1, vmax=.1)
    img(axs[1], d4s, f'source SIMP+eq.4b, common box 0.04: bitwise equal to target = {bool(np.array_equal(d4, d4s))}', cmap=DIV, vmin=-.1, vmax=.1)
    m = img(axs[2], d10, f'source native box 0.10: ‖Δρ‖₂={np.linalg.norm(d10):.2f}, cos to target step {np.dot(d4, d10)/np.linalg.norm(d4)/np.linalg.norm(d10):.3f}', cmap=DIV, vmin=-.1, vmax=.1)
    fig.colorbar(m, ax=axs, label='Δρ (first problem-(25) solve at ρ₀)', shrink=.8)
    save(fig, 'fig18_problem25_first_step_rho0.png')
    # 19 stopping metric
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for s in ['C480', 'M1', 'S480']:
        ax.plot(T[s]['outer'], T[s]['l2_drho'], color=COL[s], label=LBL[s], lw=1.2)
    ax.axhline(EPS, color=INK2, ls='--', lw=1); ax.text(150, EPS*0.62, 'ε = 0.15 (source stop: ‖Δρ‖₂ < ε, no persistence)', ha='left', va='top', fontsize=8, color=INK2)
    for k, lab in [(309, 'target rung 2'), (348, 'rung 3'), (386, 'E declared (branch B, P=20)')]:
        ax.axvline(k, color=C_T, lw=0.8, ls=':'); ax.text(k, ax.get_ylim()[0] if False else 20, lab, rotation=90, fontsize=7, color=C_T, va='top', ha='right')
    ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel('‖Δρ‖₂'); ax.legend(fontsize=7.5, loc='lower left')
    ax.set_title('Stopping metric: source ε-test vs target stage-exhaustion controller', loc='left')
    save(fig, 'fig19_stopping_metric.png')
    print('figures written')


if __name__ == '__main__':
    main()
