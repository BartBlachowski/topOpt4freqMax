#!/usr/bin/env python3
"""tr_figures -- the eleven figures required by the brief (Phase 23)."""
import os, sys, json, hashlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
CV    = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV    = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')
FIG   = os.path.join(STUDY, 'figures'); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})
CP, C1, C2, CF = '#8c8c8c', '#d62728', '#2ca02c', '#1f77b4'   # P, S1, S2, F
MESH = [('m160', 'C160x20', 160, 20), ('m320', 'C320x40', 320, 40), ('m400', 'C400x50', 400, 50)]
A = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
written = []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    written.append(dict(file=f'figures/{name}',
                        sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()))
    print('  ', name)


def csv(tag):
    return np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                         delimiter=',', names=True)


def marks(ax, m, ylim=None):
    for k, c, lab in ((m['S1']['iteration'], C1, 'S1'), (m['S2']['iteration'], C2, 'S2'),
                      (m['F']['iteration'], CF, 'F')):
        ax.axvline(k, color=c, ls='--', lw=0.9, alpha=0.85)
        y = ax.get_ylim()[1] if ylim is None else ylim
        ax.annotate(f'{lab} {k}', (k, y), xytext=(2, -9), textcoords='offset points',
                    color=c, fontsize=6.5, rotation=90, va='top')


# ---- F1  M_nd, P vs S1 vs S2 vs F, per mesh -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['Mnd'], color='0.25', lw=0.9, label='candidate trajectory')
    ax.axhline(m['production']['Mnd'], color=CP, lw=1.4, label=f"P {m['production']['Mnd']:.3f}")
    for st, c, lab in ((m['S1'], C1, 'S1'), (m['S2'], C2, 'S2'), (m['F'], CF, 'F')):
        ax.plot(st['iteration'], st['Mnd'], 'o', color=c, ms=5,
                label=f"{lab} @{st['iteration']}  {st['Mnd']:.4f}")
    ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel(r'$M_{nd}$ (%)')
    ax.set_title(f'{nx}×{ny}   $M_{{nd}}$:  P / S1 / S2 / F'); ax.legend(loc='upper right')
save(fig, 'F1_Mnd_P_S1_S2_F.png')

# ---- F2  omega1, per mesh ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['omega1'], color='0.25', lw=0.9, label=r'candidate $\omega_1$')
    ax.axhline(m['production']['omega1'], color=CP, lw=1.4,
               label=f"P {m['production']['omega1']:.4f}")
    for st, c, lab in ((m['S1'], C1, 'S1'), (m['S2'], C2, 'S2'), (m['F'], CF, 'F')):
        ax.plot(st['iteration'], st['omega1'], 'o', color=c, ms=5,
                label=f"{lab} @{st['iteration']}  {st['omega1']:.4f}")
    lo = min(m['production']['omega1'], m['S1']['omega1']) - 1.2
    hi = max(m['production']['omega1'], m['F']['omega1'], m['S2']['omega1']) + 0.9
    ax.set_ylim(lo, hi)
    ax.set_xlabel('outer iteration'); ax.set_ylabel(r'$\omega_1$')
    ax.set_title(f'{nx}×{ny}   $\\omega_1$:  P / S1 / S2 / F'); ax.legend(loc='lower right')
save(fig, 'F2_omega1_P_S1_S2_F.png')

# ---- F3  move history with the stage-1 and stage-2 E events -------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.step(T['outer'], T['move'], where='post', color='0.25', lw=1.1, label='move (four-rung)')
    k1, k2 = m['S1']['iteration'], m['S2']['iteration']
    ax.step([1, k1, k2], [0.04, 0.02, 0.02], where='post', color=C2, lw=2.0, alpha=0.6,
            label='move (two-rung counterfactual)')
    ax.axvline(k1, color=C1, ls='--', lw=1.0)
    ax.axvline(k2, color=C2, ls='--', lw=1.0)
    ax.annotate(f"E$_1$ {k1} ({m['S1']['branch']})", (k1, 0.041), fontsize=6.5, color=C1,
                rotation=90, va='bottom')
    ax.annotate(f"E$_2$ {k2} ({m['S2']['branch']})", (k2, 0.021), fontsize=6.5, color=C2,
                rotation=90, va='bottom')
    ax.set_yscale('log'); ax.set_yticks([0.005, 0.01, 0.02, 0.04])
    ax.set_yticklabels(['0.005', '0.01', '0.02', '0.04'])
    ax.set_xlabel('outer iteration'); ax.set_ylabel('move limit')
    ax.set_title(f'{nx}×{ny}   move ladder and frozen E events'); ax.legend(loc='lower left')
save(fig, 'F3_move_history_events.png')

# ---- F4  cumulative inner MMA work --------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['cumInner'], color='0.25', lw=1.0)
    for st, c, lab in ((m['S1'], C1, 'S1'), (m['S2'], C2, 'S2'), (m['F'], CF, 'F')):
        ax.plot(st['iteration'], st['innerCumulative'], 'o', color=c, ms=5,
                label=f"{lab} @{st['iteration']}  {st['innerCumulative']:,}")
    ax.axhline(m['production']['innerTotal'], color=CP, lw=1.2,
               label=f"P {m['production']['innerTotal']:,}")
    ax.set_xlabel('outer iteration'); ax.set_ylabel('cumulative inner MMA iterations')
    c = m['cost']
    ax.set_title(f"{nx}×{ny}   inner work   (stop at S2 saves {c['saved_inner_pct']:.1f}%)")
    ax.legend(loc='upper left')
save(fig, 'F4_cumulative_inner_work.png')

# ---- F5  cumulative wall time, with the reliability caveat on the face ---
fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    cw = np.cumsum(T['tOuter'])
    ax.plot(T['outer'], cw, color='0.25', lw=1.0)
    for st, c, lab in ((m['S1'], C1, 'S1'), (m['S2'], C2, 'S2'), (m['F'], CF, 'F')):
        ax.plot(st['iteration'], st['wall_s_cumulative'], 'o', color=c, ms=5,
                label=f"{lab} {st['wall_s_cumulative']:,.0f} s")
    w = m['wall_reliability']
    ax.set_xlabel('outer iteration'); ax.set_ylabel('cumulative wall time (s)')
    ax.set_title(f"{nx}×{ny}   wall time — NOT RELIABLE\n"
                 f"s/inner drifts {w['drift_ratio']:.1f}× within the run")
    ax.legend(loc='upper left')
save(fig, 'F5_cumulative_wall_time.png')

# ---- F6  rung-2 scientific value ----------------------------------------
def value_fig(block, key, title, fname, bars):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    labels = [f'{nx}×{ny}' for _, _, nx, ny in MESH]
    x = np.arange(3)
    for ax, (fld, ttl, bar, sgn) in zip(axes, bars):
        v = [A['mesh'][k][block][fld] for k, _, _, _ in MESH]
        cols = [C2 if sgn * vv >= bar else '0.72' for vv in v]
        ax.bar(x, v, color=cols, width=0.6)
        if bar is not None:
            ax.axhline(sgn * bar, color='#d62728', ls='--', lw=1.1,
                       label=f'materiality bar {sgn*bar:g}')
            ax.legend(loc='best')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(ttl)
        for xi, vv in zip(x, v):
            ax.annotate(f'{vv:+.3f}', (xi, vv), ha='center',
                        va='bottom' if vv >= 0 else 'top', fontsize=7)
    fig.suptitle(title, fontsize=10)
    save(fig, fname)

BARS = [('dMnd_rel_pct', r'$\Delta M_{nd}$ (% rel., lower is better)', 2.0, -1),
        ('domega1_rel_pct', r'$\Delta\omega_1$ (% rel., higher is better)', 0.10, +1),
        ('rho_mean_abs', r'mean $|\Delta\rho_e|$', 0.01, +1),
        ('dgray', r'$\Delta$ gray fraction', 0.01, +1)]
value_fig('rung2', 'm160', 'RUNG 2 (move = 0.02):  scientific value of S1 → S2',
          'F6_rung2_value.png', BARS)
value_fig('rung34', 'm160', 'RUNGS 3+4 (0.01, 0.005):  incremental value of S2 → F',
          'F7_rung34_value.png', BARS)

# ---- F8  rung-2 vs rungs-3+4 cost ---------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
labels = [f'{nx}×{ny}' for _, _, nx, ny in MESH]
x = np.arange(3); w = 0.36
for ax, (fld, ttl, logy) in zip(axes, [('outer', 'outer iterations', False),
                                       ('inner', 'inner MMA iterations', True)]):
    r2 = [A['mesh'][k]['cost'][f'rung2_{fld}'] for k, _, _, _ in MESH]
    r34 = [A['mesh'][k]['cost'][f'rung34_{fld}'] for k, _, _, _ in MESH]
    ax.bar(x - w/2, r2, w, color=C2, label='rung 2 (0.02)')
    ax.bar(x + w/2, r34, w, color=CF, label='rungs 3+4 (0.01, 0.005)')
    for xi, a_, b_ in zip(x, r2, r34):
        ax.annotate(f'{a_:,}', (xi - w/2, a_), ha='center', va='bottom', fontsize=6.5)
        ax.annotate(f'{b_:,}', (xi + w/2, b_), ha='center', va='bottom', fontsize=6.5)
    if logy: ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ttl)
    ax.set_title(f'cost: {ttl}'); ax.legend()
fig.suptitle('Cost of rung 2 versus rungs 3+4  (wall time excluded: unreliable)', fontsize=10)
save(fig, 'F8_rung_cost.png')

# ---- F9  S2 vs F topology ------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(11.5, 5.4))
for row, (k, tag, nx, ny) in enumerate(MESH):
    m = A['mesh'][k]
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO = np.array(h['RHO']).T
    s2 = RHO[:, m['S2']['iteration'] - 1].reshape(ny, nx, order='F')
    fF = RHO[:, m['F']['iteration'] - 1].reshape(ny, nx, order='F')
    for ax, im, ttl in ((axes[row, 0], s2, f"S2 @{m['S2']['iteration']}  (move 0.02)"),
                        (axes[row, 1], fF, f"F @{m['F']['iteration']}  ({m['F']['status']})"),
                        (axes[row, 2], fF - s2, 'F − S2')):
        if ttl == 'F − S2':
            h_ = ax.imshow(im, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        else:
            h_ = ax.imshow(1 - im, cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(h_, ax=ax, fraction=0.03)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f'{nx}×{ny}  {ttl}', fontsize=8)
    r = m['rung34']
    axes[row, 2].set_xlabel(f"mean|Δρ|={r['rho_mean_abs']:.5f}  max|Δρ|={r['rho_max_abs']:.4f}",
                            fontsize=7)
fig.suptitle('S2 versus F topology — what rungs 3+4 actually change', fontsize=10)
save(fig, 'F9_topology_S2_vs_F.png')

# ---- F10  S2 vs F multiplicity / gap ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['gap12'], color='0.3', lw=0.9, label=r'$(\omega_2-\omega_1)/\omega_1$')
    for st, c, lab in ((m['S1'], C1, 'S1'), (m['S2'], C2, 'S2'), (m['F'], CF, 'F')):
        ax.plot(st['iteration'], st['gap12'], 'o', color=c, ms=5,
                label=f"{lab}  gap {st['gap12']:.5f}")
    ax.axhline(m['production']['gap12'], color=CP, lw=1.1,
               label=f"P {m['production']['gap12']:.5f}")
    ax2 = ax.twinx(); ax2.plot(T['outer'], T['multN'], color='#ff7f0e', lw=0.8, alpha=0.7)
    ax2.set_ylim(0, 3); ax2.set_ylabel('subspace size N', color='#ff7f0e'); ax2.grid(False)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('relative gap')
    ax.set_title(f'{nx}×{ny}   multiplicity / gap'); ax.legend(loc='center right')
save(fig, 'F10_multiplicity_gap.png')

# ---- F11  architecture summary ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
x = np.arange(3); w = 0.2
for ax, (fld, ttl, fmt) in zip(
        axes, [('Mnd', r'$M_{nd}$ (%)', '{:.2f}'), ('omega1', r'$\omega_1$', '{:.2f}'),
               ('cum', 'cumulative inner MMA iterations', '{:,.0f}')]):
    for j, (nm, c) in enumerate((('production', CP), ('S1', C1), ('S2', C2), ('F', CF))):
        if fld == 'cum':
            v = [A['mesh'][k]['production']['innerTotal'] if nm == 'production'
                 else A['mesh'][k][nm]['innerCumulative'] for k, _, _, _ in MESH]
        else:
            v = [A['mesh'][k][nm][fld] for k, _, _, _ in MESH]
        ax.bar(x + (j - 1.5) * w, v, w, color=c, label={'production': 'P'}.get(nm, nm))
    if fld == 'cum': ax.set_yscale('log')
    if fld == 'omega1': ax.set_ylim(160, 175)
    ax.set_xticks(x); ax.set_xticklabels([f'{nx}×{ny}' for _, _, nx, ny in MESH])
    ax.set_ylabel(ttl); ax.set_title(ttl); ax.legend(ncol=4, fontsize=6.5)
fig.suptitle('Architecture summary:  P / S1 / S2 / F across the three primary meshes\n'
             'two-rung endpoint S2 = first frozen E on move = 0.02', fontsize=10)
save(fig, 'F11_architecture_summary.png')

json.dump(written, open(os.path.join(STUDY, 'evidence', 'figures.json'), 'w'), indent=1)
print(f'\n{len(written)} figures written')
