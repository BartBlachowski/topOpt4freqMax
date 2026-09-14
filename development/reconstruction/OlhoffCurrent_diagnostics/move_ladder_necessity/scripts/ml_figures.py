#!/usr/bin/env python3
"""ml_figures -- the twelve figures required by the brief (Phase 21)."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
CV = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')
FIG = os.path.join(STUDY, 'figures'); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})
CP, CS, CF = '#8c8c8c', '#d62728', '#1f77b4'      # production, single-stage, four-rung
written = []


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight'); plt.close(fig)
    written.append(name); print('  ', name)


A = json.load(open(os.path.join(STUDY, 'evidence', 'ladder_analysis.json')))
KEYS = ['m160', 'm320', 'm400']
LAB = {'m160': '160x20', 'm320': '320x40', 'm400': '400x50'}
D = {}
for k in KEYS:
    m = A['mesh'][k]
    D[k] = dict(m=m, T=np.genfromtxt(os.path.join(CV, 'runs', f"{m['tag']}_iterations.csv"),
                                     delimiter=',', names=True),
                kS=m['S']['iteration'], n=m['F']['iteration'], P=m['production'])

# ---------------------------------------------------------------- F1, F2
for fno, (fld, lab, prodfld) in enumerate([('Mnd', '$M_{nd}$  [%]', 'Mnd'),
                                           ('omega1', '$\\omega_1$', 'omega1')], start=1):
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for ax, k in zip(axs, KEYS):
        d = D[k]; T = d['T']
        ax.plot(T['outer'], T[fld], color=CF, lw=1.2, label='four-rung candidate F')
        ax.axhline(d['P'][prodfld], color=CP, lw=1.2, ls='-', label='production P')
        ax.axvline(d['kS'], color=CS, lw=1.2, ls='--')
        ax.plot([d['kS']], [T[fld][d['kS'] - 1]], 'o', color=CS, ms=6,
                label='single-stage endpoint S')
        ax.set_title(f"{LAB[k]}   P={d['P'][prodfld]:.4g}  "
                     f"S={T[fld][d['kS']-1]:.4g}  F={T[fld][d['n']-1]:.4g}")
        ax.set_xlabel('outer iteration')
    axs[0].set_ylabel(lab); axs[0].legend(loc='best')
    fig.suptitle(f'F{fno}  {lab}: production vs single-stage vs four-rung', y=1.02, fontsize=9)
    save(fig, f'F{fno}_{fld}_P_S_F.png')

# ---------------------------------------------------------------- F3, F4
for fno, (fld, lab, prodfld, sign) in enumerate(
        [('Mnd', 'cumulative $M_{nd}$ gain over production [%-points]', 'Mnd', -1),
         ('omega1', 'cumulative $\\omega_1$ gain over production', 'omega1', +1)], start=3):
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for ax, k in zip(axs, KEYS):
        d = D[k]; T = d['T']
        g = sign * (T[fld] - d['P'][prodfld])
        ax.plot(T['outer'], g, color=CF, lw=1.2)
        ax.axhline(0, color=CP, lw=1.0)
        ax.axvline(d['kS'], color=CS, lw=1.2, ls='--')
        tot = sign * (T[fld][d['n'] - 1] - d['P'][prodfld])
        atS = sign * (T[fld][d['kS'] - 1] - d['P'][prodfld])
        ax.set_title(f"{LAB[k]}   at S: {atS:+.4g}   final: {tot:+.4g}\n"
                     f"({100*atS/tot if tot else float('nan'):.1f} % banked at S)")
        ax.set_xlabel('outer iteration')
    axs[0].set_ylabel(lab)
    fig.suptitle(f'F{fno}  {lab} (dashed red = first A OR B exhaustion)', y=1.02, fontsize=9)
    save(fig, f'F{fno}_cumgain_{fld}.png')

# ---------------------------------------------------------------- F5
fig, axs = plt.subplots(3, 1, figsize=(6.6, 6.6))
for ax, k in zip(axs, KEYS):
    d = D[k]; T = d['T']; m = d['m']
    ax.step(T['outer'], T['move'], where='post', color=CF, lw=1.4)
    ax.axvline(d['kS'], color=CS, lw=1.3, ls='--')
    ax.annotate(f"first E: {m['S']['branch']}@{d['kS']}\nSINGLE-STAGE STOPS HERE",
                xy=(d['kS'], 0.028), xytext=(6, 0), textcoords='offset points',
                fontsize=6.5, color=CS, va='center')
    for r in m['rungs'][1:]:
        ax.axvline(r['iterFrom'], color=CF, lw=0.7, ls=':')
    ax.set_yscale('log'); ax.set_ylim(0.0042, 0.052)
    ax.set_yticks([0.005, 0.01, 0.02, 0.04]); ax.set_yticklabels(['0.005', '0.01', '0.02', '0.04'])
    ax.minorticks_off(); ax.set_ylabel('move')
    ax.set_title(f"{LAB[k]}   four-rung ends {m['F']['status']} @ {d['n']}")
axs[-1].set_xlabel('outer iteration')
fig.suptitle('F5  move history: where the single-stage policy would stop, and the rungs beyond', fontsize=9)
save(fig, 'F5_move_history.png', rect=[0, 0, 1, 0.965])

# ---------------------------------------------------------------- F6, F7
for fno, (fld, lab) in enumerate([('wall', 'cumulative wall time [s]'),
                                  ('cumInner', 'cumulative inner MMA iterations')], start=6):
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for ax, k in zip(axs, KEYS):
        d = D[k]; T = d['T']
        y = np.cumsum(T['tOuter']) if fld == 'wall' else T['cumInner']
        ax.plot(T['outer'], y, color=CF, lw=1.2)
        ax.axvline(d['kS'], color=CS, lw=1.2, ls='--')
        ax.fill_between(T['outer'], 0, y, where=(T['outer'] > d['kS']), color=CS, alpha=0.15)
        frac = 100 * (y[d['n'] - 1] - y[d['kS'] - 1]) / y[d['n'] - 1]
        ax.set_title(f"{LAB[k]}   {frac:.1f} % spent below move=0.04")
        ax.set_xlabel('outer iteration')
        if fld == 'wall':
            ax.set_yscale('log')
    axs[0].set_ylabel(lab)
    fig.suptitle(f'F{fno}  {lab}; shaded = spent on the lower rungs', y=1.02, fontsize=9)
    save(fig, f'F{fno}_cost_{fld}.png')

# ---------------------------------------------------------------- F8
fig, ax = plt.subplots(figsize=(6.0, 3.4))
x = np.arange(3); w = 0.38
mb = [A['mesh'][k]['banked']['pct_Mnd_banked_at_S'] for k in KEYS]
ob = [A['mesh'][k]['banked']['pct_omega1_banked_at_S'] for k in KEYS]
ax.bar(x - w / 2, mb, w, color=CF, label='$M_{nd}$ gain banked at S')
ax.bar(x + w / 2, ob, w, color='#2ca02c', label='$\\omega_1$ gain banked at S')
ax.axhline(98, color='k', ls='--', lw=0.9)
ax.annotate('the >=98 % fine-mesh claim', (-0.45, 98), fontsize=6.5, va='bottom')
ax.axhline(0, color='k', lw=0.8)
for xi, (a, b) in enumerate(zip(mb, ob)):
    ax.annotate(f'{a:.1f}', (xi - w / 2, a), ha='center',
                va='bottom' if a > 0 else 'top', fontsize=6.5)
    ax.annotate(f'{b:.1f}', (xi + w / 2, b), ha='center',
                va='bottom' if b > 0 else 'top', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS])
ax.set_ylabel('% of production->candidate gain already banked at S')
ax.set_title('F8  fraction of the benefit banked before the first descent\n'
             '160x20 is the exception: its $\\omega_1$ gain is entirely below the ladder')
ax.legend(loc='lower right')
save(fig, 'F8_banked_fraction.png')

# ---------------------------------------------------------------- F9
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.3))
for k in KEYS:
    m = A['mesh'][k]
    rr = m['rungs']
    axs[0].plot([r['rung'] for r in rr], [-r['dMnd_rel_pct'] for r in rr],
                'o-', ms=4, label=LAB[k])
    axs[1].plot([r['rung'] for r in rr], [max(r['wall_s'], 1) for r in rr], 'o-', ms=4, label=LAB[k])
axs[0].axhline(2.0, color='k', ls='--', lw=0.9)
axs[0].annotate('preregistered materiality bar (2 % rel)', (1.05, 2.2), fontsize=6.5)
axs[0].set_yscale('symlog', linthresh=1.0)
axs[0].set_ylabel('$M_{nd}$ improvement per rung [% rel]')
axs[1].set_yscale('log'); axs[1].set_ylabel('wall time per rung [s]')
for ax in axs:
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(['0.04', '0.02', '0.01', '0.005'])
    ax.set_xlabel('move level'); ax.legend()
fig.suptitle('F9  marginal benefit and marginal cost, rung by rung', y=1.02, fontsize=9)
save(fig, 'F9_marginal_per_rung.png')

# ---------------------------------------------------------------- F10
def rho_at(tag, k):
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        return np.array(h['RHO'][k - 1, :])

fig, axs = plt.subplots(3, 2, figsize=(9.6, 5.4))
for i, k in enumerate(KEYS):
    m = A['mesh'][k]; nx, ny = m['mesh']
    for j, (it, ttl) in enumerate([(m['S']['iteration'], 'S single-stage'),
                                   (m['F']['iteration'], 'F four-rung')]):
        axs[i][j].imshow(rho_at(m['tag'], it).reshape(nx, ny).T, cmap='gray_r',
                         vmin=0, vmax=1, aspect='equal')
        fld = m['S'] if j == 0 else m['F']
        axs[i][j].set_title(f"{LAB[k]}  {ttl} @ {it}   $M_{{nd}}$={fld['Mnd']:.2f} %  "
                            f"$\\omega_1$={fld['omega1']:.3f}", fontsize=7)
        axs[i][j].set_xticks([]); axs[i][j].set_yticks([]); axs[i][j].grid(False)
fig.suptitle('F10  final topology: single-stage S vs four-rung F', fontsize=9)
save(fig, 'F10_topology_S_vs_F.png', rect=[0, 0, 1, 0.94])

# ---------------------------------------------------------------- F11
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1))
for ax, k in zip(axs, KEYS):
    d = D[k]; T = d['T']; m = d['m']
    ax.plot(T['outer'], T['gap12'], color=CF, lw=1.1, label='relative gap $(\\omega_2-\\omega_1)/\\omega_1$')
    ax.axvline(d['kS'], color=CS, lw=1.2, ls='--')
    ax.axhline(d['P']['gap12'], color=CP, lw=1.0, ls='-', label='production')
    ax2 = ax.twinx()
    ax2.plot(T['outer'], T['multN'], color='#2ca02c', lw=1.0, ls=':')
    ax2.set_ylim(0, 3); ax2.set_yticks([0, 1, 2, 3]); ax2.grid(False)
    ax2.set_ylabel('subspace N', color='#2ca02c', fontsize=7)
    ax.set_title(f"{LAB[k]}   gap S={m['S']['gap12']:.4f}  F={m['F']['gap12']:.4f}   N=2 throughout")
    ax.set_xlabel('outer iteration')
axs[0].set_ylabel('relative gap'); axs[0].legend(loc='best')
fig.suptitle('F11  multiplicity and gap behaviour, S vs F (subspace size dotted green)', y=1.02, fontsize=9)
save(fig, 'F11_multiplicity.png')

# ---------------------------------------------------------------- F12
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.4))
dm = [-A['mesh'][k]['lower_rung_delta']['dMnd_rel_pct'] for k in KEYS]
do = [A['mesh'][k]['lower_rung_delta']['domega1_rel_pct'] for k in KEYS]
cw = [100 * A['mesh'][k]['lower_rung_delta']['frac_wall_after'] for k in KEYS]
axs[0].bar(x, dm, 0.5, color=[CF if v >= 2 else CP for v in dm])
axs[0].axhline(2.0, color='k', ls='--', lw=0.9)
axs[0].set_title('lower-rung $M_{nd}$ gain [% rel]\n(bar = 2 % materiality)')
axs[1].bar(x, do, 0.5, color=[CF if v >= 0.10 else CP for v in do])
axs[1].axhline(0.10, color='k', ls='--', lw=0.9)
axs[1].set_title('lower-rung $\\omega_1$ gain [% rel]\n(bar = 0.10 % materiality)')
axs[2].bar(x, cw, 0.5, color=CS)
axs[2].set_title('share of wall time spent below move = 0.04 [%]\n'
                 '(320x40 ends CAP_HIT after 1248 terminal-rung iterations)')
for i, ax in enumerate(axs):
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in KEYS])
    vals = [dm, do, cw][i]
    for xi, v in enumerate(vals):
        ax.annotate(f'{v:.2f}', (xi, v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=6.5)
fig.suptitle('F12  architecture trade-off: what the lower rungs buy, and what they cost', y=1.02, fontsize=9)
save(fig, 'F12_tradeoff.png')

json.dump({'figures': written}, open(os.path.join(STUDY, 'evidence', 'figures.json'), 'w'), indent=1)
print(f'\n{len(written)} figures written to {FIG}')
