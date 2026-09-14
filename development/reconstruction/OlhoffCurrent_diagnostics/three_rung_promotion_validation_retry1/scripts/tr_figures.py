#!/usr/bin/env python3
"""tr_figures -- figures for the three-rung promotion validation retry.

Every figure compares the ONE three-rung candidate run against the frozen
four-rung C320 oracle.  The oracle CSV is read from the copy extracted from git
HEAD, never from the working tree, so the pre-existing tOuter drift cannot enter
any plot.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
FIG = os.path.join(STUDY, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})

C4, C3 = '#8c8c8c', '#1f77b4'          # four-rung oracle, three-rung candidate
CA, CB = '#d62728', '#2ca02c'          # branch A, branch B
S1, S2, S3 = 274, 313, 352
written = []


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close(fig)
    written.append(name)
    print('  ', name)


new = np.genfromtxt(os.path.join(STUDY, 'runs', 'C320x40_three_rung_iterations.csv'),
                    delimiter=',', names=True)
old = np.genfromtxt(os.path.join(STUDY, 'evidence', 'oracle_C320x40_iterations_HEAD.csv'),
                    delimiter=',', names=True)
pre = json.load(open(os.path.join(STUDY, 'evidence', 'prefix_equivalence.json')))

n3, n4 = len(new['outer']), len(old['outer'])


def events(ax, upto=None):
    # S1/S2/S3 sit within 78 iterations of each other on an axis up to 1600, so
    # the labels are staggered vertically -- otherwise they overprint.
    for i, (k, c, lab) in enumerate(((S1, CA, 'S1 274 (A)'),
                                     (S2, CB, 'S2 313 (B)'),
                                     (S3, CB, 'S3 352 (B)'))):
        if upto is None or k <= upto:
            ax.axvline(k, color=c, lw=0.8, ls='--', alpha=0.8)
            ax.annotate(lab, (k, ax.get_ylim()[1]), xytext=(3, -8 - 9 * i),
                        textcoords='offset points', fontsize=6, color=c, va='top')


# --- F1  the move ladder, both arms ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.step(old['outer'], old['move'], where='post', color=C4, lw=1.4,
        label=f'four-rung oracle  (CAP_HIT @{n4})')
ax.step(new['outer'], new['move'], where='post', color=C3, lw=2.0,
        label=f'three-rung candidate  (CONVERGED @{n3})')
ax.axhline(0.005, color='#d62728', lw=0.7, ls=':', label='0.005 — removed rung')
ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel('move limit')
ax.set_title('Move ladder: the candidate stops where the oracle descended to 0.005')
ax.legend(loc='upper right')
events(ax)
save(fig, 'F1_move_ladder.png')

# --- F2  prefix equivalence of the controller trace ------------------------
fig, axs = plt.subplots(3, 1, figsize=(7, 5.4), sharex=True)
for ax, key, lab in ((axs[0], 'exAmp', r'$\|\Delta\rho\|_2$  (amp)'),
                     (axs[1], 'exMedcos', r'med$_{20}\cos$'),
                     (axs[2], 'exNA', 'persistence counters')):
    if key == 'exNA':
        ax.plot(old['outer'][:S3], old['exNA'][:S3], color=C4, lw=1.6)
        ax.plot(old['outer'][:S3], old['exNB'][:S3], color=C4, lw=1.6, ls=':')
        ax.plot(new['outer'][:S3], new['exNA'][:S3], color=CA, lw=0.9, label='nA')
        ax.plot(new['outer'][:S3], new['exNB'][:S3], color=CB, lw=0.9, label='nB')
        ax.axhline(20, color='k', lw=0.6, ls='--')
        ax.legend(loc='upper left')
    else:
        ax.plot(old['outer'][:S3], old[key][:S3], color=C4, lw=1.8,
                label='four-rung oracle')
        ax.plot(new['outer'][:S3], new[key][:S3], color=C3, lw=0.9,
                label='three-rung candidate')
        ax.legend(loc='best')
    ax.set_ylabel(lab)
    events(ax, S3)
axs[0].set_yscale('log')
axs[0].axhline(0.1, color='k', lw=0.6, ls='--')
axs[0].set_title('Controller trace, iterations 1–352: the two arms are bitwise identical')
axs[1].axhline(0, color='k', lw=0.6)
axs[-1].set_xlabel('outer iteration')
save(fig, 'F2_controller_prefix.png')

# --- F3  scientific state over the prefix ---------------------------------
fig, axs = plt.subplots(3, 1, figsize=(7, 5.4), sharex=True)
for ax, key, lab in ((axs[0], 'omega1', r'$\omega_1$'),
                     (axs[1], 'Mnd', r'$M_{nd}$  [%]'),
                     (axs[2], 'volume', 'volume fraction')):
    ax.plot(old['outer'], old[key], color=C4, lw=1.6, label='four-rung oracle (full 1600)')
    ax.plot(new['outer'], new[key], color=C3, lw=1.0, label='three-rung candidate')
    ax.set_ylabel(lab); ax.legend(loc='best')
    ax.axvline(S3, color=CB, lw=0.8, ls='--')
axs[0].set_title(r'Scientific state: identical to 352, then the oracle continues on rung 4')
axs[-1].set_xlabel('outer iteration')
save(fig, 'F3_scientific_state.png')

# --- F4  what the removed rung bought ------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
ax = axs[0]
ax.plot(old['outer'], old['cumInner'], color=C4, lw=1.6, label='four-rung oracle')
ax.plot(new['outer'], new['cumInner'], color=C3, lw=1.6, label='three-rung candidate')
ax.axvline(S3, color=CB, lw=0.8, ls='--')
ax.set_xlabel('outer iteration'); ax.set_ylabel('cumulative inner MMA')
ax.set_title('Inner work'); ax.legend(loc='upper left')
ax = axs[1]
c = pre['cost']
bars = ax.bar(['outer\niterations', 'inner MMA\niterations'],
              [100 * c['outerThree'] / c['outerFour'], 100 * c['innerThree'] / c['innerFour']],
              color=C3, width=0.55)
ax.bar(['outer\niterations', 'inner MMA\niterations'], [100, 100],
       color=C4, width=0.55, zorder=0, alpha=0.45)
for b, v in zip(bars, [c['outerSavedPct'], c['innerSavedPct']]):
    ax.annotate(f'−{v:.2f}%', (b.get_x() + b.get_width() / 2, b.get_height()),
                ha='center', va='bottom', fontsize=8, color=C3)
ax.set_ylim(0, 118); ax.set_ylabel('% of the four-rung cost')
ax.set_title('Cost eliminated by dropping rung 4')
save(fig, 'F4_cost.png')

# --- F5  the S3 decision, magnified ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 2.8))
w = slice(320, min(n4, 400))
ax.step(old['outer'][w], old['move'][w], where='post', color=C4, lw=2.2,
        label='four-rung: descends to 0.005 at 353')
ax.step(new['outer'][330:n3], new['move'][330:n3], where='post', color=C3, lw=2.2,
        label='three-rung: CONVERGED at 352')
ax.plot([n3], [new['move'][n3 - 1]], 'o', color=C3, ms=6)
ax.axvline(333, color='#999999', lw=0.7, ls=':')
ax.axvline(S3, color=CB, lw=1.0, ls='--')
ax.annotate('declBegin 333', (333, 0.0125), fontsize=6, rotation=90, va='bottom')
ax.annotate('declIter 352\n(Branch B, P=20)', (S3, 0.0128), fontsize=6,
            va='bottom', ha='right')
ax.set_yscale('log'); ax.set_ylim(0.004, 0.03)
ax.set_xlabel('outer iteration'); ax.set_ylabel('move limit')
ax.set_title('The S3 decision: the single point at which the two policies differ')
ax.legend(loc='lower left')
save(fig, 'F5_S3_decision.png')

json.dump({'figures': written}, open(os.path.join(STUDY, 'evidence', 'figures.json'), 'w'),
          indent=2)
print(f'\n{len(written)} figures written to {FIG}')
