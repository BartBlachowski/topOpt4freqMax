#!/usr/bin/env python3
"""cv_figures -- the fourteen figures required by the brief (Phase 26)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
REPO = os.path.dirname(os.path.dirname(ROOT))
FIG = os.path.join(STUDY, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})

CP, CC = '#8c8c8c', '#1f77b4'          # production, candidate
CA, CB = '#d62728', '#2ca02c'          # branch A, branch B
written = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    written.append(name); print('  ', name)


def csv(path):
    a = np.genfromtxt(path, delimiter=',', names=True)
    return a


MESHES = [('160x20', 'm160', 'C160x20',
           os.path.join(ROOT, 'diagnostics', 'move_stop', 'runs', 'baseline_160x20_iterations.csv')),
          ('320x40', 'm320', 'C320x40',
           os.path.join(ROOT, 'diagnostics', 'move_stop', 'runs', 'baseline_320x40_iterations.csv')),
          ('400x50', 'm400', 'C400x50',
           os.path.join(ROOT, 'diagnostics', 'move_activity_400', 'runs', 'P400_400x50_iterations.csv'))]

A = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
B = json.load(open(os.path.join(STUDY, 'evidence', 'baselines.json')))

D = {}
for label, key, tag, pcsv in MESHES:
    cpath = os.path.join(STUDY, 'runs', tag + '_iterations.csv')
    if not os.path.isfile(cpath):
        continue
    rec = json.load(open(os.path.join(STUDY, 'runs', tag + '_record.json')))
    D[key] = dict(label=label, tag=tag, cand=csv(cpath), prod=csv(pcsv), rec=rec,
                  base=B[key], an=A['mesh'].get(key))

KEYS = [k for k in ('m160', 'm320', 'm400') if k in D]


def prodMnd(d):
    return d['prod']['Mnd_pct']


def descents(d):
    dd = np.atleast_2d(np.array(d['rec']['descents'], dtype=float))
    return dd if dd.size else np.zeros((0, 4))


def branches(d):
    b = d['rec']['eventBranch']
    return [b] if isinstance(b, str) else list(b)


# ---------------------------------------------------------------- F1-F3
for n, k in enumerate(KEYS, start=1):
    d = D[k]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(d['prod']['outer'], prodMnd(d), color=CP, lw=1.4, label='production (beta ladder)')
    ax.plot(d['cand']['outer'], d['cand']['Mnd'], color=CC, lw=1.4, label='candidate (A OR B)')
    ax.axhline(d['base']['Mnd'], color=CP, ls=':', lw=0.9)
    ax.axhline(d['rec']['Mnd_final'], color=CC, ls=':', lw=0.9)
    for j, row in enumerate(descents(d)):
        ax.axvline(row[0], color=CA if branches(d)[j] == 'A' else CB, lw=0.8, ls='--')
    ax.axvline(d['base']['firstDescentIter'], color=CP, lw=0.8, ls='--')
    ax.set_xlabel('outer iteration'); ax.set_ylabel('$M_{nd}$  [%]')
    ax.set_title(f"F{n}  {d['label']}: $M_{{nd}}$ history, production vs candidate\n"
                 f"final {d['base']['Mnd']:.3f} -> {d['rec']['Mnd_final']:.3f} %"
                 f"  ({d['an']['delta']['Mnd_rel_pct']:+.2f} %)")
    ax.legend(loc='upper right')
    save(fig, f'F{n}_Mnd_{d["label"]}.png')

# ---------------------------------------------------------------- F4
fig, axs = plt.subplots(1, len(KEYS), figsize=(3.1 * len(KEYS), 3.0), squeeze=False)
for i, k in enumerate(KEYS):
    d = D[k]; ax = axs[0][i]
    ax.plot(d['prod']['outer'], d['prod']['omega1'], color=CP, lw=1.3, label='production')
    ax.plot(d['cand']['outer'], d['cand']['omega1'], color=CC, lw=1.3, label='candidate')
    for j, row in enumerate(descents(d)):
        ax.axvline(row[0], color=CA if branches(d)[j] == 'A' else CB, lw=0.7, ls='--')
    ax.set_title(f"{d['label']}   $\\omega_1$: {d['base']['omega1']:.3f} -> "
                 f"{d['rec']['omega1']:.3f}\n({d['an']['delta']['omega1_rel_pct']:+.3f} %)")
    ax.set_xlabel('outer iteration')
    if i == 0:
        ax.set_ylabel('$\\omega_1$'); ax.legend(loc='lower right')
fig.suptitle('F4  first eigenfrequency, production vs candidate', y=1.02, fontsize=9)
save(fig, 'F4_omega1_all.png')

# ---------------------------------------------------------------- F5
fig, axs = plt.subplots(len(KEYS), 1, figsize=(6.4, 2.3 * len(KEYS)), squeeze=False)
for i, k in enumerate(KEYS):
    d = D[k]; ax = axs[i][0]
    ax.step(d['cand']['outer'], d['cand']['move'], where='post', color=CC, lw=1.4,
            label='candidate move')
    ax.step(d['prod']['outer'], d['prod']['move'], where='post', color=CP, lw=1.2,
            label='production move')
    ax.step(d['cand']['outer'], d['cand']['prodMoveShadow'], where='post', color=CP,
            lw=0.9, ls=':', label="production rule replayed on candidate path")
    for j, row in enumerate(descents(d)):
        br = branches(d)[j]
        ax.axvline(row[0], color=CA if br == 'A' else CB, lw=0.9, ls='--')
        ax.annotate(f'{br}@{int(row[0])}\nwin {int(row[3])}-{int(row[2])}',
                    xy=(row[0], 0.04), xytext=(4, -2), textcoords='offset points',
                    fontsize=6, color=CA if br == 'A' else CB, va='top')
    t = d['an']['term']
    ax.axvline(d['rec']['nOuter'], color='k', lw=0.9)
    ax.annotate(f"{d['rec']['status']} @ {d['rec']['nOuter']}\nterminal {t['branch']}",
                xy=(d['rec']['nOuter'], 0.02), xytext=(-60, 0), textcoords='offset points',
                fontsize=6)
    ax.set_yscale('log'); ax.set_yticks([0.005, 0.01, 0.02, 0.04])
    ax.set_yticklabels(['0.005', '0.01', '0.02', '0.04'])
    ax.set_ylabel('move'); ax.set_title(f"{d['label']}")
    if i == 0:
        ax.legend(loc='upper right')
axs[-1][0].set_xlabel('outer iteration')
fig.suptitle('F5  candidate move ladder with A/B transition events', y=1.005, fontsize=9)
save(fig, 'F5_move_events.png')

# ---------------------------------------------------------------- F6
fig, axs = plt.subplots(len(KEYS), 1, figsize=(6.4, 2.2 * len(KEYS)), squeeze=False)
for i, k in enumerate(KEYS):
    d = D[k]; ax = axs[i][0]; c = d['cand']
    ax.plot(c['outer'], c['betaStallRel'], color='#9467bd', lw=1.0, label=r'$\beta$ stall statistic')
    ax.axhline(5e-3, color='#9467bd', ls=':', lw=0.9, label=r'$\beta$ stall threshold 5e-3')
    fires = c['outer'][c['betaStallFires'] > 0]
    if fires.size:
        ax.plot(fires, np.full_like(fires, 5e-3), '.', ms=2, color='#9467bd')
    for j, row in enumerate(descents(d)):
        br = branches(d)[j]
        ax.axvline(row[0], color=CA if br == 'A' else CB, lw=1.0, ls='--')
    ax.axvline(d['rec']['betaStallFirst'], color='#9467bd', lw=1.2)
    ax.annotate(f"first $\\beta$ stall {int(d['rec']['betaStallFirst'])}\n"
                f"(production descended here)",
                xy=(d['rec']['betaStallFirst'], 0), xytext=(6, 14),
                textcoords='offset points', fontsize=6, color='#9467bd')
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.set_ylabel('rel. beta progress'); ax.set_title(d['label'])
    if i == 0:
        ax.legend(loc='upper right')
axs[-1][0].set_xlabel('outer iteration')
fig.suptitle(r'F6  $\beta$-stall events vs candidate exhaustion events', y=1.005, fontsize=9)
save(fig, 'F6_beta_vs_exhaustion.png')

# ---------------------------------------------------------------- F7
rows = sum(len(descents(D[k])) + 1 for k in KEYS)
fig, axs = plt.subplots(rows, 1, figsize=(6.4, 1.9 * rows), squeeze=False)
r = 0
for k in KEYS:
    d = D[k]; c = d['cand']
    evs = [(int(row[0]), branches(d)[j]) for j, row in enumerate(descents(d))]
    evs.append((int(d['rec']['nOuter']), d['rec']['terminalBranch'] or '-'))
    for it, br in evs:
        ax = axs[r][0]; r += 1
        lo, hi = max(1, it - 60), min(len(c['outer']), it + 40)
        sl = slice(lo - 1, hi)
        ax.plot(c['outer'][sl], c['exMedcos'][sl], color='#1f77b4', lw=1.1, label=r'med$_{20}\cos\theta$')
        ax.plot(c['outer'][sl], c['exMednet'][sl], color='#ff7f0e', lw=1.1, label='med$_{20}$ net/path')
        ax.plot(c['outer'][sl], c['exAmp'][sl] / c['exTol'][sl], color='#2ca02c', lw=1.1,
                label=r'$\|\Delta\rho\|_2/\mathrm{tol}$')
        ax.axhline(0.0, color='#1f77b4', ls=':', lw=0.7)
        ax.axhline(0.5, color='#ff7f0e', ls=':', lw=0.7)
        ax.axhline(1.0, color='#2ca02c', ls=':', lw=0.7)
        ax.axvline(it, color=CA if br == 'A' else CB, lw=1.2, ls='--')
        ax.set_title(f"{d['label']}  event at {it}  (branch {br})")
        ax.set_ylim(-1.4, 2.6)
        if r == 1:
            ax.legend(loc='upper left', ncol=3)
axs[-1][0].set_xlabel('outer iteration')
fig.suptitle(r'F7  $\cos\theta$, net/path and amplitude around every candidate event',
             y=1.003, fontsize=9)
save(fig, 'F7_signals_at_events.png')

# ---------------------------------------------------------------- F8
def loadrho(path):
    with h5py.File(path, 'r') as f:
        R = f['RHO']
        return np.array(R[-1, :])            # MATLAB v7.3 stores transposed


fig, axs = plt.subplots(len(KEYS), 2, figsize=(9.0, 1.5 * len(KEYS)), squeeze=False)
for i, k in enumerate(KEYS):
    d = D[k]
    nelx, nely = [int(x) for x in d['label'].split('x')]
    cpath = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation',
                         d['tag'] + '_trajectory.mat')
    b = d['base']
    if b.get('rho_recovered'):
        ppath = os.path.join(REPO, b['rho_recovered_file'])
    elif b.get('rho_available'):
        ppath = os.path.join(REPO, b['trajectory'])
    else:
        ppath = None
    if os.path.isfile(cpath):
        axs[i][1].imshow(loadrho(cpath).reshape(nelx, nely).T, cmap='gray_r',
                         vmin=0, vmax=1, aspect='equal')
        axs[i][1].set_title(f"candidate {d['label']}  $M_{{nd}}$={d['rec']['Mnd_final']:.2f}%", fontsize=7)
    else:
        axs[i][1].text(0.5, 0.5, 'candidate density field\nUNAVAILABLE\n(raw .mat lost)',
                       ha='center', va='center', fontsize=7, transform=axs[i][1].transAxes)
        axs[i][1].set_title(f"candidate {d['label']}  $M_{{nd}}$={d['rec']['Mnd_final']:.2f}%", fontsize=7)
    if ppath and os.path.isfile(ppath):
        axs[i][0].imshow(loadrho(ppath).reshape(nelx, nely).T, cmap='gray_r',
                         vmin=0, vmax=1, aspect='equal')
        axs[i][0].set_title(f"production {d['label']}  $M_{{nd}}$={d['base']['Mnd']:.2f}%", fontsize=7)
    else:
        axs[i][0].text(0.5, 0.5, 'production density field\nUNAVAILABLE\n(raw .mat lost)',
                       ha='center', va='center', fontsize=7, transform=axs[i][0].transAxes)
        axs[i][0].set_title(f"production {d['label']}", fontsize=7)
    for a in axs[i]:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.suptitle('F8  final topology, production vs candidate', y=1.02, fontsize=9)
save(fig, 'F8_topology.png')

# ---------------------------------------------------------------- F9-F12
labels = [D[k]['label'] for k in KEYS]
x = np.arange(len(KEYS)); w = 0.36


def bars(vals_p, vals_c, ylab, title, name, fmt='{:.3f}', logy=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - w / 2, vals_p, w, color=CP, label='production')
    ax.bar(x + w / 2, vals_c, w, color=CC, label='candidate')
    for xi, (p, c) in enumerate(zip(vals_p, vals_c)):
        ax.annotate(fmt.format(p), (xi - w / 2, p), ha='center', va='bottom', fontsize=6)
        ax.annotate(fmt.format(c), (xi + w / 2, c), ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylab)
    if logy:
        ax.set_yscale('log')
    ax.set_title(title); ax.legend()
    save(fig, name)


bars([D[k]['base']['Mnd'] for k in KEYS], [D[k]['rec']['Mnd_final'] for k in KEYS],
     '$M_{nd}$ [%]', 'F9  final $M_{nd}$', 'F9_final_Mnd.png')
bars([D[k]['base']['omega1'] for k in KEYS], [D[k]['rec']['omega1'] for k in KEYS],
     '$\\omega_1$', 'F10  final $\\omega_1$', 'F10_final_omega1.png')

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, (pv, cv, t) in zip(axs, [
        ([D[k]['base']['nOuter'] for k in KEYS], [D[k]['rec']['nOuter'] for k in KEYS], 'outer iterations'),
        ([D[k]['base']['innerTotal'] for k in KEYS], [D[k]['rec']['innerTotal'] for k in KEYS], 'inner MMA iterations'),
        ([D[k]['base']['wall_s'] for k in KEYS], [D[k]['rec']['wall_s'] for k in KEYS], 'wall time [s]')]):
    ax.bar(x - w / 2, pv, w, color=CP, label='production')
    ax.bar(x + w / 2, cv, w, color=CC, label='candidate')
    for xi, (p, c) in enumerate(zip(pv, cv)):
        ax.annotate(f'x{c / p:.2f}', (xi + w / 2, c), ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(t)
axs[0].legend()
fig.suptitle('F11  computational cost, and the multiplier the improvement costs', y=1.02, fontsize=9)
save(fig, 'F11_cost.png')

fig, ax = plt.subplots(figsize=(6.0, 3.2))
for i, k in enumerate(KEYS):
    d = D[k]
    ax.plot([i - w], [d['base']['firstDescentIter']], 'o', color=CP, ms=7)
    for j, row in enumerate(descents(d)):
        ax.plot([i + w], [row[0]], 'o', color=CA if branches(d)[j] == 'A' else CB, ms=7)
        ax.annotate(f"{branches(d)[j]}@{int(row[0])}", (i + w, row[0]), xytext=(6, -2),
                    textcoords='offset points', fontsize=6)
    ax.plot([i + w], [d['rec']['nOuter']], 's', color='k', ms=6)
    ax.annotate(f"stop {d['rec']['nOuter']}", (i + w, d['rec']['nOuter']), xytext=(6, -2),
                textcoords='offset points', fontsize=6)
    ax.annotate(f"prod {int(d['base']['firstDescentIter'])}", (i - w, d['base']['firstDescentIter']),
                xytext=(-52, -2), textcoords='offset points', fontsize=6, color=CP)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('outer iteration')
ax.set_title('F12  move-transition schedule: production (grey) vs candidate (A red / B green)')
save(fig, 'F12_transitions.png')

# ---------------------------------------------------------------- F13
fig, axs = plt.subplots(len(KEYS), 1, figsize=(6.4, 2.2 * len(KEYS)), squeeze=False)
for i, k in enumerate(KEYS):
    d = D[k]; c = d['cand']; ax = axs[i][0]
    n = int(d['rec']['nOuter'])
    lo = max(1, n - 120)
    sl = slice(lo - 1, n)
    ax.plot(c['outer'][sl], c['exNA'][sl], color=CA, lw=1.2, label='persistence counter A')
    ax.plot(c['outer'][sl], c['exNB'][sl], color=CB, lw=1.2, label='persistence counter B')
    ax.axhline(20, color='k', ls=':', lw=0.9, label='P = 20')
    ax.axvline(n, color='k', lw=1.0)
    t = d['an']['term']
    ax.set_title(f"{d['label']}  {d['rec']['status']} at {n}: move={t['move']}, stage={t['stage']}, "
                 f"branch {t['branch']}, window {t['declBegin']}-{t['declIter']}", fontsize=8)
    ax.set_ylabel('consecutive iterations')
    if i == 0:
        ax.legend(loc='upper left')
axs[-1][0].set_xlabel('outer iteration')
fig.suptitle('F13  terminal-admission evidence: the frozen persistence at move = 0.005',
             y=1.005, fontsize=9)
save(fig, 'F13_terminal_admission.png')

# ---------------------------------------------------------------- F14
fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2))
dm = [D[k]['an']['delta']['Mnd_rel_pct'] for k in KEYS]
do = [D[k]['an']['delta']['omega1_rel_pct'] for k in KEYS]
dc = [D[k]['an']['delta']['outer_mult'] for k in KEYS]
delay = [D[k]['an']['delta']['descentDelay'] for k in KEYS]
axs[0].bar(x, dm, 0.5, color=[CB if v < 0 else CA for v in dm])
axs[0].axhline(-20, color='k', ls='--', lw=0.9)
axs[0].annotate('preregistered fine-mesh bar (-20 %)', (0.5, -20), fontsize=6,
                va='bottom', ha='center')
axs[0].set_title('relative $M_{nd}$ change [%]\n(negative = better)')
axs[1].bar(x, do, 0.5, color=[CB if v > 0 else CA for v in do])
axs[1].axhline(-1.0, color='k', ls='--', lw=0.9)
axs[1].set_title('relative $\\omega_1$ change [%]\n(positive = better; bound -1 %)')
axs[2].bar(x, dc, 0.5, color=CC)
axs[2].axhline(8, color='k', ls='--', lw=0.9)
axs[2].set_title('outer-iteration multiplier\n(preregistered bound 8x)')
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(labels)
# Annotate INSIDE the axes, just under the top of each bar, so the label can
# never collide with the x tick labels for a deep bar (400x50 reaches -52 %).
for i, v in enumerate(delay):
    axs[0].annotate(f'held move=0.04\n{int(v)} iters longer', (i, dm[i]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6)
axs[0].margins(y=0.18)
fig.suptitle('F14  causal effect of replacing beta-stall continuation with the frozen A OR B rule',
             y=1.02, fontsize=9)
save(fig, 'F14_summary.png')

json.dump({'figures': written}, open(os.path.join(STUDY, 'evidence', 'figures.json'), 'w'), indent=1)
print(f'\n{len(written)} figures written to {FIG}')
