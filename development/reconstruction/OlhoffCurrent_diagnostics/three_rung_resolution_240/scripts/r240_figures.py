#!/usr/bin/env python3
"""r240_figures -- the sixteen figures required by the brief (Phase 27)."""
import os, sys, json, hashlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
EV    = os.path.join(ROOT, 'evidence', 'three_rung_resolution_240')
FIG   = os.path.join(STUDY, 'figures'); os.makedirs(FIG, exist_ok=True)
TAG, NX, NY = 'C240x30', 240, 30

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})
C1, C2, C3, CF = '#d62728', '#2ca02c', '#9467bd', '#1f77b4'
RUNGC = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
A = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
M = json.load(open(os.path.join(STUDY, 'METRICS.json')))
BAR = A['thresholds']['omega1_rel_pct']
T = np.genfromtxt(os.path.join(STUDY, 'runs', f'{TAG}_iterations.csv'), delimiter=',', names=True)
n = A['nOuter']
STATES = (('S1', C1), ('S2', C2), ('S3', C3), ('F', CF))
written = []


def save(fig, name, tight=True):
    p = os.path.join(FIG, name)
    if tight: fig.tight_layout()
    fig.savefig(p); plt.close(fig)
    written.append(dict(file=f'figures/{name}',
                        sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()))
    print('  ', name)


def mark(ax, ylim=None):
    for nm, c in STATES:
        ax.axvline(A[nm]['iteration'], color=c, ls='--', lw=1.0, alpha=0.85)


# ---- F1 omega1 history ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, (lo, hi, ttl) in zip(axes, [(None, None, 'full history'),
                                    (166.9, 167.10, 'zoom: the four endpoints')]):
    ax.plot(T['outer'], T['omega1'], color='0.3', lw=0.8)
    for nm, c in STATES:
        s = A[nm]
        ax.plot(s['iteration'], s['omega1'], 'o', color=c, ms=6,
                label=f"{nm} @{s['iteration']} ({s.get('branch')})  {s['omega1']:.6f}")
    if lo: ax.set_ylim(lo, hi)
    ax.set_xlabel('outer iteration'); ax.set_ylabel(r'$\omega_1$')
    ax.set_title(f'C240×30  $\\omega_1$ — {ttl}'); ax.legend(loc='lower right')
fig.suptitle('F1  C240×30 objective history, S1/S2/S3/F marked', fontsize=10)
save(fig, 'F01_omega1_history.png')

# ---- F2 M_nd history -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, logy in ((axes[0], True), (axes[1], False)):
    ax.plot(T['outer'], T['Mnd'], color='0.3', lw=0.8)
    for nm, c in STATES:
        s = A[nm]
        ax.plot(s['iteration'], s['Mnd'], 'o', color=c, ms=6,
                label=f"{nm} @{s['iteration']}  {s['Mnd']:.5f}")
    if logy: ax.set_yscale('log'); ax.set_title('C240×30  $M_{nd}$ — full history')
    else: ax.set_ylim(12.85, 13.05); ax.set_title('C240×30  $M_{nd}$ — zoom')
    ax.set_xlabel('outer iteration'); ax.set_ylabel(r'$M_{nd}$ (%)'); ax.legend(loc='best')
fig.suptitle('F2  C240×30 $M_{nd}$ history, S1/S2/S3/F marked', fontsize=10)
save(fig, 'F02_Mnd_history.png')

# ---- F3 move / stage history --------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.step(T['outer'], T['move'], where='post', color='0.25', lw=1.3, label='move (four-rung)')
k3 = A['S3']['iteration']
ax.step([1, A['S1']['iteration'], A['S2']['iteration'], k3],
        [0.04, 0.02, 0.01, 0.01], where='post', color=C3, lw=2.4, alpha=0.55,
        label='move (three-rung counterfactual, terminates at S3)')
for nm, c in STATES:
    s = A[nm]
    ax.axvline(s['iteration'], color=c, ls='--', lw=1.0)
    ax.annotate(f"{nm} {s['iteration']}", (s['iteration'], 0.045), color=c, fontsize=7,
                rotation=90, va='bottom')
ax.set_yscale('log'); ax.set_yticks([0.005, 0.01, 0.02, 0.04])
ax.set_yticklabels(['0.005', '0.01', '0.02', '0.04'])
ax.set_xlabel('outer iteration'); ax.set_ylabel('move limit')
ax.set_title('F3  C240×30 move ladder — stage starts 1 / 207 / 246 / 285, CONVERGED @1358')
ax.legend(loc='lower left')
save(fig, 'F03_move_stage_history.png')

# ---- F4 A / B / E by stage ----------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(10, 6.4), sharex=False)
bounds = [(1, 206), (207, 245), (246, 284), (285, n)]
for ax, (a, b), st in zip(axes, bounds, A['declaration_timing']):
    sl = slice(a - 1, b)
    x = T['outer'][sl]
    ax.fill_between(x, 0, T['exE'][sl], step='mid', color='#c7e9c0', label='E = A OR B')
    ax.step(x, T['exA'][sl] * 0.95, where='mid', color='#d62728', lw=0.9, label='A')
    ax.step(x, T['exB'][sl] * 0.9, where='mid', color='#1f77b4', lw=0.9, label='B')
    ax.axvline(st['first_evaluable_iter'], color='#1f77b4', ls=':', lw=1.2)
    ax.axvline(st['earliest_possible_declaration'], color='#d62728', ls=':', lw=1.2)
    ax.axvline(st['offline_decl'], color='#2ca02c', lw=1.6)
    ax.set_ylim(-0.05, 1.15); ax.set_yticks([0, 1])
    ax.set_ylabel(f"r{st['stage']}\n{st['move']:g}", fontsize=8)
    ax.set_title(f"stage {st['stage']}  [{a}..{b}]   first evaluable {st['first_evaluable_iter']}"
                 f" · earliest possible {st['earliest_possible_declaration']}"
                 f" · declared {st['offline_decl']} (offset {st['declaration_offset']})"
                 f" · E true {100*st['E_true_fraction_after_first_evaluable']:.1f} % of the window",
                 fontsize=8)
    if ax is axes[0]: ax.legend(loc='center left', ncol=3)
axes[-1].set_xlabel('outer iteration')
fig.suptitle('F4  A / B / E by stage.  Blue dotted = first evaluable, red dotted = earliest '
             'possible declaration, green = actual declaration', fontsize=9.5)
save(fig, 'F04_ABE_by_stage.png')

# ---- F5 declaration timing by stage -------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.4))
for st in A['declaration_timing']:
    y = st['stage']
    ax.plot([st['stageStart'], st['stageEnd']], [y, y], color='0.85', lw=9, solid_capstyle='butt')
    ax.plot(st['first_evaluable_iter'], y, marker='|', color='#1f77b4', ms=16, mew=2.2)
    ax.plot(st['earliest_possible_declaration'], y, marker='|', color='#d62728', ms=16, mew=1.8)
    if st['first_E_true_iter']:
        ax.plot(st['first_E_true_iter'], y, marker='v', color='#ff7f0e', ms=7)
    ax.plot(st['offline_decl'], y, 'o', color='#2ca02c', ms=8)
    dx, ha = (8, 'left') if st['declaration_offset'] < 500 else (-8, 'right')
    ax.annotate(f"offset {st['declaration_offset']}", (st['offline_decl'], y),
                xytext=(dx, 9), textcoords='offset points', fontsize=8, ha=ha,
                color=('#2ca02c' if st['declaration_offset'] == 38 else '#d62728'),
                fontweight=('normal' if st['declaration_offset'] == 38 else 'bold'))
ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(['r1 0.04', 'r2 0.02', 'r3 0.01', 'r4 0.005'])
ax.invert_yaxis(); ax.set_xlabel('outer iteration')
h = [plt.Line2D([], [], marker='|', ls='none', color='#1f77b4', ms=12, mew=2, label='first evaluable (start+19)'),
     plt.Line2D([], [], marker='|', ls='none', color='#d62728', ms=12, mew=1.8, label='earliest possible (start+38)'),
     plt.Line2D([], [], marker='v', ls='none', color='#ff7f0e', ms=7, label='first E true'),
     plt.Line2D([], [], marker='o', ls='none', color='#2ca02c', ms=7, label='declaration')]
ax.legend(handles=h, loc='center right', fontsize=7)
ax.set_title('F5  Declaration timing.  Stages 2 and 3 declare at the arithmetic minimum (38); '
             'stage 4 takes 1073', fontsize=9.5)
save(fig, 'F05_declaration_timing.png')


def rungfig(field, ttl, fname, bar=None, sign=+1, fmt='{:+.5f}', ylab=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    v = [A['rungs'][f'rung{i}'][field] for i in (2, 3, 4)]
    cols = [RUNGC[i + 1] if (bar is None or sign * v[i] >= bar) else '0.75' for i in range(3)]
    x = np.arange(3)
    ax.bar(x, v, color=cols, width=0.6)
    if bar is not None:
        ax.axhline(sign * bar, color='#d62728', ls='--', lw=1.3,
                   label=f'materiality bar {sign*bar:g}')
        ax.legend()
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(['rung 2\n0.02', 'rung 3\n0.01', 'rung 4\n0.005'])
    for xi, vv in zip(x, v):
        ax.annotate(fmt.format(vv), (xi, vv), ha='center',
                    va='bottom' if vv >= 0 else 'top', fontsize=8)
    ax.set_ylabel(ylab or ttl); ax.set_title(ttl)
    save(fig, fname)


rungfig('domega1_rel_pct', r'F6  C240×30 rung-by-rung $\Delta\omega_1$ (% rel.)  '
        '— rung 1 omitted (+144 %)', 'F06_rung_omega1.png', bar=BAR, sign=+1,
        ylab=r'$\Delta\omega_1$ (% relative)')
rungfig('dMnd_rel_pct', r'F7  C240×30 rung-by-rung $\Delta M_{nd}$ (% rel., negative is better)',
        'F07_rung_Mnd.png', bar=2.0, sign=-1, fmt='{:+.4f}',
        ylab=r'$\Delta M_{nd}$ (% relative)')
rungfig('rho_mean_abs', r'F8  C240×30 rung-by-rung topology distance  mean $|\Delta\rho_e|$',
        'F08_rung_topology.png', bar=0.01, sign=+1, fmt='{:.6f}',
        ylab=r'mean $|\Delta\rho_e|$')

# ---- F9 / F10 cumulative work -------------------------------------------
for fld, ttl, fname in (('outer', 'cumulative outer iterations', 'F09_cumulative_outer.png'),
                        ('cumInner', 'cumulative inner MMA iterations', 'F10_cumulative_inner.png')):
    fig, ax = plt.subplots(figsize=(9, 3.3))
    y = T['outer'] if fld == 'outer' else T['cumInner']
    ax.plot(T['outer'], y, color='0.3', lw=1.1)
    for nm, c in STATES:
        s = A[nm]
        val = s['iteration'] if fld == 'outer' else s['innerCumulative']
        ax.plot(s['iteration'], val, 'o', color=c, ms=6,
                label=f"{nm} @{s['iteration']}  {val:,}")
    ax.set_xlabel('outer iteration'); ax.set_ylabel(ttl)
    c = A['cost']
    pct = c['saved_outer_pct'] if fld == 'outer' else c['saved_inner_pct']
    ax.set_title(f'{fname[:3]}  C240×30 {ttl} — terminating at S3 saves {pct:.1f} %')
    ax.legend(loc='upper left')
    save(fig, fname)

# ---- F11 S3 vs F topology ------------------------------------------------
with h5py.File(os.path.join(EV, f'{TAG}_trajectory.mat'), 'r') as h:
    RHO = np.array(h['RHO']).T
s3 = RHO[:, A['S3']['iteration'] - 1].reshape(NY, NX, order='F')
fF = RHO[:, A['F']['iteration'] - 1].reshape(NY, NX, order='F')
fig, axes = plt.subplots(3, 1, figsize=(9, 5.2))
for ax, im, ttl in ((axes[0], s3, f"S3 @{A['S3']['iteration']}  (move 0.01)"),
                    (axes[1], fF, f"F @{A['F']['iteration']}  (move 0.005, {A['status']})"),
                    (axes[2], fF - s3, 'F − S3')):
    if ttl == 'F − S3':
        hh = ax.imshow(im, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    else:
        hh = ax.imshow(1 - im, cmap='gray', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(hh, ax=ax, fraction=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(ttl, fontsize=9)
r4 = A['rungs']['rung4']
axes[2].set_xlabel(f"mean|Δρ|={r4['rho_mean_abs']:.6f}  max|Δρ|={r4['rho_max_abs']:.5f}  "
                   f"(materiality bar 0.01)", fontsize=8)
fig.suptitle('F11  C240×30  S3 versus F topology — what the final move = 0.005 rung changes',
             fontsize=10)
save(fig, 'F11_topology_S3_vs_F.png')

# ---- F12 gap / multiplicity ---------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.3))
ax.plot(T['outer'], T['gap12'], color='0.3', lw=0.9, label=r'$(\omega_2-\omega_1)/\omega_1$')
for nm, c in STATES:
    s = A[nm]
    ax.plot(s['iteration'], s['gap12'], 'o', color=c, ms=6,
            label=f"{nm}  gap {s['gap12']:.6f}")
ax2 = ax.twinx(); ax2.plot(T['outer'], T['multN'], color='#ff7f0e', lw=0.9, alpha=0.75)
ax2.set_ylim(0, 3); ax2.set_ylabel('subspace size N', color='#ff7f0e'); ax2.grid(False)
ax.set_xlabel('outer iteration'); ax.set_ylabel('relative gap')
ax.set_title('F12  C240×30 gap and multiplicity — N = 2 throughout, no mode crossing')
ax.legend(loc='lower right')
save(fig, 'F12_gap_multiplicity.png')

# ---- F13 / F14 / F15 cross-mesh rung-4 -----------------------------------
order = ['m160', 'm240', 'm320', 'm400']
lab = ['160×20', '240×30', '320×40', '400×50']
X = M['cross_mesh']
for fld, ttl, fname, bar, sign, fmt in (
        ('rung4_omega1_rel_pct', r'F13  Cross-mesh rung-4 $\Delta\omega_1$ (% relative)',
         'F13_crossmesh_rung4_omega1.png', BAR, +1, '{:+.5f}'),
        ('rung4_Mnd_rel_pct', r'F14  Cross-mesh rung-4 $\Delta M_{nd}$ (% rel., negative better)',
         'F14_crossmesh_rung4_Mnd.png', 2.0, -1, '{:+.4f}'),
        ('rung4_pct_total_inner', 'F15  Cross-mesh rung-4 share of total inner MMA work (%)',
         'F15_crossmesh_rung4_cost.png', None, +1, '{:.1f}')):
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    v = [X[k][fld] for k in order]
    cols = ['#c44e52' if k == 'm240' else RUNGC[0] for k in order]
    ax.bar(np.arange(4), v, color=cols, width=0.6)
    if bar is not None:
        ax.axhline(sign * bar, color='#d62728', ls='--', lw=1.3,
                   label=f'materiality bar {sign*bar:g}')
        ax.legend()
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab)
    for xi, vv in zip(np.arange(4), v):
        ax.annotate(fmt.format(vv), (xi, vv), ha='center',
                    va='bottom' if vv >= 0 else 'top', fontsize=8)
    for xi, k in enumerate(order):
        ax.annotate(X[k]['F_status'], (xi, 0), xytext=(0, -26), textcoords='offset points',
                    ha='center', fontsize=6.5, color='0.35')
    ax.set_title(ttl + '   (red = the new run; no scaling law is fitted)', fontsize=9)
    save(fig, fname)

# ---- F16 threshold-splitting resolution ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
tr3 = json.load(open(os.path.join(ROOT, 'diagnostics', 'three_rung_architecture', 'METRICS.json')))
sp = tr3['threshold_splitting']
vals = [sp['rung3_omega1_pct'], sp['rung4_omega1_pct'], sp['combined_S2_to_F_omega1_pct']]
ax.bar([0, 1, 2.4], vals, color=['#55a868', '#c44e52', '0.45'], width=0.6)
ax.axhline(BAR, color='#d62728', ls='--', lw=1.4, label=f'bar {BAR}%')
ax.set_xticks([0, 1, 2.4]); ax.set_xticklabels(['rung 3', 'rung 4', 'r3+r4\ncombined'])
for xi, vv in zip([0, 1, 2.4], vals):
    ax.annotate(f'{vv:+.5f}', (xi, vv), ha='center', va='bottom', fontsize=8)
ax.set_ylabel(r'$\Delta\omega_1$ (% rel.)'); ax.legend(fontsize=7)
ax.set_title('THE CONCERN (160×20, prior study)\ncombined above the bar, each half below it',
             fontsize=9)

ax = axes[1]
v = [X[k]['rung4_omega1_rel_pct'] for k in order]
ax.bar(np.arange(4), v, color=['#c44e52' if k == 'm240' else RUNGC[0] for k in order], width=0.6)
ax.axhline(BAR, color='#d62728', ls='--', lw=1.4, label=f'bar {BAR}%')
tb = M['tail_analysis']['best_omega1_rel_pct']
ax.plot([1], [tb], marker='D', color='k', ms=7, ls='none',
        label=f'240×30 running best over the whole tail {tb:+.5f}%')
ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab)
for xi, vv in zip(np.arange(4), v):
    ax.annotate(f'{vv:+.5f}', (xi, vv), ha='center', va='bottom', fontsize=8)
ax.set_ylabel(r'rung-4 $\Delta\omega_1$ (% rel.)'); ax.legend(fontsize=6.5, loc='upper left')
ax.set_title('THE RESOLUTION — rung 4 measured directly on four meshes\n'
             'every value far below the bar, including the new 240×30 run', fontsize=9)
fig.suptitle('F16  Threshold-splitting: concern and resolution   →   '
             'THRESHOLD_SPLITTING_CONCERN_RESOLVED', fontsize=10)
save(fig, 'F16_threshold_splitting_resolution.png')

json.dump(written, open(os.path.join(STUDY, 'evidence', 'figures.json'), 'w'), indent=1)
print(f'\n{len(written)} figures written')
