#!/usr/bin/env python3
"""tr3_figures -- the fourteen figures required by the brief (Phase 25)."""
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
CP, C1, C2, C3, CF = '#8c8c8c', '#d62728', '#2ca02c', '#9467bd', '#1f77b4'
RUNGC = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']         # rungs 1..4
MESH = [('m160', 'C160x20', 160, 20), ('m320', 'C320x40', 320, 40), ('m400', 'C400x50', 400, 50)]
A = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
M = json.load(open(os.path.join(STUDY, 'METRICS.json')))
BAR = A['thresholds']['omega1_rel_pct']
written = []


def save(fig, name, tight=True):
    p = os.path.join(FIG, name)
    if tight:
        fig.tight_layout()
    fig.savefig(p); plt.close(fig)
    written.append(dict(file=f'figures/{name}',
                        sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()))
    print('  ', name)


def csv(tag):
    return np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                         delimiter=',', names=True)


STATES = (('S1', C1), ('S2', C2), ('S3', C3), ('F', CF))

# ---- F1  omega1, P/S1/S2/S3/F, per mesh ---------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['omega1'], color='0.3', lw=0.9, label=r'candidate $\omega_1$')
    ax.axhline(m['production']['omega1'], color=CP, lw=1.4,
               label=f"P {m['production']['omega1']:.4f}")
    for nm, c in STATES:
        s = m[nm]
        ax.plot(s['iteration'], s['omega1'], 'o', color=c, ms=5,
                label=f"{nm} @{s['iteration']}  {s['omega1']:.4f}")
    lo = min(m['production']['omega1'], m['S1']['omega1']) - 1.0
    hi = max(m['production']['omega1'], m['F']['omega1']) + 0.8
    ax.set_ylim(lo, hi)
    ax.set_xlabel('outer iteration'); ax.set_ylabel(r'$\omega_1$')
    ax.set_title(f'{nx}×{ny}   $\\omega_1$:  P / S1 / S2 / S3 / F')
    ax.legend(loc='lower right')
save(fig, 'F1_omega1_P_S1_S2_S3_F.png')

# ---- F2  M_nd ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['Mnd'], color='0.3', lw=0.9, label='candidate trajectory')
    ax.axhline(m['production']['Mnd'], color=CP, lw=1.4, label=f"P {m['production']['Mnd']:.3f}")
    for nm, c in STATES:
        s = m[nm]
        ax.plot(s['iteration'], s['Mnd'], 'o', color=c, ms=5,
                label=f"{nm} @{s['iteration']}  {s['Mnd']:.4f}")
    ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel(r'$M_{nd}$ (%)')
    ax.set_title(f'{nx}×{ny}   $M_{{nd}}$:  P / S1 / S2 / S3 / F'); ax.legend(loc='upper right')
save(fig, 'F2_Mnd_P_S1_S2_S3_F.png')


def rungbar(field, title, fname, bar=None, sign=+1, ylabel=None, logy=False, pct=True):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, (k, tag, nx, ny) in zip(axes, MESH):
        m = A['mesh'][k]
        v = [m['rungs'][f'rung{i}'][field] for i in (1, 2, 3, 4)]
        cols = list(RUNGC)
        if bar is not None:
            cols = [RUNGC[i] if sign * v[i] >= bar else '0.75' for i in range(4)]
        x = np.arange(4)
        ax.bar(x, v, color=cols, width=0.62)
        if bar is not None:
            ax.axhline(sign * bar, color='#d62728', ls='--', lw=1.1,
                       label=f'materiality bar {sign*bar:g}')
            ax.legend(loc='best')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(['r1\n0.04', 'r2\n0.02', 'r3\n0.01', 'r4\n0.005'])
        if logy: ax.set_yscale('symlog', linthresh=1e-3)
        for xi, vv in zip(x, v):
            ax.annotate(f'{vv:+.4g}' if pct else f'{vv:,.0f}', (xi, vv), ha='center',
                        va='bottom' if vv >= 0 else 'top', fontsize=6.8)
        ax.set_title(f'{nx}×{ny}')
        if ylabel: ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10)
    save(fig, fname)


# ---- F3  rung-by-rung omega1 (log, because rung 1 dwarfs the rest) ------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m = A['mesh'][k]
    v = [m['rungs'][f'rung{i}']['domega1_rel_pct'] for i in (2, 3, 4)]
    cols = [RUNGC[i + 1] if v[i] >= BAR else '0.75' for i in range(3)]
    x = np.arange(3)
    ax.bar(x, v, color=cols, width=0.6)
    ax.axhline(BAR, color='#d62728', ls='--', lw=1.2, label=f'materiality bar {BAR}%')
    if k == 'm160':
        comb = m['rungs']['rungs34']['domega1_rel_pct']
        ax.plot([1.5], [comb], marker='D', color='k', ms=6, ls='none',
                label=f'r3+r4 combined {comb:+.4f}%')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(['rung 2\n0.02', 'rung 3\n0.01', 'rung 4\n0.005'])
    ax.set_ylabel(r'$\Delta\omega_1$ (% relative)')
    for xi, vv in zip(x, v):
        ax.annotate(f'{vv:+.5f}', (xi, vv), ha='center',
                    va='bottom' if vv >= 0 else 'top', fontsize=7)
    ax.set_title(f'{nx}×{ny}'); ax.legend(loc='best')
fig.suptitle(r'Rung-by-rung $\omega_1$ contribution (rung 1 omitted: +143–147 %, off scale)'
             '\nat 160×20 rungs 3 and 4 are individually below the bar but their sum is above it',
             fontsize=9.5)
save(fig, 'F3_rung_omega1.png')

# ---- F4  rung-by-rung M_nd ----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m = A['mesh'][k]
    v = [m['rungs'][f'rung{i}']['dMnd_rel_pct'] for i in (2, 3, 4)]
    cols = [RUNGC[i + 1] if -v[i] >= A['thresholds']['Mnd_rel_pct'] else '0.75' for i in range(3)]
    x = np.arange(3)
    ax.bar(x, v, color=cols, width=0.6)
    ax.axhline(-A['thresholds']['Mnd_rel_pct'], color='#d62728', ls='--', lw=1.2,
               label='materiality bar −2 %')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(['rung 2\n0.02', 'rung 3\n0.01', 'rung 4\n0.005'])
    ax.set_ylabel(r'$\Delta M_{nd}$ (% relative)')
    for xi, vv in zip(x, v):
        ax.annotate(f'{vv:+.4f}', (xi, vv), ha='center', va='top', fontsize=7)
    ax.set_title(f'{nx}×{ny}'); ax.legend(loc='best')
fig.suptitle(r'Rung-by-rung $M_{nd}$ contribution (rung 1 omitted: −87 %, off scale) — '
             'no lower rung is material on any mesh', fontsize=9.5)
save(fig, 'F4_rung_Mnd.png')

# ---- F5  rung-by-rung topology change -----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m = A['mesh'][k]
    v = [m['rungs'][f'rung{i}']['rho_mean_abs'] for i in (2, 3, 4)]
    x = np.arange(3)
    ax.bar(x, v, color=[RUNGC[1], RUNGC[2], RUNGC[3]], width=0.6)
    ax.axhline(A['thresholds']['rho_mean_abs'], color='#d62728', ls='--', lw=1.2,
               label=r'materiality bar  mean$|\Delta\rho_e|$ = 0.01')
    ax.set_xticks(x); ax.set_xticklabels(['rung 2\n0.02', 'rung 3\n0.01', 'rung 4\n0.005'])
    ax.set_ylabel(r'mean $|\Delta\rho_e|$ over the rung')
    ax.set_ylim(0, 0.011)
    for xi, vv in zip(x, v):
        ax.annotate(f'{vv:.5f}', (xi, vv), ha='center', va='bottom', fontsize=7)
    ax.set_title(f'{nx}×{ny}'); ax.legend(loc='upper right')
fig.suptitle('Rung-by-rung topology change — every lower rung is an order of magnitude '
             'below the bar', fontsize=9.5)
save(fig, 'F5_rung_topology.png')

# ---- F6 / F7  rung-by-rung outer and inner work -------------------------
for field, ttl, fname, logy in (('d_outer', 'outer iterations', 'F6_rung_outer_work.png', True),
                                ('d_inner', 'inner MMA iterations', 'F7_rung_inner_work.png', True)):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, (k, tag, nx, ny) in zip(axes, MESH):
        m = A['mesh'][k]
        v = [m['rungs'][f'rung{i}'][field] for i in (1, 2, 3, 4)]
        x = np.arange(4)
        ax.bar(x, v, color=RUNGC, width=0.62)
        if logy: ax.set_yscale('log')
        ax.set_xticks(x); ax.set_xticklabels(['r1\n0.04', 'r2\n0.02', 'r3\n0.01', 'r4\n0.005'])
        ax.set_ylabel(ttl)
        tot = sum(v)
        for xi, vv in zip(x, v):
            ax.annotate(f'{vv:,}\n{100*vv/tot:.1f}%', (xi, vv), ha='center', va='bottom',
                        fontsize=6.5)
        ax.set_title(f'{nx}×{ny}   ({tag})')
    fig.suptitle(f'Rung-by-rung cost: {ttl}   (320×40 rung 4 = the CAP_HIT stage)', fontsize=10)
    save(fig, fname)

# ---- F8  scientific benefit per inner MMA work --------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.arange(3); w = 0.26
labels = [f'{nx}×{ny}' for _, _, nx, ny in MESH]
for ax, (fld, ttl) in zip(axes, [('domega1_per_1000_inner', r'$\Delta\omega_1$ per 1000 inner MMA'),
                                 ('dMnd_per_1000_inner', r'$\Delta M_{nd}$ per 1000 inner MMA')]):
    for j, r in enumerate(('rung2', 'rung3', 'rung4')):
        v = [A['mesh'][k]['rungs'][r][fld] for k, _, _, _ in MESH]
        ax.bar(x + (j - 1) * w, v, w, color=RUNGC[j + 1],
               label={'rung2': 'rung 2 (0.02)', 'rung3': 'rung 3 (0.01)',
                      'rung4': 'rung 4 (0.005)'}[r])
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ttl)
    ax.set_title(ttl); ax.legend()
fig.suptitle('Marginal scientific benefit per unit of inner MMA work', fontsize=10)
save(fig, 'F8_benefit_per_inner_work.png')

# ---- F9  S3 vs F topology ------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(11.5, 5.4))
for row, (k, tag, nx, ny) in enumerate(MESH):
    m = A['mesh'][k]
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO = np.array(h['RHO']).T
    s3 = RHO[:, m['S3']['iteration'] - 1].reshape(ny, nx, order='F')
    fF = RHO[:, m['F']['iteration'] - 1].reshape(ny, nx, order='F')
    for ax, im, ttl in ((axes[row, 0], s3, f"S3 @{m['S3']['iteration']}  (move 0.01)"),
                        (axes[row, 1], fF, f"F @{m['F']['iteration']}  ({m['F']['status']})"),
                        (axes[row, 2], fF - s3, 'F − S3')):
        if ttl == 'F − S3':
            h_ = ax.imshow(im, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        else:
            h_ = ax.imshow(1 - im, cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(h_, ax=ax, fraction=0.03)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f'{nx}×{ny}  {ttl}', fontsize=8)
    r = m['rungs']['rung4']
    axes[row, 2].set_xlabel(f"mean|Δρ|={r['rho_mean_abs']:.6f}  max|Δρ|={r['rho_max_abs']:.4f}",
                            fontsize=7)
fig.suptitle('S3 versus F topology — what the final move = 0.005 rung actually changes',
             fontsize=10)
save(fig, 'F9_topology_S3_vs_F.png')

# ---- F10  S3 vs F multiplicity / gap ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    m, T = A['mesh'][k], csv(tag)
    ax.plot(T['outer'], T['gap12'], color='0.3', lw=0.9, label=r'$(\omega_2-\omega_1)/\omega_1$')
    for nm, c in STATES:
        s = m[nm]
        ax.plot(s['iteration'], s['gap12'], 'o', color=c, ms=5,
                label=f"{nm}  gap {s['gap12']:.5f}")
    ax.axhline(m['production']['gap12'], color=CP, lw=1.1,
               label=f"P {m['production']['gap12']:.5f}")
    ax2 = ax.twinx(); ax2.plot(T['outer'], T['multN'], color='#ff7f0e', lw=0.8, alpha=0.7)
    ax2.set_ylim(0, 3); ax2.set_ylabel('subspace size N', color='#ff7f0e'); ax2.grid(False)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('relative gap')
    ax.set_title(f'{nx}×{ny}   multiplicity / gap'); ax.legend(loc='center right')
save(fig, 'F10_multiplicity_gap.png')

# ---- F11  declaration offset across all stages and meshes ---------------
fig, ax = plt.subplots(figsize=(9.5, 4.0))
x = []; y = []; cols = []; lab = []; missing = []
for mi, (k, tag, nx, ny) in enumerate(MESH):
    for s_ in A['mesh'][k]['declaration_timing']:
        off = s_['declaration_offset']
        xi = mi * 5.2 + s_['stage']
        x.append(xi); cols.append(RUNGC[s_['stage'] - 1]); lab.append(f"s{s_['stage']}")
        if off is None:
            y.append(0.0); missing.append(xi)
        else:
            y.append(off)
ax.bar(x, y, color=cols, width=0.74)
ax.axhline(38, color='#d62728', ls='--', lw=1.3,
           label='earliest arithmetically possible offset = 38')
for xi, yi in zip(x, y):
    if xi in missing:
        ax.annotate('never\ndeclares', (xi, 1.6), ha='center', va='bottom',
                    fontsize=7, color='#d62728', fontweight='bold')
    else:
        ax.annotate(f'{int(yi)}', (xi, yi), ha='center', va='bottom', fontsize=7.5)
ax.set_yscale('log'); ax.set_ylim(1, 1500)
ax.set_xticks(x); ax.set_xticklabels(lab)
for mi, (_, _, nx, ny) in enumerate(MESH):
    ax.text(mi * 5.2 + 2.5, -0.085, f'{nx}×{ny}', ha='center', va='top', fontsize=10,
            fontweight='bold', transform=ax.get_xaxis_transform(), clip_on=False)
ax.set_ylabel('declaration − stageStart')
ax.set_title('Declaration offset by stage\n'
             'stage 1 takes 101–387 iterations; every lower stage that fires takes '
             'exactly 38 — the arithmetic minimum', fontsize=9.5)
ax.legend(loc='upper right')
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, 'F11_declaration_offset.png', tight=False)

# ---- F12  first-evaluable window vs declaration -------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
for ax, (k, tag, nx, ny) in zip(axes, MESH):
    D = A['mesh'][k]['declaration_timing']
    for s in D:
        y = s['stage']
        a, b = s['stageStart'], s['stageEnd']
        ax.plot([a, b], [y, y], color='0.8', lw=7, solid_capstyle='butt')
        fe = s['first_evaluable_iter']
        ax.plot([a, fe], [y, y], color='#c7c7c7', lw=7, solid_capstyle='butt')
        ax.plot(fe, y, marker='|', color='#1f77b4', ms=14, mew=2)
        ep = s['earliest_possible_declaration']
        ax.plot(ep, y, marker='|', color='#d62728', ms=14, mew=1.5)
        if s['offline_decl'] is not None:
            ax.plot(s['offline_decl'], y, 'o', color='#2ca02c', ms=6)
        else:
            ax.annotate('never declares', (b, y), xytext=(-4, -14),
                        textcoords='offset points', fontsize=7, color='#d62728',
                        ha='right', va='center')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['r1 0.04', 'r2 0.02', 'r3 0.01', 'r4 0.005'])
    ax.invert_yaxis(); ax.set_xlabel('outer iteration')
    ax.set_title(f'{nx}×{ny}')
    if k == 'm320': ax.set_xscale('log')
h = [plt.Line2D([], [], marker='|', ls='none', color='#1f77b4', ms=12, mew=2,
                label='first mathematically evaluable (stageStart+19)'),
     plt.Line2D([], [], marker='|', ls='none', color='#d62728', ms=12, mew=1.5,
                label='earliest possible declaration (stageStart+38)'),
     plt.Line2D([], [], marker='o', ls='none', color='#2ca02c', ms=6,
                label='actual declaration')]
fig.suptitle('First evaluable window versus actual declaration.  In every lower stage the '
             'green dot sits exactly on the red tick.', fontsize=9.5)
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, 0.015))
save(fig, 'F12_first_evaluable_vs_declaration.png', tight=False)

# ---- F13  320x40: S3 termination vs the four-rung CAP_HIT path ----------
m, T = A['mesh']['m320'], csv('C320x40')
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for ax, (fld, ttl, ylab) in zip(axes, [('Mnd', r'$M_{nd}$', r'$M_{nd}$ (%)'),
                                       ('omega1', r'$\omega_1$', r'$\omega_1$'),
                                       ('cumInner', 'cumulative inner MMA', 'inner MMA iterations')]):
    k3 = m['S3']['iteration']
    ax.plot(T['outer'][:k3], T[fld][:k3], color='#2ca02c', lw=1.6,
            label='three-rung path (executed, then STOP)')
    ax.plot(T['outer'][k3 - 1:], T[fld][k3 - 1:], color=CF, lw=1.0, alpha=0.8,
            label='four-rung continuation → CAP_HIT')
    ax.axvline(k3, color='#2ca02c', ls='--', lw=1.2)
    ax.annotate(f'S3 @{k3}\nterminate', (k3, ax.get_ylim()[1]), xytext=(6, -20),
                textcoords='offset points', color='#2ca02c', fontsize=7.5, va='top')
    ax.axvline(1600, color=CF, ls=':', lw=1.2)
    ax.annotate('CAP_HIT\n@1600', (1600, ax.get_ylim()[0]), xytext=(-40, 14),
                textcoords='offset points', color=CF, fontsize=7.5)
    if fld == 'cumInner': ax.set_yscale('log')
    if fld == 'omega1': ax.set_ylim(166.30, 166.55)
    if fld == 'Mnd': ax.set_ylim(12.85, 13.15)
    ax.set_xlabel('outer iteration'); ax.set_ylabel(ylab); ax.set_title(f'320×40  {ttl}')
    ax.legend(loc='best')
fig.suptitle('320×40: terminating at S3 (move = 0.01) versus the four-rung move = 0.005 stage.\n'
             '1248 further outer iterations and 70 034 inner MMA iterations buy '
             r'$\Delta M_{nd}=-0.13\%$ and $\Delta\omega_1=-0.005\%$, and never terminate.',
             fontsize=9.5)
save(fig, 'F13_C320_S3_vs_CAP_HIT.png')

# ---- F14  architecture summary ------------------------------------------
fig = plt.figure(figsize=(12, 6.6))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.30,
                      left=0.06, right=0.98, top=0.93, bottom=0.04,
                      height_ratios=[1.25, 1.0])

ax = fig.add_subplot(gs[0, 0])
m = A['mesh']['m160']
seq = [('P', m['production']['omega1'], CP), ('S1', m['S1']['omega1'], C1),
       ('S2', m['S2']['omega1'], C2), ('S3', m['S3']['omega1'], C3),
       ('F', m['F']['omega1'], CF)]
ax.bar(range(5), [v for _, v, _ in seq], color=[c for _, _, c in seq], width=0.62)
ax.axhline(m['production']['omega1'], color=CP, ls='--', lw=1.0)
ax.set_ylim(168.6, 170.3); ax.set_xticks(range(5)); ax.set_xticklabels([n for n, _, _ in seq])
for i, (_, v, _) in enumerate(seq):
    ax.annotate(f'{v:.4f}', (i, v), ha='center', va='bottom', fontsize=7)
ax.set_ylabel(r'$\omega_1$'); ax.set_title(r'160×20 $\omega_1$: S1 regresses, S2/S3 do not')

ax = fig.add_subplot(gs[0, 1])
v = [A['mesh']['m160']['rungs'][f'rung{i}']['domega1_rel_pct'] for i in (2, 3, 4)]
comb = A['mesh']['m160']['rungs']['rungs34']['domega1_rel_pct']
ax.bar([0, 1, 2], v, color=[RUNGC[1], RUNGC[2], RUNGC[3]], width=0.6)
ax.bar([3.2], [comb], color='0.4', width=0.6)
ax.axhline(BAR, color='#d62728', ls='--', lw=1.3, label=f'bar {BAR}%')
ax.set_xticks([0, 1, 2, 3.2])
ax.set_xticklabels(['r2', 'r3', 'r4', 'r3+r4\n(residual)'])
for xi, vv in zip([0, 1, 2, 3.2], v + [comb]):
    ax.annotate(f'{vv:+.4f}', (xi, vv), ha='center', va='bottom', fontsize=7)
ax.set_ylabel(r'$\Delta\omega_1$ (% rel.)'); ax.legend(loc='upper right', fontsize=6.5)
ax.set_title('THRESHOLD SPLITTING at 160×20:\nr3 and r4 each below the bar, their sum above it')

ax = fig.add_subplot(gs[0, 2])
x = np.arange(3); w = 0.38
sv = [A['mesh'][k]['cost']['S3_inner'] for k, _, _, _ in MESH]
fv = [A['mesh'][k]['cost']['total_inner'] for k, _, _, _ in MESH]
ax.bar(x - w/2, sv, w, color=C3, label='three-rung (stop at S3)')
ax.bar(x + w/2, fv, w, color=CF, label='four-rung total')
ax.set_yscale('log'); ax.set_xticks(x)
ax.set_xticklabels([f'{nx}×{ny}' for _, _, nx, ny in MESH])
for xi, a_, b_ in zip(x, sv, fv):
    ax.annotate(f'{a_:,}', (xi - w/2, a_), ha='center', va='bottom', fontsize=6.5)
    ax.annotate(f'{b_:,}', (xi + w/2, b_), ha='center', va='bottom', fontsize=6.5)
ax.set_ylabel('inner MMA iterations'); ax.legend(fontsize=6.5)
ax.set_title('Cost of the final rung')

ax = fig.add_subplot(gs[1, :])
ax.axis('off')
rows = [['mesh', 'S3', 'branch', 'E on 0.01', 'rung 3 material', 'rung 4 material',
         'gates @S3', 'four-rung status', 'inner saved']]
for k, tag, nx, ny in MESH:
    p = M['per_mesh'][k]; c = A['mesh'][k]['cost']
    rows.append([f'{nx}×{ny}', str(p['S3_iteration']), p['S3_branch'],
                 'yes' if p['E_on_0p01'] else 'NO',
                 'yes' if p['rung3_material_any'] else 'no',
                 'yes' if p['rung4_material_any'] else 'no',
                 'PASS' if p['gates_all_pass'] else 'FAIL',
                 A['mesh'][k]['F']['status'],
                 f"{c['saved_inner']:,} ({c['saved_inner_pct']:.1f} %)"])
t = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.6)
for j in range(len(rows[0])):
    t[0, j].set_facecolor('#e8e8e8'); t[0, j].set_text_props(weight='bold')
ax.set_title('Architecture summary — every mesh terminates at S3; no rung below 0.02 is '
             'individually material\nTHREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED '
             '(threshold-splitting guard)', fontsize=10, pad=18)
save(fig, 'F14_architecture_summary.png', tight=False)

json.dump(written, open(os.path.join(STUDY, 'evidence', 'figures.json'), 'w'), indent=1)
print(f'\n{len(written)} figures written')
