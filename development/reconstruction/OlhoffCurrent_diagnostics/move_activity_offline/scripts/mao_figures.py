#!/usr/bin/env python3
"""mao_figures -- figures for the offline move-activity study.

Brief sec. 16 asks for ten figures.  Three of them (5, 6, 7: spatial u_e maps and
a persistence map) require PER-ELEMENT data that does not survive -- see
DATA_INVENTORY.md.  They are NOT faked.  In their place this script draws the
strongest honest substitute the surviving aggregates permit, and each such panel
says on its face what it is and what it is not.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mao_common as M

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})
FIG = os.path.join(M.OUT, 'figures')
L = {r['key']: M.load(r) for r in M.RUNS}
A, B = L['ms_fixedmove_160x20'], L['ms_fixedmove_320x40']
THRS = ['1e-4', 'epsRMS', '1e-3', '1e-2']
UEQ = {'1e-4': 0.0025, 'epsRMS': 0.0221, '1e-3': 0.025, '1e-2': 0.25}
PDESC = {'160x20': 79, '320x40': 130}
written = []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout(); fig.savefig(p); plt.close(fig); written.append(name)
    print('  ', name)


def mark(ax, mesh, extra=()):
    ax.axvline(PDESC[mesh], color='crimson', lw=1.2, ls='--', zorder=5,
               label=f'production descent (it {PDESC[mesh]})')
    for x in extra:
        ax.axvline(x, color='crimson', lw=0.7, ls=':', alpha=0.6, zorder=5)


# ---- fig 1 & 2 : utilisation distribution summary -------------------------
for mesh, base, fm, extra in [('160x20', 'ms_baseline_160x20', 'ms_fixedmove_160x20', [90]),
                              ('320x40', 'ms_baseline_320x40', 'ms_fixedmove_320x40', [])]:
    R = L[fm]; Bl = L[base]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    it = R['outer']
    ax[0].plot(it, R['maxU'], color='#1f77b4', lw=0.9, label='max(u)  [= recorded r_rho]')
    ax[0].plot(it, R['rmsU'], color='#d62728', lw=1.1, label='RMS(u)')
    ax[0].axhline(0.5, color='0.4', lw=0.8, ls=':', label='r_rho = 0.5 (previous study gate)')
    mark(ax[0], mesh, extra)
    ax[0].set_ylabel('utilisation  u = |drho|/move'); ax[0].set_yscale('log')
    ax[0].set_title(f'{mesh}  fixed move = 0.04 (exact counterfactual continuation of production)\n'
                    'PERCENTILES P75/P90/P95/P99 ARE NOT RECOVERABLE: per-element history did not survive')
    ax[0].legend(loc='lower left', ncol=2)
    ax[1].plot(it, R['Neff'], color='#2ca02c', lw=1.0,
               label=r'participation number  $N_{eff}=(\|\Delta\rho\|_2/\max|\Delta\rho|)^2$')
    ax[1].axhline(R['NE'], color='0.5', lw=0.8, ls='--', label=f'NE = {R["NE"]}')
    mark(ax[1], mesh, extra)
    ax[1].set_yscale('log'); ax[1].set_ylabel('effective # elements'); ax[1].set_xlabel('outer iteration')
    ax[1].legend(loc='upper right')
    save(fig, f'fig{1 if mesh=="160x20" else 2}_utilisation_distribution_{mesh}.png')

# ---- fig 3 & 4 : exact active fractions -----------------------------------
for mesh, fm, extra in [('160x20', 'ms_fixedmove_160x20', [90]), ('320x40', 'ms_fixedmove_320x40', [])]:
    R = L[fm]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for t, c in zip(THRS, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
        ax.plot(R['outer'], [x/R['NE'] for x in R['nActive'][t]], lw=1.0, color=c,
                label=f'|drho| > {t}   (u > {UEQ[t]:g} at move 0.04)')
    mark(ax, mesh, extra)
    ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel('active fraction')
    ax.set_title(f'{mesh}  EXACT active fractions (recorded counts, not estimated) — fixed move 0.04')
    ax.legend(loc='lower left')
    save(fig, f'fig{3 if mesh=="160x20" else 4}_active_fractions_{mesh}.png')

# ---- fig 5 & 6 : SUBSTITUTE for the spatial u_e map -----------------------
for mesh, fm, i in [('160x20', 'ms_fixedmove_160x20', 77), ('320x40', 'ms_fixedmove_320x40', 128)]:
    R = L[fm]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    tt = [10**(-4 + 4.3*j/300.) for j in range(301)]
    cap = [min(1.0, R['rmsU'][i]**2/(t*t)) for t in tt]
    ax.plot(tt, cap, color='#1f77b4', lw=1.4, label=r'EXACT upper bound $\;frac(u\geq t)\leq RMS(u)^2/t^2$')
    xs = [UEQ[t] for t in THRS]; ys = [R['nActive'][t][i]/R['NE'] for t in THRS]
    ax.plot(xs, ys, 'o', color='#d62728', ms=6, label='EXACT measured points (recorded counts)')
    for x, y, t in zip(xs, ys, THRS):
        ax.annotate(f'>{t}', (x, y), textcoords='offset points', xytext=(4, 5), fontsize=6.5)
    ax.axvline(R['maxU'][i], color='k', lw=1.0, ls='--', label=f'max(u) = {R["maxU"][i]:.3f}')
    ax.axhline(1.0/R['NE'], color='0.5', lw=0.8, ls=':', label=f'1 element = 1/{R["NE"]}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-4, 2); ax.set_ylim(1e-5, 2)
    ax.set_xlabel('utilisation threshold t'); ax.set_ylabel('fraction of elements with u >= t')
    ax.set_title(f'{mesh}: what is KNOWN about the utilisation distribution at iteration {R["outer"][i]}\n'
                 f'(the iteration before production descends).  NOT the requested spatial map — '
                 'per-element\ndata did not survive, so no map, no element identity, no localisation.')
    ax.legend(loc='lower left')
    save(fig, f'fig{5 if mesh=="160x20" else 6}_distribution_at_descent_{mesh}.png')

# ---- fig 7 : SUBSTITUTE for the persistence map ---------------------------
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
for a, R, mesh, extra in [(ax[0], A, '160x20', [90]), (ax[1], B, '320x40', [])]:
    a.plot(R['outer'], R['Neff'], color='#2ca02c', lw=1.0, label=r'$N_{eff}$ (effective # active elements)')
    a.plot(R['outer'], R['nActive']['1e-3'], color='#1f77b4', lw=1.0, label='count |drho| > 1e-3')
    a.plot(R['outer'], R['nActive']['1e-2'], color='#d62728', lw=1.0, label='count |drho| > 1e-2')
    a.axhline(R['NE'], color='0.5', lw=0.8, ls='--', label=f'NE = {R["NE"]}')
    mark(a, mesh, extra); a.set_yscale('log'); a.set_xlabel('outer iteration')
    a.set_title(f'{mesh}'); a.legend(loc='lower left', fontsize=6.2)
ax[0].set_ylabel('number of elements')
fig.suptitle('Size of the active population over time — the aggregate substitute for the requested\n'
             'persistence map.  Element IDENTITY did not survive, so Jaccard overlap, birth/death rate\n'
             'and persistence duration are NOT computable and are not reported.', fontsize=8)
save(fig, 'fig7_active_population_size.png')

# ---- fig 8 : candidate statistics across meshes, on a comparable axis -----
def completion(R):
    M0, Mf = R['Mnd'][0], R['Mnd'][-1]
    return [(M0-R['Mnd'][k])/(M0-Mf) for k in range(R['n'])]
cA, cB = completion(A), completion(B)
panels = [('max(u)', lambda R: R['maxU'], 'log'),
          ('RMS(u)', lambda R: R['rmsU'], 'log'),
          ('active fraction  |drho|>1e-3', lambda R: [x/R['NE'] for x in R['nActive']['1e-3']], 'log'),
          ('active count  |drho|>1e-3', lambda R: R['nActive']['1e-3'], 'log'),
          (r'count / $NE^{0.8}$ (fitted)', lambda R: [x/R['NE']**0.8 for x in R['nActive']['1e-3']], 'log'),
          (r'$N_{eff}$', lambda R: R['Neff'], 'log')]
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6))
for a, (nm, fn, sc) in zip(axs.ravel(), panels):
    a.plot(cA, fn(A), color='#1f77b4', lw=1.0, label='160x20')
    a.plot(cB, fn(B), color='#d62728', lw=1.0, label='320x40')
    a.plot([cA[77]], [fn(A)[77]], 'o', color='#1f77b4', ms=7, mec='k', mew=0.8)
    a.plot([cB[128]], [fn(B)[128]], 'o', color='#d62728', ms=7, mec='k', mew=0.8)
    a.set_yscale(sc); a.set_title(nm); a.set_xlabel('M_nd completion c'); a.legend(fontsize=6.5)
fig.suptitle('Candidate statistics on a mesh-comparable axis (M_nd completion, not iteration).\n'
             'Circles = the iteration before production descends.  A usable rule needs the 160x20 circle\n'
             'BELOW the 320x40 circle (permit at 160x20, block at 320x40); only the count-like panels do.',
             fontsize=8.5)
save(fig, 'fig8_candidate_statistics_across_meshes.png')

# ---- fig 9 : M_nd with activity statistics -------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.8))
for a, R, mesh, i, extra in [(axs[0], A, '160x20', 77, [90]), (axs[1], B, '320x40', 128, [])]:
    a.plot(R['outer'], R['Mnd'], color='k', lw=1.4, label='M_nd (%)')
    a.axhline(R['Mnd'][-1], color='0.5', ls='--', lw=0.8, label=f'fixed-move final {R["Mnd"][-1]:.2f}%')
    a.fill_between([R['outer'][i], R['outer'][-1]], R['Mnd'][-1], R['Mnd'][i],
                   color='crimson', alpha=0.10)
    a.annotate(f'{R["Mnd"][i]-R["Mnd"][-1]:.2f} points of M_nd\nSTILL TO COME '
               f'({100*(R["Mnd"][i]-R["Mnd"][-1])/R["Mnd"][i]:.1f}%)',
               (R['outer'][i], R['Mnd'][i]), textcoords='offset points', xytext=(24, 12),
               fontsize=7, color='crimson')
    mark(a, mesh, extra)
    a2 = a.twinx(); a2.grid(False)
    a2.plot(R['outer'], [x/R['NE'] for x in R['nActive']['1e-3']], color='#2ca02c', lw=0.9,
            label='active fraction >1e-3')
    a2.plot(R['outer'], R['maxU'], color='#1f77b4', lw=0.7, alpha=0.8, label='max(u)')
    a2.set_yscale('log'); a2.set_ylabel('activity', color='#2ca02c')
    a.set_xlabel('outer iteration'); a.set_ylabel('M_nd (%)'); a.set_title(f'{mesh}  fixed move 0.04')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1+h2, l1+l2, loc='upper right', fontsize=6.3)
fig.suptitle('M_nd together with activity statistics.  max(u) stays pinned near 1.0 at 160x20 for the '
             'entire run\nwhile M_nd matures — it carries almost no maturity information there '
             '(Spearman 0.066).', fontsize=8.5)
save(fig, 'fig9_Mnd_with_activity.png')

# ---- fig 10 : omega1 with activity statistics -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.8))
for a, R, mesh, i, extra in [(axs[0], A, '160x20', 77, [90]), (axs[1], B, '320x40', 128, [])]:
    a.plot(R['outer'], R['omega1'], color='k', lw=1.4, label='omega_1')
    mark(a, mesh, extra)
    w0, w1 = R['omega1'][i], R['omega1'][-1]
    m0, m1 = R['Mnd'][i], R['Mnd'][-1]
    a.annotate(f'from here to the end:\nomega_1 moves {100*abs(w1-w0)/w0:.2f}%\n'
               f'M_nd moves {100*abs(m1-m0)/m0:.1f}%\n({(abs(m1-m0)/m0)/(abs(w1-w0)/w0):.0f}x more)',
               (R['outer'][i], w0), textcoords='offset points', xytext=(20, -46), fontsize=7)
    a2 = a.twinx(); a2.grid(False)
    a2.plot(R['outer'], [x/R['NE'] for x in R['nActive']['1e-3']], color='#2ca02c', lw=0.9,
            label='active fraction >1e-3')
    a2.set_yscale('log'); a2.set_ylabel('active fraction', color='#2ca02c')
    a.set_xlabel('outer iteration'); a.set_ylabel('omega_1'); a.set_title(f'{mesh}  fixed move 0.04')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1+h2, l1+l2, loc='lower right', fontsize=6.5)
fig.suptitle('omega_1 together with activity.  omega_1 is visually settled long before the topology is: '
             'it is a\npoor maturity proxy, and this analysis confirms rather than overturns that '
             'earlier finding.', fontsize=8.5)
save(fig, 'fig10_omega1_with_activity.png')

# ---- fig 11 : the normalisation-exponent result ---------------------------
import json
Mx = json.load(open(os.path.join(M.OUT, 'METRICS.json')))
rows = [r for r in Mx['normalisation_exponent']['results']
        if r.get('admissible') and r['c_min'] == 0.99]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
for j, r in enumerate(rows):
    ax.plot([r['alpha_lo'], r['alpha_hi']], [j, j], lw=5, solid_capstyle='butt',
            color='#2ca02c' if r['alpha_lo'] <= 0.5 <= r['alpha_hi'] else '#1f77b4')
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([(f"N_eff, D={r['persistence']}" if r['quantity']=='Neff'
                     else f"count |drho|>{r['quantity']}, D={r['persistence']}") for r in rows], fontsize=7)
for x, lab, c in [(0, 'constant COUNT', 'crimson'), (0.5, 'interface length ~sqrt(NE)', '#ff7f0e'),
                  (1, 'constant FRACTION', 'crimson')]:
    ax.axvline(x, color=c, lw=1.2, ls='--')
    ax.annotate(lab, (x, len(rows)-0.4), rotation=90, fontsize=6.5, color=c,
                ha='right', va='top')
# overlay the independent collapse estimate
for r in Mx['mesh_collapse']['results']:
    if not r['collapses']: continue
    lo, hi = r['alpha_band_1p5x']
    ax.axvspan(lo, hi, color='#2ca02c', alpha=0.10, zorder=0)
ax.axvspan(0.71, 0.92, color='#2ca02c', alpha=0.18, zorder=0,
           label='independent mesh-collapse estimate (low thresholds)')
ax.legend(loc='upper left', fontsize=6.5, framealpha=0.92)
ax.set_xlabel(r'admissible normalisation exponent $\alpha$   (statistic = activeCount / $NE^{\alpha}$)')
ax.set_title('Which normalisation could make one threshold work at BOTH meshes?  (c_min = 0.99)\n'
             'Constant COUNT is excluded everywhere.  The admissible band sits between the interface\n'
             'and area hypotheses — and TWO meshes cannot pin one exponent.', fontsize=8.5)
ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.8, len(rows)-0.2)
save(fig, 'fig11_normalisation_exponent.png')

print(f'\n{len(written)} figures written to figures/')
