#!/usr/bin/env python3
"""ma4_figures -- the thirteen figures required by brief sec. C16."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ma4_traj, ma4_scaling as S

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})
OUT = S.OUT
FIG = os.path.join(OUT, 'figures')
written = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    written.append(name); print('  ', name)


def main():
    P = S.load(os.path.join(OUT, 'runs/P400_400x50_iterations.csv'))
    F = S.load(os.path.join(OUT, 'runs/F400_400x50_iterations.csv'))
    SC = json.load(open(os.path.join(OUT, 'SCALING_ANALYSIS.json')))
    di = next(r for r in SC['remaining_evolution'] if r['mesh'] == '400x50')
    dIter = di['descentIter']; dLast = di['lastIterAtMove004']

    def mark(ax, label=True):
        ax.axvline(dIter, color='crimson', lw=1.3, ls='--', zorder=5,
                   label=(f'production first descent (it {dIter})' if label else None))

    # ---- 1  M_nd -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(P['outer'], P['Mnd_pct'], color='#1f77b4', lw=1.5, label='ARM P400 (production ladder)')
    ax.plot(F['outer'], F['Mnd_pct'], color='#d62728', lw=1.3, label='ARM F400 (fixed move 0.04)')
    ax.axhline(F['Mnd_pct'][-1], color='0.5', ls=':', lw=1.0,
               label=f"F400 endpoint M_nd = {F['Mnd_pct'][-1]:.3f}%")
    mark(ax)
    i = F['outer'].index(dLast)
    ax.fill_between([dLast, F['outer'][-1]], F['Mnd_pct'][-1], F['Mnd_pct'][i],
                    color='crimson', alpha=0.10)
    ax.annotate(f"{F['Mnd_pct'][i]-F['Mnd_pct'][-1]:.2f} pts of M_nd still to come\n"
                f"({100*di['remaining_relative']:.1f}% of M_nd at the descent)",
                (dLast, F['Mnd_pct'][i]), textcoords='offset points', xytext=(30, 18),
                fontsize=7.5, color='crimson')
    ax.set_xlabel('outer iteration'); ax.set_ylabel('M_nd (%)')
    ax.set_title('400x50: M_nd, production vs fixed-move counterfactual'); ax.legend()
    save(fig, 'fig1_Mnd_vs_iteration.png')

    # ---- 2  omega1 ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(P['outer'], P['omega1'], color='#1f77b4', lw=1.5, label='ARM P400')
    ax.plot(F['outer'], F['omega1'], color='#d62728', lw=1.3, label='ARM F400')
    mark(ax)
    ax.annotate(f"from the descent to the F400 endpoint:\n"
                f"omega_1 moves {100*di['omega1_relChange']:.3f}%, "
                f"M_nd moves {100*di['remaining_relative']:.1f}%",
                (dLast, F['omega1'][i]), textcoords='offset points', xytext=(30, -40), fontsize=7.5)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('omega_1'); ax.legend()
    ax.set_title('400x50: omega_1'); save(fig, 'fig2_omega1_vs_iteration.png')

    # ---- 3  move --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.step(P['outer'], P['move'], where='post', color='#1f77b4', lw=1.5, label='ARM P400')
    ax.step(F['outer'], F['move'], where='post', color='#d62728', lw=1.3, ls='--', label='ARM F400')
    mark(ax); ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel('move')
    ax.set_title('400x50: move limit'); ax.legend()
    save(fig, 'fig3_move_vs_iteration.png')

    # ---- 4  activity percentiles ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for c, lab in [('u_max', 'max'), ('u_P99', 'P99'), ('u_P975', 'P97.5'), ('u_P95', 'P95'),
                   ('u_P90', 'P90'), ('u_P75', 'P75'), ('u_P50', 'P50'), ('u_RMS', 'RMS')]:
        ax.plot(F['outer'], F[c], lw=1.0, label=lab)
    mark(ax); ax.set_yscale('log'); ax.set_xlabel('outer iteration')
    ax.set_ylabel('u = |drho|/move')
    ax.set_title('400x50 ARM F400: FULL utilisation distribution\n'
                 '(recoverable here only because the raw trajectory was retained)')
    ax.legend(ncol=4, loc='lower left')
    save(fig, 'fig4_activity_percentiles.png')

    # ---- 5 / 6  active counts and fractions ----------------------------
    cols = [('nActive_1e4', '1e-4'), ('nActive_epsRMS', 'epsRMS'),
            ('nActive_1e3', '1e-3'), ('nActive_1e2', '1e-2')]
    for idx, (norm, ylab, fname) in enumerate(
            [(1.0, 'active count', 'fig5_active_counts.png'),
             (20000.0, 'active fraction', 'fig6_active_fractions.png')]):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for c, lab in cols:
            ax.plot(F['outer'], [v/norm for v in F[c]], lw=1.0, label=f'|drho| > {lab}')
        mark(ax); ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel(ylab)
        ax.set_title(f'400x50 ARM F400: {ylab} (thresholds on RAW |drho|)'); ax.legend()
        save(fig, fname)

    # ---- 7-10  spatial maps at the descent ------------------------------
    T = ma4_traj.Traj('F')
    rho = T.rho(dLast); d = np.abs(T.drho(dLast)); u = T.u(dLast)
    panels = [(T.grid(rho), 'rho', 'gray_r', None, 'fig7_map_rho.png',
               f'physical density at iteration {dLast}'),
              (T.grid(d), '|drho|', 'magma', None, 'fig8_map_absdrho.png',
               f'|Delta rho| at iteration {dLast}'),
              (T.grid(u), 'u = |drho|/move', 'magma', None, 'fig9_map_u.png',
               f'utilisation u at iteration {dLast}'),
              (T.grid((d > 1e-3).astype(float)), 'active (|drho|>1e-3)', 'gray_r', (0, 1),
               'fig10_map_active_mask.png',
               f'low-threshold active mask at iteration {dLast}')]
    for img, lab, cmap, clim, fname, title in panels:
        fig, ax = plt.subplots(figsize=(9.0, 2.0))
        im = ax.imshow(img, cmap=cmap, aspect='equal', origin='upper',
                       vmin=None if clim is None else clim[0],
                       vmax=None if clim is None else clim[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.01, label=lab)
        ax.set_title(f'400x50 ARM F400 — {title}  (immediately before production first descent)',
                     fontsize=8)
        save(fig, fname)
    T.close()

    # ---- 11  N_active vs NE, log-log ------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=False)
    for a, c in zip(axs, S.CGRID):
        for tau, col in zip(S.TAUS, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
            rows = [r for r in SC['scaling_matched_maturity']
                    if r['threshold'] == tau and r['c'] == c][0]
            NE = [p['NE'] for p in rows['points']]; N = [p['N_active'] for p in rows['points']]
            a.plot(NE, N, 'o-', color=col, ms=5, lw=1.0,
                   label=f"{tau}  a={rows['alpha_global']:.2f}")
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('NE')
        a.set_title(f'matched maturity c = {c}'); a.legend(fontsize=6.4)
    axs[0].set_ylabel('N_active')
    fig.suptitle('Active-set size vs mesh size at MATCHED M_nd maturity, three meshes', fontsize=9)
    save(fig, 'fig11_Nactive_vs_NE_loglog.png')

    # ---- 12  pairwise vs global alpha -----------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    labels, ypos = [], 0
    for tau in S.TAUS:
        for c in S.CGRID:
            r = [x for x in SC['scaling_matched_maturity']
                 if x['threshold'] == tau and x['c'] == c][0]
            vals = [r['alpha_160_320'], r['alpha_320_400'], r['alpha_160_400']]
            ax.plot(vals, [ypos]*3, 'o', ms=5,
                    color='#1f77b4' if ypos % 2 == 0 else '#4c9ad4')
            ax.plot([min(vals), max(vals)], [ypos]*2, '-', color='0.6', lw=1.0, zorder=0)
            ax.plot(r['alpha_global'], ypos, 'x', ms=8, color='crimson', mew=1.8)
            labels.append(f'{tau}, c={c}'); ypos += 1
    for x, lab, c in [(0, 'constant COUNT', 'crimson'), (0.5, 'sqrt(NE)', '#ff7f0e'),
                      (1, 'constant FRACTION', 'crimson')]:
        ax.axvline(x, color=c, ls='--', lw=1.1)
        ax.annotate(lab, (x, ypos-0.4), rotation=90, fontsize=6.5, color=c, ha='right', va='top')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel(r'exponent $\alpha$'); ax.set_ylim(-0.8, ypos-0.2)
    ax.set_title('Pairwise (blue) vs global three-point (red x) exponents\n'
                 'A tight global fit must not hide incompatible pairwise scaling', fontsize=8.5)
    save(fig, 'fig12_pairwise_vs_global_alpha.png')

    # ---- 13  remaining evolution vs mesh --------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    rr = SC['remaining_evolution']
    NEs = [r['NE'] for r in rr]; rel = [100*r['remaining_relative'] for r in rr]
    ax.plot(NEs, rel, 'o-', color='#d62728', ms=8, lw=1.4)
    for r in rr:
        ax.annotate(f"{r['mesh']}\nit {r['descentIter']}\n{100*r['remaining_relative']:.1f}%",
                    (r['NE'], 100*r['remaining_relative']), textcoords='offset points',
                    xytext=(8, -4), fontsize=7)
    ax.set_xscale('log'); ax.set_xlabel('NE'); ax.set_ylabel('% of M_nd still to come')
    ax.set_title('Topology evolution remaining when production first descends from move=0.04')
    save(fig, 'fig13_remaining_evolution_vs_mesh.png')

    print(f'\n{len(written)} figures written')


if __name__ == '__main__':
    main()
