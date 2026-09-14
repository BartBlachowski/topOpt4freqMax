#!/usr/bin/env python3
"""btm_figures -- figures for the beta mechanism audit (brief sec. 19).

Figure 10 (beta vs gradient norms) cannot be produced: gradient statistics were
never recorded by the production recorder.  It is replaced by the observable
through which gradient scale actually enters beta -- the predicted gain
g = (beta-lambda)/lambda and g/move -- and labelled as such, not reconstructed.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import btm_common as B

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 8,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'axes.labelsize': 8})
FIG = os.path.join(B.OUT, 'figures')
COL = {'160x20': '#1f77b4', '320x40': '#ff7f0e', '400x50': '#d62728'}
written = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    written.append(name); print('  ', name)


D = {}
for mesh, spec in B.RUNS.items():
    P = B.load(spec['prod']); F = B.load(spec['fixed'])
    rel = B.stall_rel(P['beta'])
    d = [P['outer'][k] for k in range(1, P['n']) if P['move'][k] < P['move'][k-1]]
    D[mesh] = dict(P=P, F=F, rel=rel, desc=d, NE=spec['NE'])


def main():
    # ---- 1  beta vs iteration -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for m, d in D.items():
        ax.plot(d['P']['outer'], d['P']['beta'], color=COL[m], lw=1.2, label=f'{m} beta')
        ax.plot(d['P']['outer'], [w**2 for w in d['P']['omega1']], color=COL[m], lw=0.8,
                ls=':', label=f'{m} omega_1^2')
        ax.axvline(d['desc'][0], color=COL[m], ls='--', lw=0.9, alpha=0.7)
    ax.set_xlim(0, 200); ax.set_xlabel('outer iteration'); ax.set_ylabel('beta  /  omega_1^2')
    ax.set_title('beta is the bound variable of Eq. (25): the subproblem-PREDICTED eigenvalue bound.\n'
                 'It collapses onto omega_1^2 long before the topology matures. '
                 'Dashed = first production descent.')
    ax.legend(ncol=3); save(fig, 'fig1_beta_vs_iteration.png')

    # ---- 2  stall metric -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for m, d in D.items():
        ax.plot(d['P']['outer'], d['rel'], color=COL[m], lw=1.1, label=m)
        for x in d['desc']:
            ax.plot([x], [d['rel'][x-1]], 'v', color=COL[m], ms=7, mec='k', mew=0.6)
    ax.axhline(B.TOL, color='k', ls='--', lw=1.2, label=f'tolerance = {B.TOL}')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlim(0, 250); ax.set_ylim(-0.02, 0.06)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('rel = 10-iter relative increase of beta')
    ax.set_title('The stall metric. Triangles = descents.  Once rel crosses the tolerance it STAYS\n'
                 'below it, so every later rung fires at the dwell minimum (W+1 = 11 iterations).')
    ax.legend(); save(fig, 'fig2_stall_metric_vs_iteration.png')

    # ---- 3  move with stall events ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for m, d in D.items():
        ax.step(d['P']['outer'], d['P']['move'], where='post', color=COL[m], lw=1.3, label=m)
        for x in d['desc']:
            ax.plot([x], [d['P']['move'][x-1]], 'v', color=COL[m], ms=7, mec='k', mew=0.6)
    ax.set_yscale('log'); ax.set_xlim(0, 250)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('move')
    ax.set_title('Move ladder with beta-stall descent events (triangles); spacing after the first\n'
                 'descent is exactly 11 = W+1 at both meshes that reached the ladder floor')
    ax.legend(); save(fig, 'fig3_move_with_stall_events.png')

    # ---- 4  aligned at first descent -------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for m, d in D.items():
        d0 = d['desc'][0]
        off = [o - d0 for o in d['P']['outer']]
        axs[0].plot(off, d['rel'], color=COL[m], lw=1.2, label=m)
        g = [(d['P']['beta'][i]-d['P']['omega1'][i]**2)/d['P']['omega1'][i]**2
             for i in range(d['P']['n'])]
        axs[1].plot(off, g, color=COL[m], lw=1.2, label=m)
    for a, yl, tt in [(axs[0], 'stall metric rel', 'stall metric, aligned at first descent'),
                      (axs[1], 'predicted gain g = (beta-lambda)/lambda',
                       'predicted eigenvalue gain, aligned')]:
        a.axvline(0, color='k', ls='--', lw=1.1); a.set_xlim(-60, 60)
        a.set_xlabel('iterations relative to first descent'); a.set_ylabel(yl)
        a.set_title(tt); a.legend()
    axs[0].axhline(B.TOL, color='0.4', ls=':', lw=1.0); axs[0].set_ylim(-0.02, 0.05)
    axs[1].set_yscale('log')
    save(fig, 'fig4_aligned_at_first_descent.png')

    # ---- 5  remaining evolution vs mesh ----------------------------------
    M = json.load(open(os.path.join(B.OUT, 'METRICS.json')))
    ev = M['event_aligned']['results']
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    NE = [e['NE'] for e in ev]; rem = [100*e['remaining_Mnd_relative'] for e in ev]
    rel = [e['stall_rel_at_descent'] for e in ev]
    ax.plot(NE, rem, 'o-', color='#d62728', ms=9, lw=1.5, label='remaining M_nd evolution')
    for e in ev:
        ax.annotate(f"{e['mesh']}\nrel={e['stall_rel_at_descent']:.5f}\n"
                    f"{100*e['remaining_Mnd_relative']:.1f}% left",
                    (e['NE'], 100*e['remaining_Mnd_relative']),
                    textcoords='offset points',
                    xytext=(-58 if e['mesh'] == '400x50' else 10, -6), fontsize=7.5)
    a2 = ax.twinx(); a2.grid(False)
    a2.plot(NE, rel, 's--', color='#1f77b4', ms=7, lw=1.2, label='stall metric at descent')
    a2.axhline(B.TOL, color='0.4', ls=':', lw=1.0)
    a2.set_ylabel('stall metric rel at descent', color='#1f77b4'); a2.set_ylim(0, 0.006)
    ax.set_xscale('log'); ax.set_xlabel('NE'); ax.set_ylabel('% of M_nd evolution remaining')
    ax.set_ylim(0, 60)
    ax.set_title('THE CENTRAL CONTRADICTION: the stall metric fires at essentially the same value\n'
                 'at all three meshes (spread 3.3e-4) while remaining topology evolution varies 5.6x')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='center right')
    save(fig, 'fig5_remaining_evolution_vs_mesh.png')

    # ---- 6  beta vs M_nd --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for m, d in D.items():
        F = d['F']; d0 = d['desc'][0]
        ax.plot(F['Mnd_pct'], F['beta'], color=COL[m], lw=1.2, label=f'{m} (fixed move)')
        i = min(d0-1, F['n']-1)
        ax.plot([F['Mnd_pct'][i]], [F['beta'][i]], 'o', color=COL[m], ms=9, mec='k', mew=0.8)
    ax.set_xlabel('M_nd (%)  [decreasing = maturing ->]'); ax.set_ylabel('beta')
    ax.invert_xaxis()
    ax.set_title('beta against topology maturity.  Circles = where production descends.\n'
                 'beta is already flat while M_nd still has most of its journey left at fine meshes.')
    ax.legend(); save(fig, 'fig6_beta_vs_Mnd.png')

    # ---- 7  beta vs rho-change -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for m, d in D.items():
        F = d['F']
        ax.plot(F['outer'], [x/((d['NE'])**0.5) for x in F['l2']], color=COL[m], lw=1.0,
                label=f'{m} RMS(drho)')
    for m, d in D.items():
        ax.axvline(d['desc'][0], color=COL[m], ls='--', lw=0.9, alpha=0.7)
    ax.set_yscale('log'); ax.set_xlim(0, 400)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('RMS(drho)')
    ax.set_title('Design motion continues after beta stalls (dashed = first descent)')
    ax.legend(); save(fig, 'fig7_beta_vs_rho_change.png')

    # ---- 8  beta vs inner iterations -------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for m, d in D.items():
        ax.plot(d['P']['outer'], d['P']['nInner'], color=COL[m], lw=0.9, label=m)
        ax.axvline(d['desc'][0], color=COL[m], ls='--', lw=0.9, alpha=0.7)
    ax.set_xlim(0, 250); ax.set_xlabel('outer iteration'); ax.set_ylabel('inner MMA iterations')
    ax.set_title('Inner subproblem effort. At the fine meshes it is steady (17-21) and always\n'
                 'converged: beta is stable because each subproblem is solved consistently.')
    ax.legend(); save(fig, 'fig8_beta_vs_inner_iterations.png')

    # ---- 9  bound-active fraction ----------------------------------------
    ba = json.load(open(os.path.join(B.OUT, 'METRICS.json')))['bound_activity_400x50']
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    if 'samples' in ba:
        s = ba['samples']
        ax.plot([r['outer'] for r in s], [r['maxU'] for r in s], 'o-', color='#d62728',
                lw=1.3, ms=6, label='max(u) = max|drho|/move, 400x50')
        ax.plot([r['outer'] for r in s], [r['nAtBound'] for r in s], 's-', color='#1f77b4',
                lw=1.3, ms=6, label='elements AT the move bound')
        ax.axvline(137, color='k', ls='--', lw=1.1, label='production descent')
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('bound activity')
    ax.set_title('ZERO elements are ever at the move bound at 400x50.  The move box is NOT binding\n'
                 'when production descends, so the descent cannot mean "the move is used up".')
    ax.legend(); save(fig, 'fig9_bound_active.png')

    # ---- 10  gradient-scale substitute ------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for m, d in D.items():
        P = d['P']
        gm = [((P['beta'][i]-P['omega1'][i]**2)/P['omega1'][i]**2)/P['move'][i]
              for i in range(P['n'])]
        ax.plot(P['outer'], gm, color=COL[m], lw=1.1, label=m)
        ax.axvline(d['desc'][0], color=COL[m], ls='--', lw=0.9, alpha=0.7)
    ax.set_yscale('log'); ax.set_xlim(0, 200)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('g / move')
    ax.set_title('SUBSTITUTE FOR FIG 10 -- gradient norms were never recorded, so they are not\n'
                 'reconstructed.  g/move is the observable gradient scale enters beta through;\n'
                 'it is ~2.0 at all three meshes early, i.e. no mesh-dependent scaling defect.')
    ax.legend(); save(fig, 'fig10_gradient_scale_substitute.png')

    # ---- 11  move/asymptote scale vs beta ---------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for m, d in D.items():
        P = d['P']
        g = [(P['beta'][i]-P['omega1'][i]**2)/P['omega1'][i]**2 for i in range(P['n'])]
        ax.plot(P['move'], g, '.', color=COL[m], ms=3, alpha=0.5, label=m)
    mm = np.array([0.005, 0.04])
    ax.plot(mm, 2.0*mm, 'k--', lw=1.2, label='g = 2.0 * move (early linear regime)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('move (which sets the box and hence the MMA asymptote spread)')
    ax.set_ylabel('predicted gain g')
    ax.set_title('move -> box -> asymptotes -> beta.  Early the coupling is exactly linear (g=2m);\n'
                 'by the descent the gain has collapsed far below the line, so beta is no longer\n'
                 'move-limited -- which is why descending barely changes it (<=0.115%).')
    ax.legend(); save(fig, 'fig11_move_asymptote_vs_beta.png')

    # ---- 12  dependency diagram -------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.axis('off')
    boxes = [(0.06, 0.78, 'move  m'), (0.30, 0.78, 'box  |drho| <= m'),
             (0.56, 0.78, 'MMA asymptotes\nlow/upp = x -/+ 0.5(xmax-xmin)\nxmax-xmin = 2m'),
             (0.84, 0.78, 'subproblem\ngeometry'),
             (0.84, 0.50, 'beta = max bound\n= lambda + gain'),
             (0.56, 0.50, 'gain ~ 2m  (early)\ngain -> 0  (late)'),
             (0.30, 0.50, 'stall metric\nrel = d(mean beta)/mean beta'),
             (0.06, 0.50, 'rel < 5e-3 ?'),
             (0.06, 0.22, 'DESCEND\none rung'),
             (0.42, 0.22, 'dwell guard\nouter - lastStage > 10'),
             (0.76, 0.22, 'rho distribution\n(M_nd, grayness)')]
    for x, y, t in boxes:
        ax.text(x, y, t, ha='center', va='center', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.35', fc='#eef3f8', ec='#3a6ea5', lw=1.0))
    arr = dict(arrowstyle='-|>', color='#3a6ea5', lw=1.3)
    for a, b in [((0.13, 0.78), (0.24, 0.78)), ((0.39, 0.78), (0.47, 0.78)),
                 ((0.68, 0.78), (0.78, 0.78)), ((0.84, 0.72), (0.84, 0.56)),
                 ((0.75, 0.50), (0.66, 0.50)), ((0.46, 0.50), (0.40, 0.50)),
                 ((0.21, 0.50), (0.14, 0.50)), ((0.06, 0.44), (0.06, 0.28))]:
        ax.annotate('', xy=b, xytext=a, arrowprops=arr)
    ax.annotate('', xy=(0.06, 0.72), xytext=(0.06, 0.28),
                arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.6,
                                connectionstyle='arc3,rad=-0.55'))
    ax.text(0.015, 0.50, 'the loop', rotation=90, color='#c0392b', fontsize=7.5,
            ha='center', va='center')
    ax.annotate('', xy=(0.36, 0.22), xytext=(0.14, 0.22), arrowprops=arr)
    ax.text(0.76, 0.36, 'NOT AN INPUT\nto beta at all', color='#c0392b', fontsize=8,
            ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(0.80, 0.30), xytext=(0.84, 0.44),
                arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.4, ls=':'))
    ax.text(0.5, 0.05,
            'MEASURED: the move->beta coupling is real early (g = 2m) but has COLLAPSED by the\n'
            'descent (beta changes <=0.115% across it, 0 elements at the bound), so there is NO\n'
            'circular feedback. The defect is that rho\'s DISTRIBUTION never enters beta at all.',
            ha='center', va='center', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.95)
    save(fig, 'fig12_dependency_diagram.png')

    print(f'\n{len(written)} figures written')


if __name__ == '__main__':
    main()
