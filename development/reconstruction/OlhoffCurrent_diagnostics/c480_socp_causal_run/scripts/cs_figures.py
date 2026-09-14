"""The 20 minimum figures.  Read-only on evaluations/*.  PNG + SVG.

Colors: reference data-viz palette. Control = categorical slot 1 (blue),
treatment = slot 2 (orange); differences use the blue<->red diverging pair with
a neutral gray midpoint; density uses a light->dark gray ramp (0 void, 1 solid).
Two-series charts always carry a legend; no dual axes.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-cs')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from cs_common import *
from cs_trajectory_metrics import read_csv

C_CTRL, C_TREAT = '#2a78d6', '#eb6834'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f19a99', '#b3302f'])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 1.6, 'savefig.facecolor': 'white'})
EXT = [0, 8, 0, 1]


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=160)
    fig.savefig(FIG / f'{name}.svg')
    plt.close(fig)


def img(ax, a, title, **kw):
    m = ax.imshow(a, origin='lower', extent=EXT, aspect='equal', interpolation='nearest', **kw)
    ax.set_title(title, loc='left', fontsize=9, color=INK); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(False)
    return m


def stage_lines(ax, tr, color):
    st = tr['stage']
    for k in np.flatnonzero(np.diff(st) > 0):
        ax.axvline(tr['outer'][k + 1], color=color, ls=':', lw=.9)


def two_series(ax, ctr, ttr, key, ylabel, scale=1.0):
    ax.plot(ctr['outer'], ctr[key] * scale, color=C_CTRL, label='control (repeated MMA)')
    ax.plot(ttr['outer'], ttr[key] * scale, color=C_TREAT, label='treatment (exact SOCP)')
    stage_lines(ax, ctr, C_CTRL); stage_lines(ax, ttr, C_TREAT)
    ax.set_xlabel('outer iteration'); ax.set_ylabel(ylabel); ax.legend(loc='best')


def main():
    FIG.mkdir(exist_ok=True)
    A = json.loads((EV / 'analysis.json').read_text())
    E = np.load(EV / 'endpoint_fields.npz')
    rc, rt = E['rho_control'], E['rho_treatment']
    ctr, ttr = read_csv(EV / 'traj_control.csv'), read_csv(EV / 'traj_treatment.csv')
    gC, gT = A['control']['geometry'], A['treatment']['geometry']
    om = lambda s: A[s]['spectral']['omega1']
    to2 = lambda r: r.reshape(NX, NY).T
    gray = LinearSegmentedColormap.from_list('dens', ['#fcfcfb', '#0b0b0b'])

    # 1-3 density fields and difference
    MI = A['matched_iteration']; kT = MI['iteration']; gCk = MI['control_geometry']
    with h5py.File(CONTROL_TRAJ, 'r') as f:
        rck = f['RHO'][kT - 1]
    term = A['treatment']['termination'] or A['treatment']['status']
    fig, axs = plt.subplots(2, 1, figsize=(10, 3.6), layout='constrained')
    img(axs[0], to2(rc), f'control final ρ (outer 386, CONVERGED) — ω₁ = {om("control"):.4f}, M_nd = {gC["Mnd_percent"]:.2f} %, gray = {100*gC["gray_fraction"]:.2f} %', cmap=gray, vmin=0, vmax=1)
    m = img(axs[1], to2(rck), f'control ρ at matched outer {kT} (stage 1) — ω₁ = {MI["control_omega1_at_rho_k"]:.4f}, M_nd = {gCk["Mnd_percent"]:.2f} %', cmap=gray, vmin=0, vmax=1)
    fig.colorbar(m, ax=axs, label='ρ', shrink=.8); save(fig, 'FIG_01_control_final_rho')
    fig, ax = plt.subplots(figsize=(10, 1.9), layout='constrained')
    m = img(ax, to2(rt), f'treatment ρ at termination (last accepted outer {kT}; {term} at {kT+1}) — ω₁ = {om("treatment"):.4f}, M_nd = {gT["Mnd_percent"]:.2f} %', cmap=gray, vmin=0, vmax=1)
    fig.colorbar(m, ax=ax, label='ρ', shrink=.9); save(fig, 'FIG_02_treatment_final_rho')
    fig, axs = plt.subplots(2, 1, figsize=(10, 3.6), layout='constrained')
    img(axs[0], to2(rt - rck), f'treatment − control at matched outer {kT}  (‖Δρ‖₂ = {MI["topology"]["l2"]:.2f}, relocation = {100*MI["topology"]["material_relocation_fraction"]:.1f} %)', cmap=DIV, vmin=-1, vmax=1)
    m = img(axs[1], to2(rt - rc), f'treatment (outer {kT}) − control final (outer 386)  (‖Δρ‖₂ = {A["topology"]["l2"]:.2f}) — not like-for-like', cmap=DIV, vmin=-1, vmax=1)
    fig.colorbar(m, ax=axs, label='Δρ', shrink=.8); save(fig, 'FIG_03_treatment_minus_control')

    # 4 histograms
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout='constrained')
    bins = np.linspace(0, 1, 51)
    for r, c, l, ls in [(rc, C_CTRL, 'control final (386)', '-'), (rck, C_CTRL, f'control at outer {kT}', '--'), (rt, C_TREAT, f'treatment at termination ({kT})', '-')]:
        h, _ = np.histogram(r, bins=bins); ax.stairs(100 * h / r.size, bins, color=c, lw=1.8, label=l, ls=ls)
    ax.set_yscale('log'); ax.set_xlabel('ρ'); ax.set_ylabel('% of elements per bin (log)'); ax.legend()
    ax.axvspan(.1, .9, color=GRID, alpha=.35, lw=0); ax.set_title('Final density histograms (shaded: gray band 0.1–0.9)', loc='left')
    save(fig, 'FIG_04_density_histograms')

    # 5 grouped bars
    keys = [('Mnd_percent', 'M_nd', 1), ('gray_fraction', 'gray', 100), ('mid_fraction', 'mid', 100), ('broad_core_fraction', 'broad core', 100)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout='constrained')
    x = np.arange(len(keys)); w = .26
    for off, g, c, l in [(-w - .02, gC, C_CTRL, 'control final (386)'), (0, gCk, '#86b6ef', f'control at outer {kT}'), (w + .02, gT, C_TREAT, f'treatment at termination ({kT})')]:
        vals = [g[k] * s for k, _, s in keys]
        b = ax.bar(x + off, vals, w, color=c, label=l)
        ax.bar_label(b, fmt='%.1f', fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, [k[1] for k in keys]); ax.set_ylabel('% of domain (M_nd in %)'); ax.legend(); ax.grid(axis='x', visible=False)
    ax.set_title('Final grayness metrics', loc='left'); save(fig, 'FIG_05_gray_mid_broad_comparison')

    # 6-9 trajectories
    for name, key, yl, sc in [('FIG_06_Mnd_vs_iteration', 'Mnd_percent', 'M_nd (%)', 1), ('FIG_07_gray_vs_iteration', 'gray_fraction', 'gray fraction (%)', 100),
                              ('FIG_08_mid_vs_iteration', 'mid_fraction', 'mid fraction (%)', 100)]:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), layout='constrained', width_ratios=[2, 1])
        two_series(axs[0], ctr, ttr, key, yl, sc); two_series(axs[1], ctr, ttr, key, yl, sc)
        axs[1].set_xlim(0, 30); axs[1].axvline(kT + 1, color=C_TREAT, lw=1, ls='--'); axs[1].get_legend().remove()
        axs[0].set_title(f'{yl} after each update (dotted: stage starts)', loc='left'); axs[1].set_title(f'zoom 1–30 (dashed: rejected outer {kT+1})', loc='left')
        save(fig, name)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.4), layout='constrained', width_ratios=[2, 1])
    for j, (key, yl) in enumerate([('omega1', 'ω₁ (pre-update)'), ('lam1', 'λ₁ = ω₁²')]):
        two_series(axs[j, 0], ctr, ttr, key, yl); two_series(axs[j, 1], ctr, ttr, key, yl)
        axs[j, 1].set_xlim(0, 30); axs[j, 1].get_legend().remove(); axs[j, 1].axvline(kT + 1, color=C_TREAT, lw=1, ls='--')
    axs[0, 0].set_title('Fundamental eigenfrequency and eigenvalue', loc='left'); axs[0, 1].set_title('zoom 1–30', loc='left')
    save(fig, 'FIG_09_omega1_lambda1_vs_iteration')

    # 10 predicted vs realized
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for ax, tr, c, l in [(axs[0], ctr, C_CTRL, 'control (MMA β)'), (axs[1], ttr, C_TREAT, 'treatment (SOCP β)')]:
        el = tr['eligible'] > 0
        ax.scatter(tr['pred'][el], tr['act'][el], s=10, color=c, alpha=.7, lw=0, label=l)
        lim = max(np.abs(tr['pred'][el]).max(), np.abs(tr['act'][el]).max()) if el.any() else 1
        ax.plot([0, lim], [0, lim], color=INK2, lw=.8, ls='--'); ax.axhline(0, color=INK2, lw=.6)
        ax.set_xscale('symlog', linthresh=1); ax.set_yscale('symlog', linthresh=1)
        ax.set_xlabel('predicted gain β − λ₁'); ax.set_ylabel('realized gain λ₁(k+1) − λ₁(k)'); ax.legend(loc='upper left')
    fig.suptitle('Predicted vs realized λ₁ gain per accepted step (dashed: r = 1)', x=.01, ha='left'); save(fig, 'FIG_10_predicted_vs_realized_gain')

    # 11 r_k
    fig, ax = plt.subplots(figsize=(7, 3.2), layout='constrained')
    for tr, c, l in [(ctr, C_CTRL, 'control'), (ttr, C_TREAT, 'treatment')]:
        ax.plot(tr['outer'], tr['r'], '.', ms=4, color=c, label=l)
    ax.axhline(1, color=INK2, lw=.8, ls='--'); ax.axhline(0, color=INK2, lw=.6)
    ax.set_yscale('symlog', linthresh=.1); stage_lines(ax, ttr, C_TREAT); stage_lines(ax, ctr, C_CTRL)
    ax.set_xlabel('outer iteration'); ax.set_ylabel('r_k = realized / predicted'); ax.legend(); ax.set_title(f'Model realization ratio (treatment: {kT} steps, fewer than the 20 required for a verdict)', loc='left')
    save(fig, 'FIG_11_realization_ratio_vs_iteration')

    # 12 bound saturation
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), layout='constrained', sharex=True)
    for ax, tr, lab in [(axs[0], ttr, 'treatment'), (axs[1], ctr, 'control')]:
        ax.plot(tr['outer'], 100 * tr['frac_any_bound'], color=C_TREAT if lab == 'treatment' else C_CTRL, label='any bound')
        ax.plot(tr['outer'], 100 * tr['frac_pm_move'], color='#1baf7a', label='± move bound')
        ax.plot(tr['outer'], 100 * tr['frac_density_limited'], color='#4a3aa7', label='density bound')
        ax.plot(tr['outer'], 100 * tr['gray_full_move_frac'], color='#e87ba4', ls='--', label='gray elements at ± move')
        ax.set_ylabel(f'{lab}: % of elements'); ax.legend(ncol=4, fontsize=8, loc='lower right'); stage_lines(ax, tr, INK2)
    for ax in axs: ax.set_xlim(0, 40); ax.set_ylim(-3, 103)
    axs[1].set_xlabel('outer iteration (first 40 shown; control continues to 386 with 0 % at bounds)'); axs[0].set_title('Bound saturation of the accepted increment (tol 1e-6·box width)', loc='left')
    save(fig, 'FIG_12_bound_saturation_vs_iteration')

    # 13 reversal
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), layout='constrained', sharex=True)
    two_series(axs[0], ctr, ttr, 'cos_prev', 'cos(Δρ_k, Δρ_{k−1})'); axs[0].axhline(-.5, color=INK2, lw=.6, ls='--')
    two_series(axs[1], ctr, ttr, 'sign_reversal_frac', 'element sign-reversal %', 100)
    for ax in axs: ax.set_xlim(0, 40)
    axs[0].set_title('Step coherence and sign reversal (first 40 outer iterations)', loc='left'); save(fig, 'FIG_13_reversal_cosine_vs_iteration')

    # 14 controller timeline
    fig, ax = plt.subplots(figsize=(7, 3), layout='constrained')
    for tr, c, l, s in [(ctr, C_CTRL, 'control', 'control'), (ttr, C_TREAT, 'treatment', 'treatment')]:
        ax.step(tr['outer'], tr['move'], where='post', color=c, label=l)
        for e in A[s]['events']:
            ax.plot(e['iter'], e['move'], 'o', ms=8, color=c, mec='white', mew=1.5)
            ax.annotate(f"{e['iter']} ({e['branch']})", (e['iter'], e['move']), textcoords='offset points', xytext=(4, 6), fontsize=8, color=INK2)
    ax.set_yscale('log'); ax.set_yticks([.01, .02, .04], ['0.01', '0.02', '0.04']); ax.set_xlabel('outer iteration'); ax.set_ylabel('move limit')
    ax.legend(); ax.set_title('Move ladder and exhaustion declarations (iteration, branch)', loc='left'); save(fig, 'FIG_14_controller_timeline')

    # 15-16 KKT maps
    kc, kt = np.load(EV / 'kkt_control.npz'), (np.load(EV / 'kkt_treatment.npz') if (EV / 'kkt_treatment.npz').exists() else None)
    kck = np.load(EV / 'kkt_control14.npz')
    if kt is not None:
        for name, key, title in [('FIG_15_physical_KKT_maps', 'raw_grayfit', 'physical reduced gradient (gray-fit dual) / raw interior RMS'),
                                 ('FIG_16_filtered_residual_maps', 'filtered_grayfit', 'filtered sub-problem residual (gray-fit dual) / raw interior RMS')]:
            a, a2, b = kc[key] / kc['raw_scale'], kck[key] / kck['raw_scale'], kt[key] / kt['raw_scale']
            lim = float(np.quantile(np.abs(np.r_[a, a2, b]), .995))
            fig, axs = plt.subplots(3, 1, figsize=(10, 5.2), layout='constrained')
            img(axs[0], to2(a), f'control final (386) — {title}', cmap=DIV, vmin=-lim, vmax=lim)
            img(axs[1], to2(a2), f'control at outer {kT}', cmap=DIV, vmin=-lim, vmax=lim)
            m = img(axs[2], to2(b), f'treatment at termination (outer {kT})', cmap=DIV, vmin=-lim, vmax=lim)
            fig.colorbar(m, ax=axs, shrink=.8, label='clipped at pooled 99.5 %'); save(fig, name)

    # 17 gray components
    fig, axs = plt.subplots(2, 1, figsize=(10, 3.6), layout='constrained')
    for ax, lab, key, g in [(axs[0], 'control', 'labels_control', gC), (axs[1], 'treatment', 'labels_treatment', gT)]:
        L = E[key].astype(float); n = int(L.max())
        base = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
        cols = ['#fcfcfb'] + [base[i % 8] for i in range(max(n, 1))]
        img(ax, L, f'{lab}: {g["gray_components_4"]} gray components (4-conn.), largest {g["largest_gray_component_area"]:.3f}, broad-core area {g["broad_core_area"]:.3f}, max depth/R {g["max_depth_over_R"]:.2f}',
            cmap=ListedColormap(cols), vmin=-.5, vmax=max(n, 1) + .5)
    save(fig, 'FIG_17_gray_component_maps')

    # 18 model error by stage
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout='constrained')
    data, labels, colors = [], [], []
    for tr, c, l in [(ctr, C_CTRL, 'C'), (ttr, C_TREAT, 'T')]:
        for st in np.unique(tr['stage']):
            m = (tr['stage'] == st) & (tr['eligible'] > 0)
            if m.any():
                data.append(tr['model_err_rel'][m]); labels.append(f'{l} s{int(st)}\nmv {tr["move"][m][0]:.2g}'); colors.append(c)
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for p_, c in zip(bp['boxes'], colors): p_.set_facecolor(c); p_.set_alpha(.55)
    ax.axhline(0, color=INK2, lw=.7); ax.set_ylabel('(realized − predicted) / predicted'); ax.grid(axis='x', visible=False)
    ax.set_title('Relative model error by stage (C control, T treatment)', loc='left'); save(fig, 'FIG_18_model_error_by_stage')

    # 19 SOCP cost
    T = read_csv(RUN / 'C480x60_socp_socp_iterations.csv')
    fig, ax = plt.subplots(figsize=(7, 3.2), layout='constrained')
    ax.plot(T['outer'], T['tSolve'], color=C_TREAT, label='accepted SOCP solve')
    ax.plot(T['outer'], T['tCertificate'], color='#4a3aa7', label='certificate')
    ax.plot(T['outer'], T['tAssembly'], color='#1baf7a', label='assembly')
    ax.plot(T['outer'], T['cross_tSolve'], color='#eda100', label='cross-solver diagnostic (excluded)')
    ax.plot(ctr['outer'], ctr['tInner'], color=C_CTRL, label='control inner MMA')
    ax.set_yscale('log'); ax.set_xlabel('outer iteration'); ax.set_ylabel('seconds'); ax.legend(ncol=2, fontsize=8)
    ax.set_title('Inner-solver cost per outer iteration', loc='left'); save(fig, 'FIG_19_socp_cost_vs_iteration')

    # 20 causal summary
    V = A['verdicts']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained', width_ratios=[1.2, 1])
    names = ['M_nd', 'gray', 'mid', 'broad core']
    keysG = ['Mnd_percent', 'gray_fraction', 'mid_fraction', 'broad_core_fraction']
    y = np.arange(len(names)); h = .38
    b1 = axs[0].barh(y + h / 2 + .01, [gT[k] / gCk[k] for k in keysG], h, color=C_TREAT, label=f'treatment / control, both at outer {kT}')
    b2 = axs[0].barh(y - h / 2 - .01, [gT[k] / gC[k] for k in keysG], h, color=INK2, label='treatment (outer 14) / control final (386)')
    for b in (b1, b2): axs[0].bar_label(b, fmt='%.2f', padding=3, fontsize=8, color=INK2)
    axs[0].axvline(1, color=INK, lw=.8); axs[0].set_yticks(y, names); axs[0].invert_yaxis(); axs[0].grid(axis='y', visible=False)
    axs[0].set_xlabel('ratio (descriptive only; no causal verdict possible)'); axs[0].legend(loc='lower right', fontsize=8)
    axs[1].axis('off'); axs[1].grid(False)
    txt = [f"Termination: {A['treatment']['termination']} at outer {kT+1}",
           'Rejected point: at the SOC apex (predicted λ₁ = λ₂),', '  both backends exit 1, feasible; best dual gap 3.6e-4',
           '', 'Preregistered verdicts:', V['causal'], V['coverage'], V['realization'], V['socp_candidate'], V['filter_gate'], V['performance_gate']]
    for i, t_ in enumerate(txt):
        axs[1].text(0, 1 - i * .085, t_, fontsize=8, color=INK if i < 4 else INK2, transform=axs[1].transAxes, va='top', family='monospace' if i >= 5 else None)
    axs[0].set_title('Causal summary — treatment stopped fail-closed in stage 1', loc='left')
    save(fig, 'FIG_20_causal_summary')
    print('figures written:', len(list(FIG.glob('*.png'))))


if __name__ == '__main__':
    main()
