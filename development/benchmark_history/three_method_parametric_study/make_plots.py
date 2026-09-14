#!/usr/bin/env python3
"""Figures for the three-method parametric study."""
import csv, json, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)

C = {'Olhoff': '#1f77b4', 'Yuksel': '#d62728', 'Proposed': '#2ca02c'}
MK = {'Olhoff': 'o', 'Yuksel': 's', 'Proposed': '^'}

def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return float('nan')

def load(n):
    p = os.path.join(RES, n)
    if not os.path.exists(p): return []
    with open(p) as fh: return list(csv.DictReader(fh))

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=140); plt.close(fig)
    print('  ', name)

pareto = load('method_pareto_profiles.csv')
elig = load('eligibility_ledger.csv')
ledger = load('parametric_run_ledger.csv')
traj = load('olhoff_trajectory_summary.csv')
wins = load('trajectory_window_costs.csv')
qlev = load('within_method_quality_levels.csv')
hold = load('cross_resolution_validation.csv')

# --- 1 & 2: runtime / iterations vs spectral quality, all methods ----------
for xkey, xlab, fname in (
        ('cost_loop_s', 'practical-stop optimization-loop time (s)', 'fig01_runtime_vs_quality.png'),
        ('n_iter_practical', 'iterations to practical stop', 'fig02_iterations_vs_quality.png')):
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for m in C:
        pts = [(fnum(r[xkey]), fnum(r['quality_omega1_E1']), r) for r in pareto if r['method'] == m]
        if not pts: continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=C[m], marker=MK[m],
                   s=52, label=m, alpha=.85, edgecolor='k', linewidth=.4)
        for x, y, r in pts:
            if r['selection_status'].startswith('SELECTED'):
                ax.scatter([x], [y], s=250, facecolors='none', edgecolors=C[m], linewidths=2.0)
                ax.annotate(r['run_id'].replace(m.lower() + '_', ''), (x, y),
                            textcoords='offset points', xytext=(8, 6), fontsize=7.5, color=C[m])
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'common raw E1  $\omega_1$  at practical stop  (rad/s)')
    ax.set_title('Cost vs common-evaluator spectral quality (240$\\times$30)\ncircled = selected profile')
    ax.grid(alpha=.3); ax.legend()
    save(fig, fname)

# --- 3: per-method Pareto fronts (own scales) -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, m in zip(axes, C):
    pts = sorted([(fnum(r['cost_loop_s']), fnum(r['quality_omega1_E1']), r)
                  for r in pareto if r['method'] == m])
    if not pts:
        ax.set_visible(False); continue
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c='0.6', s=40, label='dominated')
    nd = sorted([p for p in pts if p[2]['nondominated'] == 'True'])
    ax.plot([p[0] for p in nd], [p[1] for p in nd], '-', color=C[m], lw=1.6, zorder=3)
    ax.scatter([p[0] for p in nd], [p[1] for p in nd], c=C[m], s=70,
               marker=MK[m], zorder=4, edgecolor='k', linewidth=.4, label='non-dominated')
    for x, y, r in pts:
        if r['selection_status'].startswith('SELECTED'):
            ax.scatter([x], [y], s=280, facecolors='none', edgecolors='k', linewidths=1.8, zorder=5)
    span = max(p[1] for p in pts) - min(p[1] for p in pts)
    ax.set_title(f'{m}   (quality span {100*span/max(p[1] for p in pts):.3f}%)')
    ax.set_xlabel('loop time to practical stop (s)')
    ax.set_ylabel(r'$\omega_1$ common raw E1')
    ax.grid(alpha=.3); ax.legend(fontsize=7.5)
save(fig, 'fig03_per_method_pareto.png')

# --- 4: runtime vs iteration count ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for m in C:
    pts = [(fnum(r['n_iter_practical']), fnum(r['cost_loop_s'])) for r in pareto if r['method'] == m]
    if not pts: continue
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=C[m], marker=MK[m], s=55,
               label=f"{m}  ({1000*sum(p[1] for p in pts)/sum(p[0] for p in pts):.0f} ms/iter)",
               edgecolor='k', linewidth=.4)
ax.set_xlabel('iterations to practical stop'); ax.set_ylabel('loop time to practical stop (s)')
ax.set_title('Runtime vs iteration count (240$\\times$30)\nslope = mean per-iteration cost')
ax.grid(alpha=.3); ax.legend()
save(fig, 'fig04_runtime_vs_iterations.png')

# --- 6: per-iteration cost and eigensolve share along the trajectory ------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for m in C:
    by = defaultdict(list)
    for w in wins:
        if w['method'] != m: continue
        if m != 'Olhoff' and w['pass'] != 'stage_a_v2': continue
        by[w['run_id']].append(w)
    for i, (run, ws) in enumerate(sorted(by.items())):
        ws.sort(key=lambda w: int(w['window_start']))
        x = [int(w['window_start']) for w in ws]
        axes[0].plot(x, [1000 * fnum(w['loop_time_per_iter_s']) for w in ws],
                     color=C[m], alpha=.35, lw=1,
                     label=m if i == 0 else None)
        sh = [fnum(w['eig_share']) for w in ws if w['eig_share'] != '']
        if sh:
            axes[1].plot(x[:len(sh)], sh, color=C[m], alpha=.35, lw=1,
                         label=m if i == 0 else None)
axes[0].set_xlabel('iteration (window start)'); axes[0].set_ylabel('ms per iteration')
axes[0].set_title('WP22: per-iteration cost is trajectory dependent'); axes[0].set_xscale('symlog')
axes[1].set_xlabel('iteration (window start)'); axes[1].set_ylabel('eigensolve share of loop time')
axes[1].set_title('Eigensolve share (Olhoff telemetry only)')
for a in axes: a.grid(alpha=.3); a.legend(fontsize=8)
save(fig, 'fig06_iteration_cost_and_eig_share.png')

# --- 7: Olhoff move vs iterations and quality ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
tr = sorted(traj, key=lambda r: fnum(r['move']))
mv = [fnum(r['move']) for r in tr]
det = {fnum(r['move']): r for r in load('olhoff_detector_grid.csv')
       if r['family'] == 'hybrid' and r['lag'] == '2' and r['level'] == 'strict'}
fire = [fnum(det[m]['fire_iter']) if m in det else float('nan') for m in mv]
bim = [r['bimodal_strict_1pct'] == '1' for r in tr]
axes[0].bar([str(m) for m in mv], [f if f == f else 0 for f in fire],
            color=['#1f77b4' if b else '#bbbbbb' for b in bim])
for i, (f, b) in enumerate(zip(fire, bim)):
    axes[0].text(i, 30, 'never' if f != f else str(int(f)), ha='center', fontsize=7.5, rotation=90,
                 color='white' if (f == f and f > 300) else 'black')
axes[0].set_xlabel('move limit'); axes[0].set_ylabel('iteration of practical stop')
axes[0].set_title('Olhoff: move vs practical stop\nblue = bimodal (gap$_{12}\\leq$1%), grey = not bimodal')
axes[0].tick_params(axis='x', rotation=45); axes[0].grid(alpha=.3, axis='y')
axes[1].plot(mv, [100 * fnum(r['gap12_rel_final']) for r in tr], 'o-', color='#1f77b4')
axes[1].axhline(1.0, color='r', ls='--', lw=1, label='1% bimodality criterion')
axes[1].axhline(5.0, color='orange', ls=':', lw=1, label="method's own tolMult = 5%")
axes[1].set_xlabel('move limit'); axes[1].set_ylabel(r'terminal $\omega_2/\omega_1-1$  (%)')
axes[1].set_title('Olhoff: move is a bifurcation parameter, not a rate knob')
axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
save(fig, 'fig07_olhoff_move.png')

# --- 8: Yuksel stage decomposition ---------------------------------------
yl = [r for r in ledger if r['method'] == 'Yuksel' and r['pass'] == 'stage_a_v2']
yl.sort(key=lambda r: fnum(r['practical_stop_iter']))
fig, ax = plt.subplots(figsize=(9.5, 4.8))
lbl = [r['run_id'].replace('yuksel_', '') for r in yl]
s1 = [fnum(r['n_stage1']) for r in yl]
s2 = [max(0.0, fnum(r['practical_stop_iter']) - fnum(r['n_stage1'])) for r in yl]
ax.bar(lbl, s1, color='#f4a582', label=r'stage 1 (compliance, no eigensolve)  $N_1$')
ax.bar(lbl, s2, bottom=s1, color='#d62728', label=r'stage 2 to native stop  $N_2$')
for i, r in enumerate(yl):
    ax.text(i, s1[i] + s2[i] + 8, f"{fnum(r['loop_time_practical_s']):.1f}s", ha='center', fontsize=7.5)
ax.set_ylabel('iterations'); ax.tick_params(axis='x', rotation=40)
ax.set_title(r'Yuksel stage decomposition at the native stop ($N_{\rm total}=N_1+N_2$)')
ax.grid(alpha=.3, axis='y'); ax.legend()
save(fig, 'fig08_yuksel_stages.png')

# --- 9: Proposed native-parameter sensitivity ----------------------------
pl = [r for r in ledger if r['method'] == 'Proposed' and r['pass'] == 'stage_a_v2']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
tolr = sorted([r for r in pl if fnum(r['move']) == 0.2], key=lambda r: fnum(r['tol']))
axes[0].semilogx([fnum(r['tol']) for r in tolr], [fnum(r['practical_stop_iter']) for r in tolr],
                 'o-', color=C['Proposed'])
axes[0].set_xlabel('native convergence tolerance'); axes[0].set_ylabel('iterations to native stop')
axes[0].set_title('Proposed: tolerance is the dominant cost control')
mvr = sorted([r for r in pl if fnum(r['tol']) == 0.001], key=lambda r: fnum(r['move']))
axes[1].plot([fnum(r['move']) for r in mvr], [fnum(r['practical_stop_iter']) for r in mvr],
             's-', color=C['Proposed'])
axes[1].set_xlabel('move limit'); axes[1].set_ylabel('iterations to native stop')
axes[1].set_title('Proposed: a larger move costs MORE iterations')
for a in axes: a.grid(alpha=.3)
save(fig, 'fig09_proposed_sensitivity.png')

# --- 12: failure / robustness map ----------------------------------------
reasons = sorted({r['reason'] for r in elig})
fig, ax = plt.subplots(figsize=(11, 5.4))
ys = []
for m in C:
    rs = [r for r in elig if r['method'] == m]
    rs.sort(key=lambda r: r['run_id'])
    for r in rs:
        ys.append((m, r['run_id'] + ('' if r['pass'] == 'stage_a' else ' [v2]'), r['reason']))
pal = plt.get_cmap('tab20')
cmap = {rs: pal(i % 20) for i, rs in enumerate(reasons)}
for i, (m, run, reason) in enumerate(ys):
    ax.barh(i, 1, color=cmap[reason], edgecolor='k', linewidth=.3)
    ax.text(1.02, i, reason, va='center', fontsize=6.6)
ax.set_yticks(range(len(ys)))
ax.set_yticklabels([f'{m[:3]}  {run}' for m, run, _ in ys], fontsize=6.6)
ax.set_xlim(0, 2.4); ax.set_xticks([]); ax.invert_yaxis()
ax.set_title('Stage A eligibility / failure map (every attempted configuration)')
save(fig, 'fig12_failure_map.png')

# --- 5, 10, 11: cross-resolution (only once hold-outs exist) --------------
if hold:
    hr = [r for r in hold if r['status'] == 'COMPLETED']
    prof = sorted({r['profile'] for r in hr})
    VALID = ('CONVERGED_BIMODAL', 'CONVERGED_NATIVE')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for p in prof:
        rows = sorted([r for r in hr if r['profile'] == p], key=lambda r: int(r['nelx']))
        m = rows[0]['method']
        n = [int(r['nelx']) * int(r['nely']) for r in rows]
        axes[0].plot(n, [1000 * fnum(r['loop_time_per_iter_s']) for r in rows], 'o-',
                     color=C[m], alpha=.8, label=p)
        it = [fnum(r['practical_stop_iter']) for r in rows]
        axes[1].plot(n, it, 'o-', color=C[m], alpha=.8, label=p)
        axes[2].plot(n, [fnum(r['omega1_common_raw_E1_practical']) for r in rows], 'o-',
                     color=C[m], alpha=.8, label=p)
        # A stop that failed its look-ahead is not a result: cross it out so the
        # point cannot be read as achieved quality.
        bad = [(nn, r) for nn, r in zip(n, rows) if r['convergence_status'] not in VALID
               and fnum(r['practical_stop_iter']) == fnum(r['practical_stop_iter'])]
        for nn, r in bad:
            for ax, val in ((axes[1], fnum(r['practical_stop_iter'])),
                            (axes[2], fnum(r['omega1_common_raw_E1_practical']))):
                ax.scatter([nn], [val], marker='x', s=150, color='k', zorder=6, linewidths=2.2)
                ax.annotate('false\nconvergence', (nn, val), textcoords='offset points',
                            xytext=(-46, -6), fontsize=7, color='k')
    for a, t, yl_ in zip(axes,
                         ['WP22/5: per-iteration cost vs mesh',
                          'WP17: iterations to practical stop vs mesh',
                          'WP17: common-evaluator quality vs mesh'],
                         ['ms per iteration', 'iterations (NaN = never fired)',
                          r'$\omega_1$ common raw E1']):
        a.set_xlabel('elements (nelx$\\times$nely)'); a.set_ylabel(yl_)
        a.set_title(t); a.set_xscale('log'); a.grid(alpha=.3); a.legend(fontsize=6.5)
    axes[0].set_yscale('log')
    save(fig, 'fig05_10_11_cross_resolution.png')
else:
    print('   (cross-resolution figures deferred: hold-out results not present yet)')

print('figures written to', FIG)
