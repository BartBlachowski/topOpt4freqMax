#!/usr/bin/env python3
"""cp_figures.py -- the only figures this study's available data supports.

Every one of the twelve figures the task asks for needs CANARY trajectories.
No canary ran, so none of the twelve exists; figures/NOT_REACHED.md names each
missing input.  The three figures produced here are drawn from RETAINED
HISTORICAL and LEGACY records, are labelled as such in their titles, and each
supports a specific statement made elsewhere in the study.  None is decorative.
"""
import csv, json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import ROOT

AUD = ROOT / 'analysis/OlhoffCurrent/diagnostics/nine_mesh_campaign_audit'
FIG = Path(__file__).parents[1] / 'figures'
FIG.mkdir(exist_ok=True)
exp = json.load(open(Path(__file__).parents[1] / 'evidence/cost_expectation.json'))
legacy = {int(r['NE']): r for r in csv.DictReader(open(AUD / 'MASTER_TABLE.csv'))}


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# ===================== FIG A: validated three-rung structure ==============
S1 = {3200: 102, 7200: 206, 12800: 274, 20000: 388}
TOT = {3200: 180, 7200: 284, 12800: 352, 20000: 466}
C, p = exp['S1_fit']['C'], exp['S1_fit']['p']

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ne = sorted(S1)
ax[0].plot(ne, [S1[k] for k in ne], 'o-', label='S1 declaration (measured)')
ax[0].plot(ne, [TOT[k] for k in ne], 's-', label='total outer = S1 + 78 (measured)')
xs = [3000, 100000]
gx = [3000 * (100000 / 3000) ** (i / 60) for i in range(61)]
ax[0].plot(gx, [C * x ** p for x in gx], 'k--', lw=1,
           label=f'fit S1 = {C:.3g}·NE^{p:.3f}  (R²log={exp["S1_fit"]["R2_log"]:.3f})')
for m, k in (('480x60', 28800), ('800x100', 80000)):
    pr = exp['projection'][m]
    ax[0].plot(k, pr['projected_outer_total'], 'r*', ms=14)
    ax[0].annotate(f"{m}\nprojected {pr['projected_outer_total']}",
                   (k, pr['projected_outer_total']), textcoords='offset points',
                   xytext=(6, -22), color='firebrick', fontsize=8)
ax[0].axhline(1600, color='firebrick', ls=':', lw=1.2)
ax[0].annotate('cap = 1600 (frozen)', (3200, 1650), color='firebrick', fontsize=8)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('NE'); ax[0].set_ylabel('outer iterations')
ax[0].set_title('HISTORICAL validated three-rung structure\n+ EXTRAPOLATED canary budget (not a result)',
                fontsize=9)
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which='both')

stages = {}
for r in csv.DictReader(open(AUD / 'HISTORICAL_STAGE_WORK.csv')):
    stages.setdefault(r['mesh'], {})[int(r['stage'])] = int(r['outer'])
meshes = ['160x20', '240x30', '320x40', '400x50']
bottom = [0] * 4
for s, col in zip((1, 2, 3), ('#4C72B0', '#DD8452', '#55A868')):
    v = [stages[m][s] for m in meshes]
    ax[1].bar(meshes, v, bottom=bottom, label=f'stage {s} (move {[0.04,0.02,0.01][s-1]})', color=col)
    bottom = [b + x for b, x in zip(bottom, v)]
for i, m in enumerate(meshes):
    ax[1].text(i, bottom[i] + 8, f'{bottom[i]}', ha='center', fontsize=8)
ax[1].set_ylabel('outer iterations'); ax[1].legend(fontsize=7)
ax[1].set_title('HISTORICAL three-rung stage occupancy\nS2 and S3 sit at the minimum dwell 39 on every validated mesh',
                fontsize=9)
ax[1].grid(alpha=.3, axis='y')
save(fig, 'FIG_A_historical_three_rung_structure')

# ===================== FIG B: legacy next-mode warning regime =============
ne = sorted(legacy)
frac = [100 * int(legacy[k]['multJ_warning_count']) / int(legacy[k]['outer']) for k in ne]
gap12 = [(float(legacy[k]['omega2']) - float(legacy[k]['omega1'])) / float(legacy[k]['omega1']) for k in ne]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(ne, frac, 'o-', color='firebrick', label='iterations with next-mode (J) warning  [%]')
ax.set_xlabel('NE'); ax.set_ylabel('% of outer iterations', color='firebrick')
ax.set_xscale('log'); ax.grid(alpha=.3, which='both')
for k, f in zip(ne, frac):
    if f > 10:
        ax.annotate(f'{legacy[k]["mesh"]}\n{int(legacy[k]["multJ_warning_count"])}/{legacy[k]["outer"]}',
                    (k, f), textcoords='offset points', xytext=(-34, 4), fontsize=8, color='firebrick')
ax2 = ax.twinx()
ax2.plot(ne, gap12, 's--', color='#4C72B0', label='gap12 = (ω₂-ω₁)/ω₁ at the endpoint')
ax2.set_ylabel('gap12', color='#4C72B0')
ax.axvline(28800, color='gray', ls=':'); ax.axvline(80000, color='gray', ls=':')
ax.annotate('480x60', (28800, max(frac) * .82), rotation=90, fontsize=7, color='gray')
ax.annotate('800x100', (80000, max(frac) * .82), rotation=90, fontsize=7, color='gray')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
ax.set_title('LEGACY beta / four-rung campaign only — NOT canary data\n'
             'next-mode warning incidence and first-gap collapse under refinement', fontsize=9)
save(fig, 'FIG_B_legacy_next_mode_warning_regime')

# ===================== FIG C: the 720 -> 800 runtime inversion ============
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ne = sorted(legacy)
ax[0].plot(ne, [float(legacy[k]['runtime_total_s']) for k in ne], 'o-', label='total wall [s]')
ax[0].plot(ne, [float(legacy[k]['wall_per_outer_s']) * 100 for k in ne], 's--',
           label='wall per outer [s] × 100')
ax[0].plot(ne, [int(legacy[k]['outer']) for k in ne], '^:', label='outer iterations')
for k in (64800, 80000):
    ax[0].axvline(k, color='gray', ls=':', lw=.8)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].grid(alpha=.3, which='both')
ax[0].set_xlabel('NE'); ax[0].legend(fontsize=8)
ax[0].set_title('LEGACY campaign: total wall falls 720→800 while\nper-outer cost RISES — the inversion is the iteration count',
                fontsize=9)

lbl = ['720x90', '800x100']
kk = [64800, 80000]
comp = ['eigen_per_outer_s', 'gradient_per_outer_s', 'inner_per_outer_s']
names = ['assembly+eigensolve', 'gradients', 'inner MMA']
bottom = [0, 0]
for cpt, nm, col in zip(comp, names, ('#4C72B0', '#DD8452', '#55A868')):
    v = [float(legacy[k][cpt]) for k in kk]
    ax[1].bar(lbl, v, bottom=bottom, label=nm, color=col)
    bottom = [b + x for b, x in zip(bottom, v)]
for i, k in enumerate(kk):
    ax[1].text(i, bottom[i] + .2,
               f"{bottom[i]:.2f} s/outer\n× {legacy[k]['outer']} outer\n= {float(legacy[k]['runtime_total_s']):.0f} s",
               ha='center', fontsize=8)
ax[1].set_ylabel('seconds per outer iteration'); ax[1].legend(fontsize=8)
ax[1].set_ylim(0, max(bottom) * 1.45)
ax[1].set_title('LEGACY per-outer decomposition — NOT canary data', fontsize=9)
ax[1].grid(alpha=.3, axis='y')
save(fig, 'FIG_C_legacy_runtime_inversion')
