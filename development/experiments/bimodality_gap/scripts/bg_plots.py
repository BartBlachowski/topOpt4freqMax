#!/usr/bin/env python3
"""Figures and LaTeX tables for docs/bimodality_gap (reads data/*.csv and runs/*.mat)."""
import csv, json, os, glob, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
data = os.path.join(root, 'data'); figs = os.path.join(root, 'figures'); runs = os.path.join(root, 'runs')
os.makedirs(figs, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 150, 'savefig.dpi': 200,
                     'lines.markersize': 4, 'lines.linewidth': 1.0})
MESHES9 = [(160,20),(240,30),(320,40),(400,50),(480,60),(560,70),(640,80),(720,90),(800,100)]
A, B = 8.0, 1.0

def readcsv(p):
    with open(p) as f: return list(csv.DictReader(f))
def fl(x):
    try: return float(x)
    except: return float('nan')

camp = readcsv(os.path.join(data, 'campaign_metrics.csv'))
P = {(int(r['nelx']), int(r['nely'])): r for r in camp if r['formulation'] == 'P_pedersen_adaptive'}
S = {(int(r['nelx']), int(r['nely'])): r for r in camp if r['formulation'] == 'S_simp_ladder'}
vec = json.load(open(os.path.join(data, 'campaign_metrics_vectors.json')))['rows']
V = {(s['formulation'], s['nelx'], s['nely']): s for s in vec}
NE = lambda m: m[0]*m[1]

# ------------------------------------------------------------------ new runs
def load_run(path):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return d
arms = {}
for p in sorted(glob.glob(os.path.join(runs, 'BG_*.mat'))):
    d = load_run(p)
    arm = str(d['arm']); mesh = tuple(int(x) for x in d['mesh'])
    arms.setdefault(arm, {})[mesh] = d
print('arms loaded:', {a: sorted(v.keys()) for a, v in arms.items()})

# ------------------------------------------------------------------ Fig 1: M_nd and gray fraction vs NE (campaigns)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
ne = [NE(m) for m in MESHES9]
for key, lab, mk, col in [(P, 'Pedersen + linear mass, adaptive box (production)', 'o', 'C0'), (S, 'SIMP + eq.(4b), beta-stall ladder (historical)', 's', 'C3')]:
    ax[0].plot(ne, [fl(key[m]['Mnd']) for m in MESHES9], mk+'-', color=col, label=lab)
    ax[1].plot(ne, [fl(key[m]['gray_01_09']) for m in MESHES9], mk+'-', color=col, label=lab)
    ax[2].plot(ne, [fl(key[m]['mid_04_06']) for m in MESHES9], mk+'-', color=col, label=lab)
for a_, t in zip(ax, ['$M_{nd}$', 'fraction $0.1<\\rho<0.9$', 'fraction $0.4\\leq\\rho\\leq0.6$']):
    a_.set_xscale('log'); a_.set_xlabel('number of elements $N_E$'); a_.set_title(t); a_.grid(alpha=.3)
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(figs, 'fig1_campaign_trend.pdf')); fig.savefig(os.path.join(figs, 'fig1_campaign_trend.png')); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: histograms
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
edges = np.arange(0, 1.0001, 0.05); ctr = (edges[:-1] + edges[1:])/2
for k, (form, ttl) in enumerate([('P_pedersen_adaptive', 'Pedersen + linear mass, adaptive box'), ('S_simp_ladder', 'SIMP + eq.(4b), beta-stall ladder')]):
    for m, col in zip([(160,20),(400,50),(800,100)], ['C0','C2','C3']):
        h = np.array(V[(form, m[0], m[1])]['hist20'])
        ax[k].step(edges, np.r_[h, h[-1]], where='post', color=col, label='%dx%d' % m)
    ax[k].set_yscale('log'); ax[k].set_ylim(5e-4, 1); ax[k].set_xlabel(r'$\rho$'); ax[k].set_title(ttl); ax[k].grid(alpha=.3)
ax[0].set_ylabel('fraction of elements (bin width 0.05)'); ax[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig2_histograms.pdf')); fig.savefig(os.path.join(figs, 'fig2_histograms.png')); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: interface geometry
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
for key, lab, mk, col in [(P, 'Pedersen/adaptive', 'o', 'C0'), (S, 'SIMP/ladder', 's', 'C3')]:
    ax[0].plot(ne, [fl(key[m]['L_iso05']) for m in MESHES9], mk+'-', color=col, label=lab)
    ax[1].plot(ne, [fl(key[m]['w_gray_phys']) for m in MESHES9], mk+'-', color=col, label=lab)
    ax[2].plot(ne, [fl(key[m]['w_gray_el']) for m in MESHES9], mk+'-', color=col, label=lab)
ax[1].axhline(0.06, color='k', ls=':', lw=.8); ax[1].text(3500, 0.062, 'filter radius $R=0.06$', fontsize=6)
ax[2].plot(ne, [0.06*m[1] for m in MESHES9], 'k:', lw=.8, label='$R/h$')
for a_, t in zip(ax, ['length $L$ of the $\\rho=0.5$ contour', 'gray-band width $w=A_{gray}/L$ (physical)', 'gray-band width $w/h$ (elements)']):
    a_.set_xscale('log'); a_.set_xlabel('$N_E$'); a_.set_title(t); a_.grid(alpha=.3)
ax[2].set_yscale('log'); ax[0].legend(); ax[2].legend()
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig3_interface.pdf')); fig.savefig(os.path.join(figs, 'fig3_interface.png')); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: M_nd trajectories, Pedersen nine meshes
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
cm = plt.get_cmap('viridis')
for i, m in enumerate(MESHES9):
    T = readcsv(os.path.join(data, 'iter_P_%dx%d.csv' % m))
    k = [int(r['k']) for r in T]; Mnd = [fl(r['Mnd']) for r in T]; l2 = [fl(r['dxNorm2'])/fl(P[m]['eps']) for r in T]
    ax[0].plot(k, Mnd, color=cm(i/8), label='%dx%d' % m); ax[0].plot(k[-1], Mnd[-1], 'o', color=cm(i/8), ms=3)
    ax[1].plot(k, l2, color=cm(i/8))
ax[0].set_ylim(0, 0.6); ax[0].set_xlabel('outer iteration $k$'); ax[0].set_ylabel('$M_{nd}$'); ax[0].grid(alpha=.3); ax[0].legend(ncol=3, fontsize=5.5)
ax[1].set_yscale('log'); ax[1].axhline(1, color='k', ls=':'); ax[1].set_xlabel('outer iteration $k$'); ax[1].set_ylabel(r'$\|\Delta\rho\|_2/\varepsilon$'); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig4_pedersen_trajectories.pdf')); fig.savefig(os.path.join(figs, 'fig4_pedersen_trajectories.png')); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: SIMP ladder vs fixed move (recorded interventions)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
for j, (m, files) in enumerate([((160,20), ('iter_S_movestop_baseline_160x20.csv','iter_S_movestop_fixedmove_160x20.csv')),
                                ((320,40), ('iter_S_movestop_baseline_320x40.csv','iter_S_movestop_fixedmove_320x40.csv')),
                                ((400,50), ('iter_S_p400_400x50.csv','iter_S_f400_400x50.csv'))]):
    for f, lab, col, lw, z in zip(files[::-1], ['fixed move 0.04', 'production ladder'], ['C1', 'C3'], [2.2, 1.0], [1, 2]):
        T = readcsv(os.path.join(data, f))
        if 'Mnd_pct' in T[0]: k = [int(r['outer']) for r in T]; Mnd = [fl(r['Mnd_pct'])/100 for r in T]; mv = [fl(r['move']) for r in T]
        else: k = [int(r['k']) for r in T]; Mnd = [fl(r['Mnd']) for r in T]; mv = [fl(r['move']) for r in T]
        ax[j].plot(k, Mnd, color=col, label=lab, lw=lw, zorder=z); ax[j].plot(k[-1], Mnd[-1], 'o', color=col, ms=4, zorder=z)
        # mark move descents
        for i in range(1, len(mv)):
            if mv[i] < mv[i-1]: ax[j].axvline(k[i], color=col, ls=':', lw=.7)
    ax[j].set_title('SIMP + eq.(4b), %dx%d' % m); ax[j].set_xlabel('outer iteration $k$'); ax[j].set_ylim(0, 0.6); ax[j].grid(alpha=.3)
ax[0].set_ylabel('$M_{nd}$'); ax[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig5_simp_ladder_vs_fixed.pdf')); fig.savefig(os.path.join(figs, 'fig5_simp_ladder_vs_fixed.png')); plt.close(fig)

# ------------------------------------------------------------------ Fig 6: density fields (P and S at three meshes)
def rho_grid(path):
    d = sio.loadmat(path); r = d['rho'].ravel(); nx, ny = int(np.asarray(d['nelx']).ravel()[0]), int(np.asarray(d['nely']).ravel()[0])
    return r.reshape(nx, ny).T   # column-major e=(i-1)*nely+j -> (ny, nx)
fig, ax = plt.subplots(3, 2, figsize=(7.2, 2.9))
for j, form in enumerate(['P', 'S']):
    for i, m in enumerate([(160,20),(400,50),(800,100)]):
        R = rho_grid(os.path.join(data, 'rho_%s_%dx%d.mat' % (form, m[0], m[1])))
        ax[i, j].imshow(1-R, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, j].set_title('%s %dx%d   $M_{nd}$=%.3f' % ('Pedersen/adaptive' if form=='P' else 'SIMP/ladder', m[0], m[1], fl((P if form=='P' else S)[m]['Mnd'])), fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig6_fields.pdf')); fig.savefig(os.path.join(figs, 'fig6_fields.png')); plt.close(fig)

# ------------------------------------------------------------------ Fig 7: where the gray elements sit (P 160x20 vs 800x100)
fig, ax = plt.subplots(2, 1, figsize=(7.2, 2.4))
for i, m in enumerate([(160,20),(800,100)]):
    R = rho_grid(os.path.join(data, 'rho_P_%dx%d.mat' % m))
    cls = np.zeros(R.shape); cls[R >= 0.9] = 2; cls[(R > 0.1) & (R < 0.9)] = 1
    ax[i].imshow(cls, cmap=matplotlib.colors.ListedColormap(['white', 'orange', 'black']), interpolation='nearest', aspect='equal', vmin=0, vmax=2)
    ax[i].set_xticks([]); ax[i].set_yticks([]); ax[i].set_title('Pedersen/adaptive %dx%d: black $\\rho\\geq0.9$, orange $0.1<\\rho<0.9$, white $\\rho\\leq0.1$' % m, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig7_gray_location.pdf')); fig.savefig(os.path.join(figs, 'fig7_gray_location.png')); plt.close(fig)

# ------------------------------------------------------------------ Table: campaign metrics (LaTeX)
def tex_table(rows, cols, hdr, fmt, path, caption=None, colsep='3.5pt'):
    with open(path, 'w') as f:
        f.write('\\setlength{\\tabcolsep}{' + colsep + '}\n\\begin{tabular}{' + 'l' + 'r'*(len(cols)-1) + '}\n\\toprule\n' + ' & '.join(hdr) + ' \\\\\n\\midrule\n')
        for r in rows:
            cells = [fm % fl(r[c]) if fm != '%s' else str(r[c]) for c, fm in zip(cols, fmt)]
            f.write(' & '.join('--' if x.strip() == 'nan' else x for x in cells) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
for form, key in [('P', P), ('S', S)]:
    rows = []
    for m in MESHES9:
        r = dict(key[m]); r['mesh'] = '%dx%d' % m; r['rminEl'] = 0.06*m[1]; r['gap12pct'] = 100*fl(r['gap12'])
        r['Mnd100'] = 100*fl(r['Mnd']); r['gray100'] = 100*fl(r['gray_01_09']); r['mid100'] = 100*fl(r['mid_04_06'])
        r['wR'] = fl(r['w_gray_phys'])/0.06; r['band100'] = 100*fl(r['gray_within_R_both'])
        rows.append(r)
    tex_table(rows, ['mesh','rminEl','nOuter','omega1','gap12pct','Mnd100','gray100','mid100','L_iso05','w_gray_phys','w_gray_el','band100'],
              ['mesh','$R/h$','$k_{stop}$','$\\omega_1$','gap$_{12}$ [\\%]','$M_{nd}$ [\\%]','$f_{0.1-0.9}$ [\\%]','$f_{0.4-0.6}$ [\\%]','$L$','$w$','$w/h$','band [\\%]'],
              ['%s','%.1f','%.0f','%.2f','%.1f','%.2f','%.2f','%.2f','%.1f','%.4f','%.2f','%.0f'],
              os.path.join(figs, 'tab_campaign_%s.tex' % form))
print('campaign figures written')

# ================================================================== NEW ARMS (if present)
newf = os.path.join(data, 'new_run_metrics.csv')
if os.path.exists(newf):
    NR = readcsv(newf)
    ARM = {}
    for r in NR: ARM.setdefault(r['arm'], {})[(int(r['nelx']), int(r['nely']))] = r
    M4 = [(160,20),(240,30),(320,40),(400,50)]; M5 = M4 + [(480,60)]
    # ---- Fig 8: 2x2 factorial (material x controller) --------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    series = [(P, 'Pedersen material, adaptive box (production)', 'o-', 'C0', MESHES9),
              (ARM.get('simpAdaptive', {}), 'SIMP+eq.(4b) material, adaptive box', 'o--', 'C1', M5),
              (ARM.get('pedersenLadder', {}), 'Pedersen material, beta-stall ladder', 's--', 'C2', M5),
              (S, 'SIMP+eq.(4b) material, beta-stall ladder (historical)', 's-', 'C3', MESHES9)]
    for key, lab, st, col, ms in series:
        ms = [m for m in ms if m in key]
        if not ms: continue
        ax[0].plot([NE(m) for m in ms], [fl(key[m]['Mnd']) for m in ms], st, color=col, label=lab)
        ax[1].plot([NE(m) for m in ms], [fl(key[m]['w_gray_phys']) for m in ms], st, color=col, label=lab)
        ax[2].plot([NE(m) for m in ms], [fl(key[m]['nOuter']) for m in ms], st, color=col, label=lab)
    for a_, t in zip(ax, ['$M_{nd}$ at the stop', 'gray-band width $w$ (physical)', 'outer iterations at the stop']):
        a_.set_xscale('log'); a_.set_xlabel('$N_E$'); a_.set_title(t); a_.grid(alpha=.3)
    ax[0].set_ylim(0, 0.6); ax[1].axhline(0.06, color='k', ls=':', lw=.8)
    ax[0].legend(fontsize=5.5, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig8_factorial.pdf')); fig.savefig(os.path.join(figs, 'fig8_factorial.png')); plt.close(fig)
    # ---- Fig 9: filter arms -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    series = [(P, '$R=0.06$ physical (production)', 'o-', 'C0', MESHES9),
              (ARM.get('filterEl3', {}), '$r_{min}=3$ elements at every mesh', '^--', 'C4', M4),
              (ARM.get('R012', {}), '$R=0.12$ physical', 'v--', 'C5', M4)]
    for key, lab, st, col, ms in series:
        ms = [m for m in ms if m in key]
        if not ms: continue
        ax[0].plot([NE(m) for m in ms], [fl(key[m]['Mnd']) for m in ms], st, color=col, label=lab)
        ax[1].plot([NE(m) for m in ms], [fl(key[m]['gray_01_09']) for m in ms], st, color=col, label=lab)
        ax[2].plot([NE(m) for m in ms], [fl(key[m]['w_gray_phys']) for m in ms], st, color=col, label=lab)
    for a_, t in zip(ax, ['$M_{nd}$', 'fraction $0.1<\\rho<0.9$', 'gray-band width $w$ (physical)']):
        a_.set_xscale('log'); a_.set_xlabel('$N_E$'); a_.set_title(t); a_.grid(alpha=.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig9_filter_arms.pdf')); fig.savefig(os.path.join(figs, 'fig9_filter_arms.png')); plt.close(fig)
    # ---- Fig 10: fixed budget trajectories -------------------------------------------------
    if 'budget400' in ARM:
        BM = [m for m in MESHES9 if m in ARM['budget400']]
        fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
        for i, m in enumerate(BM):
            col = cm(MESHES9.index(m)/8)
            T = readcsv(os.path.join(data, 'iter_budget400_%dx%d.csv' % m))
            k = [int(r['k']) for r in T]; Mnd = [fl(r['Mnd']) for r in T]; w1 = [fl(r['omega1']) for r in T]
            l2 = [fl(r['dxNorm2'])/(0.05*math.sqrt(NE(m)/3200)) for r in T]
            ks = int(fl(P[m]['nOuter']))
            ax[0].plot(k, Mnd, color=col, label='%dx%d' % m); ax[0].plot(ks, Mnd[ks-1], 'o', color=col, ms=4)
            ax[1].plot(k, [100*(v/w1[ks-1]-1) for v in w1], color=col); ax[1].plot(ks, 0, 'o', color=col, ms=4)
            ax[2].plot(k, l2, color=col)
        ax[0].set_ylim(0.10, 0.30); ax[0].set_xlabel('outer iteration $k$'); ax[0].set_ylabel('$M_{nd}$'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=5.5, ncol=2)
        ax[1].set_ylim(-1.5, 0.5); ax[1].set_xlabel('outer iteration $k$'); ax[1].set_ylabel(r'$\omega_1/\omega_1(k_{stop})-1$ [%]'); ax[1].grid(alpha=.3)
        ax[2].set_yscale('log'); ax[2].axhline(1, color='k', ls=':'); ax[2].set_xlabel('outer iteration $k$'); ax[2].set_ylabel(r'$\|\Delta\rho\|_2/\varepsilon$'); ax[2].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig10_budget.pdf')); fig.savefig(os.path.join(figs, 'fig10_budget.png')); plt.close(fig)
        rows = []
        for m in BM:
            T = readcsv(os.path.join(data, 'iter_budget400_%dx%d.csv' % m)); Mnd = [fl(r['Mnd']) for r in T]; w1 = [fl(r['omega1']) for r in T]
            ks = int(fl(P[m]['nOuter'])); imin = int(np.argmin(Mnd))
            rows.append({'mesh': '%dx%d' % m, 'ks': ks, 'Ms': Mnd[ks-1], 'Mmin': Mnd[imin], 'kmin': imin+1, 'M400': fl(ARM['budget400'][m]['Mnd']),
                         'ws': w1[ks-1], 'wmax': max(w1), 'kwmax': int(np.argmax(w1))+1, 'w400': fl(ARM['budget400'][m]['omega1']),
                         'b400': fl(ARM['budget400'][m]['Mnd_bulk']), 'bs': fl(P[m]['Mnd_bulk'])})
        tex_table(rows, ['mesh','ks','Ms','Mmin','kmin','M400','bs','b400','wmax','kwmax','w400'],
                  ['mesh','$k_{stop}$','$M_{nd}(k_{stop})$','min $M_{nd}$','at $k$','$M_{nd}(400)$','bulk$(k_{stop})$','bulk$(400)$','max $\\omega_1$','at $k$','$\\omega_1(400)$'],
                  ['%s','%.0f','%.4f','%.4f','%.0f','%.4f','%.4f','%.4f','%.3f','%.0f','%.3f'], os.path.join(figs, 'tab_budget.tex'))
    # ---- Table: new arms ---------------------------------------------------------------------
    rows = []
    for arm in ['simpAdaptive', 'pedersenLadder', 'filterEl3', 'R012', 'budget400', 'box02', 'box005']:
        for m in M5 + [(640,80),(800,100)]:
            if arm in ARM and m in ARM[arm]:
                r = dict(ARM[arm][m]); r['armmesh'] = '%s %dx%d' % (arm, m[0], m[1]); r['gap12pct'] = 100*fl(r['gap12'])
                r['Mnd100'] = 100*fl(r['Mnd']); r['gray100'] = 100*fl(r['gray_01_09']); r['band100'] = 100*fl(r['gray_within_R_both'])
                r['stat'] = r['status'].replace('_', '\\_')
                if r.get('final_has_localized_mode', '0') == '1': r['stat'] += ' (loc.\\ mode)'
                rows.append(r)
    tex_table(rows, ['armmesh','rminEl','stat','nOuter','n_spikes_30pct','omega1','gap12pct','Mnd100','gray100','w_gray_phys','band100'],
              ['arm, mesh','$R/h$','status','$k_{stop}$','spikes','$\\omega_1$','gap$_{12}$ [\\%]','$M_{nd}$ [\\%]','$f_{0.1-0.9}$ [\\%]','$w$','band [\\%]'],
              ['%s','%.1f','%s','%.0f','%.0f','%.2f','%.1f','%.2f','%.2f','%.4f','%.0f'], os.path.join(figs, 'tab_new_arms.tex'), colsep='2.2pt')
    print('new-arm figures written')

# ================================================================== matched-iteration table (Pedersen campaign)
rows = []
for m in MESHES9:
    T = readcsv(os.path.join(data, 'iter_P_%dx%d.csv' % m)); Mnd = [fl(r['Mnd']) for r in T]
    def at(k): return Mnd[k-1] if k <= len(Mnd) else float('nan')
    k20 = next((i+1 for i, v in enumerate(Mnd) if v < 0.20), float('nan'))
    k15 = next((i+1 for i, v in enumerate(Mnd) if v < 0.15), float('nan'))
    rows.append({'mesh': '%dx%d' % m, 'k50': at(50), 'k93': at(93), 'kstop': len(Mnd), 'Mstop': Mnd[-1], 'k20': k20, 'k15': k15, 'h': 1.0/m[1]})
tex_table(rows, ['mesh','k50','k93','k20','k15','kstop','Mstop'],
          ['mesh','$M_{nd}(k{=}50)$','$M_{nd}(k{=}93)$','$k: M_{nd}<0.20$','$k: M_{nd}<0.15$','$k_{stop}$','$M_{nd}(k_{stop})$'],
          ['%s','%.3f','%.3f','%.0f','%.0f','%.0f','%.3f'], os.path.join(figs, 'tab_matched_k.tex'))
kk = np.array([r['k20'] for r in rows], float); hh = np.array([r['h'] for r in rows], float)
sl, ic = np.polyfit(np.log(1/hh), np.log(kk), 1)
with open(os.path.join(data, 'clearing_time_fit.txt'), 'w') as f: f.write('k(Mnd<0.20) ~ C * (1/h)^p : p = %.3f, C = %.4f, over nine P meshes\n' % (sl, math.exp(ic)))
print('clearing-time exponent vs 1/h: %.3f' % sl)
for r in rows: print('P %s: Mnd(50)=%.3f Mnd(93)=%.3f k<0.20=%s k<0.15=%s kstop=%d' % (r['mesh'], r['k50'], r['k93'], r['k20'], r['k15'], r['kstop']))

# ================================================================== Fig 11: decomposition of M_nd (band / bulk / tails)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
x = np.arange(len(MESHES9))
for j, (key, ttl) in enumerate([(P, 'Pedersen/adaptive (production)'), (S, 'SIMP/ladder (historical)')]):
    band = np.array([fl(key[m]['Mnd_band']) for m in MESHES9]); bulk = np.array([fl(key[m]['Mnd_bulk']) for m in MESHES9]); tails = np.array([fl(key[m]['Mnd_tails']) for m in MESHES9])
    ax[j].bar(x, band, color='C0', label='interface band (within $R$ of both phases)')
    ax[j].bar(x, bulk, bottom=band, color='C1', label='bulk gray (not within $R$ of both phases)')
    ax[j].bar(x, tails, bottom=band+bulk, color='0.7', label='near-mode tails ($\\rho\\leq0.1$ or $\\geq0.9$)')
    ax[j].set_xticks(x); ax[j].set_xticklabels(['%dx%d' % m for m in MESHES9], rotation=60, fontsize=6); ax[j].set_title(ttl); ax[j].grid(alpha=.3, axis='y')
ax[0].set_ylabel('contribution to $M_{nd}$'); ax[0].legend(fontsize=5.5, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig11_decomposition.pdf')); fig.savefig(os.path.join(figs, 'fig11_decomposition.png')); plt.close(fig)
# decomposition table
rows = []
for m in MESHES9:
    rows.append({'mesh': '%dx%d' % m, 'Pb': fl(P[m]['Mnd_band']), 'Pu': fl(P[m]['Mnd_bulk']), 'Pt': fl(P[m]['Mnd_tails']), 'PM': fl(P[m]['Mnd']),
                 'Sb': fl(S[m]['Mnd_band']), 'Su': fl(S[m]['Mnd_bulk']), 'St': fl(S[m]['Mnd_tails']), 'SM': fl(S[m]['Mnd'])})
tex_table(rows, ['mesh','Pb','Pu','Pt','PM','Sb','Su','St','SM'], ['mesh','P band','P bulk','P tails','P $\\Mnd$','S band','S bulk','S tails','S $\\Mnd$'],
          ['%s','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f'], os.path.join(figs, 'tab_decomposition.tex'))
print('decomposition written')


# ================================================================== Fig 12: box-ceiling arms (step-bound test)
if os.path.exists(newf) and ('box02' in ARM or 'box005' in ARM):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    rows = []
    for j, m in enumerate([(160,20),(320,40),(400,50)]):
        for arm, box, col in [('box005', 0.05, 'C4'), ('P', 0.10, 'C0'), ('box02', 0.20, 'C3')]:
            f = os.path.join(data, 'iter_%s_%dx%d.csv' % (arm, m[0], m[1]))
            if not os.path.exists(f): continue
            T = readcsv(f); k = [int(r['k']) for r in T]; Mnd = [fl(r['Mnd']) for r in T]
            ax[j].plot(k, Mnd, color=col, label='box ceiling %.2f' % box); ax[j].plot(k[-1], Mnd[-1], 'o', color=col, ms=3)
            k20 = next((i+1 for i, v in enumerate(Mnd) if v < 0.20), float('nan')); k15 = next((i+1 for i, v in enumerate(Mnd) if v < 0.15), float('nan'))
            rows.append({'mesh': '%dx%d' % m, 'box': box, 'k20': k20, 'k15': k15, 'kstop': len(Mnd), 'Mend': Mnd[-1],
                         'w': fl((P if arm == 'P' else ARM[arm])[m]['w_gray_phys']), 'w1': fl((P if arm == 'P' else ARM[arm])[m]['omega1'])})
        ax[j].set_title('Pedersen/adaptive %dx%d' % m); ax[j].set_xlabel('outer iteration $k$'); ax[j].set_ylim(0, 0.6); ax[j].set_xlim(0, 240); ax[j].grid(alpha=.3)
    ax[0].set_ylabel('$M_{nd}$'); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'fig12_box.pdf')); fig.savefig(os.path.join(figs, 'fig12_box.png')); plt.close(fig)
    tex_table(rows, ['mesh','box','k20','k15','kstop','Mend','w','w1'], ['mesh','box ceiling','$k: M_{nd}<0.20$','$k: M_{nd}<0.15$','$k_{stop}$','$M_{nd}(k_{stop})$','$w$','$\\omega_1$'],
              ['%s','%.2f','%.0f','%.0f','%.0f','%.3f','%.4f','%.2f'], os.path.join(figs, 'tab_box.tex'))
    print('box figure written')

# ================================================================== Table: coreless-member classification
cf = os.path.join(data, 'bulk_gray_classification.csv')
if os.path.exists(cf):
    C = readcsv(cf); rows = []
    lab = {'P': 'P, $R=0.06$', 'R012': '$R=0.12$', 'filterEl3': '$r_{min}=3$ el', 'budget400': 'P, stop off'}
    for r in C:
        if r['arm'] not in ('P', 'R012', 'budget400'): continue
        if r['arm'] == 'budget400' and r['nelx'] not in ('640', '800'): continue
        rr = dict(r); rr['name'] = '%s %sx%s' % (lab[r['arm']], r['nelx'], r['nely'])
        rr['wR'] = r['coreless_width_median_over_R']
        rows.append(rr)
    tex_table(rows, ['name','R_over_h','Mnd','Mnd_bulk','Mnd_coreless','Mnd_plateau','n_coreless_components','wR'],
              ['design','$R/h$','$M_{nd}$','bulk','coreless members','gray plateau','members','median width$/R$'],
              ['%s','%.1f','%.4f','%.4f','%.4f','%.4f','%.0f','%.1f'], os.path.join(figs, 'tab_coreless.tex'))
    print('coreless table written')
