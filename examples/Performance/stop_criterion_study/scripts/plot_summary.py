import sys, os, csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT, FIG = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(OUT, 'replay_all.csv'))))
MESHES = ['160x20','240x30','320x40','400x50','480x60','560x70','640x80','720x90','800x100']
RULES = [('max|dx|<=0.01', 'max|dx| <= 0.01 (Proposed/Yuksel production)'),
         ('rms<8.8e-4 (Olh c=.05)', 'RMS(dx) < 8.8e-4  (= Olhoff production, c=0.05)'),
         ('relL2<0.001', '||dx||/||x|| < 1e-3'),
         ('max|dx|<=0.04', 'max|dx| <= 0.04 (current driver)'),
         ('relL2<0.005', '||dx||/||x|| < 5e-3'),
         ('max|dx|<=0.04 W10', 'max|dx| <= 0.04 for 10 consecutive it.'),
         ('obj<1e-3 & Mnd<5e-3 W10', 'objective range <1e-3 AND M_nd range <5e-3 over 10 it. (proposed common rule)')]
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
for row, method in enumerate(['proposed','yuksel','olhoff']):
    axk, axl = axes[row]
    x = np.arange(len(MESHES)); width = 0.11
    kst = {}
    for r in rows:
        if r['method']==method: kst[r['mesh']] = int(float(r['kstag']))
    for j, (rule, label) in enumerate(RULES):
        ks, ls = [], []
        for m in MESHES:
            rr = [r for r in rows if r['method']==method and r['mesh']==m and r['rule']==rule]
            if not rr or rr[0]['k']=='': ks.append(np.nan); ls.append(np.nan)
            else: ks.append(float(rr[0]['k'])); ls.append(float(rr[0]['w1_loss_pct']))
        axk.bar(x + (j-3)*width, ks, width, label=label)
        axl.bar(x + (j-3)*width, ls, width)
    axk.plot(x, [kst.get(m, np.nan) for m in MESHES], 'k_', ms=14, mew=2, label='k_stag (omega1 within 0.5% and M_nd within 0.01 of the 300/600-iteration end state)')
    axk.set_xticks(x); axk.set_xticklabels(MESHES, rotation=30); axl.set_xticks(x); axl.set_xticklabels(MESHES, rotation=30)
    axk.set_ylabel('stop iteration k*'); axl.set_ylabel('omega1(k*) loss vs end state [%]')
    axk.set_title(f'{method}: stop iteration (missing bar = never fired within the record)'); axl.set_title(f'{method}: omega1 loss at the stop (negative = better than the end state)')
    axl.axhline(0, color='k', lw=0.5); axl.axhline(0.5, color='r', lw=0.5, ls='--')
    if row == 0: handles, labels = axk.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, 'stop_rules_summary.png'), dpi=130)
print('ok')
