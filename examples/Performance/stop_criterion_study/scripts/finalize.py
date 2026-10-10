"""Fill the report template from replay_all.csv and tables.md; copy the study into the repo."""
import sys, os, csv, shutil, glob, numpy as np
S = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(S, 'out'); DEST = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(OUT, 'replay_all.csv'))))
r1 = [r for r in rows if r['rule'] == 'obj<1e-3 & Mnd<5e-3 W10']
hit = [r for r in r1 if r['k'] != '']
ratios = [float(r['ratio']) for r in hit]; losses = [float(r['w1_loss_pct']) for r in hit]
n_hist = len(set((r['method'], r['mesh']) for r in rows))
worst = max(hit, key=lambda r: float(r['w1_loss_pct']))
per_method = {}
for m in ['proposed','yuksel','olhoff']:
    h = [r for r in hit if r['method']==m]
    per_method[m] = (min(float(r['ratio']) for r in h), max(float(r['ratio']) for r in h), min(float(r['w1_loss_pct']) for r in h), max(float(r['w1_loss_pct']) for r in h), len(h), len([r for r in r1 if r['method']==m]))
summary = (f"R1 fired on all {len(hit)} of {len(r1)} histories. Stop iteration relative to k_stag: "
           f"{per_method['proposed'][0]:.2f}–{per_method['proposed'][1]:.2f} (Proposed), {per_method['yuksel'][0]:.2f}–{per_method['yuksel'][1]:.2f} (Yuksel stage 2), {per_method['olhoff'][0]:.2f}–{per_method['olhoff'][1]:.2f} (Du–Olhoff); "
           f"ω₁ at the stop relative to the end state: {per_method['proposed'][2]:+.2f} … {per_method['proposed'][3]:+.2f} % (Proposed), {per_method['yuksel'][2]:+.2f} … {per_method['yuksel'][3]:+.2f} % (Yuksel), {per_method['olhoff'][2]:+.2f} … {per_method['olhoff'][3]:+.2f} % (Du–Olhoff); "
           f"the worst case is {worst['method']} {worst['mesh']} (+{float(worst['w1_loss_pct']):.2f} %). M_nd at the stop exceeds its end value by at most {max(float(r['dMnd']) for r in hit):.3f}. Per-mesh numbers: Table C.")
tpl = open(os.path.join(S, 'REPORT_template.md')).read()
tpl = tpl.replace('{{N_HIST}}', str(n_hist)).replace('{{R1_RATIO}}', f"{min(ratios):.2f}–{max(ratios):.2f}×").replace('{{R1_LOSS}}', f"{max(losses):.2f}")
tpl = tpl.replace('{{R1_SUMMARY}}', summary).replace('{{TABLES}}', '## Appendix: tables from the replay\n\n' + open(os.path.join(OUT, 'tables.md')).read())
os.makedirs(DEST, exist_ok=True)
for sub in ['fig', 'data', 'scripts']: os.makedirs(os.path.join(DEST, sub), exist_ok=True)
open(os.path.join(DEST, 'REPORT.md'), 'w').write(tpl)
for f in glob.glob(os.path.join(S, 'fig', '*.png')): shutil.copy(f, os.path.join(DEST, 'fig'))
for f in glob.glob(os.path.join(OUT, '*.csv')) + [os.path.join(OUT, 'tables.md'), os.path.join(OUT, 'replay_log.txt')]: shutil.copy(f, os.path.join(DEST, 'data'))
for f in ['stopstudy_metrics.m','stopstudy_proposed.m','stopstudy_yuksel.m','stopstudy_olhoff.m','replay.py','make_tables.py','plots.py','plot_summary.py','finalize.py']:
    shutil.copy(os.path.join(S, f), os.path.join(DEST, 'scripts'))
print('wrote', DEST); print(summary)
