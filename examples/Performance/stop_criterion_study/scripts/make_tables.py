"""Emit markdown tables for the report from the recorded histories."""
import sys, os, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay
OUT = sys.argv[1]; replay.OUT = OUT
MESHES = replay.MESHES
out = []
# ---- Table A: production/current rules and the stagnation iteration, per method/mesh
out.append('### Table A. Where the current rules stop, versus where the design has actually stagnated\n')
out.append('k_stag = first iteration after which omega_1 stays within 0.5 % and M_nd within 0.01 of the end of the extended record (300 iterations for Proposed and Du-Olhoff, stage-2 cap 600 for Yuksel). Entries: stop iteration (omega_1 loss vs end state, %). "none" = the rule never fired within the record.\n')
rules = [('max|dx|<=0.01','max\\|dx\\| <= 0.01'),('rms<8.8e-4 (Olh c=.05)','RMS(dx) < 8.8e-4 (= 0.05*sqrt(n_e/3200))'),('relL2<0.001','\\|\\|dx\\|\\|/\\|\\|x\\|\\| < 1e-3'),('max|dx|<=0.04','max\\|dx\\| <= 0.04'),('rms<3.5e-3 (c=.2)','RMS(dx) < 3.5e-3 (c = 0.2)'),('relL2<0.005','\\|\\|dx\\|\\|/\\|\\|x\\|\\| < 5e-3')]
allrows = {m: replay.evaluate(m) for m in ['proposed','yuksel','olhoff']}
def cell(rs, mesh, rule):
    r = [x for x in rs if x['mesh']==mesh and x['rule']==rule]
    if not r: return '–'
    r = r[0]
    return 'none' if r['k'] is None else f"{r['k']} ({r['w1_loss_pct']:+.2f})"
for method, label in [('proposed','Proposed'),('yuksel','Yuksel (stage 2)'),('olhoff','Du-Olhoff')]:
    rs = allrows[method]
    if not rs: continue
    meshes = [m for m in MESHES if any(x['mesh']==m for x in rs)]
    out.append(f'\n**{label}**\n')
    out.append('| mesh | record | k_stag | ' + ' | '.join(l for _, l in rules) + ' |')
    out.append('|---|---|---|' + '---|'*len(rules))
    for m in meshes:
        r0 = [x for x in rs if x['mesh']==m][0]
        out.append(f"| {m} | {r0['n']} | {r0['kstag']} | " + ' | '.join(cell(rs, m, k) for k, _ in rules) + ' |')
# ---- Table B: metric level at stagnation
out.append('\n### Table B. Level of the per-iteration change metrics once the design has stagnated\n')
out.append('Median over the 20 iterations after k_stag (left) and over the last 20 recorded iterations (right). These are the values a threshold must exceed to stop at stagnation, and the floor it must stay above to stop at all.\n')
out.append('| method | mesh | k_stag | max\\|dx\\| @stag | rel-L2 @stag | RMS @stag | max\\|dx\\| end | rel-L2 end | RMS end | omega_1 @stag | omega_1 max | omega_1 end | M_nd @stag | M_nd end |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for method in ['proposed','yuksel','olhoff']:
    for mesh in MESHES:
        T = replay.load(method, mesh)
        if T is None: continue
        ks = replay.stagnation(T); n = T['n']; a = slice(ks, min(ks+20, n)); b = slice(max(0, n-20), n)
        med = lambda key, sl: float(np.nanmedian(T[key][sl]))
        out.append(f"| {method} | {mesh} | {ks} | {med('dmax_native',a):.3f} | {med('drel',a):.4f} | {med('drms',a):.4f} | {med('dmax_native',b):.3f} | {med('drel',b):.4f} | {med('drms',b):.4f} | {T['w1'][ks]:.1f} | {np.nanmax(T['w1']):.1f} | {T['w1'][-1]:.1f} | {T['Mnd'][ks]:.3f} | {T['Mnd'][-1]:.3f} |")
# ---- Table C: candidate common rules
out.append('\n### Table C. Candidate common rules replayed on every recorded history\n')
out.append('Entries: stop iteration (omega_1 loss vs end state, %). Yuksel: rule applied to stage 2 only (stage-1 handoff kept native).\n')
cand = [('obj<1e-3 & Mnd<5e-3 W10','R1: objective range < 1e-3 AND M_nd range < 5e-3 over 10 it.'),('obj<5e-4 & Mnd<5e-3 W10','R1 tight: 5e-4 / 5e-3 / 10 it.'),('obj rel W10 <1e-3','objective range < 1e-3 over 10 it. only'),('max|dx|<=0.04 W10','R2: max\\|dx\\| <= 0.04 for 10 consecutive it.'),('max|dx|<=0.025','max\\|dx\\| <= 0.025 (single it.)'),('Mnd abs W10 <5e-3','M_nd range < 5e-3 over 10 it. only')]
for method, label in [('proposed','Proposed'),('yuksel','Yuksel (stage 2)'),('olhoff','Du-Olhoff')]:
    rs = allrows[method]
    if not rs: continue
    meshes = [m for m in MESHES if any(x['mesh']==m for x in rs)]
    out.append(f'\n**{label}**\n')
    out.append('| mesh | k_stag | ' + ' | '.join(l for _, l in cand) + ' |')
    out.append('|---|---|' + '---|'*len(cand))
    for m in meshes:
        r0 = [x for x in rs if x['mesh']==m][0]
        out.append(f"| {m} | {r0['kstag']} | " + ' | '.join(cell(rs, m, k) for k, _ in cand) + ' |')
# ---- Table D: summary
out.append('\n### Table D. Summary over all recorded histories (methods x meshes)\n')
out.append('worst loss = largest omega_1 deficit at the stop relative to the end state; ratio = k*/k_stag (1 = stops exactly at stagnation, <1 = before, >1 = after).\n')
out.append('| rule | histories | never fired | worst omega_1 loss [%] | worst M_nd excess | k*/k_stag min / median / max |')
out.append('|---|---|---|---|---|---|')
flat = [r for rs in allrows.values() for r in rs]
summ = []
for name in replay.RULES:
    d = [r for r in flat if r['rule']==name]; hit = [r for r in d if r['k'] is not None]
    if not hit: continue
    rat = [r['ratio'] for r in hit]
    summ.append((max(r['w1_loss_pct'] for r in hit), name, len(d), len(d)-len(hit), max(r['dMnd'] for r in hit), min(rat), float(np.median(rat)), max(rat)))
for s in sorted(summ):
    out.append(f"| {s[1].replace(chr(124), chr(92)+chr(124))} | {s[2]} | {s[3]} | {s[0]:.2f} | {s[4]:+.3f} | {s[5]:.2f} / {s[6]:.2f} / {s[7]:.2f} |")
open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
