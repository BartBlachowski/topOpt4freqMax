"""Rebuild results/r3/w2_timing_runs.csv and w2_timing.csv from w2_timing.log
(the first W2 run finished all 90 timed runs but failed while writing the CSV)."""
import re, statistics, os
d = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'r3')
pat = re.compile(r'\s*(\d+)x(\d+)\s+(\w+)\s+run (\d)\s+total\s+([\d.]+) s \(solver\s+([\d.]+)\)\s+it (\S+)\s+omega ([\d.]+) ([\d.]+)')
paper = {'proposed': [1.13,1.52,2.50,3.75,5.69,7.50,10.92,20.15,28.22],
         'yuksel':   [1.35,2.08,3.31,5.52,8.06,11.86,17.77,31.18,39.36]}
nx = [160,240,320,400,480,560,640,720,800]
runs = [m.groups() for m in map(pat.match, open(os.path.join(d,'w2_timing.log'))) if m]
with open(os.path.join(d,'w2_timing_runs.csv'),'w') as f:
    f.write('mesh,method,run,total_s,solver_total_s,iterations,omega1,omega2\n')
    for r in runs: f.write(f'{r[0]}x{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]}\n')
groups = {}
for r in runs: groups.setdefault((int(r[0]),int(r[1]),r[2]),[]).append(r)
with open(os.path.join(d,'w2_timing.csv'),'w') as f:
    f.write('mesh,Ne,method,median_s,min_s,max_s,median_solver_s,iterations,omega1,omega2,table2_total_s,diff_pct,ratio_median_yuksel_over_proposed\n')
    for (a,b,m),g in groups.items():
        t=[float(x[4]) for x in g]; ts=[float(x[5]) for x in g]; pv=paper[m][nx.index(a)]
        med=statistics.median(t)
        other=[float(x[4]) for x in groups[(a,b,'yuksel' if m=='proposed' else 'proposed')]]
        ratio = statistics.median(t)/statistics.median(other) if m=='yuksel' else statistics.median(other)/med
        f.write(f'{a}x{b},{a*b},{m},{med:.3f},{min(t):.3f},{max(t):.3f},{statistics.median(ts):.3f},{g[-1][6]},{g[-1][7]},{g[-1][8]},{pv},{100*(med/pv-1):.1f},{ratio:.3f}\n')
