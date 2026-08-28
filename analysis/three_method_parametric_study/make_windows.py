#!/usr/bin/env python3
"""WP22 -- trajectory-window cost decomposition.

Per-iteration cost is not an intrinsic constant of a method.  This writes the
measured optimization-loop seconds spent in each 50-iteration window of every
Stage A trajectory, plus the eigensolve share where the telemetry records it.
"""
import csv, os
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
WIN = 50

def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return float('nan')

rows = defaultdict(list)
with open(os.path.join(RES, 'objective_traces.csv')) as fh:
    for t in csv.DictReader(fh):
        rows[(t['run_id'], t['pass'], t['method'])].append(t)

out = []
for (run, pas, meth), tr in rows.items():
    tr.sort(key=lambda t: int(t['iter']))
    prev_t = prev_e = 0.0
    for lo in range(0, len(tr), WIN):
        blk = tr[lo:lo + WIN]
        if not blk:
            continue
        t_end = fnum(blk[-1]['loop_time_s'])
        e_end = fnum(blk[-1]['eig_time_cum_s'])
        dt = t_end - prev_t
        de = (e_end - prev_e) if e_end == e_end else float('nan')
        out.append({
            'run_id': run, 'pass': pas, 'method': meth,
            'window_start': int(blk[0]['iter']), 'window_end': int(blk[-1]['iter']),
            'n_in_window': len(blk),
            'loop_time_s': round(dt, 6),
            'loop_time_per_iter_s': round(dt / len(blk), 8),
            'eig_time_s': '' if de != de else round(de, 6),
            'eig_share': '' if de != de or dt <= 0 else round(de / dt, 5),
            'stage_span': f"{blk[0]['stage']}-{blk[-1]['stage']}",
        })
        prev_t = t_end
        if e_end == e_end:
            prev_e = e_end

out.sort(key=lambda r: (r['method'], r['run_id'], r['window_start']))
with open(os.path.join(RES, 'trajectory_window_costs.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(f'wrote trajectory_window_costs.csv ({len(out)} windows)')

# Headline check: does per-iteration cost drift within a trajectory?
by_run = defaultdict(list)
for r in out:
    by_run[(r['method'], r['run_id'], r['pass'])].append(r)
print(f"\n{'method':9s} {'run_id':22s} {'it1-50':>9s} {'last win':>9s} {'ratio':>7s} {'eig sh 1':>9s} {'eig sh N':>9s}")
for (meth, run, pas), ws in sorted(by_run.items()):
    if pas == 'stage_a' and meth != 'Olhoff':
        continue
    a, b = ws[0], ws[-1]
    ratio = b['loop_time_per_iter_s'] / a['loop_time_per_iter_s'] if a['loop_time_per_iter_s'] else float('nan')
    print(f"{meth:9s} {run:22s} {a['loop_time_per_iter_s']*1000:8.2f}m {b['loop_time_per_iter_s']*1000:8.2f}m "
          f"{ratio:7.2f} {str(a['eig_share']):>9s} {str(b['eig_share']):>9s}")
