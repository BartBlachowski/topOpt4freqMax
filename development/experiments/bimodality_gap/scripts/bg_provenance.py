#!/usr/bin/env python3
"""Provenance table of every new run (from runs/*.json) -> figures/tab_provenance.tex, data/RUN_INDEX.json."""
import json, glob, os, subprocess, hashlib
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); repo = os.path.dirname(os.path.dirname(root))
runs = sorted(glob.glob(os.path.join(root, 'runs', 'BG_*.json')))
idx = []
for p in runs:
    J = json.load(open(p))
    mat = p[:-5] + '.mat'
    sha = hashlib.sha256(open(mat, 'rb').read()).hexdigest() if os.path.exists(mat) else ''
    idx.append({'name': J['name'], 'arm': J['arm'], 'mesh': J['mesh'], 'upstreamPreset': J['upstreamPreset'], 'overrides': J['overrides'],
                'cfgHash': J['cfgHash'], 'rminEl': J['rminEl'], 'eps': J['eps'], 'status': J['status'], 'nOuter': J['nOuter'],
                'omega1': J['omega1'], 'omega2': J['omega2'], 'Mnd_final': J['Mnd_final'], 'gray_final': J['gray_final'],
                'wall_s': J['wall_s'], 'repo_head': J['repo_head'], 'impl_tree_sha256': J['impl_tree_sha256'], 'matlab': J['matlab_version'],
                'threads': J['threads'], 'timestamp': J['timestamp'], 'mat_sha256': sha})
json.dump(idx, open(os.path.join(root, 'data', 'RUN_INDEX.json'), 'w'), indent=1)
order = ['simpAdaptive', 'pedersenLadder', 'filterEl3', 'R012', 'budget400', 'box02', 'box005']
idx.sort(key=lambda r: (order.index(r['arm']) if r['arm'] in order else 99, r['mesh'][0]))
with open(os.path.join(root, 'figures', 'tab_provenance.tex'), 'w') as f:
    f.write('\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llp{6.2cm}lrrrl}\n\\toprule\narm & mesh & overrides on the upstream preset & cfg hash & $k$ & $\\omega_1$ & wall [s] & status \\\\\n\\midrule\n')
    for r in idx:
        ov = ', '.join('%s=%s' % (r['overrides'][i], r['overrides'][i+1]) for i in range(0, len(r['overrides']), 2)) if isinstance(r['overrides'], list) else str(r['overrides'])
        ov = ov.replace('material.', '').replace('filter.', '').replace('stop.', '').replace('_', '\\_').replace('[]', '[\\,]')
        f.write('%s & %dx%d & \\code{%s} & \\code{%s} & %d & %.3f & %.0f & %s \\\\\n' % (r['arm'], r['mesh'][0], r['mesh'][1], ov, r['cfgHash'][:12], r['nOuter'], r['omega1'], r['wall_s'], r['status'].replace('_', '\\_')))
    f.write('\\bottomrule\n\\end{tabular}\n')
heads = sorted(set(r['repo_head'] for r in idx)); impls = sorted(set(r['impl_tree_sha256'] for r in idx))
print('runs:', len(idx), 'repo heads:', heads, 'impl trees:', impls)
