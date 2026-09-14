#!/usr/bin/env python3
"""cv_report -- emit CAUSAL_ANALYSIS.md and METRICS.json from evidence/analysis.json.

Pure formatting of already-computed evidence.  Recomputes nothing, decides
nothing: every number it prints comes from analysis.json, which cv_analyze.py
produced from the tracked per-iteration CSVs and the frozen baselines.
"""
import json, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
KEYS = ['m160', 'm320', 'm400']
LBL = {'m160': '160x20', 'm320': '320x40', 'm400': '400x50'}


def f(x, n=4, dash='n/a'):
    if x is None:
        return dash
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int,)) or (isinstance(x, float) and x == int(x) and abs(x) < 1e6):
        return str(int(x))
    return f'{x:.{n}f}'


def main():
    A = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
    present = [k for k in KEYS if k in A['mesh']]
    L = []
    w = L.append
    w('# CAUSAL_ANALYSIS — production vs the frozen two-branch controller\n')
    w('Generated from `evidence/analysis.json`; every number traces to a tracked')
    w('per-iteration CSV or the frozen `evidence/baselines.json`.\n')
    w(f"Preregistration SHA-256 `{A['preregistration_sha256']}`.\n")
    w(f"Runs executed: **{', '.join(A['runsExecuted'])}**"
      + (f"; missing: {', '.join(A['runsMissing'])}" if A['runsMissing'] else '') + '\n')

    # ---- headline table
    w('\n## 1. Headline causal comparison\n')
    hdr = '| quantity | ' + ' | '.join(
        f'{LBL[k]} prod → cand' for k in present) + ' |'
    w(hdr); w('|---|' + '---|' * len(present))
    rows = [
        ('terminal status', lambda m: f"{m['prod']['status']} → **{m['cand']['status']}**"),
        ('outer iterations', lambda m: f"{m['prod']['nOuter']} → {m['cand']['nOuter']}"
                                       f"  (×{m['delta']['outer_mult']:.2f})"),
        ('inner MMA total', lambda m: f"{m['prod']['innerTotal']} → {m['cand']['innerTotal']}"
                                      f"  (×{m['delta']['inner_mult']:.2f})"),
        ('wall s', lambda m: f"{m['prod']['wall_s']:.1f} → {m['cand']['wall_s']:.1f}"
                             f"  (×{m['delta']['wall_mult']:.2f})"),
        ('first descent iter', lambda m: f"{m['delta']['firstDescent_prod']} → "
                                         f"{m['delta']['firstDescent_cand']}"
                                         f"  ({m['delta']['descentDelay']:+d} later)"),
        ('final move', lambda m: f"{m['prod']['move_final']} → {m['cand']['move_final']}"),
        ('M_nd %', lambda m: f"{m['prod']['Mnd']:.4f} → **{m['cand']['Mnd_final']:.4f}**"
                             f"  ({m['delta']['Mnd_rel_pct']:+.2f} %)"),
        ('omega1', lambda m: f"{m['prod']['omega1']:.6f} → **{m['cand']['omega1']:.6f}**"
                             f"  ({m['delta']['omega1_rel_pct']:+.3f} %)"),
        ('omega2', lambda m: f"{m['prod']['omega2']:.4f} → {m['cand']['omega2']:.4f}"),
        ('gap12', lambda m: f"{m['prod']['gap12']:.5f} → {m['cand']['gap12']:.5f}"),
        ('gray frac', lambda m: f"{m['prod']['gray']:.6f} → {m['cand']['gray_final']:.6f}"),
        ('mid frac', lambda m: f"{m['prod']['mid']:.6f} → {m['cand']['mid_final']:.6f}"),
        ('volume', lambda m: f"{m['prod']['volume']:.9f} → {m['cand']['volume_final']:.9f}"),
    ]
    for name, fn in rows:
        cells = []
        for k in present:
            try:
                cells.append(fn(A['mesh'][k]))
            except Exception:
                cells.append('n/a')
        w(f'| {name} | ' + ' | '.join(cells) + ' |')

    # ---- transition audit
    w('\n## 2. Transition audit (Phase 13)\n')
    w('No transition may occur without the frozen `E = A OR B`; none may occur')
    w('because β stalls. Every row below carries its declaring counter at 20.\n')
    w('| mesh | iter | move | branch | decl | window | nA | nB | β stalled? | prod would be at | ω₁ | M_nd | gray | vol |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k in present:
        m = A['mesh'][k]
        for t in m['transitions']:
            w(f"| {LBL[k]} | {t['iter']} | {t['moveBefore']} → {t['moveAfter']} | "
              f"**{t['branch']}** | {t['declIter']} | {t['declBegin']}–{t['declIter']} | "
              f"{t['nA']} | {t['nB']} | {'yes' if t['betaStall'] else 'no'} | "
              f"stage {t['prodStageShadow']} | {t['omega1']:.4f} | {t['Mnd']:.4f} | "
              f"{t['gray']:.4f} | {t['volume']:.6f} |")

    # ---- termination audit
    w('\n## 3. Termination audit (Phase 14)\n')
    w('A candidate may report `CONVERGED` only at `move = 0.005` after the frozen')
    w('terminal persistence.\n')
    w('| mesh | status | iter | move | A | B | E | branch | nB | amp | tol | med₂₀cos | ‖Δρ‖₂ | honest | genuine |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k in present:
        t = A['mesh'][k]['term']
        w(f"| {LBL[k]} | **{t['status']}** | {t['iter']} | {t['move']} | {f(t['A'])} | "
          f"{f(t['B'])} | {f(t['E'])} | {t['branch'] or '—'} | {f(t['nB'])} | "
          f"{A['mesh'][k]['terminalStage']['amp_median']:.5f} | {A['mesh'][k]['tol']:.4f} | "
          f"{A['mesh'][k]['terminalStage']['medcos_median']:.4f} | {t['l2Drho']:.5f} | "
          f"{'yes' if A['mesh'][k]['convergenceHonest'] else 'NO'} | "
          f"{'yes' if A['mesh'][k]['genuineTerminalExhaustion'] else 'no'} |")

    # ---- gates
    w('\n## 4. Preregistered promotion gates (Phase 20)\n')
    w('| gate | requirement (abbreviated) | result |')
    w('|---|---|---|')
    desc = json.load(open(os.path.join(STUDY, 'evidence', 'gate_text.json'))) \
        if os.path.isfile(os.path.join(STUDY, 'evidence', 'gate_text.json')) else {}
    for g, v in A['gates'].items():
        w(f"| **{g}** | {desc.get(g,'')} | {'**PASS**' if v else '**FAIL**'} |")

    open(os.path.join(STUDY, 'CAUSAL_ANALYSIS.md'), 'w').write('\n'.join(L) + '\n')
    print('wrote CAUSAL_ANALYSIS.md')

    # ---- METRICS.json
    M = {'schema': 'olhoff_current_metrics/1',
         'study': 'two_branch_controller_validation',
         'generated': datetime.datetime.now(datetime.timezone.utc)
                      .strftime('%Y-%m-%dT%H:%M:%SZ'),
         'preregistration_sha256': A['preregistration_sha256'],
         'runsExecuted': A['runsExecuted'], 'runsMissing': A['runsMissing'],
         'gates': A['gates'],
         'perMesh': {k: {'mesh': A['mesh'][k]['mesh'],
                         'prod': {kk: A['mesh'][k]['prod'].get(kk) for kk in
                                  ('status','nOuter','innerTotal','wall_s','omega1',
                                   'omega2','gap12','volume','Mnd','gray','mid',
                                   'firstDescentIter','move_final')},
                         'cand': {kk: A['mesh'][k]['cand'].get(kk) for kk in
                                  ('status','nOuter','innerTotal','wall_s','omega1',
                                   'omega2','gap12','volume_final','Mnd_final',
                                   'gray_final','mid_final','move_final','stage_final',
                                   'rho_sha256')},
                         'delta': A['mesh'][k]['delta'],
                         'gainDuringDelay': A['mesh'][k]['gainDuringDelay'],
                         'physics': A['mesh'][k]['physics'],
                         'beta': A['mesh'][k]['beta'],
                         'terminalStage': A['mesh'][k]['terminalStage'],
                         'transitions': A['mesh'][k]['transitions'],
                         'termination': A['mesh'][k]['term']}
                     for k in present}}
    json.dump(M, open(os.path.join(STUDY, 'METRICS.json'), 'w'), indent=1)
    print('wrote METRICS.json')


if __name__ == '__main__':
    main()
