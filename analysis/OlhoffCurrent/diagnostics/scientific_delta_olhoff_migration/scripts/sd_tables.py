#!/usr/bin/env python3
"""File-level map (Part 5) and effective-configuration comparison (Part 4 / CONFIG_COMPARISON)."""
import difflib
from sd_common import *

IMPL = OC / '+impl'
BASE = EVAL / 'base_695f03b'
EXEC_DIRS = ['algo', 'fem', 'filter', 'mma', 'mma_published', 'architecture/+olh', 'architecture/olhoffSolve.m',
             'architecture/legacy', 'architecture/docs']


def files_under(root):
    out = set()
    for d in EXEC_DIRS:
        p = root / d
        if p.is_file():
            out.add(d)
        elif p.is_dir():
            out |= {str(q.relative_to(root)) for q in p.rglob('*') if q.is_file() and q.name != '.DS_Store'}
    return out


def nlines(a, b):
    if a is None or b is None:
        return None
    la = a.read_text(errors='replace').splitlines(); lb = b.read_text(errors='replace').splitlines()
    return sum(1 for l in difflib.unified_diff(la, lb, lineterm='', n=0) if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---')))


def main():
    t = files_under(IMPL); s = files_under(SNAP)
    rows = []
    for f in sorted(t | s):
        tp = IMPL / f if f in t else None; sp = SNAP / f if f in s else None
        bp = BASE / f if (BASE / f).exists() else None
        th = file_sha256(tp) if tp else None; sh = file_sha256(sp) if sp else None
        status = ('IDENTICAL' if th == sh else 'DIFFERENT') if (tp and sp) else ('SOURCE_ONLY' if sp else 'TARGET_ONLY')
        rows.append(dict(path=f, status=status, target_sha256=th, source_sha256=sh,
                         diff_lines_target_vs_source=nlines(tp, sp) if (tp and sp and th != sh) else (0 if th == sh else None),
                         diff_lines_base_vs_target=nlines(bp, tp) if (bp and tp) else None,
                         diff_lines_base_vs_source=nlines(bp, sp) if (bp and sp) else None))
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    # plan's "unchanged and not re-copied" claim
    claimed = ['mma/mmasub.m', 'mma/subsolv.m', 'mma_published/mmasub.m', 'mma_published/subsolv.m', 'mma_published/README.md',
               'filter/applyFilter.m', 'filter/prepFilter.m', 'filter/projChain.m', 'filter/projDensityField.m', 'filter/projectDensity.m',
               'filter/top88_reference.m', 'fem/model2D.m', 'fem/elemMats2D.m', 'fem/massScale.m', 'algo/innerLoop.m',
               'algo/innerLoopLP.m', 'algo/deltaLambda.m', 'algo/multRule.m', 'algo/moveControl.m', 'algo/useMMA.m']
    byp = {r['path']: r for r in rows}
    claim = {c: byp[c]['status'] if c in byp else 'ABSENT' for c in claimed}
    jdump(dict(counts=counts, rows=rows, plan_unchanged_claim=claim, plan_unchanged_claim_all_identical=all(v == 'IDENTICAL' for v in claim.values())),
          EVAL / 'file_map.json')
    print(counts, 'unchanged-claim all identical:', all(v == 'IDENTICAL' for v in claim.values()))
    for r in rows:
        if r['status'] != 'IDENTICAL':
            print(f"{r['status']:12s} {r['path']:55s} t/s {r['diff_lines_target_vs_source']} base->t {r['diff_lines_base_vs_target']} base->s {r['diff_lines_base_vs_source']}")

    # ---- effective configurations ---------------------------------------------
    C = json.loads((EVAL / 'effective_configs_480.json').read_text())
    names = list(C.keys())
    maps = {n: {e['path']: e['value'] for e in C[n]} for n in names}
    allp = sorted(set().union(*[set(m) for m in maps.values()]))
    diffs = []
    for p in allp:
        vals = [maps[n].get(p, '<absent>') for n in names]
        if len(set(vals)) > 1:
            diffs.append(dict(path=p, **dict(zip(names, vals))))
    ident = [p for p in allp if len({maps[n].get(p, '<absent>') for n in names}) == 1]
    jdump(dict(configs=names, n_leaves=len(allp), n_identical=len(ident), differing=diffs, identical_paths=ident), EVAL / 'config_comparison.json')
    print('config leaves', len(allp), 'identical in all four', len(ident), 'differing', len(diffs))
    for d in diffs:
        print(' ', d['path'], '|', ' | '.join(str(d[n])[:28] for n in names))


if __name__ == '__main__':
    main()
