"""P2: source-diff audit of the study driver against production olhoffSolve.m.

Rule (AUDIT_PREREGISTRATION.md 2.2): every line of the copy that differs from
production carries a %CS:<category>% tag; every replaced production line is kept
as a '%CS-ORIG%' comment.  Dropping tagged lines and un-commenting CS-ORIG lines
must give back production EXACTLY.  Categories must be in the permitted set.
Also proves the reused certificate/formulation files are byte-identical copies.
"""
import difflib
import re
from cs_common import *

PERMITTED = {'signature', 'dispatch', 'failclosed', 'telemetry', 'checkpoint', 'preflight', 'identity'}


def main():
    prod_p = ROOT / '+impl' / 'architecture' / 'olhoffSolve.m'
    copy_p = HERE / 'cs_olhoffSolveSOCP.m'
    prod = prod_p.read_text().splitlines()
    copy = copy_p.read_text().splitlines()
    rebuilt, tagged, cats = [], [], {}
    for i, line in enumerate(copy, 1):
        if line.startswith('%CS-ORIG%'):
            rebuilt.append(line[len('%CS-ORIG%'):])
            tagged.append((i, 'ORIG', line))
            continue
        m = re.findall(r'%CS:([A-Za-z]+)(?:<category>)?%', line)
        if '%CS:' in line:
            cat = m[0] if m else 'UNPARSED'
            cats.setdefault(cat, []).append(i)
            tagged.append((i, cat, line))
            continue
        rebuilt.append(line)
    exact = rebuilt == prod
    first_mismatch = None
    if not exact:
        for k, (a, b) in enumerate(zip(rebuilt, prod)):
            if a != b:
                first_mismatch = {'line': k + 1, 'rebuilt': a, 'production': b}
                break
        if first_mismatch is None:
            first_mismatch = {'length_rebuilt': len(rebuilt), 'length_production': len(prod)}
    diff = list(difflib.unified_diff(prod, copy, 'production/+impl/architecture/olhoffSolve.m',
                                     'study/scripts/cs_olhoffSolveSOCP.m', lineterm='', n=1))
    (EV / 'olhoffSolve_vs_study_driver.diff').write_text('\n'.join(diff) + '\n')
    ref = DIAG / 'frozen_problem25_reference' / 'scripts'
    copies = {}
    for f in ['fp_problem.m', 'fp_dualbound.m', 'fp_kkt.m']:
        copies[f] = {'reference': sha256_file(ref / f), 'study': sha256_file(HERE / f)}
        copies[f]['identical'] = copies[f]['reference'] == copies[f]['study']
    # numeric-expression lines of production that the copy touches: none allowed except dispatch head
    orig_lines = [t[2] for t in tagged if t[1] == 'ORIG']
    o = {
        'production_sha256': sha256_file(prod_p), 'study_driver_sha256': sha256_file(copy_p),
        'production_lines': len(prod), 'study_lines': len(copy),
        'reconstruction_exact': exact, 'first_mismatch': first_mismatch,
        'tagged_line_counts': {k: len(v) for k, v in cats.items()},
        'replaced_production_lines': orig_lines,
        'categories_permitted': sorted(set(cats) - {'ORIG'}) and set(cats) <= PERMITTED,
        'unpermitted_categories': sorted(set(cats) - PERMITTED),
        'byte_identical_copies': copies,
        'diff_file': 'evaluations/olhoffSolve_vs_study_driver.diff',
        'diff_hunks': sum(1 for d in diff if d.startswith('@@')),
    }
    o['pass'] = bool(exact and not o['unpermitted_categories'] and all(c['identical'] for c in copies.values()))
    dump(EV / 'diff_audit.json', o)
    print(json.dumps({k: o[k] for k in ['reconstruction_exact', 'tagged_line_counts', 'replaced_production_lines',
                                         'unpermitted_categories', 'diff_hunks', 'pass']}, indent=1))


if __name__ == '__main__':
    main()
