#!/usr/bin/env python3
"""Dependency firewall: current code must not depend on development/.

    analysis/  --X-->  development/
    examples/  --X-->  development/
    tests/     --X-->  development/
    tools/     --X-->  development/

Run from anywhere:

    python3 tests/test_development_firewall.py          # exit 0 = PASS
    python3 -m pytest tests/test_development_firewall.py

What is checked, in every code/config file under the current roots
(.m .py .sh .bash .zsh .json .yaml .yml .toml; recorded campaign output under
examples/Performance/conference_benchmark/ is data, not code, and is skipped):

  F1  no reference to `development` (case-insensitive: macOS paths are), so
      neither 'development/x' nor fullfile(repo, 'development', 'x') passes;
  F2  no reference to a KNOWN HISTORICAL TREE by name.  A literal-only check is
      not enough: MATLAB builds paths with fullfile(), so the historical
      directory names themselves are forbidden, including the path forms of the
      three renamed trees (analysis/OlhoffCurrent, analysis/YukselApproach,
      analysis/ourApproach);
  F3  no genpath() sweep (it is how historical trees used to leak onto the
      MATLAB path);
  F4  no current directory carries a historical tree name;
  F5  exactly one definition of every public entry point, and no duplicate
      bare MATLAB function name across the current roots (package +dir and
      class @dir members are invisible to bare-name resolution and skipped);
  F6  the Olhoff path gate forbids the whole development/ tree at run time
      (the dynamic complement of F1/F2: it catches path construction no static
      scan can see).

Whole-line comments are stripped before F1-F3.  Every remaining hit must be
listed in tests/firewall_allowlist.tsv (path, exact substring, reason).  An
allowlist entry that no longer matches anything is itself a failure, so the
allowlist cannot silently outlive the code it excuses.
"""
import csv, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ROOTS = ['analysis', 'examples', 'tests', 'tools']
EXTS = {'.m', '.py', '.sh', '.bash', '.zsh', '.json', '.yaml', '.yml', '.toml'}
SKIP_DIRS = {'.git', '__pycache__', '.venv'}
SKIP_PREFIXES = ['examples/Performance/conference_benchmark/']   # recorded output, not code
ALLOWLIST = os.path.join(REPO, 'tests', 'firewall_allowlist.tsv')

HISTORICAL_NAMES = [
    'OlhoffApproachExactOpus', 'OlhoffApproachExact', 'OlhoffApproach', 'OlhoffM4Reconstruction',
    'OlhoffRegularized', 'OlhoffReproduced2007', 'OlhoffExperiments', 'reproduction2007',
    'olhoff_stabilization_audit', 'olhoff_fixed_budget_audit', 'olhoff_native_convergence',
    'olhoff_nested_mma_route_audit', 'olhoff_practical_convergence_audit',
    'iteration_efficiency_', 'iteration_count_audit', 'performance_campaign_forensic_audit',
    'performance_campaign_targeted_replays', 'three_method_parametric_study', 'Revision_v1',
    'conference_benchmark_v1', 'LabandaApproach', 'source_of_truth', 'phase5_evidence',
    'legacy_r3', 'final_campaign_profile', 'OlhoffCurrent_diagnostics', 'OlhoffCurrent_evidence',
    'OlhoffCurrent_study_governance',
]
SEP = r"""['"]?\s*[,/\\]\s*['"]?"""
PATTERNS = (
    [('F1', re.compile(r'(?<![A-Za-z0-9_])development(?![A-Za-z0-9_])', re.I))]
    + [('F2', re.compile(re.escape(n))) for n in HISTORICAL_NAMES]
    + [('F2', re.compile(r'analysis' + SEP + r'(OlhoffCurrent|YukselApproach|ourApproach)(?![A-Za-z0-9_])')),
       ('F2', re.compile(r'(OlhoffCurrent|YukselApproach|ourApproach)' + SEP + r'(\+impl|diagnostics|evidence|Matlab|Python)(?![A-Za-z0-9_])')),
       ('F2', re.compile(r'YukselApproach\.Python')),
       ('F3', re.compile(r'genpath\s*\('))]
)
PUBLIC_ENTRY_POINTS = ['olhoffcurrent_run.m', 'topopt_freq.m', 'topopt_freq.py',
                       'top99neo_inertial_freq.m', 'run_topopt_from_json.m', 'run_topopt_from_json.py',
                       'study_base_config.m', 'study_evaluate_design.m', 'performance_comparison.m']
# Package-hidden public entry points: resolvable only through their package, never by bare name.
PACKAGE_ENTRY_POINTS = ['analysis/Olhoff/+impl/architecture/olhoffSolve.m']
# Bare-name duplicates that are not competing implementations.
DUPLICATE_OK = {
    'BeamTopOpt.m': 'per-example SCRIPT in examples/ClampedBeam and examples/HingedBeam; each is run from its own folder (cd to scriptDir), neither is on the MATLAB path',
}


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, '/')


def walk():
    for root in ROOTS:
        base = os.path.join(REPO, root)
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in fn:
                yield os.path.join(dp, f)


def code_lines(path):
    ext = os.path.splitext(path)[1]
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            lines = fh.read().splitlines()
    except OSError:
        return
    in_block = False
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if ext == '.m':
            if s in ('%{',):
                in_block = True; continue
            if s in ('%}',):
                in_block = False; continue
            if in_block or s.startswith('%'):
                continue
        elif ext in ('.py', '.sh', '.bash', '.zsh', '.yaml', '.yml', '.toml'):
            if s.startswith('#'):
                continue
        yield i, line


def load_allowlist():
    rows = []
    with open(ALLOWLIST, newline='') as fh:
        for r in csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            if not r or r[0].startswith('#') or r[0] == 'path':
                continue
            if len(r) < 3 or not r[2].strip():
                raise SystemExit(f'allowlist row without a reason: {r}')
            rows.append({'path': r[0], 'substr': r[1], 'reason': r[2], 'used': 0})
    return rows


def scan():
    allow = load_allowlist()
    violations = []
    for p in walk():
        rp = rel(p)
        if os.path.splitext(p)[1] not in EXTS or any(rp.startswith(x) for x in SKIP_PREFIXES):
            continue
        if rp == 'tests/test_development_firewall.py':
            continue   # this file names what it forbids
        for i, line in code_lines(p):
            for tag, rx in PATTERNS:
                if not rx.search(line):
                    continue
                ok = [a for a in allow if a['path'] == rp and a['substr'] in line]
                if ok:
                    for a in ok: a['used'] += 1
                else:
                    violations.append(f'{tag} {rp}:{i}: {line.strip()[:160]}')
    stale = [f"stale allowlist entry: {a['path']} :: {a['substr']}" for a in allow if a['used'] == 0]
    return violations, stale, allow


def dirs_with_historical_names():
    bad = []
    for root in ROOTS:
        for dp, dn, fn in os.walk(os.path.join(REPO, root)):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for d in dn:
                if any(d.startswith(n) for n in HISTORICAL_NAMES) or d in ('OlhoffCurrent', 'YukselApproach', 'ourApproach'):
                    bad.append(rel(os.path.join(dp, d)))
    return bad


def uniqueness():
    seen = {}
    for p in walk():
        rp = rel(p)
        parts = rp.split('/')
        if any(x.startswith('+') or x.startswith('@') for x in parts[:-1]):
            continue
        if any(rp.startswith(x) for x in SKIP_PREFIXES):
            continue
        name = parts[-1]
        if name.endswith('.m') or name in PUBLIC_ENTRY_POINTS:
            seen.setdefault(name, []).append(rp)
    problems = []
    for ep in PUBLIC_ENTRY_POINTS:
        n = len(seen.get(ep, []))
        if n != 1:
            problems.append(f'F5 public entry point {ep}: {n} definitions {seen.get(ep, [])}')
    for ep in PACKAGE_ENTRY_POINTS:
        if not os.path.isfile(os.path.join(REPO, ep)):
            problems.append(f'F5 package entry point missing: {ep}')
    for name, where in sorted(seen.items()):
        if name.endswith('.m') and len(where) > 1 and name not in DUPLICATE_OK:
            problems.append(f'F5 duplicate MATLAB name {name}: {where}')
    return problems, seen


def gate_forbids_development():
    f = os.path.join(REPO, 'analysis', 'Olhoff', 'olhoffcurrent_forbidden_paths.m')
    if not os.path.isfile(f):
        return [f'F6 {rel(f)} missing']
    body = '\n'.join(l for _, l in code_lines(f))
    return [] if re.search(r"fullfile\(\s*'development'\s*\)|'development'", body) else \
        ['F6 olhoffcurrent_forbidden_paths does not forbid development/']


def run():
    violations, stale, allow = scan()
    named = [f'F4 current directory with a historical name: {d}' for d in dirs_with_historical_names()]
    uniq, _ = uniqueness()
    gate = gate_forbids_development()
    problems = violations + stale + named + uniq + gate
    print(f'scanned roots {ROOTS}; allowlisted hits: {sum(a["used"] for a in allow)} '
          f'across {sum(1 for a in allow if a["used"])} entries')
    for p in problems:
        print('  ' + p)
    print('FIREWALL ' + ('PASS' if not problems else f'FAIL ({len(problems)} problem(s))'))
    return problems


def test_development_firewall():
    assert run() == []


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
