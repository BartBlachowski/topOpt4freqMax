#!/usr/bin/env python3
"""Execute the moves defined by manifest_rules.py.  Moves only -- never deletes.

    python3 execute_migration.py --dry-run     # print the ordered operations
    python3 execute_migration.py --execute     # perform them, logging to MIGRATION_LOG.tsv

Ordering
  1. a rule whose NEW path is a prefix of another rule's NEW path goes first
     (references -> paper/references before docs/*.pdf -> paper/references/library/*);
  2. otherwise the deepest OLD path goes first, so children leave a directory
     before the directory itself is renamed (OlhoffCurrent/diagnostics before
     OlhoffCurrent -> Olhoff).

Safety
  * fail-stop if a destination already exists (no merge, no overwrite);
  * `git mv` when the source contains tracked files (a directory rename carries
    its untracked, ignored and empty contents with it); plain rename otherwise;
  * every operation is appended to MIGRATION_LOG.tsv before the next one starts.
"""
import os, subprocess, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest_rules as M

REPO = M.REPO
LOG = os.path.join(M.HERE, 'MIGRATION_LOG.tsv')


def tracked_under(p):
    out = subprocess.run(['git', '-C', REPO, 'ls-files', '-z', '--', p], capture_output=True, check=True).stdout
    return len([x for x in out.split(b'\0') if x])


def is_prefix(a, b):
    return b == a or b.startswith(a + '/')


def ordered_moves():
    M.lit_rules()
    moves = [r for r in M.R if r[0] != r[1] and os.path.lexists(os.path.join(REPO, r[0]))]
    # stable topological order on the two constraints
    moves.sort(key=lambda r: (-r[0].count('/'), r[0]))
    changed = True
    while changed:
        changed = False
        for i in range(len(moves)):
            for j in range(i + 1, len(moves)):
                a, b = moves[i], moves[j]
                if is_prefix(b[1], a[1]) and b[1] != a[1]:      # b's destination contains a's: b first
                    moves.insert(i, moves.pop(j)); changed = True; break
            if changed:
                break
    return moves


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--dry-run'
    moves = ordered_moves()
    for r in moves:
        old, new = r[0], r[1]
        kind = 'git mv' if tracked_under(old) else 'mv'
        if mode != '--execute':
            print(f'{kind}\t{old}\t{new}'); continue
        src, dst = os.path.join(REPO, old), os.path.join(REPO, new)
        if os.path.lexists(dst):
            sys.exit(f'STOP: destination exists: {new}')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if kind == 'git mv':
            subprocess.run(['git', '-C', REPO, 'mv', '--', old, new], check=True)
        else:
            os.rename(src, dst)
        with open(LOG, 'a') as fh:
            fh.write(f'{datetime.datetime.now().isoformat(timespec="seconds")}\t{kind}\t{old}\t{new}\n')
    if mode == '--execute':
        print(f'{len(moves)} operations executed')


if __name__ == '__main__':
    main()
