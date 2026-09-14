#!/usr/bin/env python3
"""cp_verify.py -- re-verify FINAL_SHA256.txt against the files on disk."""
import hashlib, sys
from pathlib import Path
HERE = Path(__file__).parents[1]
bad = miss = n = 0
for line in (HERE / 'FINAL_SHA256.txt').read_text().splitlines():
    if '  ' not in line or line.startswith(('=', 'FINAL', 'generated', 'branch',
                                            'HEAD', 'implTree', 'scientific')):
        continue
    h, _, rel = line.partition('  ')
    if len(h) != 64:
        continue
    f = HERE / rel
    n += 1
    if not f.exists():
        print('MISSING ', rel); miss += 1
    elif hashlib.sha256(f.read_bytes()).hexdigest() != h:
        print('MISMATCH', rel); bad += 1
print(f'{n} entries: {n-bad-miss} match, {bad} mismatch, {miss} missing')
sys.exit(1 if (bad or miss) else 0)
