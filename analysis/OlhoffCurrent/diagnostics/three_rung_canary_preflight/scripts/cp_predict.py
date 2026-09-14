#!/usr/bin/env python3
"""Cross-validate the offline three-rung construction, then emit the canary hashes."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_config import canary_hash, canary_config, nested, CAP
from cp_confighash import config_hash, ROOT

# --- validation 1: the recorded VALIDATED C320 three-rung configuration -----
VALIDATED_C320 = 'afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab'
h320, _ = canary_hash(320, 40)
print(f'C320 three-rung  recorded={VALIDATED_C320}')
print(f'C320 three-rung  offline ={h320}   {"MATCH" if h320 == VALIDATED_C320 else "MISMATCH"}')

# --- validation 2: the four-rung 'C' arm at 320 must NOT collide ------------
h320_4, _ = canary_hash(320, 40, levels=[0.04, 0.02, 0.01, 0.005])
print(f'C320 four-rung   offline ={h320_4}   (distinct: {h320_4 != h320})')

out = {'validated_c320_recorded': VALIDATED_C320,
       'validated_c320_offline': h320, 'c320_match': h320 == VALIDATED_C320,
       'cap': CAP, 'meshes': {}}
for nx, ny in [(480, 60), (800, 100)]:
    h, lines = canary_hash(nx, ny)
    out['meshes'][f'{nx}x{ny}'] = {
        'nelx': nx, 'nely': ny, 'NE': nx * ny,
        'predicted_config_hash': h,
        'stop_tolerance': canary_config(nx, ny)['stop.tolerance'],
    }
    Path(__file__).parents[1].joinpath(f'evidence/predicted_config_{nx}x{ny}.txt').write_text('\n'.join(lines) + '\n')
    print(f'{nx}x{ny} predicted three-rung cfgHash = {h}')
print(json.dumps(out['meshes'], indent=1))
Path(__file__).parents[1].joinpath('evidence/predicted_config_hashes.json').write_text(json.dumps(out, indent=1) + '\n')
sys.exit(0 if out['c320_match'] else 1)
