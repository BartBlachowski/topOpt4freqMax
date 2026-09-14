#!/usr/bin/env python3
"""Emit EFFECTIVE_CONFIG.json -- the config the canary driver WOULD resolve."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_config import canary_config, canary_hash, nested, CAP

HERE = Path(__file__).parents[1]
out = {
    'status': 'PREDICTED_NOT_RUNTIME_RESOLVED',
    'why': ('MATLAB could not be started on this host (network licence server '
            'unreachable), so olh.config.resolve was never executed and no '
            'RUNTIME-resolved effective configuration exists.  The tables below '
            'are an offline reconstruction of the resolve chain, validated by '
            'reproducing the nine recorded legacy campaign config hashes (9/9) '
            'and the recorded VALIDATED three-rung C320 hash exactly.  They are '
            'the FROZEN EXPECTATION cp_preflight.m asserts against.  They do NOT '
            'discharge the Part A requirement, which is a runtime resolution.'),
    'builder_chain': 'cp_run.m -> cp_preflight.m -> cp_config.m -> tr_config.m -> cv_config.m -> olh.config.resolve',
    'preset': 'duOlhoffFrozenM4  (via olhoffcurrent_preset().upstreamPreset)',
    'cap': CAP,
    'validation': {
        'legacy_nine_hashes_reproduced': '9/9',
        'validated_c320_three_rung_hash_reproduced': True,
        'validated_c320_hash': 'afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab',
    },
    'meshes': {},
}
for nx, ny in [(480, 60), (800, 100)]:
    h, _ = canary_hash(nx, ny)
    out['meshes'][f'{nx}x{ny}'] = {
        'predicted_config_hash': h,
        'free_DOF': 2 * (nx + 1) * (ny + 1) - 4,
        'effective_config': nested(canary_config(nx, ny)),
    }
(HERE / 'EFFECTIVE_CONFIG.json').write_text(json.dumps(out, indent=1) + '\n')
print('wrote EFFECTIVE_CONFIG.json')
