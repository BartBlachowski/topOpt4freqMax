#!/usr/bin/env python3
"""cp_freeze.py -- write the FROZEN pre-run assertion manifest.

Written BEFORE canary 1.  cp_preflight.m fails closed against it: any
difference between the runtime-resolved configuration and this file aborts
before olhoffSolve is ever called.
"""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_config import canary_hash, canary_config, CAP, THREE_RUNG_LEVELS
from cp_confighash import ROOT

HERE = Path(__file__).parents[1]


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ''


integrity = json.load(open(HERE / 'evidence/integrity.json'))

meshes = {}
for nx, ny in [(480, 60), (800, 100)]:
    NE = nx * ny
    h, _ = canary_hash(nx, ny)
    cfg = canary_config(nx, ny)
    meshes[f'{nx}x{ny}'] = {
        'nelx': nx, 'nely': ny, 'NE': NE,
        'free_DOF_expected': None,          # recorded at runtime from mdl, not asserted
        'stop_tolerance': cfg['stop.tolerance'],
        'filter_radiusPhysical': cfg['filter.radiusPhysical'],
        'rminEl_expected': cfg['filter.radiusPhysical'] * ny / 1.0,   # R/(b/nely)
        'predicted_config_hash': h,
        'retention_policy': ('full RHO and DRHO retained for every outer iteration, '
                             'plus hist, res.diag, res.exhaustion and res.log; '
                             'no post-hoc discard is permitted'),
        'output_path': f'runs/C{nx}x{ny}_three_rung',
    }

man = {
    'manifest_schema': 'three_rung_canary_preflight/1',
    'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'purpose': ('Pre-run assertion manifest.  cp_preflight.m MUST fail closed '
                'against this file before any optimization is executed.'),
    'branch': sh('git -C %s rev-parse --abbrev-ref HEAD' % ROOT),
    'head': sh('git -C %s rev-parse HEAD' % ROOT),
    'cap': CAP,
    'cap_provenance': ('two_branch_controller_validation/PREREGISTRATION.md sec. 5; '
                       'the same cap the validated C160/C240/C320/C400 runs used. '
                       'It is NOT raised or lowered after seeing any trajectory.'),
    'policy': {
        'move.levels': THREE_RUNG_LEVELS,
        'move.continuation.signal': 'stageExhaustion',
        'stop.rule': 'stageExhaustion',
        'exhaustion_rule': 'frozen two-branch E = A OR B, W=20, P=20, Wnp=10',
        'beta_continuation_authority': False,
        'beta_stop_authority': False,
    },
    'impl': {
        'root': integrity['impl_root'],
        'tree_sha256': integrity['actual_tree_sha256'],
        'n_files': integrity['n_files_actual'],
        'matches_validated_c320_implTree': (
            integrity['actual_tree_sha256']
            == 'edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb'),
    },
    'production_preset_state': {
        'production_preset': 'duOlhoffFixedPenaltySensitivityFiltered',
        'delegates_to': 'duOlhoffFrozenM4',
        'promoted_to_three_rung': False,
        'disclosure': ('Production is STILL the legacy four-rung / boundVariable / '
                       'designChange policy.  This driver therefore applies the '
                       'validated overrides explicitly through tr_config, and does '
                       'NOT resolve through olhoffcurrent_config.'),
    },
    'controller_path': {
        'entry': 'cp_run.m -> cp_preflight.m -> cp_config.m -> tr_config.m -> cv_config.m',
        'solver': 'olhoffSolve.m',
        'descent': '+olh/+move/limit.m  stageExhaustion branch',
        'detector': '+olh/+move/exhaustion.m',
        'terminal_admission': 'olhoffSolve.m  exhaustStop branch',
        'sources': integrity['controller_path_sources'],
    },
    'environment_record_required': [
        'matlab version', 'computer', 'os', 'cpu', 'ncores', 'ram_bytes',
        'threads', 'blas thread env', 'hostname', 'loadavg', 'swap', 'free RAM'],
    'environment_record_note': (
        'These are RECORDED by cp_preflight at run time and written into the run '
        'record.  They are deliberately NOT asserted against frozen values: no '
        'MATLAB has ever been started on this host under this study, so freezing '
        'an expected toolchain version would be fabrication rather than a check.'),
    'host_at_freeze_time': {
        'cpu': sh('sysctl -n machdep.cpu.brand_string'),
        'ncores': sh('sysctl -n hw.ncpu'),
        'ram_bytes': sh('sysctl -n hw.memsize'),
        'os': sh('sw_vers -productVersion'),
        'kernel': sh('uname -r'),
        'hostname': sh('hostname'),
        'matlab_app': '/Applications/MATLAB_R2025b.app',
        'matlab_runtime_version': None,
    },
    'timing_schema': {
        'per_outer': ['tOuter', 'tEig', 'tGrad', 'tInner'],
        'derived': ['tOther = tOuter - tEig - tGrad - tInner'],
        'class': ('NONDETERMINISTIC PERFORMANCE TELEMETRY.  Never read back by the '
                  'optimizer; excluded from every scientific-state comparison.'),
    },
    'meshes': meshes,
}
(HERE / 'PREFLIGHT_MANIFEST.json').write_text(json.dumps(man, indent=1) + '\n')
print(json.dumps({k: man[k] for k in ('branch', 'head', 'cap', 'policy', 'impl')}, indent=1))
for k, v in meshes.items():
    print(f"{k}: cfgHash={v['predicted_config_hash']} eps={v['stop_tolerance']} rminEl={v['rminEl_expected']}")
