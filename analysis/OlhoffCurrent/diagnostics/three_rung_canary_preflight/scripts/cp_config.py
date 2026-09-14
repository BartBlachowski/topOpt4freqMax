#!/usr/bin/env python3
"""cp_config.py -- offline reconstruction of the THREE-RUNG canary configuration.

This mirrors, field for field, the chain the MATLAB canary driver executes:

    olh.config.defaults()                        <- schema.m column 3
      -> olh.presets.duOlhoffFrozenM4            <- the production upstream preset
      -> the two_branch_controller_validation 'C' override list
      -> the three_rung single factor move.levels = [0.04 0.02 0.01]
      -> derived rule stop.tolerance = 0.05*sqrt(NE/3200)

It is a PREDICTION, not a runtime resolution.  Its only sanctioned use is to
name the expected config hash inside the frozen pre-run manifest so that
cp_run.m can FAIL CLOSED before any optimization if the runtime disagrees.
Nothing here is permitted to stand in for the runtime resolution itself.

Validated two ways (cp_validate.py, cp_predict.py):
  * the nine recorded legacy campaign hashes reproduce 9/9;
  * the recorded VALIDATED C320 three-rung hash reproduces exactly.
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import config_hash, schema_paths, IMPL


def _mlit(s):
    """Evaluate the MATLAB literal used in a schema default / preset assign."""
    s = s.strip()
    if s.startswith("'"):
        return s[1:-1]
    if s in ('true', 'false'):
        return s == 'true'
    if s == '[]':
        return []
    if s.startswith('['):
        return [float(x) for x in s[1:-1].replace(',', ' ').split()]
    return float(s)


def defaults():
    """olh.config.defaults(): column 3 of schema.m, in schema order."""
    body = (IMPL / 'schema.m').read_text().split('S = {', 1)[1]
    out = {}
    for line in body.splitlines():
        m = re.match(r"^'([^']+)',\s*'(\w+)',\s*(.+?),\s*(\[.*?\]|\{.*?\}),\s*'([A-D])'", line)
        if m:
            out[m.group(1)] = _mlit(m.group(3))
    missing = set(schema_paths()) - set(out)
    if missing:
        raise RuntimeError(f'schema defaults not parsed for: {sorted(missing)}')
    return out


def preset_frozen_m4(cfg):
    """olh.presets.duOlhoffFrozenM4, parsed from the preset source itself."""
    src = (IMPL.parent / '+presets/duOlhoffFrozenM4.m').read_text()
    body = src.split('cfg = olh.config.assign(cfg, ...', 1)[1].split(');', 1)[0]
    for path, val in re.findall(r"'([\w.]+)',\s*([^,\n]+?),?\s*\.\.\.", body + ' ...'):
        cfg[path] = _mlit(val)
    for path, val in re.findall(r"olh\.config\.assign\(cfg,\s*'([\w.]+)',\s*([^)]+)\)", src):
        cfg[path] = _mlit(val)
    return cfg


CAP = 1600  # two_branch_controller_validation PREREGISTRATION sec. 5; unchanged here
THREE_RUNG_LEVELS = [0.04, 0.02, 0.01]


def canary_config(nelx, nely, levels=THREE_RUNG_LEVELS, cap=CAP, diagnostics=True):
    cfg = preset_frozen_m4(defaults())
    # --- cv_config('C') common + arm overrides, then the three-rung factor ---
    cfg['domain.mesh.nelx'] = float(nelx)
    cfg['domain.mesh.nely'] = float(nely)
    cfg['runtime.maxOuter'] = float(cap)
    cfg['runtime.singleThread'] = True
    cfg['runtime.diagnostics'] = bool(diagnostics)
    cfg['runtime.verbose'] = False
    cfg['runtime.name'] = f'CAN3_{nelx}x{nely}'          # excluded from the hash
    cfg['move.continuation.signal'] = 'stageExhaustion'
    cfg['stop.rule'] = 'stageExhaustion'
    cfg['move.levels'] = list(levels)
    # --- derived rule, applied AFTER overrides (olh.config.resolve) ---------
    if cfg['stop.toleranceRule'] == 'meshScaled':
        cfg['stop.tolerance'] = 0.05 * ((nelx * nely / 3200) ** 0.5)
    return cfg


def nested(flatcfg):
    out = {}
    for path, v in flatcfg.items():
        d = out
        parts = path.split('.')
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def canary_hash(nelx, nely, **kw):
    return config_hash(nested(canary_config(nelx, nely, **kw)))
