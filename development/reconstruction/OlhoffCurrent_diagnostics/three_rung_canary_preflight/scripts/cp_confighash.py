#!/usr/bin/env python3
"""cp_confighash.py -- offline reconstruction of olhoffcurrent_config_hash.

The MATLAB digest is SHA-256 over schema-ordered 'path=value' lines with
runtime.name excluded and MATLAB mat2str(v,17) number formatting.  Reproducing
it here is what lets the PRE-RUN manifest name the expected hash BEFORE any
solve, so the canary runner can fail closed on a mismatch.

Validated against the nine recorded legacy campaign hashes (cp_validate).
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]          # repo root
IMPL = ROOT / 'analysis/OlhoffCurrent/+impl/architecture/+olh/+config'


def schema_paths():
    txt = (IMPL / 'schema.m').read_text()
    body = txt.split('S = {', 1)[1]
    return re.findall(r"^'([^']+)'\s*,", body, re.M)


def show(v):
    """MATLAB local_show of olhoffcurrent_config_hash."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, list):
        if not v:
            return '[]'
        return '[' + ' '.join(show(x) for x in v) + ']'
    if isinstance(v, (int, float)):
        return format(float(v), '.17g')
    raise ValueError(repr(v))


def flat(d, p=''):
    out = {}
    for k, v in d.items():
        q = f'{p}.{k}' if p else k
        if isinstance(v, dict):
            out.update(flat(v, q))
        else:
            out[q] = v
    return out


def config_hash(cfg):
    f = flat(cfg)
    lines = [p + '=' + ('<excluded>' if p == 'runtime.name' else show(f[p]))
             for p in schema_paths()]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest(), lines
