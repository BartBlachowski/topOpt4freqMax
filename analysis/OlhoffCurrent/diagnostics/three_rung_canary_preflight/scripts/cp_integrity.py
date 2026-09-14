#!/usr/bin/env python3
"""cp_integrity.py -- implementation-tree and driver-source integrity, offline.

Reproduces olhoffcurrent_source_manifest's tree hash: SHA-256 of the sorted
"<relpath>  <sha256>" lines over every SOURCE file under +impl/, with
filesystem/editor artifacts stepped over exactly as olhoffcurrent_is_artifact
defines them.
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import ROOT

CORE = ROOT / 'analysis/OlhoffCurrent/+impl'
MAN = ROOT / 'analysis/OlhoffCurrent/SOURCE_MANIFEST.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def is_artifact(name):
    return name == '.DS_Store' or name.endswith(('.asv', '~'))


def tree():
    files = sorted(p for p in CORE.rglob('*') if p.is_file() and not is_artifact(p.name))
    rows = [(str(p.relative_to(CORE)), sha(p)) for p in files]
    lines = [f'{r}  {h}' for r, h in sorted(rows)]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest(), dict(rows)


def main():
    th, rows = tree()
    man = json.load(open(MAN))
    rec = {r['path']: r['sha256'] for r in man['files']}
    mismatch = sorted(p for p in rec if p in rows and rows[p] != rec[p])
    missing = sorted(set(rec) - set(rows))
    extra = sorted(set(rows) - set(rec))
    out = {
        'impl_root': man['root'],
        'recorded_tree_sha256': man['tree_sha256'],
        'actual_tree_sha256': th,
        'tree_match': th == man['tree_sha256'],
        'n_files_recorded': man['n_files'],
        'n_files_actual': len(rows),
        'file_mismatches': mismatch,
        'file_missing': missing,
        'file_extra': extra,
        'per_file_match': f'{len(rec) - len(mismatch) - len(missing)}/{len(rec)}',
    }
    # driver sources of THIS study, hashed for the frozen manifest
    here = Path(__file__).parents[1]
    out['driver_sources'] = {
        str(p.relative_to(here)): sha(p)
        for p in sorted(here.joinpath('scripts').rglob('*')) if p.is_file()
    }
    # the production entry points the driver reaches through
    out['production_entry_points'] = {
        rel: sha(ROOT / 'analysis/OlhoffCurrent' / rel)
        for rel in ['olhoffcurrent_preset.m', 'olhoffcurrent_config.m',
                    'olhoffcurrent_paths.m', 'olhoffcurrent_config_hash.m',
                    'olhoffcurrent_source_manifest.m', 'olhoffcurrent_assert_dispatch.m']
    }
    out['controller_path_sources'] = {
        rel: sha(CORE / rel)
        for rel in ['architecture/olhoffSolve.m',
                    'architecture/+olh/+move/exhaustion.m',
                    'architecture/+olh/+move/limit.m',
                    'architecture/+olh/+config/resolve.m',
                    'architecture/+olh/+config/schema.m',
                    'architecture/+olh/+config/validate.m',
                    'architecture/+olh/+presets/duOlhoffFrozenM4.m']
    }
    here.joinpath('evidence/integrity.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in
                      ('driver_sources', 'production_entry_points', 'controller_path_sources')}, indent=1))
    return 0 if out['tree_match'] else 1


if __name__ == '__main__':
    sys.exit(main())
