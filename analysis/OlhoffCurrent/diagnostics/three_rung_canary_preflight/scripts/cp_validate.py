#!/usr/bin/env python3
"""Validate cp_confighash against the nine recorded legacy campaign hashes."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cp_confighash import config_hash, ROOT

AUD = ROOT / 'analysis/OlhoffCurrent/diagnostics/nine_mesh_campaign_audit'
cfgs = json.load(open(AUD / 'effective_configs.json'))
rows = list(csv.DictReader(open(AUD / 'MASTER_TABLE.csv')))
ok = 0
for c, r in zip(cfgs, rows):
    h, _ = config_hash(c)
    match = h == r['effective_config_hash']
    ok += match
    print(f"{r['mesh']:9s} recorded={r['effective_config_hash'][:16]} recomputed={h[:16]} {'MATCH' if match else 'MISMATCH'}")
print(f"\n{ok}/{len(rows)} match")
sys.exit(0 if ok == len(rows) else 1)
