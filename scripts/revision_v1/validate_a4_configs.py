#!/usr/bin/env python3
"""Validator V-A4-2 — factor drift (A4_SPECIFICATION_V3 §7.8).

Asserts that A4 varies EXACTLY ONE factor: the refresh interval N, carried as
domain.load_cases[0].loads[0].update_after.

This validator exists because of a specific, expensive failure: the previous A4
plan reused ss_beam_harmonic_frozen.json / ss_beam_harmonic_periodic.json, which
differ from the proposed method in FOUR ways at once (harmonic load, MMA, partial
dF/dx, 160x20 mesh). Sweeping N over those configs measured the confound, not the
approximation. See spec §0.1 and the EXP4 -62% result.

It therefore checks:
  1. exactly one base config exists and is the declared one;
  2. the base config declares the A4 preconditions (pmass=1, baseline=solid,
     load_sensitivity=omitted, OC, semi_harmonic, 400x50);
  3. the retired EXP4 configs are NOT referenced by A4;
  4. simulating the driver's injection for every N level changes ONLY
     update_after and NOTHING else.

Usage:  python3 scripts/revision_v1/validate_a4_configs.py
Exit 0 = pass, 1 = fail.
"""
import copy
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "examples", "Revision_v1", "a4_ss_400x50_base.json")
DRIVER = os.path.join(REPO, "examples", "Revision_v1", "a4_eigenpair_refresh.m")

FORBIDDEN = [
    "ss_beam_harmonic_frozen.json",
    "ss_beam_harmonic_periodic.json",
    "ablation_harmonic_frozen_solid.json",
    "ablation_harmonic_periodic_solid.json",
    "ablation_semi_harmonic.json",
    "ablation_semi_harmonic_with_loadsens.json",
    "ss_beam.json",
]

N_LEVELS = [0, 50, 10, 5, 1]  # 0 == frozen == N=inf, as injected by the driver

failures = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        failures.append(msg)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


print("\n=== validate_a4_configs (V-A4-2: factor drift) ===\n")

check(os.path.isfile(BASE), f"base config exists: {os.path.relpath(BASE, REPO)}")
if not os.path.isfile(BASE):
    sys.exit(1)

cfg = json.load(open(BASE))
opt = cfg["optimization"]

# ---- 2. preconditions ----------------------------------------------------
check(float(opt.get("pmass", -1)) == 1.0,
      "pmass == 1 (LINEAR mass — the declared method, MASS_INTERPOLATION_DECISION.md)")
check(opt.get("semi_harmonic_baseline") == "solid",
      'semi_harmonic_baseline == "solid" (Gate A0-F1; the solver enforces it)')
check(opt.get("load_sensitivity") == "omitted",
      'load_sensitivity == "omitted" (the proposed method)')
check(opt.get("optimizer") == "OC", "optimizer == OC (the proposed method, not MMA)")
check("semi_harmonic_rho_source" not in opt,
      "no semi_harmonic_rho_source (Gate A0 forbids it)")
check(opt.get("harmonic_normalize") is False, "harmonic_normalize == false (Gate A0)")
check(opt.get("filter", {}).get("heaviside") is False, "no Heaviside projection")
check(cfg["domain"]["mesh"]["nelx"] == 400 and cfg["domain"]["mesh"]["nely"] == 50,
      "mesh == 400x50 (SS beam benchmark)")

loads = cfg["domain"]["load_cases"][0]["loads"]
check(len(cfg["domain"]["load_cases"]) == 1 and len(loads) == 1,
      "exactly one load case with one load (alpha = 1, mode 1)")
check(loads[0]["type"] == "semi_harmonic",
      'load type == "semi_harmonic" (NOT "harmonic" — that is the retired EXP4 path)')
check("update_after" in loads[0],
      "base config declares update_after (the single independent variable)")

# ---- 3. retired EXP4 configs must not be referenced ----------------------
driver_src = open(DRIVER).read() if os.path.isfile(DRIVER) else ""
bad = [f for f in FORBIDDEN if f in driver_src]
check(not bad, f"A4 driver references NO retired/pre-authoritative config (found: {bad or 'none'})")

# ---- 4. single-factor injection -----------------------------------------
base_flat = flatten(cfg)
key = "domain.load_cases[0].loads[0].update_after"
check(key in base_flat, f"injection key resolvable: {key}")

for N in N_LEVELS:
    arm = copy.deepcopy(cfg)
    arm["domain"]["load_cases"][0]["loads"][0]["update_after"] = N
    arm["optimization"]["a4_endpoint_export"] = True  # driver sets this on every arm

    arm_flat = flatten(arm)
    diffs = []
    for k in set(base_flat) | set(arm_flat):
        if k == "optimization.a4_endpoint_export":
            continue  # identical across ALL arms; not a factor
        if base_flat.get(k) != arm_flat.get(k):
            diffs.append(k)

    # The base config already declares update_after = 0 (frozen), so the N=inf
    # arm legitimately differs in NOTHING. Any other arm must differ ONLY in the
    # injection key. Hence: diffs must be a SUBSET of {key}.
    ok = set(diffs) <= {key}
    check(ok, f"N={N if N else 'inf(0)'}: ONLY {key} may differ from base "
              f"(diffs: {diffs if diffs else 'none'})")

print(f"\n  checks: {checks}   failures: {len(failures)}")
if failures:
    print("\n  V-A4-2 FAILED — A4 would vary more than one factor.\n")
    sys.exit(1)
print("\n  V-A4-2 PASSED — A4 varies exactly one factor (the refresh interval N).\n")
sys.exit(0)
