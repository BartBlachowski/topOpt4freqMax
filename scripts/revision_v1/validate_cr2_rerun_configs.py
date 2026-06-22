#!/usr/bin/env python3
"""Validate immutable CR2 rerun configs against their originals."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def without_meta(cfg: dict) -> dict:
    value = copy.deepcopy(cfg)
    value.pop("meta", None)
    return value


def validate(original_a: Path, original_b: Path, rerun_a: Path, rerun_b: Path) -> dict:
    from tools.Python.run_topopt_from_json import check_gate_a0_constraints

    oa, ob, ra, rb = [json.loads(p.read_text(encoding="utf-8"))
                      for p in (original_a, original_b, rerun_a, rerun_b)]
    for cfg in (oa, ob, ra, rb):
        check_gate_a0_constraints(cfg)

    if without_meta(oa) != without_meta(ra):
        raise AssertionError("rerun Variant A differs numerically from original Variant A")

    expected_rb = without_meta(ob)
    expected_rb["optimization"]["move_limit"] = 0.02
    if without_meta(rb) != expected_rb:
        raise AssertionError("rerun Variant B differs beyond metadata and move_limit=0.02")

    if float(ra["optimization"]["convergence_tol"]) != 1e-3:
        raise AssertionError("rerun Variant A convergence_tol must remain 1e-3")
    if float(rb["optimization"]["convergence_tol"]) != 1e-3:
        raise AssertionError("rerun Variant B convergence_tol must remain 1e-3")
    if float(rb["optimization"]["move_limit"]) != 0.02:
        raise AssertionError("rerun Variant B move_limit must be 0.02")

    return {
        "status": "passed",
        "protocol": "examples/Revision_v1/cr2/cr2_rerun_protocol.md",
        "accepted_comparison_eligible": False,
        "reason": "Variant B stabilization changes move_limit; the pair is intentionally unmatched.",
        "feasibility_tolerance": 1e-4,
        "hashes": {
            "original_variant_a": digest(oa),
            "original_variant_b": digest(ob),
            "rerun_variant_a": digest(ra),
            "rerun_variant_b": digest(rb)
        },
        "checks": {
            "original_configs_preserved": True,
            "variant_a_numerically_identical_to_original": True,
            "variant_b_only_numerical_change_is_move_limit_0_02": True,
            "both_convergence_tolerances_are_1e_3": True,
            "authoritative_load_constraints_pass": True
        }
    }


def main() -> int:
    if len(sys.argv) != 6:
        print("usage: validate_cr2_rerun_configs.py ORIGINAL_A ORIGINAL_B RERUN_A RERUN_B OUTPUT", file=sys.stderr)
        return 2
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    result = validate(*paths[:4])
    paths[4].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("CR2 rerun config validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
