#!/usr/bin/env python3
"""V1c static validation for a matched authoritative-load CR2 config pair."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate_variant(cfg: dict, expected_mode: str, label: str) -> None:
    from tools.Python.run_topopt_from_json import check_gate_a0_constraints

    opt = cfg.get("optimization", {})
    if opt.get("gate_a0_diagnostics") is not True:
        raise AssertionError(f"{label}: gate_a0_diagnostics must be true")
    check_gate_a0_constraints(cfg)
    if opt.get("semi_harmonic_baseline") != "solid":
        raise AssertionError(f"{label}: semi_harmonic_baseline must explicitly be solid")
    if opt.get("load_sensitivity") != expected_mode:
        raise AssertionError(f"{label}: load_sensitivity must be {expected_mode}")
    if opt.get("harmonic_normalize", False) is not False:
        raise AssertionError(f"{label}: harmonic_normalize must be false or absent")
    for obsolete in ("semi_harmonic_rho_source", "semi_harmonic_load_sensitivity"):
        if obsolete in opt:
            raise AssertionError(f"{label}: obsolete key {obsolete} must be absent")
    cases = cfg.get("domain", {}).get("load_cases", [])
    if len(cases) != 1 or len(cases[0].get("loads", [])) != 1:
        raise AssertionError(f"{label}: expected exactly one authoritative modal load")
    load = cases[0]["loads"][0]
    if load.get("type") != "semi_harmonic" or load.get("mode") != 1:
        raise AssertionError(f"{label}: mode tracking must use frozen solid-reference mode 1")

def validate(path_a: Path, path_b: Path) -> dict:
    cfg_a = json.loads(path_a.read_text(encoding="utf-8"))
    cfg_b = json.loads(path_b.read_text(encoding="utf-8"))
    _validate_variant(cfg_a, "omitted", "Variant A")
    _validate_variant(cfg_b, "complete", "Variant B")

    shared_a = copy.deepcopy(cfg_a)
    shared_b = copy.deepcopy(cfg_b)
    shared_a.pop("meta", None)
    shared_b.pop("meta", None)
    del shared_a["optimization"]["load_sensitivity"]
    del shared_b["optimization"]["load_sensitivity"]
    if shared_a != shared_b:
        raise AssertionError(
            "V1c mismatch: CR2 configs differ beyond metadata and optimization.load_sensitivity"
        )

    return {
        "gate": "V1c",
        "status": "passed",
        "authoritative_load": "F(x) = omega0^2 * M(x) * Phi0",
        "variant_a": {"path": str(path_a), "load_sensitivity": "omitted", "sha256": _sha256(cfg_a)},
        "variant_b": {"path": str(path_b), "load_sensitivity": "complete", "sha256": _sha256(cfg_b)},
        "shared_configuration_sha256": _sha256(shared_a),
        "allowed_differences": ["meta", "optimization.load_sensitivity"],
        "checks": {
            "solid_reference": True,
            "gate_a0_diagnostics": True,
            "load_normalization_disabled": True,
            "obsolete_rho_source_absent": True,
            "same_initial_design_mesh_filter_stopping_and_mode_tracking": True
        }
    }


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: validate_v1c_cr2_configs.py VARIANT_A.json VARIANT_B.json OUTPUT.json", file=sys.stderr)
        return 2
    path_a, path_b, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
    result = validate(path_a, path_b)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("V1c CR2 configuration validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
