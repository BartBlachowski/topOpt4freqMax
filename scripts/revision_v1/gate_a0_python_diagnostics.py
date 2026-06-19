#!/usr/bin/env python3
"""Generate and locally validate Python Gate A0 diagnostics."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


ABS_TOL = 1e-12
REL_TOL = 1e-8


def _assert_close(name: str, actual, expected) -> None:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} != {e.shape}")
    allowed = ABS_TOL + REL_TOL * np.abs(e)
    error = np.abs(a - e)
    if np.any(~np.isfinite(a)) or np.any(error > allowed):
        raise AssertionError(f"{name}: max error {np.max(error):.3e} exceeds tolerance")


def _required(diag: dict, name: str):
    if name not in diag:
        raise AssertionError(f"missing Gate A0 diagnostic: {name}")
    return diag[name]


def _validate_local(diag: dict, selected_mode: str) -> None:
    required = (
        "reference_omega", "reference_omega_sq", "reference_modes",
        "reference_modal_mass", "current_mass_matrix", "load_vector",
        "objective", "omitted_sensitivity", "complete_sensitivity",
        "selected_sensitivity", "selected_load_sensitivity",
        "load_normalization_enabled", "obsolete_rho_source_used",
    )
    for name in required:
        _required(diag, name)

    if diag["load_normalization_enabled"]:
        raise AssertionError("load normalization is enabled")
    if diag["obsolete_rho_source_used"]:
        raise AssertionError("obsolete rho-source behavior was used")
    if diag["selected_load_sensitivity"] != selected_mode:
        raise AssertionError("selected load-sensitivity mode was not propagated")

    modal_mass = np.asarray(diag["reference_modal_mass"], dtype=float)
    _assert_close("reference modal mass", modal_mass, np.ones_like(modal_mass))

    omega_sq = np.asarray(diag["reference_omega_sq"], dtype=float)
    phi0 = np.asarray(diag["reference_modes"], dtype=float)
    mass = diag["current_mass_matrix"]
    expected_load = omega_sq[0] * (mass @ phi0[:, 0])
    actual_load = np.asarray(diag["load_vector"], dtype=float)[:, 0]
    _assert_close("independent omega0^2*M(x)*Phi0 load", actual_load, expected_load)

    expected_selected = (
        diag["complete_sensitivity"] if selected_mode == "complete"
        else diag["omitted_sensitivity"]
    )
    _assert_close("selected analytical sensitivity", diag["selected_sensitivity"], expected_selected)


def _serializable(diag: dict) -> dict:
    def vec(name: str) -> list[float]:
        return np.asarray(diag[name], dtype=float).reshape(-1, order="F").tolist()

    return {
        "reference_omega": vec("reference_omega"),
        "reference_omega_sq": vec("reference_omega_sq"),
        "reference_modes": vec("reference_modes"),
        "reference_modal_mass": vec("reference_modal_mass"),
        "load_vector": vec("load_vector"),
        "objective": float(diag["objective"]),
        "omitted_sensitivity": vec("omitted_sensitivity"),
        "complete_sensitivity": vec("complete_sensitivity"),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: gate_a0_python_diagnostics.py FIXTURE_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    fixture = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from tools.Python.run_topopt_from_json import run_topopt_from_json

    cfg = json.loads(fixture.read_text(encoding="utf-8"))
    runs: dict[str, dict] = {}
    for mode in ("omitted", "complete"):
        run_cfg = copy.deepcopy(cfg)
        run_cfg["optimization"]["load_sensitivity"] = mode
        _, _, _, _, info = run_topopt_from_json(run_cfg, return_diagnostics=True)
        if not isinstance(info, dict) or "gate_a0" not in info:
            raise AssertionError("Python solver did not return gate_a0 diagnostics")
        diag = info["gate_a0"]
        _validate_local(diag, mode)
        runs[mode] = _serializable(diag)

    for invariant in (
        "reference_omega", "reference_omega_sq", "reference_modes",
        "reference_modal_mass", "load_vector", "objective",
        "omitted_sensitivity", "complete_sensitivity",
    ):
        _assert_close(
            f"omitted/complete invariant {invariant}",
            runs["complete"][invariant], runs["omitted"][invariant],
        )

    output.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Python Gate A0 diagnostics passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
