#!/usr/bin/env python3
"""Verify modal normalization, deterministic orientation, and squared MAC."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


LOCAL_TOL = 1e-12


def _require(diag: dict, name: str):
    if name not in diag:
        raise AssertionError(f"V1b missing Gate A0 diagnostic: {name}")
    return diag[name]


def _assert_close(name: str, actual, expected, tolerance: float = LOCAL_TOL) -> None:
    error = np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f"V1b {name} error {error:.3e} exceeds {tolerance:.1e}")


def verify(fixture: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from analysis.ourApproach.Python.modal_utils import (
        normalize_and_orient_modes,
        squared_mass_weighted_mac,
    )
    from tools.Python.run_topopt_from_json import run_topopt_from_json

    cfg = json.loads(fixture.read_text(encoding="utf-8"))
    run_cfg = copy.deepcopy(cfg)
    run_cfg["optimization"]["load_sensitivity"] = "complete"
    _, _, _, _, info = run_topopt_from_json(run_cfg, return_diagnostics=True)
    if not isinstance(info, dict) or "gate_a0" not in info:
        raise AssertionError("V1b run returned no Gate A0 diagnostics")
    diag = info["gate_a0"]
    mass = _require(diag, "current_mass_matrix")
    reference_modes = np.asarray(_require(diag, "reference_modes"), dtype=float)
    reference_modal_mass = np.asarray(_require(diag, "reference_modal_mass"), dtype=float)
    if reference_modes.ndim != 2 or reference_modes.shape[1] < 1:
        raise AssertionError("V1b reference mode diagnostic is empty")
    _assert_close("reference modal mass", reference_modal_mass, np.ones_like(reference_modal_mass))

    first = reference_modes[:, 0]
    seed = np.linspace(1.0, 2.0, first.size)
    seed -= first * (first @ (mass @ seed)) / (first @ (mass @ first))
    raw_modes = np.column_stack((-3.25 * first, 2.75 * seed))
    normalized = normalize_and_orient_modes(raw_modes, mass)
    modal_mass_matrix = normalized.T @ (mass @ normalized)
    modal_masses = np.diag(modal_mass_matrix)
    largest_dof_indices = np.argmax(np.abs(normalized), axis=0)
    largest_dof_values = normalized[largest_dof_indices, np.arange(normalized.shape[1])]

    identical_mac = squared_mass_weighted_mac(normalized[:, 0], normalized[:, 0], mass)
    sign_invariant_mac = squared_mass_weighted_mac(normalized[:, 0], -normalized[:, 0], mass)
    scale_invariant_mac = squared_mass_weighted_mac(3.0*normalized[:, 0], -7.0*normalized[:, 0], mass)
    orthogonal_mac = squared_mass_weighted_mac(normalized[:, 0], normalized[:, 1], mass)
    mac_matrix = squared_mass_weighted_mac(normalized, normalized, mass)

    _assert_close("unit modal mass", modal_masses, np.ones(2))
    if np.any(largest_dof_values < 0.0):
        raise AssertionError("V1b deterministic orientation left a negative phase-defining DOF")
    _assert_close("identical-mode MAC", identical_mac, 1.0)
    _assert_close("sign-invariant MAC", sign_invariant_mac, 1.0)
    _assert_close("scale-invariant MAC", scale_invariant_mac, 1.0)
    _assert_close("M-orthogonal MAC", orthogonal_mac, 0.0)
    _assert_close("pairwise MAC matrix", mac_matrix, np.eye(2))

    return {
        "status": "passed",
        "modal_mass_tolerance": LOCAL_TOL,
        "mac_tolerance": LOCAL_TOL,
        "normalization_definition": "phi' * M * phi = 1",
        "phase_definition": "largest-magnitude DOF is nonnegative",
        "mac_definition": "(phi' * M * psi)^2 / ((phi' * M * phi) * (psi' * M * psi))",
        "modal_masses": modal_masses.tolist(),
        "largest_magnitude_dof_indices": (largest_dof_indices + 1).tolist(),
        "largest_magnitude_dof_values": largest_dof_values.tolist(),
        "normalized_oriented_modes": normalized.reshape(-1, order="F").tolist(),
        "identical_mode_mac": identical_mac,
        "sign_invariant_mac": sign_invariant_mac,
        "scale_invariant_mac": scale_invariant_mac,
        "orthogonal_mode_mac": orthogonal_mac,
        "pairwise_mac_matrix": mac_matrix.reshape(-1, order="F").tolist(),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_v1b_mac.py FIXTURE_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    fixture = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if not fixture.is_file():
        raise FileNotFoundError(f"V1b fixture not found: {fixture}")
    result = verify(fixture)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Python V1b MAC verification passed; orthogonal MAC = {result['orthogonal_mode_mac']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
