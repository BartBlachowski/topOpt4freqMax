#!/usr/bin/env python3
"""Central finite-difference verification of the complete Gate A0 sensitivity."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve


FD_STEP = 1e-6
REL_TOL = 1e-5
REL_FLOOR = 1e-14


def _require(diag: dict, name: str):
    if name not in diag:
        raise AssertionError(f"V1a missing Gate A0 diagnostic: {name}")
    return diag[name]


def _element_data(nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_el = nelx * nely
    edof = np.zeros((n_el, 8), dtype=np.int64)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof[el] = (2*n1, 2*n1+1, 2*n2, 2*n2+1,
                        2*(n2+1), 2*(n2+1)+1, 2*(n1+1), 2*(n1+1)+1)
    i_k = np.kron(edof, np.ones((8, 1), dtype=np.int64)).ravel()
    j_k = np.kron(edof, np.ones((1, 8), dtype=np.int64)).ravel()
    return edof, i_k, j_k


def _fixed_dofs(cfg: dict, nelx: int, nely: int, length: float, height: float) -> np.ndarray:
    x_coords = np.repeat(np.linspace(0.0, length, nelx + 1), nely + 1)
    y_coords = np.tile(np.linspace(0.0, height, nely + 1), nelx + 1)
    fixed: set[int] = set()
    for support in cfg["bc"]["supports"]:
        kind = support["type"]
        if kind == "vertical_line":
            tol = float(support.get("tol", 1e-9))
            nodes = np.flatnonzero(np.abs(x_coords - float(support["x"])) <= tol)
        elif kind == "closest_point":
            point = np.asarray(support["location"], dtype=float)
            distance_sq = (x_coords - point[0])**2 + (y_coords - point[1])**2
            nodes = np.array([int(np.argmin(distance_sq))])
        else:
            raise AssertionError(f"V1a fixture uses unsupported support type: {kind}")
        for node in nodes:
            if "ux" in support["dofs"]:
                fixed.add(2 * int(node))
            if "uy" in support["dofs"]:
                fixed.add(2 * int(node) + 1)
    if not fixed:
        raise AssertionError("V1a fixture produced no fixed degrees of freedom")
    return np.asarray(sorted(fixed), dtype=np.int64)


def _objective(x: np.ndarray, cfg: dict, diag: dict) -> float:
    from analysis.ourApproach.Python.topopt_freq import lk, lm

    nelx = int(cfg["domain"]["mesh"]["nelx"])
    nely = int(cfg["domain"]["mesh"]["nely"])
    length = float(cfg["domain"]["size"]["length"])
    height = float(cfg["domain"]["size"]["height"])
    ndof = 2 * (nelx + 1) * (nely + 1)
    _, i_k, j_k = _element_data(nelx, nely)
    ke = lk(length / nelx, height / nely, float(cfg["material"]["nu"]))
    me = lm(length / nelx, height / nely)

    e0 = float(cfg["material"]["E"])
    emin = e0 * float(cfg["void_material"]["E_min_ratio"])
    rho0 = float(cfg["material"]["rho"])
    rho_min = float(cfg["void_material"]["rho_min"])
    penal = float(cfg["optimization"]["penalization"])
    pmass = 1.0
    stiffness = emin + x**penal * (e0 - emin)
    density = rho_min + x**pmass * (rho0 - rho_min)
    s_k = (ke.ravel()[:, None] * stiffness).ravel(order="F")
    s_m = (me.ravel()[:, None] * density).ravel(order="F")
    k_matrix = coo_array((s_k, (i_k, j_k)), shape=(ndof, ndof)).tocsc()
    m_matrix = coo_array((s_m, (i_k, j_k)), shape=(ndof, ndof)).tocsc()

    phi0 = np.asarray(_require(diag, "reference_modes"), dtype=float)[:, 0]
    omega0_sq = float(np.asarray(_require(diag, "reference_omega_sq"), dtype=float).reshape(-1)[0])
    force = omega0_sq * (m_matrix @ phi0)
    fixed = _fixed_dofs(cfg, nelx, nely, length, height)
    free = np.setdiff1d(np.arange(ndof, dtype=np.int64), fixed)
    displacement = np.zeros(ndof, dtype=float)
    displacement[free] = spsolve(k_matrix[free][:, free], force[free])
    objective = float(displacement @ (k_matrix @ displacement))
    if not np.isfinite(objective):
        raise AssertionError("V1a perturbed objective is not finite")
    return objective


def verify(fixture: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from tools.Python.run_topopt_from_json import run_topopt_from_json

    cfg = json.loads(fixture.read_text(encoding="utf-8"))
    run_cfg = copy.deepcopy(cfg)
    run_cfg["optimization"]["load_sensitivity"] = "complete"
    _, _, _, _, info = run_topopt_from_json(run_cfg, return_diagnostics=True)
    if not isinstance(info, dict) or "gate_a0" not in info:
        raise AssertionError("V1a complete run returned no Gate A0 diagnostics")
    diag = info["gate_a0"]
    for name in ("current_x", "complete_sensitivity", "omitted_sensitivity",
                 "reference_omega_sq", "reference_modes", "selected_load_sensitivity"):
        _require(diag, name)
    if diag["selected_load_sensitivity"] != "complete":
        raise AssertionError("V1a did not execute the complete sensitivity branch")

    x = np.asarray(diag["current_x"], dtype=float).reshape(-1)
    analytical = np.asarray(diag["complete_sensitivity"], dtype=float).reshape(-1)
    omitted = np.asarray(diag["omitted_sensitivity"], dtype=float).reshape(-1)
    if analytical.size != x.size or omitted.size != x.size:
        raise AssertionError("V1a sensitivity diagnostic has the wrong size")
    indices = np.unique(np.floor(np.linspace(0, x.size - 1, 6) + 0.5).astype(int))
    if np.any(x[indices] - FD_STEP <= 0.0) or np.any(x[indices] + FD_STEP >= 1.0):
        raise AssertionError("V1a perturbation would leave the open density interval")

    fd_values = []
    analytical_values = analytical[indices]
    omitted_values = omitted[indices]
    for index in indices:
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[index] += FD_STEP
        x_minus[index] -= FD_STEP
        fd_values.append((_objective(x_plus, cfg, diag) - _objective(x_minus, cfg, diag)) / (2.0 * FD_STEP))
    fd_values = np.asarray(fd_values)
    denominator = np.maximum(np.maximum(np.abs(fd_values), np.abs(analytical_values)), REL_FLOOR)
    relative_errors = np.abs(fd_values - analytical_values) / denominator
    omitted_denominator = np.maximum(np.maximum(np.abs(fd_values), np.abs(omitted_values)), REL_FLOOR)
    omitted_relative_errors = np.abs(fd_values - omitted_values) / omitted_denominator

    if np.any(~np.isfinite(relative_errors)):
        raise AssertionError("V1a produced non-finite complete relative errors")
    if np.any(relative_errors > REL_TOL):
        worst = int(np.argmax(relative_errors))
        raise AssertionError(
            f"V1a complete FD relative error {relative_errors[worst]:.3e} at element "
            f"{indices[worst] + 1} exceeds {REL_TOL:.1e}"
        )
    if np.allclose(analytical_values, omitted_values, rtol=REL_TOL, atol=REL_FLOOR):
        raise AssertionError("V1a omitted and complete sensitivities are unexpectedly indistinguishable")

    return {
        "status": "passed",
        "formulation": "F(x) = omega0^2 * M(x) * Phi0",
        "perturbation_size": FD_STEP,
        "relative_error_tolerance": REL_TOL,
        "tested_element_indices": (indices + 1).tolist(),
        "finite_difference_values": fd_values.tolist(),
        "complete_analytical_values": analytical_values.tolist(),
        "complete_relative_errors": relative_errors.tolist(),
        "omitted_analytical_values": omitted_values.tolist(),
        "omitted_relative_errors_vs_complete_fd": omitted_relative_errors.tolist(),
        "omitted_expected_to_match_complete_fd": False,
        "omitted_confirmation": "The omitted branch excludes the nonzero load derivative and is not expected to match the complete finite-difference derivative.",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_v1a_fd.py FIXTURE_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    fixture = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if not fixture.is_file():
        raise FileNotFoundError(f"V1a fixture not found: {fixture}")
    result = verify(fixture)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Python V1a FD verification passed; max relative error = {max(result['complete_relative_errors']):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
