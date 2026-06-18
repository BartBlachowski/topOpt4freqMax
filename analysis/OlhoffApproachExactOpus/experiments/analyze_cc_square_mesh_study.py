#!/usr/bin/env python3
"""Analyze CC square-element mesh study outputs.

Reads results written by run_cc_square_mesh_study.m and computes morphology
metrics without changing solver data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import scipy.io as sio
from scipy.ndimage import label


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments" / "results_square_mesh"
REPORT = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments" / "cc_square_mesh_study_result.md"
JSON_OUT = RESULTS / "cc_square_mesh_metrics.json"


@dataclass
class Morphology:
    components_4: int
    components_8: int
    effective_components_8: int
    effective_component_threshold: int
    component_sizes_8: list[int]
    central_density: float
    cc_solid: float
    min_central_col_solid: int
    max_central_col_solid: int
    diagonal_braces: bool
    brace_score: float
    topology_class: str
    fig3c_similarity: float
    qualitative_similarity: str


def matlab_struct_field(obj, name):
    return getattr(obj, name)


def as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape((-1, 1))
    return arr


def density_grid(rho: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    return np.reshape(as_1d(rho), (nely, nelx), order="F")


def connected_components(binary: np.ndarray, conn: int) -> tuple[int, list[int]]:
    if conn == 8:
        structure = np.ones((3, 3), dtype=int)
    elif conn == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    else:
        raise ValueError(conn)
    lab, n = label(binary.astype(bool), structure=structure)
    sizes = sorted((int((lab == i).sum()) for i in range(1, n + 1)), reverse=True)
    return int(n), sizes


def brace_score(grid: np.ndarray) -> tuple[bool, float]:
    """Heuristic for X-bracing in a long beam.

    A cross-braced topology should carry material in both upper and lower half
    across the middle span and show left-right crossing tendencies.  This is
    intentionally conservative and reported as a heuristic, not a classifier
    trained on images.
    """
    nely, nelx = grid.shape
    x0, x1 = nelx // 4, 3 * nelx // 4
    mid_cols = grid[:, x0:x1]
    if mid_cols.size == 0:
        return False, 0.0

    lower = mid_cols[: max(1, nely // 2), :]
    upper = mid_cols[nely // 2 :, :]
    lower_cols = np.mean(lower > 0.35, axis=0)
    upper_cols = np.mean(upper > 0.35, axis=0)
    both_halves = float(np.mean((lower_cols > 0) & (upper_cols > 0)))

    # Weighted vertical centroid variation across columns; X/truss members
    # typically create a non-flat centroid path through the middle span.
    weights = np.maximum(mid_cols - 0.15, 0.0)
    y = np.linspace(0.0, 1.0, nely)[:, None]
    col_mass = weights.sum(axis=0)
    valid = col_mass > 1e-9
    if np.any(valid):
        centroid = (weights * y).sum(axis=0)[valid] / col_mass[valid]
        centroid_span = float(np.max(centroid) - np.min(centroid))
        coverage = float(np.mean(valid))
    else:
        centroid_span = 0.0
        coverage = 0.0

    score = 0.45 * both_halves + 0.35 * min(1.0, centroid_span / 0.35) + 0.20 * coverage
    return bool(score >= 0.45 and coverage >= 0.45), float(score)


def classify_morphology(rho: np.ndarray, nelx: int, nely: int) -> Morphology:
    g = density_grid(rho, nelx, nely)
    solid = g > 0.5
    comp4, _ = connected_components(solid, 4)
    comp8, sizes8 = connected_components(solid, 8)
    eff_threshold = max(10, int(round(0.005 * nelx * nely)))
    eff_sizes8 = [s for s in sizes8 if s >= eff_threshold]
    eff_comp8 = len(eff_sizes8)

    c0, c1 = nelx // 3, 2 * nelx // 3
    central = g[:, c0:c1]
    central_density = float(np.mean(central))
    central_solid = central > 0.5
    cc_solid = float(np.mean(central_solid))
    colsolid = np.sum(central_solid, axis=0)
    min_col = int(np.min(colsolid)) if colsolid.size else 0
    max_col = int(np.max(colsolid)) if colsolid.size else 0

    braces, bscore = brace_score(g)

    if eff_comp8 >= 2 and cc_solid < 0.08:
        tclass = "disconnected blocks"
    elif eff_comp8 <= 1 and braces:
        tclass = "connected cross-braced truss"
    elif eff_comp8 <= 1:
        tclass = "connected frame"
    else:
        tclass = "other"

    connected_score = 1.0 if eff_comp8 <= 1 else 0.0
    central_score = min(1.0, central_density / 0.30)
    solid_score = min(1.0, cc_solid / 0.20)
    similarity = 0.40 * connected_score + 0.35 * bscore + 0.15 * central_score + 0.10 * solid_score

    if similarity >= 0.75:
        qualitative = "high"
    elif similarity >= 0.50:
        qualitative = "moderate"
    elif similarity >= 0.25:
        qualitative = "low"
    else:
        qualitative = "very low"

    return Morphology(
        components_4=comp4,
        components_8=comp8,
        effective_components_8=eff_comp8,
        effective_component_threshold=eff_threshold,
        component_sizes_8=sizes8,
        central_density=central_density,
        cc_solid=cc_solid,
        min_central_col_solid=min_col,
        max_central_col_solid=max_col,
        diagonal_braces=braces,
        brace_score=bscore,
        topology_class=tclass,
        fig3c_similarity=similarity,
        qualitative_similarity=qualitative,
    )


def load_case(path: Path) -> dict:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    cfg = data["cfg"]
    hist = data["hist"]
    rho = as_1d(data["rho_final"])
    nelx = int(matlab_struct_field(cfg, "nelx"))
    nely = int(matlab_struct_field(cfg, "nely"))
    omega = as_2d(matlab_struct_field(hist, "omega"))
    omega_trial = as_2d(matlab_struct_field(hist, "omega_trial"))
    if np.all(np.isnan(omega_trial)):
        omega_for_best = omega
    else:
        omega_for_best = omega_trial
    w1 = omega_for_best[:, 0]
    best_idx = int(np.nanargmax(w1))
    volume = as_1d(matlab_struct_field(hist, "volume"))
    n_trial = as_1d(matlab_struct_field(hist, "N_trial"))
    final_omega = as_1d(matlab_struct_field(hist, "final_omega"))
    initial_omega = omega[0, :3]
    morph = classify_morphology(rho, nelx, nely)

    return {
        "mesh": f"{nelx}x{nely}",
        "nelx": nelx,
        "nely": nely,
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "outer_iters": int(matlab_struct_field(hist, "outer_iters")),
        "initial_omega": [float(v) for v in initial_omega],
        "best": {
            "iteration": best_idx + 1,
            "omega1": float(omega_for_best[best_idx, 0]),
            "omega2": float(omega_for_best[best_idx, 1]),
            "N": int(n_trial[best_idx]) if np.isfinite(n_trial[best_idx]) else None,
            "volume": float(volume[best_idx]),
        },
        "final": {
            "omega1": float(final_omega[0]),
            "omega2": float(final_omega[1]),
            "N": int(matlab_struct_field(hist, "final_N")),
            "volume": float(matlab_struct_field(hist, "final_volume")),
        },
        "morphology": morph.__dict__,
    }


def fmt3(vals: list[float]) -> str:
    return ", ".join(f"{v:.3f}" for v in vals[:3])


def write_report(rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# CC square-element mesh resolution study\n")
    lines.append("Configuration: audit-baseline `run_cc_meshcompare_b.m`; only `nelx,nely` changed. Domain remains L=8, H=1. `rmin_elem=2.5` is held as the existing solver setting.\n")
    lines.append("## Measurements\n")
    for r in rows:
        m = r["morphology"]
        lines.append(f"### Mesh {r['mesh']}\n")
        lines.append(f"- Runtime: {r['runtime_s']:.1f} s; outer iterations: {r['outer_iters']}")
        lines.append(f"- Initial omega: {fmt3(r['initial_omega'])}")
        lines.append(f"- Best design: iter {r['best']['iteration']}, omega1={r['best']['omega1']:.3f}, omega2={r['best']['omega2']:.3f}, N={r['best']['N']}, volume={r['best']['volume']:.4f}")
        lines.append(f"- Final design: omega1={r['final']['omega1']:.3f}, omega2={r['final']['omega2']:.3f}, N={r['final']['N']}, volume={r['final']['volume']:.4f}")
        lines.append(f"- Morphology: {m['topology_class']}; raw components 4/8={m['components_4']}/{m['components_8']}; effective 8-components={m['effective_components_8']} (ignore islands <{m['effective_component_threshold']} elems); ccSolid={m['cc_solid']:.3f}; central density={m['central_density']:.3f}; diagonal braces={m['diagonal_braces']} (score={m['brace_score']:.3f})")
        lines.append(f"- Fig. 3c similarity: {m['qualitative_similarity']} ({m['fig3c_similarity']:.3f})\n")

    lines.append("## Convergence table\n")
    lines.append("| mesh | connected? | ccSolid | omega1best | omega1final | Nfinal |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["morphology"]
        connected = "yes" if m["effective_components_8"] <= 1 else "no"
        lines.append(f"| {r['mesh']} | {connected} | {m['cc_solid']:.3f} | {r['best']['omega1']:.3f} | {r['final']['omega1']:.3f} | {r['final']['N']} |")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = [
        p for p in RESULTS.glob("cc_square_*.mat")
        if p.stem.split("_")[-1].split("x")[0].isdigit()
    ]
    paths = sorted(paths, key=lambda p: int(p.stem.split("_")[-1].split("x")[0]))
    if not paths:
        raise SystemExit(f"No result files found in {RESULTS}")
    rows = [load_case(p) for p in paths]
    RESULTS.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(rows)
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
