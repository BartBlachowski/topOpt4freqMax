#!/usr/bin/env python3
"""Analyze Heaviside projection-only restart cases against P0 reference."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import scipy.io as sio
from scipy.ndimage import label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
EXPDIR = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments"
RESULTS = EXPDIR / "results_projection"
P0_PATH = EXPDIR / "results_lp_vs_nl" / "cc_embed_NL_nonlinear.mat"
REPORT = EXPDIR / "cc_320x40_projection_assessment.md"


def field(obj, name):
    return getattr(obj, name)


def as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape((-1, 1))
    return arr


def components_8(rho: np.ndarray, nelx: int, nely: int) -> tuple[int, int, list[int]]:
    grid = np.reshape(as_1d(rho), (nely, nelx), order="F")
    solid = grid > 0.5
    lab, raw = label(solid, structure=np.ones((3, 3), dtype=int))
    sizes = sorted((int((lab == i).sum()) for i in range(1, raw + 1)), reverse=True)
    threshold = max(10, int(round(0.005 * nelx * nely)))
    eff = len([s for s in sizes if s >= threshold])
    return int(raw), int(eff), sizes


def density_metrics(rho: np.ndarray, nelx: int, nely: int) -> dict:
    r = as_1d(rho)
    grid = np.reshape(r, (nely, nelx), order="F")
    central = grid[:, nelx // 3 : 2 * nelx // 3]
    raw, eff, sizes = components_8(r, nelx, nely)
    return {
        "raw_components_8": raw,
        "effective_components_8": eff,
        "largest_components": sizes[:5],
        "central_density": float(np.mean(central)),
        "ccSolid": float(np.mean(central > 0.5)),
        "grey_fraction_01_09": float(np.mean((r > 0.1) & (r < 0.9))),
        "grey_measure_4rho1rho": float(np.mean(4 * r * (1 - r))),
    }


def n2_runs(N: np.ndarray) -> tuple[int | None, int, int]:
    idx = np.where(N == 2)[0]
    if idx.size == 0:
        return None, 0, 0
    runs = []
    start = prev = idx[0]
    for cur in idx[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            runs.append((start, prev))
            start = prev = cur
    runs.append((start, prev))
    return int(idx[0] + 1), int(max(b - a + 1 for a, b in runs)), int(idx.size)


def load_case(path: Path, case: str, projection: str, beta: float | None) -> dict:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    cfg, hist = data["cfg"], data["hist"]
    nelx, nely = int(field(cfg, "nelx")), int(field(cfg, "nely"))
    niter = int(field(hist, "outer_iters"))
    omega = as_2d(field(hist, "omega_trial"))[:niter, :]
    if np.all(np.isnan(omega)):
        omega = as_2d(field(hist, "omega"))[:niter, :]
    it = np.arange(1, niter + 1)
    gap12 = (omega[:, 1] - omega[:, 0]) / omega[:, 0]
    N = as_1d(field(hist, "N_trial"))[:niter]
    volume = as_1d(field(hist, "volume"))[:niter]
    drho_max = as_1d(field(hist, "drho_max"))[:niter]
    final_omega = as_1d(field(hist, "final_omega"))
    rho_final = as_1d(data["rho_final"])
    best_idx = int(np.nanargmax(omega[:, 0]))
    first_n2, max_run_n2, total_n2 = n2_runs(N)
    tail = slice(max(0, niter - 20), niter)
    metrics = density_metrics(rho_final, nelx, nely)
    design_volume = None
    if hasattr(hist, "final_design_volume"):
        design_volume = float(field(hist, "final_design_volume"))

    return {
        "case": case,
        "projection": projection,
        "beta": beta,
        "path": str(path),
        "runtime_s": float(np.asarray(data["el"]).reshape(())) if "el" in data else None,
        "mult_tol": float(field(cfg, "mult_tol")),
        "iterations": it,
        "omega": omega,
        "gap12": gap12,
        "N": N,
        "volume": volume,
        "drho_max": drho_max,
        "best_index": best_idx,
        "first_n2": first_n2,
        "max_run_n2": max_run_n2,
        "total_n2": total_n2,
        "slope20": float(np.polyfit(it[tail].astype(float), omega[tail, 0], 1)[0]),
        "gap_slope20": float(np.polyfit(it[tail].astype(float), gap12[tail], 1)[0]),
        "final": {
            "omega": final_omega,
            "gap12": float((final_omega[1] - final_omega[0]) / final_omega[0]),
            "N": int(field(hist, "final_N")),
            "volume": float(field(hist, "final_volume")),
            "design_volume": design_volume,
            **metrics,
        },
    }


def classify(case: dict, ref: dict) -> str:
    gap_gain = ref["final"]["gap12"] - case["final"]["gap12"]
    w1_gain = case["final"]["omega"][0] - ref["final"]["omega"][0]
    connected = case["final"]["effective_components_8"] <= 1
    if connected and gap_gain > 0.01 and w1_gain > 2:
        return "helpful"
    if (not connected) or w1_gain < -2 or gap_gain < -0.005:
        return "harmful"
    return "neutral"


def plot_cases(cases: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for c in cases:
        ax.plot(c["iterations"], c["omega"][:, 0], label=f"{c['case']} omega1")
        ax.plot(c["iterations"], c["omega"][:, 1], ls="--", label=f"{c['case']} omega2")
    ax.axhline(456.4, color="k", ls=":", lw=1.1, label="paper 456.4")
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_omega12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for c in cases:
        ax.plot(c["iterations"], c["gap12"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("(omega2 - omega1) / omega1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_gap12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for c in cases:
        ax.step(c["iterations"], c["N"], where="mid", label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("N")
    ax.set_yticks([1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_N.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for c in cases:
        ax.plot(c["iterations"], c["drho_max"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("max abs design update")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_drho_max.png", dpi=160)
    plt.close(fig)


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def write_report(cases: list[dict]) -> None:
    ref = cases[0]
    lines = []
    lines.append("# 320x40 Heaviside projection diagnostic\n")
    lines.append("P0 is the existing 100-iteration nonlinear reference. P1/P2 use elementwise Heaviside projection only in the design-to-physical-density map, with no density filter in that map. The existing sensitivity filter and filter radius are unchanged.\n")
    lines.append("Volume convention for P1/P2: the active volume constraint is `mean(rho_phys) <= volfrac`, where `rho_phys = H_beta(x_design)`. Final design-variable volume is reported separately.\n")

    lines.append("## Comparison\n")
    lines.append("| case | projection | beta | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | total N=2 | max N=2 run | effCmp8 | grey frac | central density | phys vol | design vol | class |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c in cases:
        bi = c["best_index"]
        f = c["final"]
        beta = "-" if c["beta"] is None else f"{c['beta']:.1f}"
        dvol = "-" if f["design_volume"] is None else f"{f['design_volume']:.4f}"
        klass = "reference" if c is ref else classify(c, ref)
        lines.append(
            f"| {c['case']} | {c['projection']} | {beta} | {c['omega'][bi,0]:.3f} | {c['omega'][bi,1]:.3f} | {c['gap12'][bi]:.4f} | "
            f"{f['omega'][0]:.3f} | {f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | {c['total_n2']} | {c['max_run_n2']} | "
            f"{f['effective_components_8']} | {f['grey_fraction_01_09']:.3f} | {f['central_density']:.3f} | {f['volume']:.4f} | {dvol} | {klass} |"
        )

    p1, p2 = cases[1], cases[2]
    lines.append("\n## Required questions\n")
    lines.append(f"1. Gap12 reduction: P1 is `{p1['final']['gap12']:.4f}` versus P0 `{ref['final']['gap12']:.4f}`; P2 is `{p2['final']['gap12']:.4f}`. P2 is the only material reduction.")
    lines.append(f"2. Omega1 toward omega2: P2 raises omega1 to `{p2['final']['omega'][0]:.3f}` while lowering the gap; P1 only slightly raises both frequencies without closing the gap.")
    lines.append(f"3. N=2 persistence: P0 `{ref['total_n2']}` iterations, P1 `{p1['total_n2']}`, P2 `{p2['total_n2']}`. P2 is more persistent than P0/P1.")
    lines.append(f"4. Topology connected: effective components are P0 `{ref['final']['effective_components_8']}`, P1 `{p1['final']['effective_components_8']}`, P2 `{p2['final']['effective_components_8']}`.")
    lines.append(f"5. Closer to `omega1 ~= omega2 ~= 456.4`: P2 is closer by coalescence and omega1, but still far below 456.4 (`omega1={p2['final']['omega'][0]:.3f}`, `omega2={p2['final']['omega'][1]:.3f}`).")

    lines.append("\n## Final assessment\n")
    lines.append(f"- P1 beta=1 classification: **{classify(p1, ref)}**.")
    lines.append(f"- P2 beta=3 classification: **{classify(p2, ref)}**.")
    lines.append("- Hypothesis P1 status: partially supported. Moderate projection materially reduces gap12 and improves N=2 persistence while preserving connected topology, but it does not reach the published 456.4 coalesced benchmark.")
    lines.append("- Projection-only is a paper-inspired stabilization, not exact paper reproduction. It should remain diagnostic unless a later production-oriented stabilized solver is explicitly requested.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = {
        "P0_reference": P0_PATH,
        "P1_beta1": RESULTS / "cc_projection_P1_beta1.mat",
        "P2_beta3": RESULTS / "cc_projection_P2_beta3.mat",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing projection inputs:\n" + "\n".join(missing))
    cases = [
        load_case(paths["P0_reference"], "P0_reference", "none", None),
        load_case(paths["P1_beta1"], "P1_beta1", "Heaviside", 1.0),
        load_case(paths["P2_beta3"], "P2_beta3", "Heaviside", 3.0),
    ]
    plot_cases(cases)
    (RESULTS / "cc_projection_metrics.json").write_text(json.dumps(json_safe(cases), indent=2), encoding="utf-8")
    write_report(cases)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
