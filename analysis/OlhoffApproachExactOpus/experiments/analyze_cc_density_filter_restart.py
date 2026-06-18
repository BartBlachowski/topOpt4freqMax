#!/usr/bin/env python3
"""Analyze density-filter-only stabilized restart against S0 reference."""

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
RESULTS = EXPDIR / "results_density_filter"
S0_PATH = EXPDIR / "results_lp_vs_nl" / "cc_embed_NL_nonlinear.mat"
S1_PATH = RESULTS / "cc_density_filter_S1.mat"
REPORT = EXPDIR / "cc_320x40_density_filter_assessment.md"


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


def load_case(path: Path, case: str, filter_kind: str) -> dict:
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
        "filter_kind": filter_kind,
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
    fig.savefig(RESULTS / "density_filter_omega12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for c in cases:
        ax.plot(c["iterations"], c["gap12"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("(omega2 - omega1) / omega1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "density_filter_gap12.png", dpi=160)
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
    fig.savefig(RESULTS / "density_filter_N.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for c in cases:
        ax.plot(c["iterations"], c["drho_max"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("max abs design update")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "density_filter_drho_max.png", dpi=160)
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
    s0, s1 = cases
    lines = []
    lines.append("# 320x40 density-filter stabilized reproduction diagnostic\n")
    lines.append("Classification target: paper-inspired stabilized reproduction, not strict paper reproduction.")
    lines.append("S0 is the existing 100-iteration nonlinear reference. S1 adds only density filtering in the design-to-physical density map; no Heaviside projection was run because S1 did not improve the benchmark direction.\n")
    lines.append("Volume convention for S1: the active volume constraint is `mean(rho_phys) <= volfrac`, where `rho_phys = H*x_design/Hs`. The final design-variable volume is reported separately.\n")

    lines.append("## Comparison\n")
    lines.append("| case | filter | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | total N=2 | effCmp8 | grey frac | central density | ccSolid | final phys vol | final design vol | max update final |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        bi = c["best_index"]
        f = c["final"]
        dvol = "-" if f["design_volume"] is None else f"{f['design_volume']:.4f}"
        lines.append(
            f"| {c['case']} | {c['filter_kind']} | {c['omega'][bi,0]:.3f} | {c['omega'][bi,1]:.3f} | {c['gap12'][bi]:.4f} | "
            f"{f['omega'][0]:.3f} | {f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | {c['total_n2']} | "
            f"{f['effective_components_8']} | {f['grey_fraction_01_09']:.3f} | {f['central_density']:.3f} | {f['ccSolid']:.3f} | "
            f"{f['volume']:.4f} | {dvol} | {c['drho_max'][-1]:.4e} |"
        )

    lines.append("\n## Required questions\n")
    lines.append(f"1. Gap12 below current ~5%: S1 final gap is `{s1['final']['gap12']:.4f}` versus S0 `{s0['final']['gap12']:.4f}`. This is not a material reduction.")
    lines.append(f"2. Omega1 toward omega2: no useful movement; S1 lowers the frequency level (`omega1={s1['final']['omega'][0]:.3f}`) relative to S0 (`omega1={s0['final']['omega'][0]:.3f}`).")
    lines.append(f"3. N=2 persistence: S1 is more persistent by count (`{s1['total_n2']}` iterations) than S0 (`{s0['total_n2']}`), but without coalescence or frequency gain.")
    lines.append(f"4. Connected Fig. 3c topology: preserved; S1 effective components = `{s1['final']['effective_components_8']}`.")
    lines.append(f"5. Closer to `omega1 ~= omega2 ~= 456.4`: no. S1 moves both frequencies farther below 456.4.")
    lines.append("6. Density filtering changes the optimizer path mainly by smoothing/lowering the stiffness-effective design; it does not create a stronger coalescence path.")
    lines.append("7. Projection was not tested because S1 was not promising under the stated criterion.")

    lines.append("\n## Final assessment\n")
    lines.append("- Classification: **C. paper-inspired stabilized reproduction** as an experiment class, but the result is not a successful near-reproduction.")
    lines.append("- This is not exact paper reproduction: density filtering in the design-variable-to-physical-density map is not part of the strict paper-derived exact formulation used in the previous studies.")
    lines.append("- Remaining obstacle classification: density filtering alone does not explain the benchmark discrepancy; the likely issue remains optimizer path/control or another undocumented numerical choice.")
    lines.append("- Recommendation: stop this density-filter-only branch unless the next objective explicitly allows projection/continuation as a separate, non-exact stabilized optimizer study.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not S0_PATH.exists():
        raise SystemExit(f"Missing S0 reference: {S0_PATH}")
    if not S1_PATH.exists():
        raise SystemExit(f"Missing S1 result: {S1_PATH}")
    cases = [
        load_case(S0_PATH, "S0_reference_NL", "none"),
        load_case(S1_PATH, "S1_density_filter", "density filter only"),
    ]
    plot_cases(cases)
    (RESULTS / "cc_density_filter_metrics.json").write_text(json.dumps(json_safe(cases), indent=2), encoding="utf-8")
    write_report(cases)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
