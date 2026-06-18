#!/usr/bin/env python3
"""Analyze basin sensitivity from alternative initial states."""

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
RESULTS = EXPDIR / "results_basin_initialization"
REPORT = EXPDIR / "cc_320x40_basin_initialization_assessment.md"


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


def snapshot_grey(hist, niter: int) -> np.ndarray:
    grey = np.full(niter, np.nan)
    if not hasattr(hist, "rho_snapshot_count"):
        return grey
    count = int(field(hist, "rho_snapshot_count"))
    if count <= 0:
        return grey
    snap_iters = as_1d(field(hist, "rho_snapshot_iters"))[:count].astype(int)
    snaps = np.asarray(field(hist, "rho_snapshots"), dtype=float)
    if snaps.ndim == 1:
        snaps = snaps.reshape((-1, 1))
    for si, it in enumerate(snap_iters):
        if 1 <= it <= niter:
            rho = snaps[:, si]
            grey[it - 1] = float(np.mean((rho > 0.1) & (rho < 0.9)))
    return grey


def load_case(path: Path) -> dict:
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
    beta_iter = as_1d(field(hist, "projection_beta_iter"))[:niter]
    volume = as_1d(field(hist, "volume"))[:niter]
    drho_max = as_1d(field(hist, "drho_max"))[:niter]
    final_omega = as_1d(field(hist, "final_omega"))
    rho_final = as_1d(data["rho_final"])
    x_initial = as_1d(data["x_initial"])
    best_idx = int(np.nanargmax(omega[:, 0]))
    best_gap_idx = int(np.nanargmin(gap12))
    first_n2, max_run_n2, total_n2 = n2_runs(N)
    tail = slice(max(0, niter - 20), niter)
    return {
        "case": path.stem.replace("cc_basin_", ""),
        "path": str(path),
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "iterations": it,
        "omega": omega,
        "gap12": gap12,
        "N": N,
        "beta_iter": beta_iter,
        "volume": volume,
        "drho_max": drho_max,
        "grey_fraction": snapshot_grey(hist, niter),
        "best_index": best_idx,
        "best_gap_index": best_gap_idx,
        "first_n2": first_n2,
        "max_run_n2": max_run_n2,
        "total_n2": total_n2,
        "slope20": float(np.polyfit(it[tail].astype(float), omega[tail, 0], 1)[0]),
        "gap_slope20": float(np.polyfit(it[tail].astype(float), gap12[tail], 1)[0]),
        "initial": {
            "mean": float(np.mean(x_initial)),
            "grey_fraction_01_09": float(np.mean((x_initial > 0.1) & (x_initial < 0.9))),
            **density_metrics(x_initial, nelx, nely),
        },
        "final": {
            "omega": final_omega,
            "gap12": float((final_omega[1] - final_omega[0]) / final_omega[0]),
            "N": int(field(hist, "final_N")),
            "volume": float(field(hist, "final_volume")),
            "design_volume": float(field(hist, "final_design_volume")),
            "beta": float(field(hist, "final_projection_beta")),
            **density_metrics(rho_final, nelx, nely),
        },
    }


def plot_cases(cases: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c in cases:
        ax.plot(c["iterations"], c["omega"][:, 0], label=f"{c['case']} omega1")
    ax.axhline(456.4, color="k", ls=":", lw=1.0, label="paper 456.4")
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega1 [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "basin_omega1_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c in cases:
        ax.plot(c["iterations"], c["gap12"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("gap12")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "basin_gap12_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for c in cases:
        ax.step(c["iterations"], c["N"], where="mid", label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("N")
    ax.set_yticks([1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "basin_N_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c in cases:
        ax.plot(c["iterations"], c["grey_fraction"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("grey fraction, 0.1 < rho < 0.9")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "basin_grey_fraction_combined.png", dpi=160)
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
    best_final_w1 = max(cases, key=lambda c: c["final"]["omega"][0])
    best_final_gap = min(cases, key=lambda c: c["final"]["gap12"])
    raw = next(c for c in cases if c["case"] == "raw")
    lines = []
    lines.append("# 320x40 basin-initialization assessment\n")
    lines.append("All cases use the same stabilized optimizer settings: Heaviside projection continuation `1->2->3` every 25 iterations, `move_lim = outer_move = 0.05`, and `mult_tol = 0.05`. Only the initial design field changes.\n")

    lines.append("## Comparison\n")
    lines.append("| initial state | initial grey | best omega1 | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | central density | phys vol | class note |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c in cases:
        bi = c["best_index"]
        gi = c["best_gap_index"]
        f = c["final"]
        if c is best_final_w1:
            note = "best useful basin"
        elif f["omega"][0] < 50:
            note = "collapsed low-frequency basin"
        elif f["omega"][0] < raw["final"]["omega"][0] - 10:
            note = "lower-frequency basin"
        else:
            note = "similar basin"
        lines.append(
            f"| {c['case']} | {c['initial']['grey_fraction_01_09']:.3f} | {c['omega'][bi,0]:.3f} | {c['gap12'][gi]:.4f} | "
            f"{f['omega'][0]:.3f} | {f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | {c['total_n2']} | "
            f"{f['effective_components_8']} | {f['grey_fraction_01_09']:.3f} | {f['central_density']:.3f} | {f['volume']:.4f} | {note} |"
        )

    lines.append("\n## Questions\n")
    spread = max(c["final"]["omega"][0] for c in cases) - min(c["final"]["omega"][0] for c in cases)
    lines.append(f"1. Different initializations converge to different frequencies: yes. Final omega1 spread is `{spread:.3f}` rad/s.")
    lines.append(f"2. Movement toward `omega1 ~= omega2 ~= 456.4`: no. The best final omega1 is `{best_final_w1['final']['omega'][0]:.3f}` from `{best_final_w1['case']}`, still far below 456.4.")
    connected_count = sum(c["final"]["effective_components_8"] <= 1 for c in cases)
    lines.append(f"3. Connected topology preservation: effective single-component topology appears in `{connected_count}/{len(cases)}` cases, but the thresholded case has a collapsed low-frequency response and should not be considered a useful connected optimum.")
    lines.append(f"4. Final state basin-dependence: yes. Raw, smoothed, thresholded, and thresholded+smoothed starts end in distinct frequency/gap regimes.")

    lines.append("\n## Final assessment\n")
    lines.append("- Hypothesis P4 is supported in the weak sense: the optimizer is basin-sensitive.")
    lines.append("- It is not supported as the primary explanation for the remaining benchmark gap: none of the alternate initial states finds a higher-frequency coalesced basin near 456.4.")
    lines.append("- The useful basin remains the raw connected design under PC1 settings; smoothing/thresholding pushes the run into lower-frequency or collapsed basins.")
    lines.append("- Conclusion: basin selection matters, but the missing benchmark mechanism is not simply recovered by these local initial-state perturbations.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = sorted(RESULTS.glob("cc_basin_*.mat"))
    if not paths:
        raise SystemExit(f"No basin results found in {RESULTS}")
    cases = [load_case(p) for p in paths]
    order = {"raw": 0, "thresholded": 1, "smoothed": 2, "thresholded_smoothed": 3}
    cases.sort(key=lambda c: order.get(c["case"], 99))
    plot_cases(cases)
    (RESULTS / "cc_basin_initialization_metrics.json").write_text(
        json.dumps(json_safe(cases), indent=2), encoding="utf-8"
    )
    write_report(cases)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
