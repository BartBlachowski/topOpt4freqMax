#!/usr/bin/env python3
"""Analyze 320x40 long-horizon convergence-to-coalescence runs."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import scipy.io as sio
from scipy.ndimage import label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments" / "results_long_horizon"
REPORT = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments" / "cc_320x40_long_horizon_assessment.md"


def as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape((-1, 1))
    return arr


def field(obj, name):
    return getattr(obj, name)


def components_8(rho: np.ndarray, nelx: int, nely: int) -> tuple[int, int, list[int]]:
    g = np.reshape(as_1d(rho), (nely, nelx), order="F")
    solid = g > 0.5
    lab, n = label(solid, structure=np.ones((3, 3), dtype=int))
    sizes = sorted((int((lab == i).sum()) for i in range(1, n + 1)), reverse=True)
    threshold = max(10, int(round(0.005 * nelx * nely)))
    effective = len([s for s in sizes if s >= threshold])
    return int(n), effective, sizes


def central_metrics(rho: np.ndarray, nelx: int, nely: int) -> tuple[float, float]:
    g = np.reshape(as_1d(rho), (nely, nelx), order="F")
    c0, c1 = nelx // 3, 2 * nelx // 3
    central = g[:, c0:c1]
    return float(np.mean(central)), float(np.mean(central > 0.5))


def load_run(path: Path) -> dict:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    cfg = data["cfg"]
    hist = data["hist"]
    nelx, nely = int(field(cfg, "nelx")), int(field(cfg, "nely"))
    omega = as_2d(field(hist, "omega_trial"))
    if np.all(np.isnan(omega)):
        omega = as_2d(field(hist, "omega"))
    niter = int(field(hist, "outer_iters"))
    omega = omega[:niter, :]
    it = np.arange(1, niter + 1)
    volume = as_1d(field(hist, "volume"))[:niter]
    ntrial = as_1d(field(hist, "N_trial"))[:niter]
    drho_norm = as_1d(field(hist, "drho_norm"))[:niter]
    drho_max = as_1d(field(hist, "drho_max"))[:niter]
    gap12 = (omega[:, 1] - omega[:, 0]) / omega[:, 0]
    gap23 = (omega[:, 2] - omega[:, 1]) / omega[:, 1]

    snap_iters = []
    snap_rows = []
    if hasattr(hist, "rho_snapshot_count"):
        count = int(field(hist, "rho_snapshot_count"))
        if count > 0:
            snap_iters = [int(v) for v in as_1d(field(hist, "rho_snapshot_iters"))[:count]]
            snaps = np.asarray(field(hist, "rho_snapshots"), dtype=float)
            if snaps.ndim == 1:
                snaps = snaps.reshape((-1, 1))
            snaps = snaps[:, :count]
            for si, siter in enumerate(snap_iters):
                raw, eff, sizes = components_8(snaps[:, si], nelx, nely)
                cden, ccsolid = central_metrics(snaps[:, si], nelx, nely)
                idx = siter - 1
                snap_rows.append({
                    "iteration": siter,
                    "omega1": float(omega[idx, 0]),
                    "omega2": float(omega[idx, 1]),
                    "omega3": float(omega[idx, 2]),
                    "gap12": float(gap12[idx]),
                    "gap23": float(gap23[idx]),
                    "N": int(ntrial[idx]) if np.isfinite(ntrial[idx]) else None,
                    "volume": float(volume[idx]),
                    "drho_max": float(drho_max[idx]),
                    "raw_components_8": raw,
                    "effective_components_8": eff,
                    "largest_components": sizes[:5],
                    "central_density": cden,
                    "ccSolid": ccsolid,
                })

    rho_final = as_1d(data["rho_final"])
    raw_final, eff_final, sizes_final = components_8(rho_final, nelx, nely)
    cden_final, ccsolid_final = central_metrics(rho_final, nelx, nely)

    best_idx = int(np.nanargmax(omega[:, 0]))
    tail = slice(max(0, niter - 20), niter)
    x_tail = it[tail].astype(float)
    y_tail = omega[tail, 0]
    if len(x_tail) >= 2:
        slope20 = float(np.polyfit(x_tail, y_tail, 1)[0])
    else:
        slope20 = float("nan")

    return {
        "path": str(path),
        "nelx": nelx,
        "nely": nely,
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "iterations": it,
        "omega": omega,
        "volume": volume,
        "N": ntrial,
        "drho_norm": drho_norm,
        "drho_max": drho_max,
        "gap12": gap12,
        "gap23": gap23,
        "snapshots": snap_rows,
        "best_index": best_idx,
        "slope20": slope20,
        "final": {
            "omega": as_1d(field(hist, "final_omega")),
            "N": int(field(hist, "final_N")),
            "volume": float(field(hist, "final_volume")),
            "raw_components_8": raw_final,
            "effective_components_8": eff_final,
            "largest_components": sizes_final[:5],
            "central_density": cden_final,
            "ccSolid": ccsolid_final,
        },
    }


def plot_run(run: dict, stem: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    it = run["iterations"]
    omega = run["omega"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, lab in enumerate(["omega1", "omega2", "omega3"]):
        ax.plot(it, omega[:, j], label=lab)
    ax.axhline(456.4, color="k", ls="--", lw=1, label="paper 456.4")
    ax.set_xlabel("iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_omega.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(it, run["gap12"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("(omega2 - omega1) / omega1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_gap12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(it, run["volume"])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("volume")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_volume.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(it, run["drho_max"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("max abs Delta rho")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_drho_max.png", dpi=160)
    plt.close(fig)


def write_report(run: dict, stem: str) -> None:
    best = run["best_index"]
    omega = run["omega"]
    final = run["final"]
    last = len(run["iterations"]) - 1
    gap_last = run["gap12"][last]
    gap_best = run["gap12"][best]
    connected = final["effective_components_8"] <= 1
    slope20 = run["slope20"]

    if abs(slope20) < 0.05:
        freq_status = "plateauing"
    elif slope20 > 0:
        freq_status = "converging upward"
    else:
        freq_status = "declining/oscillating"

    lines = []
    lines.append("# 320x40 long-horizon convergence assessment\n")
    lines.append(f"Source: `{Path(run['path']).name}`")
    lines.append(f"Runtime: {run['runtime_s']:.1f} s; iterations: {len(run['iterations'])}\n")
    lines.append("## Summary\n")
    lines.append(f"- Best: iter {best+1}, omega1={omega[best,0]:.3f}, omega2={omega[best,1]:.3f}, omega3={omega[best,2]:.3f}, gap12={gap_best:.4f}, N={int(run['N'][best])}")
    lines.append(f"- Final: omega1={final['omega'][0]:.3f}, omega2={final['omega'][1]:.3f}, omega3={final['omega'][2]:.3f}, gap12={gap_last:.4f}, N={final['N']}, volume={final['volume']:.4f}")
    lines.append(f"- Final topology: effective components={final['effective_components_8']}, raw components={final['raw_components_8']}, ccSolid={final['ccSolid']:.3f}, central density={final['central_density']:.3f}")
    lines.append(f"- Last-20-iteration omega1 slope: {slope20:.4f} rad/s per iteration ({freq_status})\n")

    lines.append("## Snapshot table\n")
    lines.append("| iter | omega1 | omega2 | omega3 | gap12 | gap23 | N | volume | maxAbsDrho | effCmp8 | ccSolid |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in run["snapshots"]:
        lines.append(
            f"| {s['iteration']} | {s['omega1']:.3f} | {s['omega2']:.3f} | {s['omega3']:.3f} | "
            f"{s['gap12']:.4f} | {s['gap23']:.4f} | {s['N']} | {s['volume']:.4f} | "
            f"{s['drho_max']:.4e} | {s['effective_components_8']} | {s['ccSolid']:.3f} |"
        )

    lines.append("\n## Final assessment\n")
    lines.append(f"A. Connected topology status: {'preserved' if connected else 'degraded/disconnected'}.")
    lines.append(f"B. Frequency status: {freq_status}.")
    if final["N"] >= 2:
        mult = "achieved"
    elif gap_last < 0.02:
        mult = "approaching"
    else:
        mult = "not approaching"
    lines.append(f"C. Multiplicity status: {mult}.")
    if final["N"] >= 2 and abs(final["omega"][0] - 456.4) < 15:
        prob = 0.85
    elif connected and slope20 > 0.1:
        prob = 0.55
    elif connected and gap_last < 0.15:
        prob = 0.40
    else:
        prob = 0.25
    lines.append(f"D. Estimated probability that omega1 ~= omega2 ~= 456.4 is reachable under unchanged settings: {prob:.2f}.")
    lines.append("E. Most likely remaining obstacle if not reached: the unchanged optimizer path is not driving the connected branch to eigenvalue coalescence; the first mode remains separated from the second despite active volume and stable connectivity.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("matfile", nargs="?", default=None)
    args = ap.parse_args()

    if args.matfile:
        path = Path(args.matfile)
    else:
        candidates = sorted(RESULTS.glob("cc_long_320x40_*iter.mat"))
        if not candidates:
            raise SystemExit(f"No long-horizon result found in {RESULTS}")
        path = candidates[-1]

    run = load_run(path)
    stem = path.stem
    plot_run(run, stem)
    serial = {
        k: v for k, v in run.items()
        if k not in {"iterations", "omega", "volume", "N", "drho_norm", "drho_max", "gap12", "gap23"}
    }
    serial["best"] = {
        "iteration": int(run["best_index"] + 1),
        "omega": [float(x) for x in run["omega"][run["best_index"], :3]],
        "gap12": float(run["gap12"][run["best_index"]]),
        "N": int(run["N"][run["best_index"]]),
    }
    serial["last"] = {
        "iteration": int(run["iterations"][-1]),
        "omega": [float(x) for x in run["omega"][-1, :3]],
        "gap12": float(run["gap12"][-1]),
        "gap23": float(run["gap23"][-1]),
        "N": int(run["N"][-1]),
        "volume": float(run["volume"][-1]),
        "drho_max": float(run["drho_max"][-1]),
    }
    (RESULTS / f"{stem}_metrics.json").write_text(json.dumps(json_safe(serial), indent=2), encoding="utf-8")
    write_report(run, stem)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots in {RESULTS}")


if __name__ == "__main__":
    main()
