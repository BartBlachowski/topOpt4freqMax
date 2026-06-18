#!/usr/bin/env python3
"""Analyze local move-control restart cases A-C."""

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
RESULTS = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments" / "results_local_move"
REPORT = ROOT / "analysis" / "OlhoffApproachExactOpus" / "experiments" / "cc_320x40_local_move_restart_assessment.md"


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
    g = np.reshape(as_1d(rho), (nely, nelx), order="F")
    solid = g > 0.5
    lab, n = label(solid, structure=np.ones((3, 3), dtype=int))
    sizes = sorted((int((lab == i).sum()) for i in range(1, n + 1)), reverse=True)
    threshold = max(10, int(round(0.005 * nelx * nely)))
    eff = len([s for s in sizes if s >= threshold])
    return int(n), eff, sizes


def central_metrics(rho: np.ndarray, nelx: int, nely: int) -> tuple[float, float]:
    g = np.reshape(as_1d(rho), (nely, nelx), order="F")
    c0, c1 = nelx // 3, 2 * nelx // 3
    central = g[:, c0:c1]
    return float(np.mean(central)), float(np.mean(central > 0.5))


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
    vol = as_1d(field(hist, "volume"))[:niter]
    ntrial = as_1d(field(hist, "N_trial"))[:niter]
    drho_max = as_1d(field(hist, "drho_max"))[:niter]
    final_omega = as_1d(field(hist, "final_omega"))
    rho_final = as_1d(data["rho_final"])
    raw, eff, sizes = components_8(rho_final, nelx, nely)
    cden, ccsolid = central_metrics(rho_final, nelx, nely)
    best_idx = int(np.nanargmax(omega[:, 0]))
    tail = slice(max(0, niter - 20), niter)
    slope20 = float(np.polyfit(it[tail].astype(float), omega[tail, 0], 1)[0])
    gap_slope20 = float(np.polyfit(it[tail].astype(float), gap12[tail], 1)[0])
    osc20 = float(np.std(omega[tail, 0]))
    return {
        "path": str(path),
        "case": path.stem.replace("cc_local_", ""),
        "move": float(field(cfg, "move_lim")),
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "iterations": it,
        "omega": omega,
        "gap12": gap12,
        "volume": vol,
        "N": ntrial,
        "drho_max": drho_max,
        "best_index": best_idx,
        "slope20": slope20,
        "gap_slope20": gap_slope20,
        "osc20": osc20,
        "final": {
            "omega": final_omega,
            "gap12": float((final_omega[1] - final_omega[0]) / final_omega[0]),
            "N": int(field(hist, "final_N")),
            "volume": float(field(hist, "final_volume")),
            "raw_components_8": raw,
            "effective_components_8": eff,
            "largest_components": sizes[:5],
            "central_density": cden,
            "ccSolid": ccsolid,
        },
    }


def plot_case(case: dict) -> None:
    it = case["iterations"]
    stem = f"cc_local_{case['case']}"
    RESULTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, case["omega"][:, 0], label="omega1")
    ax.plot(it, case["omega"][:, 1], label="omega2")
    ax.axhline(456.4, color="k", ls="--", lw=1, label="paper 456.4")
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_omega12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(it, case["gap12"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("(omega2 - omega1) / omega1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_gap12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(it, case["drho_max"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("max abs Delta rho")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_drho_max.png", dpi=160)
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
    lines = []
    lines.append("# 320x40 local move-control restart assessment\n")
    lines.append("Starting point: saved long-horizon snapshot at iteration 150 (`omega1=388.281`, `omega2=460.834`).")
    lines.append("Only `move_lim` and `outer_move` differ across cases; all other settings are inherited from the 320x40 connected run.\n")

    lines.append("## Comparison\n")
    lines.append("| case | move | best iter | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | effCmp8 | maxDrho final | omega1 slope last20 | omega1 std last20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        bi = c["best_index"]
        f = c["final"]
        lines.append(
            f"| {c['case']} | {c['move']:.3f} | {bi+1} | {c['omega'][bi,0]:.3f} | "
            f"{c['omega'][bi,1]:.3f} | {c['gap12'][bi]:.4f} | {f['omega'][0]:.3f} | "
            f"{f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | {f['effective_components_8']} | "
            f"{c['drho_max'][-1]:.4e} | {c['slope20']:.4f} | {c['osc20']:.3f} |"
        )

    lines.append("\n## Per-case notes\n")
    for c in cases:
        f = c["final"]
        bi = c["best_index"]
        lines.append(f"### {c['case']}")
        lines.append(f"- Runtime: {c['runtime_s']:.1f} s; iterations: {len(c['iterations'])}")
        lines.append(f"- Best: iter {bi+1}, omega1={c['omega'][bi,0]:.3f}, omega2={c['omega'][bi,1]:.3f}, omega3={c['omega'][bi,2]:.3f}, gap12={c['gap12'][bi]:.4f}, N={int(c['N'][bi])}")
        lines.append(f"- Final: omega1={f['omega'][0]:.3f}, omega2={f['omega'][1]:.3f}, omega3={f['omega'][2]:.3f}, gap12={f['gap12']:.4f}, N={f['N']}, volume={f['volume']:.4f}")
        lines.append(f"- Connectivity: effective components={f['effective_components_8']}, raw components={f['raw_components_8']}, ccSolid={f['ccSolid']:.3f}, central density={f['central_density']:.3f}")
        lines.append(f"- Last-20 behavior: omega1 slope={c['slope20']:.4f}/iter, gap12 slope={c['gap_slope20']:.5f}/iter, omega1 std={c['osc20']:.3f}\n")

    lines.append("## Final assessment\n")
    best_case = max(cases, key=lambda c: c["omega"][c["best_index"], 0])
    smallest_gap = min(cases, key=lambda c: c["final"]["gap12"])
    all_connected = all(c["final"]["effective_components_8"] <= 1 for c in cases)
    lines.append(f"- Connected topology status: {'preserved in all cases' if all_connected else 'not preserved in all cases'}.")
    lines.append(f"- Smallest final gap12: {smallest_gap['case']} = {smallest_gap['final']['gap12']:.4f}.")
    lines.append(f"- Highest best omega1: {best_case['case']} = {best_case['omega'][best_case['best_index'],0]:.3f}.")
    if smallest_gap["final"]["N"] >= 2:
        mult = "achieved"
    elif smallest_gap["final"]["gap12"] < 0.05:
        mult = "approaching"
    else:
        mult = "not approaching"
    lines.append(f"- Multiplicity status: {mult}.")
    lines.append("- Obstacle classification: compare the reduction in last-20 omega1 oscillation and final gap across A-C. If smaller move sharply reduces oscillation and gap, obstacle A (step size) dominates; if not, the remaining obstacle is the generalized-gradient/MMA path rather than local move size alone.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = sorted(RESULTS.glob("cc_local_*.mat"))
    if not paths:
        raise SystemExit(f"No local move results found in {RESULTS}")
    cases = [load_case(p) for p in paths]
    order = {"A_baseline_move020": 0, "B_move005": 1, "C_move0025": 2}
    cases.sort(key=lambda c: order.get(c["case"], 99))
    for c in cases:
        plot_case(c)
    (RESULTS / "cc_local_move_metrics.json").write_text(json.dumps(json_safe(cases), indent=2), encoding="utf-8")
    write_report(cases)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
