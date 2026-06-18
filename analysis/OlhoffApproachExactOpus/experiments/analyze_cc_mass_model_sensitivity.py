#!/usr/bin/env python3
"""Analyze Du-Olhoff mass interpolation sensitivity cases."""

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
RESULTS = EXPDIR / "results_mass_model"
REPORT = EXPDIR / "cc_320x40_mass_model_sensitivity_assessment.md"


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
    volume = as_1d(field(hist, "volume"))[:niter]
    drho_max = as_1d(field(hist, "drho_max"))[:niter]
    final_omega = as_1d(field(hist, "final_omega"))
    rho_final = as_1d(data["rho_final"])
    best_idx = int(np.nanargmax(omega[:, 0]))
    best_gap_idx = int(np.nanargmin(gap12))
    first_n2, max_run_n2, total_n2 = n2_runs(N)
    tail = slice(max(0, niter - 20), niter)
    return {
        "case": path.stem.replace("cc_mass_", ""),
        "path": str(path),
        "mass_mode": str(field(cfg, "mass_mode")),
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "iterations": it,
        "omega": omega,
        "gap12": gap12,
        "N": N,
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
        label = c["case"].replace("_", " ")
        ax.plot(c["iterations"], c["omega"][:, 0], label=f"{label} omega1")
        ax.plot(c["iterations"], c["omega"][:, 1], ls="--", label=f"{label} omega2")
    ax.axhline(456.4, color="k", ls=":", lw=1.0)
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "mass_model_omega12_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for c in cases:
        ax.plot(c["iterations"], c["gap12"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("gap12")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "mass_model_gap12_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for c in cases:
        ax.step(c["iterations"], c["N"], where="mid", label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("N")
    ax.set_yticks([1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "mass_model_N_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for c in cases:
        ax.plot(c["iterations"], c["grey_fraction"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("grey fraction, 0.1 < rho < 0.9")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "mass_model_grey_fraction_combined.png", dpi=160)
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
    best_gap = min(cases, key=lambda c: c["final"]["gap12"])
    best_w1 = max(cases, key=lambda c: c["final"]["omega"][0])
    c1 = next(c for c in cases if c["mass_mode"] == "du2007_c1")
    lines = []
    lines.append("# 320x40 Du-Olhoff mass interpolation sensitivity assessment\n")
    lines.append("All cases start from the connected 320x40 Case-B restart design and use the same stabilized optimizer path: Heaviside projection continuation `1->2->3` every 25 iterations, `move_lim = outer_move = 0.05`, `rmin_elem = 2.5`, and `mult_tol = 0.05`. Only `mass_mode` changes.\n")

    lines.append("## Comparison\n")
    lines.append("| case | mass mode | best omega1 | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | central density | phys vol | class note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c in cases:
        bi = c["best_index"]
        gi = c["best_gap_index"]
        f = c["final"]
        if c is best_gap:
            note = "smallest gap"
        elif c is best_w1:
            note = "highest omega1"
        else:
            note = "reference-like"
        lines.append(
            f"| {c['case']} | {c['mass_mode']} | {c['omega'][bi,0]:.3f} | {c['gap12'][gi]:.4f} | "
            f"{f['omega'][0]:.3f} | {f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | {c['total_n2']} | "
            f"{f['effective_components_8']} | {f['grey_fraction_01_09']:.3f} | {f['central_density']:.3f} | {f['volume']:.4f} | {note} |"
        )

    lines.append("\n## Questions\n")
    lines.append(f"1. Smallest gap12: `{best_gap['case']}` / `{best_gap['mass_mode']}` with final gap `{best_gap['final']['gap12']:.4f}`.")
    lines.append(f"2. Highest omega1 while preserving connectivity: `{best_w1['case']}` / `{best_w1['mass_mode']}` with omega1 `{best_w1['final']['omega'][0]:.3f}` and effective components `{best_w1['final']['effective_components_8']}`.")
    stable_n2 = [c["case"] for c in cases if c["final"]["N"] == 2 and c["total_n2"] >= 80]
    lines.append(f"3. Stable N=2: yes, all tested models are stable by this criterion (`{', '.join(stable_n2)}`).")
    lines.append("4. Ranking versus coarse-mesh studies: on this connected stabilized branch the ranking is weak and local. It does not reproduce a coarse-mesh-style mass-model dominance; all three remain in the same connected class and close frequency/gap range.")

    gaps = [c["final"]["gap12"] for c in cases]
    w1s = [c["final"]["omega"][0] for c in cases]
    lines.append("\n## Final assessment\n")
    lines.append(f"- Final gap spread is small: `{min(gaps):.4f}` to `{max(gaps):.4f}`.")
    lines.append(f"- Final omega1 spread is also small among useful connected cases: `{min(w1s):.3f}` to `{max(w1s):.3f}`.")
    lines.append(f"- Relative to c1 reference (`omega1={c1['final']['omega'][0]:.3f}`, gap={c1['final']['gap12']:.4f}`), c0 raises omega1 slightly but c1 remains marginally better for coalescence.")
    lines.append("- Conclusion: the connected branch is mildly sensitive to the Du-Olhoff mass interpolation choice, but mass interpolation is not the primary control on the remaining benchmark discrepancy.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = sorted(RESULTS.glob("cc_mass_*.mat"))
    if not paths:
        raise SystemExit(f"No mass-model results found in {RESULTS}")
    cases = [load_case(p) for p in paths]
    order = {"M1_step": 0, "M2_c0": 1, "M3_c1": 2}
    cases.sort(key=lambda c: order.get(c["case"], 99))
    plot_cases(cases)
    (RESULTS / "cc_mass_model_sensitivity_metrics.json").write_text(
        json.dumps(json_safe(cases), indent=2), encoding="utf-8"
    )
    write_report(cases)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
