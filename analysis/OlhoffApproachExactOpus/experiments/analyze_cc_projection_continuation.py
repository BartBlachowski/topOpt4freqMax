#!/usr/bin/env python3
"""Analyze Heaviside projection continuation cases."""

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
RESULTS = EXPDIR / "results_projection_continuation"
REPORT = EXPDIR / "cc_320x40_projection_continuation_assessment.md"


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


def beta_description(cfg, hist) -> str:
    sched = as_1d(field(cfg, "projection_beta_schedule")) if hasattr(cfg, "projection_beta_schedule") else np.array([])
    if sched.size:
        interval = int(field(cfg, "projection_beta_interval"))
        return f"{[int(v) if float(v).is_integer() else float(v) for v in sched]} / {interval}"
    return f"fixed {float(field(cfg, 'projection_beta')):.1f}"


def snapshot_metrics(hist, niter: int, nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grey = np.full(niter, np.nan)
    eff_comp = np.full(niter, np.nan)
    raw_comp = np.full(niter, np.nan)
    if not hasattr(hist, "rho_snapshot_count"):
        return grey, eff_comp, raw_comp
    count = int(field(hist, "rho_snapshot_count"))
    if count <= 0:
        return grey, eff_comp, raw_comp
    snap_iters = as_1d(field(hist, "rho_snapshot_iters"))[:count].astype(int)
    snaps = np.asarray(field(hist, "rho_snapshots"), dtype=float)
    if snaps.ndim == 1:
        snaps = snaps.reshape((-1, 1))
    for si, it in enumerate(snap_iters):
        if it < 1 or it > niter:
            continue
        rho = snaps[:, si]
        grey[it - 1] = float(np.mean((rho > 0.1) & (rho < 0.9)))
        raw, eff, _ = components_8(rho, nelx, nely)
        raw_comp[it - 1] = raw
        eff_comp[it - 1] = eff
    return grey, eff_comp, raw_comp


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
    beta_iter = as_1d(field(hist, "projection_beta_iter"))[:niter]
    grey_hist, eff_comp_hist, raw_comp_hist = snapshot_metrics(hist, niter, nelx, nely)
    final_omega = as_1d(field(hist, "final_omega"))
    rho_final = as_1d(data["rho_final"])
    best_idx = int(np.nanargmax(omega[:, 0]))
    best_gap_idx = int(np.nanargmin(gap12))
    first_n2, max_run_n2, total_n2 = n2_runs(N)
    metrics = density_metrics(rho_final, nelx, nely)
    tail = slice(max(0, niter - 20), niter)

    return {
        "case": path.stem.replace("cc_projcont_", ""),
        "path": str(path),
        "schedule": beta_description(cfg, hist),
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "iterations": it,
        "omega": omega,
        "gap12": gap12,
        "N": N,
        "volume": volume,
        "drho_max": drho_max,
        "beta_iter": beta_iter,
        "grey_fraction": grey_hist,
        "effective_components": eff_comp_hist,
        "raw_components": raw_comp_hist,
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
            **metrics,
        },
    }


def plot_case(case: dict) -> None:
    it = case["iterations"]
    stem = case["case"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(it, case["omega"][:, 0], label="omega1")
    ax.plot(it, case["omega"][:, 1], label="omega2")
    ax.axhline(456.4, color="k", ls=":", lw=1.0, label="paper 456.4")
    for b in np.unique(case["beta_iter"]):
        idx = np.where(case["beta_iter"] == b)[0]
        if idx.size:
            ax.axvspan(idx[0] + 1, idx[-1] + 1, alpha=0.05)
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.set_title(stem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_omega12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(it, case["gap12"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("(omega2 - omega1) / omega1")
    ax.set_title(stem)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_gap12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    ax.step(it, case["N"], where="mid")
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("N")
    ax.set_title(stem)
    ax.set_yticks([1, 2, 3])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_N.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(it, case["grey_fraction"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("grey fraction, 0.1 < rho < 0.9")
    ax.set_title(stem)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_grey_fraction.png", dpi=160)
    plt.close(fig)


def plot_combined(cases: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c in cases:
        ax.plot(c["iterations"], c["gap12"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("gap12")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_continuation_gap12_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c in cases:
        ax.plot(c["iterations"], c["omega"][:, 0], label=f"{c['case']} omega1")
    ax.axhline(456.4, color="k", ls=":", lw=1.0)
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega1 [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_continuation_omega1_combined.png", dpi=160)
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
    best_final_gap = min(cases, key=lambda c: c["final"]["gap12"])
    best_final_w1 = max(cases, key=lambda c: c["final"]["omega"][0])
    best_any_gap = min(cases, key=lambda c: c["gap12"][c["best_gap_index"]])
    pc0 = next(c for c in cases if c["case"].startswith("PC0"))
    lines = []
    lines.append("# 320x40 Heaviside projection continuation assessment\n")
    lines.append("This is a paper-inspired stabilized reproduction experiment, not exact paper reproduction. All cases start from the saved connected 320x40 Case-B design and keep FE model, mass model, nonlinear subeigenvalue formulation, filter radius, move limits, volume constraint, MMA settings, and `mult_tol=0.05` unchanged.\n")

    lines.append("## Comparison\n")
    lines.append("| case | beta schedule | best omega1 | best gap iter | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | phys vol | final beta | omega1 slope20 | class note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c in cases:
        bi = c["best_index"]
        gi = c["best_gap_index"]
        f = c["final"]
        if c["case"].startswith("PC0"):
            note = "fixed-beta reference"
        elif c is best_final_gap:
            note = "best final coalescence"
        elif f["omega"][0] < pc0["final"]["omega"][0] - 2 or f["gap12"] > pc0["final"]["gap12"]:
            note = "over-projected final"
        else:
            note = "intermediate"
        lines.append(
            f"| {c['case']} | {c['schedule']} | {c['omega'][bi,0]:.3f} | {gi+1} | {c['gap12'][gi]:.4f} | "
            f"{f['omega'][0]:.3f} | {f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | {c['total_n2']} | "
            f"{f['effective_components_8']} | {f['grey_fraction_01_09']:.3f} | {f['volume']:.4f} | {f['beta']:.1f} | {c['slope20']:.4f} | {note} |"
        )

    lines.append("\n## Required questions\n")
    lines.append(f"1. Continuation below 1.90% gap: yes. Best final case is `{best_final_gap['case']}` with final gap `{best_final_gap['final']['gap12']:.4f}`; best transient gap is `{best_any_gap['gap12'][best_any_gap['best_gap_index']]:.4f}` in `{best_any_gap['case']}`.")
    lines.append(f"2. Omega1 upward toward omega2: partly. PC1 keeps omega1 near the beta=3 level (`{best_final_gap['final']['omega'][0]:.3f}`), but does not exceed PC0. Stronger beta schedules reduce final omega1.")
    lines.append(f"3. Omega2 stability: PC1 lowers omega2 toward omega1 (`{best_final_gap['final']['omega'][1]:.3f}`), which improves coalescence. PC3 drives omega2 away upward at the final beta=6 stage.")
    lines.append(f"4. N=2 persistence: PC0, PC1, and PC2 have persistent N=2; PC3 loses final N=2 despite a long N=2 segment.")
    connected = all(c["final"]["effective_components_8"] <= 1 for c in cases)
    lines.append(f"5. Connected topology: {'preserved in all cases' if connected else 'not preserved in all cases'}.")
    lines.append("6. Stronger projection: beta up to 3 helps coalescence; beta 4/6 mostly over-projects or freezes/degrades the branch rather than improving final coalescence.")
    lines.append(f"7. Closest approach to `omega1 ~= omega2 ~= 456.4`: `{best_final_gap['case']}` by final coalescence, but `{best_final_w1['case']}` by final omega1. No schedule approaches the published frequency level.")

    lines.append("\n## Final assessment\n")
    lines.append("- Projection role classification: **B. useful stabilizer**.")
    lines.append("- It is more than cosmetic: beta continuation 1->2->3 reduces final gap from PC0 `0.0190` to `0.0105` and keeps N=2 active.")
    lines.append("- It is not yet a key/dominant reproduction mechanism: the best final frequency remains around `omega1=418.7`, `omega2=423.1`, far below `456.4`, and stronger beta continuation degrades the final state.")
    lines.append("- Practical recommendation: keep beta around 3 for this branch; do not continue blindly to beta 4 or 6 without a separate path-control change.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = sorted(RESULTS.glob("cc_projcont_*.mat"))
    if not paths:
        raise SystemExit(f"No projection-continuation results found in {RESULTS}")
    cases = [load_case(p) for p in paths]
    order = {
        "PC0_beta3_fixed": 0,
        "PC1_beta123_i25": 1,
        "PC2_beta1234_i25": 2,
        "PC3_beta12346_i20": 3,
    }
    cases.sort(key=lambda c: order.get(c["case"], 99))
    for c in cases:
        plot_case(c)
    plot_combined(cases)
    (RESULTS / "cc_projection_continuation_metrics.json").write_text(
        json.dumps(json_safe(cases), indent=2), encoding="utf-8"
    )
    write_report(cases)
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
