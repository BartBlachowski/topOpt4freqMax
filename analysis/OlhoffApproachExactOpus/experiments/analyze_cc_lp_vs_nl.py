#!/usr/bin/env python3
"""Analyze nonlinear vs LP-reduced multiple-eigenvalue restart cases."""

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
RESULTS = EXPDIR / "results_lp_vs_nl"
REPORT = EXPDIR / "cc_320x40_lp_vs_nl_assessment.md"


def field(obj, name):
    return getattr(obj, name)


def text_field(obj, name) -> str:
    value = field(obj, name)
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.reshape(-1).astype(str)).strip()
    return str(value)


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
    lab, n_raw = label(solid, structure=np.ones((3, 3), dtype=int))
    sizes = sorted((int((lab == i).sum()) for i in range(1, n_raw + 1)), reverse=True)
    threshold = max(10, int(round(0.005 * nelx * nely)))
    n_eff = len([s for s in sizes if s >= threshold])
    return int(n_raw), int(n_eff), sizes


def central_metrics(rho: np.ndarray, nelx: int, nely: int) -> tuple[float, float]:
    grid = np.reshape(as_1d(rho), (nely, nelx), order="F")
    central = grid[:, nelx // 3 : 2 * nelx // 3]
    return float(np.mean(central)), float(np.mean(central > 0.5))


def activation_runs(N: np.ndarray, target: int) -> tuple[int | None, int, int]:
    idx = np.where(N == target)[0]
    if idx.size == 0:
        return None, 0, 0
    runs = []
    start = idx[0]
    prev = idx[0]
    for cur in idx[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            runs.append((start, prev))
            start = prev = cur
    runs.append((start, prev))
    return int(idx[0] + 1), int(max(end - start + 1 for start, end in runs)), int(idx.size)


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
    gap23 = (omega[:, 2] - omega[:, 1]) / omega[:, 1]
    N = as_1d(field(hist, "N_trial"))[:niter]
    volume = as_1d(field(hist, "volume"))[:niter]
    drho_max = as_1d(field(hist, "drho_max"))[:niter]
    if hasattr(hist, "inner_offdiag_max"):
        offdiag = as_1d(field(hist, "inner_offdiag_max"))[:niter]
    else:
        offdiag = np.full(niter, np.nan)

    final_omega = as_1d(field(hist, "final_omega"))
    final_gap12 = float((final_omega[1] - final_omega[0]) / final_omega[0])
    rho_final = as_1d(data["rho_final"])
    raw_comp, eff_comp, sizes = components_8(rho_final, nelx, nely)
    central_density, cc_solid = central_metrics(rho_final, nelx, nely)

    best_idx = int(np.nanargmax(omega[:, 0]))
    first_n2, max_run_n2, total_n2 = activation_runs(N, 2)
    tail = slice(max(0, niter - 20), niter)
    slope20 = float(np.polyfit(it[tail].astype(float), omega[tail, 0], 1)[0])
    gap_slope20 = float(np.polyfit(it[tail].astype(float), gap12[tail], 1)[0])
    oscillation20 = float(np.std(omega[tail, 0]))

    return {
        "path": str(path),
        "case": path.stem.replace("cc_embed_", ""),
        "formulation": text_field(cfg, "subproblem_formulation"),
        "mult_tol": float(field(cfg, "mult_tol")),
        "runtime_s": float(np.asarray(data["el"]).reshape(())),
        "iterations": it,
        "omega": omega,
        "gap12": gap12,
        "gap23": gap23,
        "N": N,
        "volume": volume,
        "drho_max": drho_max,
        "offdiag_max": offdiag,
        "best_index": best_idx,
        "first_n2": first_n2,
        "max_run_n2": max_run_n2,
        "total_n2": total_n2,
        "slope20": slope20,
        "gap_slope20": gap_slope20,
        "oscillation20": oscillation20,
        "final": {
            "omega": final_omega,
            "gap12": final_gap12,
            "N": int(field(hist, "final_N")),
            "volume": float(field(hist, "final_volume")),
            "raw_components_8": raw_comp,
            "effective_components_8": eff_comp,
            "largest_components": sizes[:5],
            "central_density": central_density,
            "ccSolid": cc_solid,
        },
    }


def plot_combined(cases: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    styles = {
        "NL_nonlinear": {"c1": "tab:blue", "c2": "tab:cyan"},
        "LP_reduced": {"c1": "tab:orange", "c2": "tab:red"},
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for c in cases:
        st = styles.get(c["case"], {"c1": None, "c2": None})
        ax.plot(c["iterations"], c["omega"][:, 0], color=st["c1"], label=f"{c['case']} omega1")
        ax.plot(c["iterations"], c["omega"][:, 1], color=st["c2"], ls="--", label=f"{c['case']} omega2")
    ax.axhline(456.4, color="k", ls=":", lw=1.1, label="paper 456.4")
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "lp_vs_nl_omega12.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for c in cases:
        ax.plot(c["iterations"], c["gap12"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("(omega2 - omega1) / omega1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "lp_vs_nl_gap12.png", dpi=160)
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
    fig.savefig(RESULTS / "lp_vs_nl_N.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for c in cases:
        ax.plot(c["iterations"], c["drho_max"], label=c["case"])
    ax.set_xlabel("restart iteration")
    ax.set_ylabel("max abs Delta rho")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "lp_vs_nl_drho_max.png", dpi=160)
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


def report_lines(cases: list[dict]) -> list[str]:
    nl = next(c for c in cases if c["case"].startswith("NL"))
    lp = next(c for c in cases if c["case"].startswith("LP"))
    best_w1 = max(cases, key=lambda c: c["omega"][c["best_index"], 0])
    best_gap = min(cases, key=lambda c: c["final"]["gap12"])
    all_connected = all(c["final"]["effective_components_8"] <= 1 for c in cases)

    lp_offdiag = lp["offdiag_max"]
    finite_lp_offdiag = lp_offdiag[np.isfinite(lp_offdiag)]
    max_lp_offdiag = float(np.max(finite_lp_offdiag)) if finite_lp_offdiag.size else float("nan")
    final_lp_offdiag = float(finite_lp_offdiag[-1]) if finite_lp_offdiag.size else float("nan")

    lines = []
    lines.append("# 320x40 LP reduction vs nonlinear subeigenvalue assessment\n")
    lines.append("Starting point: saved connected 320x40 Case-B local move-control final design.")
    lines.append("Both cases use `move_lim = outer_move = 0.05`, 100 restart iterations, and `mult_tol = 0.05` so the N>1 subproblem is exercised. Only `subproblem_formulation` differs.\n")

    lines.append("## Comparison\n")
    lines.append("| case | formulation | first N=2 | max consecutive N=2 | total N=2 iters | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | effCmp8 | final volume | maxDrho final | omega1 slope last20 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        bi = c["best_index"]
        f = c["final"]
        first = c["first_n2"] if c["first_n2"] is not None else "-"
        lines.append(
            f"| {c['case']} | {c['formulation']} | {first} | {c['max_run_n2']} | {c['total_n2']} | "
            f"{c['omega'][bi,0]:.3f} | {c['omega'][bi,1]:.3f} | {c['gap12'][bi]:.4f} | "
            f"{f['omega'][0]:.3f} | {f['omega'][1]:.3f} | {f['gap12']:.4f} | {f['N']} | "
            f"{f['effective_components_8']} | {f['volume']:.4f} | {c['drho_max'][-1]:.4e} | {c['slope20']:.4f} |"
        )

    lines.append("\n## LP off-diagonal diagnostic\n")
    lines.append("For the LP-reduced case, the paired off-diagonal constraints target `f_sk^T Delta rho = 0` for `s != k` inside the inner MMA step.")
    lines.append(f"- Maximum recorded absolute residual: `{max_lp_offdiag:.6e}`.")
    lines.append(f"- Final recorded absolute residual: `{final_lp_offdiag:.6e}`.")
    if finite_lp_offdiag.size and max_lp_offdiag > 1e-6:
        lines.append("- The residual is not numerically zero, so the MMA embedding enforces the LP equalities approximately rather than as an exact equality-constrained LP solve.")
    else:
        lines.append("- The recorded residuals are near numerical zero at this scale.")

    lines.append("\n## Questions\n")
    lines.append(f"1. LP movement of omega1 toward omega2: {'yes' if lp['final']['gap12'] < nl['final']['gap12'] else 'no'}; final gaps are LP `{lp['final']['gap12']:.4f}` and NL `{nl['final']['gap12']:.4f}`.")
    lines.append(f"2. Faster gap decrease: {'LP' if lp['gap_slope20'] < nl['gap_slope20'] else 'NL'} has the more negative last-20 gap slope (`{lp['gap_slope20']:.5f}` vs `{nl['gap_slope20']:.5f}`).")
    lines.append(f"3. N=2 persistence: LP has `{lp['total_n2']}` N=2 iterations, NL has `{nl['total_n2']}`.")
    lines.append(f"4. Connected topology: {'preserved in both cases' if all_connected else 'not preserved in both cases'}.")
    lines.append(f"5. Optimization path alteration: best omega1 comes from `{best_w1['case']}` at `{best_w1['omega'][best_w1['best_index'],0]:.3f}`.")
    lines.append(f"6. Closest to 456.4: final omega1 is LP `{lp['final']['omega'][0]:.3f}` and NL `{nl['final']['omega'][0]:.3f}`; best omega1 is LP `{lp['omega'][lp['best_index'],0]:.3f}` and NL `{nl['omega'][nl['best_index'],0]:.3f}`.")
    lines.append(f"7. Benchmark-gap explanation: the best gap/coalescence case is `{best_gap['case']}` with final gap `{best_gap['final']['gap12']:.4f}`.")

    lines.append("\n## Per-case notes\n")
    for c in cases:
        bi = c["best_index"]
        f = c["final"]
        lines.append(f"### {c['case']}")
        lines.append(f"- Runtime: {c['runtime_s']:.1f} s; iterations: {len(c['iterations'])}.")
        lines.append(f"- Best: iter {bi+1}, omega1={c['omega'][bi,0]:.3f}, omega2={c['omega'][bi,1]:.3f}, omega3={c['omega'][bi,2]:.3f}, gap12={c['gap12'][bi]:.4f}, N={int(c['N'][bi])}.")
        lines.append(f"- Final: omega1={f['omega'][0]:.3f}, omega2={f['omega'][1]:.3f}, omega3={f['omega'][2]:.3f}, gap12={f['gap12']:.4f}, N={f['N']}, volume={f['volume']:.4f}.")
        lines.append(f"- Connectivity: effective components={f['effective_components_8']}, raw components={f['raw_components_8']}, ccSolid={f['ccSolid']:.3f}, central density={f['central_density']:.3f}.")
        lines.append(f"- Last-20 behavior: omega1 slope={c['slope20']:.4f}/iter, gap12 slope={c['gap_slope20']:.5f}/iter, omega1 std={c['oscillation20']:.3f}.\n")

    lines.append("## Final assessment\n")
    if lp["final"]["gap12"] + 1e-3 < nl["final"]["gap12"] and lp["omega"][lp["best_index"], 0] > nl["omega"][nl["best_index"], 0] + 1.0:
        classification = "A. nonlinear subeigenvalue embedding is a material contributor."
    elif abs(lp["final"]["gap12"] - nl["final"]["gap12"]) < 0.01 and abs(lp["omega"][lp["best_index"], 0] - nl["omega"][nl["best_index"], 0]) < 5.0:
        classification = "B. optimizer path independent of embedding is the primary explanation."
    else:
        classification = "C. another undocumented numerical choice or optimizer mechanism remains likely."
    lines.append(f"- Remaining discrepancy classification: {classification}")
    lines.append(f"- Connected topology status: {'preserved' if all_connected else 'not consistently preserved'}.")
    lines.append(f"- Frequency/coalescence status: best final gap is `{best_gap['final']['gap12']:.4f}` from `{best_gap['case']}`; neither case is judged coalesced unless this is near zero.")
    lines.append("- The LP-reduced experiment should be interpreted as diagnostic only: it keeps MMA and the rest of the update mechanism unchanged, so it tests the embedding effect without replacing the optimizer by a true standalone LP solver.")
    return lines


def main() -> None:
    paths = sorted(RESULTS.glob("cc_embed_*.mat"))
    if not paths:
        raise SystemExit(f"No LP-vs-NL results found in {RESULTS}")
    cases = [load_case(p) for p in paths]
    order = {"NL_nonlinear": 0, "LP_reduced": 1}
    cases.sort(key=lambda c: order.get(c["case"], 99))
    plot_combined(cases)
    (RESULTS / "cc_lp_vs_nl_metrics.json").write_text(json.dumps(json_safe(cases), indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(report_lines(cases)), encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote plots and metrics in {RESULTS}")


if __name__ == "__main__":
    main()
