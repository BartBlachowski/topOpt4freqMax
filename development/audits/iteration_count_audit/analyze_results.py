from __future__ import annotations

import csv
from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as spla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.YukselApproach.Python.solver import _build_ke_ke0

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def read_vec(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name], dtype=float).reshape(-1)


def first_below(values: np.ndarray, threshold: float) -> int:
    idx = np.flatnonzero(values < threshold)
    return int(idx[0] + 1) if idx.size else -1


def first_persistent_below(values: np.ndarray, threshold: float) -> int:
    bad = np.flatnonzero(values >= threshold)
    return int(bad[-1] + 2) if bad.size and bad[-1] + 1 < values.size else (1 if not bad.size else -1)


def topology_metrics(x: np.ndarray, nely: int, nelx: int) -> dict[str, float]:
    a = x.reshape((nely, nelx), order="F")
    solid = a >= 0.5
    void = ~solid
    _, n_solid = ndimage.label(solid, structure=np.ones((3, 3), dtype=int))
    labels_v, n_void = ndimage.label(void, structure=np.ones((3, 3), dtype=int))
    boundary_labels = np.unique(np.concatenate((labels_v[0], labels_v[-1], labels_v[:, 0], labels_v[:, -1])))
    holes = sum(i not in boundary_labels for i in range(1, n_void + 1))
    checker = 0.0
    if nely > 1 and nelx > 1:
        q00, q01 = solid[:-1, :-1], solid[:-1, 1:]
        q10, q11 = solid[1:, :-1], solid[1:, 1:]
        checker = np.mean((q00 == q11) & (q01 == q10) & (q00 != q01))
    return {
        "solid_components": float(n_solid),
        "holes": float(holes),
        "checkerboard_fraction": float(checker),
        "symmetry_l1": float(np.mean(np.abs(a - np.flipud(a)))),
        "gray_fraction_01_09": float(np.mean((a > 0.1) & (a < 0.9))),
    }


def stiffness_condition_proxy(x: np.ndarray, nely: int, nelx: int) -> dict[str, float]:
    """Reassemble Yuksel K and estimate its free-DOF spectral condition number."""
    n_dof = 2 * (nely + 1) * (nelx + 1)
    nodes = np.arange(1, (nely + 1) * (nelx + 1) + 1, dtype=np.int64).reshape(
        (nely + 1, nelx + 1), order="F"
    )
    c_vec = (2 * nodes[:-1, :-1] + 1).reshape(-1, order="F")
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    c_mat = c_vec[:, None] + offsets[None, :] - 1
    si, sj = [], []
    for j in range(8):
        si.extend(range(j, 8))
        sj.extend([j] * (8 - j))
    ik = c_mat[:, np.asarray(si)].T.reshape(-1, order="F")
    jk = c_mat[:, np.asarray(sj)].T.reshape(-1, order="F")
    rows, cols = np.maximum(ik, jk), np.minimum(ik, jk)
    ke_lower, _ = _build_ke_ke0(0.3)
    modulus = 10.0 + x**3 * (1.0e7 - 10.0)
    k_mat = sparse.coo_matrix(
        (np.kron(modulus, ke_lower), (rows, cols)), shape=(n_dof, n_dof)
    ).tocsr()
    k_mat = k_mat + k_mat.T - sparse.diags(k_mat.diagonal())
    mid_row = int(np.floor(nely / 2 + 0.5)) + 1
    left, right = int(nodes[mid_row - 1, 0]), int(nodes[mid_row - 1, -1])
    fixed = np.array([2 * left - 2, 2 * left - 1, 2 * right - 2, 2 * right - 1])
    free = np.setdiff1d(np.arange(n_dof), fixed)
    k_free = k_mat[free][:, free]
    largest = float(spla.eigsh(k_free, k=1, which="LA", return_eigenvectors=False, tol=1e-4, maxiter=2000)[0])
    smallest = float(spla.eigsh(k_free, k=1, sigma=0, which="LM", return_eigenvectors=False, tol=1e-4, maxiter=2000)[0])
    diag = k_free.diagonal()
    return {
        "smallest_stiffness_eigenvalue": smallest,
        "largest_stiffness_eigenvalue": largest,
        "spectral_condition_proxy": largest / smallest,
        "diagonal_ratio": float(diag.max() / diag.min()),
    }


def load_case(path: Path) -> dict:
    with h5py.File(path) as f:
        result = f["result"]
        case = result["case"]
        stage2 = result["info"]["stage2"]
        audit = stage2["audit"]
        nelx = int(np.asarray(case["nelx"]).reshape(-1)[0])
        nely = int(np.asarray(case["nely"]).reshape(-1)[0])
        tag = "".join(chr(int(v)) for v in np.asarray(case["tag"]).reshape(-1))
        data = {
            "tag": tag,
            "nelx": nelx,
            "nely": nely,
            "rmin": float(np.asarray(case["rmin"]).reshape(-1)[0]),
            "freeze_mode": bool(np.asarray(case["freezeMode"]).reshape(-1)[0]),
            "freeze_load": bool(np.asarray(case["freezeLoad"]).reshape(-1)[0]),
            "c": read_vec(stage2, "c"),
            "ch": read_vec(stage2, "ch"),
            "v": read_vec(stage2, "v"),
            "x_final": read_vec(stage2, "xFinal"),
            "omega1": float(np.asarray(stage2["omega1"]).reshape(-1)[0]),
        }
        for field in audit:
            arr = np.asarray(audit[field], dtype=float)
            if field == "xPhysSnapshots":
                data[field] = arr
            else:
                data[field] = arr.reshape(-1)
        # MATLAB column-major array is exposed transposed by HDF5.
        snaps = data["xPhysSnapshots"]
        if snaps.shape[1] != nelx * nely:
            snaps = snaps.T
        data["xPhysSnapshots"] = snaps
        return data


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cases = {p.stem: load_case(p) for p in sorted(RESULTS.glob("*.mat"))}
    baselines = [cases[f"baseline_{nx}x{ny}"] for nx, ny in ((160, 20), (240, 30), (320, 40), (400, 50))]
    tol = 3e-3

    diagnostic_rows = []
    topology_rows = []
    sensitivity_rows = []
    for d in cases.values():
        n = len(d["ch"])
        top_seq = [topology_metrics(x, d["nely"], d["nelx"]) for x in d["xPhysSnapshots"]]
        snap_iters = d["snapshotIter"].astype(int)
        binary_changes = []
        for left, right in zip(d["xPhysSnapshots"][:-1], d["xPhysSnapshots"][1:]):
            binary_changes.append(float(np.mean((left >= 0.5) != (right >= 0.5))))
        binary_changes = np.asarray(binary_changes)
        significant = np.flatnonzero(binary_changes > 1e-3)
        last_transition = int(snap_iters[significant[-1] + 1]) if significant.size else int(snap_iters[0])
        hole_counts = np.asarray([m["holes"] for m in top_seq])
        hole_change = np.flatnonzero(np.diff(hole_counts) != 0)
        last_hole_change = int(snap_iters[hole_change[-1] + 1]) if hole_change.size else int(snap_iters[0])
        final_top = topology_metrics(d["x_final"], d["nely"], d["nelx"])

        diagnostic_rows.append(
            {
                "tag": d["tag"],
                "stage2_iterations": n,
                "omega1": d["omega1"],
                "first_max_below_tol": first_below(d["ch"], tol),
                "first_mean_below_tol": first_below(d["stepMean"], tol),
                "first_rms_below_tol": first_below(d["stepRms"], tol),
                "first_p95_below_tol": first_below(d["stepP95"], tol),
                "persistent_mean_below_tol": first_persistent_below(d["stepMean"], tol),
                "persistent_rms_below_tol": first_persistent_below(d["stepRms"], tol),
                "persistent_p95_below_tol": first_persistent_below(d["stepP95"], tol),
                "persistent_mode_angle_below_0p1deg": first_persistent_below(d["modeAngleDeg"], 0.1),
                "persistent_mac_above_0p999999": first_persistent_below(1 - d["modeCos"] ** 2, 1e-6),
                "persistent_du_below_tol": first_persistent_below(d["du"], tol),
                "last_binary_transition_gt_0p1pct": last_transition,
                "last_hole_count_change": last_hole_change,
                "snapshots_with_binary_change_gt_0p1pct": int(significant.size),
                "cumulative_binary_change": float(np.sum(binary_changes)),
                "max_binary_change_per_10_iters": float(np.max(binary_changes)) if binary_changes.size else 0,
                "final_step_mean": d["stepMean"][-1],
                "final_step_rms": d["stepRms"][-1],
                "final_step_p95": d["stepP95"][-1],
                "final_step_max": d["ch"][-1],
                "final_max_to_p95_ratio": d["ch"][-1] / max(d["stepP95"][-1], 1e-30),
                "final_fraction_active": d["stepActiveFrac"][-1],
                "max_mode_angle_deg": np.nanmax(d["modeAngleDeg"]),
                "tail_mode_angle_deg_median": np.nanmedian(d["modeAngleDeg"][n // 2 :]),
                "tail_objective_rel_change_median": np.nanmedian(
                    np.abs(np.diff(d["c"][n // 2 :])) / np.maximum(np.abs(d["c"][n // 2 : -1]), 1e-30)
                ),
                "final_solid_components": int(final_top["solid_components"]),
                "final_holes": int(final_top["holes"]),
                "final_checkerboard_fraction": final_top["checkerboard_fraction"],
                "final_symmetry_l1": final_top["symmetry_l1"],
                "final_gray_fraction": final_top["gray_fraction_01_09"],
            }
        )

        for it, tm, prev in zip(snap_iters, top_seq, [np.nan] + binary_changes.tolist()):
            topology_rows.append(
                {
                    "tag": d["tag"],
                    "iteration": int(it),
                    "binary_change_from_prior_snapshot": prev,
                    **tm,
                }
            )

        for label, idx in (("start", 0), ("quarter", n // 4), ("half", n // 2), ("three_quarter", 3 * n // 4), ("final", n - 1)):
            sensitivity_rows.append(
                {
                    "tag": d["tag"],
                    "point": label,
                    "iteration": idx + 1,
                    "dc_mean": d["dcMean"][idx],
                    "dc_std": d["dcStd"][idx],
                    "dc_max_abs": d["dcMaxAbs"][idx],
                    "dc_p95_abs": d["dcP95Abs"][idx],
                    "oc_arg_mean": d["ocArgMean"][idx],
                    "oc_arg_cv": d["ocArgCV"][idx],
                    "oc_lambda": d["lambdaOC"][idx],
                    "normalized_oc_driver_mean": d["ocArgMean"][idx] / max(d["lambdaOC"][idx], 1e-30),
                }
            )

    write_csv(RESULTS / "diagnostic_summary.csv", diagnostic_rows)
    write_csv(RESULTS / "topology_metrics.csv", topology_rows)
    write_csv(RESULTS / "sensitivity_scaling.csv", sensitivity_rows)

    condition_rows = []
    for d in baselines:
        condition_rows.append(
            {
                "tag": d["tag"],
                "stage2_iterations": len(d["ch"]),
                **stiffness_condition_proxy(d["x_final"], d["nely"], d["nelx"]),
            }
        )
    write_csv(RESULTS / "condition_proxies.csv", condition_rows)

    perf = {}
    with (ROOT / "examples" / "Performance" / "table1_performance.csv").open() as f:
        for row in csv.DictReader(f):
            perf[(row["Method"], row["Mesh"])] = row
    decomposition = []
    for nx, ny in ((160, 20), (240, 30), (320, 40), (400, 50), (480, 60), (560, 70), (640, 80), (720, 90), (800, 100)):
        row = perf[("Yuksel", f"{nx}x{ny}")]
        s1, s2, total = int(row["IterStage1"]), int(row["IterStage2"]), int(row["Iterations"])
        decomposition.append(
            {
                "mesh": f"{nx}x{ny}",
                "total": total,
                "stage1": s1,
                "stage2": s2,
                "stage2_fraction_percent": 100 * s2 / total,
            }
        )
    write_csv(RESULTS / "iteration_decomposition.csv", decomposition)

    colors = ["#4477AA", "#228833", "#CC6677", "#AA3377"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for color, d in zip(colors, baselines):
        label = f"{d['nelx']}×{d['nely']} ({len(d['ch'])} it.)"
        it = np.arange(1, len(d["ch"]) + 1)
        axes[0, 0].semilogy(it, d["ch"], color=color, label=label)
        axes[0, 1].semilogy(it, d["stepMean"], color=color, label=label)
        axes[0, 1].semilogy(it, d["stepP95"], color=color, linestyle="--")
        axes[1, 0].semilogy(it, np.maximum(d["modeAngleDeg"], 1e-12), color=color, label=label)
        axes[1, 1].plot(it, d["lambdaOC"] / d["lambdaOC"][0], color=color, label=label)
        axes[2, 0].semilogy(it, d["dcMaxAbs"] / np.maximum(d["dcP95Abs"], 1e-30), color=color, label=label)
        rel_obj = np.r_[np.nan, np.abs(np.diff(d["c"])) / np.maximum(np.abs(d["c"][:-1]), 1e-30)]
        axes[2, 1].semilogy(it, np.maximum(rel_obj, 1e-14), color=color, label=label)
    axes[0, 0].axhline(tol, color="black", linestyle=":", label="tol=0.003")
    axes[0, 1].axhline(tol, color="black", linestyle=":")
    axes[0, 0].set_title("Maximum accepted density change (actual stop)")
    axes[0, 1].set_title("Mean (solid) and 95th-percentile (dashed) step")
    axes[1, 0].set_title("Angle between consecutive mode estimates")
    axes[1, 1].set_title("OC multiplier / initial multiplier")
    axes[2, 0].set_title("Sensitivity max / 95th percentile")
    axes[2, 1].set_title("Relative change of moving compliance value")
    for ax in axes.ravel():
        ax.set_xlabel("Stage-2 iteration")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(RESULTS / "diagnostic_histories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    order = [
        "baseline_320x40",
        "fixed_physical_radius_320x40",
        "frozen_mode_320x40",
        "frozen_load_320x40",
    ]
    vals = [len(cases[k]["ch"]) for k in order]
    labels = ["Baseline\nr=2 elem", "Fixed physical\nr=4 elem", "Frozen mode\nestimate", "Frozen inertial\nload"]
    bars = ax.bar(labels, vals, color=["#CC6677", "#4477AA", "#228833", "#AA3377"])
    ax.bar_label(bars)
    ax.set_ylabel("Stage-2 iterations")
    ax.set_title("320×40 causal controls (single-factor effects; not additive)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(RESULTS / "ablation_iteration_counts.png", dpi=180)
    plt.close(fig)

    selected = [cases["baseline_160x20"], cases["baseline_320x40"], cases["baseline_400x50"]]
    fig, axes = plt.subplots(len(selected), 5, figsize=(15, 5.5), constrained_layout=True)
    for row, d in enumerate(selected):
        snap_iters = d["snapshotIter"].astype(int)
        targets = np.linspace(1, len(d["ch"]), 5)
        indices = [int(np.argmin(np.abs(snap_iters - t))) for t in targets]
        for col, idx in enumerate(indices):
            a = d["xPhysSnapshots"][idx].reshape((d["nely"], d["nelx"]), order="F")
            axes[row, col].imshow(1 - a, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
            axes[row, col].set_title(f"{d['nelx']}×{d['nely']}, it {snap_iters[idx]}", fontsize=9)
            axes[row, col].axis("off")
    fig.savefig(RESULTS / "topology_evolution.png", dpi=180)
    plt.close(fig)

    print(f"Wrote diagnostics to {RESULTS}")


if __name__ == "__main__":
    main()
