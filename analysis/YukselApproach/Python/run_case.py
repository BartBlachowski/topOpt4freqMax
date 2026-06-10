#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

try:
    from .cases import CASES, get_case
    from .solver import top99neo_dynamic_freq, top99neo_inertial_freq
except ImportError:  # direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from examples.fromPaper.Yuksel.cases import CASES, get_case
    from examples.fromPaper.Yuksel.solver import top99neo_dynamic_freq, top99neo_inertial_freq

matplotlib.use("Agg")


def _plot_stage_topologies(stage1: np.ndarray, stage2: np.ndarray, nely: int, nelx: int, omega1_s1: float, omega1_s2: float, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)
    axes[0].imshow(1 - stage1.reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[0].set_axis_off()
    axes[0].set_title(f"Stage 1 | w1={omega1_s1:.2f}")

    axes[1].imshow(1 - stage2.reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[1].set_axis_off()
    axes[1].set_title(f"Stage 2 | w1={omega1_s2:.2f}")

    fig.suptitle(title)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _plot_dynamic_history(omega_hist: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    it = np.arange(1, omega_hist.shape[0] + 1)
    for j in range(min(3, omega_hist.shape[1])):
        ax.plot(it, omega_hist[:, j], label=f"mode {j+1}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("omega [rad/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_case(
    case: str = "fig4",
    quick: bool = False,
    out: Path | None = None,
    with_dynamic: bool = False,
    snapshot_every: int = 0,
) -> Path:
    params = get_case(case, quick=quick)

    base_out = out if out is not None else Path(__file__).resolve().parent / "output"
    case_out = base_out / case
    case_out.mkdir(parents=True, exist_ok=True)

    res = top99neo_inertial_freq(
        nelx=params["nelx"],
        nely=params["nely"],
        volfrac=params["volfrac"],
        penal=params["penal"],
        rmin=params["rmin"],
        ft=params["ft"],
        ftBC=params["ftBC"],
        eta=params["eta"],
        beta=params["beta"],
        move=params["move"],
        maxit=params["maxit"],
        stage1_maxit=params["stage1_maxit"],
        bcType=params["bcType"],
        nHistModes=3,
        output_dir=case_out,
        snapshot_every=max(0, snapshot_every),
        verbose=True,
    )

    _plot_stage_topologies(
        stage1=np.asarray(res.info["stage1"]["xFinal"], dtype=np.float64),
        stage2=np.asarray(res.xPhys_stage2, dtype=np.float64),
        nely=params["nely"],
        nelx=params["nelx"],
        omega1_s1=float(res.info["stage1"]["omega1"]),
        omega1_s2=float(res.info["stage2"]["omega1"]),
        out_png=case_out / "final_plot.png",
        title=f"Yuksel {case} ({params['bcType']})",
    )

    np.savez(
        case_out / "final_design_field.npz",
        x_stage1=np.asarray(res.info["stage1"]["xFinal"], dtype=np.float64),
        x_stage2=np.asarray(res.xPhys_stage2, dtype=np.float64),
        omega1_stage1=float(res.info["stage1"]["omega1"]),
        omega1_stage2=float(res.info["stage2"]["omega1"]),
        stage1_c=np.asarray(res.info["stage1"]["c"], dtype=np.float64),
        stage2_c=np.asarray(res.info["stage2"]["c"], dtype=np.float64),
        stage1_v=np.asarray(res.info["stage1"]["v"], dtype=np.float64),
        stage2_v=np.asarray(res.info["stage2"]["v"], dtype=np.float64),
        stage1_ch=np.asarray(res.info["stage1"]["ch"], dtype=np.float64),
        stage2_ch=np.asarray(res.info["stage2"]["ch"], dtype=np.float64),
        nelx=params["nelx"],
        nely=params["nely"],
    )

    stage1_log = np.column_stack(
        (
            np.arange(1, len(res.info["stage1"]["c"]) + 1),
            np.asarray(res.info["stage1"]["c"], dtype=np.float64),
            np.asarray(res.info["stage1"]["v"], dtype=np.float64),
            np.asarray(res.info["stage1"]["ch"], dtype=np.float64),
        )
    )
    np.savetxt(case_out / "stage1_iteration_log.csv", stage1_log, delimiter=",", header="iter,compliance,volume,change", comments="")

    stage2_log = np.column_stack(
        (
            np.arange(1, len(res.info["stage2"]["c"]) + 1),
            np.asarray(res.info["stage2"]["c"], dtype=np.float64),
            np.asarray(res.info["stage2"]["v"], dtype=np.float64),
            np.asarray(res.info["stage2"]["ch"], dtype=np.float64),
        )
    )
    np.savetxt(case_out / "stage2_iteration_log.csv", stage2_log, delimiter=",", header="iter,compliance,volume,change", comments="")

    if with_dynamic:
        _, dyn_info = top99neo_dynamic_freq(
            nelx=params["nelx"],
            nely=params["nely"],
            volfrac=params["volfrac"],
            penal=params["penal"],
            rmin=params["rmin"],
            ft=params["ft"],
            ftBC=params["ftBC"],
            move=params["dyn_move"],
            maxit=params["dyn_maxit"],
            bcType=params["bcType"],
            output_dir=case_out,
            snapshot_every=max(0, snapshot_every),
            verbose=True,
        )

        _plot_dynamic_history(dyn_info["omegaHist"], case_out / "dynamic_convergence.png", f"Dynamic history ({case})")
        dyn_log = np.column_stack(
            (
                np.arange(1, dyn_info["omegaHist"].shape[0] + 1),
                dyn_info["omegaHist"],
                dyn_info["chHist"],
                dyn_info["repActive"].astype(float),
            )
        )
        np.savetxt(
            case_out / "dynamic_iteration_log.csv",
            dyn_log,
            delimiter=",",
            header="iter,omega1,omega2,omega3,change,repeated_mode",
            comments="",
        )

    return case_out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Yuksel inertial/dynamic benchmark.")
    parser.add_argument("--case", default="fig4", choices=sorted(CASES.keys()), help="Benchmark case")
    parser.add_argument("--quick", action="store_true", help="Reduced mesh and iteration counts")
    parser.add_argument("--out", type=Path, default=None, help="Output base folder")
    parser.add_argument("--with-dynamic", action="store_true", help="Also run dynamic frequency solver")
    parser.add_argument("--snapshot-every", type=int, default=0, help="Snapshot period (0 disables)")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    out = run_case(
        case=args.case,
        quick=args.quick,
        out=args.out,
        with_dynamic=args.with_dynamic,
        snapshot_every=args.snapshot_every,
    )
    print(f"Saved outputs in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
