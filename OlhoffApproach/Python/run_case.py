#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

try:
    from .cases import PAPER_TARGETS, case_config
    from .solver import run_optimization
except ImportError:  # direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from OlhoffApproach.Python.cases import PAPER_TARGETS, case_config
    from OlhoffApproach.Python.solver import run_optimization

matplotlib.use("Agg")


def _plot_topology(x_phys: np.ndarray, nely: int, nelx: int, omega: float, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.imshow(1 - x_phys.reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"{title} | omega1={omega:.2f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_case(
    case: str = "all",
    quick: bool = False,
    out: Path | None = None,
    max_iter: int | None = None,
    seed: int | None = None,
    snapshot_every: int = 5,
) -> list[Path]:
    base_out = out if out is not None else Path(__file__).resolve().parent / "output"
    base_out.mkdir(parents=True, exist_ok=True)

    cases = ["CC", "CS", "SS"] if case.lower() == "all" else [case.upper()]
    created: list[Path] = []

    for c in cases:
        case_out = base_out / c
        case_out.mkdir(parents=True, exist_ok=True)

        cfg = case_config(c, quick=quick, seed=seed)
        if max_iter is not None:
            cfg.maxiter = max_iter

        res = run_optimization(
            cfg,
            output_dir=case_out,
            verbose=True,
            do_diagnostic=True,
            diag_modes=5,
            snapshot_every=max(0, snapshot_every),
        )

        _plot_topology(res.xPhys_best, cfg.nely, cfg.nelx, res.omega_best, case_out / "final_plot.png", f"Olhoff {c}")
        np.savez(
            case_out / "final_design_field.npz",
            xPhys_best=res.xPhys_best,
            omega_best=res.omega_best,
            objective_history=res.objective_history,
            volume_history=res.volume_history,
            grayness_history=res.grayness_history,
            nelx=cfg.nelx,
            nely=cfg.nely,
        )

        target = PAPER_TARGETS[c]
        summary = (
            f"Case {c}\n"
            f"initial paper omega1: {target['init']:.1f}\n"
            f"optimal paper omega1: {target['opt']:.1f}\n"
            f"code omega1: {res.omega_best:.4f}\n"
        )
        (case_out / "summary.txt").write_text(summary, encoding="utf-8")
        created.append(case_out)

    return created


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Olhoff SIMP+MMA benchmark.")
    parser.add_argument("--case", default="all", choices=["all", "CC", "CS", "SS"], help="Boundary-condition case")
    parser.add_argument("--quick", action="store_true", help="Reduced mesh/iterations for smoke tests")
    parser.add_argument("--out", type=Path, default=None, help="Output base folder")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--snapshot-every", type=int, default=5, help="Snapshot frequency (0 disables)")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    outputs = run_case(
        case=args.case,
        quick=args.quick,
        out=args.out,
        max_iter=args.max_iter,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
    )
    for p in outputs:
        print(f"Saved outputs in: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
