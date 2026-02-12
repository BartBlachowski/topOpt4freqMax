#!/usr/bin/env python3
"""Quick performance test for the optimized scripts."""

from __future__ import annotations

import time
import sys
from pathlib import Path

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OlhoffApproach.Python.solver import OlhoffConfig, run_optimization


def run_quick_test() -> None:
    """Run a quick test with reduced iterations."""
    cfg = OlhoffConfig()
    cfg.nelx = 80  # Reduced from 240
    cfg.nely = 10  # Reduced from 30
    cfg.support_type = "CC"
    cfg.maxiter = 20  # Reduced from 300
    cfg.rmin = 2 * cfg.L / cfg.nelx
    cfg.beta_schedule = (1, 2, 4, 8)  # Reduced from (1, 2, 4, 8, 16, 32, 64)
    cfg.beta_interval = 8  # Reduced from 40
    cfg.n_polish = 5  # Reduced from 40
    cfg.eigs_strategy = "shift-invert"
    cfg.finalize()

    print("=" * 70)
    print("PERFORMANCE TEST - Reduced problem size for quick validation")
    print("=" * 70)
    print(f"Problem size: {cfg.nelx}x{cfg.nely} elements")
    print(f"Max iterations: {cfg.maxiter}")
    print(f"Plot iterations: False (optimized)")
    print("=" * 70)

    start_time = time.time()

    res = run_optimization(
        cfg,
        verbose=True,
        do_diagnostic=False,
        diag_modes=3,
        plot_iterations=False,  # Optimized version
    )

    elapsed_time = time.time() - start_time

    print("=" * 70)
    print(f"✓ Test completed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Final omega: {res.omega_best:.2f}")
    print(f"Average time per iteration: {elapsed_time/cfg.maxiter:.3f} seconds")
    print("=" * 70)
    print("\nWith plot_iterations=True, this would take 5-10x longer")
    print("due to GUI updates every iteration.")
    print("=" * 70)


if __name__ == "__main__":
    run_quick_test()
