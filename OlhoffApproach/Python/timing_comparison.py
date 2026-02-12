#!/usr/bin/env python3
"""Direct timing comparison between different calling methods."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OlhoffApproach.Python.solver import OlhoffConfig, run_optimization
from OlhoffApproach.Python.cases import case_config


def time_method_1_manual_config():
    """Method 1: Manual configuration (like run_clamped_clamped.py)"""
    print("\n" + "="*70)
    print("METHOD 1: Manual configuration (run_clamped_clamped.py style)")
    print("="*70)

    cfg = OlhoffConfig()
    cfg.nelx = 80
    cfg.nely = 10
    cfg.support_type = "CC"
    cfg.maxiter = 15
    cfg.rmin = 2 * cfg.L / cfg.nelx
    cfg.beta_schedule = (1, 2, 4, 8)
    cfg.beta_interval = 8
    cfg.eigs_strategy = "shift-invert"
    cfg.finalize()

    start = time.time()
    res = run_optimization(
        cfg,
        verbose=False,
        do_diagnostic=False,
        diag_modes=3,
        plot_iterations=False,
    )
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s | omega: {res.omega_best:.2f}")
    return elapsed


def time_method_2_case_config():
    """Method 2: Using case_config (like run_case.py)"""
    print("\n" + "="*70)
    print("METHOD 2: Using case_config() (run_case.py style)")
    print("="*70)

    cfg = case_config('CC', quick=True, seed=None)
    cfg.maxiter = 15  # Match method 1

    start = time.time()
    res = run_optimization(
        cfg,
        output_dir=None,  # Disable output for fair comparison
        verbose=False,
        do_diagnostic=False,
        diag_modes=3,
        snapshot_every=0,  # Disable snapshots for fair comparison
    )
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s | omega: {res.omega_best:.2f}")
    return elapsed


def main():
    print("\n" + "="*70)
    print("TIMING COMPARISON - Quick Test (80x10, 15 iterations)")
    print("="*70)

    # Run each method
    time1 = time_method_1_manual_config()
    time2 = time_method_2_case_config()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Method 1 (manual config):    {time1:.2f}s")
    print(f"Method 2 (case_config):      {time2:.2f}s")
    print(f"Difference:                  {abs(time1-time2):.2f}s")
    print(f"Ratio:                       {max(time1,time2)/min(time1,time2):.2f}x")

    if abs(time1 - time2) < 0.5:
        print("\n✓ Performance is essentially identical!")
    elif time1 < time2:
        print(f"\n→ Method 1 is {time2/time1:.1f}x faster")
    else:
        print(f"\n→ Method 2 is {time1/time2:.1f}x faster")

    print("="*70)


if __name__ == "__main__":
    main()
