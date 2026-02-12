#!/usr/bin/env python3
"""Compare actual script performance."""

import subprocess
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Test 1: Inline version simulating run_clamped_clamped.py
print("="*70)
print("TEST 1: Simulating run_clamped_clamped.py (optimized)")
print("="*70)

from OlhoffApproach.Python.solver import OlhoffConfig, run_optimization

cfg = OlhoffConfig()
cfg.nelx = 80
cfg.nely = 10
cfg.support_type = 'CC'
cfg.maxiter = 15
cfg.rmin = 2 * cfg.L / cfg.nelx
cfg.beta_schedule = (1, 2, 4, 8)
cfg.beta_interval = 8
cfg.eigs_strategy = 'shift-invert'
cfg.finalize()

start = time.time()
res1 = run_optimization(cfg, verbose=False, do_diagnostic=False, diag_modes=3, plot_iterations=False)
time1 = time.time() - start

print(f"Time: {time1:.2f}s | omega: {res1.omega_best:.2f}")

# Test 2: Using case_config like run_case.py
print("\n" + "="*70)
print("TEST 2: Simulating run_case.py method")
print("="*70)

from OlhoffApproach.Python.cases import case_config

cfg2 = case_config('CC', quick=True, seed=None)
cfg2.maxiter = 15  # Match test 1

start = time.time()
res2 = run_optimization(cfg2, output_dir=None, verbose=False, do_diagnostic=False, diag_modes=3, snapshot_every=0)
time2 = time.time() - start

print(f"Time: {time2:.2f}s | omega: {res2.omega_best:.2f}")

# Summary
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Test 1 (run_* style):  {time1:.2f}s")
print(f"Test 2 (run_case style): {time2:.2f}s")
print(f"Difference: {abs(time1-time2):.2f}s ({abs(time1-time2)/max(time1,time2)*100:.1f}%)")

if abs(time1 - time2) < 0.5:
    print("\n✓ Both methods have essentially the same performance!")
    print("\nConclusion: Your optimizations worked. Both run at the same speed now.")
elif time2 < time1:
    print(f"\n! run_case.py style is {time1/time2:.1f}x faster")
    print("\nThis requires further investigation...")
else:
    print(f"\n! run_* style is {time2/time1:.1f}x faster")

print("="*70)
