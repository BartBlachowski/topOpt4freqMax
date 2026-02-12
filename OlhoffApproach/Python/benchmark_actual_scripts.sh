#!/bin/bash
# Benchmark actual scripts with small problem size

echo "========================================================================"
echo "BENCHMARKING ACTUAL SCRIPTS"
echo "========================================================================"

cd "$(dirname "$0")"

# Modify scripts temporarily to use quick mode
echo ""
echo "Testing optimized run_clamped_clamped.py (quick mode)..."
echo "------------------------------------------------------------------------"
time python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

from OlhoffApproach.Python.solver import OlhoffConfig, run_optimization
import matplotlib
matplotlib.use('Agg')

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

res = run_optimization(cfg, verbose=False, do_diagnostic=False, diag_modes=3, plot_iterations=False)
print(f'Result: omega={res.omega_best:.2f}')
"

echo ""
echo "Testing run_case.py (quick mode)..."
echo "------------------------------------------------------------------------"
time python run_case.py --case CC --quick --max-iter 15 --out /tmp/olhoff_test --snapshot-every 0 2>&1 | grep -E "(omega|Saved outputs)"

echo ""
echo "========================================================================"
echo "If times are similar, optimizations worked!"
echo "If run_case.py is much faster, there may be another issue."
echo "========================================================================"
