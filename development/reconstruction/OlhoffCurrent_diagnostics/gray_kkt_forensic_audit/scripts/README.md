# Audit-only reproduction

From repository root, using the repository .venv:

1. `python scripts/identity.py` (prefix paths with this audit directory): validate source/evidence hashes and exact endpoints, recover configs.
2. `python scripts/geometry.py`: read saved trajectories, generate geometry and deterministic sample files.
3. Only after step 2 completes, MATLAB `frozen_evaluate`: native FE/gradient/FD kernels, no optimizer. Existing valid fd_*.mat cases are skipped. It requires MATLAB/license access. Do not run cp_fixedwork, any production solver or test suite.
4. `python scripts/stationarity.py`, then `python scripts/cluster_robustness.py`, `python scripts/local_core.py`, `python scripts/timeline.py`: offline array diagnostics. The cluster fit is an exploratory dual-space least-squares lower bound, not density optimization.
5. `python scripts/write_report.py`, then `python scripts/finalize.py`: write documents/manifests, verify input/source integrity and output hashes.

Set MPLCONFIGDIR to a writable temporary directory for plots. Invalid integer FD recordings are not inputs to any analysis. Source inputs and production are read-only. No runs directory is created. All density perturbations in the MATLAB evaluator are independent derivative checks around one immutable saved state; no perturbation becomes the next baseline.
