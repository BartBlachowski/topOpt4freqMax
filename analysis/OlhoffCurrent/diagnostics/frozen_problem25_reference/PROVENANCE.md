# PROVENANCE — frozen_problem25_reference

## 1. Locks honoured

| lock | status |
|---|---|
| topology-optimization runs | **0** — no `olhoffSolve` / `olhoffcurrent_run` call |
| outer iterations | **0** |
| accepted ρ updates | **0** — `rho + drho` never formed as a design; every `drho` discarded after metrics |
| controller transitions | **0** |
| production files modified | **0** — nothing under `+impl/` or the OlhoffCurrent root written; `git status` shows only this study directory as new |
| move limit / filter / p / MMA constants altered | **none** |
| packages installed | **0** — `coneprog`, `fmincon`, `linprog` from the installed Optimization Toolbox 25.2 |

Every solve is labelled FROZEN-SUBPROBLEM REFERENCE or FROZEN-SUBPROBLEM
REPLAY in the scripts and in the JSON it writes.

## 2. Preregistration

`AUDIT_PREREGISTRATION.md`, SHA-256
`37cbbb6b10a9fc7db4fe7f7f2555724795a151cb3641bc96eea9579ca4ca6320`, written
before any solve (`evaluations/preregistration_sha256.txt`). No bar, tolerance,
start point, metric or verdict rule was changed after a result was seen. The
deviations below are disclosed, not corrected.

## 3. Disclosed deviations and imperfections

1. **Non-preregistered sub-checks in the first equivalence run.** The first
   version of `fp_equivalence.m` additionally tested "dlam sorted" (fails
   trivially for the offset form — the increments are measured from different
   baselines; the preregistered criterion `e₁ ≤ e₂` passes) and a literal
   sign test at points constructed to be exactly active (production value 0.0,
   cone residual 1e−15 of either sign). The script was corrected to the
   preregistered criteria plus a three-way class with a ±1e−12 active band.
   No preregistered bar changed; both versions' outputs are described in
   `CLUSTER_CONSTRAINT_REDUCTION.md` and `SOCP_EQUIVALENCE.md`.
2. **`coneprog` did not give exitflag 1 at the preregistered tolerance.** All
   1e−10 runs stalled (−7) with converged primal and unconverged duals. A
   sweep of linear solvers and an exact unit-box reparametrization was run
   (`fp_coneprog_sweep.m`); the reference was chosen by the smallest
   *independently certified* duality gap (x-form, `schur`, 6.4e−12), not by
   exit flag. The only exitflag-1 runs (1e−8, t-form) stop 3.2e−5 short and
   were rejected. The sweep script's own mechanical selection preferred
   exitflag 1; `fp_reference.m` overrides it. `CONIC_REFERENCE.md`.
3. **The KKT verdict uses certificate multipliers, not `coneprog`'s.** The
   solver's returned duals give a stationarity residual of 0.42 (FAIL); the
   maximized-dual-bound multipliers give 1.9e−15 with a verified gap of
   9.2e−12 (PASS). Both are reported; the preregistration's "mathematically
   correct KKT treatment" clause is what the certificate multipliers satisfy.
4. **The certificate multipliers were re-aligned once.** The free maximizer
   tilts the cone direction by ~1e−6 for a bound better by <1e−12; the aligned
   direction (exact subgradient at x) was preferred on ties so that
   stationarity is exact. Both gaps (6.4e−12, 9.2e−12) are reported.
5. **Gradient-check step.** The preregistered central-difference step 1e−8 is
   below the roundoff-optimal scale; rows 2 and 4 "fail" at 1e−8 with a 1/h
   error signature and pass at 1e−6…1e−5 (min errors 2.8e−10 … 6.4e−8). A step
   sweep replaced the single step. `FMINCON_CROSSCHECK.md`.
6. **`fmincon` first launch crashed** on a wrapper bug (missing argument in
   the output-function "get" call) before any solve; fixed and relaunched; no
   result came from the crashed process.
7. **L-BFGS cross-check incomplete.** Only S0 finished (1.5 h per start); the
   process was stopped at finalization. Exact-Hessian runs cover all seven
   starts.
8. **SQP not viable** at n = 28 801 (killed at 34:46 inside its first dense
   QP, 20.4 GB); the OutputFcn cap cannot interrupt a QP.
9. **Trajectory classification is E by the preregistered rules**, although the
   distance decreases monotonically over 1 000–5 000; the rules' thresholds
   were not loosened to obtain a label.
10. **The preregistered "move-bound dominated" rule** (≥ 50 % at a move-limited
    bound) returns "no" (33.7 %) for a solution with 99.94 % of variables on a
    bound, because 19 140 boxes are narrower than the move on one side. Both
    readings are reported; the preregistered one is the verdict input.
11. The task prompt expected `dOff` absent; it is present. The cone was derived
    for the offset form; all equivalence tests were done on the actual form.

## 4. Compute performed — all read-only

| activity | calls / iterations | wall |
|---|---|---|
| state identity: FE assembly + eigensolve + gradients at ρ₃₈₅ | 1 | ~1 min |
| equivalence tests | 15 points × 3 `bs` + toy `coneprog` + Hessian FD | ~1 min (×3 runs) |
| `coneprog` | 1 preregistered + 20 sweep + 3 reference re-runs | 3–54 s each |
| dual-bound maximization (`fminsearch`) | per candidate | seconds |
| `fmincon` interior-point, exact Hessian | 7 starts | 17–25 s each |
| `fmincon` interior-point, L-BFGS | 1 start completed | 5 634 s |
| `fmincon` sqp | 0 iterations | killed at 2 086 s |
| frozen MMA replay | 5 000 sub-iterations | 5 984 s |
| `linprog` insight | 1 | seconds |

## 5. Audit-only code and its relation to production

`scripts/fp_replay.m` mirrors `+impl/algo/innerLoop.m` character for character
in every numeric expression (differences: captured duals, checkpoints, stop
tolerance 1e−12). Its fidelity is proved by bitwise reproduction of
`DRHO(:,386)` at 19 and of the prior audit's 500-iterate at 500.
`scripts/fp_problem.m::local_evalProd` mirrors the constraint block of
`innerLoop.m` and calls the production `deltaLambda`. Everything else
(`fp_setup, fp_state, fp_hash, fp_testpoints, fp_equivalence, fp_kkt,
fp_dualbound, fp_coneprog, fp_coneprog_sweep, fp_reference, fp_fmincon,
fp_fmincon_sqp, fp_gradcheck, fp_lp_insight, fp_compare, fp_figures.py,
fp_finalize.py`) is new to this task.

## 6. Host and toolchain

Apple M1 Max, 10 cores, 64 GiB; macOS 26.6.2; MATLAB 25.2.0 (R2025b),
Optimization Toolbox 25.2; `maxNumCompThreads(1)`. Python 3.13 with numpy
2.3.4, matplotlib 3.10.7, h5py 3.16 for figures and finalization. **The host
was shared** with three unrelated single-threaded MATLAB jobs
(`run_repro`, 800×100) for the whole task; no reported quantity is a timing
measurement.

## 7. Epistemic class

The reference is the exact solution of the **production reconstruction** of
Du & Olhoff's problem (25) at one frozen state, including the reconstruction's
move box. It is not a statement about the authors' implementation, nor about
the paper's move-box-free (25). Class C, unchanged.
