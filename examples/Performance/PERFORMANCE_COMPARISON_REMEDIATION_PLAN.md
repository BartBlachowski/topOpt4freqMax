# Proposed remediation plan for the retired performance comparison

Status: proposal only; not implemented  
Depends on: `PERFORMANCE_COMPARISON_AUDIT.md`  
Audit decision: **D. REMOVE**

This plan does not attempt to rescue the current Table 1. It describes how to
contain the retired artifact and, only if scientifically necessary, design a
new experiment.

## Phase 0: immediate containment

1. Keep the quantitative cross-code table absent from the manuscript,
   supplement, response letter, and figures.
2. Rename the legacy driver and CSV so neither contains `table1`, and place
   them under an explicitly non-evidentiary internal-benchmark/archive path.
3. Add a machine-readable retirement marker and a guard that refuses
   manuscript-style export from the legacy driver.
4. Reconcile the stale formulation, settings, and endpoint statements listed
   in audit Section 14 against accepted manifests. Do this as a separate
   manuscript change with its own review.

Exit criterion: repository searches cannot mistake the legacy CSV/driver for
accepted reviewer evidence, and the active manuscript cites only validated
result manifests.

## Phase 1: decide whether a new comparison is scientifically needed

Write and approve a one-page comparison protocol before coding. It must choose
one of:

- a common direct first-eigenfrequency optimization problem;
- a common frozen-load compliance-surrogate ablation; or
- a clearly labeled comparison of exact software implementations.

The protocol must state the permissible claim. It must not label local inspired
codes as the published methods without a separate fidelity validation.

Exit criterion: one mathematical problem, one estimand, and one attribution
policy are frozen.

## Phase 2: shared formulation and implementation

Put all controlled variants behind a shared FE and instrumentation layer:

- identical mesh, geometry, constrained DOFs, passive regions, and volume
  accounting;
- identical stiffness and continuous mass interpolation;
- identical physical filter radius, boundary treatment, projection, and
  continuation;
- identical initial physical design and, if applicable, modal reference;
- explicitly frozen or fully differentiated design-dependent load;
- one optimizer and stopping rule for formulation ablations, or a justified
  end-to-end algorithm comparison when optimizers are intrinsic;
- common independent final eigensolver.

Correct the known Olhoff auxiliary-variable derivative before any use. Remove
or explicitly justify Yuksel's discontinuous mass branch. Unsupported JSON
keys must be rejected, not ignored. Every run must emit requested and effective
configuration values.

Exit criterion: branch-equivalence tests demonstrate that controlled settings
produce the same K, M, filter operator, BCs, volume, and initial state.

## Phase 3: verification gates

Before optimization campaigns:

1. Central-FD or complex-step checks at uniform, perturbed, and representative
   late designs.
2. Separate frozen-load and full design-dependent-load checks.
3. Eigenvalue sensitivity checks away from multiplicities, plus a documented
   clustered-mode treatment where needed.
4. Filter/projection chain checks, volume checks, and elemental K/M
   interpolation checks.
5. Deterministic unit tests for effective configuration and unsupported fields.

Suggested numerical gate: relative \(L_2\) gradient error below `1e-4` for
smooth, nondegenerate fixtures, with pointwise exceptions explained. A
sensitivity heuristic that cannot pass must be labeled as a heuristic and must
not be paired with KKT/stationarity claims.

Exit criterion: all applicable gradient/configuration tests pass in CI and
their raw results are archived.

## Phase 4: convergence and checkpoint instrumentation

Every run should persist:

- `converged` and a controlled `termination_reason`;
- cap hit, final objective, frequency, compliance, volume, all constraint
  violations, and stationarity/KKT or optimizer-appropriate residuals;
- raw and physical \(L_\infty/L_2\) design changes;
- move-limit activity, MMA asymptotes/duals, or OC multiplier diagnostics;
- objective/frequency/load/change histories;
- modal-load change, MAC, modal separation, selected mode, and refresh events
  when modal refresh is active;
- final 100 compact state hashes and restartable checkpoints, including
  optimizer history;
- environment, code/config hashes, random seed, and artifact schema version.

The exporter must reject any run that reaches its cap without satisfying the
declared convergence gate.

Exit criterion: a deliberately capped smoke run is marked non-converged and
cannot enter an accepted summary.

## Phase 5: independent endpoint validation

For every retained final design:

1. Reassemble K/M independently under the protocol's common definitions.
2. Compute at least the first several modes.
3. Record residuals, ordering, gaps, and rigid-mode checks.
4. Inspect modal localization and support connectivity.
5. Recompute physical volume, grayness, and minimum-feature/connectivity
   measures.
6. Compare optimizer-returned and independent frequencies within a declared
   tolerance.

Exit criterion: every endpoint passes all structural and modal gates; failures
remain failures and are not silently converted to scalar frequencies.

## Phase 6: performance protocol, only after scientific gates pass

- Run each method in an isolated MATLAB process.
- Use an untimed warm-up, then randomized/counterbalanced order.
- Measure direct end-to-end wall time over the same scope.
- Record phase times and operation counts: K/M assemblies, linear solves,
  factorizations, eigensolves, eigenpairs, trials, filter passes, optimizer
  subproblems, and refreshes.
- Use an external OS-level process peak-RSS tool over the full run.
- Use enough repetitions to report median, dispersion, and confidence
  intervals; select the count from a pilot variance analysis.
- Archive hardware, OS, MATLAB, thread count, BLAS, code commit, and configs.

Exit criterion: timing/memory statistics are reproducible and iteration totals
are never used as a proxy for common work without operation counts.

## Phase 7: claim review

Create a new result manifest that maps every proposed table cell and sentence
to an accepted artifact and gate status. Obtain an independent code/scientific
review before manuscript insertion.

Even after all phases, the result should be described as a comparison of the
implemented formulations actually tested. Attribution to published methods
requires a separate fidelity demonstration.

## Explicit non-actions in this audit

- No optimizer, wrapper, configuration, or manuscript correction was applied.
- No legacy artifact was moved or renamed.
- No full `160x20`/`240x30` rerun was launched.
- No timing or memory campaign was performed.
- No non-converged endpoint was relabeled as converged.
