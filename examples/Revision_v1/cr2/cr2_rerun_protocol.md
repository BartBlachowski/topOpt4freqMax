# CR2 Rerun Protocol

**Status:** Predeclared plan only. No configuration has been edited and no
experiment has been run under this protocol.

**Study:** Omitted versus complete load sensitivity for the authoritative load

```text
F(x) = omega0^2 * M(x) * Phi0
```

**Source evidence:** `output/cr2_failure_diagnosis.md` and the preserved CR2
production artifacts. The failed endpoints are diagnostic inputs only; they are
not accepted scientific comparison endpoints.

## 1. Principles

1. Formal convergence is determined from design change, feasibility, mode
   validity, solver termination, and artifact completeness. Objective plateau is
   not a substitute for formal convergence.
2. A mode-index change is not itself mode loss. The target remains valid when
   the maximum squared mass-weighted MAC against the frozen reference mode is at
   least 0.8, even if the matching eigenmode index changes.
3. A stabilization screen that changes only Variant B is diagnostic. It is not
   an accepted A-versus-B comparison because the optimizer settings are no
   longer matched.
4. A quantitative CR2 comparison requires a final matched pair whose configs
   differ only in `load_sensitivity` and descriptive metadata.
5. No endpoint selected from an oscillatory trajectory may be described as an
   optimum or a converged design.

## 2. Formal Acceptance Criteria

The following criteria are predeclared for every candidate production endpoint.
All must pass.

| Criterion | Formal threshold |
|---|---:|
| Solver completion | `solver_success=true`, no exception |
| Iteration termination | stop before the configured cap |
| Design convergence | final `design_change <= 1e-3` |
| Feasibility | final residual `<= 1e-4` |
| Target-mode validity | final squared mass-weighted `MAC >= 0.8` |
| Tracking coverage | target searched in at least the lowest 6 modes |
| Formulation | Gate A0/V1c authoritative-load checks pass |
| Evidence | required MAT, CSV, topology, history, figure, log, and manifest artifacts exist |

The design-change threshold remains `1e-3`. Acceptance must not be triggered by
an objective plateau or by a single hand-selected phase of a limit cycle.

The feasibility threshold is revised from `1e-8` to `1e-4`. This is justified as
an explicit numerical constraint tolerance for the OC volume bisection: it is an
absolute volume-fraction residual of 0.0001, or 0.02% relative to the prescribed
volume fraction 0.5. It must be applied identically to both variants and reported
without rounding. This relaxation does not relax design convergence.

## 3. Objective-Plateau Diagnostics

Objective plateau is recorded separately and never changes formal acceptance.
For the final 50 iterations, report:

- relative objective range,
  `(max(J)-min(J))/max(abs(mean(J)), eps)`;
- relative tracked-frequency range using the same definition;
- least-squares relative objective slope per iteration;
- median and maximum design change;
- median lag-1 and lag-2 objective differences;
- median lag-1 and lag-2 topology differences;
- fraction of iterations at 99% or more of the move limit;
- feasibility range, mode indices, and minimum/maximum MAC.

A **diagnostic plateau** may be reported when the final-50 relative objective
range and tracked-frequency range are each at most 0.5%. This means only that
the response quantities are flat over that window. It does not mean the design
is converged and cannot make a capped run acceptable.

## 4. Variant A Tolerance Decision

### `conv_tol=5e-3`: not justified as formal acceptance

The observed final design change was `5.245e-3`, already above the proposed
threshold. More importantly, the diagnosed cycle occupied approximately
`0.004-0.009`, with larger spikes. A one-step `5e-3` threshold could accept or
reject the same stable cycle depending only on its phase. It was also selected
after inspecting the failed trajectory. Therefore:

- retain `1e-3` as the formal design-convergence threshold;
- use `5e-3` only as a descriptive marker in plateau diagnostics;
- do not relabel the previous Variant A endpoint as converged.

### `feas_tol=1e-4`: justified with disclosure

The observed feasibility cycle was approximately zero to `1e-4`, consistent
with OC volume-bisection resolution. The threshold is small relative to the
prescribed volume and can be applied symmetrically. It is acceptable as the
formal feasibility tolerance for the rerun, provided the exact residual and
relative volume violation are saved.

Variant A's previous run is consequently classified as **diagnostic objective
plateau with OC micro-oscillation**, not accepted convergence.

## 5. Variant B Controlled Stabilization

The first and only stabilization change is:

```text
move_limit: 0.20 -> 0.02
```

Do not switch to MMA in this first attempt. Do not change the mesh, filter,
initial design, solid reference, load definition, sensitivity formula, volume
fraction, penalization, stopping tolerance, mode tracking, or iteration cap.

This B-only run is a **stabilization screen**. Its purpose is to determine
whether the complete-gradient OC two-cycle is caused by move-limit overshoot.
Regardless of whether it converges, it cannot by itself be compared
quantitatively with Variant A at `move_limit=0.20`.

### Follow-up after the screen

- If B at `move_limit=0.02` formally converges, create a matched confirmatory
  pair by running both A and B at `move_limit=0.02`. Those two configs may differ
  only in `load_sensitivity` and metadata.
- If B remains capped with a documented non-decaying cycle, preserve it as
  diagnostic algorithm-failure evidence. Do not add MMA in the same attempt.
- MMA may be planned only as a later, separately labelled optimizer study. It
  must not be merged into the initial move-limit test or represented as the same
  CR2 comparison.

## 6. Outcome Classification

### A. Accepted converged comparison

Count the CR2 comparison as accepted only when a matched A/B pair satisfies all
formal criteria in Section 2 under identical settings except
`load_sensitivity`.

Required evidence:

- both solvers stop before the cap;
- both final design changes are `<=1e-3`;
- both feasibility residuals are `<=1e-4`;
- both final target-mode MAC values are `>=0.8`;
- complete histories show the tracked physical mode and any index changes;
- all required artifacts and the pairwise topology-difference metrics exist;
- V1a verifies the complete analytical gradient used by Variant B;
- V1c confirms config equivalence apart from the sensitivity selector.

Allowed claims:

- report the measured objective, tracked-frequency, topology, grayness,
  feasibility, and iteration differences for the matched converged pair;
- state which variant attained the reported converged values;
- discuss practical effect size using the saved values.

Not automatically allowed:

- "negligible", "equivalent", or "significant" without a separately declared
  equivalence/effect margin;
- general claims beyond this mesh, optimizer, filter, and initialization;
- causal claims that omission is universally superior.

### B. Diagnostic algorithm-failure evidence

Classify a run as diagnostic algorithm failure when all of the following hold:

- solver execution is numerically successful and all diagnostics are finite;
- the run reaches the cap without formal convergence;
- configuration, load, sensitivity, and mode-tracking validation pass;
- required failure artifacts are complete;
- the final 50 iterations show at least one predeclared instability signature:
  - design change is at least 99% of the move limit in at least 90% of the
    window; or
  - median relative lag-2 objective difference is `<=1e-4` while median
    relative lag-1 difference is `>=1e-3`; or
  - design change stays above `1e-3` throughout the window with no decreasing
    trend and a reproducible bounded cycle is present.

Mode index changes are reported, but only `MAC<0.8` constitutes mode invalidity.

Allowed claims:

- state that the tested algorithm/configuration failed to converge;
- describe the observed move-limit saturation, cycle period, plateau, mode-index
  history, and feasibility behavior;
- state that reducing the move limit did or did not stabilize Variant B under
  the tested conditions.

Forbidden claims:

- endpoint objective/frequency superiority between A and B;
- topology comparison as a comparison of converged optima;
- "the complete sensitivity is worse" or "omission is negligible";
- extrapolation from OC failure to all optimizers.

### C. Inconclusive result

Classify the result as inconclusive if any of these occurs:

- eigensolve, linear solve, optimizer, or artifact export raises an exception;
- required histories or artifacts are missing or non-finite;
- Gate A0, V1a, or V1c validation fails;
- the target mode cannot be identified with `MAC>=0.8` in the tracked spectrum;
- the run is capped but does not meet a predeclared algorithm-failure signature;
- A and B settings differ in an unplanned way;
- the stabilization run terminates for a reason not covered above.

Allowed claims:

- report only that the attempt was inconclusive and identify the failed check;
- describe diagnostics needed to resolve the uncertainty.

No scientific conclusion about the load-sensitivity omission is allowed.

## 7. Rerun Sequence

1. Archive current failed/partial artifacts unchanged.
2. Create, but do not overwrite, a B stabilization config differing only by
   `move_limit=0.02` and metadata.
3. Run V1c-style validation against the original B config and assert that the
   move limit is the sole numerical difference.
4. Run the B stabilization screen and classify it using Section 6.
5. If B formally converges, create matched A/B `move_limit=0.02` configs and
   validate that `load_sensitivity` is their only scientific difference.
6. Run the matched pair, preserve all artifacts regardless of outcome, and apply
   formal acceptance before inspecting endpoint differences.
7. Mark Gate E4 CR2 passed only for outcome A. Outcomes B and C remain
   failed/partial and permit no comparative scientific claim.

## 8. Required Rerun Artifacts

For every run save:

- immutable config copy and SHA-256;
- MAT result and complete run log;
- objective, six-mode frequency, design-change, feasibility, volume, and
  grayness histories;
- target-mode index, MAC, and tracked-frequency histories;
- omitted-versus-complete sensitivity-difference norms;
- final topology CSV and image;
- final-50 plateau and cycle diagnostics;
- acceptance report with every criterion shown separately;
- machine-readable manifest, including missing artifacts for failed runs;
- for an accepted matched pair, topology-difference CSV/image/metrics.

No config edit or experiment execution is authorized by this document.
