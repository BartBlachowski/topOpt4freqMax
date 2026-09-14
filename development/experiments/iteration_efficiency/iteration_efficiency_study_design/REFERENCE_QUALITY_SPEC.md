# Phase 1C reference-quality specification

Status: binding design for independent delta audit; not authorization to run Phase 2.

## 1. Defect being repaired

Phase 1A defined each reference as the best sustained floor anywhere in that method's
measurement observer horizon. Because those horizons were 900, 2000 per Yuksel stage,
and 3200 method-level updates, the reported `k_enter` was partly a function of an unequal
safety budget. Phase 1C accepts Critical Finding C2 and separates reference generation
from endpoint measurement.

## 2. Alternatives considered

| Construction | Fairness and reproducibility | Runs/censoring | Decision |
|---|---|---|---|
| common fixed terminal horizon | same cap, but the cap itself becomes the reference definition and can freeze a still-improving method at an arbitrary point | one long reference run; failure/cap endpoints remain semantically unequal | rejected as the baseline |
| method-independent stabilization rule | reference has the same stopping semantics for every method; the cap controls availability, not reference magnitude | separate reference run; honest `REFERENCE_NOT_ESTABLISHED` possible | **selected baseline** |
| frozen performance-campaign endpoints | no new reference run | terminal meanings differ; Proposed/Yuksel fields are absent; Olhoff is fixed-work/failure-censored | rejected |
| report reference as a function of horizon | transparent but leaves no single frozen denominator for the measurement study | many horizon-dependent results | supplementary diagnostic only |

## 3. Reference-run boundary

For each method `m` and mesh `j`, run a dedicated deterministic **reference trajectory**
from the frozen initialization/profile. It is distinct from the measurement trajectory.
Its fixed resource cap is

`B_ref = 3200` acceptance-eligible method-level updates,

chosen before Phase 2 as the largest Phase-1A outer/loop safety horizon and equal to
`32P` at `P=100`. For Proposed these are OC updates; for Yuksel they are Stage-2 updates
after the separately counted Stage 1; for Olhoff they are outer updates. The different
operations are not treated as equal computational work. The common cap is only a
censoring/resource boundary and never supplies a terminal reference value.

The reference run records the entire prefix up to `B_ref` or a required-solver
termination. Native stops are observed but do not end it. Common evaluation remains
offline. A reference may freeze at an earlier first-passage stabilization point; later
stored states cannot revise it. Later failures are disclosed.

Reference/calibration updates and time are published as `N_reference` and `T_reference`
but are never charged to `k_enter`, `k_cert`, `T_enter`, or `T_cert`.

## 4. Base-valid states

Let `H0_mj(k)` be solver health, relative raw-volume feasibility, repaired topology
validity, and the method-specific validity condition, with no spectral threshold. The
topology rule is defined in `TOPOLOGY_SANITY_SPEC.md`. Only `H0=1` states may form a
reference window.

For evaluator `e in {E1,E2,E3}`, let `Q_e,mj(k)` be the Candidate C actual-gray lowest
unanimously valid structural-mode frequency. Define the
cumulative best sustained floor through update `b`:

\[
F_{e,mj}(b)=\max_{1\le a\le b-P+1\atop H0(k)=1\;\forall k\in[a,a+P-1]}
\;\min_{k=a}^{a+P-1}Q_{e,mj}(k).
\]

`F_e(b)` is undefined until a base-valid P-window exists and is nondecreasing thereafter.

## 5. Method-independent stabilization rule

Freeze these constants before the reference run:

- block length: `P=100` updates;
- look-back: `L_ref=5P=500` updates;
- reference resolution: `epsilon_ref=0.001` (0.1%).

The 0.1% resolution is one fifth of the tightest primary quality deficit (0.5%); the
500-update look-back is five common persistence windows. Neither depends on a method,
mesh, future ranking, or native stopping rule.

At block endpoints `b=tP`, starting only when `F_e(b-L_ref)` exists for all evaluators,
compute

\[
g_e(b)=\frac{F_e(b)-F_e(b-L_{ref})}{F_e(b)}.
\]

The **reference freeze index** is the first block endpoint satisfying

\[
b_ref=\min\{b:g_e(b)\le\epsilon_ref\quad\text{for all }e\in\{E1,E2,E3\}\}.
\]

Then freeze, separately for each evaluator,

\[
Q^{ref}_{e,mj}=F_{e,mj}(b_ref).
\]

This is a causal first-passage rule on the reference prefix. The offline engine must stop
its logical scan at the first qualifying `b_ref`; it may not inspect later quality to
choose a different reference. A later observer failure is reported but cannot revise an
already frozen reference.

If no `b_ref` exists by `B_ref`, return `REFERENCE_NOT_ESTABLISHED`. If a required solver
terminates before `b_ref`, return `REFERENCE_SOLVER_TERMINATION` with the exact backend
classification. **There is no fallback to the best floor at the cap.** This is what
prevents the resource horizon from lowering the quality bar.

## 6. Measurement independence and identity

After reference values, `b_ref`, and their provenance hashes are frozen, start a separate
measurement trajectory from the identical initialization/profile. Its horizon is fixed by
the binding Phase-1E equation

\[
B_{meas}=\min\{\max(B_0,b_{ref}+P-1),B_{ref}\},
\]

with no progress-triggered extension. This can affect whether an endpoint is observed, but
cannot alter any `Q_ref`. Verify reference/measurement trajectory fingerprints at several
shared counts and at every reported endpoint. A mismatch is an implementation failure, not
a new reference opportunity.

The trajectory runner receives the frozen `b_ref` only to calculate `B_meas`; the acceptance
scan receives only the frozen triplet `(Q_ref_E1,Q_ref_E2,Q_ref_E3)` and provenance. Neither
component can recompute reference values from the measurement horizon.

## 7. Sensitivity and failure interpretation

Mandatory offline reference diagnostics are:

- `F_e(b)` versus `b` for E1/E2/E3;
- the first-passage `b_ref` and all `g_e(b)` values;
- a horizon diagnostic showing the reference that the superseded Phase-1A rule would
  have produced at 900, 1600, 2000, and 3200 where those prefixes exist;
- native stop, method-gate satisfaction, solver termination, and cap locations.

These diagnostics test C2 closure; they cannot select another reference rule. Reference
stability is not optimizer stationarity and must not be called convergence.

## 8. Conditional A and mandatory best-observed comparison

Engineering estimand A remains absent unless `Omega_req(mesh)` has an independent source
whose content and provenance hash are locked into the protocol manifest before any
production run. The acceptance engine hard-fails if this record is absent or postdates
production. No study trajectory may supply `Omega_req`.

The non-engineering **best-observed benchmark** is mandatory. For evaluator `e`:

\[
Q^{BO}_{e,j}=\max_m Q^{ref}_{e,mj}.
\]

Report attainment/status relative to `q Q_BO` under the same quality levels as R. It is a
symmetric descriptive comparison, never A, absolute adequacy, or an engineering
requirement.

## Phase 2H evaluator amendment

Every `Q_e(k)`, floor `F_e(b)`, gain, and `Q_ref_e` uses the actual-gray Candidate C
selected structural frequency, never raw mode 1 and never the binary diagnostic. Reference
windows containing an invalid evaluator state are not base-valid. If no structural mode
can be selected, `STRUCTURAL_MODE_NOT_FOUND` propagates and reference establishment fails
closed. `B_ref=3200`, `P=100`, `L_ref=500`, and `epsilon_ref=0.001` are unchanged.
