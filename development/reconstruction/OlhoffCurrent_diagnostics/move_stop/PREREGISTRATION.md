# PREREGISTRATION — canonical move/stopping diagnostic

Frozen **before the first diagnostic optimization run**. Everything below —
hypotheses, arms, meshes, caps, metrics, thresholds and the decision rule — is
fixed in advance. The SHA-256 of this file at freeze time is recorded in
`FINAL_SHA256.txt` alongside the run artifacts.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Baseline HEAD (Phase A) | `a19d13eccc8dfe6487145f6c10ad6f8e024be525` |
| Implementation | `analysis/OlhoffCurrent` (the sole production Olhoff) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 source files) |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` |
| MATLAB | R2025b Update 1 (25.2.0.3042426), **1 computational thread** |

---

## 1. Question

Is the refinement-dependent grayness of the canonical, sensitivity-filtered,
projection-free Du–Olhoff reconstruction driven primarily by the interaction of

* the **move-limit ladder**,
* the **L2 convergence measure on Δρ**, and
* the **`settledMove` admission guard**,

rather than by filtering, `p`, mass interpolation, multiplicity or projection?

All three of those ingredients are class **C** reconstruction choices in
`PROVENANCE_LEDGER.md`; none has a numeric value anywhere in the Du–Olhoff
lineage.

## 2. Hypotheses

**H1 (primary).** The move-limit descent mechanically reduces the L2
design-change measure enough that the stopping test can be satisfied before the
topology has re-equilibrated, and this effect grows with mesh refinement.

**H2 (secondary).** Removing the descent, while leaving the scientific
formulation unchanged, substantially reduces the refinement-driven excess
grayness.

**Explicitly NOT predicted:** a binary topology. Earlier evidence suggests a
**residual grayness floor may remain** after the refinement-driven excess is
removed. A non-zero floor in the fixed-move arm does **not** refute H1 or H2.

**H0.** Grayness is not materially associated with move-descent events; the
fixed-move arm shows no substantial reduction in refinement-driven excess.

## 3. Arms

Exactly two. **One conceptual factor differs.**

| Arm | Configuration | Label in all figures/tables |
|---|---|---|
| **Baseline** | the production preset, **unchanged**, no scientific override | `Production: move ladder` |
| **Fixed move** | identical, except `move.policy = fixed`, `move.initial = 0.04` | `Diagnostic: fixed move 0.04` |

Held identical in both arms: `p = 3` fixed, mass `eq4b`, `q = 1`, sensitivity
filter, `R = 0.06·b`, `filter.applyTo = all`, projection **off**, multiplicity
`subspace`/`N = 2`/off-diagonals on, published MMA on the increment,
`tolInner = 0.05`, `stop.field = designVariable`, `stop.norm = l2`,
`stop.toleranceRule = meshScaled`, `settledMove = true`.

`move = 0.04` is the production ladder's **first level**, chosen for that reason
alone. **It will not be re-chosen after seeing results.**

The fixed-move arm is a **diagnostic instrument**, not a candidate production
method.

## 4. Meshes

**160×20 and 320×40 only.** No 800×100. No nine-mesh campaign. 400×50 may be
authorized as a confirmatory mesh **only** if the primary result is ambiguous,
and only after the primary analysis is complete. Sub-160×20 meshes may be used
for software mechanics only, never as scientific evidence.

## 5. Iteration cap and the meaning of termination

**Diagnostic cap = 400 outer iterations, both arms, both meshes.** This is the
production `runtime.maxOuter`, deliberately reused so that no new number is
invented for this experiment.

The inherited stopping rule is **not** modified: `stop.tolerance`,
`stop.norm`, `stop.field` and `settledMove` are untouched in both arms. No
threshold will be introduced or adjusted after observing a trajectory.

A run that reaches the cap is reported **`CAP_HIT`** and is **not** relabelled as
converged. `CAP_HIT` does not erase trajectory evidence: for such a run the
report will state separately whether ω₁ stabilized, whether M_nd stabilized,
whether the design change stabilized, and whether ‖Δρ‖₂ remained above ε.

**Prediction registered in advance:** the fixed-move arm may fail to satisfy the
inherited L2 tolerance at 160×20 even when its topology metrics are stable. That
outcome is anticipated and is *evidence*, not failure.

## 6. Metrics (all on the physical density)

Because projection is off and density filtering is off, the design variable **is**
the physical FE density. This will be **verified**, not assumed.

* `M_nd = 100 · mean(4ρ(1−ρ))` — measure of non-discreteness (%)
* gray fraction — fraction of elements with `0.1 < ρ < 0.9`
* mid-density fraction — fraction with `0.4 ≤ ρ ≤ 0.6`
* `ω₁`, `ω₂`, `gap12`, volume

### Per-iteration recorders

From `res.hist`: `ω`, `N`, `β` (bound variable), `nInner`, `innerConv`,
`cumInner`, `dxOuter = max|Δρ|`, `dxNorm2 = ‖Δρ‖₂`, `vol`, `move`, `stage`,
`gap12`, `volErr`, `degen`, `multJ`.

From `res.diag.drho{k}` (`runtime.diagnostics = true`, documented purely
additive and bitwise inert — **which the baseline reproduction in §8 tests**):
the full per-iteration Δρ vector, from which ρ_k is reconstructed exactly as the
solver forms it, `ρ_k = min(1, max(ρ_min, ρ_{k−1} + Δρ_k))` with
`ρ_0 = 0.5`, and **validated against `hist.vol(k) = mean(ρ_k)`**.

Derived per iteration: `RMS(Δρ) = ‖Δρ‖₂/√NE`, the stop predicate, the
`settledMove` state, `M_nd`, gray and mid-density fractions.

### Active-set counts (registered thresholds, fixed in advance)

`N_active(τ) = #{e : |Δρ_e| > τ}` for

* `τ = ε_RMS = ε/√NE = 8.8388347648318442e-04` (mesh-independent by construction), and
* fixed diagnostic levels `τ ∈ {1e-4, 1e-3, 1e-2}`.

## 7. Analyses, specified in advance

**A1 — transition response.** For every move descent in the baseline, the values
of `move`, `‖Δρ‖₂`, `RMS(Δρ)`, `max|Δρ|`, `M_nd`, `ω₁`, volume immediately
before and immediately after; the step-measure reduction factor; and the
*simultaneous* topology change. Plus `stop_iteration − last_move_descent_iteration`
per mesh.

**A2 — mesh scaling.** Measure `N_active`; do **not** assume it. Test whether
`N_active` is approximately constant under refinement and hence whether
`RMS(Δρ) ≈ move·√(N_active/NE)` dilutes the global statistic as NE grows. **The
data will not be forced to this model if it does not fit.**

**A3 — grayness decomposition** at 320×40: baseline vs fixed move on `M_nd`,
gray fraction, mid-density fraction, `ω₁`, volume.

## 8. Baseline validity gate

The baseline arm must reproduce the archived production anchors
(`examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat`)
for the final design, `ω₁`, outer count and inner count. **Bitwise where the
archived field exists.** Because the baseline is run with
`runtime.diagnostics = true` while the archive was produced with it off, this
simultaneously tests the recorder's documented inertness.

Failure ⇒ **`BASELINE_REPRODUCTION_FAIL`**, and the diagnostic stops.

## 9. Decision rule, fixed in advance

| Verdict | Condition |
|---|---|
| **`MOVE_STOP_INTERACTION_CONFIRMED`** | Baseline termination follows closely after a move descent at **both** meshes, **and** the descent reduces `‖Δρ‖₂` by a factor materially larger than the simultaneous change in `M_nd`/topology, **and** the fixed-move arm substantially reduces `M_nd` at 320×40 |
| **`MOVE_STOP_INTERACTION_PARTIAL`** | The descent–termination coupling holds, but the fixed-move arm reduces `M_nd` only modestly; or the effect is clear at one mesh and not the other |
| **`MOVE_STOP_INTERACTION_REFUTED`** | Termination is not associated with descents, **or** the fixed-move arm does not reduce `M_nd` |
| **`INCONCLUSIVE`** | Evidence insufficient or internally contradictory at these two meshes |

"Substantially reduces" is fixed here as a **relative reduction in `M_nd` at
320×40 of ≥ 25 %** between `Production: move ladder` and
`Diagnostic: fixed move 0.04`, evaluated at each arm's own terminal iteration.
"Modestly" is a reduction that is positive but < 25 %.

Production decision is fixed in advance regardless of outcome:
**`KEEP_PRODUCTION_PRESET_PENDING_REVIEW`**. No production file will be changed
in this task.

## 10. Prohibited in this task

No projection, no density filtering, no β sharpening, no p continuation, no mass
continuation, no tolerance tuning, no filter-type change, no change to `p`, mass
interpolation or multiplicity, no promotion of fixed move into production, no
800×100, no nine-mesh campaign, no reorganization of historical directories.
