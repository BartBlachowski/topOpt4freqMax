# A4 Recovery Phase 2 — Scientific Specification

**Status:** authoritative. This document is the Source of Truth for the Phase 2 implementation.
**Supersedes, for the matters in scope:** `A4_SPECIFICATION_V3.md` §4.1, §4.3.1, §5.2 (code B3), §6.1 Gate A4-Pre checkpoint set.
**Leaves untouched:** every other provision of `A4_SPECIFICATION_V3.md`, which remains binding.

**Scope:** defects **C-1** (mode-window truncation) and **C-2** (screening exposure confounded with `N`) only.
**Already closed in Phase 1 and not reopened:** **C-3** (threshold void floor) and **C-4** (deterministic hashing), per `A4_RECOVERY_PHASE1_REPORT.md`.

**Authoritative inputs, treated as settled:**

| Input | Standing |
|---|---|
| `examples/Revision_v1/A4_SPECIFICATION_V3.md` | Experiment definition; binding except where this document explicitly amends it |
| `A4_SCIENTIFIC_AUDIT.md` (2026-07-20) | Independent verification; findings treated as fact, not as hypotheses |
| `A4_RECOVERY_PHASE1_REPORT.md` | Phase 1 corrections; C-3/C-4 are closed |

**Prohibitions carried into Phase 2.** No previously settled question may be reopened by the
implementation: not the mass model (`pmass = 1`), not the baseline (`solid`), not the equivalence
margin (`δ = 5%`), not the level set `N ∈ {∞, 50, 10, 5, 1}`, not the optimizer, mesh, filter,
load model, or sensitivity model. Phase 2 corrects **how the experiment observes itself**. It does
not change what the experiment asks.

---

## Section 1 — Scientific objective

### 1.1 The hypothesis A4 exists to answer

Unchanged from `A4_SPECIFICATION_V3.md` §1.1–§1.2, restated here so that Phase 2 cannot drift
from it:

> **Does optimizing the quasi-static surrogate with a permanently frozen reference eigenpair
> `(ω₀, Φ₀)` produce a design whose true fundamental frequency is materially worse than one
> obtained by periodically refreshing that eigenpair on the evolving design — and if so, at what
> refresh interval `N*`, and by what mechanism, does the frozen mode cease to be a valid proxy?**

Judged against a pre-registered equivalence margin **`δ = 5%`** on the tracked true fundamental
frequency `ω₁ᵗʳᵃᶜᵏ`, obtained from an independent exact eigensolve of the final design:

- **H₀ (freezing is benign):** for every finite `N`, `[ω₁ᵗʳᵃᶜᵏ(N) − ω₁ᵗʳᵃᶜᵏ(∞)] / ω₁ᵗʳᵃᶜᵏ(∞) ≤ δ`.
- **H₁ (freezing costs accuracy):** some finite `N`, from a run that is clean, yields
  `ω₁ᵗʳᵃᶜᵏ(N) > ω₁ᵗʳᵃᶜᵏ(∞)·(1 + δ)`.
- **Pre-registered third outcome:** the refresh reference is *unavailable* — no admissible
  refresh reference can be constructed on the intermediate designs, so no refreshed arm can
  referee the frozen arm.

`δ`, the hypotheses, and the four pre-registered decision outcomes of `A4_SPECIFICATION_V3.md`
§5.3 are **frozen**. Phase 2 introduces no new outcome and re-weights none.

### 1.2 What Phase 2 changes, and what it does not

The audit established, by direct re-execution and not by inference, that the completed run did not
measure the hypothesis above. It measured a property of its own instrument: with a hard-coded
20-mode search window, the admissibility screen cannot locate the physical mode on designs between
roughly iteration 2 and 40 of **any** arm, including the frozen arm that never refreshes. The
reported causal chain — *small `N` → contaminated spectrum → reference unavailable → INDETERMINATE*
— has a false first link. The observed chain is *small `N` → screened earlier → screened inside the
void-mode regime → window too narrow → arm terminated*.

The distinction Phase 2 must enforce throughout:

| Scientific objective | Implementation strategy |
|---|---|
| Whether freezing `(ω₀, Φ₀)` costs true frequency, and at what `N*` | How many modes must be computed to *find* the physical mode on a given design |
| Whether refreshing destabilizes the optimization | When and how often the design is *inspected* |
| Whether the intermediate spectra admit a valid refresh reference at all | The numerical width of the eigensolve window used to answer that question |

Everything in the left column is a finding. Everything in the right column is apparatus. The
completed run published an item from the right column as if it belonged to the left. Phase 2's
entire purpose is to make that substitution impossible:

- **C-1 correction.** The mode-search window ceases to be a fixed implementation constant and
  becomes an **adaptive, deterministic search with a declared ceiling**, whose *outcome* — the
  window size actually required — is promoted to a **recorded response variable**. The number of
  modes needed to find the physical mode on an intermediate design is a genuine and reportable
  property of a formulation in which `E_min/E₀ = ρ_min/ρ₀ = 10⁻⁹` gives void the same wave speed
  as solid. It was previously invisible because it was hard-coded.
- **C-2 correction.** Admissibility screening ceases to be an event triggered by refreshing and
  becomes a **common, `N`-independent diagnostic schedule applied identically to all five arms,
  including `N = ∞`**. Screening exposure therefore stops being a function of the independent
  variable.

Phase 2 does **not** attempt to answer the hypothesis differently, add arms, add benchmarks, alter
the decision rule, or reinterpret the frozen arm. Those are separate matters, some of them
out of scope (§7.6, §9.5).

### 1.3 Standard of success

Phase 2 succeeds if, after it, a reviewer can state — from the artifacts alone, without
re-execution — **which mode was selected at every screening event of every arm, out of which
candidate set, under which window, and why every other candidate was rejected.** Nothing less
discharges the audit, because the audit's central finding was that this information had never been
recorded and had to be reconstructed by re-running the frozen trajectory by hand.

---

## Section 2 — Independent variable

### 2.1 The only experimental treatment

| Factor | Levels | Operational definition |
|---|---|---|
| **Refresh interval `N`** | `{∞, 50, 10, 5, 1}` | The iteration schedule on which the reference eigenpair `(ω₀, Φ₀)` used to build the load `f = ω₀²M(x)Φ₀` is **replaced**. `N = ∞`: computed once on the solid reference domain, never replaced. `N = k`: replacement is *attempted* at every iteration `i` with `i mod k = 0`. |

`N` is injected by the driver as the sole override on a single base configuration file,
`examples/Revision_v1/a4_ss_400x50_base.json`. Sibling configuration files remain prohibited.

Note the wording: `N` schedules an **attempt**. Whether an attempt succeeds is a measured outcome
(§5.4), not part of the treatment definition.

### 2.2 What is explicitly NOT an independent variable

Each item below either was, or could be mistaken for, a second factor. Each is fixed, or is
demoted to a response variable, or is made identical across arms by construction.

| Quantity | Status in Phase 2 | Why it must not vary with `N` |
|---|---|---|
| **Mode-search window size** | **Response variable.** Determined per screening event by the deterministic protocol of §3; never configured per arm. | It was the true cause of the falsified B3 verdict. As a constant it silently determined the outcome; as a response it becomes evidence. |
| **Screening schedule** | **Fixed common grid**, identical for all arms including `N = ∞` (§4). | This is the C-2 confound. When screening happened only at refreshes, `N` controlled both the treatment and the observation schedule. |
| **Diagnostic instrumentation** | Fixed, mandatory, identical, and provably non-perturbing (§4.4). | Instrumentation that varies with the arm reintroduces the confound in a new form. |
| **Screen thresholds** | Fixed and config-declared: support-component kinetic fraction `≥ 0.5`; dominant component touches both supports; low-density strain fraction `≤ 0.5` evaluated at `x < 0.1`; MAC continuity `≥ 0.8`. | The completed implementation drifted the low-density threshold to `0.05` without declaration (audit m-6). Thresholds are part of the protocol, not of the code. |
| **Eigensolver settings** | Fixed (shift-invert, declared shift, declared tolerance), independent of window rung except for the number of modes requested. | Solver settings that co-vary with the window would make the window's effect uninterpretable. |
| **Threshold void floor** | Fixed at the configured `rho_min = 10⁻⁹`. Closed in Phase 1 (C-3). | Not reopened. |
| **Provenance hash algorithm** | Fixed: wrapping FNV-1a 32-bit. Closed in Phase 1 (C-4). | Not reopened. |
| **Iteration cap, tolerance, mesh, optimizer, filter, `V_f`, `p_K`, `pmass`, baseline, load and sensitivity model** | Fixed per `A4_SPECIFICATION_V3.md` §2.2. | Unchanged. |

### 2.3 The single-factor guarantee, restated operationally

After Phase 2, two arms `N = a` and `N = b` shall differ in exactly one observable respect: **the
set of iterations at which the reference eigenpair is replaced.** They shall be identical in the
iterations at which they are inspected, in the number of modes computed at each inspection (given
the same design), in the thresholds applied, and in the consequences of a failed inspection.

This is a testable statement, and §11 requires it to be tested rather than asserted.

---

## Section 3 — Adaptive mode-search protocol

This section replaces the fixed `nModes = 20` window of `A4_SPECIFICATION_V3.md` §4.1/§4.3.1. It
governs **every** screening event, diagnostic and operational alike, in every arm.

### 3.1 Determinism requirement

The protocol is a deterministic function of the current density field, the previously used
reference eigenvector, and the declared constants below. Two independent implementations,
presented with the same design and the same previous reference, shall compute the same window
sequence, select the same mode index, and emit the same admissibility decision. No tolerance,
threshold, ceiling, or tie-break may be left to implementer discretion. Where a tie is
numerically possible, §3.6 resolves it.

### 3.2 Declared constants

| Symbol | Value | Role |
|---|---|---|
| `m₀` | **20** | Initial search window (modes requested). Retained at 20 so that the completed run's screening decisions remain directly comparable to the new ones. |
| `W` | **(20, 40, 80, 160, 320)** | The window ladder. Expansion is by doubling; no intermediate values are permitted. |
| `M_max` | **320** | Maximum window. The ceiling is declared, not discovered. |
| `τ_MAC` | **0.80** | Minimum mass-weighted MAC to the previously used reference for a candidate to be admissible (continuity). |
| `τ_stab` | **0.99** | Minimum mass-weighted MAC between the modes selected at two consecutive rungs for the selection to be declared stable. |
| `τ_kin` | **0.50** | Minimum kinetic-energy fraction on the largest support-connected solid component. |
| `τ_strain` | **0.50** | Maximum strain-energy fraction in low-density elements. |
| `x_low` | **0.10** | Density below which an element is "low-density" for the strain-fraction measure. |

**Justification of `M_max = 320`.** The audit located the physical mode at index 49 (iteration 25)
and index 37 (iteration 30), and verified by direct re-execution that extending the window from 60
to 120 changes nothing — index 49 / 37 is the true location. A ceiling of 320 is more than six
times the largest observed requirement and four rungs above the previously fatal value. It is
chosen to be generous enough that reaching it constitutes evidence about the *design's spectrum*
rather than about the apparatus. If any Phase 2 arm reaches `M_max` at any event, that fact is a
first-class reportable finding (§7.3) and is **not** grounds for silently raising the ceiling
mid-campaign (§11, R-4).

### 3.3 Admissibility

A candidate eigenvector `φ` from the current window is **admissible** iff all four hold:

1. kinetic-energy fraction on the largest support-connected solid component `≥ τ_kin`;
2. that dominant component touches **both** supports;
3. strain-energy fraction in elements with `x < x_low` is `≤ τ_strain`;
4. mass-weighted MAC to the **previously used reference eigenvector** `≥ τ_MAC`.

Conditions 1–3 are the support-connectivity screen of `A4_SPECIFICATION_V3.md` §4.3.1, unchanged.
Condition 4 is continuity. For the first screening event of an arm, the "previously used
reference" is `Φ₀` on the solid domain.

The four conditions are evaluated independently and **all four outcomes are recorded for every
candidate**, even after the first failure. Short-circuit evaluation is prohibited: the rejection
reason must state *every* condition a candidate failed, because distinguishing "no
support-connected mode exists" from "a support-connected mode exists but has drifted below MAC
continuity" is precisely the discrimination the completed run could not make (§7).

### 3.4 Search and stopping rule

At a screening event, let `A(m)` denote the admissible set within a window of `m` modes and
`j*(m) = argmax` mass-weighted MAC to the previous reference over `A(m)`.

1. Solve for the lowest `m = m₀` modes. Apply the screen to all `m` candidates.
2. **If `A(m) = ∅`:** if `m < M_max`, advance to the next rung of `W` and repeat step 1. If
   `m = M_max`, terminate the search with outcome **REFERENCE_UNAVAILABLE** (§7.3).
3. **If `A(m) ≠ ∅` and `m = M_max`:** terminate with outcome **SELECTED**, selecting `j*(M_max)`,
   and set the stability flag to `unconfirmed` (§7.5).
4. **If `A(m) ≠ ∅` and `m < M_max`:** advance to the next rung `m′` and solve again. Compare the
   eigenvector selected at `m` with the one selected at `m′` by mass-weighted MAC:
   - **`MAC(φ_{j*(m)}, φ_{j*(m′)}) ≥ τ_stab`** → terminate with outcome **SELECTED** and stability
     flag `confirmed`.
   - **otherwise** → the selection has not stabilized; treat `m′` as the current rung and return
     to step 2. (Expansion can only add higher modes, so a changed selection means a better-matching
     candidate was previously outside the window — the exact condition that invalidated the
     completed run.)

**A confirmation expansion is mandatory.** Every successful screening event solves at least two
rungs. This is deliberate: a mode found near the edge of a window is not evidence that the window
was wide enough, and the completed run's failure was in every case a window-edge effect.

### 3.5 Reported values come from the widest window solved

All quantities reported for a screening event — the selected mode's frequency and index, its MAC
values, its energy fractions, the full candidate table, and the spectrum — are taken from the
**final (widest) eigensolve performed at that event**, never from an earlier rung.

This makes every reported number independent of the expansion path taken to reach it, which is
required for the determinism guarantee of §3.1 and for replay (§11, V-P2-6). Mode indices are
therefore always indices into the `m_final` spectrum and must be reported alongside `m_final`; an
index without its window is meaningless.

### 3.6 Tie-breaking

If two admissible candidates have mass-weighted MAC to the previous reference equal to within
`10⁻¹²`, select the one with the **lower mode index** in the `m_final` spectrum. Record that a tie
occurred. Ties are not expected; an occurrence is a signal worth reading, not a defect.

### 3.7 Failure conditions

| Condition | Outcome | Consequence |
|---|---|---|
| Eigensolver fails to converge at any rung | **SOLVER_FAILURE** | The event, the arm, and the run are classified as implementation failure (§7.2). The arm is halted, but all accumulated telemetry is harvested and persisted before halting (§4.5). |
| `NaN` or `Inf` in any assembled matrix, eigenvalue, eigenvector, or screen quantity | **SOLVER_FAILURE** | As above. |
| `M_max` reached with `A = ∅`, solver healthy at every rung | **REFERENCE_UNAVAILABLE** | Not a failure. A measured property of the design's spectrum (§7.3). Operational consequence in §5.4. |
| `M_max` reached with `A ≠ ∅` but no stable selection | **SELECTED (unconfirmed)** | Not a failure. Selection proceeds at `M_max`; the arm carries a warning (§8.3). |
| The selected mode fails any admissibility condition | **INADMISSIBLE_SELECTION** | Impossible by construction. If it is ever emitted, it is an implementation failure (§7.4), never a scientific class. |

### 3.8 Cost

Screening at wide rungs is expensive, and the diagnostic schedule of §4 applies it to all five
arms. Per `A4_SPECIFICATION_V3.md` §4.5, wall-clock time is recorded for provenance only and may
not appear in any table, figure, or claim. Cost is explicitly **not** a permitted reason to narrow
the ladder, thin the grid, or lower the ceiling once Phase 2 has begun (§11, R-4). If the campaign
budget cannot absorb the protocol, that is a scheduling decision to be taken **before** execution
and recorded as an amendment to this document, not a runtime adaptation.

---

## Section 4 — Common diagnostic schedule

This section discharges C-2. Its governing principle: **every arm is observed identically, and
observation never touches the optimization.**

### 4.1 The grid

Diagnostic screening is performed at each iteration in the fixed set

> **G = {1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000}**

intersected with the arm's realized iteration range, **plus the arm's final iteration** whenever
that iteration is not already in `G`.

`G` is identical for all five arms, including `N = ∞`. It is not a function of `N`, of the arm's
convergence behaviour, or of any observed quantity.

**Justification of the grid.** It is dense in the regime the audit identified as decisive — the
completed arms died at iterations 2, 25 and 30, and the frozen trajectory fails the 20-mode screen
at 2, 20, 25 and 30 — and it thins toward the cap where the design is nearly stationary. It
subsumes the checkpoints of Gate A4-Pre `{100, 300, 600, 2000}`, which the audit showed sample only
the regime where the answer is always PASS (M-6). Gate A4-Pre shall use `G` restricted to the
pre-flight trajectory rather than its former four points.

### 4.2 Recorded quantities per grid point

At every grid point, for every arm, the following are recorded:

| Quantity | Notes |
|---|---|
| Full adaptive-search outcome | Window rungs solved, `m_final`, search outcome, stability flag (§3) |
| Full candidate table | Every candidate in the `m_final` spectrum, per §6 |
| Selected mode | Index (in `m_final`), `ω`, MAC to previous reference, MAC to `Φ₀` |
| `ω₁ᵐⁱⁿ` | Lowest eigenvalue of the current design, whatever mode it is |
| `ω₁ᵗʳᵃᶜᵏ − ω₁ᵐⁱⁿ` | The spurious-mode gap on the intermediate design |
| `ω₁/ω₂` separation | Reviewer 2's coalescence observable |
| Number of disconnected solid components | Topological state of the current design |
| Low-density kinetic fraction | Retained so that its being `≈ 0` is documented, not assumed |
| `max|Δx_e|` | Also recorded every iteration, not only on `G` (§4.3) |
| `|V(x) − V_f|/V_f` | Feasibility |
| Surrogate objective | Diagnostic only; never compared across arms |
| Omitted-term ratio | Recorded if available; `null` if not measured — never a default (§7.6) |
| `min x_e` | The audit's re-execution showed this reaching `10⁻²⁷⁷`; it is diagnostic of the density field's collapse |

### 4.3 Per-iteration histories (not restricted to `G`)

The following are recorded at **every** iteration of **every** arm, because
`A4_SPECIFICATION_V3.md` §4.3 requires them and the completed run retained none of them (audit
M-4), making four of the seven specified figures unproducible:

`max|Δx_e|`, surrogate objective, feasibility, MAC of the currently used reference to `Φ₀`,
tracked index `j*` of the currently used reference, and the identity of the reference eigenpair in
force.

These are cheap: none requires an eigensolve.

### 4.4 Diagnostics shall not change optimization behaviour

This is a hard constraint, not a design preference.

1. Diagnostic screening is **read-only**. It may not modify the density field, the reference
   eigenpair, the filter state, the OC multiplier or its bracket, any accumulator, or any solver
   state.
2. Diagnostic screening may not consume randomness, and may not alter the state of any random or
   iterative-solver seed used by the optimization.
3. A diagnostic outcome — including `REFERENCE_UNAVAILABLE` — may not terminate, restart, or
   redirect an arm. Its only effects are on recorded telemetry.
4. Diagnostic eigensolves are performed on a copy of the design and their results are never
   written back into the optimization path.

**Validation is mandatory and is a Phase 2 stop condition.** Running any arm with the diagnostic
schedule enabled and disabled shall produce **bit-identical** design trajectories, endpoints, and
topologies. §9.2 makes the `N = ∞` arm the production-scale instance of this test.

### 4.5 Telemetry survives failure

If an arm halts for any reason — solver failure, exception, cap, or operator interrupt — all
telemetry accumulated to that point shall be persisted before the halt propagates. The completed
run's driver discarded everything the solver had accumulated inside its `catch` block, so three of
five arms recorded `iterations: 0`, `refresh_events: []`, and every endpoint `null` (audit M-5) —
destroying precisely the observations immediately preceding each failure, which for an experiment
whose stated purpose is to characterize failure are the primary evidence.

A halted arm shall record its true iteration count, its true refresh count, its complete event
log, its complete per-iteration histories, and its complete candidate telemetry. Recording
`iterations: 0` for an arm that executed 30 iterations is, in Phase 2, an implementation failure
(§7.4).

---

## Section 5 — Operational refresh

### 5.1 The distinction Phase 2 introduces

| | **Diagnostic screening** | **Operational refresh** |
|---|---|---|
| **When** | At every iteration in `G` (§4.1) | At iterations `i` with `i mod N = 0`; never for `N = ∞` |
| **Schedule depends on `N`?** | **No** | **Yes — this is the treatment** |
| **Effect on the optimization** | **None, ever** (§4.4) | Replaces the reference eigenpair used to build `f = ω₀²M(x)Φ₀` |
| **Effect on classification** | Contributes evidence; cannot alone determine an arm's class | Determines whether the arm's treatment was actually delivered |
| **Uses the §3 search protocol** | Yes, identically | Yes, identically |

An iteration may be both: `G` and the refresh schedule overlap by construction (e.g. iteration 50
for `N = 50`, every grid point for `N = 1`). When they coincide, **one** search is performed and
its result serves both purposes. The search is identical either way; only the consequence differs.
This is what makes the two schedules commensurable.

### 5.2 When a refresh changes the optimization — exhaustively

The optimization changes **if and only if** all of the following hold:

1. the arm has finite `N`; **and**
2. the current iteration `i` satisfies `i mod N = 0`; **and**
3. the §3 search at iteration `i` terminated with outcome **SELECTED**.

When all three hold, the selected eigenpair `(ω, φ)` from the `m_final` solve replaces the
reference used to construct the load, effective from iteration `i` onward, until the next
successful replacement. This is recorded as an **effective refresh**.

Under every other circumstance the optimization is unchanged. In particular:

- A diagnostic screening outcome never changes the optimization, whatever it is.
- A search outcome of `REFERENCE_UNAVAILABLE` at an operational refresh never changes the
  optimization (§5.4).
- The `N = ∞` arm has no operational refreshes; its reference is computed once on the solid domain
  and never replaced. The refresh code path shall be inert for `N = ∞`, and its inertness is
  verified bitwise (§9.2).

### 5.3 Refresh accounting

Two counts are recorded and reported separately, and must never be conflated:

- **Scheduled refreshes:** `⌊n_iter / N⌋` — attempts, a function of the treatment and the realized
  iteration count alone.
- **Effective refreshes:** the number of attempts that satisfied §5.2 condition 3 and actually
  replaced the reference.

The predicted eigensolve count of `A4_SPECIFICATION_V3.md` §4.3 / V-A4-3 (`1 + ⌊(n_iter−1)/N⌋`)
applies to **scheduled** refreshes plus initialization. The completed implementation checked
`⌊n_iter/N⌋`, a different formula (audit M-8); Phase 2 shall implement the specified formula and
report both counts so the discrepancy is visible rather than absorbed.

Note that eigensolve *count* and eigensolve *cost* now diverge: an event may solve up to five
rungs. Both are recorded; per `A4_SPECIFICATION_V3.md` §4.5 the analytic operation count is what
may be reported, and wall-clock is provenance only.

### 5.4 Deferral: what happens when an operational refresh finds no admissible reference

> **The arm continues. The previous reference eigenpair is retained. The event is recorded as a
> deferred refresh. The next attempt occurs at the next scheduled refresh iteration, not earlier.**

The arm is **not** terminated, **not** classified from this event alone, and **not** disqualified.

**Justification.** Retaining the previous reference is the minimal well-defined behaviour: it
leaves the optimization in a state the method already supports — running on a reference eigenpair
older than the current design, which is exactly what `N = ∞` does permanently. No new numerical
regime is introduced, no arbitrary substitute mode is invented, and the treatment degrades
gracefully toward the frozen arm rather than toward an undefined state. Terminating instead — the
completed run's behaviour — converts a property of the intermediate spectrum into an arm-level
scientific verdict, destroys all subsequent observation, and, as the audit demonstrated, produced a
published causal claim that its own artifacts contradicted.

**Deferral is a first-class response variable.** Per arm, Phase 2 records the number of deferred
refreshes, their iterations, the deferral fraction (deferred ÷ scheduled), and the longest run of
consecutive deferrals.

**Degenerate-arm condition.** If an arm defers **every** scheduled refresh, its reference eigenpair
was never replaced and it is numerically identical to `N = ∞`. Such an arm did not receive the
treatment its level names. It shall be reported as **degenerate**, its endpoint shall not be used
as an accuracy reference at that level, and the fact shall be stated explicitly in Table A4-1 and
in the manuscript text. A degenerate arm is a strong and reportable result about the method's
diagnosability at that `N` — it is the honest form of what the completed run tried to say with B3,
supported by evidence the completed run did not have.

### 5.5 Why this removes the verified causal flaw

The audit established C-2 as follows: because the screen ran only at refresh events, `N` controlled
both the treatment and the observation schedule; the screen fails on all early designs regardless
of refreshing (the frozen trajectory itself fails at iterations 2, 20, 25, 30); therefore `N` was
perfectly confounded with *how early the design is first screened*, and `N = 50` survived only
because its first refresh lands at iteration 50, one step past the window in which every arm fails.

Phase 2 breaks the confound in two independent places, and either alone would be insufficient:

1. **Observation is decoupled from treatment.** All arms are screened on `G`. Screening exposure is
   now a constant function of iteration, identical across arms. `N` no longer influences *when* the
   design is inspected — only *when the inspection's result is allowed to change the load*. The
   frozen arm is screened on the same grid as `N = 1`, so its early-iteration screen failures — if
   they recur under the adaptive window — become visible data about the formulation rather than an
   asymmetry between arms.
2. **Screening loses the power to terminate.** Under §5.4 a failed screen can no longer end an arm,
   so a property of the intermediate spectrum can no longer masquerade as an arm-level scientific
   verdict. Even if the adaptive window were still too narrow at some design, the consequence would
   be a recorded deferral in a completed arm, not three arms with `iterations: 0` and a false
   contamination diagnosis.

Together with C-1's correction, the specific mechanism that produced the false B3 verdict is
eliminated at three points: the window is no longer fixed (so the mode is found), the screen is no
longer `N`-dependent (so failures are not attributable to the treatment), and screen failure no
longer terminates (so failures do not destroy the evidence that would refute the diagnosis).

---

## Section 6 — Candidate telemetry

### 6.1 Requirement

For **every candidate mode**, in the `m_final` spectrum, at **every screening event**, of **every
arm** — diagnostic and operational alike — the following record is stored. There is no sampling,
no truncation to the admissible set, and no summarization.

| Field | Definition |
|---|---|
| `arm_N` | Arm level (`"inf"`, `50`, `10`, `5`, `1`) |
| `iteration` | Iteration index of the screening event |
| `event_kind` | `diagnostic` \| `operational` \| `both` |
| `event_id` | Unique, monotone within the arm; joins this row to the event record of §6.2 |
| `window_m_final` | Width of the eigensolve from which this candidate is drawn |
| `mode_index` | Index in the `m_final` spectrum, ascending by eigenvalue, 1-based |
| `omega` | Frequency (rad/s) of this candidate |
| `mac_prev` | Mass-weighted MAC to the previously used reference eigenvector |
| `mac_phi0` | Mass-weighted MAC to the original solid-domain `Φ₀` |
| `mac_solid` | Mass-weighted MAC to the solid-domain fundamental mode (equal to `mac_phi0` by definition of the baseline; recorded explicitly so the two are never inferred from each other) |
| `support_kinetic_fraction` | Kinetic-energy fraction on the largest support-connected solid component |
| `low_density_strain_fraction` | Strain-energy fraction in elements with `x < x_low` |
| `low_density_kinetic_fraction` | Retained for documentation (§4.2) |
| `support_connectivity` | Whether the dominant component touches both supports (boolean) |
| `cond_kinetic_pass`, `cond_supports_pass`, `cond_strain_pass`, `cond_mac_pass` | The four admissibility conditions of §3.3, each evaluated independently — **all four recorded regardless of earlier failures** |
| `rejection_reason` | Empty if admissible; otherwise the complete list of failed conditions with the measured value and the threshold for each |
| `admissible` | Final admissibility decision (boolean) |
| `selected` | Whether this candidate was the selected reference (boolean) |
| `tie_flag` | Whether this candidate tied within `10⁻¹²` with the selected one (§3.6) |
| `eigensolver_status` | Convergence status and residual for this eigenpair |

### 6.2 Per-event record

Alongside the candidate rows, one record per screening event:

`arm_N`, `iteration`, `event_kind`, `event_id`, the ordered list of window rungs solved,
`m_final`, `search_outcome` (`SELECTED` \| `REFERENCE_UNAVAILABLE` \| `SOLVER_FAILURE`),
`stability_flag` (`confirmed` \| `unconfirmed` \| `n/a`), `n_candidates`, `n_admissible`,
`selected_index`, `selected_omega`, `omega_min`, `omega1_omega2_gap`, `n_solid_components`,
`reference_changed` (boolean — the §5.2 test), `deferred` (boolean), `eigensolve_count_at_event`,
and the wall-clock cost of the event (provenance only).

### 6.3 Reconstructability standard

The telemetry shall be sufficient for a reviewer, working from the artifacts alone, to answer for
any screening event: *which modes were available, what each one's screen quantities were, which
conditions each one failed and by how much, which was chosen, whether the choice was confirmed by
window expansion, and whether the choice changed the optimization.*

If any of those questions requires re-execution to answer, the telemetry is non-compliant. This
standard exists because the audit had to re-run the frozen trajectory in MATLAB to discover that a
MAC-0.98 admissible mode sat at index 49 — a fact the run's own artifacts should have contained.

### 6.4 Volume

The grid has 25 points, the ladder tops out at 320 modes, and there are five arms, bounding the
candidate table at roughly 10⁵ rows for a full sweep. This is a modest CSV and is not a reason to
sample. Storage cost is never a permitted justification for reducing telemetry (§11, R-4).

---

## Section 7 — Failure taxonomy

### 7.1 What was wrong with B3

`A4_SPECIFICATION_V3.md` §5.2 defines **B3 — spurious-mode contamination** with the interpretation
*"the refresh locked onto a disconnected-island mode."* The completed run emitted B3 for three
arms. The audit falsified this on the run's own evidence: all three arms record `Solid components =
1` in their own exception messages, and a single connected solid body cannot produce a
disconnected-island mode. Two distinct problems are present:

1. **The label asserted a mechanism that was never measured.** B3 as written is a causal claim
   about *which mode the refresh selected*. In the completed run **no mode was selected at all** —
   the admissible set was empty. The condition that fired ("no admissible candidate") and the
   condition B3 describes ("an inadmissible candidate was chosen") are different events, and the
   implementation used one code for both.
2. **The condition that did fire was an artifact of the window.** An admissible mode existed at
   both failure points and passed the full screen; it lay outside the 20-mode window.

### 7.2 Terminology decision

> **B3 is retired as a breakdown mechanism. It is not renamed; it is decomposed.**

"Reference unavailable" is the right *phrase* for the condition that actually fired, but adopting it
as a rename of B3 would be insufficient, because B3 in the completed implementation absorbed at
least three distinguishable conditions. A rename would preserve the conflation under a more honest
name. Phase 2 therefore replaces the single code with five mutually exclusive classes, each with
necessary and sufficient conditions.

A second, deliberate change of kind: **"reference unavailable" is an observability state, not an
approximation-breakdown mechanism.** The B-codes of `A4_SPECIFICATION_V3.md` §5.2 assert things
about *the frozen-eigenpair approximation*. Whether a wide enough eigensolve can locate the
physical mode on an intermediate design asserts something about *the spectrum of the formulation*.
Filing the latter under the former is the category error that produced the published claim. The new
classes are therefore recorded in their own namespace (`E-*`, for *event class*) and mapped to
arm-level outcomes explicitly in §8.

### 7.3 The five classes

Conditions are stated as necessary **and** sufficient. The classes are mutually exclusive and, over
the domain of screening events, exhaustive.

---

**E-0 — CLEAN SELECTION**

*Necessary and sufficient:* the §3 search terminated with outcome `SELECTED`; the selected
candidate satisfies all four admissibility conditions of §3.3; the stability flag is `confirmed`.

*Interpretation:* the screening event succeeded. No finding.

---

**E-1 — WINDOW TRUNCATION**

*Necessary and sufficient:* the §3 search terminated with outcome `SELECTED`, **and** the selected
mode index exceeds `m₀ = 20`, i.e. the mode would not have been found under the completed run's
fixed window.

*Interpretation:* **This is the audit's C-1 finding, converted into a measurement.** It is not a
failure and carries no penalty. It is the primary new observable of Phase 2 and quantifies how many
genuine void modes descend below the structural mode on intermediate designs — a real property of a
formulation with `E_min/E₀ = ρ_min/ρ₀ = 10⁻⁹`, under which void has the same wave speed as solid.

*Required reporting:* every E-1 event is reported with its iteration, its selected index, and
`m_final`. The maximum index over all events, per arm, is a headline quantity of Table A4-2.

*Note:* E-1 events are expected in abundance around iterations 25–30 based on the audit's direct
measurements (index 49 and 37). Their absence would itself be a finding requiring explanation.

---

**E-2 — REFERENCE UNAVAILABLE**

*Necessary and sufficient, all required:*
1. the window was expanded to `M_max = 320`;
2. the eigensolver converged successfully at every rung, with no `NaN`/`Inf` anywhere;
3. the admissible set is empty at `M_max`;
4. the rejection reasons for all `M_max` candidates are recorded per §6.

*Interpretation:* on this design, no mode within the declared ceiling is simultaneously
support-connected, non-void-dominated, and continuous with the previous reference. This is a
measured property of the intermediate design's spectrum. It is **not** a claim that a spurious mode
was selected, and it is **not** a claim about the frozen approximation.

*Sub-classification is mandatory* and follows from the recorded per-condition outcomes:
- **E-2a — no connected candidate:** no candidate passes conditions 1–3. The design's spectrum
  offers no physical mode at all within the ceiling.
- **E-2b — no continuous candidate:** at least one candidate passes conditions 1–3 but none passes
  condition 4 (MAC continuity `≥ τ_MAC`). A physical mode exists but has drifted away from the
  previous reference — which is a statement about **mode evolution**, potentially bearing on H₁,
  and must not be conflated with E-2a.

This E-2a/E-2b split is the discrimination the completed implementation could not make, and is why
§3.3 forbids short-circuit evaluation.

*Operational consequence:* deferral, per §5.4. Never termination.

---

**E-3 — INADMISSIBLE SELECTION**

*Necessary and sufficient:* a reference eigenpair was adopted (`reference_changed = true`) whose
selected candidate fails at least one admissibility condition of §3.3.

*Interpretation:* **this is the condition B3 was written to describe, and it is impossible under
the Phase 2 protocol**, which selects only from the admissible set. Its emission therefore
indicates that the implementation departed from this specification.

*Consequence:* implementation failure (E-4). Phase 2 is halted and the defect fixed. E-3 is
retained in the taxonomy specifically as a tripwire, so that the condition has a name and a
detector even though it should never fire.

---

**E-4 — DISCONNECTED-MODE CONTAMINATION**

*Necessary and sufficient, both required:*
1. the current design has `n_solid_components ≥ 2` (a topological fact about the design); **and**
2. the highest-`mac_prev` candidate in the `m_final` spectrum has support-component kinetic
   fraction `< τ_kin` (a modal fact about the spectrum).

*Interpretation:* the design genuinely carries disconnected solid material and the best-matching
mode genuinely lives on it. **This is the mechanism the completed run named but never
demonstrated.** Both a topological and a modal criterion are required precisely because the
completed run asserted the mechanism while its own artifacts recorded `Solid components = 1`.

*Relationship to E-2:* E-4 may co-occur with an E-2 outcome, in which case both are recorded — E-2
describes what the search returned, E-4 describes why. E-4 is the only class that may be cited in
the manuscript as evidence of spurious-mode contamination, and it may be cited **only** where both
conditions are recorded in the artifacts.

*Note:* the completed `N = 1` arm recorded `Solid components = 4` at iteration 2, so E-4's first
condition may well be met there. Its second condition was never measured. Phase 2 will measure it.

---

**E-5 — IMPLEMENTATION FAILURE**

*Necessary and sufficient:* any one of —
- eigensolver non-convergence at any rung, or `NaN`/`Inf` in any assembled matrix, eigenvalue,
  eigenvector, or screen quantity;
- an E-3 event;
- a diagnostic-enabled run differing in any bit from the corresponding diagnostic-disabled run
  (§4.4);
- a halted arm whose persisted telemetry does not match its true execution (e.g. `iterations: 0`
  for an arm that executed iterations) (§4.5);
- factor drift: an arm's configuration differing from the base in any respect other than `N`,
  detected by the corrected hash (Phase 1, C-4);
- non-deterministic replay (§11, V-P2-6);
- a required artifact absent.

*Interpretation:* the machinery is broken; the run says nothing. Fix and re-run. Maps to
`A4_SPECIFICATION_V3.md` §5.2 **Class A — REJECTED**, unchanged.

### 7.4 Separation summary

| Question | Class |
|---|---|
| Was the mode outside the old fixed window? | **E-1** |
| Was no admissible mode found within the declared ceiling? | **E-2** (a: none connected; b: none continuous) |
| Was an inadmissible mode actually adopted? | **E-3** → implementation failure |
| Is there disconnected solid material carrying the best-matching mode? | **E-4** |
| Did the apparatus break? | **E-5** |

The completed run mapped the first, second and fourth of these onto a single code, B3, and
published the fourth's interpretation for the first's occurrence.

### 7.5 Warning conditions (not classes)

Recorded per event, orthogonal to the class:

- `stability_flag = unconfirmed` — selection made at `M_max` without confirmation (§3.4 step 3).
- `m_final = M_max` at any event — the ceiling was reached. Whether or not a mode was found, this
  means the ceiling is potentially binding and must be reported (§8.3).
- `tie_flag` — a tie was broken by index (§3.6).

### 7.6 Out of scope, explicitly

The following audit findings bear on classification but are **not** Phase 2 work, and Phase 2 shall
neither implement nor pre-empt them. They are named here so that the Phase 2 implementation does
not collide with them and so that their absence is not mistaken for an oversight:

| Finding | Status |
|---|---|
| **M-1** — the Class B criterion `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵗʰʳᵉˢʰ` is not implemented | Deferred. Phase 1's C-3 fix made this criterion meaningful (the corrected `ω₁ᵗʰʳᵉˢʰ` values are 162.47 / 162.48 against tracked 159.57 / 159.60, i.e. within 1.8%); implementing it is Phase 3. Phase 2 shall record `ω₁ᵗʰʳᵉˢʰ` and its MAC for every completed arm so that Phase 3 has the data. |
| **M-2** — `N = ∞` labelled B4 without B4 evidence | Deferred. B4 requires a limit cycle **and** an omitted-term ratio above threshold; the arm records neither. Phase 2 does not alter B4 or introduce a replacement code. |
| **M-3** — `limit_cycle` is a hard-coded default | Partially enabled, not resolved. Phase 2's §4.3 per-iteration `max|Δx_e|` history makes a limit-cycle detector *possible*; specifying and implementing the detector is Phase 3. Phase 2 shall emit `limit_cycle: null` (not measured) rather than `false`, since a default presented as data is what the audit objected to. |
| **M-7** — `INDETERMINATE` is not a pre-registered outcome | Deferred. It arises from M-2. Phase 2 does not resolve it and **shall not** emit a final scientific decision (§9.5). |
| **M-9** — no baseline `ω₁⁽⁰⁾` recorded | Deferred to Phase 3 (validator V-A4-1). |

**Consequence, stated plainly:** because M-2 is unresolved, the `N = ∞` arm will again classify as
`ACCEPTED_WITH_BREAKDOWN/B4` and the campaign decision will again be `INDETERMINATE`. Phase 2's
deliverable is a **corrected measurement**, not a final scientific verdict. Phase 2 shall not be
declared a failure on this ground, and shall not "fix" it by expanding scope.

---

## Section 8 — Acceptance rules

These rules classify **arms** and the **Phase 2 run as a whole**. They are stated so that no
outcome requires a judgement call.

### 8.1 An arm is REJECTED

*If and only if* at least one E-5 condition (§7.3) is recorded for that arm.

A rejected arm carries no scientific content. The defect is fixed and the arm re-run. Rejection is
never triggered by mode loss, deferral, iteration cap, `MAC < 0.8`, `j* ≠ 1`, `E-1`, `E-2`, or
`E-4` — every one of those is a measurement (`A4_SPECIFICATION_V3.md` §4.6, unchanged).

### 8.2 An arm is ACCEPTED

*If and only if* all of the following hold:

1. no E-5 condition is recorded;
2. the arm ran to a terminal state — convergence (`max|Δx_e| < 10⁻³`) or the iteration cap — with
   its true iteration count recorded;
3. every scheduled refresh was attempted, and every attempt is recorded with its full event and
   candidate telemetry (§6);
4. every diagnostic grid point within the arm's realized range, plus the final iteration, was
   screened and recorded;
5. per-iteration histories (§4.3) are complete and of length equal to the realized iteration count;
6. the primary endpoint `ω₁ᵗʳᵃᶜᵏ` and all companion measures (`ω₁ᵐⁱⁿ`, `ω₁ᵗʰʳᵉˢʰ`,
   `ω₁/ω₂` gap, `j*`, MAC to `Φ₀`, grayness, feasibility) exist and are finite;
7. no deferred refreshes occurred (deferral count `= 0`);
8. no warning condition of §8.3 applies.

Acceptance is a statement about **measurement integrity only**. It does not assert that the arm
converged, that its mode was retained, or that it supports any hypothesis. Those are the
`A4_SPECIFICATION_V3.md` §5.2 B-classes, which are applied on top of this and are unchanged.

### 8.3 An arm is ACCEPTED WITH WARNING

*If and only if* §8.2 conditions 1–6 hold **and** at least one of the following:

| Warning | Condition | Reporting requirement |
|---|---|---|
| **W-1 — deferral** | deferral count `≥ 1` and `<` scheduled count | Report deferral count, deferral fraction, iterations, and longest consecutive run. The arm's endpoint remains usable as an accuracy reference, but every use must be annotated with its deferral fraction. |
| **W-2 — ceiling reached** | `m_final = M_max` at any event | Report the events. The ceiling may be binding; the arm's screening decisions are conditional on `M_max = 320`. |
| **W-3 — unconfirmed selection** | `stability_flag = unconfirmed` at any event | Report the events. The selected mode was not confirmed by window expansion. |
| **W-4 — deep truncation** | any E-1 event with selected index `> 160` | Report. The physical mode sat in the top half of the ladder; the ceiling's adequacy should be reconsidered for future work (never mid-campaign). |
| **W-5 — contamination observed** | any E-4 event | Report with both conditions' measured values. |

An arm with warnings is a **valid scientific observation** and is reported in full. Warnings are
never grounds for exclusion; they are grounds for annotation. Every table and figure presenting a
warned arm shall carry its warning codes.

### 8.4 An arm is UNAVAILABLE

*If and only if* §8.2 conditions 1–6 hold **and** the arm is **degenerate** per §5.4 — every
scheduled refresh deferred, so the reference eigenpair was never replaced.

An UNAVAILABLE arm did not receive its nominal treatment. Its endpoint is numerically that of
`N = ∞` and **shall not** be reported as `ω₁ᵗʳᵃᶜᵏ(N)` for that level, shall not enter the H₀/H₁
comparison at that level, and shall not appear in Figure 1 as a data point. It **shall** be reported
explicitly — in Table A4-1, in the text, and in the figure's annotation — as *"refresh reference
unavailable at every scheduled attempt; arm degenerates to the frozen method."*

This is the pre-registered third outcome of `A4_SPECIFICATION_V3.md` §1.2/§5.3 (outcome 3) reaching
the arm level, now supported by per-event evidence rather than by a terminated run.

### 8.5 The Phase 2 run as a whole

| Verdict | Condition |
|---|---|
| **COMPLETE** | All five arms are ACCEPTED or ACCEPTED WITH WARNING or UNAVAILABLE; the `N = ∞` bit-identity gate (§9.2) passed; every item of §11 is satisfied. |
| **INCOMPLETE** | Any arm is REJECTED, or any §11 item is unsatisfied. Phase 2 is not complete and shall not be reported as such. |
| **HALTED** | The `N = ∞` bit-identity gate failed. This is a stop condition: it means the diagnostic instrumentation perturbed the optimization, invalidating every arm. No Phase 2 result may be reported until it is resolved. |

There is no "partially complete" verdict and no discretionary override.

---

## Section 9 — Rerun plan

### 9.1 Summary

| Arm | Action | Existing result | Rationale |
|---|---|---|---|
| `N = ∞` | **Re-execute; scientific content preserved** | Preserved and load-bearing | §9.2 |
| `N = 50` | **Re-execute; result superseded** | Superseded | §9.3 |
| `N = 10` | **Re-execute** | Void (no data) | §9.4 |
| `N = 5` | **Re-execute** | Void (no data) | §9.4 |
| `N = 1` | **Re-execute** | Void (no data) | §9.4 |

Five arms, one sweep, one base configuration, no new levels, no new benchmarks, no changed
constants. This is the minimum: three arms produced no data at all, and the remaining two lack
every per-iteration history the specification requires and were screened under the defective
window.

### 9.2 `N = ∞` — is it preserved?

> **Yes. The scientific content of the `N = ∞` arm is preserved and remains authoritative. The arm
> is nevertheless re-executed, and re-execution must reproduce it bit-identically.**

These are not in tension, and the distinction is load-bearing.

**Why the result is preserved.** The `N = ∞` arm never refreshes. Under Phase 2 its diagnostic
screening is read-only and provably non-perturbing (§4.4), and the refresh code path is inert. No
Phase 2 change — adaptive window, common grid, deferral policy, telemetry — can lawfully alter its
trajectory. Its endpoint has moreover been independently verified: the audit reproduced
`ω₁ᵗʳᵃᶜᵏ = 159.565627` to seven significant figures from a from-scratch Q4 plane-stress model
driven only by the published topology CSV, and Phase 1 confirmed the tracked frequency and topology
are bitwise unchanged, with topology CSV SHA-256 `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`.

**Why it must nonetheless be re-executed.** The arm has no per-iteration histories, no MAC history,
no `j*` history, no `max|Δx_e|` history, and no screening record of any kind (audit M-4) — because
it was never screened during its run at all. `A4_SPECIFICATION_V3.md` §4.3 requires all of these
for every arm, four of the seven specified figures cannot be produced without them, and the C-2
correction is *defined* as screening `N = ∞` on the same grid as every other arm. None of this can
be obtained post hoc; the data were never retained.

**The re-execution is simultaneously the strongest available validator.** The re-run shall
reproduce the preserved endpoint, topology, and trajectory **bit-identically**:

- `ω₁ᵗʳᵃᶜᵏ = 159.56562699328325` exactly;
- topology CSV SHA-256 unchanged;
- final `max|Δx_e| = 3.034903639330122e-03`, iterations `2000`, `j* = 1`,
  `MAC = 0.9996284251363903`, `ω₁/ω₂` gap `67.37267502573462`, all exactly.

A bit-identical reproduction discharges, at production scale and in a single run, three things the
completed campaign never evidenced: **V-A4-5** (determinism replay at 400×50 — audit M-8 records no
such artifact), **V-A4-6** (refresh-path inertness, previously evidenced only on a 40×5 mesh with
≤6 iterations, which `A4_IMPLEMENTATION_REPORT.md` §5 itself declares "plumbing only and never
evidence"), and the §4.4 non-perturbation requirement.

**Any deviation is a Phase 2 stop condition** (§8.5, HALTED). A mismatch means the diagnostic
instrumentation changed the optimization, which invalidates every arm in the sweep, not only this
one. It must be diagnosed and eliminated before any Phase 2 result is reported. The corrected
provenance hash `fnv1a32_c141e407` (Phase 1) makes factor drift detectable as a cause; before
Phase 1 that check passed unconditionally.

### 9.3 `N = 50` — re-execute; existing result superseded

Necessary for two independent reasons, either sufficient:

1. **Its operational refreshes were performed under the defective window.** All ten refreshes ran
   with `nModes = 20`. Under §3, each event expands to at least 40 modes and selects the
   highest-`mac_prev` admissible candidate over the wider set. The first refresh already shows the
   risk concretely: at iteration 50 the run selected **index 5** with `mac_prev = 0.9995764` out of
   **one** admissible candidate — the selection was forced, not competitive, and a candidate with
   higher MAC continuity may exist above index 20. If any refresh selects differently, the entire
   downstream trajectory differs and the existing endpoint is not the endpoint of the specified
   protocol. The existing result is therefore not merely under-instrumented; it is potentially the
   result of a different procedure.
2. **It has no per-iteration histories** (audit M-4), like `N = ∞`, and cannot supply Figures 2, 3,
   4 and 7.

The existing `N = 50` result — `ω₁ᵗʳᵃᶜᵏ = 159.6011729491971`, `+0.022%` against the frozen arm,
227× inside `δ` — is retained in the artifacts as the **pre-Phase-2 reference value** and reported
alongside the new one. If the Phase 2 re-run reproduces it, that is meaningful corroboration: it
would show the widened window did not change the selections, and would strengthen rather than
replace the one clean result the completed run owns. If it does not reproduce, the difference is
itself a headline finding about the window's influence on the trajectory. Both outcomes are
informative and both must be reported; neither may be suppressed.

### 9.4 `N = 10`, `N = 5`, `N = 1` — re-execute

These arms produced **no data**: `iterations: 0`, `n_refresh: 0`, `refresh_events: []`, every
endpoint `null`. They were terminated at iterations 30, 25 and 2 respectively, on a diagnosis the
audit falsified by direct measurement — a support-connected, MAC-0.98 admissible Φ₁-type mode
existed at index 49 (iteration 25) and index 37 (iteration 30), outside the 20-mode window. There
is nothing to preserve and no post hoc correction is possible.

They carry the entire scientific question. `N*`, the breakdown threshold the experiment exists to
locate, cannot lie at `N = ∞` or `N = 50` alone; the graded sweep is the design
(`A4_SPECIFICATION_V3.md` §2.1, "why five levels and not two"). Without these three arms A4 is a
two-point design and cannot distinguish "freezing is harmless" from "refreshing is unstable" — the
two opposite conclusions that a two-point design conflates.

`N = 1` deserves specific attention: it ran for 1.27 seconds against a 2.5-hour budget, so nothing
whatsoever is known about the fully design-dependent limit, which `A4_SPECIFICATION_V3.md` §6.2
identifies as the most fragile regime and where the CR2 interaction is expected to be strongest.
It is also the arm where E-4 (disconnected-mode contamination) has a real prospect of being
genuinely observed: it recorded `Solid components = 4` at iteration 2, satisfying E-4's first
condition. Its second condition has never been measured.

### 9.5 What Phase 2 shall not do

- **No new arms, levels, benchmarks, or `α` values.** The clamped beam, the building, `α ≠ 1`, and
  an adversarial `Φ₀` case (audit PR-13, PR-14) are out of scope.
- **No change to `δ`, to the hypotheses, or to the four pre-registered decision outcomes.**
- **No final scientific decision.** Phase 2 emits corrected measurements, per-arm classes, and
  Table A4-1/A4-2. The campaign-level H₀/H₁ decision remains blocked on the out-of-scope items of
  §7.6 — principally M-2, which will again classify `N = ∞` as B4. Phase 2 shall state this
  limitation explicitly in its report rather than emitting a decision the evidence does not yet
  support.
- **No manuscript claim.** Phase 2 prepares the evidence; §10.7 lists the manuscript edits it
  *enables*, which are executed separately and only after Phase 2 is COMPLETE.

### 9.6 Budget

Non-normative. The completed sweep took 526 s, but three arms died within seconds and no arm
performed diagnostic screening. Phase 2 adds up to 25 screening events per arm, each solving at
least two ladder rungs and potentially reaching 320 modes on ~41,000 DOF, and runs three arms to
completion for the first time. A wall-clock increase of one to two orders of magnitude should be
anticipated. Per §3.8 and §11 R-4, budget pressure is not a permitted reason to alter the protocol
once execution begins; the registry's runtime estimate shall be updated **before** the sweep (the
completed run's `estRuntimeSeconds = 57600` against 526 s actual, 110× off, would have masked a
silently truncated run — audit m-10).

---

## Section 10 — Deliverables

Every artifact below is mandatory. Missing artifacts are an E-5 implementation failure (§7.3).

### 10.1 Reports

| Artifact | Content |
|---|---|
| `A4_RECOVERY_PHASE2_REPORT.md` | Execution record: what was run, corrections implemented, per-arm classes and warnings, the `N = ∞` bit-identity verification with the compared values, the E-1 window-requirement findings, the E-2/E-4 findings, the §11 checklist with each item marked and evidenced, and an explicit statement of what remains blocked (§7.6, §9.5). |
| `A4_RECOVERY_PHASE2_VALIDATION.md` | Validator results (§11 V-P2-1…V-P2-8), each with the assertion, the measured values, and pass/fail. |

### 10.2 JSON

| Artifact | Content |
|---|---|
| `output/a4/a4_result.json` | Per-arm result to the `A4_SPECIFICATION_V3.md` §7.5 schema, extended with: `deferrals[]`, `n_refresh_scheduled`, `n_refresh_effective`, `max_window_used`, `max_selected_index`, `warnings[]`, `event_classes{}`, `degenerate` (boolean). `N = ∞` serializes as the string `"inf"`, never as `null` (audit m-8). `limit_cycle` serializes as `null` (§7.6, M-3). |
| `output/a4/a4_screening_events.json` | Every screening event record of §6.2, all arms. |
| `output/a4/a4_manifest.json`, `a4_stage_result.json`, `a4_stage_manifest.json` | Campaign provenance. The two manifests shall list an identical artifact set (audit m-5). |
| `output/a4/a4_pre_screen.json` | Gate A4-Pre on the grid `G` (§4.1), reporting the window required at each checkpoint. |

### 10.3 MAT

| Artifact | Content |
|---|---|
| `output/a4/a4_eigenpair_refresh_results.mat` | Complete result structure including full per-iteration histories (§4.3) and final topologies for all five arms. `N = ∞` stored as `Inf`. |

### 10.4 CSV

| Artifact | Content |
|---|---|
| `output/a4/a4_candidate_telemetry.csv` | Long format, one row per candidate per event per arm, exactly the fields of §6.1. **The primary reproducibility instrument.** |
| `output/a4/a4_iteration_histories.csv` | Long format, one row per iteration per arm, the quantities of §4.3. |
| `output/a4/a4_topology_{inf,50,10,5,1}.csv` | Final density fields, five files. |

### 10.5 Plots

The seven figures of `A4_SPECIFICATION_V3.md` §7.6, all of which are now producible, plus two
Phase-2-specific figures:

1. `ω₁ᵗʳᵃᶜᵏ` vs `N`, **with the `±δ = 5%` equivalence band drawn and the y-limits set to contain
   it** (audit m-1: the completed figure auto-scaled to a 0.025% y-range with no band, rendering a
   0.022% difference at opposite corners — communicating the opposite of the finding). Arms that
   are UNAVAILABLE (§8.4) are annotated, not plotted as data points.
2. MAC vs iteration, per arm, refresh events marked, deferrals distinctly marked.
3. `max|Δx_e|` vs iteration, log scale.
4. Tracked index `j*` vs iteration, five panels.
5. Spectrum and screen metrics at each event.
6. Final topologies, five panels.
7. `ω₁/ω₂` separation vs iteration.
8. **New — required window `m_final` vs iteration**, all five arms overlaid, with `m₀ = 20` and
   `M_max = 320` marked as horizontal reference lines. *This figure is the visual statement of the
   C-1 finding and of Phase 2's central correction.*
9. **New — selected mode index vs iteration**, all five arms overlaid, with `m₀ = 20` marked.
   *Shows directly how far outside the old fixed window the physical mode sat, and when.*

All text labels shall render with `'Interpreter','none'` so that class names containing
underscores are not TeX-subscripted (audit m-2).

### 10.6 Tables

| Artifact | Content |
|---|---|
| **Table A4-1** (`output/a4/a4_table.md`) | One row per `N`: `ω₁ᵗʳᵃᶜᵏ`, `ω₁ᵐⁱⁿ`, `ω₁ᵗʰʳᵉˢʰ`, MAC to `Φ₀`, `j*`, iterations, converged, scheduled/effective refreshes, grayness, feasibility, omitted-term ratio, class, warnings, and `Δω₁` vs `N = ∞`. The last column is populated **only** for arms eligible as accuracy references and is left **explicitly blank** — not zero, not a dash — otherwise. |
| **Table A4-2** (new) | Screening summary per arm: number of events, number of E-0/E-1/E-2a/E-2b/E-4 events, maximum `m_final`, maximum selected index, deferral count and fraction, longest consecutive deferral run, number of events reaching `M_max`, number of unconfirmed selections. *This table is the Phase 2 evidence base.* |

### 10.7 Manuscript updates enabled (not executed by Phase 2)

Phase 2 produces the evidence; the edits are made separately and only after verdict COMPLETE:

- `main.tex:704` — the directional claim that refresh beats frozen. Phase 2 supplies the corrected
  finite-`N` endpoints against which it must be evidenced, softened, or retracted.
- `main.tex:661` — the cross-class robustness claim. Must be narrowed regardless of Phase 2's
  outcome; A4 covers one class, as `A4_SPECIFICATION_V3.md` Part 8 concedes.
- `main.tex:665` — quotes SS-beam MAC `0.9998`; A4 measures `0.99963`. Reconcile the number
  (audit m-7).
- A new limitations sentence recording the **window requirement** finding: that on intermediate
  designs of this formulation, dozens of genuine void modes descend below the structural mode, so a
  fixed low mode count is unsafe for mode tracking. This is a real contribution, and it exists only
  because the defect was found.

### 10.8 Supplementary material

- `A4_RECOVERY_PHASE2_SPECIFICATION.md` — this document, as the protocol of record.
- `a4_candidate_telemetry.csv` and `a4_screening_events.json` — the complete screening record.
- The determinism replay record for `N = ∞` (§9.2).
- The diagnostics-on / diagnostics-off bit-identity record (§4.4).

### 10.9 Version control

**All** artifacts shall be git-tracked, including the `.mat`, the PNGs, and the topology CSVs. In
the completed run only the six JSON/MD files were tracked, so the `.mat` — a *required artifact*
per `a4_stage_result.json` — could not be recovered from the repository (audit m-4). Every artifact
shall additionally record a commit SHA that is reachable from the working state at the time of
writing (audit m-9).

---

## Section 11 — Acceptance checklist

Phase 2 is COMPLETE if and only if every item below is satisfied and evidenced in
`A4_RECOVERY_PHASE2_REPORT.md`. Unevidenced items count as unsatisfied.

### Scientific

- [ ] **S-1** The scientific question, `δ = 5%`, the hypotheses, the level set, and the four
  pre-registered decision outcomes are unchanged from `A4_SPECIFICATION_V3.md`.
- [ ] **S-2** `N` is the only treatment; every item of §2.2 is fixed, demoted, or made common, and
  each is evidenced.
- [ ] **S-3** The mode-search window is reported as a response variable in Table A4-2 and Figure 8,
  never as a configured constant.
- [ ] **S-4** Screening exposure is identical across arms: every arm was screened at
  `G ∩ [1, n_iter]` plus its final iteration, and this is demonstrated by counting recorded events
  per arm.
- [ ] **S-5** No B3 label is emitted anywhere. Every screening event carries an E-class per §7.3.
- [ ] **S-6** Every E-4 (contamination) claim, if any, is supported by both its topological and its
  modal condition, with measured values in the artifacts.
- [ ] **S-7** Every E-2 event is sub-classified E-2a or E-2b.
- [ ] **S-8** No E-3 event was emitted.
- [ ] **S-9** Out-of-scope items (§7.6) were not implemented, and their consequences — including
  that the campaign decision remains blocked — are stated explicitly in the report.

### Implementation

- [ ] **I-1** The window ladder, ceiling, thresholds, and grid are declared in configuration or in
  a single named constants block, not scattered as literals, and their values match §3.2 and §4.1
  exactly.
- [ ] **I-2** The `m₀`-then-confirm search of §3.4 is implemented, including mandatory confirmation
  expansion and the stability test.
- [ ] **I-3** All reported quantities derive from the widest window solved (§3.5).
- [ ] **I-4** Admissibility evaluates all four conditions independently, without short-circuiting
  (§3.3).
- [ ] **I-5** Diagnostic screening is read-only and cannot terminate an arm (§4.4).
- [ ] **I-6** Operational refresh changes the optimization exactly under §5.2's three conditions,
  and deferral follows §5.4.
- [ ] **I-7** Telemetry is persisted before any halt propagates (§4.5).
- [ ] **I-8** Scheduled and effective refresh counts are recorded separately; the V-A4-3 formula is
  `1 + ⌊(n_iter−1)/N⌋` as specified, not `⌊n_iter/N⌋`.
- [ ] **I-9** `limit_cycle` emits `null`, not `false`; `N = ∞` serializes as `"inf"`, not `null`.

### Validation

- [ ] **V-P2-1 — Non-perturbation.** At least one arm run with diagnostics enabled and disabled
  yields bit-identical trajectory, endpoint, and topology.
- [ ] **V-P2-2 — `N = ∞` bit-identity.** The re-executed frozen arm reproduces the preserved
  endpoint, topology SHA-256, and final design change exactly (§9.2). *Failure is a stop
  condition.*
- [ ] **V-P2-3 — Window recovery.** At iterations 25 and 30 of the frozen trajectory, the protocol
  finds an admissible Φ₁-type mode at index 49 and index 37 respectively, with MAC ≈ 0.978 and
  ≈ 0.966 — reproducing the audit's independent measurements. *This is the direct test that C-1 is
  fixed.*
- [ ] **V-P2-4 — Screening symmetry.** For a fixed design presented to the screening routine, the
  window sequence, selected index, and admissibility decisions are independent of which arm
  presented it.
- [ ] **V-P2-5 — Ladder determinism.** Repeated screening of the same design yields identical
  window sequences and selections.
- [ ] **V-P2-6 — Determinism replay.** Full replay of at least one finite-`N` arm reproduces its
  result exactly, and the record is written into the provenance block (`A4_SPECIFICATION_V3.md`
  §7.7).
- [ ] **V-P2-7 — Classifier fixtures.** Synthetic fixtures exercise each of E-0…E-5 and E-2a/E-2b,
  including a fixture where an admissible mode sits above index 20 (asserting E-1, **not** E-2) and
  a fixture reproducing the completed run's iteration-25 state (asserting E-1, **not** B3).
- [ ] **V-P2-8 — Factor drift.** All five arms share the corrected base-config hash
  `fnv1a32_c141e407`, and a negative test confirms two different configurations produce different
  hashes.
- [ ] **V-P2-9** Phase 1's regressions (`test_a4_phase1`, 10/10) still pass.

### Reproducibility

- [ ] **R-1** `a4_candidate_telemetry.csv` permits reconstruction of every screening decision
  without re-execution (§6.3), demonstrated by reconstructing at least three events by hand in the
  report — including one E-1 and one E-2 event if any occurred.
- [ ] **R-2** Every artifact of §10 exists, is git-tracked, and is listed identically in both
  manifests.
- [ ] **R-3** Every artifact records a reachable commit SHA and the corrected config hash.
- [ ] **R-4** No protocol constant — `m₀`, `W`, `M_max`, any threshold, the grid `G`, or the
  telemetry set — was altered during execution. Any change requires a written amendment to this
  document and a full re-run of all five arms.
- [ ] **R-5** The registry runtime estimate was updated before the sweep and is within one order of
  magnitude of actual.

### Documentation

- [ ] **D-1** `A4_RECOVERY_PHASE2_REPORT.md` and `A4_RECOVERY_PHASE2_VALIDATION.md` exist and are
  complete.
- [ ] **D-2** `A4_SPECIFICATION_V3.md` carries a pointer to this document at §4.1, §4.3.1, §5.2 and
  §6.1, marking those provisions as amended.
- [ ] **D-3** Table A4-1 and Table A4-2 are issued, with the `Δω₁` column blank for ineligible arms.
- [ ] **D-4** All nine figures exist, Figure 1 carries the `δ` band with y-limits containing it, and
  all labels render with `'Interpreter','none'`.
- [ ] **D-5** The `N = 50` pre-Phase-2 and Phase-2 endpoints are both reported, with the difference
  (or its absence) stated explicitly (§9.3).
- [ ] **D-6** The report states plainly which audit findings remain open (§7.6) and that the
  campaign-level scientific decision is not among Phase 2's deliverables.

---

## Closing note on scope discipline

The completed A4 run failed not because its machinery broke — the audit verified the FE model, the
eigensolve, the determinism, the cross-file consistency and the refresh telemetry as sound — but
because an implementation constant was permitted to determine a scientific verdict, and because the
apparatus that would have exposed this had been scheduled by the independent variable itself.

Phase 2 corrects exactly those two things. It is deliberately narrow. The temptation during
implementation will be to fix the adjacent findings that are now within reach — the B4 mislabel, the
limit-cycle detector, the `ω₁ᵗʰʳᵉˢʰ` Class B criterion, the missing baseline frequency. Each is
real, each is listed in §7.6, and each is Phase 3. Expanding scope mid-recovery is how the original
defects entered: a single specification carrying four simultaneous changes is the failure mode the
campaign has already survived twice.

Phase 2's deliverable is a corrected measurement of an unchanged question, with enough recorded
evidence that no future reviewer need re-run anything to audit it.
