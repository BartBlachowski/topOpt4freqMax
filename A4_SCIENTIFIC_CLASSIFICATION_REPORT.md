# A4 Scientific Classification Report

**Status:** scientific classification only. Post-production. No source code, MATLAB,
production data, threshold, constant, acceptance rule, factor definition, or hypothesis
was modified in the production of this report.

**Scope:** resolves the deferred scientific items **M-1, M-2, M-3, M-7, M-9**, the open
**B4** attribution, and validator **V-A4-1**, strictly from the completed and validated
Phase 5 production artifacts. It then evaluates whether the pre-registered A4 decision
rule can be applied.

**Relation to Phase 5 validation.** The Phase 5 production campaign is **COMPLETE** and
its measurement-integrity evidence (V-P2-1 … V-P2-9, Section 11 checklist) is treated
here as **fixed input**, not re-derived. This report is completely separated from
production validation: it consumes validated measurements and classifies them; it does
not re-establish that they are valid.

---

## Category legend (mandatory throughout)

Every substantive statement is tagged with the epistemic category it belongs to:

| Tag | Category | Meaning |
|---|---|---|
| **[M]** | Measurement | A value or fact read directly from a validated production artifact. |
| **[C]** | Classification | Application of a pre-registered rule/threshold to measurements. |
| **[I]** | Interpretation | A reasoned reading of measurements that goes beyond the pre-registered rules. Explicitly not a decision. |
| **[S]** | Scientific conclusion | A conclusion admissible into the scientific record under the decision protocol. |

Where a conclusion cannot be uniquely supported, the report states **"reference
unavailable"** or **"insufficient evidence"** rather than strengthening the claim.

---

## Section 0 — Pre-registration (declared before any pass/fail judgement)

These declarations precede every evaluation below. **No value here is new.** Each is
carried verbatim from an authoritative input; the citation is given so that no threshold
can be suspected of having been chosen after inspecting the results.

### 0.1 Baseline computation

- **[M]** The reference eigenpair `(ω₀, Φ₀)` is the fundamental mode of the **fully solid
  domain** `x_e = 1 ∀e`, `K(1)Φ = ω²M(1)Φ`, mass-normalized (A4_SPECIFICATION_V3 §3.2,
  A0-F1). The solver hard-errors on any other `semi_harmonic_baseline`
  (A4_SPECIFICATION_V3 §3.2, item 2), and the base config declares
  `semi_harmonic_baseline: "solid"` ([a4_ss_400x50_base.json:64](examples/Revision_v1/a4_ss_400x50_base.json#L64)).
- **[M]** The solid-domain baseline frequency is recorded in the production artifacts as
  `reference_omega = 136.483085251141`, `reference_identity = solid_phi0`, at iteration 1
  of every arm ([a4_iteration_histories.csv](examples/Revision_v1/output/a4/a4_iteration_histories.csv), columns 8–9).
  **No separate eigensolve was required**, and none was performed: the Baseline protocol's
  precondition ("if V-A4-1 requires `ω₁⁽⁰⁾` **and it is absent** from the production
  artifacts") is **not** triggered.

### 0.2 Numerical tolerances and comparison thresholds (all pre-existing)

| Symbol | Value | Source | Use |
|---|---|---|---|
| `δ` | 5% | A4_SPECIFICATION_V3 §1.1; frozen by A4_RECOVERY_PHASE2_SPECIFICATION §1.1 | Equivalence margin on `ω₁ᵗʳᵃᶜᵏ` |
| `τ_MAC` | 0.8 | V3 §4.3.1 / §5.2 Class B | Mode-retention & continuity |
| `τ_kin` | 0.5 | V3 §4.3.1 (support-component kinetic fraction ≥ 0.5) | Physical-mode admissibility |
| low-density strain fraction | ≤ 0.5 at `x < 0.1` | Phase 2 §2.2 | Physical-mode admissibility |
| convergence | `max\|Δx_e\| < 10⁻³` **before** cap | V3 §5.2 Class B | Class B eligibility |
| `m₀` | 20 | Phase 2 §7.3 (E-1) | Old fixed window; E-1 reference |
| `M_max` | 320 | Phase 2 §7.3 (E-2) | Adaptive-ladder ceiling |
| threshold void floor | `ρ_min = 10⁻⁹` | Phase 1 (C-3), config | `ω₁ᵗʰʳᵉˢʰ` computation |

**[M]** The definition of `Δω₁ᵗʳᵃᶜᵏ(N)` is the **signed** ratio
`[ω₁ᵗʳᵃᶜᵏ(N) − ω₁ᵗʳᵃᶜᵏ(∞)] / ω₁ᵗʳᵃᶜᵏ(∞)` (A4_RECOVERY_PHASE2_SPECIFICATION §1.1).

**Explicit non-declaration.** The M-1 criterion `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵗʰʳᵉˢʰ` has **no
pre-registered numerical tolerance** in any authoritative input. This report therefore
does **not** convert it into a binary pass/fail (doing so would require introducing a
threshold, which is forbidden). It reports the measured separation and its MAC and states
the scientific question it answers; a binding boundary is a Phase-3 act.

### 0.3 Classification boundaries (pre-existing taxonomy)

- **[M]** Arm measurement-integrity classes: ACCEPTED / ACCEPTED_WITH_WARNING /
  UNAVAILABLE / REJECTED (Phase 2 §8).
- **[M]** Scientific classes: Class A (REJECTED), Class B (clean), Class C (accepted with
  approximation breakdown) with mechanisms **B1** (mode migration, `MAC ≥ 0.8` but
  `j* ≠ 1`), **B2** (frozen-mode breakdown, `MAC < 0.8`), **B4** (sensitivity-omission
  instability = limit cycle **and** omitted-term ratio above threshold)
  (V3 §5.2). **B3 is retired and decomposed** into event classes E-1…E-5 (Phase 2 §7);
  per the "Deprecated outcomes" directive, no B3 classification is reused.
- **[M]** Pre-registered decision outcomes 1–4 (V3 §5.3), reproduced in Section 9.

---

## Section 1 — Fixed input from Phase 5 (not re-derived)

The following are taken as settled (evidence: A4_RECOVERY_PHASE5_PRODUCTION_REPORT.md,
A4_RECOVERY_PHASE5_VALIDATION.md, [a4_result.json](examples/Revision_v1/output/a4/a4_result.json),
[a4_table.md](examples/Revision_v1/output/a4/a4_table.md),
[a4_table2.md](examples/Revision_v1/output/a4/a4_table2.md)):

| N | `ω₁ᵗʳᵃᶜᵏ` [M] | `Δ` vs ∞ [C] | `ω₁ᵐⁱⁿ` [M] | `ω₁ᵗʰʳᵉˢʰ` [M] | `j*` [M] | MAC→Φ₀ [M] | iters/cap [M] | conv [M] | class [C] | integrity [C] |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---|---|
| ∞ | 159.56562699 | +0.0000% | 159.5656 | 162.4677 | 1 | 0.99963 | 2000/2000 | no | C / **B4** | ACC-WARN (W-2,W-5) |
| 50 | 159.60129670 | +0.0224% | 159.6013 | 162.4788 | 1 | 0.99963 | 540/2000 | yes | **B** | ACC-WARN (W-2,W-5) |
| 10 | 159.12290675 | −0.2775% | 98.5535 | 162.9134 | 7 | 0.99477 | 536/2000 | yes | C / **B1** | ACC-WARN (W-1,W-2,W-5) |
| 5 | 158.67273611 | −0.5596% | 158.6727 | 162.7406 | 1 | 0.99935 | 1173/2000 | yes | **B** | ACC-WARN (W-1,W-2,W-5) |
| 1 | 157.63288447 | −1.2113% | 157.6329 | 161.4558 | 1 | 0.99975 | 1040/2000 | yes | **B** | ACC-WARN (W-1,W-2,W-5) |

- **[M]** Reference (frozen) endpoint `ω₁ᵗʳᵃᶜᵏ(∞) = 159.56562699328325`, verified
  bit-identical (V-P2-1, V-P2-2; topology SHA-256 `9c3d961b…`).
- **[M]** `decision.outcome = "NOT_EMITTED_PHASE2"`; `delta = 0.05`
  ([a4_result.json](examples/Revision_v1/output/a4/a4_result.json)).
- **[C]** Every finite arm lies **within** `δ`, and **no finite arm exceeds** the frozen
  arm (the single positive deviation, N=50, is +0.0224%). The largest deviation of any
  kind is N=1 at −1.21%, i.e. **≈ 4× inside `δ`**.

---

## Section 2 — M-1: the `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵗʰʳᵉˢʰ` Class-B criterion

**Deferred question (V3 §5.2 Class B; audit M-1).** Is any arm's reported frequency a
**gray-material artifact** — i.e. does volume-preserving thresholding collapse the tracked
frequency, as the pre-recovery Table A4-1 falsely implied (`ω₁ᵗʰʳᵉˢʰ = 26.5`, audit C-3)?

**Available evidence [M]** (from [a4_result.json](examples/Revision_v1/output/a4/a4_result.json),
[a4_table.md](examples/Revision_v1/output/a4/a4_table.md)):

| N | `ω₁ᵗʳᵃᶜᵏ` | `ω₁ᵗʰʳᵉˢʰ` | `(ω₁ᵗʰʳᵉˢʰ−ω₁ᵗʳᵃᶜᵏ)/ω₁ᵗʳᵃᶜᵏ` | MAC(Φ_thresh→Φ₀) |
|---|---:|---:|---:|---:|
| ∞ | 159.5656 | 162.4677 | +1.82% | 0.99972 |
| 50 | 159.6013 | 162.4788 | +1.80% | 0.99973 |
| 10 | 159.1229 | 162.9134 | +2.38% | 0.99970 |
| 5 | 158.6727 | 162.7406 | +2.56% | 0.99921 |
| 1 | 157.6329 | 161.4558 | +2.43% | 0.99981 |

**[C]** For every arm the thresholded frequency lies within **+1.8% to +2.6%** of the
tracked frequency and is **higher**, not lower — the physical direction for
volume-preserving thresholding, which removes intermediate density and stiffens the
design. In every arm `MAC(Φ_thresh → Φ₀) ≥ 0.99921`, so the thresholded design carries
**the same physical mode**, not a spurious void mode.

**[S] Resolution — RESOLVED, high confidence.** **No arm's frequency is a gray-material
artifact.** The C-3 defect (the undeclared `10⁻³` void floor that manufactured
`ω₁ᵗʰʳᵉˢʰ = 26.5`, MAC 0.0002) is confirmed absent in production: with the declared
`ρ_min = 10⁻⁹`, `ω₁ᵗʰʳᵉˢʰ ≈ ω₁ᵗʳᵃᶜᵏ` and the mode is preserved.

**[I]** The audit's "two defects cancelled" hazard (M-1 unimplemented + C-3 artifact would
have disqualified both surviving arms) **no longer applies**: because C-3 is fixed, a
future Phase-3 implementation of the M-1 criterion on the corrected `ω₁ᵗʰʳᵉˢʰ` would
**retain** N=50 (and all arms) as passing, not disqualify them.

**Evidence missing / remaining uncertainty.** The `≈` in the criterion has **no
pre-registered tolerance** (§0.2). Binding M-1 as a binary Class-B gate is therefore a
Phase-3 act requiring a declared boundary; it is out of scope here. The **scientific
question** ("is any result a gray-material fiction?") is nonetheless answered **no** with
high confidence, since all separations are < 2.6% and all thresholded MACs ≥ 0.999 under
the only pre-registered margin available (`δ = 5%`), which every arm satisfies.

---

## Section 3 — M-2: the B4 label on the N=∞ arm

**Deferred question (audit M-2).** Is the N=∞ arm's **B4** breakdown code
(sensitivity-omission instability) supported by evidence?

**Measurement vs interpretation, kept strictly separate:**

- **[M]** N=∞ reached the iteration cap: `iterations = 2000 = cap`,
  `final_design_change = 3.0349×10⁻³` (> tolerance `10⁻³`)
  ([a4_result.json](examples/Revision_v1/output/a4/a4_result.json)). This is a
  **measured fact**: the frozen method did **not** converge within 2000 iterations.
- **[M]** The two conditions B4 is *defined* by (V3 §5.2) are **both unmeasured**:
  `limit_cycle = null` and `omitted_term_ratio = null` in the arm record, and no declared
  B4 threshold exists in any artifact.
- **[M]** The stored `class_reason` itself concedes the gap: *"B4 (unattributed):
  iteration cap reached; M-2 remains open and Phase 2 makes no campaign-level decision."*
- **[C]** Under the pre-registered B4 definition (limit cycle **AND** omitted-term ratio
  above threshold), B4 **cannot be asserted**: neither conjunct is measured.
- **[I]** The "B4" string in the production record is a **catch-all label for "capped with
  no other signature,"** exactly the misuse the audit identified — not a measurement of a
  sensitivity-omission instability.

**[S] Resolution — RESOLVED as "label unsupported."** The N=∞ arm is, on the evidence, a
**capped, non-converged run whose breakdown mechanism is unattributed**. The specific
claim encoded by the B4 code — that the non-convergence is caused by the omitted load
sensitivity — is **not supported by any production measurement** and must not enter the
record as such. The audit's recommended distinct code (e.g. "B0 — unattributed
non-convergence") is the correct home for this arm, but introducing it is a taxonomy
change reserved for Phase 3 and is **not** performed here.

**Remaining uncertainty.** Whether the true mechanism *is* B4 is **insufficient evidence**
(see Section 7).

---

## Section 4 — M-3: `limit_cycle` as measurement, not default

**Deferred question (audit M-3).** `limit_cycle` was a hard-coded default. Is there now a
basis to value it — and does the N=∞ non-convergence exhibit a limit cycle?

**Available evidence:**

- **[M]** `limit_cycle = null` for every arm
  ([a4_result.json](examples/Revision_v1/output/a4/a4_result.json)). Phase 2 correctly
  emits **null = "not measured"** rather than a default `false` (Phase 2 §7.6, M-3 row).
- **[M]** The per-iteration `max|Δx_e|` history now **exists** for every arm
  ([a4_iteration_histories.csv](examples/Revision_v1/output/a4/a4_iteration_histories.csv);
  [a4_checkpoint_inf.mat.history.jsonl](examples/Revision_v1/output/a4/a4_checkpoint_inf.mat.history.jsonl)),
  which is the raw material a detector would consume. This is the part of M-3 that Phase 2
  **enabled**.
- **[M] Observed behaviour of the N=∞ tail** (last 200 iterations, 1801–2000):
  `max|Δx_e|` stays in the band **[1.199×10⁻³, 4.098×10⁻³]**, mean 2.52×10⁻³, with
  **0 of 200** iterations below the `10⁻³` tolerance; the surrogate objective is
  **stationary at ≈ 7337** from iteration ~1200 onward (7336.9 → 7337.4).
- **[M]** The consecutive-step signs over the tail are **irregular**, not a clean period-2
  alternation (e.g. deltas +,−,+,+,−,+,+,+,+,−,−,+,−,+,+,−,+ over iters 1983–2000).

**[C]** Under the B4 definition, the "bounded limit cycle (period-2 signature)" conjunct
**cannot be confirmed**: no detector was specified or run, and the raw signature is not
cleanly period-2.

**[I]** The observed behaviour is a **bounded, non-decaying oscillation of the design
update above tolerance, with a stationary objective** — the design keeps churning at a
scale ~1% of the move limit (0.2) without improving the objective. Multiple mechanisms
remain consistent with this (see Section 7); the raw record does not, by itself, single
one out.

**[S] Resolution — the value is correctly "not measured"; formal limit-cycle
classification is INSUFFICIENT EVIDENCE.** `limit_cycle: null` is now honest data, not a
default. Phase 2 made a detector *possible* by retaining the Δx history, but the detector
itself is Phase-3 work and was not run; the raw history shows bounded non-convergence but
**not** an unambiguous period-2 limit cycle. No limit-cycle classification is emitted.

---

## Section 5 — M-7: `INDETERMINATE` is not a pre-registered outcome

**Deferred question (audit M-7).** The decision rule pre-registers exactly four outcomes
(V3 §5.3). Does the observed measurement space map onto one of them — in particular, is
there a defined outcome for the case where the **reference arm N=∞ is itself Class C**?

**[M]** N=∞ is classified Class C (not Class B): it is capped and carries a breakdown code
(Section 3). N=10 is Class C/B1 (mode migration, `j* = 7`). N=50, N=5, N=1 are Class B.

**[C]** Outcome 1 (H₀ retained) requires **all five arms Class B**. That antecedent is
**false** because the reference arm (N=∞) and one finite arm (N=10) are Class C. This is
evaluated in full in Section 9.

**[S] Resolution — CONFIRMED still open; the gap M-7 named persists.** The pre-registered
rule has **no outcome** for "equivalence satisfied for all finite arms, but the reference
arm is itself Class C." Emitting `INDETERMINATE` (or any fifth outcome) to cover it would
be exactly the post-hoc retrofitting §5.3 exists to forbid. M-7 is therefore **not
resolvable from the measurements**; it is a property of the *rule*, not of the data.
Resolving it requires either (a) making the frozen arm converge, or (b) amending §5.3 in
advance of a future run to define the reference-arm-Class-C case — **both out of scope**.

> Note on the task's framing of M-7 as a "limit-cycle / behavioural" item: the
> behavioural (limit-cycle) content is measured and classified in Sections 3 and 7; the
> **decision-completeness** content of M-7 (the audit's definition) is the one above and
> is the binding one for the decision protocol.

---

## Section 6 — M-9: the baseline reference

**Deferred question (audit M-9).** No baseline `ω₁⁽⁰⁾` was recorded, so no gain ratio
`ω̃₁/ω₁⁽⁰⁾` was computable. Is the baseline reference now **sufficient, insufficient, or
ambiguous** for the intended scientific comparison?

**Available evidence [M]:**

- The solid-domain baseline is recorded: `ω₁⁽⁰⁾(solid) = 136.483085251141`
  (`reference_identity = solid_phi0`),
  [a4_iteration_histories.csv](examples/Revision_v1/output/a4/a4_iteration_histories.csv).
  This is the ω₀ that builds the frozen load `f = ω₀²M(x)Φ₀` and is identical across all
  five arms at iteration 1. It matches the audit's independent solid-baseline value
  (136.483) to the recorded precision.
- The intended H₀/H₁ comparison is **arm-to-arm** on `ω₁ᵗʳᵃᶜᵏ(N)` vs `ω₁ᵗʳᵃᶜᵏ(∞)`
  (§1.1); it does **not** require `ω₁⁽⁰⁾`.

**[C]** Gain ratios against the declared solid baseline are now computable, e.g.
N=50: `159.601 / 136.483 = 1.169×`; N=∞: `1.169×` (all arms `1.155×`–`1.169×`).

**[I]** The residual ambiguity the audit flagged (Reviewer 2's C2) is the **choice of
baseline**: solid (136.483 → gain 1.169×) versus uniform `x = V_f = 0.5`
(68.24 → gain 2.339×) — a factor-of-two spread. A4 **declares** the solid baseline
(V3 §3.2) and proves (V3 §3.3, confirmed in Section 8) that this choice does **not** change
the design or the true `ω₁`; it changes only the *reported* `ω₁⁽⁰⁾` and hence the gain
ratio. V3 §3.4 explicitly places the cross-benchmark baseline-choice fix (the building's
gains) **outside A4's scope**.

**[S] Resolution — SUFFICIENT for A4's SS-beam scope; the cross-benchmark ambiguity is
out of scope, not a defect.** For the comparison A4 exists to make (arm-to-arm equivalence
on `ω₁ᵗʳᵃᶜᵏ`), the baseline is not even required, and the H₀/H₁ inputs are complete. For
the manuscript's *gain ratio*, the baseline is now recorded and unambiguous **given the
declared solid reference**. The remaining solid-vs-uniform ambiguity is Reviewer 2's C2,
which A4's specification assigns to another workstream; it is neither resolved nor
worsened here.

---

## Section 7 — B4 attribution assessment

**Open issue.** Does the completed evidence **resolve**, **narrow**, or **leave
unresolved** the B4 attribution for the N=∞ arm?

**What B4 requires (V3 §5.2):** (i) a bounded limit cycle (period-2 `Δx` signature)
**and** (ii) an omitted-term ratio exceeding its declared threshold.

**Evidence status:**

- **[M]** Condition (i): the Δx history exists and shows bounded, non-decaying oscillation
  (Section 4) — **but not a clean period-2 signature**, and no detector was run.
- **[M]** Condition (ii): `omitted_term_ratio = null` in every arm and every screening
  event ([a4_result.json](examples/Revision_v1/output/a4/a4_result.json),
  [a4_screening_events.json](examples/Revision_v1/output/a4/a4_screening_events.json));
  the covariate was **never exported** (audit PR-12, out of scope), and **no declared B4
  threshold exists**.

**[C] Verdict — B4 is NARROWED but UNRESOLVED.**

- **Narrowed:** the completed evidence is sufficient to state that the **current B4 label
  is unsupported** (Section 3), and that the *necessary* limit-cycle conjunct is at best
  weakly and ambiguously consistent with the data (bounded oscillation, no clean period-2).
- **Unresolved:** the **second mandatory conjunct (omitted-term ratio) was never
  measured**, so B4 can be neither confirmed nor refuted. This is **insufficient
  evidence**, by construction, and cannot be repaired without generating new data
  (forbidden here).

**[I] Explanations consistent with the observed bounded non-convergence** (reported
explicitly, none excluded by the evidence):

1. **Sensitivity-omission instability (B4 proper):** the discarded `∂f/∂x` drives a
   sustained oscillation. *Untestable as run* — `omitted_term_ratio` is null.
2. **Benign OC "chatter":** persistent small design churn near a filtered optimum with a
   flat objective — a common non-convergence of OC on fine meshes that says nothing about
   freezing. Consistent with the stationary objective and sub-move-limit Δx.
3. **Near-degenerate objective landscape:** many near-equivalent designs, so `max|Δx_e|`
   floors above tolerance while the objective is stationary.

**[S]** No single mechanism can be selected. B4 stays open; the only defensible present
statement is *"the frozen arm did not converge within 2000 iterations for an unattributed
reason."*

---

## Section 8 — V-A4-1: baseline invariance

**What V-A4-1 tests (V3 §7.8):** `MAC(Φ₀ᵘⁿⁱᶠᵒʳᵐ, Φ₀ˢᵒˡⁱᵈ) ≥ 0.9999`; the `ω₀²` ratio
matches the predicted `a/b` scalar; and the N=∞ design is invariant to the baseline
choice.

**What the validated production evidence supports:**

- **[M]** The solid baseline was the only one executed: the solver enforces
  `semi_harmonic_baseline = "solid"` and the uniform baseline is **unrunnable by design**
  (V3 §3.2, item 2). `ω₀(solid) = 136.483085251141` is recorded (Section 6).
- **[C] `ω₀²` ratio.** For a spatially uniform field `x_e = 0.5` with no passive elements,
  the SIMP stiffness scale is `a(0.5) = 10⁻⁹ + 0.5³(1−10⁻⁹) = 0.125` and the linear mass
  scale (`pmass = 1`) is `b(0.5) = 10⁻⁹ + 0.5(1−10⁻⁹) = 0.5`, so
  `ω₀²(uniform)/ω₀²(solid) = a/b = 0.25` exactly, i.e. `ω₀(uniform) = 0.5 × 136.483 =
  68.24`. This **matches** the predicted scalar (V3 §3.3) and the audit's independent
  uniform value (68.24). ✔
- **[C] MAC.** By the analytic identity of V3 §3.3 (`K(c) = a·K̂`, `M(c) = b·M̂` ⇒ the
  eigenvectors are those of the solid domain), `Φ₀ᵘⁿⁱᶠᵒʳᵐ = Φ₀ˢᵒˡⁱᵈ` **exactly**, hence
  `MAC = 1.0000 ≥ 0.9999`. ✔ (analytic; the uniform run needed to *measure* it directly is
  precluded by the solver guard).
- **[M] Design invariance.** The N=∞ topology reproduced **bit-identically** across the
  clean replay (V-P2-1/V-P2-2, SHA-256 `9c3d961b…`). This confirms determinism and
  refresh-path inertness (V-A4-5/V-A4-6) but is **not** a direct dual-baseline test.

**[S] Resolution — CONFIRMED analytically and consistent with all recorded evidence; the
direct dual-baseline empirical leg was not executed (by design).** The invariance claim
holds: mode shapes identical (MAC = 1), `ω₀²` ratio = `a/b` = 0.25 confirmed against the
recorded `ω₀`. **Scientific implication:** the solid-vs-uniform baseline choice is
immaterial to A4's design and true `ω₁`, and consequential only for the *reported*
`ω₁⁽⁰⁾`/gain — precisely the M-9 point. The one empirical leg V-A4-1 nominally asks for
(run both baselines, compare designs) is **not present in the artifacts** because the
uniform baseline cannot be run; that specific direct measurement is *insufficient
evidence*, while the invariance conclusion itself is supported by the exact analytic
identity plus the recorded `ω₀` and bit-identical design.

---

## Section 9 — Decision-rule completeness, then the decision protocol

### 9.1 Screening methodology used (not deprecated B3)

**[M]** No production event carries a B3 label (A4_RECOVERY_PHASE5_PRODUCTION_REPORT.md).
Classification here uses the corrected E-class taxonomy. Two facts from that taxonomy bear
on the decision:

- **E-1 (window truncation):** at N=∞, admissible Φ₁-type modes were found at **index 49
  (iter 25, MAC 0.9775)** and **index 37 (iter 30, MAC 0.9664)** — outside the old 20-mode
  window (V-P2-3). The former "B3 contamination" was window truncation. **[C]**
- **E-4 (disconnected-mode contamination):** fires at **iterations 1, 2, 3 in all five
  arms including N=∞**, identically (`n_solid_components = 4`, best-mode support-kinetic
  fraction 0) ([a4_screening_events.json](examples/Revision_v1/output/a4/a4_screening_events.json)).
  **[C]** Because it is present in the frozen arm too, the early-design spectral pathology
  is a property of **early gray designs, not of refreshing** — confirming the C-2 confound
  is now measured symmetrically and vindicating the retirement of B3.

### 9.2 Completeness check of the pre-registered rule over the observed space

The observed decision-relevant state is:

- **Equivalence [C]:** `|Δω₁ᵗʳᵃᶜᵏ(N)| ≤ δ` for **every** finite N (max 1.21%), and **no**
  finite arm exceeds N=∞ (max positive deviation +0.0224%).
- **Class membership [C]:** {∞: C/B4\*, 50: B, 10: C/B1, 5: B, 1: B}, where B4\* is the
  unsupported label of Section 3.

Mapping onto the four pre-registered outcomes (V3 §5.3):

| # | Outcome | Antecedent | Holds? | Why |
|---|---|---|:--:|---|
| 1 | H₀ retained | all five arms Class B **and** `Δ ≤ δ` ∀ finite N | **NO** | `Δ ≤ δ` holds, but N=∞ and N=10 are **Class C**, so "all Class B" fails |
| 2 | H₁ supported | some clean (B / C-B1 / C-B2) arm exceeds N=∞ by `> δ` | **NO** | no finite arm exceeds N=∞ at all |
| 3 | reference unavailable | N=∞ exceeds every finite arm by `> δ` **and** those are C-B3 / C-B4 | **NO** | N=∞ exceeds none by `> δ` (max 1.21%); finite arms are not B3/B4 |
| 4 | refreshing hurts | N=∞ exceeds every finite arm by `> δ`, those clean | **NO** | N=∞ exceeds none by `> δ` |

**[C] No outcome's antecedent is satisfied. The rule is INCOMPLETE over the observed
measurement space.**

**The missing decision cell, stated precisely:** *"the equivalence margin is satisfied for
all finite arms, but the reference arm N=∞ is itself Class C (and one finite arm is Class
C/B1, a benign mode-migration breakdown, not an H₁ signature)."* Outcome 1 is the only
outcome whose **consequent** (H₀) is consistent with the equivalence data, but its
**antecedent** demands that *all* arms — including the reference — be Class B, which the
capped frozen arm violates. This is exactly the gap M-7 named, and it exists **because**
M-2/M-3 leave the frozen arm's Class-C label (B4) unresolved.

### 9.3 Decision

**[S]** Per the decision-completeness protocol, this missing cell is **not** resolved by
interpretation. The rule is classified **incomplete**, and the report **stops before
issuing H₀ or H₁**. The campaign-level scientific decision is therefore:

> **NOT EMITTED — pre-registered decision rule incomplete over the observed measurement
> space; a defined outcome does not exist for "all finite arms within `δ` while the
> reference arm is Class C."** This matches, and now *explains*, the production record's
> `decision.outcome = "NOT_EMITTED_PHASE2"`.

**[I] (explicitly not a decision):** the *directional* evidence is consistent with freezing
being benign on this benchmark — no refreshed arm beats the frozen arm, and the largest
deviation is −1.21% (≈ 4× inside `δ`); the cleanest converged Class-B arm nearest the
frozen method (N=50) differs by +0.022%. This is a reading of the data, **not** a
pre-registered outcome, and must not be reported as H₀.

---

## Section 10 — Scientific decision summary

### 10.1 Resolved items

| Item | Resolution | Confidence |
|---|---|---|
| **M-1** | No arm is a gray-material artifact; `ω₁ᵗʰʳᵉˢʰ ≈ ω₁ᵗʳᵃᶜᵏ` (+1.8–2.6%, MAC ≥ 0.999). C-3 fix confirmed in production. | High |
| **M-2** | The N=∞ **B4 label is unsupported**; the arm is a capped, non-converged run with an **unattributed** mechanism. | High |
| **M-9** | Baseline `ω₁⁽⁰⁾(solid) = 136.483` **is recorded**; sufficient for A4's SS-beam scope; gains computable and unambiguous given the declared solid baseline. | High |
| **V-A4-1** | Baseline invariance **confirmed** (`MAC = 1` analytic; `ω₀²` ratio `= a/b = 0.25` confirmed against recorded `ω₀`; design bit-identical). | High (analytic + recorded); the direct uniform-run leg is precluded by the solver guard |

### 10.2 Unresolved items

| Item | Status | Blocking reason |
|---|---|---|
| **M-3** | limit_cycle correctly "not measured"; formal limit-cycle classification **insufficient evidence** | No Phase-3 detector run; raw Δx not cleanly period-2 |
| **M-7** | Decision-completeness gap **confirmed open** | No pre-registered outcome for reference-arm = Class C; not amendable post-hoc |
| **B4** | **Narrowed** (label refuted) but **unresolved** | `omitted_term_ratio` never measured (null); no declared threshold |
| **H₀/H₁** | **NOT EMITTED** | Decision rule incomplete over observed space (Section 9) |

### 10.3 Evidence chain (per conclusion)

- M-1 ← `a4_result.json` (ω₁ᵗʳᵃᶜᵏ, ω₁ᵗʰʳᵉˢʰ, mac_thresholded_to_phi0) ← Phase-1 C-3 fix.
- M-2 ← `a4_result.json` (iterations=cap, final_design_change=3.03e-3, limit_cycle=null,
  omitted_term_ratio=null, class_reason).
- M-3 ← `a4_iteration_histories.csv` + `a4_checkpoint_inf.mat.history.jsonl` (Δx tail
  band, stationary objective, non-period-2 signs); `a4_result.json` (limit_cycle=null).
- M-7 ← V3 §5.3 (four outcomes) vs Section 1 class assignments.
- M-9 ← `a4_iteration_histories.csv` (reference_omega=136.483, solid_phi0); V3 §3.2–§3.4.
- B4 ← `a4_result.json` + `a4_screening_events.json` (omitted_term_ratio=null everywhere).
- V-A4-1 ← recorded ω₀ + V3 §3.3 analytic identity + V-P2-1/2 bit-identity.
- Decision ← Section 9 truth table; `a4_result.json` decision block.

### 10.4 Recommended manuscript wording (conservative; wording only — the manuscript is **not** edited here)

> On the simply-supported-beam benchmark, periodically refreshing the reference eigenpair
> at every tested interval (N ∈ {50, 10, 5, 1}) yielded a converged design whose true
> fundamental frequency differed from the frozen-reference design by at most 1.2% (at
> N = 1), well within the pre-registered 5% equivalence margin; no refreshed arm exceeded
> the frozen design. A formal equivalence verdict is not issued: the frozen-reference arm
> did not converge within the 2000-iteration cap, and its non-convergence mechanism could
> not be attributed (the omitted load-sensitivity covariate was not measured). The
> directional evidence is consistent with the frozen eigenpair being an adequate proxy on
> this benchmark; establishing this as a formal result requires a converged frozen-arm
> reference. Reported frequency gains use the solid-domain baseline
> (ω₁⁽⁰⁾ = 136.5 rad/s); the design and true frequency are invariant to this choice, while
> the gain ratio is not.

Do **not** use: any "refreshing contaminates the spectrum" (B3) claim; any B4 /
sensitivity-omission-instability claim for the frozen arm; any gray-material-artifact
claim for the thresholded frequency.

---

## Section 11 — Traceability table

| Specification item | Evidence (validated artifact / validator) | Classification | Remaining uncertainty |
|---|---|---|---|
| M-1: `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵗʰʳᵉˢʰ` Class-B criterion | `a4_result.json`, `a4_table.md` (ω_thresh, mac_thresh) | **Resolved:** no gray-material artifact | `≈` tolerance not pre-registered → binary gate is Phase 3 |
| M-2: B4 without B4 evidence | `a4_result.json` (limit_cycle/omitted_term null; class_reason) | **Resolved:** B4 label unsupported; capped/unattributed | True mechanism unknown (→ B4 assessment) |
| M-3: limit_cycle default | `a4_iteration_histories.csv`, `*_inf*.history.jsonl` | **Partially enabled:** value = "not measured"; Δx retained | Formal limit-cycle detection: insufficient evidence |
| M-7: INDETERMINATE not pre-registered | V3 §5.3 vs Section 1 classes | **Confirmed open:** decision cell missing | Needs converged frozen arm or advance §5.3 amendment |
| M-9: baseline `ω₁⁽⁰⁾` | `a4_iteration_histories.csv` (reference_omega=136.483, solid_phi0) | **Resolved:** recorded; sufficient for SS-beam scope | Solid-vs-uniform gain ambiguity (Rev-2 C2) out of scope |
| B4 attribution | `a4_result.json`, `a4_screening_events.json` (omitted_term=null) | **Narrowed, unresolved** | Second conjunct never measured |
| V-A4-1: baseline invariance | recorded ω₀; V3 §3.3; V-P2-1/2 (SHA-256) | **Confirmed** (analytic + recorded) | Direct uniform-run leg precluded by solver guard |
| Decision rule (V3 §5.3) | Section 9 truth table | **Incomplete over observed space** | No outcome for reference-arm = Class C |
| E-1 window truncation (C-1) | V-P2-3 (idx 49 it25, idx 37 it30) | **Confirmed:** truncation, not contamination | — |
| E-4 symmetry (C-2) | `a4_screening_events.json` (it 1–3, all arms) | **Confirmed:** early-design property, not refresh | — |

---

## Section 12 — Actions taken and not taken (discipline record)

- **No** source code was modified; **no** MATLAB was run; **no** production data was
  generated; **no** threshold, constant, acceptance rule, factor definition, or hypothesis
  was changed; the manuscript and response letter were **not** touched.
- **No baseline eigensolve was performed.** The Baseline protocol authorizes exactly one
  eigensolve *only if* `ω₁⁽⁰⁾` is absent from the artifacts; it is **present**
  (`reference_omega = 136.483085251141`, solid_phi0), so the precondition did not fire.
- **No deprecated B3 classification was reused.** Classification followed the corrected
  E-class / Class-A/B/C taxonomy.
- All measurement-integrity conclusions of Phase 5 (V-P2-1 … V-P2-9) were treated as fixed
  input and were **not** re-derived.

**Final position.** M-1, M-2, M-9, and V-A4-1 are resolved with high confidence from the
validated Phase 5 evidence. M-3 and B4 are bounded but cannot be closed without unmeasured
covariates. M-7 confirms the pre-registered decision rule is **incomplete** over the
observed measurement space. Consequently **no H₀/H₁ is emitted** — the scientifically
conservative and correct outcome given the evidence.
