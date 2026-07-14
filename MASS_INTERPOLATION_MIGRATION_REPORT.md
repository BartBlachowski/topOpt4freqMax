# MASS_INTERPOLATION_MIGRATION_REPORT

**Date:** 2026-07-14
**Governing decision:** [`MASS_INTERPOLATION_DECISION.md`](MASS_INTERPOLATION_DECISION.md)
**Task type:** scientific consistency only. **No numerical algorithm, configuration, or result was
modified. No experiment was regenerated. No new experiment was introduced. A4 was not touched.**

---

## 1. The decision being integrated

The proposed method interpolates element mass **linearly** in density, with a `ρ_min` floor:

```
m_e(x_e) = [ ρ_min + x_e (ρ_0 − ρ_min) ] · m_e^0
```

SIMP `p = 3` penalizes **stiffness only**. **No low-density (Du–Olhoff) correction branch is
applied.** The implementation was always correct; the manuscript was wrong. The localized modes
may **not** be explained by claiming that a mass exponent removes them; current evidence
associates them with **disconnected solid components**, stated as evidence rather than as a
proven mechanism.

---

## 2. Every modified file

**Created (2):**

| File | Purpose |
|---|---|
| `MASS_INTERPOLATION_DECISION.md` | The authoritative decision: decision, evidence, provenance, mathematical + literature justification, rejected alternatives, impacts, open questions. |
| `MASS_INTERPOLATION_MIGRATION_REPORT.md` | This report. |

**Modified (7):**

| File | Change |
|---|---|
| **`paper/main.tex`** | Eq. (1) mass law; eigenvalue-sensitivity equation; §4.1 material paragraph; the localized-mode paragraph. See §3. |
| **`paper/reviews/revision_plan.tex`** | Three sites asserting `d=p=3` as the method's mass model and as the *cause* of the spurious modes (B1 item; §3.2 note; B4 item). |
| **`paper/reviews/REVISION_AUDIT.md`** | Demand **M7** ("demonstrate no spurious low-density modes at `d=p=3`") — obsolete premise **and** refuted claim; retracted. |
| **`SCIENTIFIC_DECISION_MEMO.md`** | §3.1 limitation reformulation; §6.1 "root cause / known fix" assessment superseded. |
| **`REVISION_R1_STATUS.md`** | Two limitation reformulations that attributed the modes to linear mass and prescribed void-mass penalization as the fix. |
| **`scripts/revision_v1/IMPLEMENTATION_MAP.md`** | S1-4 mitigation item: mass-interpolation axis marked exhausted; **no new experiment scheduled**. |
| **`examples/Revision_v1/revision_v1_update1.md`** | Workstream 6 mitigation instruction: same. |
| **`examples/Revision_v1/REVISION_CURRENT_STATE_REPORT.md`** | S1 remaining-work row. |

**Verified untouched:** `git diff --name-only -- analysis/ tools/` → **empty.** No solver, no
interpolation routine, no configuration, no result artifact.

---

## 3. Every corrected statement

### Manuscript (`paper/main.tex`)

| Was | Now |
|---|---|
| `m_e(x_e) = x_e^d m_e^0`, *"`p = d = 3` are the penalization exponents"* | `m_e(x_e) = [ρ_min + x_e(ρ_0 − ρ_min)] m_e^0`; `p = 3` applied to **stiffness only**; mass **linear**, tied to the volume measure and to the inertial reading of `f = ω₀²M(x)Φ₀` (Eq. `eqn:load`). |
| Eigenvalue sensitivity: `− λ_j · d · x_e^{d−1} · φᵀm_e^0φ` | `− λ_j · (ρ_0 − ρ_min) · φᵀm_e^0φ` — no power of `x_e`, because the mass law is linear. |
| §4.1: *"The SIMP penalization exponent is `p = 3` for both stiffness and mass."* | *"`p = 3` … applied to the stiffness only; the mass is interpolated linearly, as in (1), and no low-density correction branch is used."* |
| *"Following the recommendation of Du and Olhoff, the mass interpolation uses a higher penalization exponent `d ≫ p` for elements with `x_e ≤ 0.1`; … `d = p = 3` is used in the proposed approach **without inducing localized modes in the examples reported here**."* | **Rewritten.** The **global mass law** and a **low-density correction branch** are now explicitly distinguished as two different devices. Du & Olhoff's branch (linear above 0.1, higher-order below) is described and attributed to the local `OlhoffApproach`. The proposed method is stated to apply **no** low-density correction. The claim *"without inducing localized modes"* is **withdrawn**. The disconnected-component evidence is reported, explicitly hedged as current evidence and not an established mechanism, with uncertainty stated (`E_min` role not isolated; mass exponent changes which mode is lowest without removing the family). |

**Validated:** the SIMP equation is genuinely equation (1) (no earlier equation environment
exists), so the in-text "(1)" references are correct; `eqn:load`, `sec:discussion`, and
`sec:numericalExamples` all resolve; `Deng2024` exists in `literature.bib`. *(An earlier draft of
this edit referenced two labels — `eqn:opt`, `sec:proposed` — that do not exist; that would have
broken the build and was caught and corrected.)*

### Project documents

| File | Corrected statement |
|---|---|
| `revision_plan.tex` B1 | *"they arise from using `d=p=3` for the mass interpolation in the proposed method"* → withdrawn; mass axis exhausted; disconnected-component evidence. |
| `revision_plan.tex` §3.2 note | *"void elements with `x_e ≈ 10⁻³` and `d=p=3`"* → attribution removed. |
| `revision_plan.tex` B4 | *"SIMP with `d=p=3` for the mass generates spurious localized modes"* → superseded; RAMP may be *discussed* but not presented as the identified cause. |
| `REVISION_AUDIT.md` M7 | *"Demonstrate no spurious low-density modes at `d=p=3`" — ✅* → **❌ retracted**: obsolete premise **and** refuted claim. |
| `SCIENTIFIC_DECISION_MEMO.md` §6.1 | *"well-understood SIMP artifact … its solution is also well-understood: Heaviside projection combined with void-mass penalization"* → withdrawn; the "known fix" is **not known to be a fix**. |
| `SCIENTIFIC_DECISION_MEMO.md` §3.1 | Limitation text rewritten: linear mass declared; mass not the cause; disconnection evidence, hedged. |
| `REVISION_R1_STATUS.md` (×2) | *"void-mass penalization is required for a clean spectrum"* → removed; three mass models tested, none removes the family. |
| `IMPLEMENTATION_MAP.md` S1-4, `revision_v1_update1.md` WS6 | Mitigation framing no longer premised on the mass model being the cause. **No new experiment scheduled.** |

---

## 4. Part 3 — classification of every remaining occurrence

| Class | Files | Action |
|---|---|---|
| **CORRECT** — already consistent with the decision | `analysis/ourApproach/Matlab/topopt_freq.m` (`pmass` default 1); `analysis/ourApproach/Python/topopt_freq.py` (same); `our_mass_interpolation.m` (linear `power` law + `du2007_c1` available but off); `analysis/YukselApproach/*` (linear mass); `tools/Matlab/run_topopt_from_json.m` (passes `optimization.pmass` only when present); `s1_mitigation_400x50_pilot.m` (explicitly calls `pmass=1` "the baseline linear") | **None.** These *are* the declared method. |
| **HISTORICAL** — preserved deliberately | `paper/reviews/final_review_V1.tex`, `final_review_V2.tex` (reviewer inputs quoting the v1 manuscript's `d=p=3` claim — **reviewer comments must not be rewritten**); `OLHOFF_EXACT_MIGRATION_REPORT.md`; `NUMERICAL_BEHAVIOR_FREEZE.md` | **None.** Rewriting reviewer text or a dated migration record would falsify the record. |
| **DIAGNOSTIC ONLY** | `archive/diagnostics/eq4b_hypothesis_test/` (Du–Olhoff Eq. 4b, refuted); `archive/diagnostics/s1_mode_diagnostic/`, `localized_mode_onset/`; `analysis/OlhoffApproachExact*/` (incl. `mass_interp.m`, `disconnected_local_mode_audit_report.md`, `disconnection_analysis.md`, `initial_frequency_verification.md`) | **None.** Governed archive; not reviewer evidence. Cited in the decision only as diagnostics. |
| **CORRECT (comparator)** | `analysis/OlhoffApproach/*` — uses a low-density branch (`d = 6`), which is *correct for that implementation* and is now explicitly contrasted with ours in the manuscript. | **None.** |
| **OBSOLETE — contradicts the decision, NOT silently resolved** | `examples/Revision_v1/exp2_clamped_beam.m:314` and `examples/Revision_v1/A4_SPECIFICATION_V3.md` (4 sites) | **Reported below, not edited.** See §5. |

---

## 5. Remaining contradictions — reported, not silently resolved

> Per the task's own instruction: *"If any contradiction remains, report it instead of silently
> resolving it."* Two remain. **Neither was edited, and both are deliberate.**

### 5.1 `examples/Revision_v1/A4_SPECIFICATION_V3.md` — ~~BLOCKER~~ **RESOLVED 2026-07-14**

Originally reported as a blocker and left unedited under the instruction *"Do NOT touch A4."*
**A follow-up task authorized the synchronization, and it is complete.** All four contradicting
sites are corrected:

| Site | Was | Now |
|---|---|---|
| §0.2 | *"The mass-interpolation exponent contradicts the manuscript — and it is the S1 pathology"*; claimed to explain EXP4's −62% | **Retracted in place.** Declares the linear model; states the void-mass mechanism was empirically refuted (`low_density_kinetic_fraction = 0.0000`); records that EXP4's −62% is **unexplained in mechanism**. |
| Part 2 fixed-factor table | `p_M = 3`, *"mandatory… A4 must test the method as published"* | **`pmass = 1` (linear)**, per the decision; stated explicitly in the base config so arms cannot drift. |
| Part 6 (S1 dependency) | *"pin `p_M = 3` and the pathology collapses; A4 is then free of S1"* | **Withdrawn.** No mass setting removes the confound. A4 must *detect* it and *measure* (Gate A4-Pre) whether the SS beam is affected at all. The dependency is **harder**, not softer. |
| Part 9 preconditions + Appendix | *"`pmass = 3` must be set"* | **`pmass = 1`**; appendix rewritten as a retraction. |
| Part 7 (base config, runner guard) | `pmass: 3`; fail loud if `pmass ≠ 3` | `pmass: 1`; fail loud if `pmass ≠ 1`. |

**One latent defect was exposed and fixed in the process — the most important outcome of this
synchronization.** The **B3 contamination detector** was specified to fire on *kinetic energy in
low-density elements*. That quantity is **`0.0000` for every observed mode, at every `pmass`
tested**. As written, **the detector would never have fired**: A4 would have refreshed into a
polluted spectrum, failed to detect the pollution, and published it as accuracy evidence — exactly
the EXP4 failure the specification exists to prevent. It is now a **support-connectivity screen**
(new §4.3.1), keyed on `largest_support_component_kinetic_fraction` (0.937 physical vs 0.0006
spurious), `dominant_component_touches_both_supports`, and `low_density_strain_fraction` — a
three-orders-of-magnitude separation, so it is not threshold-sensitive.

**Unaffected and still standing:** the single-factor design, the true-`ω₁` endpoint, the ban on the
surrogate judging itself, the unique `N=∞` baseline and its invariance result, the three-class
acceptance framework, the pre-registered null outcome, and the finding that `update_after` does not
exist on the `semi_harmonic` path.

### 5.2 `examples/Revision_v1/exp2_clamped_beam.m:314` — obsolete

```matlab
Emin=1e-6*E0; penal=3; rhoMin=1e-6; pmass=3;
```

This retired, pre-authoritative script *optimized* at `pmass = 1` (the solver default) and then
*verified its own frequency* at `pmass = 3` — an internal inconsistency independent of this
decision. **Not modified:** this is a code change, and the task is explicitly not an
implementation task. It is already unregistered as a stage and **denied by name in the runner's
preflight P2**, so it cannot enter the campaign. Recommended: archive it under
`archive/obsolete_evidence/` in a future code-scoped task.

---

## 6. Remaining uncertainties (carried forward, not resolved)

1. **The disconnected-component association is evidence, not proof.** The role of `E_min`
   (`10⁻⁶E₀` clamped/building vs `10⁻⁹E₀` SS beam) has not been isolated; no connectivity
   treatment has been tested. The manuscript states this uncertainty explicitly.
2. **No remedy exists for the artefact the paper names.** Linear, `pmass=6`, and Du–Olhoff Eq. 4b
   all fail to remove the mode family. The paper now reports this as a limitation rather than
   claiming a solution.
3. **`ρ_min` is not uniform across the examples** (`10⁻⁹ρ₀` in §4.1; `10⁻⁶` in §4.2/§4.3).
   Faithfully reported, never justified.
4. **The `x_e ∈ [10⁻³, 1]` lower bound** interacts with `ρ_min` and `E_min` in the void regime in
   ways the manuscript does not document.
5. **S1's scientific goal remains unmet.** This decision changes the *explanation* that may be
   offered, not the outcome.

## 7. Remaining scientific dependencies

- **S1 → EXP2b, EXP3** — unchanged; the localized-mode question still gates them.
- **A4** — now additionally blocked on the A4-spec correction in §5.1 (it currently pins the wrong
  mass model).
- **CR2** — the omitted load-sensitivity term is `∂f/∂x = ω₀²(∂M/∂x)Φ₀`. Under the linear law
  `∂M/∂x` is **constant**, which is the regime in which CR2's **71.3%** figure was measured. That
  measurement is therefore *consistent* with the declared method and does **not** need re-deriving —
  a dependency that would have arisen had `d = 3` been adopted.
- **Manuscript §4.2 / §4.3** state "SIMP penalization `p = 3`" without qualifying stiffness vs
  mass. These are not contradictions (they do not claim mass penalization) and were left
  unmodified per *"do not rewrite unrelated text"*; a future editorial pass may wish to make them
  explicitly parallel to §4.1.

---

## 8. Confirmation

> ### The manuscript and the repository now describe the same mass interpolation model.

| Check | Result |
|---|---|
| Manuscript mass law | `m_e = [ρ_min + x_e(ρ_0 − ρ_min)] m_e^0` — **linear**, `ρ_min` floor |
| MATLAB implementation | `pmass = 1` default; `rho_e = rho_min + x^1·(rho0 − rho_min)` — **identical** |
| Python implementation | `pmass = 1` default; same expression — **identical** |
| Active production configs | none sets `pmass` → all inherit linear — **consistent** |
| Manuscript stiffness penalization | `p = 3`, stiffness only — **matches code** |
| Low-density correction branch | manuscript: none; code: `du2007_c1` exists but **off by default** — **consistent** |
| Surviving global-`d=3` claims outside the archive, reviewer inputs, and the two reported items | **none** |
| Solver / algorithm / config / result files modified | **zero** |
| Experiments regenerated | **zero** |
| New experiments introduced | **zero** |
| Computational cost of this decision | **zero hours** — every existing result was already computed with the now-declared model |

**Status of the two reported contradictions:**

- **§5.1 — A4 specification: RESOLVED** (2026-07-14, follow-up task). `A4_SPECIFICATION_V3.md` now
  pins `pmass = 1`, retracts its §0.2 causal narrative in place, hardens the S1 dependency, and —
  critically — replaces its inoperative B3 detector with a support-connectivity screen.
- **§5.2 — `exp2_clamped_beam.m:314`: OPEN, contained.** Retired, unregistered, and denied by name
  in preflight P2, so it cannot enter the campaign. Fixing it is a code change and belongs to a
  code-scoped task.

With that single contained exception — named explicitly, not silent — **the repository and the
manuscript now describe exactly the method that is actually implemented.**
