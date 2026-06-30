# SCIENTIFIC DECISION MEMO

**Date:** 2026-06-29
**Purpose:** Conservative, reviewer-safe assessment of which original manuscript
claims are supportable, which must be removed, and which can be reformulated
as limitations based solely on documented artifacts.
**Scope:** A0, I1, V1, CR2, Exp2, Exp3, S1 baseline, S1 pmass=6 mitigation,
Eq.4b hypothesis test. Does not cover A4, P1, MS, RP.
**Constraint:** No experiment reruns. No solver code edits. No manuscript edits.

---

## 1. Which original claims are still supportable?

Claims listed here have documented artifact evidence sufficient to assert them
in a response letter or revised manuscript, with appropriate scope qualifications.

### 1.1 Formulation correctness (A0)

**Claim:** The proposed method uses the authoritative inertial load
F_j(x) = omega_0j^2 · M(x) · Phi_0j, where omega_0j and Phi_0j are eigenpairs
of the fully solid domain, mass-normalized, with deterministic sign convention.

**Evidence:** `a0_matlab_result.json` — 10-item MATLAB gate passed; FD error
4.13e-8 (tol 1e-5); mass normalization exact; no rho_nodal double-scaling;
solid reference eigenpair confirmed.

**Scope qualification:** Verified in MATLAB R2025b on a 6×2 deterministic mesh.
Python parity not certified (A0-G not run).

---

### 1.2 Multi-mode objective linearity (V1-3)

**Claim:** The multi-mode compliance objective is a weighted sum of per-case
objectives; per-case objective scales as factor^2 of the load factor.

**Evidence:** `v1_3_multimode_results.json` — additivity to floating-point zero;
factor^2 scaling to rel err 1.8e-15.

**Scope:** Verified on the 6×2 fixture.

---

### 1.3 Load sensitivity term is mathematically non-negligible (V1-4)

**Claim:** The term dF/dx_e = omega_0^2 · (dM/dx_e) · Phi_0 is nonzero and
contributes substantially to the complete gradient.

**Evidence:** `v1_4_sensitivity_results.json` — max relative branch difference
71.3% between omitted and complete sensitivities at the first-iteration design;
complete branch matches FD to 4.1e-8.

**Important:** This claim directly constrains what can be said about "omission"
of the term. The term is NOT zero and is NOT negligible in the mathematical sense.

---

### 1.4 MAC-based mode tracking is correct (V1-6)

**Claim:** The mass-weighted squared MAC correctly identifies the target mode
regardless of frequency ordering; the metric is phase-invariant.

**Evidence:** `v1_6_reordering_results.json` — permuted-set tracking accurate
to MAC=1; cross-design MAC=1.000; phase invariance confirmed.

---

### 1.5 Single-mode method produces a converged result on 200×25 clamped beam (Exp2 α=1)

**Claim:** The proposed method (α=1, load from mode 1) converges on the 200×25
clamped-clamped beam and produces a nontrivial solid-void topology with
omega_1 = 141.79 rad/s.

**Evidence:** `exp2_authoritative_alpha_1_00_result.json` — 1052 iterations,
dc=1.66e-4, MAC=0.993, feasibility=0, grayness=8.6%, A5 check passed.

**Scope qualification:** Single mesh (200×25). Same result is shared with
Exp3 200×25. Cannot be extended to finer meshes (400×50 mode-invalid).

---

### 1.6 Localized low-density modes exist at 400×50 (S1 diagnostic evidence)

**Claim:** The 400×50 clamped-beam final design under SIMP (pmass=1) contains
localized low-density modes. Modes 2–10 of the final design have strain-energy
fraction in low-density regions ≥ 0.96; mode 1 is ambiguous.

**Evidence:** `s1_exp3_400x50_mode_summary.json` — 0 physical global, 8 localized,
2 ambiguous modes from energy-fraction analysis with documented criteria.

**Claim form for response letter:** "Energy analysis of the 400×50 final design
confirms localized low-density modes: modes 2–10 have >96% of their strain
energy concentrated in elements with ρ < 0.05. This is a known SIMP artifact."

---

### 1.7 pmass=6 recovers mode 1 but not the spectrum

**Claim:** Increasing the mass penalization exponent to pmass=6 produces a
physically meaningful mode 1 (MAC=0.974, omega_1=131.93 rad/s) at 400×50,
but does not eliminate localized modes from the adjacent spectrum.

**Evidence:** `s1_mitigation_400x50_mode_summary.json` — mode 1 classified
"physical global" (kinetic fraction on support-connected component = 93.7%);
modes 2–10 all "localized low-density".

**Scope:** This is a partial mitigation, not a solution. It demonstrates the
problem can be partially suppressed but shows the requirement for a stronger
length-scale enforcement.

---

### 1.8 Variant B (complete sensitivity, OC) exhibits period-2 instability

**Claim:** OC with complete load sensitivity and move_limit=0.20 on the
160×20 SS beam exhibits a period-2 cycle from iteration 9; reducing move_limit
to 0.02 does not eliminate the instability.

**Evidence:** CR2 rerun artifacts — dc=0.20=move_limit throughout 400 iterations;
objective alternates 4833↔4959; same pattern at move_limit=0.02.

**Claim form for response letter:** "We document a period-2 instability in the OC
optimizer when complete load sensitivity is used; this is consistent with known
OC instability patterns under gradient overshooting. The practical effect of
sensitivity omission on converged objectives could not be quantified because
Variant B did not converge."

---

## 2. Which claims must be removed?

These claims are either directly contradicted by documented evidence or
require evidence that does not exist.

### 2.1 "No spurious low-density modes"

**Original form:** The method avoids spurious low-density modes.

**Why it must be removed:** S1 baseline shows 8 localized low-density modes in
the first 10 modes of the 400×50 final design (pmass=1). S1 mitigation (pmass=6)
leaves 9 of 10 modes localized. Eq.4b runs capped. No tested configuration
produces a clean spectrum. The claim is directly contradicted.

**Action:** Remove from manuscript and response letter in all forms.

---

### 2.2 "Omitted load sensitivity is negligible" or "effectively zero"

**Why it must be removed:** V1-4 documents 71.3% max relative branch difference
at the first-iteration design. CR2 rerun protocol Section 6.B explicitly forbids
the "negligible" characterization. The term is mathematically non-negligible.

**Action:** Replace everywhere with: "The load-sensitivity term dF/dx is nonzero
and has been omitted to avoid OC instability (demonstrated by Variant B, CR2).
The practical impact on converged results could not be quantified within this
work [cite Variant B failure]."

---

### 2.3 Multi-mode targeting (α sweep) as a working feature

**Why it must be removed:** α=0.50 and α=0.25 produced mode-invalid results
(localized low-density modes suppressed the physical response). α=0.75 and
α=0.00 completed in 1 iteration with a uniform topology — no optimization
occurred. The multi-mode capability is not demonstrated in any scientifically
meaningful way beyond the single-mode case (α=1).

**Action:** Remove all α-sweep narratives, monotonicity claims, and two-mode
design comparisons. The α parameter may be described in the method section but
cannot be supported with experimental evidence from the current runs.

---

### 2.4 Mesh convergence of the proposed method

**Why it must be removed:** Exp3 400×50 is mode-invalid (MAC=0.786, omega_1
dropped from 141.79 to 64.39 rad/s, topology correlation = −0.088). The
forensic audit confirmed no setup inconsistency; the divergence is likely
due to localized-mode pathology. No mesh-convergence evidence exists.

**Action:** Remove any claim that the method produces mesh-converging results.

---

### 2.5 Performance claims: 7.1× speedup, 8.6% frequency gap, 4.61× building gain

**Why they must be removed:** P1 not run. No instrumented timing data
exists under the authoritative formulation. The frequency gap cannot be
stated without a converged Olhoff-inspired comparison on the same problem.
The building gain cannot be stated without converged building results.

**Action:** Remove all three numbers from manuscript and response letter.

---

### 2.6 O(N_e^1.3) scaling with stated precision

**Why it must be removed:** Table 3 values in `algorithms_comparison.tex`
are explicitly marked as placeholders ("final values should be updated after
rerunning"). The log-log fit of the current placeholder data gives β≈1.07,
not 1.30. P1 not run.

**Action:** Remove scaling exponent claims. If the algorithms_comparison.tex
document is retained for the review response, add an explicit caveat that
exponents will be recomputed from instrumented benchmarks.

---

### 2.7 N=1 refresh equivalence to Yuksel-Yilmaz

**Why it must be removed:** A4 not run; no equivalence evidence exists.

**Action:** Remove from manuscript.

---

### 2.8 Comparator labeled as "Olhoff" or "Yuksel-Yilmaz" without qualification

**Why it must be removed:** Implementations are local modifications of those
methods. Faithful reproduction has not been verified.

**Action:** Relabel as "Olhoff-inspired implementation" and
"Yuksel-Yilmaz-inspired implementation" throughout all tables, figures,
and text. Remove "canonical" language.

---

## 3. Which claims can be reformulated as limitations?

These are claims where partial evidence exists and an honest reformulation
is possible without overstating the findings.

### 3.1 Localized-mode problem as a stated limitation

**Original claim (to remove):** The method avoids spurious modes.

**Reformulation:** "Under SIMP with linear mass interpolation (pmass=1), localized
low-density modes appear in the frequency spectrum at fine mesh resolution
(400×50 clamped beam). Energy analysis confirms 8 of the first 10 modes
of the final design are localized in low-density regions with strain-energy
fractions >96%. Increasing the mass penalization exponent to pmass=6 recovers
the fundamental mode as a physical global mode (MAC=0.974) but does not
eliminate spurious modes from the adjacent spectrum. Length-scale enforcement
(Heaviside projection or void-mass penalization) is required to fully resolve
this issue and is identified as a necessary extension."

---

### 3.2 Multi-mode method described as a formulation capability, not a demonstrated result

**Original claim (to remove):** Multi-mode designs produced and compared.

**Reformulation:** "The formulation supports multi-mode targeting via the α
weighting parameter. A preliminary single-mode result (α=1, 200×25 mesh) is
demonstrated. Extension to multi-mode designs (α<1) encountered the same
localized-mode pathology at the 200×25 resolution, preventing accepted results
for α=0.5 and α=0.25. Resolution of the spurious-mode problem is a prerequisite
for multi-mode demonstration."

---

### 3.3 Load sensitivity omission framed as a practical design choice, not a validation result

**Original claim (to remove):** Omission is negligible.

**Reformulation:** "The load sensitivity term dF_j/dx_e = omega_0j^2·(dM/dx_e)·Phi_0j
is nonzero and was verified to differ from the complete gradient by up to 71.3%
per element at the initial design. The term is omitted in the current implementation
because including it caused OC optimizer instability (period-2 cycle documented
in the sensitivity study). The practical effect on converged objective values
is unknown; quantifying it via a matched optimizer comparison is identified
as future work."

---

### 3.4 Performance advantage characterized qualitatively, not quantitatively

**Original claim (to remove):** 7.1× speedup, 8.6% gap.

**Reformulation:** "The proposed method eliminates the per-iteration eigensolve
required by the Olhoff-inspired approach, replacing it with a single static
linear solve per iteration. This is expected to reduce per-iteration cost by
a factor proportional to the Krylov solver overhead. Quantitative speedup
measurements require instrumented benchmarks and are not reported in this
revision."

---

### 3.5 Mesh sensitivity as a known open issue

**Original claim (if present):** Results are mesh-independent or mesh-converging.

**Reformulation:** "Results on the 200×25 mesh are presented; the 400×50
refinement produced a mode-invalid outcome consistent with localized low-density
mode formation. The method's behavior under mesh refinement is sensitive to
the treatment of low-density regions and requires length-scale enforcement
for robust convergence."

---

## 4. Which results are safe to show?

The following results have accepted artifact evidence and can be included
in the response letter or revised manuscript with appropriate qualification.

| Result | Artifact | Qualification required |
|---|---|---|
| α=1 converged topology on 200×25 | Exp2 α=1.00 result JSON | Single mesh; not mesh-convergence evidence |
| omega_1 = 141.79 rad/s for 200×25 | same | — |
| Grayness = 8.6% for 200×25 | same | pmass=1; fine-mesh grayness not known |
| A0 10-item gate results table | a0_matlab_result.json | MATLAB-only; Python parity not certified |
| FD verification table | v1a_fd_results.json | 6×2 mesh only |
| Branch difference 71.3% | v1_4_sensitivity_results.json | First-iteration design; not converged comparison |
| V1-3 additivity/scaling table | v1_3_multimode_results.json | 6×2 mesh |
| V1-6 MAC tracking table | v1_6_reordering_results.json | 6×2 mesh |
| S1 mode classification table | s1_exp3_400x50_mode_summary.json | As limitation evidence |
| S1 pmass=6 mode 1 recovery | s1_mitigation_400x50_mode_summary.json | As partial-mitigation evidence |
| CR2 Variant B period-2 signature | cr2_rerun_manifest.json | As diagnostic evidence of OC instability |
| Eq.4b formula validation | eq4b_validation_result.json | Formula validated; run capped; not a positive result |

---

## 5. Which figures/tables must NOT be used?

| Item | Reason |
|---|---|
| Any α-sweep comparison figure (α=0, 0.25, 0.5, 0.75, 1.0) | α=0.75/0.00 trivial; α=0.25/0.50 mode-invalid |
| Topology figures from Exp3 400×50 | Mode-invalid; localized modes; topology anti-correlated with 200×25 |
| Comparison of 200×25 vs 400×50 topologies as mesh convergence | Forbidden — designs are unrelated |
| Table 3 timing values (algorithms_comparison.tex) as stated | Placeholder values; exponents inconsistent with data |
| Any speedup table showing 7.1× or similar | No accepted evidence base |
| Any frequency gap table showing 8.6% | No accepted evidence base |
| Performance comparison table with unqualified "Olhoff" / "Yuksel-Yilmaz" labels | Local implementations; not canonical |
| Any figure claiming "no spurious modes" | Directly contradicted |
| Topology figures from pmass=6 run presented as "clean" | 9/10 modes still localized |
| Eq.4b result as a positive demonstration | Run capped and rejected |
| Multi-mode topology figures as accepted results | Only α=1.00 is a genuine accepted result |

---

## 6. Whether further experiments are likely to rescue the original narrative

### 6.1 The localized-mode problem is the central blocker

All major failures (Exp3 400×50 mode-invalid, α=0.25/0.50 mode-invalid,
Eq.4b capped) share the same root cause: localized low-density modes
overwhelming the physical response at fine mesh resolution. This is a
well-understood SIMP artifact, and its solution is also well-understood:
Heaviside projection combined with void-mass penalization. However:

- Both require solver code changes (not permitted in this memo's scope).
- pmass=6 alone is insufficient (demonstrated).
- Eq.4b alone is insufficient (run capped).
- The forensic audit found no simpler fix (BCs, filter, material all correct).

**Assessment:** Resolving the localized-mode problem requires implementing
Heaviside projection or void-mass penalization in `topopt_freq.m`. If
implemented and verified, it would likely:
- Fix the 400×50 Exp3 failure (restoring mesh-convergence evidence).
- Fix the α=0.25/0.50 Exp2 failures (restoring the α-sweep narrative).
- Eliminate spurious modes from the S1 spectrum.

This is a meaningful but technically straightforward extension. It requires
1–2 weeks of implementation + testing. **It is recommended.**

### 6.2 CR2 can potentially be resolved but requires optimizer change

Variant B (complete sensitivity, OC) has a documented period-2 instability
that is not eliminated by reducing move_limit. Resolution requires either:
(a) Switching both variants to MMA — this is the correct approach but requires
    MMA to be available or implemented for the proposed method. Moderate effort.
(b) Testing at a coarser mesh (80×10 or 160×20) where OC dynamics may be stable.
    Lower effort; results less convincing for a fine-mesh claim.
(c) Abandoning the CR2 comparison and restricting the claim to the diagnostic
    failure narrative. **This is the current fallback.**

**Assessment:** CR2 can be rescued only if an accepted matched comparison pair
can be produced. Option (a) is scientifically defensible; option (c) is the
minimum safe path for R1.

### 6.3 Performance claims require P1 but are structurally achievable

The timing measurements needed for P1 require solver instrumentation and
≥10 benchmark runs per mesh. There is no scientific barrier — the code works;
timing just needs to be measured. The corrected scaling exponent will likely
be ≈1.07–1.11 based on current placeholder data, which still qualitatively
supports the complexity analysis.

**Assessment:** P1 is achievable with ≈1–2 weeks of implementation and
benchmark runs. The headline speedup claim may change when measured properly,
but the qualitative advantage of eliminating eigensolves per iteration is
structurally sound.

### 6.4 The α-sweep narrative depends on fixing the localized-mode problem

If Heaviside projection is implemented and the localized-mode problem is
resolved, the α-sweep (Exp2) can be rerun and may produce accepted results
for α=0.25 and α=0.50. Whether the α sweep shows monotone behavior cannot
be predicted in advance.

**Assessment:** The multi-mode narrative is contingent on (6.1). It cannot be
rescued independently.

### 6.5 Overall assessment

The original narrative — that the proposed method is a fast, convergent,
multi-mode approach with no spurious modes — faces two scientific issues
that cannot be addressed by simply running more experiments with the current
solver configuration:

1. **Localized-mode pathology** requires a solver modification (Heaviside
   or void-mass penalization) that is known-feasible but not yet implemented.
2. **OC instability with complete sensitivity** requires an optimizer change
   (MMA) to produce a CR2 accepted comparison.

With those two changes, the original narrative could largely be rescued.
Without them, the revision must report a substantially narrowed scope:
single-mode result on a single mesh, qualitative complexity analysis without
specific numbers, and explicit limitations on spurious modes.

---

## 7. Gate-specific analysis

### 7.1 CR2

No converged matched omitted/complete comparison exists. The documented evidence
is algorithm-failure diagnostic data: Variant B period-2 cycle (both move_limit=0.20
and 0.02), Variant A objective plateau without formal convergence. These findings
are reportable as diagnostic results but cannot substitute for a converged comparison.

**Minimum acceptable response-letter treatment:** State the CR2 failure explicitly.
Report the Variant B period-2 signature as evidence that OC is not appropriate
for complete-sensitivity optimization on this problem. State that a matched
comparison using MMA is identified as future work. Do NOT claim that the omission
is negligible or that Variant A's plateau proves omission has no effect.

### 7.2 Exp2

The α sweep produced one genuine accepted result (α=1.00, 200×25) and four
non-results (two trivial, two mode-invalid). The trivial α=0.75/0.00 outcomes
(1 iteration, grayness≈100%) are a convergence-criterion artifact, not scientific
evidence. The mode-invalid α=0.25/0.50 outcomes demonstrate the same localized-mode
pathology seen in Exp3.

**Minimum acceptable response-letter treatment:** Present α=1.00 as the single
demonstrated result. Explain that α=0.25/0.50 encountered the localized-mode
pathology (cite S1 evidence). Do not present the α sweep as a systematic comparison.

### 7.3 Exp3

The 400×50 run is mode-invalid; the topology and frequency diverge from the
200×25 accepted result (correlation = −0.088, frequency drop 55%). The S1
baseline confirmed the cause: localized low-density modes. The forensic audit
ruled out setup errors.

**Minimum acceptable response-letter treatment:** State that mesh convergence
was attempted but the 400×50 run failed the MAC gate due to localized-mode
formation. Cite the S1 energy analysis as the diagnostic evidence. State that
length-scale enforcement is required. Do NOT claim the method is mesh-convergent.

### 7.4 S1

The localized low-density mode problem persists across all tested configurations:
- pmass=1 (baseline): 0/10 modes physical.
- pmass=6 (mitigation): 1/10 modes physical, 9/10 localized.
- Eq.4b (hypothesis test): run capped, result rejected.

No configuration tested in the current revision produces a clean mode spectrum.

**Minimum acceptable response-letter treatment:** Report S1 findings as a
discovered limitation. Present the pmass=6 result as a partial mitigation.
Explicitly acknowledge that the no-spurious-mode claim cannot be supported
and must be removed. Identify Heaviside projection as the recommended fix
(with citation to standard SIMP literature on void-mass penalization).

### 7.5 Performance claims

No instrumented timing data exists. The placeholder values in Table 3
(`algorithms_comparison.tex`) do not constitute evidence and the stated exponents
are inconsistent with the data they supposedly summarize. All headline performance
numbers (speedup, frequency gap, scaling exponent) must be removed until P1 is run.

**Minimum acceptable response-letter treatment:** State that timing benchmarks
are being regenerated with isolated diagnostic overhead removed. Provide a
structural argument for the qualitative cost advantage (no eigensolve per
iteration) without specific numerical claims. Remove Table 3 from the submission
until exponents are recomputed from instrumented data.

---

*This memo is based exclusively on documented artifact evidence as of 2026-06-29.
No experiment reruns, no solver modifications, and no manuscript edits have been
performed in preparing this analysis.*
