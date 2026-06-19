# Audit — does `examples/Revision_v1/` fulfil the reviewer demands?

**Manuscript:** SAMO-D-26-00346, *"Frequency maximization of planar structures using quasi-static approximation in topology optimization"*
**Scope:** experiments/implementation under `examples/Revision_v1/`, audited against
`paper/reviews/revision_plan.tex` (authors' own A/B/C consolidation),
`review1.txt` (Reviewers 1/3/4), the Reviewer 3 PDF (13 comments),
and `final_review_V1.tex` / `final_review_V2.tex` (AI consensus: CR/MR/M items).
**Date of audit:** 2026-06-19

---

## Verdict

**NO — the revision is not complete.** Of the six experiments, **four produced
saved results (exp1, exp2b, exp4, exp5) and two FAILED with no output
(exp2 = clamped beam, exp3 = mesh convergence).** The two that failed are the
ones that carry the **single most important critical issue** (CR1 — the Table 3
α=0.75 non-monotonicity, flagged Critical #1 by *both* AI review branches) and a
**major** demand (MR5/V3 — mesh-convergence study). In addition, several of the
results that *did* run contradict the manuscript's headline claims or argue
against the method, and the manuscript text (`main.tex`) has only partially
absorbed the new data.

This audit was extended (Jun 19) with a direct read of `main.tex`, which
**confirms** that the two definitional issues at the heart of the revision
(A0 / C2 — load formula and reference-design) are still unresolved in the
manuscript, and that the CR1 monotonicity claim is still asserted in text that
its own table contradicts. See §2 and §4 for line-level evidence.
(All `.mat` structures were read via Python/scipy; no implementation files were
modified by this audit.)

Evidence for the failures (from `all_revision_results.mat`, saved Jun 3 14:46):

| Experiment | Status in master run | Saved result file |
|---|---|---|
| exp1 perf table | OK (timing 56 449 s ≈ 15.7 h) | `exp1_perf_table_results.mat` ✓ |
| **exp2 clamped beam** | **EMPTY / FAILED (timing NaN)** | **none** ✗ |
| exp2b building | OK (6 542 s) | `exp2b_building_results.mat` ✓ |
| **exp3 mesh convergence** | **EMPTY / FAILED (timing NaN)** | **none** ✗ |
| exp4 sensitivity ablation | OK (37 s) | `exp4_sensitivity_ablation_results.mat` ✓ |
| exp5 scaling | OK (18 s) | `exp5_scaling_results.mat` ✓ |

Corroborating signs that exp2/exp3 never completed: there is **no**
`exp2_clamped_beam_results.mat`, **no** `exp3_mesh_convergence_results.mat`
(anywhere in the repo), **no** `exp2_*_correlation.csv` / `exp3_*_correlation.csv`
copies, and **no** M5 mode-shape plots (`exp2_alpha1_topo_mode{1,2}.png`).
Note also that `exp1/exp2/exp2b` `.m` files were edited **Jun 10**, *after* the
Jun 3 run — i.e. the authors were still mid-debug and never regenerated a clean
consolidated result set.

---

## 1. Demand-by-demand coverage (experiment side)

Legend: ✅ delivered with usable data · ⚠️ delivered but data is problematic ·
🟡 partially / function exists but no output · ❌ not delivered · 📝 manuscript-text task (outside `Revision_v1`).

### Urgent (A-group, revision_plan.tex) and AI-review Critical (CR)

| ID | Demand | Where | Status |
|---|---|---|---|
| A0 | Align frozen-load definition (ω vs ω², solid vs initial baseline); rerun | exp2 (semi_harmonic vs harmonic-Eq.7 comparison) | ❌ designed in exp2 but **exp2 failed → no data**, and **still unresolved in `main.tex`** (see §2.5) |
| A1 | Reframe novelty vs Yuksel | main.tex §Related work | 📝 partial |
| A2 | Table 1 runtime breakdown (setup / per-iter / iters) | exp1 (`tSetup`, `tProbe`, `tPerIterAdj`), `save_frequency_iterations=false` | ⚠️ delivered, **but Olhoff capped at 2000 iters (see §2)** |
| A3 | Explain Yuksel 1083/1723-iter discrepancy | exp1 reproduces high counts (298/698/1610/878) | 🟡 data only; **no explanation** (text task) |
| A4 | Approx-error / N-refresh study (plan: SS-beam 400×50, N∈{∞,50,10,5,1}) | exp4 variants C/D | ⚠️ only **160×20, N∈{0,50}**, and result is adverse (§2) |
| A5 | Mode coalescence + a-posteriori mode check | exp2 (α=0.75 near-coalescence), exp2b MAC | 🟡 exp2b only; **exp2 failed** |
| A6 | Reframe efficiency claim | main.tex §6 | 📝 depends on exp1 timing (problematic) |
| **CR1** | **Verify/correct Table 3 Φ₂ α=0.75 non-monotone gain** | **exp2** full MAC/freq diagnostic | **❌ exp2 failed → no output, and `main.tex` L469 still asserts the monotonicity its own table breaks (see §2.6)** |
| CR2 | Validate omitted load sensitivity (Eq. 6) | exp4 FD check + Variant A vs B + direct ratio | ⚠️ rigorous, **but validates the implemented `semi_harmonic` load, not the documented Eq. 5–6 (`ω²M(x)Φ`)** — FD relerr ≈ 1e-7; A vs B = +1.13 % (see §2.5) |
| CR3 | Relabel Olhoff comparator "in-house"; include/exclude table | main.tex (+ `algorithms_comparison.tex`) | 📝 "in-house" appears once; partial |

### Major (AI-review MR + Reviewer-3/4 B-group)

| ID | Demand | Where | Status |
|---|---|---|---|
| MR1 | Std devs on Table 1 + hardware spec | exp1 (`omega_std`, `mem_std`, `tTotal_std`) + `exp6_hardware_info` | ✅ (std ≈ 0 because runs are deterministic — honest) |
| MR2 / M2 | ω₁ for every method × mesh | exp1 `omega_mean` | ⚠️ delivered but **contradicts the 8.6 % gap claim (§2)** |
| MR3 / V5 | State clamped-beam ω₁⁰, ω₂⁰ | exp2 `localSolidEigenpairs` | 🟡 function exists; **exp2 failed → not produced** |
| MR4 / V4 | Define + apply MAC validity threshold | exp2b (0.8 threshold; flags building Φ₂) | ✅ for building; 🟡 clamped (exp2 failed) |
| **MR5 / V3** | **Mesh-convergence study (≥2 meshes)** | **exp3** | **❌ exp3 failed → no data** |
| MR6 / C4 | Frozen-mode reliability diagnostic | exp4 C vs D | ⚠️ delivered but **adverse: −62 %** (§2) |
| MR7 | Scope headline claims to benchmark/mesh | main.tex | 📝 |
| B1 / M5 | Classify/plot two new modes at α=1 (clamped) | exp2 `localSaveModeShapes` | ❌ exp2 failed → **no plots** |
| B2 | High-index Φ₂ tracking = spurious modes | exp2b (indices 33/58/81/69/71) + spurious check | ✅ for building |
| B3 | Same discussion for building | exp2b | ✅ |
| B4 | RAMP interpolation discussion | main.tex (RAMP ×1) | 📝 discussion-only (per plan, acceptable) |
| B5 / M6 | Grey regions / discreteness metric; optional Heaviside run | exp2b grayness `g`; Heaviside run not done | 🟡 metric ✅ (building); clamped failed; Heaviside demo ❌ |
| B6 | Why frozen-load avoids spurious modes | exp2b spurious check + main.tex | 🟡 |
| M7 | Demonstrate no spurious low-density modes at d=p=3 | exp2b `localPrintSpuriousModeCheck` | ✅ (building) |
| M8 | Address Huang et al. (2025) gap claim | main.tex (Huang ×4) | 📝 done |

### Minor (C-group / mn) — mostly manuscript text, outside `Revision_v1`

- C5 hardware platform → **delivered** by `exp6_hardware_info` (MATLAB ver, CPU, cores, RAM, OS, BLAS).
- mn8 / M4 scaling exponent → **delivered** by exp5 (see §3).
- C1–C4, C6–C11, mn1–mn7, mn9–mn11 (references, notation, Eq.9 mass-weighting, void-material/tolerance justification, "no eigensolve in loop" wording, figure-number fix) → 📝 manuscript edits; not assessable from `Revision_v1`.

---

## 2. Substantive correctness concerns in the data that *did* run

These are arguably more serious than the missing experiments, because they would
**undermine the paper if dropped in as-is**:

1. **exp1 Olhoff does not converge — it is capped at exactly 2000 iterations at
   every mesh** (`nIter_mean = [2000,2000,2000,2000]`). Consequences:
   - Total times balloon to 493/796/1532/2570 s vs the paper's 32/74/136/222 s
     (≈ 11× at 400×50). The "clean timing" the reviewers asked for (A2/A6) is
     therefore *not* achieved; the comparator ran to a cap, not to convergence.
   - **The 8.6 % accuracy-gap headline is not reproduced.** exp1 gives
     Proposed-vs-Olhoff ω₁ gaps of **−0.53 %, +0.58 %, +0.43 %, +0.49 %**
     (Proposed is sometimes *above* Olhoff). The manuscript still states "within
     8.6 %" (main.tex L98, L661) and "7.1×" (L349, L639). With exp1 numbers the
     speedup would be ≈ 80×, not 7.1×. The regenerated data **contradicts the
     manuscript's two central claims** and cannot be inserted without
     reconciliation.
   - The **setup/per-iteration decomposition itself is not trustworthy**
     (methodologically unsound, not merely capped). For `ourApproach`, `tSetup`
     is a *standalone* eigensolve timed with **hardcoded, mismatched BCs**
     (`localTimeSingleEigensolve` pins mid-height nodes as an "SS approximation"
     regardless of the actual config), and `tPerIterAdj = (tTotal − tSetup)/nIter`
     subtracts that separately-measured time from `tTotal = tIter·nIter`, risking
     a double-subtraction. For Yuksel the "setup" is Stage 1 (~200 static-compliance
     iters), which a `max_iters=1` probe **cannot capture**. The three columns are
     therefore measured in inconsistent frames and rest on a non-converged Olhoff.

2. **exp4 frozen-vs-periodic (MR6/C4/A4) gives the opposite of the desired
   conclusion.** Frozen (Variant C) ω₁ = 131.24 rad/s; periodic-refresh every 50
   iters (Variant D) ω₁ = **49.84 rad/s** (−62 %). The periodic JSON's own note
   says *"large frequency differences indicate the frozen approximation is
   inaccurate."* As produced, this argues the frozen mode is unreliable (most
   likely the refreshed eigenpair locks onto a spurious low-density mode — the
   very M7/B1 pollution the reviewers warned about). This needs to be diagnosed
   and either fixed (mode-tracking on refresh) or it must not be cited as
   evidence of frozen-mode reliability.

3. **All exp4 variants ran to the 400-iteration cap** (`nIter = [400,400,400,400]`,
   `ch ≈ 0.005–0.006 > tol`), i.e. none satisfied the 1e-3 convergence
   criterion. Variant A also ends quite grey (g = 0.25). The A-vs-B load-sens
   conclusion (+1.13 %) still stands, but it is a comparison of two
   non-converged designs.

4. **exp2b building Φ₂ gains are all below the MAC threshold** (MAC₂ = 0.55,
   0.69, 0.40, 0.27, 0.51 < 0.8). Applying the threshold the reviewers demanded
   (MR4/V4) effectively **invalidates every multi-mode Φ₂ building result** —
   honest, but it guts a chunk of the multi-mode contribution and must be
   reflected in the manuscript. exp2b also hits the 2000-iter cap at α=1.0/0.75.

5. **A0 / C2 confirmed unresolved in `main.tex` — and exp4's CR2 validation is on
   the wrong load.** The manuscript documents the load as
   `f = (ω⁽⁰⁾)² M(x) Φ` (L130, L239, L250) but states *three different reference
   designs*: "initial **uniform-density** field xₑ=V_f" (L241, §3.2) vs "initial
   **solid** domain" (L264, multi-mode) vs "fully **solid** design domain" (L508,
   building). Meanwhile the reported results were produced by the **`semi_harmonic`**
   path (`ρ_nodal(x)·ω₀·M₀·Φ₀` — `ω` not `ω²`, `M₀` solid not `M(x)`). exp4's
   headline CR2 evidence (the FD check and the A-vs-B ablation) validates the
   gradient of the **`semi_harmonic`** load actually in the code, *not* the
   `ω²M(x)Φ` load written in Eqs. 5–6. The documented (harmonic) load only appears
   in variants C/D — which are the adverse, unconverged pair. So the central
   code↔manuscript mismatch is still open, and the strongest validation in the
   package answers it for the undocumented formulation.

6. **CR1 is provably still open in the text.** `main.tex` L469 still asserts the
   Φ₂ gain "increases monotonically as α decreases: 1.99× (α=1) … 2.45× (α=0)" —
   silently **omitting the α=0.75 value (1.73×)** that breaks the monotone trend
   (the exact arithmetic both AI reviews flagged as Critical #1). L471 adds a
   mode-lifting narrative for the *Φ₁* ordering, but the *Φ₂* monotonicity claim
   is unchanged and unsupported by Table 3's own numbers. The experiment meant to
   re-derive a corrected Table 3 under the new mass-weighted MAC (exp2) is one of
   the two that failed, so there is no data to fix it.

---

## 3. What is genuinely solid

- **exp5 scaling (M4/mn8):** clean log-log fit, β = 0.90 (Olhoff), 0.91
  (Yuksel), 1.04 (Proposed), R² = 0.97–0.998. Decisively corrects the wrong
  "O(nₑ^1.3)" claim. Figure `exp5_scaling_loglog.png` produced. ✅
- **exp4 CR2 load-sensitivity validation (machinery):** FD gradient check passes
  (relerr ≈ 1e-7 on significant elements), the omitted-term ratio is computed
  analytically, and the A-vs-B ablation isolates exactly the load-sens term
  (+1.13 %). This is the most rigorous, well-controlled part of the package — but
  see §2.5: it validates the implemented `semi_harmonic` load, not the `ω²M(x)Φ`
  load documented in Eqs. 5–6, so it only closes CR2 once A0 is resolved. ⚠️
- **exp1 structure & exp6 hardware block:** the *design* (ω₁ column, std devs,
  separated setup time, hardware spec) matches MR1/MR2/R1/R2/C5 exactly — only
  the Olhoff convergence / numbers need fixing.
- **exp2b building:** reproduces Tables 4/5, applies the MAC threshold, runs the
  spurious-mode check, and exposes the high Φ₂ indices (B2/B3/M7). ✅

---

## 4. Manuscript-incorporation gap (`paper/main.tex`)

Even for experiments that succeeded, the data is largely **not yet written into
the manuscript**. Inspection of `main.tex` (modified Jun 19) finds:
present — Huang ×4 (M8), RAMP ×1 (B4), "in-house" ×1 (CR3), Svanberg ×3 (C4),
Heaviside ×5, spurious ×3; **absent** — any "finite difference" (CR2 validation),
"standard deviation"/`\pm` (MR1), "mesh convergence"/"refinement" (MR5),
"MAC threshold" (MR4), "init/setup time" (A2/R1), explicit "load sensitivity"
discussion (C9). There is **no response-to-reviewers letter** in the repo.

Beyond what is merely *missing*, two demanded fixes are **contradicted by text
that is still present**:
- **A0 / C2 (reference design + load):** still inconsistent — uniform `xₑ=V_f`
  at L241 vs "solid domain" at L264/L508; documented load `(ω⁽⁰⁾)²M(x)Φ`
  (L130/L239/L250) ≠ the `semi_harmonic` load used for the results.
- **CR1 (monotonicity):** L469 still claims the Φ₂ gain is monotone while dropping
  the α=0.75 = 1.73× counterexample.

---

## 5. Concrete remaining work to actually close the revision

1. **Fix and rerun exp2 (clamped beam).** It carries CR1 (the top critical
   issue), MR3, A0 (semi vs Eq.7), A5, B1/M5, and clamped M6/B5/B2. Determine
   the failure cause (it errors fast — likely in the solver call or MAC CSV
   path for this config), then regenerate `exp2_clamped_beam_results.mat`, the
   α=0.75 diagnostic, and the M5 mode plots. **Without this the revision cannot
   answer its #1 reviewer demand.**
2. **Fix and rerun exp3 (mesh convergence, MR5/V3).** Same — no data currently.
3. **Make the Olhoff comparator converge in exp1** (its own grayness<0.05 / a
   sane iteration cap), then reconcile the regenerated ω₁/timing with the paper's
   8.6 % gap and 7.1× speedup — or revise those headline claims to match.
4. **Diagnose the exp4 periodic-refresh collapse** (−62 %). Add mode-tracking on
   refresh, or reframe the result. Extend A4 to the planned SS-beam 400×50 with
   N ∈ {∞,50,10,5,1} rather than one 160×20 point.
5. **Confirm convergence** for exp4 (and exp2b) instead of hitting the iteration
   cap, so comparisons are between converged designs.
6. **Resolve A0/C2 in the text first (it gates everything else):** pick *one*
   load definition (`ω` reference-modal vs `ω²` physical inertial) and *one*
   reference design (uniform `xₑ=V_f` vs solid), then make Eqs. 5–7, L241/L264/L508,
   the JSON examples, and the code agree. Until this is fixed, exp4's CR2 evidence
   validates a load the paper doesn't actually claim.
7. **Correct the CR1 sentence (L469)** to match Table 3's real numbers (or the
   corrected exp2 Table 3 once it runs), and stop omitting the α=0.75 = 1.73×
   value; reframe contribution (iii) as MAC-tracked shape-stiffening where the
   sequence is non-monotone.
8. **Write the produced results into `main.tex`** (std devs, ω₁/mesh, setup time,
   FD validation, MAC threshold + invalidated building Φ₂ gains, scaling
   correction) and draft the response-to-reviewers letter.

**Bottom line:** `Revision_v1` is a well-structured scaffold that maps cleanly
onto the reviewer demands and *fully* nails two of them (load-sensitivity
validation CR2, scaling correction M4). But it does **not** currently fulfil the
demands completely: two experiments fail (one of them owning the top critical
issue), three of the experiments that run produce numbers that contradict the
paper or the method, and most results are not yet in the manuscript.

---

## 6. Validation of the proposed 9-phase completion plan

**Validated 2026-06-19.**
The plan is sound and correctly ordered. Every substantive problem in §§2–5 is
assigned to a specific phase; the gates are concrete and binary; and the
governing principle — *results must not be forced to support the original
conclusions* — is exactly what the current implementation violates.

The plan is accepted with **two wording corrections** and **six structural
additions** detailed below. These do not change the phase ordering or gates;
they sharpen what each phase must deliver.

### Wording corrections (mandatory — propagate into every script, comment, and paper sentence that touches these concepts)

**W1 — Omitted sensitivity ≠ negligible sensitivity.**
Do not write "approximately zero," "neglected term," or "ε-small." The load
sensitivity `∂f/∂xₑ` is **nonzero**; it is *dropped by assumption*. The paper's
contribution is precisely that discarding a nonzero term still yields good
designs. Any experiment, diary note, or manuscript sentence that softens this to
"small" or "approximately zero" misrepresents the approximation and understates
the CR2 demand. Correct phrasing: *"the load-sensitivity term is omitted from
the adjoint; its magnitude relative to the compliance term is measured a
posteriori."*

**W2 — N=1 refresh is not automatically the Yuksel–Yilmaz method.**
The N=1 variant (refresh every iteration) is a **fully-refreshed reference
design**, not the published Yuksel–Yilmaz scheme unless algorithmic equivalence
is demonstrated step-by-step. Yuksel–Yilmaz has a specific two-stage structure
(Stage 1: static-compliance topology; Stage 2: frequency maximisation from that
starting point) with its own convergence behaviour. Labelling N=1 as "Yuksel" in
scripts, tables, or text is a category error; label it "N=1 (fully refreshed)"
until equivalence is proved.

### Three-study distinction (structural — each study answers a different question and must not be conflated)

| Study | Variable | Question answered | Correct load |
|---|---|---|---|
| **CR2** (load-sensitivity ablation) | Omit vs retain `∂f/∂xₑ` | Does dropping the load-sensitivity term change the converged objective? | **Evolving** `M(x)` — the paper's Eqs. 5–6 |
| **A4** (eigenpair-refresh ablation) | Frozen vs N∈{∞,50,10,5,1} | Does stale eigenpair quality degrade the optimum? | Fixed at whichever N is being tested |
| **Comparator** (exp1/exp2) | Proposed vs Olhoff vs Yuksel | How does the proposed method rank against canonical algorithms? | Each algorithm uses its own documented load; Yuksel is its own method, not N=1 |

These three studies use overlapping machinery (eigensolve, adjoint, MMA/OC) but
measure orthogonal effects. Mixing results across studies — e.g. reporting the
A4 N=1 ω₁ against Olhoff's converged ω₁ as a "performance comparison" — is
invalid. Phases 3, 5, and 6 of the plan must keep them separate.

### Phase-by-phase assessment

**Phase 1 — Freeze the authoritative method.** Accepted with one addition.
The formula `f_j⁰ = (ω_j⁰)² M(x_ref) Φ_j⁰` is correct for the *constant-load*
variant, but cf. W1: for the CR2 study `M(x)` must evolve with the design; it is
only `x_ref` (the reference eigenpair snapshot point) that is frozen. The gate
must state ≤1e-8 relative tolerance on load-vector entries for the MATLAB/Python
cross-check. The MAC formula must be mass-weighted (physically correct for FE
modes); update Eq. 9 accordingly.

**Phase 2 — Repair experiment infrastructure.** No changes. "Capped/empty/NaN ⇒
nonzero exit status," per-experiment output directories, git hash in manifest,
and explicit artifact manifest are each necessary and sufficient to prevent the
silent-failure mode that killed exp2 and exp3.

**Phase 3 — Validate the corrected solver.** Accepted. Finite-difference check
(relerr ≤ 1e-5, central differences, h = 1e-6) on the **evolving-M(x)** load
objective closes CR2 machinery once Phase 1 is done. Note that the existing FD
check (relerr ≈ 1e-7 in `exp4_variant1_diary.txt`) validates the `semi_harmonic`
gradient — a different quantity; retain it as an internal regression test but do
not cite it as the CR2 result.

**Phase 4 — Recover exp2 and exp3.** Highest priority. Document the failure
cause in the manifest before rerunning. For the multi-mode comparator in exp2
(M3 / MR3), scope it to the α=1 single-mode case or define an explicit fair
comparator; do not compare the two-mode α-weighted objective directly against
Olhoff's single-mode result. The α=0.75 MAC diagnostic must appear in the
rerun; it is the specific output CR1 demanded.

**Phase 5 — Replace exp4 with a valid approximation study.** Accepted with one
correction. Do not remove the A-vs-B semi_harmonic ablation; after Phase 1
freezes the authoritative formulation, rerun A-vs-B for that formulation and
retain it as the CR2 validation — the FD check is the most rigorous work in the
package and should be preserved under the correct load. The N∈{1,5,10,50,∞}
refresh sweep must pin the SS-beam 400×50 mesh (A4's spec), confirm N=1 itself
converges, and apply MAC+frequency-continuity mode tracking to diagnose the
131.24 vs 49.84 collapse (almost certainly spurious-mode capture by the
periodic refresh — which would vindicate the frozen-mode approach). Per W2, the
N=1 column must be labelled "fully refreshed," not "Yuksel."

**Phase 6 — Rebuild exp1 performance evidence.** Accepted with one explicit risk
acknowledgement. Instrument timings *inside* each solver; require Olhoff
convergence (grayness < 0.05 or `Δω/ω < 1e-3`) before entering any speedup or
gap calculation. The 8.6 % accuracy-gap and 7.1× speedup claims may not survive
converged runs; if they cannot be reproduced, replace them with the measured
values labelled relative to the authors' modified implementation. The std-dev on
ω₁ is **timing-only variability** — the eigenfrequency objective of a
deterministic code does not vary between runs; the revision text must say so
explicitly (answers MR1's "why does a deterministic code have standard
deviation?"). Per W2, if N=1 results appear in exp1, label them accordingly.

**Phase 7 — Characterize low modes and grayness.** No changes. Element-wise
kinetic and strain-energy fraction in low-density regions is the correct
spurious-mode discriminator; MAC alone is insufficient. Repeat capped building
cases (α=1.0/0.75 hit 2000-iter cap) to convergence before characterizing their
mode spectra.

**Phase 8 — Revise the manuscript.** Accepted with two additions:
(a) Update Eq. 9 explicitly to the mass-weighted MAC formula — currently the
code uses mass-weighted MAC while Eq. 9 shows the Euclidean form; this is item
E3 in `final_review_V2.tex` and is called out in `exp2b_building.m` line 43.
(b) Add an explicit checklist for the C/mn minor corrections: Φ vs φ notation
(mn2), "quasi-static" non-standard wording (mn9), void-material penalty
(1e-9 vs 1e-6 inconsistency, R4), convergence tolerance (1e-3 vs 2e-3, R5/A4),
and the Huang et al. (M8) citation (already ×4, confirm placement is adequate).
Rewrite, not merely update, any narrative that depends on the 8.6 % / 7.1× /
monotonicity claims; expect structural prose changes in §§4–6.

**Phase 9 — Reproducibility package.** No changes. Manifest, immutable release
tag, checksums, machine-readable completion report, and supplementary-material
inventory close R1–R4, C8, mn4–6, and Reviewer 3 comment 13.

### Experiment-tier classification (against a time crunch)

| Tier | Experiments | Gate for resubmission? |
|---|---|---|
| **Blocking** | exp2 (clamped beam, CR1+A0), exp3 (MR5), exp1 convergence (A2/A6), A-vs-B under corrected load (CR2), A4 refresh sweep (A4/MR6) | Yes — cannot resubmit without these |
| **Strong-to-have** | Phase 7 low-mode characterization (M7/B1–B6), mode-tracking fix for periodic refresh | Yes if building caps not resolved otherwise |
| **Nice-to-have** | A5 a-posteriori mode check commentary, Heaviside filter demo (B5) | No — discuss in limitations if omitted |

A capped or partial result must **not** be re-inserted to make a deadline.
Phase 2's infrastructure explicitly prevents this.

### Acceptance criteria (unchanged from the plan)

The revision is complete when all of the following hold:
1. exp2 and exp3 produce non-empty, non-capped results.
2. Every numerical claim in the manuscript traces to a convergence-verified artifact.
3. CR2 is validated for the evolving-M(x) load (W1 correction applied throughout).
4. The N=1 column, if present, is labelled "fully refreshed" not "Yuksel" (W2).
5. The 8.6 % / 7.1× / monotonicity claims are either reproduced by converged data
   or replaced by the actual measured values with honest labelling.
6. Adverse findings (exp4 periodic collapse, MAC-below-threshold Φ₂ gains) are
   reported, diagnosed, and discussed — not omitted.
7. A fresh independent audit finds no remaining ❌ or ⚠️ in §1's coverage matrix.
