# Second-Round Independent Scientific Review of Revision V2

Report reviewed: `faithful_reconstruction_report_v2.md`  
Round 1 review: `independent_scientific_audit_phase_b.md`

## Overall assessment

Revision V2 has addressed the central scientific problem identified in Round 1.
The report now distinguishes the paper-explicit formulation, the tested
reconstruction, and the undocumented historical implementation; withdraws the
claim that contraction is demonstrated; treats fail-closed acceptance as an
audit convention rather than an optimality theorem; removes the causal ranking
of the residual frequency gap; limits the LP diagnosis to the initial
simple-eigenvalue state; and replaces “exact limit cycle” by finite-budget,
period-2-like behavior. The convergence and first-step-collapse evidence remain
strong.

Within its revised scope, the main scientific conclusions are supported:

1. the exact full-box LP update at the initial uniform design collapses the
   reconstructed clamped–clamped structure on both tested meshes;
2. this initial collapse is not an artefact of stopping MMA at 30 iterations;
3. none of the 19 tested trajectories reaches the declared design-convergence
   criterion within its observed budget; and
4. the tested reconstruction does not retain a clustered lowest eigenvalue.

The report still contains a few local statements that are stronger than its new
methodological qualifications. They do not overturn the conclusions above and
do not require new experiments, but they should be corrected before the report
is treated as final.

## Disposition of every Round 1 issue

The labels below follow Appendix B of Revision V2. Where Round 1 repeated the
same underlying issue in its critical, major, internal-consistency, and minor
lists, the later entry cross-refers to the first disposition rather than
repeating the argument.

### Critical findings

| Round 1 issue | Classification | Evaluation |
|---|---|---|
| C-1 — Verdict conflated the tested reconstruction with the historical Du–Olhoff implementation | **Partially resolved** | Sections 0.0, C6, 9.1, and 11 now make the required scientific distinction and explicitly disclaim a verdict on the undocumented 2007 implementation. The principal verdict is appropriately scoped. The report title, however, remains “Faithful Reconstruction of the Du & Olhoff (2007) Optimization Procedure,” which still implies established historical fidelity despite 18 acknowledged reconstruction choices. This is now a framing defect rather than a defect in the body’s central inference. |
| C-1b — “Mesh-independent” was unsupported by two meshes | **Resolved** | V2 consistently uses “both tested meshes” or equivalent language and expressly says that two meshes do not establish general mesh independence. |
| C-2a — A move limit is an upper bound, not the “smallest step” | **Resolved** | The erroneous lower-bound argument is explicitly withdrawn. V2 reports the observed saturation of the cap, which is the quantity actually measured. |
| C-2b — Basin width and an “order 10” contraction factor were not measured | **Resolved** | Both claims are withdrawn. No numerical contraction factor or inferred basin width remains. |
| C-2c — Contraction was untested | **Resolved** | Contraction is now graded [C], expressly described as unimplemented and untested, and separated from the demonstrated failure of the fixed 0.2 restriction. |
| C-2d — Regime B changed `move_lim`, `outer_move`, and `alpha` together | **Partially resolved** | V2 correctly acknowledges that the trajectory comparison is confounded, and the fixed-direction \(t\)-profile genuinely isolates scalar step length at the first iteration. However, the new claim that the inner-budget transition is a second “clean isolation” because “only the realised step length differs” is not justified: changing the number of MMA iterations changes the approximate iterate’s direction and component distribution as well as its norm, a fact V2 itself acknowledges in §5.3. This second line is corroborative, not isolating. The first-step overshoot conclusion remains supported by the \(t\)-profile. |
| C-3a — Fail-closed acceptance and its successive-iterate test were treated as mathematical validity | **Partially resolved** | The new definitions and §4.1 caveats correctly state that the test is not a KKT, dual, or optimality certificate and that an inexact inner step may be legitimate. Several later statements nevertheless say that a failed test means the configuration “does not solve” Eq. (25), that its increments “do not solve” the subproblem, or that the gate rejects “invalid steps” (§§2.2, 4.1–4.3, 5.3, C2). Those categorical claims contradict the report’s correct statement that failure of the test does not prove non-solution. The evidence supports “did not meet the declared stopping test” and “cannot be certified as a solution,” not “is not a solution.” |
| C-3b — The 9% V4/VR difference was causally attributed to invalidity | **Resolved** | V2 now treats truncation as a materially different effective update, states that both terminal values are non-converged, and declines to rank either as closer to a fixed point. |
| C-4a — Multiplicity alternatives were not excluded | **Partially resolved** | V2 now gives the appropriately bounded conclusion: the implemented forced-\(N=2\) model under the existing controls did not retain the tested cluster; detection alone is not a sufficient explanation; step length is only a supported contributor; and the unverified \(N=2\) formulation, \(\bar\lambda\) substitution, modal window, tolerance-path effects, and sample size remain open. That resolves the main causal overreach. One sentence in §6.3 still says that low individual MAC values show that both steps “destroy mode identity,” despite the adjacent, correct concession that individual MAC is non-diagnostic near multiplicity. That sentence should be removed or qualified. |
| C-4b — Individual-mode MAC is fragile near a multiple eigenvalue | **Partially resolved** | V2 adds the basis-rotation caveat and correctly rests non-retention on the basis-invariant eigengap. The residual “destroy mode identity (MAC ≈ 0.01)” sentence is still unsupported without a subspace diagnostic. It is not needed for the eigengap-based conclusion. |
| C-5 — The residual frequency gap could not be causally decomposed | **Resolved** | The causal ranking is withdrawn. C5 now states only that the terminal values are not optima, the initial scalar frequency agrees, bimodality is not retained, and path-level reconstruction discrepancies remain possible. |
| C-6 — “Exact period-2 limit cycle” exceeded the finite-tail evidence | **Resolved** | V2 consistently uses “period-2-like oscillation over the observed tail,” restricts the detailed correlations to the run for which they were computed, and explicitly explains why exact and asymptotic language is unavailable. |

### Internal-consistency findings

| Round 1 issue | Classification | Evaluation |
|---|---|---|
| I-1 — Two executive-summary oscillation frequencies were wrong/omitted | **Resolved** | All four V4/VR values are now reported correctly. |
| I-2 — The \(-0.982\) correlation was generalized to four runs | **Resolved** | It is now restricted to V4 at 160 × 20; the four-run statement uses identically computed lag ratios. |
| I-3 — “End in a mechanism” conflicted with V5a’s recovered terminal state | **Resolved** | V2 distinguishes a trajectory-minimum classifier from a terminal state and reports V5a’s final 354.49 rad/s value. |
| I-4 — “Every budget from 20 to 2000” overstated sampled budgets | **Resolved** | The text now says “every tested budget from 20 upward,” enumerates them, and discloses the non-collapsing 1–10 cases. |
| I-5 — “Every qualitative behavior transfers” exceeded the paired mesh design | **Resolved** | The transfer claim is limited to V0, V4, V5, and VR, the four variants run at both meshes. |
| I-6 — Singular-warning prose conflicted with seven warnings for V4 at 240 × 30 | **Resolved** | The complete counts are reconciled and the causal conclusion is stated at the supported level. |
| I-7 — “All production progress ever made” exceeded documented executions | **Partially resolved** | Section 4.3 is correctly narrowed to documented configurations. C2 nevertheless retains “Every design update the production solver has ever applied to this benchmark,” which reintroduces the same unsupported universal claim. It should say “every documented update examined in this and the preceding campaign.” |
| I-8 — Summed gate totals were treated as a scientific scale | **Resolved** | V2 explicitly withdraws score-based inference and interprets the gates individually. Retaining the raw-count column is unnecessary and conflicts editorially with the heading “Why gate totals are not reported,” but no conclusion now relies on it. |

### Major criticisms

| Round 1 issue | Classification | Evaluation |
|---|---|---|
| M-1 — Exact/asymptotic cycling was not shown for all four runs | **Resolved** | See C-6 and I-2. |
| M-2 — Regime B bundled three controls | **Partially resolved** | See C-2d. The confounding is now admitted, but the budget sweep is still incorrectly described as changing step length alone. |
| M-3 — The initial exact-LP result was generalized to every outer iteration | **Resolved** | V2 limits the result to the initial \(N=1\), inactive-J state and explicitly withdraws the global 0/1-vertex claim. |
| M-4 — Individual MAC was used as decisive evidence near multiplicity | **Partially resolved** | See C-4b. The eigengap now carries the conclusion, but one unsupported mode-identity statement remains. |
| M-5 — The forced-\(N=2\) implementation was not independently verified | **Resolved** | This is now expressly listed as an open alternative; behavioral failure is no longer presented as validation. |
| M-6 — Three schedules did not eliminate continuation ambiguity | **Resolved** | V2 scopes its findings to the three ladders/stage lengths and names trigger, stage-convergence, and state-transfer semantics as untested. |
| M-7 — The classifier conflated transient mechanism entry with terminal collapse | **Resolved** | The classifier’s semantics, p-dependent threshold, and V5a exception are disclosed, and no central conclusion rests on the marginal V5/V5a label. |
| M-8 — G5 included a subjective limb and gate totals mixed unlike criteria | **Resolved** | The implemented objective G5 rule is now stated accurately, and aggregate scores are not used inferentially. |
| M-9 — Spectral validity was a final-state gate, not a trajectory property | **Resolved** | G4 is explicitly scoped to the final design and the prose uses “spectrally valid final design.” |
| M-10 — Initial-frequency agreement did not exclude optimization-path discrepancies | **Resolved** | C5 now makes exactly that limitation. |
| M-11 — Historical production claims exceeded the executions examined | **Partially resolved** | See I-7; one universal sentence remains in C2. |
| M-12 — The V4 singular-warning inconsistency required reconciliation | **Resolved** | See I-6. |

### Minor criticisms

| Round 1 issue | Classification | Evaluation |
|---|---|---|
| m-1 — Correct the four oscillation frequencies | **Resolved** | Corrected. |
| m-2 — Qualify “continuation harmful” | **Resolved** | The claim is limited to the tested reconstructed schedules and controls. |
| m-3 — Use “every tested budget” | **Resolved** | Corrected. |
| m-4 — Use “consistent on both tested meshes” | **Resolved** | Corrected. |
| m-5 — Distinguish the 160 × 20 near-cluster from a detected \(N=2\) state | **Resolved** | V2 makes the distinction and identifies the sole detected non-mechanism \(N=2\) event at 240 × 30. |
| m-6 — Define “structural component” versus raw component count | **Resolved** | The thresholded `n_members` definition and its relation to raw 8-connected counts are now explicit. |
| m-7 — Define prediction-ratio handling near a zero denominator | **Resolved** | The `NaN` guard and the fact that it was never activated are reported. |
| m-8 — Clarify actual versus counterfactual rejected steps | **Resolved** | The distinction is explicit in §5.2. |
| m-9 — Do not use a summed gate score | **Resolved** | The scores no longer carry inferential weight. |
| m-10 — Relabel conclusion previews | **Resolved** | `[CONC-preview]` is defined as a forward reference to a graded conclusion. |

### Other substantive Round 1 qualifications

| Round 1 issue | Classification | Evaluation |
|---|---|---|
| Finite trajectories cannot establish non-convergence for unlimited iteration count | **Resolved** | V2 consistently scopes non-convergence to observed budgets and reserves its evidence for rejecting convergence of the reported runs. |
| “Random walk” conflicted with a strong period-2 component | **Resolved** | V2 replaces both “random walk” and “exact cycle” by a decomposition into a dominant alternating component and a non-repeating residual. |
| Continuation probes from uniform states were not full continuation trajectories | **Resolved** | V2 separates the per-\(p\) probes from the three full reconstructed schedules and limits each inference accordingly. |
| C7 appeared to prohibit every reviewer-facing use of the directory | **Resolved** | V2 distinguishes forbidden convergence/optimality claims from usable forward-model evidence, qualified topology observations, and diagnostic findings. |

## Remaining issues

### Critical (publication-blocking)

None.

### Major

None. No remaining issue overturns the first-step LP result, the rejection of
convergence for the reported runs, or the failure to retain bimodality.

### Minor

1. **Make the inner-solve language consistent with the stated evidence.** Replace
   “invalid step,” “does not solve Eq. (25),” and “increments do not solve the
   subproblem” by “did not meet the declared stopping test” or “cannot be
   certified as a solution of Eq. (25).” Also replace “mathematically valid” in
   the decisive question with the defined term “meeting the declared
   inner-solve stopping test.” A failed successive-iterate test is not proof of
   non-optimality.

2. **Do not call the inner-budget sweep an isolation of step length.** More MMA
   iterations alter the full approximate increment, not only its magnitude.
   Retain it as corroborative evidence and rely on the fixed-direction
   \(t\)-profile as the actual isolation experiment. Correspondingly, phrase the
   [B] claim as an additional step restriction being supported as necessary to
   avoid the demonstrated first-step overshoot, rather than as an unqualified
   universal requirement for any viable trajectory.

3. **Remove the remaining individual-MAC inference.** In §6.3, “both destroy
   mode identity (MAC ≈ 0.01)” is not supported near multiplicity. The
   basis-invariant eigengap already establishes non-retention and is sufficient.

4. **Close the remaining scope leaks.** Narrow the main title to the tested
   full-box reconstruction or printed formulation, and replace the C2 phrase
   “every design update the production solver has ever applied” with “every
   documented update examined.” These changes would make the title and C2
   consistent with §§0.0, 4.3, C6, and 11.

### Editorial

1. Remove the raw `x/8` gate-count column. Although it is now labelled “not a
   score,” retaining it serves no scientific purpose and conflicts with
   §8.3’s statement that gate totals are not reported.
2. Remove the duplicated `## 5. Phase 5` and `## 7. Phase 7` headings.

## Material effect of the remaining issues

The remaining issues do **not** materially affect the report’s supported
scientific conclusions. The fixed-direction profile and exact LP solve support
the initial full-step overshoot without the inner-budget “isolation” claim. The
eigengap supports cluster non-retention without individual MAC. The absence of a
KKT certificate affects how the budget-30 inner increments may be described,
but not the exact-LP collapse, the measured failure of the declared stopping
test, or the observed lack of outer convergence.

Revision V2 is therefore scientifically supported **as a negative result about
the tested full-box reconstruction and the tested bounded variants**, not as a
reconstruction of every undocumented detail of the historical 2007
implementation.

## Recommendation

**Accept with minor revision**

The central evidential overreach identified in Round 1 has been corrected. The
remaining changes are local qualifications and consistency edits; they require
no new computation and do not alter the supported conclusions.

## Summary for the Authors

Revision V2 is a substantial and successful response to Round 1. The most
important improvement is the explicit separation of the paper-explicit
formulation, the tested reconstruction, and the undocumented historical
implementation. The final verdict is now appropriately confined to the tested
full-box reconstruction. The report also correctly withdraws the untested
contraction factor, treats contraction as speculative, limits the LP diagnosis
to the initial \(N=1\) state, removes the causal ranking of the residual
frequency gap, acknowledges unresolved \(N=2\) and continuation choices, and
replaces “exact limit cycle” with a finite-budget period-2-like description.
The corrected gate, mechanism, mesh-transfer, warning-count, and topology
language materially improves internal consistency.

Four small revisions remain. First, failure of the chosen successive-iterate
test must not be restated as proof that an increment is not a solution of
Eq. (25); use “did not meet the declared stopping test” or “was not certified.”
Second, the inner-budget sweep is corroborative but does not isolate step length,
because the approximate MMA direction can change with iteration count; the
fixed-direction \(t\)-profile is the valid isolation experiment. Third, remove
the statement that individual MAC proves destruction of mode identity near
multiplicity; the eigengap already supports non-retention. Fourth, narrow the
main title and the remaining “ever applied” sentence to the documented tested
scope. The raw gate-count column and duplicate headings should also be removed.

Subject to these local edits, the reconstruction report can be considered
scientifically reliable for its stated negative conclusions: the tested
full-box initial update collapses, the tested trajectories do not converge
within their budgets, and persistent bimodality is not reproduced. It does not
establish how the undocumented historical implementation behaved.
