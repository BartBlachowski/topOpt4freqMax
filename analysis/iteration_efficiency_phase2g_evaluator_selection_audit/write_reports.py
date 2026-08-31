#!/usr/bin/env python3
"""Render Phase-2G narrative deliverables from the reduced evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def write(name, text):
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


S = json.loads((OUT / "audit_summary.json").read_text())
A = S["adaptive"]; B = S["binary"]; SM = S["smoothness_pooled"]; BC = S["boundary_counts"]
mac = rows("MODE_IDENTITY_AUDIT.csv")
hardmac = [r for r in mac if r.get("hard_gate_pass") == "True"]
precision = rows("PRECISION_PAIR_AUDIT.csv")
erole = rows("EVALUATOR_ROLE.csv")
e23max = max(float(r["E2_E3_max_relative_difference"]) for r in erole)


write("CANDIDATE_DEFINITIONS_VERIFIED.md", r"""
# Verified candidate definitions

This reconstruction comes from the frozen MATLAB evaluator, the Phase-2F modal engine,
the completed survey archive, and the exact-count topology implementation. It does not
substitute the prompt's shorthand for executable semantics.

## A — frozen evaluator

- Field: actual clipped gray density.
- Stiffness: E1 `1e7(1e-6+(1-1e-6)x^3)`; E2
  `1e7(1e-9+(1-1e-9)x^3)`; E3 `1e7 max(x,1e-3)^3`.
- Mass: E1 linear with its floor; E2/E3 Du–Olhoff Eq.(4), `g=x^6` at
  `x<=0.1`, otherwise `g=x` (with E2's floor and E3's effective-density clamp).
- Solver/selection: MATLAB `eigs`, three modes, algebraically lowest mode; no
  eigenvector diagnostic, escalation, or modal-invalid status.
- Hard gate: separate exact-count connectivity/island gate; it does not validate a mode.

## B — continuous mass, algebraic-lowest

- Field and stiffness: same gray field and E1/E2/E3 stiffness conventions.
- Mass: E1 unchanged; E2/E3 continuous Eq.(4a), `g=1e5*x^6` at `x<=0.1`,
  otherwise `g=x`.
- Selection: algebraically lowest mode, with no validity rule. Phase-2F computed 12
  modes diagnostically, but extra modes do not change B's definition.
- Failure semantics: no explicit modal-invalid result.

## C — adaptive lowest valid structural mode

- Field/stiffness/mass: actual gray field; E1 unchanged; E2/E3 Eq.(4a); E3 diagnostics
  use `rho_eff=max(x,1e-3)` consistently with its pencil.
- For every mode require all three physical conditions:
  1. `voidKE(rho_eff<=0.1) < 0.5`;
  2. `voidSE(rho_eff<=0.1) < 0.5`;
  3. density-weighted kinetic participation `sum(KE_n*rho_eff) > 0.5`.
- A mode is structurally valid only when all three conditions hold. Select the
  algebraically lowest valid mode. IPR is stored as a nonbinding localization cross-check;
  it has no mesh-invariant natural cutoff.
- Adaptive schedule: `3 -> 6 -> 12 -> 24 -> 48 -> ...`; continue until a valid mode is
  found or the eigensolver/resource limit is reached. A limit returns
  `STRUCTURAL_MODE_NOT_FOUND`; the highest computed mode is never substituted.
- Hard gate: unchanged and logically separate. Both modal evaluation and the pointwise
  gate must succeed.

The Phase-2F survey's operational shortcut used `voidKE<0.5` alone. This audit found 24
selected modes contradicted by both strain energy and density participation. The verified
Candidate C is the already-investigated three-diagnostic concept above, not that shortcut.

## D — exact-count binary

- Field: stable descending-density projection with increasing-index tie break and exactly
  `round(0.5*N)` solid elements.
- E1/E2/E3 pencil: same conventions evaluated on the binary field (Phase-2F used six modes).
- Selection: algebraically lowest binary mode; no adaptive/modal validity rule.
- Hard gate: applied to the same binary topology, but checks connectivity and component
  area rather than dynamic adequacy.
""")


write("METHOD_NEUTRALITY.md", f"""
# Method neutrality

The selected C rule contains no method label, optimizer constant, rank result, or native
objective. Its inputs are only the density field, the common evaluator's own effective
density/interpolation, and eigenpairs. Repeating the scorecard with method labels hidden
therefore selects C for the same reasons: continuity, actual-gray fidelity, joint modal
validity, and threshold robustness. No Proposed ranking was available to improve.

Mechanism neutrality is established by construction. Empirical cross-method neutrality is
not established: every full stored density trajectory is Olhoff LP. Proposed and Yuksel
artifacts retain scalar/final evidence but not trajectory density fields. Different methods
may produce different numbers of rejected modes because their designs differ; that is not
bias unless the classifier becomes ambiguous or threshold-sensitive. On Olhoff, all
{S['survey_state_evaluator_records']:,} selected state/evaluator records were unanimous on
the three diagnostics, but this must be repeated on Proposed and Yuksel histories before
production claims of demonstrated cross-method neutrality.

Candidate D is label-blind in code but can correlate with method behavior through cutoff
ties, quantization, and rate of binarization. That is a stronger neutrality risk than C's
physical rejection count.
""")


write("REFERENCE_PERSISTENCE_IMPACT.md", f"""
# Reference and persistence impact

Within the available survey, the selected C frequency is invariant throughout the exact
threshold plateau: density partition `tau=0.02..0.50` with nearby simultaneous-condition cuts (the
intersection contains `0.49..0.54`; at `tau=0.1`, `0.48..0.56`). Thus the available C
quality sequences do not change within that plateau.

Formal `b_ref`, `B_meas`, `k_enter`, and `k_cert` cannot be recomputed. The reference rule
requires a separate density trajectory through `B_ref=3200`; the longest stored production
density history has 1,601 snapshots and the reference-length artifacts retain quality
arrays, not densities. Therefore:

- sensitivity of `b_ref`: **not exercisable**, not zero;
- sensitivity of `k_enter`: **not exercisable**, not zero;
- sensitivity of `k_cert`: **not exercisable**, not zero;
- `B_meas`: cannot be regenerated because it depends on the unavailable C-based `b_ref`.

Once C is implemented, unresolved modal states must make `Q_e(k)` unavailable and cannot
contribute to a valid sustained floor or acceptance window. They must never be imputed.
The formulas, `P`, `B_ref`, `B_meas`, q levels, and topology gates otherwise remain unchanged.
""")


write("COMPUTATIONAL_COST.md", f"""
# Computational cost

Phase-2F measured sparse shift-invert solves of 0.07 s at 160x20 and 2.97 s at 720x90 for
12 modes, with scaling approximately `T proportional N_el^1.264`. At 160x20, requesting
3/6/12/24/48 modes cost 0.0465/0.0550/0.0663/0.0903/0.1557 s. Eigenvector energy reduction
added 9.3% at 160x20 and 2.9% at 720x90; 12 eigenvectors used 12.6 MB at 720x90.

Only {A['gt3']} of {A['records']:,} records required more than three modes; {A['gt6']}
required more than six, and five required the 12-to-24 escalation. A geometric schedule is
therefore practical. Reusing a factorization/subspace is an implementation optimization,
not a scientific rule.

The Phase-2F estimate is about 30 single-threaded hours for three methods x three evaluators
over the eight available meshes at 1,600 states. Extrapolating the measured scaling to the
missing 800x100 mesh gives roughly 45 hours for all nine meshes, or order 6 hours with
eight-way state/mesh parallelism. This remains practical post hoc. All evaluator work stays
outside native optimizer timing.
""")


maxprec = max(float(r["selected_omega_relative_change"]) for r in precision)
write("PRECISION_IMPACT.md", f"""
# Precision impact

Stored paired evidence covers {len(precision)} evaluator/state records (24x4 and 96x12,
double versus float32 images). Under C, selected ordinals changed in 0 records, hard-gate
decisions changed in 0 records, and the maximum selected-frequency relative change was
{maxprec:.3e}.

This is encouraging but not a qualification. The paired samples do not contain the complete
3,200-state density sequence needed to verify `b_ref`, `B_meas`, acceptance, `k_enter`, or
`k_cert`. A fresh post-refreeze qualification is required and must bind its artifact to the
new evaluator and contract hashes.

The Phase-2B single-versus-double negative result remains historically valid for the
Eq.(4) frozen evaluator and its discontinuity mechanism. C does not retroactively invalidate
that experiment, and it does not prove the downstream C decisions precision-invariant.
""")


write("HARNESS_IMPACT.md", r"""
# Final-harness impact

Nothing was edited in this audit. A later controlled implementation must:

1. add the C modal classifier, adaptive batches, eigenvector diagnostics, residual checks,
   selected ordinal/frequency, vote margins, IPR, requested-mode counts, and explicit
   `STRUCTURAL_MODE_NOT_FOUND` status;
2. route C-selected E1/E2/E3 values into `Q`, reference, persistence, and tables while
   leaving the hard topology gate separate and unchanged;
3. keep evaluator wall time outside optimizer timing and report it separately;
4. extend regression tests, negative controls, provenance hashes, and precision-artifact
   binding;
5. retain E1/E2/E3 and robust normalized minimum; do not simplify the family in this phase;
6. support `olhoff.variant = 'lp' | 'mma' | 'both'`.

Olhoff accounting remains route-specific:

- LP (principal): outer updates and LP calls, separately; one successful LP call per outer
  in the settled route. Do not call simplex/HiGHS internals optimizer iterations.
- Nested MMA (secondary): outer updates; total MMA inner iterations; mean, median, and p95
  inner iterations; cap hits; and converged-inner fraction. Never fold these into a
  fictitious universal iteration count.

The LP/MMA selection is unchanged: LP is the principal comparator and nested MMA is the
secondary paper-literal sensitivity variant.
""")


write("METHODOLOGY_DELTA_MAP.md", r"""
# Collateral methodology delta map

| artifact | category | required consequence |
|---|---|---|
| `analysis/three_method_parametric_study/study_evaluate_design.m` | evaluator-definition | implement Eq.(4a), eigenvectors, three-condition intersection, adaptive selection, fail-closed status |
| `QUALITY_EFFORT_SPEC.md` | evaluator-definition/reporting | replace algebraic mode 1 by C; state loss of E2/E3 native identity and retained family |
| `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` | evaluator-definition/reference | define C, unresolved semantics, and unchanged persistence formulas |
| `iteration_efficiency_contract.json` | evaluator-definition | update evaluator definitions, source hash, status, adaptive/failure contract |
| `+ie2a/evaluate_common.m`, `analyze_trajectory.m`, reference/persistence integration | reference/persistence consequence | consume C-selected values and propagate missing status fail-closed |
| `+ie2a/production_preflight.m` | precision consequence | bind qualification to evaluator/contract hashes and require new qualification |
| `+ie2a/frozen_contract_sha256.m`, implementation provenance | evaluator-definition | update hashes only in the controlled refreeze |
| `FAIRNESS_RISK_REGISTER.md` | reporting consequence | record modal-classification and missing cross-method evidence risks |
| `PHASE2_FINAL_READINESS.md`, `METHODOLOGY_FREEZE_RECORD.md` | reporting/refreeze | replace stale Eq.(4) language and record amendment |
| endpoint tables/figures/status taxonomy | reporting consequence | selected ordinal, diagnostics, escalation, unresolved status, evaluator cost |
| timing specification/replays | timing consequence | explicitly exclude C post-processing; report overhead separately |
| Olhoff/Yuksel/Proposed native optimizers and profiles | no change | remain byte-identical |
| `P`, `B_ref`, `B_meas` formula, q levels, topology gates | no change | preserve exactly |
| Olhoff route ruling | no change | LP principal; nested MMA secondary |

Historical audit and implementation reports are not rewritten; they remain records of the
instrument in force when produced.
""")


findings = [
    ["G01","WP3","CRITICAL","A","Eq.(4) has a finite factor-1e5 branch jump at rho=0.1; prior independent perturbations reach 2.6736%.","REJECT_A"],
    ["G02","WP3","MAJOR","A","Six of 9,668 algebraic-lowest records fail unanimous modal validity; all six are recovered by mode 2, so fixed three covers this A survey but is not universally qualified.","DISCONTINUITY_AND_MODAL_FAILURE_REJECT_A"],
    ["G03","WP4","CRITICAL","B","Eq.(4a) is continuous, but 623/16,536 algebraic-lowest records are rejected by all three diagnostics and 20 more have diagnostic disagreement; all eight meshes are affected.","REJECT_B"],
    ["G04","WP5-7","MAJOR","C","The Phase-2F voidKE-only shortcut accepted 24 modes contradicted by both voidSE and density participation.","DO_NOT_ADOPT_SHORTCUT"],
    ["G05","WP5-7","PASS","C","The exact unanimous three-condition rule resolves all 16,536 records and has a broad tau/cut plateau.","ADOPT_RULE"],
    ["G06","WP8-9","PASS","C","Maximum ordinal 18; five 12-to-24 escalations; zero unresolved after escalation.","ADAPTIVE_FAIL_CLOSED"],
    ["G07","WP10-11","PASS","C","160x20 MAC shows smooth crossing behavior; pooled hard-gate step max 1.070%, none above 2% after the joint rule.","STABLE_OBSERVED"],
    ["G08","WP12","PASS","D","Binary mode-1 voidKE is zero to numerical scale (E2 maximum 1.85e-8).","ACKNOWLEDGE_ADVANTAGE"],
    ["G09","WP13-15","CRITICAL","D","667/2,704 hard-gate-passing states are below half gray structural frequency.","REJECT_D_PRIMARY"],
    ["G10","WP13-14","MAJOR","D","At 400x50 k=833, mode 1 lives almost entirely on permitted detached solid islands, not the spanning component.","PROJECTION_ARTIFACT"],
    ["G11","WP16/27","LIMITATION","C","Rule is method-blind, but full density trajectories exist only for Olhoff.","CROSS_METHOD_PREPRODUCTION_CHECK"],
    ["G12","WP17","NOTE","C","E2/E3 are effectively redundant after C (maximum observed relative difference 5.37e-4), but simplification is out of scope.","RETAIN_ALL_THREE"],
    ["G13","WP18","LIMITATION","C","No reference-length density history exists; b_ref/k_enter/k_cert sensitivity is not exercisable.","DO_NOT_FABRICATE"],
    ["G14","WP24","MAJOR","C","Stored precision pairs pass selected-mode and gate checks, but downstream reference decisions remain unqualified.","FRESH_QUALIFICATION"],
    ["G15","WP19","PASS","C","Adaptive post-hoc cost is practical and outside optimizer timing.","NOT_A_DISCRIMINATOR"],
    ["G16","WP22","DECISION","C","C wins the method-blind scorecard 47/60 and is the only candidate passing mandatory validity gates.","SELECT_C"],
]
with (OUT / "FINDINGS.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","work_package","severity","candidate","finding","disposition"]); w.writerows(findings)


write("EXECUTIVE_VERDICT.md", f"""
# Executive verdict

**Select Candidate C: adaptive lowest-valid-structural-mode evaluation of the actual gray
design.** It wins the unweighted method-blind scorecard (47/60) and is the only candidate
that passes continuity, intended-estimand, modal-validity, threshold-plateau, and practical
cost gates.

The selection is narrower than “use Phase-2F C.” The voidKE-only survey shortcut is not
acceptable; it selected 24 modes contradicted by the two other energy/participation
diagnostics. The accepted rule is the exact unanimous three-condition rule in
`CANDIDATE_DEFINITIONS_VERIFIED.md`. All {S['survey_state_evaluator_records']:,} ultimately
selected modes are unanimous; the threshold plateau spans `tau=0.02..0.50` and nearby
simultaneous diagnostic cuts around 0.5.

A is rejected by its rho=0.1 discontinuity and six invalid algebraic-lowest records. B fixes
only the discontinuity: 623 algebraic-lowest records fail all three diagnostics and 20 more
have disagreement, so 643 fail the unanimous rule. D removes void-kinetic modes but changes the estimand and fails
severely in {B['severe_binary_lt_half_gray']}/{B['hard_gate_passing_states']} hard-gate-pass
states. At 400x50 k=833 its 4.67 mode is a permitted detached-solid-island mechanism, not
the 170.93 spanning gray structure.

This audit supports a separate controlled methodology refreeze to C. It performs no
refreeze and authorizes no production. Missing Proposed/Yuksel density histories and a
fresh reference-length precision qualification remain pre-production obligations.
""")


answers = f"""
# Phase-2G final independent audit

## Decision basis

The audit reconstructed all candidates from code, consumed the completed 15 MB Phase-2F
survey, independently reproduced seven anchor roles plus the worst D ratio, used dense
LAPACK at 160x20 k=252 (maximum sparse/dense difference 3.19e-8), replayed 708 precision
records, and kept every write inside this directory. Protected-source checks have zero
mismatches.

Candidate C is selected with the exact unanimous three-diagnostic rule. The important
falsification result is that Phase-2F's voidKE-only shortcut is not the selected rule: it
misclassifies 24 modes, including the 240x30 k=594 step outlier. Adding the already-computed
strain-energy and density-participation conditions removes those false selections without tuning;
all selected modes become unanimous and the threshold plateau is broad.

## Required final summary

1. **Verified A.** Actual gray field; frozen E1/E2/E3 stiffness; E1 linear mass and E2/E3 discontinuous Eq.(4); three `eigs` modes; algebraic lowest; no classification/escalation.
2. **Verified B.** Actual gray field; same stiffness; E1 unchanged and E2/E3 continuous Eq.(4a); algebraic lowest; no modal validity.
3. **Verified C.** Actual gray field and B mass laws; lowest mode satisfying all three physical diagnostic conditions; adaptive extraction; fail closed.
4. **Verified D.** Stable exact-count binary projection; binary E1/E2/E3 pencil; algebraic lowest; separate hard gate.
5. **Is A continuous/stable at rho=0.1?** No. E2/E3 Eq.(4) jump by factor 1e5 in mass value; observed perturbations reach 2.6736%.
6. **Is A's fixed mode count adequate?** Three modes cover this surveyed A/Olhoff set: 6/9,668 algebraic-lowest modes fail unanimous validity, but every case is recovered at mode 2. That is not cross-method/universal proof, and A is independently rejected by continuity.
7. **Does B remove the discontinuity?** Yes; Eq.(4a) is value-continuous at 0.1.
8. **Does B eliminate artificial low modes?** No. In 623/16,536 evaluator records all three diagnostics reject the algebraic-lowest mode; 20 additional records have diagnostic disagreement. Thus 643 algebraic-lowest modes fail the adopted unanimous validity rule; all eight meshes are affected.
9. **Do C populations separate cleanly?** The jointly corroborated extremes do: pooled structural voidKE ends at 0.4850 and all-three-artificial voidKE begins at 0.5007. Diagnostic-disagreement modes bridge scalar populations, so they are rejected rather than voted through. Their incidence declines from 4,822 modes in the early stored-state tercile to 2,656 mid and 2,135 late, and varies by evaluator; mesh/evaluator/stage distributions are tabulated. All 16,536 selected modes satisfy all three conditions.
10. **Strongest diagnostics.** Joint void kinetic energy, void strain energy, and density-weighted participation. IPR corroborates localization but is nonbinding and mesh-dependent.
11. **Measurable overlap?** Yes at individual-diagnostic level: 9,613 modes have diagnostic disagreement and 28 such modes lie below the final selection. Each of those 28 satisfies only one condition and exhibits the localized/artificial physical pattern; none is promoted by voting. Every selected mode satisfies all three conditions.
12. **Broad threshold plateau?** Yes. Exact hard-gate-pass selections persist at `tau=0.1` for cuts 0.48..0.56. Wider exploratory cuts alter at most one near-degenerate frequency by 0.15%; the full grid is reported separately from the hard-gate subset.
13. **Exact modal-validity rule.** Require all three: `voidKE(rho_eff<=0.1)<0.5`, `voidSE(rho_eff<=0.1)<0.5`, and density participation `>0.5`; select the lowest mode satisfying all three; store IPR as nonbinding QA. Diagnostic disagreement is an invalid/not-established mode, not a vote.
14. **Maximum first-structural ordinal.** {A['max_ordinal']}.
15. **Number of >3 cases.** {A['gt3']} state/evaluator records.
16. **Number of >6 cases.** {A['gt6']}.
17. **Number of >10 cases.** {A['gt10']}.
18. **Number of >12 cases.** {A['gt12']}.
19. **Adaptive escalations.** Five Phase-2F 12-to-24 events; a 3-based schedule would trigger 244 first escalations and 318 total batch expansions (244+69+5).
20. **Unresolved states.** 0 after the observed 12-to-24 escalations.
21. **Cause of each unresolved state.** None observed. Future solver/resource exhaustion must return `STRUCTURAL_MODE_NOT_FOUND`.
22. **Fixed 3 adequate?** No for C: 244 records need more.
23. **Fixed 10 adequate?** No: 6 records need more.
24. **Fixed 12 adequate?** No: 5 records need more.
25. **Fixed 24 adequate observed?** Yes; observed maximum is 18 and no state remained unresolved.
26. **Is adaptivity preferable?** Yes; it is cheaper for the majority and preserves fail-closed semantics beyond observed evidence.
27. **Escalation schedule.** `3->6->12->24->48->...`, geometrically, preferably reusing factorization/subspace.
28. **Ceiling/failure policy.** No scientific fixed-mode ceiling. Stop only at solver dimension or preregistered resource failure and return `STRUCTURAL_MODE_NOT_FOUND`; never accept the last mode by default.
29. **Mode crossings.** Robust on the full 160x20 E2 sequence: 16 ordinal changes/1,599 transitions, median MAC 0.9963, no transition combining MAC<0.5 with a >2% jump.
30. **Largest unexplained C step.** None above 2%. The observed hard-gate pooled maximum is {100*float(SM['max']):.4f}%; it retains the same validated classification and is consistent with a genuine trajectory update. The voidKE-only 2.3797% k=594 artifact disappears.
31. **Does D eliminate void-localized modes?** Yes by kinetic-energy incidence: E2 mode-1 voidKE maximum {S['binary_void_max']['E2']:.3e}, zero cases above 0.5.
32. **D severe failures.** {B['severe_binary_lt_half_gray']}/{B['hard_gate_passing_states']} = {100*float(B['severe_fraction_of_hard_gate_pass']):.2f}%.
33. **Worst verified D mechanism.** Ratio minimum: 160x20 k=639, binary 1.88196 versus gray structural 165.7600. Required anchor: 400x50 k=833, 4.67271 versus 170.93147.
34. **Numerical or structural?** Not eigensolver errors and not void-kinetic modes. They are projection-created detached-solid/local structural mechanisms regularized by void stiffness.
35. **Why hard gate misses them.** It requires one spanning component but permits each detached component below `A_sig`; k=833 has a 9,985-element spanning component plus ten permitted 1–6 element islands, and mode 1 has only 2.82e-10 KE share on the spanning component.
36. **D valid early?** No; projection ties and disconnected/island mechanisms dominate.
37. **D valid mid-trajectory?** No; severe failures remain common and mesh-dependent.
38. **D near convergence?** Often numerically close (400x50 k=1577: 172.1398 versus 170.8055), but not uniformly proven across every mesh.
39. **Future D role.** Endpoint/manufacturability and topology-presentation diagnostic after modal/component checks; not primary trajectory maturation.
40. **Does C introduce method-correlated bias?** No label/mechanism bias is evident; differential rejection can reflect real morphology. Empirical incidence bias is untested outside Olhoff.
41. **Cross-method neutrality demonstrated?** No.
42. **Unavailable-trajectory limitation.** Proposed/Yuksel rejection rates, threshold margins, escalation incidence, and downstream endpoints cannot be measured.
43. **Are E1/E2/E3 all informative?** Retain all three for the current refreeze; C removes their modal pathologies but not their distinct stiffness/floor perspectives.
44. **Which evaluator binds most often?** Not determinable without C-based `Q_ref` normalization; absolute-frequency minima are not the protocol's binding definition.
45. **Are E2/E3 redundant?** Effectively nearly so after C: median differences are zero and maximum observed relative difference is {e23max:.3e}; later simplification deserves review, not action here.
46. **`b_ref` threshold sensitivity.** Not exercisable without a 3,200-state density history; available Q selections are invariant on the plateau.
47. **`k_enter` sensitivity.** Not exercisable.
48. **`k_cert` sensitivity.** Not exercisable.
49. **C overhead.** 3–24 modes in observed use, 1.43x cost for 12 versus 3 and 1.94x for 24; energy diagnostics add 2.9–9.3%; about 30 h single-threaded for the available eight-mesh 3x3 sweep.
50. **Practical for nine meshes?** Yes post hoc; extrapolated order 45 h single-threaded or about 6 h on eight workers.
51. **Scorecard winner.** C, 47/60 unweighted (A 34, B 36, D 31).
52. **Winner with labels hidden?** Yes.
53. **Selected candidate.** C — adaptive structural-mode evaluator.
54. **Exact methodology change.** Replace E2/E3 Eq.(4) algebraic-lowest evaluation with Eq.(4a) plus the unanimous three-condition adaptive rule; E1 uses the same modal rule with linear mass; propagate fail-closed status.
55. **Hard topology gate change?** No.
56. **New structural gate required?** No for C. Adding one to rescue D would duplicate/circularize the evaluator and redesign the methodology.
57. **Precision requalification required?** Yes.
58. **Phase-2B negative result historically valid?** Yes, for the pre-amendment Eq.(4) instrument.
59. **Olhoff LP/MMA decision changes?** No.
60. **Harness Olhoff variants.** `olhoff.variant='lp'|'mma'|'both'`; LP principal, nested MMA secondary, with separate work accounting.
61. **Normative amendments.** Evaluator, contract/hash, quality/protocol, Phase-2A integration/preflight/tests, fairness/readiness/freeze records, timing/reporting; native optimizers, P/B/q/topology unchanged.
62. **Refreeze authorized?** C is sufficiently supported for a separate controlled refreeze; this audit performs none.
63. **Production authorized?** No.
64. **Exact next action.** Independently implement and test the minimum C amendment in an isolated refreeze change, update all hashes/normative records, then run the fresh reference-length precision and cross-method trajectory qualifications before any production.

## Selection decision

ADOPT ADAPTIVE STRUCTURAL-MODE EVALUATOR C

METHODOLOGY REFREEZE: NOT PERFORMED

OLHOFF PRINCIPAL ROUTE: LP

OLHOFF SECONDARY ROUTE: NESTED MMA

PRODUCTION STATUS: BLOCKED
"""
write("PHASE2G_FINAL_AUDIT.md", answers)

print("reports written")
