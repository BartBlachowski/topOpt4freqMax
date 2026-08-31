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
14. **Maximum first-structural ordinal.** 18.
15. **Number of >3 cases.** 244 state/evaluator records.
16. **Number of >6 cases.** 69.
17. **Number of >10 cases.** 6.
18. **Number of >12 cases.** 5.
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
30. **Largest unexplained C step.** None above 2%. The observed hard-gate pooled maximum is 1.0700%; it retains the same validated classification and is consistent with a genuine trajectory update. The voidKE-only 2.3797% k=594 artifact disappears.
31. **Does D eliminate void-localized modes?** Yes by kinetic-energy incidence: E2 mode-1 voidKE maximum 1.845e-08, zero cases above 0.5.
32. **D severe failures.** 667/2704 = 24.67%.
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
45. **Are E2/E3 redundant?** Effectively nearly so after C: median differences are zero and maximum observed relative difference is 5.372e-04; later simplification deserves review, not action here.
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
