# FILTER_COMPARISON — Part 10

## Verdict

**The sensitivity filters are mathematically identical.** Therefore the clean source behaviour
**cannot be attributed simply to removal of the previously identified filter non-conservativity**
(filtered_subproblem_integrability_audit: Jacobian antisymmetry 0.287 at C480, 99 % from the
`A·Hess` non-commutation term). That mechanism is present, unchanged, in the source.

## Evidence

| question | source | target | evidence |
|---|---|---|---|
| sensitivity or density filtering? | sensitivity (`filter.type = sensitivity`) | same | effective configs |
| modified filter? | no — `prepFilter.m`, `applyFilter.m` byte-identical (top88 ft = 1) | — | file map |
| formula | f̃_e = Σ_i H_ei ρ_i f_i / (Hs_e · max(1e−3, ρ_e)), H_ei = max(0, r_min − dist) | same | code |
| which vectors | every f_sk (`applyTo = all`) and f_JJ | same | configs, code |
| application point | after `genGrad` (with diagonal offsets), before the inner loop; ρ weighting uses the unfiltered design | same | `olhoffSolve.m` lines are textually identical on this branch |
| radius | R = 0.06 physical → r_min = 0.06·nely = 3.6 elements at 480×60 | same | configs |
| rmin-only change? | no change at R = 0.06; the Rel sweep uses 1.3 elements (a different policy, not the audited realization) | — | sweep verification |
| ρ weighting/normalization change? | none | — | code |
| numerical identity | filtered f₁₁, f₂₂, f₁₂, f_JJ **bitwise equal** (target vs source-with-SIMP/4b) at 9 frozen states | | same_state_comparison.json |

## What does differ: the filter's input

Under Pedersen/eq.(2) the raw generalized gradients differ in the void band (row 23 of the delta
table). Because the filter averages ρ_i f_i over a 3.6-element radius, that difference leaks into
neighbouring elements: filtered f₁₁ differs by 19–31 % (L2, all elements) between the two material
laws at gray states, but only 0.4–2.5 % on ρ ≥ 0.1 elements. This is a formulation effect carried
through an identical filter operator, not a filter change.

## Stationarity note

At the endpoints the *filtered-model* gray-fit residual is 0.011 (S480, native) vs 0.049 (C480,
native) on the raw scale, while the *physical* residual is 0.334 vs 0.334 (STOPPING_COMPARISON.md).
The source endpoint is closer to stationarity of the filtered local model the algorithm sees, not of
the physical problem — consistent with, and not a removal of, the earlier finding.
