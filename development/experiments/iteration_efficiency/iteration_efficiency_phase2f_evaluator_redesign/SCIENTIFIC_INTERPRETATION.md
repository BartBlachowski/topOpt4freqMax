# WP14 — What `k_enter` / `k_cert` would mean under each candidate
PHASE 2F — EVIDENCE ONLY — NO METHODOLOGY DECISION

The study's question is:

> What is the minimum number of iterations required by each method to obtain a proper
> result?

"A proper result" has to be operationalised by the common evaluator. Each candidate
operationalises it differently, and the differences are scientific, not cosmetic.

## The estimand each candidate actually defines

**Candidate A — Eq. (4), lowest algebraic mode.**
`k_enter` = the first iteration from which, for P consecutive updates, the design's
*Du–Olhoff-suppressed* gray eigenfrequency stays within `q` of the reference. The
suppression is a numerical device internal to an optimizer, imported into the measuring
instrument. What is measured is maturation *as the Olhoff scheme's own artificial-mode
filter sees it*. Because Eq. (4) drives void mass to `ρ^6`, the low-density region
contributes essentially nothing: the quantity is close to "the eigenfrequency of the solid
skeleton, with gray material counted for stiffness but almost not for mass". That is a
coherent quantity — but it is not the eigenfrequency of the design the optimizer is
actually holding, and it carries the discontinuity as an inseparable companion.

**Candidate B — Eq. (4a), lowest algebraic mode.**
`k_enter` = the first iteration from which the *lowest algebraic eigenvalue* of the
continuously interpolated gray state stays within `q` of the reference. When a void mode
lies below the structure, this is not a structural frequency at all; the quantity silently
changes meaning between states. A trajectory that passes through a void-mode window is
scored as having *lost* dynamic performance it never had and never lost. This is not a
harder-to-interpret estimand; it is an ill-defined one.

**Candidate C — Eq. (4a), lowest physically valid structural mode.**
`k_enter` = the first iteration from which the *lowest structural* eigenfrequency of the
actual gray design stays within `q` of the reference. This is the quantity the study's
question names. The gray state is the design the optimizer holds at iteration k; its
lowest structural mode is that design's dynamic performance; maturation of that quantity is
what "the method has obtained a proper result" should mean. The price is a selection rule,
and therefore a modal-validity criterion that must be shown non-arbitrary.

**Candidate D — exact-count binary projection.**
`k_enter` = the first iteration from which the *volume-preserving binary realisation* of the
gray design stays within `q` of the reference. This answers a subtly different question:
not "does this design perform" but "does the manufacturable topology this design implies
perform". It has an honest engineering reading — intermediate densities are not
manufacturable, so the design's value is the value of the 0–1 structure it encodes. But it
is not the gray state's own dynamic property, and it introduces a rank-based projection
between the design and the score.

## Which quantity answers the study's question?

The question says "obtain a proper result", not "obtain a proper manufacturable topology"
and not "obtain a proper artificially-suppressed spectrum". Read literally, **candidate C
is the closest match**: it measures the dynamic performance of the design the method
actually holds at iteration k.

Candidate D is defensible on different grounds — it is the quantity a practitioner would
care about if only the final 0–1 structure is deliverable — but adopting it *changes the
research question* from "when does the design mature" to "when does the implied topology
mature". That change must be made explicitly and defended, not slipped in because the
projection is numerically convenient.

Two consequences follow, and neither should be decided by implementation convenience:

1. If D is adopted, the study's stated question should be reworded, and every statement of
   the estimand in the frozen specifications updated with it. A candidate that changes the
   question cannot be adopted silently.
2. If C is adopted, the modal-validity rule becomes part of the frozen methodology and
   inherits the full burden of neutrality and threshold-robustness evidence. It is a new
   normative device, not a bug fix.

## The trap to avoid

Both C and D would produce *smoother, better-behaved* quality sequences than B, and both
would produce different `k_enter` values than A. It would be easy — and wrong — to prefer
whichever yields the tidiest iteration counts. The evidence assembled in this phase is
deliberately about *validity and stability of the measurement*, not about the iteration
counts any candidate produces for any method. No candidate was scored on the `k_enter`
values it would produce, and `k_enter` cannot in fact be computed on the available
artifacts: the frozen reference requires a separate `B_ref = 3200` trajectory, and the
longest density history in the repository is 1601 states.
