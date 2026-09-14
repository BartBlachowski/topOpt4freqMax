# NEXT_ACTION — Part 15

## Verdicts

```
REPEATED_MMA_REALIZATION_FAILS_PROBLEM25_REFERENCE
FROZEN_INNER_SOLVER_STUDY_JUSTIFIED
```

## How the decision logic resolved (preregistration §8, in order)

1. A certified reference exists (`GLOBAL_PROBLEM25_REFERENCE_CERTIFIED`) — not Case E.
2. Convexity is certified, and every independent solve agrees in objective to
   7.4e−7 — not Case D.
3. `G_P19 = 0.9972 > 0.1` — not Case B.
4. `G_M5000 = 0.4630 > 0.1` — Case C fails on its objective condition
   (it would also fail the ≥ 50 % move-bound rule, 0.337).
5. ⇒ Case A.

## What the reference answers

The exact frozen production problem (25) at the authoritative 480 state has a
unique optimal objective, `bs = 1.001792128848` (β = 26 921.77, a 0.179 %
predicted eigenvalue gain), certified global to 9e−12. Its solution is a
**bang-bang vertex of the box**: every gray element moves by the full ±0.01,
void goes to the floor, solid to the ceiling, with the volume exactly
balanced; only two general constraints (minimum eigenvalue, volume) are
active. To 6e−4 of the gain it is the thresholding rule
"`F11_e/λ₁ > ν/Vtot` ⇒ up, else down" on the filtered first-mode
sensitivity.

Production's repeated MMA stops at 0.3 % of that gain; run 5 000 times longer
it reaches 54 % with a step a quarter the size, drifting slowly toward the
vertex from the inside without converging to it.

## The single highest-information next action

**Study the inner-solver realization on this frozen subproblem, with the
certified reference as the yardstick.** Concretely, on the identical frozen
data (ctx, box, scaling), with `drho` discarded:

1. Measure what a *conservative* realization does — GCMMA (Svanberg 2002)
   with its inner conservative loop — in sub-iterations to reach G ≤ 0.01 and
   in distance to the reference. The reference makes "converged" a measurable
   statement instead of a relative-step heuristic.
2. Measure the sensitivity of plain repeated MMA to its reconstruction
   choices that the paper does not fix: asymptote initialization (`asyinit`),
   asymptote reset versus persistence across outer iterations, `move = 0.5`
   in `mmasub`, and the stop rule — each against the same reference.
3. Test whether the SOCP itself is an admissible inner solver for the
   production loop: it is exact for N = 2 with offsets, solves in ~50 s
   single-threaded here, and delivers the vertex with a certificate. Whether
   a vertex step is *desirable* for the outer iteration (it is a full-move
   bang-bang update every outer step) is the scientific question that study
   must answer, and it is precisely the question the move-box reconstruction
   raises.

None of these updates a density; all are frozen-state certifications.

## What must not be concluded from this task

* Not that the move limit is wrong. The reference shows the box is what
  determines the frozen solution; whether the outer iteration *should* take
  vertex steps is not decided here, and the move limit was not changed.
* Not that the filter is to blame. The frozen problem is a convex program in
  the filtered data; its solution is what it is regardless of how the data
  were produced. The filter's non-conservativity is a separate, standing
  finding.
* Not that MMA is defective. Each `mmasub` call solves its convex
  approximation cleanly; the failure is of the *repeated-MMA realization* on
  a problem whose solution is a vertex the interior-point subsolver approaches
  geometrically slowly.
* Not that the 5 000-state is a solution of (25). It is 46 % short of the
  certified objective.

## Downstream questions explicitly left open

Density filtering versus sensitivity filtering; projection; p continuation;
why grayness grows with mesh; whether the nine-mesh campaign should run. This
task examined one frozen subproblem at one mesh.
