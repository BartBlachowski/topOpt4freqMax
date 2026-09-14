# NEXT_ACTION — Part 15

## Primary verdict

```
INNER_MMA_CERTIFICATION_FAILURE_REQUIRES_RESOLUTION
```

## Why this and not the filter study

`FILTER_FORMULATION_STUDY_JUSTIFIED` requires, in the task's own definition,
that **MMA solves its surrogate**. It does not. The final inner subproblem fails
its own KKT at 19, 500 and 5000 sub-iterates, the residual *degrades* with more
iterations (0.360 → 0.586 → 0.597), and the increment the subproblem actually
wants is **15.7× larger** than the one production returns — 97.9 % of the move
limit against 6.2 %.

That ordering is not a technicality. Until it is known what the algorithm does
when its surrogate is solved accurately, the endpoint cannot be attributed to
the filter's non-conservativity, because the endpoint is currently determined by
**where the inner loop is truncated**.

This is the task's Case B, and its interpretation rule is explicit: inner/local
optimization accuracy must be resolved before blaming the filter.

## What the highest-information next study is

**Determine whether the final outer step is well defined at all**, at the frozen
480 state, without accepting any density update.

Concretely, and in this order:

1. **Characterize the oscillation.** The relative step is non-monotone over the
   last 100 of 5000 sub-iterates, ranging [8.97e−04, 1.24e−02]. Establish
   whether the iterates cycle, wander, or drift along the move-limit face.
   Retain the full primal/dual/asymptote state — this audit's lean recorder
   deliberately did not.
2. **Test the conservative-approximation question.** Plain `mmasub` has no
   global-convergence guarantee as a fixed-point map. GCMMA's conservative outer
   loop does. Running the **same subproblem** under a conservative-approximation
   safeguard, purely as certification with `drho` discarded, would show whether
   the subproblem has a KKT point that a convergent method reaches, or whether
   the subproblem itself is the problem.
3. **Localize the failure.** The residual is 0.578 in void and 0.013 in the gray
   core, and the filter amplifies void gradients ~30× in RMS with operator row
   sums to 245. Test whether the non-convergence survives when the void region's
   contribution is examined separately.

All three are frozen-state certifications. None requires an optimization run,
a density update, or any change to the formulation.

## What must NOT be concluded yet

* **Not** that the filter should be replaced. The non-conservativity is
  established (`JACOBIAN_SYMMETRY.md`, `CLOSED_LOOP_INTEGRALS.md`) and a
  conservative alternative exists in principle (`EFFECTIVE_OBJECTIVE.md` §3) —
  but the filter is what Du & Olhoff (2007) §1 explicitly prescribes, replacing
  it departs from a printed choice, and the inner-solve question is logically
  prior.
* **Not** that MMA is defective. Its convex subproblem is solved cleanly at every
  call; complementarity sits at the interior-point barrier floor throughout.
* **Not** that any of this causes the grayness. See `GRAYNESS_IMPLICATION.md`.

## 4. Projection and p-continuation — still not recommended

The task forbids recommending either unless this audit produced evidence
specifically supporting them. It did not.

**Projection remains premature.** `PROJECTION_CANARY_EXPERIMENT_PREMATURE`
stands. Projection would change the filter *and* the variable representation
simultaneously, adding a second uncontrolled factor on top of an inner-solve
failure that is not yet understood. It is also CLASS D — absent from every Du &
Olhoff source. Nothing measured here bears on it.

**p-continuation remains unjustified.** `P_CONTINUATION_REOPENING_NOT_JUSTIFIED`
stands. This audit produced no evidence about the penalization schedule, and the
prior failed connectivity experiment is untouched.

## Standing findings this audit does establish

Independent of the inner-solve question, and not contingent on it:

* the sensitivity-filtered field is **locally non-conservative** — Jacobian
  antisymmetry 0.287 invariant across a 33× range of FD step, closed-loop
  integrals scaling as a^1.999 with exact sign reversal, and a physical positive
  control passing at 1e−06;
* **no scalar objective exists** whose gradient it is, near this state;
* the obstruction is **99 % non-commutation** (`A·Hess`), not the ρ-weighting,
  so symmetrizing the operator would not fix it — measured, 0.031 residual
  asymmetry after symmetrization;
* the exact Jacobian decomposition
  `J_filt = A·D_{g/ρ} + A·Hess − diag(g_filt/ρ)`, verified to 1e−08.

These are durable results. They will not need re-deriving whatever the
inner-solve study concludes, and they define precisely what the algorithm is
iterating on.
