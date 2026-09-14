# GRAYNESS_IMPLICATION — Part 14

## Verdict

```
SURROGATE_MISMATCH_PARTIALLY_EXPLAINS_NONSTATIONARY_GRAYNESS
```

The MODERATE level of Part 14: this audit explains **why the algorithm can stop
while the physical KKT residual stays substantial**, and does **not** explain
why grayness grows with mesh.

## What is now explained

The previous audit established the coexistence of two facts that looked
contradictory:

* the 480 endpoint is **not** KKT-stationary for the physical relaxed problem —
  best gray-only raw residual 0.334;
* the same endpoint is **nearly** stationary for the filtered local model —
  best gray-only filtered residual 0.049.

This audit supplies the mechanism, and it is not a matter of degree:

**The filtered field is not the gradient of anything.** Its Jacobian has a
measured antisymmetric part of 0.287 that is independent of the
finite-difference step over a 33× range; its closed-loop integrals scale as
a^1.999 and reverse sign to 3.9e−13; the physical gradient passes the identical
tests at the 1e−06 level. By the converse of the Poincaré lemma there is no
scalar `F` with `∇F = g_filt` near this state.

So the algorithm is not minimizing anything. It iterates on a vector field that
has a curl, and "convergence" means its increments became small enough for the
exhaustion rule to fire — a **fixed point of a heuristic update field**, not a
stationary point of any optimization problem, physical or regularized.

That is exactly why a design can sit there, unmoving, at 26.3 % M_nd, with a
flat terminal window (ω₁ +0.0037 %, M_nd −0.036 points over the last 20
iterations), while carrying a large physical first-order residual. The two
observations were never in tension; they are what a non-conservative update rule
produces.

### Where the residual lives

Measured on the final inner subproblem, normalized by the same scale:

| class | n | inner-KKT residual RMS | filter amplification RMS \|g_filt\|/\|g_phys\| |
|---|---|---|---|
| void (ρ < 0.1) | 10 264 | **0.5776** | **30.67** |
| gray shell | 4 526 | 0.0160 | 0.855 |
| gray core | 3 748 | **0.0129** | 0.987 |
| solid (ρ > 0.9) | 10 262 | 0.1712 | 0.931 |

Two things stand out. The filtered subproblem is **nearly stationary inside the
gray regions** (0.013–0.016) — the surrogate is content there, which is why the
design stops moving. And the residual is concentrated in **void**, precisely
where the filter amplifies the gradient ~30× in RMS (operator row sums reach
245×). The mismatch between "the surrogate is satisfied" and "the physics is
not" is therefore largest exactly in the regions the filter transforms most.

## What is NOT explained, and must not be claimed

**1. Why grayness grows with mesh.** This audit examined **one mesh**. It
performed no 400 or 800 evaluation. Nothing here measures how the antisymmetry,
the amplification, or the surrogate/physical gap scale with refinement. The
mesh trend (M_nd 12.9 → 15.4 → 26.3 → 34.4 % at 320/400/480/800) is untouched by
this work.

**2. That the filter causes the gray patches.** Non-conservativity and broad
gray patches coexist at this state. Coexistence is not causation, and the task's
interpretation rules forbid the stronger claim. A non-conservative field can have
attracting fixed points that are perfectly discrete; nothing about a nonzero
curl forces intermediate densities.

**3. That fixing the filter would remove the grayness.** Untested, and not
testable without running a different formulation — which this audit is forbidden
to do and which `NEXT_ACTION.md` does not recommend on this evidence alone.

**4. That MMA is at fault.** Its own subproblem's KKT does fail
(`INNER_MMA_KKT.md`), but that is a truncation-and-convergence finding about a
deliberately loose inner criterion, not a defect in the published algorithm.

## The precise statement this audit supports

> The sensitivity-filtered update field is not physically KKT-consistent and is
> locally non-conservative, allowing convergence of the heuristic filtered
> algorithm without convergence to a stationary point of the underlying FE
> optimization problem.

That is the strongest acceptable statement in the task's own interpretation
rules, and the evidence meets its conditions: Jacobian asymmetry **and**
closed-loop integrals, both surviving finite-difference refinement, both with a
passing physical positive control.

## Relation to the mesh question

One observation is worth recording without over-reading it. The obstruction is
dominated (median share 0.991) by the `A·Hess` term — non-commutation between
the filter operator and the physical Hessian. Both `A` (through `Hs`, `rmin` in
element units, and the ρ field) and `Hess` change under refinement, so there is
no reason to expect the antisymmetry to be mesh-invariant.

That is a **hypothesis for a future measurement**, not a result. Testing it
requires evaluating the same statistic at 400 and 800, which this audit was
explicitly instructed not to turn into a cross-mesh study.
