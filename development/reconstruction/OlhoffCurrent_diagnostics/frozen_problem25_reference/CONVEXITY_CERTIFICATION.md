# CONVEXITY_CERTIFICATION — Part 6

## Verdict

```
FROZEN_PROBLEM25_CONVEXITY_CERTIFIED
```

## The reduced exact problem

After `CLUSTER_CONSTRAINT_REDUCTION_PASS` (row 2 is implied by row 1 for every
`x`) and `FROZEN_PROBLEM25_SOCP_EQUIVALENCE_PASS` (row 1 is the cone of
`SOCP_DERIVATION.md` to 7e−15), the feasible set of the exact frozen production
problem is

```
minimize   −bs
subject to ‖A_c x − b_c‖₂ ≤ d_cᵀx − γ_c        (second-order cone: row 1)
           [−f_JJᵀ/lamref, 1] x ≤ λ_J/lamref     (affine: row 3, next mode)
           [1ᵀ/Vtot, 0] x ≤ (Vtot − Σρ₃₈₅)/Vtot   (affine: row 4, volume)
           xmin ≤ x ≤ xmax                        (box: (25f) ∩ move limit, bs ∈ [0,5])
```

* the objective is linear;
* the cluster condition is a second-order cone, a convex set for any `A_c, b_c,
  d_c, γ_c` (the norm is convex, the right side affine);
* rows 3 and 4 are half-spaces;
* the box is a polytope.

The intersection of convex sets is convex, so the frozen problem is a convex
second-order cone program. The certification does not depend on whether the
cone apex lies in the box (it may; see `SOCP_DERIVATION.md`), only on the
exact equivalence established pointwise.

Equivalently, the reduced constraint is `bs·lamref ≤ λ₁ + λ_min(M(drho))`
with `M` affine in `drho`; `λ_min` of an affine symmetric matrix family is
concave, so `{x : bs·lamref − λ₁ − λ_min(M(drho)) ≤ 0}` is convex.

## What the certificate licenses

A point that satisfies the KKT conditions of this convex program — or,
independently, whose objective matches a valid weak-duality bound — is a
**global** solution of the frozen problem, up to the tolerance of that
certificate. Uniqueness of `bs` follows; uniqueness of `drho` does not
(`EXACT_PROBLEM25.md`, structural observation).

## What it does not say

Nothing here concerns the outer problem `max λ₁(ρ)` — which is nonconvex —
nor the sensitivity-filtered surrogate's relation to it. Convexity is a
property of the frozen inner problem (25) alone.
