# CLUSTER_CONSTRAINT_REDUCTION — Part 3

## Verdict

```
CLUSTER_CONSTRAINT_REDUCTION_PASS
```

## Claim

For every admissible `drho`, production row 1 implies production row 2, so
the clustered family `{row 1, row 2}` reduces exactly to row 1 — the
minimum-eigenvalue constraint.

## Proof (from the implemented equations)

1. `olhoffSolve.m:245` sets `dOff = lam(idx) − lam(idx(1))` and
   `ctx.lam = lam(idx)`, so `dOff_j = λ_j − λ₁` **exactly** (floating-point
   identity checked: `(λ₂ − dOff₂) − λ₁ = 0`).
2. `deltaLambda` forms `M = diag(dOff) + A(drho)` (symmetric), takes
   `[V,D] = eig(M)`, sorts the eigenvalues ascending `e₁ ≤ e₂`, and returns
   `Δλ_j = e_j − dOff_j`.
3. Hence `λ_j + Δλ_j = λ_j − dOff_j + e_j = λ₁ + e_j`.
4. Row j is `bs·lamref ≤ λ₁ + e_j`. Since `e₁ ≤ e₂` for every `drho`
   (they are the sorted eigenvalues of one symmetric matrix),
   `bs·lamref ≤ λ₁ + e₁ ⇒ bs·lamref ≤ λ₁ + e₂`. ∎

The argument uses only the ordering of the eigenvalues of a symmetric 2×2
matrix and the exact identity in step 1. It holds for every `drho ∈ R^NE`,
not only in the box.

## A subtlety the task prompt did not anticipate

The prompt's step "verify `deltaLambda` returns `dlam₁ ≤ dlam₂`" is the
correct statement only when `dOff = 0`. With the offsets present the
*increments* are measured from different baselines, `Δλ₂ = e₂ − 7428.6`, and
they are **not** ordered (e.g. at M500: `Δλ = [19.07, −2.31]`). The
constraint-relevant quantity `λ_j + Δλ_j = λ₁ + e_j` **is** ordered at every
point. The equivalence script reports `all_dlam_sorted_NONCRITERIAL = false`
and `all_ev_sorted = true`; the preregistered criterion (§3.1) is the latter.

## Numerical verification (`evaluations/socp_equivalence.json`)

Fifteen deterministic points (drho = 0, P19, M500, five seeded random box
points, ±move clipped, `move·sign(F11)` clipped, three volume-neutral
directions, the reference), each at three `bs` values.

| check | result over all 45 evaluations |
|---|---|
| `λ₁ ≤ λ₂` | true |
| `e₁ ≤ e₂` (`ev_sorted`) | true at every point |
| `(λ₁ + Δλ₁) ≤ (λ₂ + Δλ₂) + 1e−9·lamref` | true at every point |
| `(λ₂ − dOff₂) − λ₁` | 0 (bar 4·eps·λ₁ = 2.39e−11) |
| min `e₂ − e₁` over the points | 6 500.2 (at −move everywhere) |
| row-2 slack at the reference | 0.278 (row 2 is inactive by a wide margin) |

Row 2 is therefore removed from the reference problem **only after** both the
analytic proof and the numeric check passed. It is retained in every
production-side evaluation (`fp_problem.evalProd`, the `fmincon` cross-check)
so that the nonlinear route solves the literal four-row problem.
