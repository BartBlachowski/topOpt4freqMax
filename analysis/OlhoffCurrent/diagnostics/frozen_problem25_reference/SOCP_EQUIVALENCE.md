# SOCP_EQUIVALENCE — Part 5

## Verdict

```
FROZEN_PROBLEM25_SOCP_EQUIVALENCE_PASS
```

Source: `evaluations/socp_equivalence.json` (`scripts/fp_equivalence.m`,
`scripts/fp_testpoints.m`).

## Test points

| name | construction | max\|drho\|/move | e₂ − e₁ |
|---|---|---|---|
| T0 | drho = 0 | 0 | 7 428.6 |
| T1 | production P19 = `DRHO(:,386)` | 0.062 | 7 430.9 |
| T2 | M500 (prior audit) | 0.876 | 7 407.3 |
| T4–T8 | `rng(20260912)` uniform in the box | 1.000 | 14 852 – 15 310 |
| T9 | +move everywhere, clipped (`= xmax`) | 1.000 | 23 801 |
| T10 | −move everywhere, clipped (`= xmin`) | 1.000 | 6 500 |
| T11 | `move·sign(F11)`, clipped | 1.000 | 8 354 |
| T12–T14 | seeded volume-neutral directions, scaled into the box | 0.001 | 7 428.2 – 7 428.8 |
| T15 | the conic reference | 1.000 | 7 474.4 |

T3 (M5000) is appended by re-running `fp_equivalence` once the replay has
finished; see `MMA_TRAJECTORY_TO_REFERENCE.md`. At each point `bs` is set to
`(λ₁ + e₁)/lamref·(1 + δ)`, `δ ∈ {−1e−3, 0, +1e−3}`, giving 45 evaluations.

## Results

| comparison | preregistered bar | measured (max over 45) |
|---|---|---|
| row 1, `deltaLambda` value − cone residual | ≤ 1e−10 | **6.9e−15** |
| feasible / active / infeasible class identical | all | **all 45** (15 active, 15 feasible, 15 infeasible) |
| row-1 gradient, `ddlam` form vs cone gradient, relative ∞-norm | ≤ 1e−9 | **1.3e−16** |
| `ddlam(:,1)` vs closed-form `∂e₁/∂drho`, relative ∞-norm | ≤ 1e−9 | **6.2e−15** |
| `e₁` from `eig` vs closed form (absolute, λ units) | — | 3.6e−12 |
| row 3 (next mode) vs affine form | ≤ 1e−12 | **5.3e−15** |
| row 4 (volume) vs affine form | ≤ 1e−12 | **3.1e−14** |
| objective `−bs` vs `fᵀx` | ≤ 1e−12 | **0** |
| `secondordercone` sign convention toy test | exact −2 within 1e−6 | −2.0000000000 |
| Lagrangian Hessian-vector vs FD of production gradient | ≤ 1e−4 | 1.1e−8 |

## One disclosed refinement

The preregistered sign test compares `sign(value)`. At `δ = 0` the points are
constructed to be *exactly* active — production's value is `0.0` by
construction while the cone residual is `O(1e−15)` roundoff of either sign — so
a literal sign test compares roundoff. The implemented test uses a three-way
class with a ±1e−12 band (`feasible < −1e−12 < active < 1e−12 < infeasible`)
and requires identical class. All 45 agree; the `δ = ±1e−3` evaluations agree
under the literal sign test as well.

## Meaning

The production nonlinear cluster constraint, computed by the production
`deltaLambda`, and the second-order cone of `SOCP_DERIVATION.md` are the same
function of `x` to machine precision, in value, classification and gradient;
rows 3–4 and the objective are the same affine functions. Solving the SOCP is
therefore solving the exact frozen production subproblem.
