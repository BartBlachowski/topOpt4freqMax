# FINAL_SUBPROBLEM_RECOVERY — Part 2

The exact implemented subproblem, recovered from code and retained state — not
a generic MMA problem.

## Where it was posed

At **ρ₃₈₅**, not at the frozen endpoint. `olhoffSolve` assembles, eigensolves
and forms the generalized gradients at the current density, then `innerLoop`
returns the increment that produces ρ₃₈₆. So the final subproblem's data are all
evaluated at ρ₃₈₅ with λ₁ = ω₁² = 163.93172747916933².

## Primal variables

```
x = [ drho (NE = 28 800) ; bs (1) ]      nvar = 28 801
```

`bs` is the bound variable β of (25a), **scaled** by `lamref = ctx.lam(1) = λ₁`;
`innerLoop` returns `st.beta = x(end)*lamref`.

## Bounds

```
xmin = [ max(ρmin − ρ₃₈₅, −move) ; 0 ]
xmax = [ min(1     − ρ₃₈₅, +move) ; 5 ]
move = 0.01   (stage 3)
```

Because ρ₃₈₅ never touches a box bound, every element has a two-sided,
nondegenerate interval; measured box widths are strictly positive throughout.

## Objective

```
f0(x) = −bs          df0/dx = [0 … 0 , −1]ᵀ
```

The objective gradient is **zero in every density coordinate** and −1 in the
scaled bound variable. All density information enters through the constraint
rows.

## Constraints — m = N + 2 = 4

With N = 2 (fixed subspace) and J = n + N = 3:

| row | equation | value at the returned point | dual λ |
|---|---|---|---|
| 1 | (25c) mode 1: `bs − (λ₁ + Δλ₁)/lamref` | −4.71234e−06 | **0.9972567734063279** |
| 2 | (25c) mode 2: `bs − (λ₂ + Δλ₂)/lamref` | −0.276516 | 3.6167840533154884e−07 |
| 3 | (25b) next mode: `bs − (λ_J + f_JJᵀdrho)/lamref` | −4.99559 | 2.0018108810544395e−08 |
| 4 | (25e) volume: `(Σ(ρ+drho) − Vtot)/Vtot` | −3.09101e−06 | **0.6324256597515894** |

`Vtot = volfrac·NE = 0.5 × 28 800 = 14 400`.

**Active set: row 1 (first spectral) and row 4 (volume).** Rows 2 and 3 are
slack by 0.28 and 5.0 with duals at the 1e−07 and 1e−08 level. This reproduces
the `gray_kkt_forensic_audit`'s independent finding that the admissible spectral
dual is `Q = diag(1,0)` and `f_JJ` is inactive.

## Constraint gradients

```
rows 1..N :  dfdx(j,1:NE) = −ddlam(:,j)ᵀ / lamref ,  dfdx(j,nvar) = 1
row N+1   :  dfdx(  ,1:NE) = −f_JJᵀ      / lamref ,  dfdx(  ,nvar) = 1
row N+2   :  dfdx(  ,1:NE) = 1/Vtot                (constant, ρ-independent)
```

`ddlam` comes from `deltaLambda(ctx.F, drho, dOff)`, the (25d) subeigenvalue
problem in its erratum form with the actual separation `dOff = [0, 7429.0787]`
retained on the diagonal. At `drho = 0` the subeigenvalue matrix is already
diagonal, so the basis is the identity and `ddlam(:,j) = F(:,j,j)` — **the
filtered f_jj**. `ctx.F` and `ctx.fJJ` were filtered by `olhoffSolve` before
`innerLoop` was entered.

## Constraint and objective scaling

Spectral rows are divided by `lamref = λ₁`; the volume row by `Vtot`. The
objective carries no scaling. These are the implemented scalings, reproduced
exactly, not a normalization chosen by this audit.

## Multiplicity treatment

`subspace`, N fixed at 2, `diagonalOffsets = true` (so `dOff` enters the
determinant), `offDiagonal = true` (so the full (25d) determinant is used and
the `~offDiag` LP branch with its 2N(N−1) extra rows is **not** taken).

## Inner stopping criterion

```
relStep = max|Δx_step| / max(max|drho|, 1e−12)  <  tolInner = 0.05
with minInner = 5 sub-iterates always taken, maxInner = 500
```

This is a **relative step** test, not an optimality test. It is a declared
reconstruction: `innerLoop`'s own header states the paper gives no criterion.
Nothing in the implemented criterion measures a KKT residual.

## MMA constants

`a0 = 1`, `a = 0`, `c = 1000`, `d = 0`, published Svanberg September-2007
`mmasub`. With `a = 0` and `a0 = 1` the auxiliary `z` is driven to 0 and, since
all λ ≪ c = 1000, all `y_i` are 0 — measured `y ≈ 1.0e−10`, `z = 1.0e−07`. The
MMA problem therefore reduces to problem (25) as written.

## Retained state, and what was missing

| item | retained in the trajectory? |
|---|---|
| ρ₃₈₅, ρ₃₈₆, Δρ₃₈₆ | **yes** |
| `hist.nInner`, `move`, `stage`, `beta`, `gap12`, `multJ`, `N` | **yes** |
| `res.diag.drho`, `lam`, `beta`, `dlamPred`, `fdiag`, `foff` | **yes** |
| **MMA duals** `lam, xsi, eta, mu, zet, s` | **NO** — `innerLoop` discards them with `~` |
| **MMA asymptotes** `low, upp` at the final iterate | **NO** — local to `innerLoop` |
| inner iterate history | **NO** |

So **no exact retained dual exists**. Duals were obtained by re-running the
identical `mmasub` calls on identical inputs and capturing the outputs
(`fi_innerloop_audit.m`, an audit-only mirror whose numeric expressions are
copied character-for-character from `innerLoop.m`). They are labelled
**reconstructed**, and are exact for the MMA convex subproblem.

## Reproduction is bitwise exact

| check | result |
|---|---|
| `drho` vs retained `DRHO(:,386)` | **bitwise equal**, max abs difference **0** |
| `nInner` | 19 reproduced vs 19 recorded |
| `β` | 26873.74466704797 both, difference **0** |

The reconstruction is therefore the actual final subproblem, not an
approximation of it. The preregistered precondition for Part 3 is met.
