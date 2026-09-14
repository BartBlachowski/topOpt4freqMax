# EXACT_PROBLEM25 — Part 2

The exact problem production attempts to solve at outer iteration 386, recovered
from `+impl/algo/innerLoop.m`, `+impl/algo/deltaLambda.m`,
`+impl/architecture/olhoffSolve.m` (lines 225–331) and the frozen `ctx`.
Every expression below is the implemented one; nothing is cleaned up.

## Variables

```
x = [ drho ∈ R^28800 ; bs ∈ R ]        nvar = 28 801
bs = β / lamref,   lamref = λ₁(ρ₃₈₅) = 26873.611274304643
```

`innerLoop` returns `st.beta = x(end)·lamref`.

## Objective

```
minimize  f0(x) = −bs            ∇f0 = [0 … 0, −1]ᵀ
```

## Box (25f) plus the move limit

```
xmin = [ max(ρmin − ρ₃₈₅, −move) ; 0 ]      ρmin = 1e−3, move = 0.01
xmax = [ min(1    − ρ₃₈₅, +move) ; 5 ]
```

## Constraints, m = N + 2 = 4, all in production scaling

Let `A(drho)` be the symmetric 2×2 matrix `A_sk = F(:,s,k)ᵀ drho` and
`M(drho) = diag(dOff) + A(drho)`, `dOff = [0, λ₂ − λ₁] = [0, 7428.636632030943]`.
`deltaLambda(F, drho, dOff)` returns the ascending eigenvalues `e₁ ≤ e₂` of
`M(drho)` as `Δλ_j = e_j − dOff_j`, and the gradients
`ddlam(:,j) = Σ_{s,k} v_js v_jk F(:,s,k)` with `v_j` the j-th eigenvector.

| row | implemented expression | gradient in drho | gradient in bs | origin |
|---|---|---|---|---|
| 1 | `bs − (λ₁ + Δλ₁)/lamref ≤ 0` | `−ddlam(:,1)ᵀ/lamref` | 1 | (25c), j = n |
| 2 | `bs − (λ₂ + Δλ₂)/lamref ≤ 0` | `−ddlam(:,2)ᵀ/lamref` | 1 | (25c), j = n+1 |
| 3 | `bs − (λ_J + f_JJᵀ drho)/lamref ≤ 0` | `−f_JJᵀ/lamref` | 1 | (25b), J = 3 |
| 4 | `(Σ_e(ρ₃₈₅,e + drho_e) − Vtot)/Vtot ≤ 0`, `Vtot = 0.5·NE = 14 400` | `1/Vtot` | 0 | (25e) |

Since `dOff_j = λ_j − λ₁` exactly, row j reads
`bs·lamref ≤ λ₁ + e_j(M(drho))`, i.e. **the j-th ordered eigenvalue of
`diag(λ₁, λ₂) + A(drho)`**. This is the diagonal-offset form of (25d)
(`deltaLambda` header, `olhoffSolve.m:240–245`): the actual eigenvalue
separation is retained on the diagonal, and the printed (25d) — which assumes
an exactly N-fold eigenvalue — is recovered when `dOff = 0`.

`F` and `f_JJ` are the **sensitivity-filtered** generalized gradients
(`applyFilter`, Sigmund 1997, radius 0.06, applied to every f_sk including the
off-diagonal one). The diagonal blocks use their own λ_j, the off-diagonal
block uses λ̃ = λ₁ (eq. 19/24).

## The MMA layer

`innerLoop` hands this problem to the published `mmasub` with
`a0 = 1, a = 0, c = 1000, d = 0`. The MMA problem is
`min −bs + z + 1000·Σy_i` s.t. `f_i(x) − y_i ≤ 0`, `y ≥ 0`, `z ≥ 0`. With
`a = 0` the variable `z` is driven to 0, and because every multiplier stays far
below `c = 1000` all `y_i` are 0 (measured along the replay: `max y ≤ 1e−9`,
`z ≤ 1e−6`, `evaluations/mma_replay.mat`, `H.ymax`, `H.zmma`). The MMA problem
is therefore (25) as written above. The repeated-MMA *sequence* is
`x⁰ = [0; 1]`, asymptotes reset to the box each outer iteration, published
asymptote update rules, and the relative-step stop
`max|Δx_step|/max(max|drho|, 1e−12) < 0.05` after at least 5 sub-iterates.

## What is paper and what is reconstruction

| element | class |
|---|---|
| bound variable β, maximize β (25a) | paper |
| (25b) next-mode linearized constraint | paper |
| (25c) cluster constraints on β | paper |
| (25d) sub-eigenvalue problem, erratum form with Δ | paper (with the published erratum) |
| (25e) volume, (25f) density box | paper |
| MMA as the mathematical-programming method | paper ("the MMA method (Svanberg 1987) has been used") |
| sensitivity filter on the f_sk | paper §1 names Sigmund (1997) applied to the sensitivities; radius, and application to the off-diagonal f_sk, are reconstruction |
| **hard move box ±0.01 on drho** | **reconstruction** — (25f) bounds only the density; `olhoffSolve.m:289` says "NONE of this is specified by the paper: the only printed bound on drho is the box (25f)" |
| diagonal-offset (25d) with `dOff = λ_j − λ₁` | reconstruction (class C, `deltaLambda` header) |
| fixed subspace N = 2 (no multiplicity classifier) | reconstruction (`olh.multi.detect` header) |
| scaling by `lamref`, `Vtot`; `bs ∈ [0, 5]` | reconstruction |
| MMA constants, asymptote reset per outer iteration, published `move = 0.5`, `asyinit = 0.5` | reconstruction / Svanberg defaults |
| inner stop rule (relative step 0.05, min 5, max 500) | reconstruction — the paper says only "Increments Δρe converged?" |
| gradient formula `ddlam = Σ v_s v_k F_sk` | reconstruction |

See `PAPER_VS_RECONSTRUCTION.md` for the paper-side detail.

## Structural observation (recorded before solving, preregistration §6)

The constraints depend on `drho` only through five linear functionals:
`a = F11ᵀdrho`, `b = F12ᵀdrho`, `c = F22ᵀdrho`, `f_JJᵀdrho`, `Σdrho`.
The optimal `bs` is therefore a property of the image of the box under these
five maps; the optimal `drho` is any preimage of the optimal image point inside
the box and may be non-unique on a face of the box.
