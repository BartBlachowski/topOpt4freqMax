# SOCP_ELIGIBILITY — Part 3

## The proved supported class

Frozen evidence: `frozen_problem25_reference/SOCP_DERIVATION.md`,
`frozen_inner_solver_study/SOCP_GENERALIZATION.md`.

For N = 2, production builds, with a = F11ᵀΔρ, b = F12ᵀΔρ, c = F22ᵀΔρ, d₂ = dOff(2):

```
M(Δρ) = [ a      b     ]      e₁ = (a+c+d₂)/2 − √(((a−c−d₂)/2)² + b²)
        [ b   c + d₂   ]
```

Row 1 of (25c), `bs − (λ₁ + e₁)/λ_ref ≤ 0`, is exactly the second-order cone
`‖[(a−c−d₂)/2; b]‖ ≤ (a+c+d₂)/2 + λ₁ − λ_ref·bs`. Because
`dOff = λ(idx) − λ(idx(1))` is formed by `olhoffSolve` itself, every
λ_j − dOff_j equals λ₁ up to roundoff. Row 2 is then implied by row 1
(e₁ ≤ e₂), so the cluster constraint set is exactly the cone. Rows (25b)
(next mode) and (25e) (volume) are affine. The box (25f) with the move limit, and
the bs box [0, 5], are bounds. Problem (25) as posed by production is therefore
one convex SOCP, and it stays convex regardless of whether the frozen filtered
coefficients are integrable as an outer gradient field.

## Checks executed before every solve

| id | check | why it is needed |
|---|---|---|
| E1 | N = 2, J = 3 ≤ Jcalc | the 2×2 PSD ⇔ SOC identity holds only for N = 2 |
| E2 | F finite, NE×2×2, F(:,1,2) ≡ F(:,2,1) bitwise | A(Δρ) must be symmetric with fixed coefficients |
| E3 | dOff present and ≡ λ − λ₁ bitwise, dOff₁ = 0, dOff₂ ≥ 0 | consistent offsets ⇒ b_j = λ₁ for all j |
| E4 | \|(λ_j − dOff_j) − λ₁\| ≤ 4 eps λ₁ | row-1 dominance to roundoff |
| E5 | 0 < λ₁ ≤ λ₂ ≤ λ_J, fJJ finite | ordered spectrum, finite next-mode row |
| E6 | offDiag = true | the offDiag = false equality route is a different LP, not preregistered |
| E7 | no volFun, no projection, sensitivity filter, `innerLoop` path | nonlinear volume or chained variables are outside the proof |
| E8 | lo ≤ 0 ≤ hi, move > 0, ρ ∈ [ρ_min, 1] | Δρ = 0 lies in the box |
| E9 | next-mode and volume rows (production arithmetic) = Alin·x − blin, ≤ 1e-12 | affine rows are exactly the linear constraints |
| E10 | production row 1 (`deltaLambda`) = cone residual, ≤ 1e-10 | SOC is exactly the production row |
| E11 | row 2 ≤ row 1 + 1e-12 | redundancy holds at the point |
| E12 | production row-1 gradient = cone gradient, rel ≤ 1e-9 (skipped at the apex) | the smooth KKT certificate uses the production gradient |

E1–E8 run before the solve; E9–E10 run at x = [0; 1] after assembly; E9–E12 run at
the accepted point. `multJ` (next mode near-multiple) is recorded but is not a
rejection. The posed row (25b) is affine either way, and production handles it
identically ("log only").

## Unsupported cases and what happens

| case | handling |
|---|---|
| N = 1 | unsupported here: the configuration fixes N = 2 (`subspace`, size 2), and no LP path was preregistered |
| N > 2 | unsupported: needs an SDP of order N, and no validated SDP implementation exists |
| inconsistent offsets | unsupported: a lower bound on a higher ordered eigenvalue alone is nonconvex in general |
| offDiag = false | unsupported: the LP equality route was not preregistered |
| nonlinear volume / projection | unsupported: no proved conic form |
| cone apex (predicted e₁ = e₂) | supported by the generalized conic certificate (§4.3); smooth E12 skipped |

Any unsupported case terminates the run as `SOCP_UNSUPPORTED_CASE_HIT` with **no
update, no MMA fallback, no skipped iteration**. Preflight P5 demonstrated that
N = 3, inconsistent offsets, an asymmetric F perturbed by 1e-12, offDiag = false,
and a volFun are each rejected before any solve.

Run-time coverage is reported in `SOCP_CERTIFICATION.md` and `REPORT.md` (Part 14).
