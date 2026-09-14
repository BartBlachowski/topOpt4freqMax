# GRADIENT_OBJECTS — Part 4

Three objects, kept strictly separate throughout this audit.

## A. The physical FE objective derivative — `g_phys(ρ)`

```
g_phys = F(:,1,1)  from genGrad, BEFORE filtering
       = φ₁ᵀ ( ∂K/∂ρ_e − λ₁ ∂M/∂ρ_e ) φ₁
       = ∂λ₁/∂ρ_e            (modes M-orthonormal: φᵀMφ = I)
```

`genGrad` is called with `lamTilde = lam(1)` and `idx = 1`, which is exactly
what `olhoffSolve` does when it rebuilds the diagonal under
`multiplicity.diagonalOffsets = true`. Because `f₁₁` is quadratic in φ₁ it is
invariant to the eigenvector's sign, so it is well defined independently of
what `eigs` returns.

This is a genuine gradient of a scalar function wherever λ₁ is simple and
smooth. At the frozen state `gap12 = 0.1298`, so λ₁ is simple by a wide margin.
Its conservativity is the audit's positive control, and it passes at 2.6e−06.

## B. The filtered vector that actually reaches MMA — `g_filt(ρ)`

```
g_filt = applyFilter(flt, ρ, g_phys)
       = (H·(ρ∘g_phys)) ./ (Hs ∘ max(1e-3, ρ))
       = A(ρ)·g_phys           (the guard is inactive: 0 of 28 800 elements)
A(ρ)   = diag(1./(Hs∘ρ)) · H · diag(ρ)
```

**That this is the field MMA sees is not an assumption.** In `innerLoop`, the
spectral constraint rows carry `dfdx(j,1:NE) = −ddlam(:,j)ᵀ/lamref`, and
`ddlam` comes from `deltaLambda(ctx.F, drho, dOff)`. At `drho = 0` — the point
every inner solve starts from — the subeigenvalue matrix `A` in `deltaLambda`
is `diag(dOff)`, which is already diagonal, so the returned eigenvector basis
is the identity and

```
ddlam(:,j) = F(:,j,j)  = the FILTERED f_jj
```

so the active row's gradient is exactly `−g_filt/lamref`. `ctx.F` has been
filtered by `olhoffSolve` before `innerLoop` is entered.

The ρ-dependence of `A` is carried in every derivative test in this audit; it is
not frozen at ρ₃₈₆ and then differentiated as if constant.

## C. The gradient of a hypothetical scalar regularized objective — `∇F_reg`

Never assumed to exist. Whether B behaves locally like C is the question, and
`EFFECTIVE_OBJECTIVE.md` records the answer.

## What the filter does to the field, measured

| class | n | RMS \|g_phys\| | RMS \|g_filt\| | ratio |
|---|---|---|---|---|
| void (ρ < 0.1) | 10 264 | 9.216e−01 | 2.826e+01 | **30.67** |
| gray shell | 4 526 | 1.620e+00 | 1.386e+00 | 0.855 |
| gray core | 3 748 | 1.375e+00 | 1.356e+00 | 0.987 |
| solid (ρ > 0.9) | 10 262 | 6.500e+00 | 6.049e+00 | 0.931 |

In material the filter is close to magnitude-preserving. In void it amplifies by
a factor of 30 in RMS, and the operator's row sums `ρ̃/ρ` reach **245**. The
amplification is a direct consequence of dividing by `ρ_e` while the numerator
samples the solid neighbourhood. `figures/FIG_01_*` and `FIG_02_*` map both
fields and their ratio.

## What is NOT claimed here

That the filter is "wrong". Du & Olhoff (2007) §1 explicitly state the
Sigmund (1997) filter was applied to the sensitivities, and the implementation
reproduces the published top88 `ft = 1` algebra exactly. See
`LITERATURE_INTERPRETATION.md`. This section only establishes *what the two
fields are*, so that the integrability question is asked about the right object.
