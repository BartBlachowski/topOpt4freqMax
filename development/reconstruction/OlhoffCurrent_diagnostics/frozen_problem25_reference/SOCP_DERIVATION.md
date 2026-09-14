# SOCP_DERIVATION — Part 4

## From the implemented equations to a second-order cone

With N = 2 write `a = F11ᵀdrho`, `b = F12ᵀdrho`, `c = F22ᵀdrho`, `d₂ = dOff(2)`.
`deltaLambda` builds

```
M(drho) = [ a      b     ]
          [ b   c + d₂   ]
```

whose smaller eigenvalue is

```
e₁ = (a + c + d₂)/2 − sqrt( ((a − c − d₂)/2)² + b² ).
```

Row 1 of production, `bs − (λ₁ + e₁)/lamref ≤ 0`, is therefore

```
sqrt( ((a − c − d₂)/2)² + b² )  ≤  (a + c + d₂)/2 + λ₁ − lamref·bs .
```

The left side is the Euclidean norm of a vector affine in `x`; the right side
is affine in `x`. This is a second-order cone constraint. Dividing both sides
by `lamref > 0` is an exact equivalence and puts the cone in the same units as
production's row 1 (`λ₁/lamref = 1` exactly because `lamref = ctx.lam(1)`).

## `coneprog` form

MATLAB's `secondordercone(A, b, d, γ)` encodes `‖A·x − b‖ ≤ dᵀx − γ` (the
convention was verified with a toy problem whose optimum differs under the two
candidate sign conventions: obtained −2.0000000000, documented convention
confirmed). With `x = [drho; bs]`:

```
A_c = [ 0.5·(F11 − F22)ᵀ / lamref ,  0 ]        (2 × 28 801)
      [       F12ᵀ      / lamref ,  0 ]
b_c = [ d₂ / (2·lamref) ; 0 ]
d_c = [ 0.5·(F11 + F22) / lamref ; −1 ]
γ_c = −( d₂ / (2·lamref) + 1 )
```

so that `‖A_c x − b_c‖ − (d_cᵀx − γ_c) = bs − (λ₁ + e₁)/lamref` identically.

**Difference from the structure the task prompt expected.** The prompt's
template (`b_c = 0`, RHS constant `+λ₁`) is the `dOff = 0` case. On the frozen
production path `dOff(2) = 7428.636632030943` is present, so the cone centre
`b_c` carries `d₂/(2·lamref) = 0.13821…` and the constant carries
`d₂/(2·lamref) + 1`. Nothing else changes.

## The remaining rows are affine

```
row 3:  [ −f_JJᵀ/lamref , 1 ] x ≤ λ_J/lamref
row 4:  [ 1ᵀ/Vtot       , 0 ] x ≤ (Vtot − Σρ₃₈₅)/Vtot
```

together with `xmin ≤ x ≤ xmax` and the objective `f = [0; …; 0; −1]`.

## Gradient and Hessian of row 1 (used by the nonlinear cross-check)

With `u = (a − c − d₂)/2`, `v = b`, `r = sqrt(u² + v²)`:

```
∂e₁/∂drho = 0.5·(F11 + F22) − ( u·0.5·(F11 − F22) + v·F12 ) / r
∂²e₁/∂drho² = − Jᵀ H_r J ,   J = [ 0.5·(F11 − F22)ᵀ ; F12ᵀ ],
                                H_r = [ v², −uv ; −uv, u² ] / r³
```

Row 2 uses `e₂ = trace − e₁`, so `∇²row2 = −∇²row1`. The Lagrangian Hessian
is `(μ₁ − μ₂)/lamref · Jᵀ H_r J`, a rank-2 operator applied matrix-free.
Both were verified against `deltaLambda`/`ddlam`: gradient relative ∞-norm
difference ≤ 1.3e−16 at all 45 test evaluations; Hessian-vector product vs a
central difference of the production gradient, relative error 1.1e−8 (row 1)
and 2.1e−7 (row 2) at step 1e−7.

## Smoothness margin

The cone apex (`u = v = 0`, i.e. `e₁ = e₂`) is where `deltaLambda` is
nonsmooth. The crude box bound `Σ_e max(|lo_e|,|hi_e|)·|F11_e − F22_e| = 19 405.7`
exceeds `d₂ = 7 428.6`, so the apex **cannot be excluded** from the box by
that bound. Consequences: (i) the convexity certificate is unaffected — a
second-order cone is convex whether or not its apex is reachable; (ii) the
nonlinear cross-check tracks the separation `e₂ − e₁ = 2·r·lamref` along every
solve; the minimum observed on any solver path was 7 428.6 (the drho = 0
value), and at the reference the separation is **7 474.35**, i.e. the optimum
moves *away* from coalescence, not toward it. `drho = 0` is not a nonsmooth
point on this path: `M(0) = diag(0, 7428.6)` has distinct eigenvalues.
