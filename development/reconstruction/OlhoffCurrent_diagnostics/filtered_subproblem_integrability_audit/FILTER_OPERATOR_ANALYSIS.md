# FILTER_OPERATOR_ANALYSIS — Part 8

## 1. The operator, exactly as implemented

`+impl/filter/applyFilter.m`, verbatim:

```matlab
den = flt.Hs .* max(1e-3, rho);
df(:,c) = (flt.H * (rho .* df(:,c))) ./ den;
```

with `H_ei = max(0, rmin − dist(e,i))` and `Hs_e = Σ_i H_ei` from `prepFilter.m`
(the top88 `ft = 1` form). Elementwise:

```
g_filt(e) = Σ_i H_ei ρ_i g_phys(i) / ( Hs_e · max(1e-3, ρ_e) )
```

**The `max(1e-3, ρ)` guard is never active here.** The solver clamps
`ρ ≥ ρmin = 1e-3`, and at the frozen state exactly **0 of 28 800** elements have
`ρ < 1e-3`. So on the admissible set the guard is the identity and

```
g_filt = A(ρ) · g_phys ,        A_ei = H_ei ρ_i / (Hs_e ρ_e)
A(ρ)   = diag(1 ./ (Hs∘ρ)) · H · diag(ρ)
```

Verified numerically: `H` is exactly symmetric, `nnz(H) = 1 037 580`, and
`rowsum(H) − Hs = 0` exactly.

## 2. Is H symmetric? Yes. Is A? No.

| quantity | value |
|---|---|
| `H` symmetric | **yes**, exactly |
| `A` symmetric | **no** |
| ‖A − Aᵀ‖_F / ‖A‖_F | **1.4135** |

For a matrix whose symmetric and antisymmetric parts are comparable, that ratio
approaches √2 ≈ 1.41421. **A is essentially maximally asymmetric in the
Frobenius sense.** This is not a small perturbation of a symmetric operator.

### The exact symmetry condition

`A_ei = A_ie` requires, for every pair with `H_ei ≠ 0`,

```
H_ei ρ_i /(Hs_e ρ_e) = H_ei ρ_e /(Hs_i ρ_i)   ⟺   ρ_i² Hs_i = ρ_e² Hs_e
```

i.e. the quantity `ρ² Hs` must be constant across every filter stencil.
Measured over the 1 008 780 off-diagonal stencil pairs at the frozen state:

| statistic of the relative violation | value |
|---|---|
| median | 0.0503 |
| p90 | 0.9508 |
| max | 1.0000 |
| fraction below 1e−6 | 0.1157 |

The condition fails almost everywhere, and fails *hardest* exactly where ρ
varies most — at the interfaces and at the gray/void boundary.

## 3. A is not an averaging operator

Row sums are **not** 1:

```
Σ_i A_ei = (Hρ)_e /(Hs_e ρ_e) = ρ̃_e / ρ_e ,     ρ̃ = (Hρ)/Hs
```

verified to 1.42e−13. Measured range of `ρ̃/ρ`: **[0.668, 245.09]**.

So the "filter" multiplies the local sensitivity by up to **245×** where ρ is at
its floor and the neighbourhood is not. Measured RMS amplification by class:

| class | RMS \|g_filt\| / RMS \|g_phys\| |
|---|---|
| void (ρ < 0.1) | **30.67** |
| gray shell | 0.855 |
| gray core | 0.987 |
| solid (ρ > 0.9) | 0.931 |

In gray and solid material the filter roughly preserves magnitude; in void it
amplifies by more than an order of magnitude. `figures/FIG_02_*` maps this.

## 4. Is A ρ-dependent? Yes — but that is not the main problem

A depends on ρ through both `diag(ρ)` and `diag(1/ρ)`. Differentiating,

```
J_filt := ∂g_filt/∂ρ = A·D_{g/ρ}  +  A·Hess  −  diag(g_filt/ρ)          (★)
```

with `D_{g/ρ} = diag(g_phys/ρ)` and `Hess = ∂²λ₁/∂ρ²`. The first term is the
ρ-dependence of A; the second is A applied to the physical Hessian; the third is
diagonal, hence symmetric, hence irrelevant to any curl.

**(★) is verified, not assumed.** Testing it column by column against central
finite differences on full 28 800-vectors (`fi_analytic_verify.m`):

| class | ρ | relative error of (★) |
|---|---|---|
| gray core | 0.5643 | 9.6e−09 |
| gray core | 0.5643 | 9.8e−09 |
| gray shell | 0.2100 | 5.4e−09 |
| gray shell | 0.4724 | 1.2e−08 |
| solid | 0.9999 | 9.5e−11 |
| void | 0.001015 | 9.7e−05 |

The void row is larger only because its diagonal term is ≈ 2e5 and FD noise is
relatively larger there. The decomposition is exact.

## 5. Which term destroys integrability

`g_filt` is a gradient field iff `J_filt` is symmetric. The diagonal term
cancels, so

```
skew(J_filt) = skew(A·D_{g/ρ})  +  skew(A·Hess)
             =:      S1         +       S2
```

`S1` has a **closed form** needing no derivatives at all:

```
S1(e,j) = H_ej [ g_j /(Hs_e ρ_e) − g_e /(Hs_j ρ_j) ]
```

Both were evaluated on the ten preregistered direction pairs and compared with
the directly measured antisymmetry (`fi_decompose.m`). The identity
`measured = S1 + S2` closes to ≤ 1e−3 on 9 of 10 pairs (the exception is the
pair with the smallest asymmetry, where FD noise dominates). Shares:

| | median share of the measured antisymmetry |
|---|---|
| S1 — the ρ-weighting term | **0.009** |
| S2 — the A·Hess term | **0.991** |

**The dominant mechanism is not the ρ-weighting.** It is that A does not
commute with the Hessian in the required sense.

## 6. Under what conditions would `A·g_phys` be integrable?

From (★), with A constant the Jacobian is `A·Hess`, which is symmetric iff

```
A·Hess = Hess·Aᵀ
```

For a symmetric A this is exactly the commutation condition `A·Hess = Hess·A`.
So a sufficient set of conditions is:

1. **A constant in ρ** (kills S1), **and**
2. **A symmetric**, **and**
3. **A commutes with the Hessian.**

The production A satisfies **none** of the three. Condition 3 is the binding one
and cannot be arranged by construction, because Hess changes with the design.

### This was tested, not merely argued

Antisymmetry of `M·Hess` on the same ten pairs, using the measured
Hessian-vector products, for four candidate operators
(`fi_counterfactual_operators.m` — diagnostic only, no production file changed,
no run uses these):

| operator | symmetric | median relative antisymmetry |
|---|---|---|
| `A` — production sensitivity filter | no | **1.2274** |
| `B = diag(1/Hs)·H` — plain weighted average, ρ-free | no | 0.0567 |
| `Bsym = (B+Bᵀ)/2` — symmetrized | **yes** | **0.0314** |
| `Bᵀ` | no | 0.0324 |

Removing the ρ-weighting improves matters ~20×; symmetrizing on top of that
gains only a further ~1.8× and **does not reach zero**. Exactly as the
derivation predicts: symmetry alone is insufficient without commutation.

## 7. The contrast with a density filter

This is the mathematically important comparison, and it is a theorem, not a
measurement. With `W = diag(1/Hs)·H` and the *density* filter `ρ̃ = Wρ`, define

```
F(ρ) := f(Wρ)     ⟹     ∇F(ρ) = Wᵀ ∇f(Wρ)
```

which is a gradient **by construction**, and its Jacobian

```
∇²F = Wᵀ · Hess(Wρ) · W
```

is symmetric for *any* W, because `Hess` is symmetric: `(Wᵀ H W)ᵀ = Wᵀ Hᵀ W =
Wᵀ H W`. No commutation is needed.

Three structural differences separate the production sensitivity filter from
this:

1. it applies **A, not Wᵀ** — the wrong side of the transpose;
2. it inserts a **ρ-weighting** that Wᵀ does not have;
3. it evaluates `∇f` **at ρ**, not at the filtered field `Wρ`, so there is no
   composed function to differentiate.

The `Bᵀ` row of the table in §6 shows that swapping the operator alone is not
the fix — it is the **composition structure** that makes a density filter
conservative, not the transpose.

## 8. Volume constraint (Part 9)

The volume row gradient in `innerLoop` is

```matlab
dfdx(N+2,1:NE) = 1/Vtot;
```

a constant, identical on every element and **independent of ρ**. Therefore

```
∂/∂ρ_j [ λ_V / Vtot ] = 0   for every j
```

so the volume term contributes a constant vector field, whose Jacobian is the
zero matrix. Hence

```
skew(J_reduced) = skew(J_filt + const) = skew(J_filt)
```

**exactly**. Adding the volume multiplier cannot create, remove or alter any
antisymmetry, for any value of the multiplier. Objective-gradient integrability
and KKT reduced-gradient stationarity are therefore separate questions, and the
volume constraint bears on the second only. This is a proof, not an estimate.
