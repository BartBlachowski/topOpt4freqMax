# EFFECTIVE_OBJECTIVE — Part 12

## Verdict

```
EFFECTIVE_SCALAR_OBJECTIVE_NOT_IDENTIFIED
```

Not "we failed to find one". **No such function exists locally**, and the
reason is a theorem rather than a search that came up empty.

## Why the question is closed rather than open

The preregistration permits this identification attempt *only if* the filtered
field passes the local integrability tests. It does not:

| test | filtered | physical control |
|---|---|---|
| median relative Jacobian asymmetry | **0.287**, δ-independent | 2.6e−06, ∝1/δ |
| median normalized closed-loop integral | **0.285** | 2.9e−06 |
| loop amplitude exponent | **1.999** | 0.978 |
| Frobenius skew ratio (mixed partials) | 5.59e−04 | 2.07e−06 |

By the converse of the Poincaré lemma, a continuously differentiable vector
field on a simply connected neighbourhood is a gradient **iff** its Jacobian is
symmetric there. The Jacobian of `g_filt` has a measured antisymmetric part that
does not vanish under finite-difference refinement, is predicted in closed form
from the operator structure, and produces closed-loop integrals scaling exactly
as a genuine curl. Therefore no scalar `F` with `∇F = g_filt` exists in a
neighbourhood of the frozen state.

Searching for a density-filtered, smoothed, weighted or otherwise regularized
functional whose gradient is `g_filt` is not merely unsuccessful — it is
searching for something the measurements exclude.

## What the field is instead

An exactly characterized non-gradient field:

```
g_filt(ρ) = A(ρ)·∇λ₁(ρ) ,     A(ρ) = diag(1/(Hs∘ρ))·H·diag(ρ)
J_filt    = A·D_{g/ρ} + A·Hess − diag(g_filt/ρ)      (verified to 1e−8)
```

The obstruction has two parts, whose shares were measured:

| mechanism | median share |
|---|---|
| S1 — ρ-weighting, `skew(A·D_{g/ρ})` | 0.009 |
| S2 — non-commutation, `skew(A·Hess)` | **0.991** |

## A conservative alternative does exist — and it is a different formulation

With `W = diag(1/Hs)·H`, the **density** filter defines a genuine composed
objective:

```
F(ρ) := λ₁(Wρ)  ⟹  ∇F(ρ) = Wᵀ ∇λ₁(Wρ) ,  ∇²F = Wᵀ·Hess(Wρ)·W
```

`Wᵀ H W` is symmetric for any `W` because `Hess` is symmetric, so this field is
a gradient by construction, with no condition on `W` at all.

The production sensitivity filter differs from it in three ways, each of which
independently breaks the identification:

1. it applies `A`, not `Wᵀ`;
2. it inserts a ρ-weighting that `Wᵀ` does not have;
3. it evaluates `∇λ₁` at **ρ**, not at `Wρ`, so there is no composed function.

The counterfactual table in `FILTER_OPERATOR_ANALYSIS.md` §6 shows that fixing
only (1) and (2) — i.e. using a symmetric, ρ-free operator — reduces the
antisymmetry ~40× but leaves it at 0.031, still nonzero. It is (3), the
composition structure, that makes a density filter conservative. Swapping
operators inside the existing scheme would not produce a scalar objective.

**This is stated as mathematics, not as a recommendation.** No filter was
changed, no projection or density-filter run was executed, and
`NEXT_ACTION.md` is explicit that adopting such a formulation would be a
departure from the printed Du & Olhoff choice and is a decision for the owner,
not a conclusion of this audit.
