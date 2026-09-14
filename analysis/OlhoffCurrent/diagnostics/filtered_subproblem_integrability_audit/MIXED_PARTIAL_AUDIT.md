# MIXED_PARTIAL_AUDIT — Part 7

## Method

For each element `j` in a deterministic set S, the **full** Jacobian column
`J(:,j)` is taken by central differences on that single element
(`δ_e = 1e−5`), which yields `∂g_i/∂ρ_j` for every `i` at once. The `|S|×|S|`
submatrix then gives both `J_ij` and `J_ji` for every pair. 30 elements, 61
evaluations, 21.3 s. Maximum box excursion: **0** — every perturbed density
stayed inside `[ρmin, 1]`.

Selection, fixed in the preregistration: 8 broad-gray-core (deepest), 8
gray-shell (shallowest gray), 6 solid-like (ρ > 0.9), 6 void-like (ρ < 0.1),
plus the 4 nearest neighbours of the deepest core element.

## Result

| | median rel. asym. | p90 | max | **Frobenius skew ratio** |
|---|---|---|---|---|
| physical control | 4.522e−02 | 1.010 | 1.988 | **2.073e−06** |
| filtered | 9.841e−01 | 1.001 | 1.885 | **5.593e−04** |

### Read the Frobenius ratio, not the median

The elementwise ratio `|J_ij − J_ji| / max(|J_ij|,|J_ji|)` is **not** a usable
statistic for distant pairs: the true entries there are essentially zero, so the
ratio is noise divided by noise and saturates near 1 for *both* fields. That is
why the physical control shows a median of 0.045 and a p90 of 1.01 — it is not
evidence of physical non-conservativity.

The scale-aware statistic is `‖J − Jᵀ‖_F / ‖J‖_F`, and it separates the two
fields by **270×**: 2.07e−06 (physical, i.e. numerical zero) against 5.59e−04
(filtered).

### In-stencil vs out-of-stencil

| | pairs | median rel. asym., filtered |
|---|---|---|
| within the filter stencil (`H_ej ≠ 0`) | 116 of 870 | **2.163e−01** |
| outside the stencil | 754 of 870 | 9.973e−01 (noise-dominated) |

The in-stencil value, 0.216, is where the measurement has signal, and it agrees
with the direction-pair result of 0.287 from Part 5.

### By class

| class | n | median rel. asym., filtered | physical |
|---|---|---|---|
| gray core | 8 | 7.365e−01 | 1.486e−02 |
| neighbour | 2 | 7.235e−01 | 2.102e−02 |
| gray shell | 8 | 9.822e−01 | 5.119e−02 |
| solid | 6 | 9.245e−01 | 2.338e−03 |
| void | 6 | 1.000e+00 | 9.844e−01 |

The void row is noise for both fields — single-element perturbations of 1e−5 at
ρ = 1e−3 produce a physical response near the FD floor. It is reported rather
than dropped.

## Testing the analytic decomposition

`FILTER_OPERATOR_ANALYSIS.md` §5 predicts

```
skew(J_filt) = skew(A·D_{g/ρ}) + skew(A·Hess)
```

with the first term in closed form. Testing that on the submatrix alone gave a
residual **larger** than the measured skew (ratio 1.58) — as it must, since the
`A·Hess` term also contributes in-stencil and was not subtracted. The
submatrix test cannot separate the two terms.

The decomposition was therefore verified two other ways, both of which succeed:

1. **Full-column test** (`fi_analytic_verify.m`). For 6 elements spanning all
   classes, the predicted column
   `A·J_phys(:,j) + A(:,j)·g_j/ρ_j − e_j·g_filt_j/ρ_j` matches the measured
   `J_filt(:,j)` to **9.6e−09** relative (9.7e−05 for the void element, whose
   diagonal term is ≈2e5). The identity is exact.
2. **Direction-pair budget** (`fi_decompose.m`). The measured antisymmetry
   closes as `S1 + S2` to ≤1e−3 on 9 of 10 pairs, with median shares
   S1 = 0.009 and S2 = 0.991.

So the submatrix asymmetry is real and correctly predicted; it simply cannot be
attributed to one term without the other, and the two tests above do that
attribution properly.

`figures/FIG_08_mixed_partial_asymmetry.*` shows both matrices.
