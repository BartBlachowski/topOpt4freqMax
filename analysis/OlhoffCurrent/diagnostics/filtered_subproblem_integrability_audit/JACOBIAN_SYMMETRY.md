# JACOBIAN_SYMMETRY — Part 5

## Method

`J = ∂g/∂ρ` is never assembled. For each preregistered direction, `Jd` is
estimated by central differences about the **frozen** ρ₃₈₆:

```
Jd ≈ [ g(ρ + δd) − g(ρ − δd) ] / (2δ) ,   ‖d‖₂ = 1
```

Statistic per pair (u,v) and step δ:

```
r = |uᵀJv − vᵀJu| / max(|uᵀJv|, |vᵀJu|)
```

A field with a scalar potential has a symmetric Jacobian, so `r` must vanish as
δ → 0, limited only by the FD noise floor. 88 evaluations, 31.2 s. No density
was updated; every perturbed ρ is a temporary evaluation point.

The identical test on `g_phys` is the **mandatory positive control**: λ₁ is a
scalar function of ρ, so its gradient is conservative wherever λ₁ is simple and
smooth.

## Result

| δ | median r, **filtered** | median r, **physical control** |
|---|---|---|
| 1e−3 | **2.8675e−01** | 2.5567e−06 |
| 3e−4 | **2.8675e−01** | 9.3588e−06 |
| 1e−4 | **2.8681e−01** | 3.5864e−05 |
| 3e−5 | **2.8701e−01** | 1.4178e−04 |

Two facts decide the question:

1. **The filtered asymmetry is independent of δ** — 0.28675 → 0.28701 across a
   33× change in step. That is the signature of a genuine, nonzero
   antisymmetric part of the Jacobian.
2. **The physical control behaves exactly like finite-difference noise** — it
   *grows* as δ shrinks, ∝ 1/δ, from 2.6e−06 to 1.4e−04. Extrapolated to the
   optimal step it is consistent with zero.

At δ = 1e−3 the two differ by **five orders of magnitude**.
`figures/FIG_05_symmetry_vs_step.*` shows flat lines for the filtered field and
a clean 1/δ slope for the control.

## Per-pair relative asymmetry r

Columns are δ = 1e−3, 3e−4, 1e−4, 3e−5.

| pair | filtered | physical control |
|---|---|---|
| D1a,D1b | 1.85e−01 1.85e−01 1.85e−01 1.86e−01 | 2.77e−05 5.70e−04 5.28e−04 1.89e−03 |
| D2a,D2b | 2.80e−02 2.79e−02 2.80e−02 2.79e−02 | 2.38e−06 1.07e−05 1.63e−06 1.97e−06 |
| D1a,D2a | **1.60e+00** ×4 | 6.65e−08 5.66e−08 3.19e−06 3.34e−06 |
| D3,D2a | 8.15e−01 ×4 | 2.73e−06 9.11e−06 2.21e−05 1.67e−04 |
| D4a,D4b | 4.38e−03 4.28e−03 4.27e−03 4.27e−03 | 1.69e−06 1.61e−06 7.16e−06 9.19e−06 |
| D1a,D4a | 1.33e−01 ×4 | 4.38e−06 5.33e−06 2.97e−04 1.56e−04 |
| D2a,D4a | 7.78e−01 ×4 | 1.14e−06 1.47e−06 1.10e−05 1.28e−04 |
| D5D1a,D5D1b | 1.85e−01 ×4 | 2.67e−05 5.72e−04 5.28e−04 1.89e−03 |
| D5D2a,D5D4a | 3.88e−01 ×4 | 4.42e−07 9.60e−06 4.96e−05 9.05e−05 |
| D3,D4b | **1.97e+00** ×4 | 2.57e−05 2.31e−05 6.61e−05 4.37e−04 |

Every filtered value is constant to 3–4 significant figures across the whole δ
range. Values near 2 mean `uᵀJv` and `vᵀJu` have **opposite signs** — the
Jacobian is not merely asymmetric there, it is nearly antisymmetric.

The two smallest asymmetries are informative: `D4a,D4b` (two independent random
fields, 4.3e−03) and `D2a,D2b` (two disjoint bumps deep inside the gray core,
2.8e−02). Non-conservativity is weakest between spatially separated,
statistically similar probes and strongest whenever one direction is a
localized gray-core or interface feature and the other is not — precisely where
the §2 symmetry condition `ρ²Hs = const` fails hardest.

The volume-neutral variants (`D5*`) reproduce their parent values exactly
(`D5D1a,D5D1b` = `D1a,D1b` to 3 digits), which is the numerical counterpart of
the proof in `FILTER_OPERATOR_ANALYSIS.md` §8 that a constant field cannot
affect any curl.

## Multiplicity screening (Part 10)

| | |
|---|---|
| samples screened | 88 |
| excluded (`gap12 < 0.05`) | **0** |
| minimum `gap12` observed | 0.12979 |
| eigenvalue-ordering changes | **0** |

`gap12` at the frozen state is 0.1298 and never moved below 0.1297 under any
perturbation. No sample crossed a multiplicity or ordering boundary, so no
result here can be an artefact of mode switching. The preregistered 5 %
exclusion rule was never triggered and no δ was dropped.

## Verdict contribution

Preregistered conditions 1–3 for `FILTERED_FIELD_LOCALLY_NONCONSERVATIVE`:

1. median r ≥ 0.05 at the two smallest δ — **0.2868, 0.2870** ✓
2. r does not vanish under refinement (smallest δ ≥ 0.5× largest) — **flat** ✓
3. positive control ≤ 0.01 at the same δ — **3.6e−05, 1.4e−04** ✓

Condition 4 (closed loops) is in `CLOSED_LOOP_INTEGRALS.md`.
