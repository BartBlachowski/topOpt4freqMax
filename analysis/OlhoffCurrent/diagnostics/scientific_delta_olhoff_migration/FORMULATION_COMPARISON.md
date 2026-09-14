# FORMULATION_COMPARISON — Part 7 (Pedersen vs eq.(4) distinction)

```
SOURCE_TARGET_DIFFERENT_SCIENTIFIC_FORMULATION
```

## 1. What `duOlhoffAdaptivePedersen` does (source 6b08708)

Resolution chain: `duOlhoffFrozenM4` → `duOlhoffAdaptiveMove` → `duOlhoffAdaptivePedersen`.
Material fields in force (verified in every S/Rel `res.mat`):

| | source (`duOlhoffAdaptivePedersen`) | target (`duOlhoffFrozenM4`, both C480 and production) |
|---|---|---|
| stiffness law | Pedersen (2000) eq.(5): g_K = ρ³ for ρ ≥ 0.1; g_K = ρ·0.1² = ρ/100 for ρ < 0.1 | SIMP eq.(1): g_K = ρ³ everywhere |
| stiffness derivative | 3ρ² (ρ ≥ 0.1); 0.01 (ρ < 0.1) — C⁰ law, derivative jumps 0.03 → 0.01 at 0.1 | 3ρ² |
| mass law | eq.(2), q = 1: g_M = ρ everywhere | eq.(4b) C¹: ρ (ρ > 0.1); 6·10⁵ρ⁶ − 5·10⁶ρ⁷ (ρ ≤ 0.1) |
| mass derivative | 1 | 1 (ρ > 0.1); 3.6·10⁶ρ⁵ − 3.5·10⁷ρ⁶ (ρ ≤ 0.1) |
| low-density threshold | stiffness only, strict `ρ < 0.1` | mass only, `ρ ≤ 0.1` |
| eq.(4)/(4b) mass modification active? | **No.** eq.(2) replaces it (fields `lowDensityExponent`, `cutoff` are carried but unused) | Yes |
| Pedersen replaces or complements? | **Replaces** the mass cut-off: Pedersen stiffness + plain linear mass, as Pedersen (2000) uses it | n/a |
| p | 3, fixed | 3, fixed |
| ρ_min | 1e−3 | 1e−3 |

Element factors (`evaluations/lowdensity_kkt.json`):

| ρ | 0.001 | 0.01 | 0.05 | 0.1 | 0.2 | 0.5 | 1 |
|---|---|---|---|---|---|---|---|
| K, SIMP | 1e−9 | 1e−6 | 1.25e−4 | 1e−3 | 8e−3 | 0.125 | 1 |
| K, Pedersen | **1e−5** | **1e−4** | **5e−4** | 1e−3 | 8e−3 | 0.125 | 1 |
| M, eq.(4b) | 6.0e−13 | 5.5e−7 | 5.5e−3 | 0.1 | 0.2 | 0.5 | 1 |
| M, eq.(2) | **1e−3** | **1e−2** | **5e−2** | 0.1 | 0.2 | 0.5 | 1 |
| M/K, SIMP + eq.(4b) | 6e−4 | 0.55 | 43.8 | 100 | 25 | 4 | 1 |
| M/K, Pedersen + eq.(2) | 100 | 100 | 100 | 100 | 25 | 4 | 1 |

## 2. Answer

**Is the successful source sweep solving the same relaxed material problem as OlhoffCurrent? No.**

- The objective, constraints, filter, multiplicity model and bounds are the same (CONFIG_COMPARISON,
  same-state identity). The **relaxed material law differs in the band ρ < 0.1 (stiffness) and
  ρ ≤ 0.1 (mass)** and is identical above it.
- The difference is **exactly zero** on any state with every ρ_e > 0.1. It is proven, not argued:
  the source code run with the target's law (M1) reproduces the committed Pedersen run S480x60
  **bitwise for outer iterations 1–5**, and the two first differ at iteration k* = 6, the first
  iteration whose start state has an element at or below 0.1 (min ρ 0.102 → 0.0037).
- It is **not small once active**. At identical frozen designs (source code, S0 = Pedersen/eq.2 vs
  S1 = SIMP/eq.4b):

| state | fraction ρ<0.1 | λ₁ rel. diff | raw f₁₁ L2 rel. (all / ρ≥0.1 / ρ<0.1) | filtered f₁₁ L2 rel. (all / ρ≥0.1) | first inner step cos (box 0.04) |
|---|---|---|---|---|---|
| C480 iter 20 | 0.121 | 0.38 % | 0.33 / 0.005 / 2.06 | 0.31 / 0.017 | 0.990 |
| C480 final 386 | 0.356 | 0.34 % | 0.45 / 0.003 / 3.30 | 0.25 / 0.004 | 0.758 |
| S480 final | 0.425 | 0.26 % | 0.44 / 0.002 / 3.42 | 0.25 / 0.004 | 0.608 |
| M1 iter 5 | 0.093 | 0.64 % | 0.25 / 0.014 / 1.07 | 0.21 / 0.025 | 0.987 |
| M1 final 64 | 0.349 | **λ₁ 34.36 vs 163.55 rad/s** | 1.00 (different mode) | 1.00 | −0.229 |

  On a black-and-white final design the frequency effect is below 1 % (the reason the sweep can
  re-evaluate under SIMP + eq.(4) with < 0.5 % change). The **gradient** and hence the
  **step** differ strongly in the void band, and on designs with gray islands (M1 final) the SIMP
  law produces spurious localized modes that Pedersen does not (MASS_STIFFNESS_COMPARISON.md).

**Quantified formulation difference.** Δg_K(ρ) = ρ/100 − ρ³ for ρ < 0.1 (10⁴× stiffer void at
ρ_min); Δg_M(ρ) = ρ − (6·10⁵ρ⁶ − 5·10⁶ρ⁷) for ρ ≤ 0.1 (≈10⁹× heavier void at ρ_min); M/K bounded
at 100 under Pedersen, non-monotone 6e−4…100 under SIMP/4b.

## 3. Classification

- Pedersen stiffness: **class B/D.** Du & Olhoff (2007) §2.2 *name* Pedersen (2000) as an
  alternative remedy to the mass cut-off; they do not use it. Relative to OlhoffCurrent it changes
  the relaxed physics → a new scientific formulation, not a bug fix. The target's SIMP + eq.(4b)
  is itself a printed Du–Olhoff choice and is internally consistent (no class F).
- Linear mass eq.(2): **class A (printed option) / D relative to target.**
- Source's own provenance labels it "class B in the Pedersen scheme" and the preset a
  "RECONSTRUCTION (class C)". Both labels are compatible with this audit's D classification
  relative to OlhoffCurrent: the formulation is defensible, but it is a different one.

**It must not be migrated under the name `duOlhoffFixedPenaltySensitivityFiltered`**: that name
describes SIMP + eq.(4b), and the axis that changed (low-density material law) is not in it.
See MIGRATION_CLASSIFICATION.md §Preset identity.
