# FIRST_DIVERGENCE — Parts 14 and 15

```
FIRST_DIVERGENCE_OUTER_BOX_CONTROLLER
```

## 1. Definition (preregistration §9)

Levels evaluated on the M1 pairing — source code with the target material law + adaptive box vs
target C480 — at the earliest shared state ρ₀ (uniform 0.5). Controller and stopping fields are
excluded from D0 and belong to D7/D8. A first divergence is a location, not a cause.

## 2. Result at ρ₀

| level | question | result | evidence |
|---|---|---|---|
| D0 | problem/physics configuration | **identical** | M1 vs C480 differ only in `move.*`, `stop.*`, `runtime.*`, metadata (CONFIG_COMPARISON) |
| D1 | K, M, eigenpairs | **identical** | K and M SHA-256 equal; ω₁…ω₅ bitwise; both equal C480 in-run ω(1) bitwise |
| D2 | raw f_sk, f_JJ | **identical** | bitwise |
| D3 | filtered f_sk, f_JJ | **identical** | bitwise |
| D4 | N, cluster, dOff, next-mode row | **identical** | bitwise |
| D5 | problem-(25) rows and gradients at Δρ = 0 | **identical** | bitwise |
| D6 | inner solve for identical rows and identical box (0.04) | **identical** | Δρ, β, nInner bitwise; equals C480 in-run DRHO(1) bitwise |
| **D7** | box supplied by the outer controller | **DIFFERENT** | source d_e = 0.10 for all e; target 0.04 → ‖Δρ₁‖₂ 13.393 vs 4.266, max 0.09996 vs 0.03996, cos 0.960, predicted gain β − λ₁ 1959 vs 694 |
| D8 | stopping decision | different rule (ε-test vs terminal E) | — (downstream of D7) |

Verdict = lowest failing level = **D7**.

## 3. M0 (native) — reported separately

M0 additionally differs at D0 in the material law. That operator is exactly zero on ρ₀ and on every
state with all ρ_e > 0.1 (same-state S0 vs S1 at ρ₀ and C480 iteration 10: ω, f_sk, filtered f_sk and
inner step bitwise equal). On the source trajectory it activates at **outer iteration 6**
(P-prefix: S480 = M1 bitwise for iterations 1–5). On C480's trajectory the first ρ ≤ 0.1 appears at
the start of iteration 12. So in time, the native divergence sequence is:
**iteration 1: box (D7) → iteration 6: material law (D0, activated) → iteration ≈20: localized
modes in the SIMP arm → iteration 30: grayness separation between S480 and M1.**

## 4. Same-state cross-evaluation (Part 15)

Evaluators: **T** target `+impl` (SIMP/4b, C480 cfg); **S1** source snapshot with the M1 cfg
(SIMP/4b); **S0** source snapshot with the S480 cfg (Pedersen/eq.2). Separate MATLAB sessions,
isolated paths (`sd_use_target` through the production gate; `sd_use_source`). Output
`evaluations/same_state/*.mat`, comparison `evaluations/same_state_comparison.json`.

### 4a. Implementation identity (T vs S1)

At **all 9 states** (ρ₀; C480 k = 10, 20, 100, 386; S480 final; M1 k = 5, 11, 64): K, M, λ₁…λ₅,
N, J, multJ, dOff, raw f₁₁ f₂₂ f₁₂ f_JJ, filtered ones, problem-(25) values and gradients, and the
full inner solve with box 0.04 are **bitwise identical**. Answers to Q23–Q26 for the
implementation question: eigenvalues agree, raw gradients agree, filtered gradients agree, the first
local problem-(25) step agrees — exactly.

Evaluator validation against in-run records (all bitwise): T reproduces C480 Δρ and ω at iterations
1, 11, 21, 101; S1 reproduces M1 Δρ at iterations 1, 6, 12 with its native per-element box; S0 at ρ₀
equals S1 and S480's recorded ω(1).

### 4b. Formulation operator (S1 vs S0) at the same ρ

| state | ρ<0.1 | ω rel. (max, modes 1–5) | λ₁ rel. | raw f₁₁ L2 rel. | filtered f₁₁ L2 rel. | inner step cos (box 0.04) |
|---|---|---|---|---|---|---|
| ρ₀ | 0 | 0 | 0 | 0 | 0 | 1 (bitwise) |
| C480 k=10 | 0 | 0 | 0 | 0 | 0 | 1 (bitwise) |
| C480 k=20 | 0.121 | 2.2e−2 | 3.8e−3 | 0.33 | 0.31 | 0.990 |
| C480 k=100 | 0.240 | 1.3e−2 | 2.7e−3 | 0.45 | 0.23 | 0.925 |
| C480 k=386 | 0.356 | 1.0e−2 | 3.4e−3 | 0.45 | 0.25 | 0.758 |
| S480 final | 0.425 | 8.1e−3 | 2.6e−3 | 0.44 | 0.25 | 0.608 |
| M1 k=5 | 0.093 | 9.0e−2 | 6.4e−3 | 0.25 | 0.21 | 0.987 |
| M1 k=11 | 0.192 | 1.3e−2 | 2.6e−3 | 0.40 | 0.19 | 0.983 |
| M1 k=64 | 0.349 | 12.7 (localized) | 21.7 | 1.00 | 1.00 | −0.229 |

### 4c. Different trajectory vs different operator

- For every kernel on the shared path the operator is the **same**; the implementations follow
  different trajectories because (i) the controller supplies a different box from iteration 1 and
  (ii) the source's material law changes the operator once low densities appear.
- Figures: `fig13` chain, `fig15` eigenvalues, `fig16` raw gradients, `fig17` filtered gradients,
  `fig18` first problem-(25) step at ρ₀.
