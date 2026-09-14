# REFERENCE_SOLUTION — Part 10

The certified global solution of the exact frozen production problem (25) at
the authoritative 480×60 state (outer 386, built at ρ₃₈₅, move 0.01).
Source: `evaluations/reference_solution.json`, `evaluations/conic_reference.mat`.

## Objective

| quantity | value |
|---|---|
| `bs` = β/λ₁ | **1.001792128848223** (certified: no feasible point exceeds 1.001792128857) |
| physical β = `bs·lamref` | **26 921.772248** |
| predicted ω₁ = √β | 164.078555 (current ω₁ = 163.931727) |
| predicted relative eigenvalue gain `bs − 1` | **1.7921e−3** (0.179 %) |
| production P19 gain for comparison | 9.6e−6 (`bs_P19 = 1.0000096761`) |

## Increment

| quantity | value |
|---|---|
| max\|drho\| | 0.00999999999888 |
| **max\|drho\| / move** | **1.0000** (to 1.1e−10) |
| ‖drho‖₂ | 1.003271 |
| mean\|drho\| | 4.056e−3 (40.6 % of move on average) |
| RMS drho | 5.912e−3 |
| Σdrho (volume change, absolute) | +0.04580 elements |
| Σdrho / Vtot | +3.18e−6 (exactly the volume slack available at ρ₃₈₅) |
| elements with drho > 0 / < 0 / = 0 | 14 484 / 14 316 / 0 |
| image `(a, b, c)` = `(F11ᵀdrho, F12ᵀdrho, F22ᵀdrho)` | (48.16098, −0.103738, 93.87786) |
| `f_JJᵀdrho` | −1 185.80 |

## Constraints at the reference (production scaling)

| row | value | status | multiplier |
|---|---|---|---|
| 1 cluster (min eigenvalue) | 1.5e−14 | **active** | μ₁ = 1.0000000000 |
| 2 cluster (second eigenvalue) | −0.27813 | slack | 0 |
| 3 next mode (J = 3) | −4.94958 | slack | 4e−17 |
| 4 volume | 4.0e−14 | **active** | ν₂ = 0.7291136447 |
| box | 28 784 of 28 800 variables on a bound | **active** | max ξ 7.55e−3, max η 8.6e−4 |

KKT: `REFERENCE_PROBLEM25_KKT_PASS` (`REFERENCE_KKT.md`). Duality gap
9.2e−12 ⇒ `GLOBAL_PROBLEM25_REFERENCE_CERTIFIED`.

Eigenvalue separation of the sub-eigenvalue matrix at the reference:
`e₂ − e₁ = 7 474.35` (1.0062 × the current λ₂ − λ₁). The optimum does not
approach coalescence.

## Active bounds (tolerance 1e−6 × width; sweep in `REFERENCE_KKT.md`)

| at −move | at ρ floor | at +move | at ρ ceiling | interior |
|---|---|---|---|---|
| 4 906 | 9 404 | 4 788 | 9 686 | 16 |

Is it move-bound dominated? By the preregistered rule (≥ 50 % of variables at
a move-limited bound): **no, 33.7 %**. By structure: **it is a bound-saturated
bang-bang step** — every gray element (0.1 ≤ ρ ≤ 0.9) but 12 of 8 272 sits at
±move, the void sits on the density floor and the solid on the ceiling
(`ACTIVE_SET_ANALYSIS.md`). The answer to "does the move box determine the
solution" is yes for every element whose box side is the move limit.

## Spatial structure (figures 1, 10, 13)

`FIG_01_reference_drho`: the increment is ±move inside the gray band around
the members, floor-limited in the void, ceiling-limited in the solid.
`FIG_10_reference_active_bounds`: the four bound classes as a map.
`FIG_13_reduced_gradient_map`: the reduced gradient `q_e`, whose sign decides
the side; it is small only on a thin set of elements.

## What sets the solution — an interpretation check (`evaluations/lp_insight.json`)

The certificate's cone direction is `w = [−1.0000, −2.8e−5]`: at this state
the minimum-eigenvalue cone acts almost exactly like the **first-mode linear
functional** `a = F11ᵀdrho`. Solving the linear program
`max F11ᵀdrho s.t. Σdrho ≤ slack, box` with `linprog` (dual simplex) and
evaluating its increment through the production `deltaLambda` gives
`bs_LP = 1.001791054`, i.e. **99.94 % of the certified gain**, with the same
bound side on 99.91 % of the elements (cosine 0.9973 to the reference). The
true optimum is, to that accuracy, a **threshold rule on the filtered
first-mode sensitivity**: element e goes to its upper bound if
`F11_e/lamref > ν₂/Vtot = 5.06e−5` and to its lower bound otherwise. The
sub-eigenvalue coupling (b, c) contributes the remaining 6e−4 of the gain.

## Independence of the result

Six `fmincon` interior-point solves on the literal four-row production problem
(`deltaLambda` + `ddlam`, exact Lagrangian Hessian) from six deterministic
starts reach `bs = 1.00179189 … 1.00179190`, i.e. **2.3e−7 below** the
certified optimum (inside the preregistered 1e−6 agreement bar; the deficit is
the interior-point barrier offset of ~4 500 near-degenerate coordinates that
stop 1e−4·width inside their bounds), with cosine 0.9997 to the reference
increment (`FMINCON_CROSSCHECK.md`). Nothing found by any method exceeds the
certified bound.

## Primal uniqueness

`bs` is unique. `drho` is unique only up to motions inside the face spanned
by the ~20 coordinates with reduced gradient below 1e−6·max\|q\| and, more
loosely, along the `c`-functional (raising λ₂ alone changes `e₁` only at second
order), which is why `fmincon` lands 0.023 away in ‖·‖₂ with the same objective
to 2.3e−7. Every comparison against MMA states below uses the certified
`coneprog` point as "the reference"; its objective is what is certified.

`drho_ref` was never added to any density.
