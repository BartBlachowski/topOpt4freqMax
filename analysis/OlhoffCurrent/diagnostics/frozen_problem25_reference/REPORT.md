# BOTTOM LINE

**The exact frozen problem (25) that production poses at the authoritative
480×60 state is a convex second-order cone program, and its global solution
is now known and certified: `bs = 1.001792128848` (β = 26 921.77, a 0.179 %
predicted gain in λ₁), a bang-bang vertex of the box with 99.94 % of the
28 800 increments on a bound and every gray element at the full ±0.01 move.
Production's repeated-MMA step captures 0.28 % of that gain. Run to 5 000
sub-iterations it captures 54 %, with an increment a quarter the size of the
solution's, drifting slowly toward the vertex without converging to it.
Repeated MMA does not solve problem (25) at this state.**

Zero topology-optimization runs, zero outer iterations, zero density updates.
Production untouched.

## Verdicts

```
FROZEN_480_PROBLEM25_STATE_PASS
CLUSTER_CONSTRAINT_REDUCTION_PASS
FROZEN_PROBLEM25_SOCP_EQUIVALENCE_PASS
FROZEN_PROBLEM25_CONVEXITY_CERTIFIED
REFERENCE_PROBLEM25_KKT_PASS
GLOBAL_PROBLEM25_REFERENCE_CERTIFIED
REPEATED_MMA_REALIZATION_FAILS_PROBLEM25_REFERENCE
FROZEN_INNER_SOLVER_STUDY_JUSTIFIED
```

## The evidence in one table

| | reference (certified) | P19 production | M500 | M5000 |
|---|---|---|---|---|
| `bs` = β/λ₁ | **1.0017921288** | 1.0000049637 | 1.0007024808 | 1.0009624400 |
| fraction of achievable gain lost, G | 0 | **0.997** | 0.608 | **0.463** |
| ‖drho‖₂ | 1.003 | 0.008 | 0.145 | 0.274 |
| max\|drho\|/move | 1.000 | 0.062 | 0.876 | 0.979 |
| variables on a bound (1e−6·width) | 28 784 | 0 | 0 | 0 |
| cosine to reference | 1 | 0.669 | 0.598 | 0.707 |
| KKT residual, normalized RMS | 1.9e−15 (certificate duals) | 0.360 | 0.586 | 0.597 |
| duality gap | **9.2e−12** | — | — | — |

---

## Direct answers

**1. Was the exact authoritative frozen 480 state recovered?** Yes. ρ₃₈₆
`0a498a7d…e4a60`, ρ₃₈₅ `9b1443e5…3ded`, Δρ₃₈₆ `0b12cd7f…9fd`, config
`03097a28…782e`, `+impl` `edbfe47e…52cb` verified; outer 386, stage 3, move
0.01, N = 2, `multJ` = 0, `nInner` = 19. A fresh FE evaluation at ρ₃₈₅ through
the production functions reproduces the retained inner-loop context
**bitwise**. `STATE_IDENTITY.md`.

**2. Were zero outer/topology optimization runs executed?** Yes, zero.

**3. Were zero density updates accepted?** Yes, zero. Every `drho` was
discarded after metrics; `rho + drho` was never formed as a design.

**4. What exact problem is production attempting to solve?** Minimize `−bs`
over `x = [drho; bs]` subject to `bs·λ₁ ≤ λ₁ + e_j(diag(0, λ₂−λ₁) + A(drho))`
for j = 1, 2 (production rows 1–2 via `deltaLambda` with offsets),
`bs·λ₁ ≤ λ₃ + f_JJᵀdrho`, `Σ(ρ₃₈₅ + drho) ≤ 0.5·NE`, and
`max(ρmin−ρ, −0.01) ≤ drho ≤ min(1−ρ, +0.01)`, `0 ≤ bs ≤ 5`; `F`, `f_JJ`
sensitivity-filtered. The move box, offsets, scaling, `bs` bounds and stop
rule are reconstruction choices. `EXACT_PROBLEM25.md`.

**5. Is the frozen clustered problem N = 2?** Yes, fixed subspace N = 2, J = 3.

**6. Is dOff active or absent?** **Present**: `dOff = [0, 7428.636632030943]`
= `[0, λ₂ − λ₁]` exactly, contrary to the task prompt's expectation. The cone
was derived for the offset form.

**7. Are the higher clustered spectral constraints globally redundant?** Yes.
`λ_j + Δλ_j = λ₁ + e_j` exactly, and `e₁ ≤ e₂` for every drho, so row 1 implies
row 2 everywhere — proved from the code and verified at 45 evaluations (row-2
slack at the reference 0.278). `CLUSTER_CONSTRAINT_REDUCTION.md`.

**8. Is the active clustered constraint exactly representable as a 2×2 SOC?**
Yes: `‖[(a−c−d₂)/2; b]‖ ≤ (a+c+d₂)/2 + λ₁ − λ₁·bs`. `SOCP_DERIVATION.md`.

**9. Did numerical SOCP equivalence tests pass?** Yes. Value difference
≤ 6.9e−15, gradient ≤ 1.3e−16 relative, class agreement 45/45, affine rows
≤ 3.1e−14, objective 0, `secondordercone` convention confirmed by toy test.
`SOCP_EQUIVALENCE.md`.

**10. Is the exact frozen problem convex?** Yes — certified: linear objective,
one second-order cone, two half-spaces, a box. `CONVEXITY_CERTIFICATION.md`.

**11. Was coneprog or another conic solver available?** Yes, MATLAB R2025b
`coneprog`. Nothing installed.

**12. Did it converge?** Its primal converged (rows feasible to 4e−14) but its
dual stalled: exitflag −7 at the preregistered 1e−10 in every configuration.
The independent maximized dual bound certifies the primal anyway (gap
6.4e−12 / 9.2e−12 aligned). The only exitflag-1 runs (1e−8) stop 3.2e−5 short
and were rejected. `CONIC_REFERENCE.md`.

**13. Can its solution be certified as globally optimal for the frozen
problem?** Yes: convexity certified, exact-production KKT PASS with the
certificate multipliers, and weak duality shows no feasible point exceeds
`bs = 1.001792128857`. `GLOBAL_PROBLEM25_REFERENCE_CERTIFIED`.

**14. Did independent fmincon methods agree?** Yes. Seven exact-Hessian
interior-point solves (starts: zero, P19, M500, M5000, ½·xmax, ½·xmin, seeded
random) reach `bs = 1.00179189 … 1.00179190`, 2.3e−7 below the certificate
(inside the 1e−6 bar, on the correct side); the one completed L-BFGS run
7.4e−7 below. Same volume and `a` functional; `c` differs along a
second-order-flat direction (primal non-uniqueness, preregistered). SQP not
viable at n = 28 801. `FMINCON_CROSSCHECK.md`.

**15. What is the reference beta?** `bs = 1.001792128848`, β = 26 921.772248,
predicted ω₁ = 164.0786 (current 163.9317).

**16. What is max|drho_ref| / move?** 1.0000 (0.999999999888).

**17. Is the reference solution move-bound dominated?** By the preregistered
rule (≥ 50 % at a move-limited bound): **no, 33.7 %**. Structurally: it is
bound-saturated — 99.94 % of variables on a bound; every gray element but 12 of
8 272 at ±move; void on the floor, solid on the ceiling. The move limit binds
every element whose box side is the move limit. `ACTIVE_SET_ANALYSIS.md`.

**18. Which constraints are active?** The minimum-eigenvalue cone (μ = 1.000)
and the volume (ν = 0.7291), plus the box on 28 784 variables. Next mode slack
4.95; second cluster row slack 0.278.

**19. Does the reference satisfy KKT?** Yes: primal 4e−14, dual ≥ 0,
complementarity 2.9e−14 (rows) and 5.0e−7 (box, normalized), stationarity
1.9e−15 RMS with the certificate multipliers. With `coneprog`'s own stalled
duals it would not (0.42) — reported, not used. `REFERENCE_KKT.md`.

**20. How far is production P19 from the reference?** G = 0.997 (loses 99.7 %
of the gain); ‖Δ‖₂/‖ref‖ = 0.995; cosine 0.669; its norm is 0.8 % of the
reference's.

**21. How far is M500?** G = 0.608; distance 0.921; cosine 0.598; norm 14 %.

**22. How far is M5000?** G = 0.463; distance 0.830; cosine 0.707; norm 27 %;
no variable within 1e−6 of a bound; median gray increment 15 % of the
reference's.

**23. Does repeated MMA approach the reference in objective?** Not by the
preregistered rule (G₅₀₀₀ = 0.46 ≫ 0.1). The gap oscillates between 0.44 and
0.85 over the last 1 000 sub-iterations with no trend to zero.

**24. Does it approach the reference in design space?** Slowly and
monotonically in direction (distance 0.995 → 0.830, Spearman −1.00 over
1 000–5 000; sign agreement 94 %), but it remains far away and its magnitude
stays a quarter of the solution's. Preregistered classification: E,
inconclusive. `MMA_TRAJECTORY_TO_REFERENCE.md`.

**25. Is production truncation materially premature?** Yes: G_P19 = 0.997
against the 0.5 bar. The production step is 0.3 % of the certified step's
gain and 0.8 % of its norm.

**26. Is the 5000-state drift toward the actual solution?** Directionally yes
(cosine rising to 0.71, distance decreasing), but at a rate that leaves it
46 % short in objective and 83 % away in design after 5 000 sub-iterations,
with the objective oscillating. It is not converging to the solution on any
practical budget.

**27. Is repeated MMA solving problem (25) faithfully?** No. It never reaches
the vertex the problem's solution is; its KKT residual does not decrease
(0.36 → 0.60); its own duals drift away from the certified ones (volume
multiplier 1.47 vs 0.73). Each `mmasub` call is solved cleanly — the failure is
of the repeated realization on a vertex-solution problem.

**28. Does the hard move box materially determine the solution?** Yes. With
99.94 % of variables on a bound and every gray element at ±move, the frozen
solution is a function of the box geometry together with the cone and volume
rows; to 6e−4 of the gain it is a threshold rule on the filtered first-mode
sensitivity. The move limit was not changed.

**29. Which inner-solver details are paper-specified?** Problem (25a–f) with
the erratum's Δ in (25d); the four-step Fig. 1 loop with an inner "increments
converged?" test; "the MMA method (Svanberg 1987) has been used"; Sigmund's
filter on the sensitivities. `PAPER_VS_RECONSTRUCTION.md`.

**30. Which are reconstruction choices?** The move box on drho (the paper
bounds only the density), the relative-step stop rule and its 0.05/5/500,
the asymptote reset and MMA constants, coordinates and scaling, `bs ∈ [0,5]`,
the diagonal-offset (25d), the fixed subspace N = 2, the gradient formula, the
filter radius and its application to the off-diagonal f_sk, fixed p = 3.

**31. What is the single highest-information next action?** Study the
inner-solver realization on this frozen subproblem against the certified
reference: GCMMA's conservative loop, the sensitivity of repeated MMA to its
unfixed reconstruction choices, and whether an exact SOCP step is an
admissible inner solver — all frozen-state, no density update.
`NEXT_ACTION.md`.

**32. Was production left untouched?** Yes. No file under `+impl/` or the
OlhoffCurrent root was modified; the only new tree is this study directory.

---

## Figures (`figures/`)

1 reference drho field · 2 P19 drho · 3 M5000 drho · 4 reference − P19 ·
5 reference − M5000 · 6 MMA max|drho| vs sub-iteration · 7 objective gap vs
sub-iteration · 8 distance and cosine to reference vs sub-iteration · 9 KKT
residual vs sub-iteration · 10 active-bound map of the reference · 11 scatter
P19 vs reference · 12 scatter M5000 vs reference · 13 (supplementary) reduced
gradient map.

## Artifact guide

Preregistration `AUDIT_PREREGISTRATION.md` (SHA-256 `37cbbb6b…4ca6320`).
Identity and problem: `STATE_IDENTITY.md`, `EXACT_PROBLEM25.md`,
`PAPER_VS_RECONSTRUCTION.md`. Reduction and convexity:
`CLUSTER_CONSTRAINT_REDUCTION.md`, `SOCP_DERIVATION.md`,
`SOCP_EQUIVALENCE.md`, `CONVEXITY_CERTIFICATION.md`. Reference:
`CONIC_REFERENCE.md`, `FMINCON_CROSSCHECK.md`, `REFERENCE_KKT.md`,
`REFERENCE_SOLUTION.md`, `ACTIVE_SET_ANALYSIS.md`. MMA:
`MMA_REFERENCE_COMPARISON.md`, `MMA_TRAJECTORY_TO_REFERENCE.md`. Decision:
`NEXT_ACTION.md`. Integrity: `PROVENANCE.md`, `METRICS.json`,
`DATA_MANIFEST.json`, `EVIDENCE.json`, `FINAL_SHA256.txt`, `scripts/`,
`evaluations/`.

# WHAT WE LEARNED

The frozen inner problem is not a hard nonlinear problem at all: at this
state it is a convex cone program whose global solution is a bang-bang vertex
determined almost entirely by thresholding the filtered first-mode
sensitivity against the volume multiplier. Its certified optimum gives a
0.179 % predicted eigenvalue gain per outer step. Production's inner loop
returns 0.3 % of that gain because it stops on a relative-step heuristic
after 19 sub-iterations, and the repeated-MMA sequence would not reach the
vertex in any practical number of sub-iterations: an interior-point subsolver
iterated as a fixed-point map creeps toward the box from inside, its objective
oscillating around half the achievable gain. The endpoint of the 480 run is
therefore set by a truncation of an iteration that is not converging to the
problem it poses.

# WHAT WE DID NOT LEARN

Whether taking the certified vertex step at every outer iteration would
produce a better design — a full-move bang-bang update each outer step is a
different algorithm, and the outer problem is nonconvex. Whether the paper's
own (25), without a move box, has a solution that resembles this one (the box
is a reconstruction and was not removed). Anything about density filtering,
projection, p continuation, or the mesh dependence of grayness.

# WHAT I WOULD DO NEXT

Run the frozen inner-solver study with the reference as the yardstick: GCMMA
on the same frozen data, the sensitivity of repeated MMA to its unfixed
choices, and the exact SOCP step as a candidate inner solver — measuring
sub-iterations to G ≤ 0.01 and distance to the vertex. Only then decide
whether the outer iteration should take the step the inner problem actually
asks for.
