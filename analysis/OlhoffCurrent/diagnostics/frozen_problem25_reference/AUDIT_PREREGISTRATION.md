# AUDIT_PREREGISTRATION — frozen_problem25_reference

Frozen before any solve. No tolerance, bar, start point, comparison metric or
verdict rule below may be changed after a reference result is visible. Any
deviation discovered later is disclosed in `PROVENANCE.md`, never silently
corrected.

## 0. Scope and locks

This is a ZERO-DENSITY-UPDATE INDEPENDENT REFERENCE SOLUTION of the frozen
480×60 Du–Olhoff sub-optimization problem (25) at the authoritative three-rung
endpoint. It is not a topology-optimization run.

| lock | rule |
|---|---|
| topology-optimization runs | **0** — `olhoffSolve` / `olhoffcurrent_run` are never called |
| outer iterations | **0** |
| accepted ρ updates | **0** — `rho + drho` is never formed as a design; every `drho` is discarded |
| controller transitions | **0** |
| production files modified | **0** — nothing under `+impl/` or the OlhoffCurrent root is written |
| move limit / filter / p / MMA constants altered | **none** |

Authorized computations: frozen-state FE evaluation (to rebuild and verify the
retained context), `coneprog` solves, `fmincon` solves, KKT arithmetic,
repeated-MMA replay of the frozen subproblem with `drho` discarded, analytic
derivation, deterministic test points.

## 1. Authoritative frozen state (expected, from repository evidence)

| item | expected |
|---|---|
| study of origin | `diagnostics/three_rung_canary_preflight` |
| trajectory | `evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` |
| ρ₃₈₆ SHA-256 (authoritative endpoint) | `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60` |
| ρ₃₈₅ SHA-256 (density the final subproblem was built at) | `9b1443e508a6e4ecb5288d30167c7258d837be127aa3a9a42d8cc092240f3ded` |
| Δρ₃₈₆ SHA-256 (production P19 increment) | `0b12cd7f9ae32decc6e95bf63e89fa7ca4530d13ec6dc7d815b618d46af8a9fd` |
| config hash | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` |
| `+impl` tree | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| outer / stage / move | 386 / 3 / 0.01 |
| N (fixed subspace) / J | 2 / 3 |
| `multJ` at 386 / `nInner` at 386 | 0 / 19 |
| `dOff` | **expected PRESENT** = `[0, λ₂−λ₁]` ≈ `[0, 7429.08]` (from `olhoffSolve.m` line 245, `diagonalOffsets = true`) |
| `offDiag` | true |
| ρmin, volfrac | 1e−3, 0.5 |
| MMA | published Svanberg 2007, `a0=1, a=0, c=1000, d=0`, `tolInner=0.05, minInner=5, maxInner=500` |

The subproblem is the one posed at ρ₃₈₅ (the FE data are evaluated before the
update), exactly as `filtered_subproblem_integrability_audit` established.

**Note on the task prompt.** The prompt expects `dOff` to be absent. Repository
evidence (`olhoffSolve.m:245`, `FINAL_SUBPROBLEM_RECOVERY.md`) says it is
present. The reference problem is the one the code poses. The cone derivation
below is therefore written for the general offset form and reduces to the
prompt's form when `dOff = 0`.

`FROZEN_480_PROBLEM25_STATE_PASS` requires: all three hashes above match; the
source manifest verifies; outer = 386, stage = 3, move = 0.01, N = 2,
`multJ` = 0; and a fresh FE evaluation at ρ₃₈₅ through the production functions
(`assemble2D`, `eigSolve`, `genGrad`, `applyFilter`, `olh.multi.detect`)
reproduces the retained `ctx` (`F`, `fJJ`, `lam`, `lamJ`, `dOff`) with max
relative difference ≤ 1e−12 (expected bitwise, same host and MATLAB build).
Any mismatch ⇒ FAIL ⇒ STOP.

## 2. Expected exact formulation (to be recovered from code, not assumed)

Variables `x = [drho (NE=28 800); bs]`, `nvar = 28 801`, `bs = β/lamref`,
`lamref = λ₁(ρ₃₈₅)`. Objective `min −bs`.

Bounds: `xmin = [max(ρmin−ρ₃₈₅, −move); 0]`, `xmax = [min(1−ρ₃₈₅, +move); 5]`.

Constraints (m = N+2 = 4), production-scaled:

| row | expression |
|---|---|
| 1..N | `bs − (λ_j + Δλ_j(drho))/lamref ≤ 0`, `Δλ = deltaLambda(F, drho, dOff)` |
| N+1 | `bs − (λ_J + f_JJᵀ drho)/lamref ≤ 0` |
| N+2 | `(Σ(ρ₃₈₅+drho) − Vtot)/Vtot ≤ 0`, `Vtot = 0.5·NE` |

`deltaLambda` with offsets returns `Δλ_j = e_j(M(drho)) − dOff_j` where
`M(drho) = diag(dOff) + A(drho)`, `A_sk = F(:,s,k)ᵀdrho`, eigenvalues `e_1 ≤ e_2`.

The MMA auxiliary variables `y, z` are internal to `mmasub`; with `a = 0`,
`a0 = 1` and every multiplier ≪ `c = 1000` they vanish and the MMA problem is
(25) as written (established by the prior audit; re-verified here only through
the replay's `y, z` values).

## 3. Equivalence tests (Parts 3–5)

### 3.1 Cluster-constraint redundancy
Analytic claim to prove: since `dOff_j = λ_j − λ_1` exactly (by construction in
`olhoffSolve`), `λ_j + Δλ_j = λ_1 + e_j(M)`. Ordering `e_1 ≤ e_2` then makes
row 1 imply row j for every drho. Numerical check at every test point of §3.3:
`(λ_1 + Δλ_1) ≤ (λ_j + Δλ_j) + 1e−9·lamref`, and
`|(λ_j − dOff_j) − λ_1| ≤ 4·eps·λ_1`. PASS requires both at every point.

### 3.2 SOC representation (N = 2)
With `a = F11ᵀdrho`, `b = F12ᵀdrho`, `c = F22ᵀdrho`, `d₂ = dOff(2)`:
`e_1 = (a+c+d₂)/2 − sqrt(((a−c−d₂)/2)² + b²)`. Row 1 ⇔
`‖[(a−c−d₂)/2 ; b]‖ ≤ (a+c+d₂)/2 + λ_1 − lamref·bs`.
In production-scaled units (everything divided by `lamref`, an exact
equivalence), the `coneprog` cone `‖A_c x − b_c‖ ≤ d_cᵀx − γ_c` has
`A_c = [0.5(F11−F22)ᵀ/lamref, 0; F12ᵀ/lamref, 0]`, `b_c = [d₂/(2·lamref); 0]`,
`d_c = [0.5(F11+F22)/lamref; −1]`, `γ_c = −(d₂/(2·lamref) + 1)`.
The MATLAB sign convention of `secondordercone` is verified with a two-variable
toy problem whose optimum differs under the two candidate conventions (§3.4).

### 3.3 Test points (deterministic, fixed before solving)
`T0` drho = 0; `T1` production P19 (`DRHO(:,386)`); `T2` M500 (`stC.xFinal`);
`T3` M5000 (from the replay of Part 12, appended when available);
`T4–T8` five `rng(20260912,'twister')` uniform points in the box (in that draw
order); `T9` `+move` everywhere clipped to box; `T10` `−move` everywhere clipped;
`T11` `move·sign(F11)` clipped; `T12–T14` three volume-neutral directions: seeded
uniform `u ∈ [−1,1]^NE`, mean removed, scaled to 0.9·min box ratio, clipped;
`T15` the reference solution (appended post hoc).
For each point, `bs` is set to `(λ_1+e_1)/lamref·(1+δ)` for
`δ ∈ {−1e−3, 0, +1e−3}` so both feasible and infeasible classifications occur.

### 3.4 Agreement bars
Row-1 value from `deltaLambda` vs cone residual `‖A_c x − b_c‖ − d_cᵀx + γ_c`:
absolute difference ≤ 1e−10 at every point and δ. Feasible/infeasible
classification (sign) identical. Gradient of row 1 from `ddlam` vs closed-form
cone gradient: relative ∞-norm difference ≤ 1e−9 (away from the apex). Rows
N+1, N+2 and objective: exact algebraic identity checked to 1e−12. Toy sign
test: the optimum must equal the value predicted by the documented convention
to 1e−6. All must hold ⇒ `FROZEN_PROBLEM25_SOCP_EQUIVALENCE_PASS`; any failure
⇒ FAIL and the SOCP is not used as reference.

### 3.5 Smoothness margin
Report `min over the box` of `((a−c−d₂)/2)² + b²` via the bound
`((d₂ − max_box|a−c|)/2)²` with `max_box|a−c| = Σ_e max(|lo_e|,|hi_e|)·|F11_e−F22_e|`.
If that bound is positive the cone never reaches its apex inside the box and
row 1 is C² on the whole feasible box; drho = 0 is then NOT a nonsmooth point
(the offsets separate the eigenvalues). Otherwise eigenvalue separation is
tracked during every solve and any approach within 1e−6·d₂ is reported.

## 4. Solver hierarchy and tolerances

1. **Primary:** `coneprog` (interior-point SOCP), `OptimalityTolerance = 1e−10`,
   `ConstraintTolerance = 1e−10`, `MaxIterations = 500`, `LinearSolver = 'auto'`.
   Strong termination = `exitflag == 1`. Retain `x, fval, exitflag, output,
   lambda` (soc, ineqlin, lower, upper).
2. **Cross-check A:** `fmincon` interior-point with production
   `deltaLambda` constraints (all 4 rows, exact `ddlam` gradients), exact
   Lagrangian Hessian supplied through `HessianMultiplyFcn`
   (`SubproblemAlgorithm = 'cg'`), `OptimalityTolerance = 1e−10`,
   `ConstraintTolerance = 1e−10`, `StepTolerance = 1e−14`, `MaxIterations = 3000`.
3. **Cross-check B:** same as A with `HessianApproximation = 'lbfgs'`.
4. **Cross-check C:** `fmincon` `sqp` from S1 only, wall cap 30 min /
   `MaxIterations = 50`. n = 28 801 makes its dense quasi-Newton Hessian
   (6.6 GB) and dense QP subproblem likely non-viable; if it does not return
   within the cap it is recorded as **not numerically viable**, not as a
   disagreement.

Gradient check before any fmincon solve: central differences along 5 seeded
directions at T1, step `1e−6·move`; relative error ≤ 1e−5 per row required.

Deterministic starts: `S0 = [0; 1]`; `S1 = P19`; `S2 = M500`; `S3 = M5000`;
`S4 = 0.5·xmax(1:NE)`; `S5 = 0.5·xmin(1:NE)`; `S6 = rng(20260912)` uniform in
box; for S4–S6, `bs = (λ_1+e_1)/lamref`. If the smoothness margin of §3.5 is
positive, S0 is used for gradient solves as well; otherwise S0 is value-only.

## 5. KKT certification (Part 9), preregistered scales

Normalizer `sRow0 = RMS(F11/lamref)` (the row-1 gradient at drho = 0), fixed
for every point and every solver. The prior audits' per-point normalizer
`RMS(ddlam(:,1)/lamref)` is reported alongside for comparability.

Residual with the solver's own multipliers (`μ` cone or `λ_1..4`, `ν` linear,
`ξ` lower, `η` upper):
`r = ∇f0 + Σ_i λ_i ∇c_i − ξ + η`. Report `RMS(r(1:NE))/sRow0`,
`max|r(1:NE)|/sRow0`, `|r(nvar)|`.

| condition | PASS | FAIL |
|---|---|---|
| primal: max row value (scaled units) | ≤ 1e−8 | ≥ 1e−5 |
| primal: box violation (absolute) | ≤ 1e−10 | ≥ 1e−7 |
| dual: min multiplier | ≥ −1e−10 | ≤ −1e−7 |
| complementarity: max|λ_i c_i| and max(ξ·gap, η·gap)/(sRow0·move) | ≤ 1e−6 | ≥ 1e−3 |
| stationarity: normalized RMS / normalized max | ≤ 1e−6 / ≤ 1e−5 | ≥ 1e−3 |
| cone dual (SOCP): `‖w‖ ≤ w₀ + 1e−10`, gap `w₀‖A_cx−b_c‖ − wᵀ(A_cx−b_c)` ≤ 1e−8 | | |

All PASS ⇒ `REFERENCE_PROBLEM25_KKT_PASS`; any FAIL ⇒ `_FAIL`; otherwise
`_INCONCLUSIVE`. Bound activity is **never** classified by a 1e−12 tolerance;
actual multipliers are used. For reporting the active set, a variable is "at a
bound" if within `1e−6·(box width)` of it, and the count is swept over
`{1e−8, 1e−7, 1e−6, 1e−5, 1e−4, 1e−3}`.

Global certificate: the SOCP weak-duality bound computed from the returned
duals (derivation in `CONIC_REFERENCE.md`); duality gap ≤ 1e−8 (absolute, in
`bs` units) together with `FROZEN_PROBLEM25_CONVEXITY_CERTIFIED` and KKT PASS ⇒
`GLOBAL_PROBLEM25_REFERENCE_CERTIFIED`. Otherwise, if at least one fmincon
solve reaches KKT PASS, `LOCAL_PROBLEM25_REFERENCE_ONLY`; else
`PROBLEM25_REFERENCE_INCONCLUSIVE`.

## 6. Solver agreement and primal non-uniqueness

Two solutions **agree in objective** if `|bs_A − bs_B| ≤ 1e−6`. They
**materially disagree** if `|bs_A − bs_B| ≥ 1e−4`.

Structural note recorded before solving: the constraints depend on drho only
through five linear functionals `(a, b, c, f_JJᵀdrho, Σdrho)`. The optimal `bs`
is unique (convex problem, if certified) but the optimal `drho` set may be a
face of the box of positive dimension. Solver-to-solver primal differences are
therefore compared (i) in `bs`, (ii) in the five functionals (relative
difference ≤ 1e−6 = same image point), and (iii) in `drho` (reported, not a
pass/fail criterion). A primal difference with identical objective and image
is classified **objective agreement with primal non-uniqueness**, not a solver
disagreement.

## 7. Comparison metrics against repeated MMA (Parts 11–12)

For `k ∈ {P19, M500, M5000}` and at replay checkpoints:
`bs_k`, `β_k = bs_k·lamref`, objective gap `bs_ref − bs_k`, normalized gap
`G_k = (bs_ref − bs_k)/(bs_ref − 1)` (fraction of the achievable predicted
gain lost; `bs − 1` is the predicted relative eigenvalue increase),
`‖drho_k − drho_ref‖_2`, `‖·‖_∞`, `‖·‖_2/‖drho_ref‖_2`, cosine similarity,
`max|drho_k|/move`, constraint violation `max(c_i(x_k), 0)`, exact-MMA-dual
KKT residual (normalized by `sRow0`), active-set overlap (Jaccard index of the
sets "at +move", "at −move", "at floor", "at ceiling" at tolerance 1e−6·width;
for MMA iterates, which come from an interior-point subsolver, the sweep of
§5 is also reported), and the five image functionals.

Replay checkpoints: every iteration 1–50; then every 10 to 100; every 50 to
1 000; every 100 to 5 000. At each: x, MMA duals `lam, xsi, eta`, `y, z`,
relStep. The replay must reproduce `DRHO(:,386)` bitwise at iteration 19 and
`stC.xFinal` bitwise at 500; otherwise it is reported as non-reproducing and
its M5000 is labelled as such.

Trajectory classification (design space uses `dist_k = ‖drho_k−drho_ref‖_2/‖drho_ref‖_2`):
* **approaching in objective**: `G_5000 ≤ 0.1` and `G_5000 < 0.5·G_19`;
* **approaching in design**: `dist_5000 < 0.5·dist_19` and `dist_5000 ≤ 0.1`;
* **moving away**: `dist_5000 > 1.1·dist_19`;
* **orbiting**: `dist` over iterations 1 000–5 000 has range ≥ 0.2×its mean
  with no monotone trend (Spearman |ρ| < 0.5 against iteration);
* **approaching objective but not primal (D)**: objective criterion met,
  design criterion not met;
* otherwise **inconclusive**.

## 8. Decision logic (Part 15), evaluated in this order

1. No certified (global or local KKT-PASS) reference ⇒
   `PROBLEM25_REFERENCE_INCONCLUSIVE` (Case E).
2. Convexity **not** certified and ≥ 2 KKT-PASS solutions with `|Δbs| ≥ 1e−4`
   ⇒ `PROBLEM25_MULTIPLE_LOCAL_SOLUTIONS_MATERIAL` (Case D).
3. `G_P19 ≤ 0.1` ⇒ `PRODUCTION_TRUNCATION_NEAR_REFERENCE` (Case B).
4. `G_P19 ≥ 0.5`, `G_M5000 ≤ 0.1`, cosine(M5000, ref) ≥ 0.9 and reference
   has ≥ 50 % of variables at a move bound ⇒
   `PRODUCTION_INNER_TRUNCATION_MATERIALLY_PREMATURE` (Case C).
5. Otherwise ⇒ `REPEATED_MMA_REALIZATION_FAILS_PROBLEM25_REFERENCE` (Case A).

"Production truncation materially premature" (report Q25) = `G_P19 ≥ 0.5`.
"Move-bound dominated" (Q17) = ≥ 50 % of variables within 1e−6·width of a
move-limited bound.

Final: `FROZEN_INNER_SOLVER_STUDY_JUSTIFIED` if Case A, C or D;
`_NOT_JUSTIFIED` if Case B; `_PREMATURE` if Case E.

## 9. Stop conditions

STOP (report, no repair) if: state identity fails; the retained ctx cannot be
reproduced; the SOCP fails the equivalence bars; a sign/scaling error is
exposed by the toy or gradient checks and cannot be traced to a documented
derivation step; or the reference solvers materially disagree in objective
with no convexity certificate.

## 10. Reporting

Every solve of the frozen subproblem is labelled FROZEN-SUBPROBLEM REFERENCE
or FROZEN-SUBPROBLEM REPLAY and its `drho` is discarded after metrics are
taken. `rho_new = rho + drho` is never formed. The 5000-iteration MMA state is
never called "the solution of (25)".
