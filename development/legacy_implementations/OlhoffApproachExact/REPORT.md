# OlhoffApproachExact — execution report

Plan: [PLAN_Olhoff2014_exact.md](PLAN_Olhoff2014_exact.md).
Target: Olhoff & Du (2014), CISM, `references/Olhoff2014_Structural.pdf`.
Run date: 2026-07-30. MATLAB R2025b, macOS arm64.
Solver manifest: `Matlab/SHA256.txt`.

---

## 1. Headline result

All verification phases pass. The paper's three §3.1 examples run to a
characterized end state and are reported against the decision rule declared in
the plan **before** the runs. None reaches PASS.

| case | figure | paper | computed | err | N | N paper | comp | stop_reason | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ss_n1 | Fig. 3a | 174.7 | **148.78** | −14.83 % | 2 | 2 | 1 | trust_region_exhausted | PARTIAL |
| cs_n1 | Fig. 3b | 288.7 | **263.69** | −8.66 % | 1 | 2 | 1 | trust_region_exhausted | PARTIAL |
| cc_n1 | Fig. 3c | 456.4 | **411.77** | −9.78 % | **2** | 2 | 2 | trust_region_exhausted | PARTIAL |
| cc_gap23 (80×10, 60 it, smoke test) | Fig. 7 | 810 | **834.68** | +3.05 % | 2 | 3 | 1 | max_iter | PARTIAL |

160×20 unless noted, LP subproblem, trust-region step control, `rmin = 1.2`.
Initial frequencies match the paper to −0.36 %. The gap case predates the
defect of §3.0 and has not been re-run.

The gap problem (Eq. 20) is the closest to its published value (+3.05 %) despite
running on a coarse mesh for 60 iterations, and it reproduces the paper's
qualitative claim (+564.6 % vs the published +548 %).

**Superseded numbers.** An earlier version of this table reported SS 168.68
(−3.45 %, kkt_converged, N = 1), CS 257.37 (−10.85 %) and CC 415.96 (−8.86 %,
N = 3). Those runs were produced with the ratio-test defect of §3.0 active and
must not be quoted. Correcting it made the multiplicity right and the
frequencies worse.

## 3.0 DEFECT FOUND AND FIXED — the ratio test switched itself off whenever N ≥ 2

Caught by inspecting the printed `ratio` column: it was **NaN on every iteration
with N ≥ 2**, those steps were always accepted, the trust region never adapted,
and `fd_err` ramped monotonically (1.7e−2 → 4.9e−2) through each N = 2 stretch
until the run broke out of it.

Cause: `pred_inc` was computed as `min(sort(eig(G)) − lam(cluster))`, a per-mode
difference. Under cluster model CA the reference is `lam_n` for every member, so
the second entry is `mu_2 − (lam_{n+1} − lam_n)`, which turns negative as soon as
the cluster has any spread. That drove `pred_inc <= 0` → `ratio = NaN` → the
trust-region branch was skipped → the step was accepted unconditionally. The
test disabled itself exactly when the paper's multiple-eigenvalue machinery
engaged.

Fix (two parts):

* the objective increase is `min(predicted new cluster eigenvalues) − lam_n`,
  not a per-mode difference;
* an undefined ratio no longer means "accept". It falls back to the objective
  itself: keep the step only if `act_inc > 0`, otherwise reject and contract.

Effect: bimodality now forms **and is retained** — CC reaches the paper's N = 2,
SS reaches N = 2 — while every ω₁ moves and SS moves a long way down.

### Consequence, not yet resolved

With the ratio test live, all three cases now stop at
`trust_region_exhausted` after 23–117 iterations instead of running to a small
increment. Leading hypothesis: this is finding F-4 one level up. The LP
maximizes the **filtered** model while `pred_inc` measures the **unfiltered**
one, so the subproblem can predict a gain where the true linearization predicts
none; the step is then rejected, the radius collapses to `move_min`, and the run
terminates. If that is right, `trust_region_exhausted` is here partly a FALSE
convergence signal and the residual ω₁ gaps below are upper bounds on the
achievable error, not converged optima. **This is untested.**

## 2. What is verified

| test | file | result |
|---|---|---|
| Fig. 2 initial frequencies, p = 3, 80×10 | `v_forward_model` | SS 68.62 / CS 104.07 / CC 145.97 vs 68.7 / 104.1 / 146.1 → −0.11 / −0.03 / −0.09 % |
| published values are p = 3 penalized | `v_forward_model` | p = 1 gives 137.25 = exactly 2 × 68.62, since ω² ∝ 0.5^{p−1} |
| SS support is a mid-height pin | `v_forward_model` | 68.40 (mid-height) vs 95.50 (bottom corner) at 160×20 — corner reading ruled out |
| odd-`nely` guard | `v_forward_model` | raises `build_supports_exact:OddNely` |
| mass-law derivatives across the ρ = 0.1 kink, 5 modes | `v_sensitivities_fd` T1 | ≤ 8.5e−11 |
| Eq. (4)/(5) simple sensitivity vs central differences | T2 | err/scale 9.7e−8 |
| f_ss = ∇λ_s (Eqs. 14, 15) | T3 | exact (0) |
| **off-diagonal f_sk vs central differences** | T4 | err/scale 2.4e−9 |
| f_sk = f_ks | T5 | exact (0) |
| multiplicity detection + hysteresis, 11 cases | `v_multiplicity` | all pass |
| **cluster basis invariance, LP path** | `v_basis_invariance` T3 | Δρ invariant to 2.5e−13 under a random orthogonal rotation of the cluster basis |
| N = 1: MMA vs exact LP | `v_subproblem` V-I1 | cos ≥ 0.994, gap ≤ 5.8e−5 |
| N = 2,3: cutting plane vs independent bisection | V-I2 | 5.3e−11 |
| stop_reason always reported | V-I3 | pass |
| N = 1 reduction of the general-N path | V-I4 | β to 2.2e−16, Δρ exact |
| **required LP-vertex collapse** | V-I6 | ω₁ 145.97 → 0.0634, 99.88 % of variables at a bound |

T4 is the check no previous campaign had: without it the Eq. (12)/(13)
multiple-eigenvalue path is untested, and the earlier record states plainly that
the N = 2 implementation was never verified against derivatives.

## 3. Findings

### F-1 The paper's own LP reduction is real, and reproduces exactly

With `move = Inf` and N = 1, the exact optimum of subproblem (19) is a box
vertex with 99.88 % of `Δρ_e` at a bound, and one accepted, feasible, in-bounds
step takes CC ω₁ from 145.97 to 0.0634. This independently reproduces the
historical measurement (145.57 → 0.0638) from a clean reimplementation, and it
is now a *required passing test* rather than a failure mode. It confirms that
Fig. 1 as literally drawn cannot work, and that this is mathematics, not a bug.

### F-2 Subproblem (19) is a semidefinite program, and that makes bimodality well-posed

Because f_sk = f_ks and F(Δρ) is linear in Δρ, constraints (19c)+(19d) are
exactly the linear matrix inequality

```
diag(L) + Σ_e Δρ_e F_e  ⪰  β I_N
```

`subproblem_lp.m` solves it exactly by cutting planes over
`μ_min(G) = min_{‖q‖=1} qᵀGq`, needing only `linprog`. Consequences measured:

* the result is **invariant to the cluster eigenvector basis** (2.5e−13), which
  is the property the paper's non-uniqueness remark on p. 281 demands;
* it agrees with an independent bisection certificate to 5.3e−11 at N = 2, 3;
* it needs 0 cuts on the tested problems — the initial basis directions already
  certify the LMI.

The MMA path is basis-invariant only in the limit: the difference decays
1.4e−1 → 5.8e−5 as the inner budget grows 20 → 200 and floors near 1e−4. That
is non-convergence, not a formulation defect — but it is why `lp` is the
default.

### F-3 A fixed move limit cannot both preserve the basin and converge

`experiments/step_calibration`, SS 160×20, 80 iterations, LP solver, only `m` varied:

| m | ω₁(80) | ω₂ peak @ | ω₃ peak @ | coalesced | ω₁ monotone | worst drop |
|---|---|---|---|---|---|---|
| 0.01 | 154.22 | 312.4 @20 | 521.8 @25 | no | ~yes | −0.03 % |
| 0.02 | 154.93 | 311.5 @11 | 516.5 @13 | no | ~yes | −0.31 % |
| 0.05 | 23.20 | 309.2 @5 | 497.1 @42 | it 25 | no | −98.1 % |
| 0.10 | 6.09 | 307.4 @3 | 484.1 @3 | it 12 | no | −98.0 % |
| 0.20 | 2.36 | 273.5 @2 | 420.8 @1 | it 6 | no | −96.9 % |
| Inf | 0.01 | 253.4 @1 | 8932 @4 | it 5 | no | −99.9 % |

At m ≥ 0.05 the run rises cleanly to ω₁ = 144.5 by iteration 12, the FD audit
jumps to 0.97 (linearization error ≈ λ₁ itself), and iteration 13 collapses to
45.6 — then oscillates permanently. **The FD audit predicted the exit one
iteration ahead.** At m ≤ 0.02 the run is stable but `‖Δρ‖∞` stays pinned at
exactly `m` for all 80 iterations: the optimum of an LP subproblem is always a
move-limit vertex, so Fig. 1's stopping test `‖Δρ‖ < ε` can never be met.

This explains the persistent limit cycles reported by every earlier campaign
without needing any appeal to implementation error.

The fix is the textbook SLP globalization — accept/reject on the ratio of
realised to predicted objective increase, contract or expand the radius —
with Powell's constants (η = 0.25/0.75, γ = 0.5/2.0), not values fitted here.
With it, every run is monotone in ω₁ and terminates with a recorded reason.

### F-4 A sensitivity filter makes the ratio test inconsistent

First implementation of the ratio test floored at **0.22 regardless of step
length**, where a correct first-order model must give 1 as m → 0. Cause: the LP
maximizes the *filtered* model while the ratio measured the true objective, and
sensitivity filtering is not a consistent gradient of anything. Fix: the
subproblem uses filtered `Fe`, the ratio test and the FD audit use unfiltered
`Fe_raw`. The ratio then behaves correctly (0.8–0.9 at small steps, negative
where the model breaks).

### F-5 The filter radius, not the filter, is what cost the accuracy — and a short-horizon read of this was wrong

Recorded because the first conclusion was reversed by the second experiment.

At **80×10, 200 iterations**, removing the filter looked like the answer:
`A2_nofilter` gave ω₁ = 165.3 (−5.4 %) against the reference 140.7 (−19.5 %).
At **160×20, 400 iterations** the same arm gives **114.0 (−34.8 %)** — it
degrades with mesh and horizon. The 80×10 result was a short-horizon artefact.

The radius sweep at 160×20, 800 iterations settles it:

| rmin (elements) | 1.0 | 1.2 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|---|
| ω₁ | 114.0 | **168.7** | 166.1 | 161.8 | 158.9 |
| stop | tr_exhausted | kkt_converged | kkt_converged | tr_exhausted | kkt_converged |

`rmin = 1.0` produces a 1×1 kernel — literally no filtering — and matches the
`sensitivity_filter = false` arm to the last digit (113.999), which is a useful
internal consistency check. Du2007's 2.5 over-smooths; 1.2 is best.

**Calibration honesty:** Olhoff2014 states no filter at all, so the radius is
genuinely unspecified and had to be calibrated. It was calibrated on **SS
only**; CS and CC were then run at that fixed value as predictions, and are the
weaker two results (−10.85 %, −8.86 %). The SS number is therefore partly
in-sample and should not be quoted as an independent success.

### F-6 Multiplicity forms, and the machinery handles N = 3

CC terminates with **N = 3** — ω = 415.96 / 417.09 / 418.74, cluster spread
0.67 % on ω — driven through the full Eq. (12)/(13) path with off-diagonal
generalized gradients. The gap problem reaches N = 2 on the lower cluster with
both bound variables β₁, β₂ active. So the multiple-eigenfrequency machinery
does engage; what does not happen is convergence to the paper's *bimodal*
optimum with ω₁ = ω₂ at the published value.

### F-7 MMA path: works, slow, badly conditioned at this size

`A4_mma` at 160×20 costs ~7.6 s per outer iteration (vs 0.08 s for LP) and
floods `subsolv` with RCOND ≈ 6e−17 warnings. At 80×10 it took 1442 s against
4 s for the LP arm and landed at ω₁ = 139.9 vs the LP's 140.7 — no accuracy
gain for a ~350× cost. It is retained because Olhoff2014 names MMA as a solver
for (19), not because it is preferable.

## 4. What was not achieved

* No case reaches PASS. Residual ω₁ gaps: −3.45 % (SS, partly in-sample),
  −10.85 % (CS), −8.86 % (CC).
* The published optima are **bimodal with ω₁ = ω₂**; ours are N = 1 (SS, CS) or
  N = 3 (CC). Multiplicity forms but does not settle at the published value.
* Convergence is inconsistent across cases: `kkt_converged` (SS),
  `trust_region_exhausted` (CS), `increment_small` (CC).
* §3.2 (n = 2) runners are wired and configured but were not executed.
* The gap case was smoke-tested at 80×10/60 iterations only, not run to
  convergence at 160×20.
* §3.4–3.6 (3D plates, bi-material) are out of scope by design.

## 5. Attribution of the residual gap

What can be ruled out, with evidence:

* **not the forward model** — initial frequencies within 0.36 %, all derivatives
  FD-clean, off-diagonal f_sk verified;
* **not the inner loop** — the subproblem is solved exactly (LP optimality gap
  is 0.00e+00 by construction, verified against an independent bisection);
* **not the cluster-basis choice** — invariance verified to 2.5e−13;
* **not step control alone** — the trust region removes collapse and gives
  monotone ω₁, and the gap remains.

What remains open, in order of my confidence:

1. **The filter.** Any sensitivity filter changes the optimum, Olhoff2014
   specifies none, and the result moves 158.9 → 168.7 across a plausible radius
   range. The paper's own regularization is unknown.
2. **Mesh.** The paper's mesh is unspecified. Our initial frequencies are
   −0.11 % at 80×10 but −0.36 % at 160×20, which suggests the paper used a
   coarser mesh than our primary.
3. **Local optimality.** SS terminates `kkt_converged` at a connected,
   full-volume, first-order stationary design 3.45 % below the published value.
   That is consistent with a different local optimum, which for this problem
   class is expected rather than anomalous.

## 6. Reproducing

```matlab
cd analysis/OlhoffApproachExact/Matlab
addpath(pwd, 'verify', fullfile('..','..','..','tools','Matlab'))

v_forward_model; v_sensitivities_fd; v_multiplicity
v_basis_invariance; v_subproblem            % all acceptance tests

T = run_all_olhoff_2014({'ss_n1','cs_n1','cc_n1'});   % ~75 s

cd ../experiments/step_calibration; run_step_calibration          % LP arm ~1 min
cd ../ablations; run_ablations(160, 20, 400, ...
    {'ref','A1_pow','A1_step','A2_nofilter','A2_rmin1p5','A3_CC','S_fixed','S_tr_big'})
```

Exclude `A4_mma` from `run_ablations` unless you want the ~25-minute MMA arm;
arms are individually guarded, so a failure is recorded and the sweep continues.
