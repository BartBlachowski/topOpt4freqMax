# Faithful Reconstruction of the Du & Olhoff (2007) Optimization Procedure

## Revision V2

Clamped–clamped beam benchmark, production mesh **160 × 20** (diagnostic) and
**240 × 30** (primary), square elements, exact mid-height supports, even `nely`.
The 40 × 5 discretization is not used for any conclusion in this report.

Date: 2026-07-29 · MATLAB R2025b · repository commit `461022f`.
Revision V2: 2026-07-29, following an independent scientific audit. **No
experiment was rerun and no numerical result changed between V1 and V2.** V2
revises scope, causal language, terminology and internal consistency only. Every
change is itemised in Appendix B.

**Zero production files were modified.** Evidence: `results/production_sha256.txt`
(SHA-256 of all 25 files in `analysis/OlhoffApproachExact/Matlab/` plus
`tools/Matlab/mmasub.m` and `tools/Matlab/subsolv.m`) and a clean
`git status --porcelain` over `analysis/` and `tools/`. All campaign code is
additive, in this directory. The campaign's outer loop (`recon_solve.m`) is
proven **bit-identical** to the production `topopt_freq_exact.m` on its default
path, in both step-control regimes, by `tests/run_all_tests.m` (T2).

### 0.0 Scope, terminology and evidence grading

This section is normative for the whole report. It was added in V2 because the
audit correctly found that V1 used several key terms without operational
definitions.

**Object of study.** Three distinct objects must not be conflated.

| Term | Definition | What this campaign can say about it |
|---|---|---|
| **The paper-explicit formulation** | Eq. (25a–f) as printed, the Fig. 1 nested loop, and the update ρ := ρ + Δρ | Directly tested. Conclusions about it are as strong as the forward-model reconstruction it is evaluated on (§2.7) |
| **The tested reconstruction** | The paper-explicit formulation *plus* the 18 unspecified choices of §12 (mesh, ρ_min, inner tolerance and budget, MMA constants, β bound, `mult_tol`, `n_modes`, continuation schedule, …) | Directly tested. This is the subject of every trajectory-level conclusion below |
| **The historical implementation** | Whatever Du & Olhoff actually executed in 2007, including undocumented procedural details | **Never directly tested.** No conclusion in this report is a verdict on it. Statements about it are labelled [HYP] and are inferences from the paper's own published figures plus contemporaneous practice |

Wherever V1 wrote "the paper-literal procedure" as the subject of a verdict, V2
writes "the tested paper-literal reconstruction" unless the claim is shown in
§2.7 to be independent of every low-confidence reconstruction choice.

**"Validity" — four distinct senses, never used interchangeably below.**

| Sense | Operational definition | Status in this campaign |
|---|---|---|
| **Inner-solve validity** | The increment returned by the inner loop satisfied *this campaign's declared* stopping test `‖Δρₖ − Δρₖ₋₁‖ < inner_tol·√nEl`, rather than exhausting the iteration cap | Measured, per iteration, in `inner_history.csv` |
| **Acceptance-gate validity** | The five fail-closed predicates of §4.1 all hold | Measured, per iteration |
| **Spectral validity** | Gate G4 on the *final design*: the reported ω₁ is the recomputed lowest eigenvalue, the state is not a mechanism, and mode 1 carries < 10 % of its strain energy in ρ ≤ 0.1 elements | Measured on final designs only |
| **Mathematical optimality of the subproblem solution** | A KKT / duality certificate for Eq. (25) | **Not measured anywhere in this campaign.** No KKT residual, dual residual or optimality-gap certificate was computed. The single exception is the initial N = 1 state, where the subproblem is an LP and an independent exact solution is available (§2.4) |

V1's phrase "necessary for validity" is replaced throughout by the specific
sense intended, which is always *inner-solve validity*. The report does **not**
claim that its successive-iterate tolerance is a necessary or sufficient
certificate of subproblem optimality, and it does **not** claim that an inexact
inner step is mathematically illegitimate in an outer sequential method. What it
claims is narrower and is fully measured: at the recorded budget the production
configuration never met its own declared inner stopping test, so its increments
cannot be described as solutions of Eq. (25).

**Evidence grading.** Every conclusion in §§9–11 carries one of:

| Grade | Meaning |
|---|---|
| **[A] Demonstrated** | Directly measured in this campaign; alternative explanations excluded by a controlled comparison within the campaign |
| **[B] Supported hypothesis** | Consistent with all campaign measurements and with at least one independent line of argument, but alternatives are not excluded by experiment |
| **[C] Speculative** | Motivated by the campaign but untested within it. Never used as the basis of a verdict |

**Markers.** **[OBS]** marks a measurement, **[EV]** direct evidence quoted from
the paper or the code, **[INF]** an inferred mechanism, **[DEC]** an
implementation decision made for this campaign, **[HYP]** a statement about the
historical implementation, and **[CONC]** a conclusion. `[CONC-preview]` marks a
forward reference to a conclusion formally established, with its evidence grade,
in §§9–11.

---

## 0. Executive summary

Three findings dominate everything else. Each is stated at the strength its
evidence supports and no stronger; the object of each claim is named explicitly.

1. **[A] At the initial uniform design, the paper-explicit incremental
   subproblem is a linear program whose optimum is a box vertex, and that vertex
   destroys the structure.** This is not an artefact of a truncated inner solve,
   of MMA conditioning, of the absence of continuation, or of mesh resolution.
   The subproblem was solved *twice and independently*: exactly, as an LP, and
   by MMA run to full convergence (312 inner iterations). The two agree to
   0.02 % in objective and 4° in direction, and both collapse the structure —
   the exact LP step takes ω₁ from 145.57 to **0.0638 rad/s**, the converged MMA
   step to **0.0953 rad/s**, in a single accepted, feasible, in-bounds outer
   step. The linear model predicted 265.6. Because the exact LP solution bypasses
   MMA entirely, this result is independent of nine of the eighteen unspecified
   reconstruction choices, including every MMA-related one (§2.7). It is
   therefore a property of the **paper-explicit formulation** evaluated on this
   forward model, not merely of the tested reconstruction.

2. **[A] At the recorded inner budget the production configuration never
   satisfies its own declared inner stopping test; making that test binding
   halts the solver, and removing the truncation does not avert the collapse.**
   At `inner_max_iter = 30`, *no inner solve at outer iteration 1 converged
   anywhere in the campaign*, and over full trajectories the count is 0/300
   (160 × 20) and 0/120, 0/150 (240 × 30). With the fail-closed gate enabled at
   that budget every variant halts at outer iteration 1. Raising the budget until
   the inner problem meets the test makes the gate pass — and the very same step
   then collapses the structure anyway (V2: 12/13 inner solves converged,
   ω₁ → 0.02; V3: 15/15, ω₁ → 0.04). This is a statement about *inner-solve
   validity* as defined in §0.0, not about mathematical optimality.

   [B] A second reading of the same data is that truncation is itself acting as
   an undeclared step restriction: in the paper-literal box, a budget of 10
   returns ‖Δρ‖∞ = 0.165 and *raises* ω₁ from 145.57 to 184.07, while a budget of
   20 returns ‖Δρ‖∞ = 0.500 and collapses it (§2.3). The recorded budget of 30
   is on the collapsing side of that transition.

3. **[A] No run converges.** Of 19 runs, 12 halt on a failed acceptance gate (5)
   or are classified `MECHANISM_COLLAPSE` (7), and 2 more report a void-localized
   mode. The 5 that are feasible and spectrally valid all fail design convergence
   by four to five orders of magnitude. Four of them terminate in a **persistent,
   move-limit-saturated period-2-like oscillation** — V4 at ω₁ = 300.90
   (160 × 20) and 343.04 (240 × 30), VR at 328.55 (160 × 20) and 371.54
   (240 × 30) — with lag-2/lag-1 design-change ratios of 0.19, 0.19, 0.14 and
   0.08 against a declared threshold of 0.25, and with ‖δ‖∞ equal to the move
   limit at every tail iteration. The fifth (V5, 160 × 20) does not settle within
   its budget. The oscillation is **not** an exact period-2 cycle: the lag-2
   design change is small but non-zero (median ‖ρₖ − ρₖ₋₂‖∞ = 0.136 for V4 at
   160 × 20), so the correct description is a strong period-2 component plus a
   non-repeating residual (§7.3). Throughout, the linearized subproblem realises a
   median **0.2 %** of the improvement it predicts and gets the sign wrong in 144
   of 300 iterations (V4, 160 × 20). Across all 19 runs a bimodal state is
   reported at **one** outer iteration of one non-mechanism run — at ω₁ = 90.86,
   20 % of the published optimum, on a step that cut ω₁ by 74 % — and is
   abandoned immediately.

**What is and is not established about the missing ingredient.** The evidence
excludes continuation and multiplicity detection as the missing ingredient, and
excludes inner-solver truncation as an *explanation of the collapse*:

* [A] Continuation is neither necessary nor sufficient, under all three tested
  schedules, at every p on the 1 → 3 path, at both meshes.
* [A] Multiplicity detection is not the obstacle to retaining the one tested
  near-cluster: forcing N = 2, which bypasses detection entirely, also fails to
  retain it.
* [B] **A finite restriction on the step length is required for the tested
  reconstruction to produce any iteration history at all.** Three independent
  lines support this: the step profile along the *fixed* paper-literal direction,
  which isolates length from every other control (§2.5); the inner-budget
  transition of finding 2 above; and the survival of the Regime-B runs. It
  remains a hypothesis rather than a demonstrated necessity because the
  Regime-B comparison changes `move_lim`, `outer_move` and `alpha` together, so
  only the first of the three lines isolates step length alone, and only at
  outer iteration 1.
* [C] **That such a restriction must additionally *contract* as the iteration
  proceeds is untested.** It is motivated by the observation that a fixed
  restriction keeps the iteration alive but does not let it converge or hold a
  coalesced state. No contracting variant was implemented, no basin width was
  measured, and no contraction factor is claimed. V1's assertion of a required
  contraction factor "of order 10" is withdrawn as unsupported.
* [HYP] Whether the historical 2007 implementation contained a step restriction
  is **not determined by this campaign**. The paper's own Fig. 4, which rises
  smoothly over ~100 iterations, is inconsistent with an unrestricted full-box
  step, but this is indirect evidence about the published computation, not
  implementation evidence.

Gate results are reported individually in §8 and are **not** summed into a
score in V2; V1's "best gate score 6/8" framing is withdrawn, because the eight
gates are heterogeneous, some passes are vacuous, and G7 is `n/a` for runs
without a mesh sibling. The two gates that define the research question, G5
(multiplicity retained) and G6 (trajectory converged), **fail for all 19 runs**.
The highest frequency observed, 371.54 rad/s (81.4 % of the published 456.4),
comes from the reference configuration in which no inner solve ever met the
declared inner stopping test, and is in any case a sample from a non-converged
oscillation rather than an optimum.

---

## 1. Phase 1 — The exact current formulations

### 1.1 Execution paths

`topopt_freq_exact.m` has grown a large amount of default-disabled diagnostic
machinery. On the **default path** used by both regimes, all of the following
are inert: `globalization_enabled`, `forensic_enabled`, `acceptance_check`,
`persistent_mma_state`, `post_coalescence_trust_enabled`, `density_symmetry`,
`forced_solid_mask`, and every `filter_type` other than `sensitivity`. With
`filter_type = 'sensitivity'` the physical density equals the design density
(`topopt_freq_exact.m:710-711`), so no density filter is applied anywhere.

| # | Element | Where | What it does |
|---|---|---|---|
| 1 | **Regime A** (paper-literal) | `run_clamped_clamped_exact.m:11-24` | `move_lim = Inf`, `outer_move = Inf` (default), `alpha = 1`, `acceptance_check = false`, `outer_tol = 1e-4`, `outer_max_iter = 300`, `inner_max_iter = 30` |
| 2 | **Regime B** (stabilized) | `audit_optimizer_nochange.m:12-31` | `move_lim = 0.2`, `outer_move = 0.2`, `alpha = 0.5`, `outer_tol = 1e-6`, `outer_max_iter = 80`, `inner_max_iter = 30`, `mult_tol = 1e-3`, `n_modes = 4` |
| 3 | **Inner MMA subproblem** | `inner_loop_mma.m` | variables `x = [β̂; Δρ]`; `m = N + 1 + has_J` constraints; up to `inner_max_iter` calls to `mmasub`; converges on `‖Δρₖ − Δρₖ₋₁‖ < inner_tol·√nEl` |
| 4 | **Multiplicity detection** | `detect_multiplicity.m` | upward scan from mode *n*; \|ω_j − ω_n\| / max(ω_n, eps) ≤ `mult_tol`; `J = n + N` |
| 5 | **Generalized gradients** | `compute_generalized_gradients.m` | `f_sk[e] = φ_sᵀ(K′_e − λ̄ M′_e)φ_k`, Eq. (19), then Sigmund sensitivity-filtered per (s,k) pair |
| 6 | **Outer update** | `topopt_freq_exact.m:377-378` | `ρ ← clamp(ρ + α·Δρ, ρ_min, 1)` |
| 7 | **Stopping criterion** | `topopt_freq_exact.m:549` | `‖ρ_new − ρ‖₂/√nEl < outer_tol` **and** a full step was taken |
| 8 | **Continuation** | — | **none exists**; `penal` is a fixed scalar (`set_defaults`, default 3.0) |
| 9 | **Acceptance / rejection** | `topopt_freq_exact.m:413-418` | the update is applied **unconditionally**; `hist.inner_converged` is recorded but never read |

Item 9 is the single most consequential fact in this section: on the default
path there is no acceptance test of any kind, and the inner convergence flag —
which the code already computes — is discarded.

### 1.2 Explicit algorithmic trace

`phase1_trace.m` emits a narrated, per-step trace. Full output:
`results/phase1_trace_CC_160x20_regimeA.txt` and `…_regimeB.txt`. Abridged, for
regime A, outer iteration 1:

```
[MESH]  nEl=3200  nDof=6762  fixed DOF=84  free DOF=6678  dx=0.05 dy=0.05
[INIT]  rho_e = volfrac = 0.5 for all 3200 elements

STEP 1  omega  = [145.5692  363.0493  622.5581  641.2279]
        M-orthonormality max|Phi'M Phi - I| = 2.887e-15
STEP 1b relative gap |w2-w1|/w1 = 1.493998e+00   mult_tol = 0.001
        => N = 1   cluster = [1]   J = n+N = 2
STEP 2  f_sk raw      min=-3.1749e+01 max=5.2259e+02 ||.||=3.4265e+03
        f_sk filtered min=-3.0345e+01 max=3.7280e+02 ||.||=3.2493e+03
        lambda_J/lambda_bar = 6.2200
STEP 3  n_var = 3201, m = 3 (1 cluster + 1 J-mode + 1 volume)
        bounds (25f): Delta_rho in [-0.4990, 0.5000]
        beta_hat in [0, 1e6]                      <-- reconstruction
        MMA constants a0=1, a=0, c=1e3, d=1        <-- reconstruction
        inner loop: 30 iterations, converged = 0, reason = max_iterations
        inner test ||dDrho|| < inner_tol*sqrt(nEl) = 5.6569e-03
        last change = 4.0043e-01  ->  NOT MET
        Delta_rho: min=-0.498950 max=+0.499994 ||.||inf=0.499994
        fraction within 1% of a box bound = 0.8150
        beta = 7.015829e+04 -> sqrt(beta) = 264.8741 rad/s
VOLUME  predicted mean(rho+Drho) = 0.49973525 (residual -2.647e-04)
        NO explicit volume projection or correction anywhere
STEP 4  rho := rho + 1.0*Delta_rho ; ||rho_new-rho||inf = 0.499994
ACCEPT  production accepts unconditionally
        fail-closed predicate ==> WOULD REJECT
SPECTRUM omega_new = [0.1143  0.1812  0.2630  0.3676]
        realised dlambda = -2.119038e+04  predicted = +4.896790e+04  ratio = -0.4327
```

[OBS] At outer iteration 1 the paper-literal step drives 81.5 % of design
variables to within 1 % of a box bound, the inner problem is **not** converged,
the step is nevertheless accepted, and ω₁ falls by a factor of 1273.

### 1.3 Provenance of every procedural detail

| Procedural detail | Paper explicit | Inferred | Project reconstruction | Confidence |
|---|:-:|:-:|---|:-:|
| Bound formulation, max β s.t. β ≤ ω_j² | ✔ Eq. (15), (25a,c) | | as written | high |
| J-mode bound constraint, J = n + N | ✔ Eq. (25b), §3.5.3 | | as written | high |
| Subeigenproblem det(f_skᵀΔρ − δ_sk Δω²) = 0 | ✔ Eq. (25d) | | as written | high |
| Volume constraint Σ(ρ+Δρ)V_e ≤ V* | ✔ Eq. (25e) | | as written | high |
| Box bounds ρ_min ≤ ρ+Δρ ≤ 1 | ✔ Eq. (25f) | | ρ_min = 1e-3 chosen | high |
| Generalized gradients f_sk, Eq. (19) | ✔ | | as written | high |
| Nested outer/inner loop, Fig. 1 | ✔ §3.5 | | as written | high |
| Update ρ := ρ + Δρ | ✔ Fig. 1 box 4 | | α damping added in regime B | high |
| Stop on ‖Δρ‖ < ε | ✔ Fig. 1 | | RMS norm; ε = 1e-4 / 1e-6 | med |
| MMA used to solve Eq. (25) | ✔ §3.5.3 | | Svanberg 1987 `mmasub.m` | high |
| Subproblem "reduces to a linear program" | ✔ §3.5.3 | | verified numerically, §3.2 | high |
| Sensitivity filter (Sigmund) on objective sensitivities | ✔ §2 | | applied to every f_sk pair | high |
| Mass interpolation Eq. (4b), c₁=6e5, c₂=−5e6 | ✔ | | as written | high |
| **p increasing from 1 to 3 during optimization** | ✔ §2.1 | | schedule invented, §4 | **high (that it happens) / low (how)** |
| Mesh / element count | ✘ | ✘ | 160×20, 240×30 chosen | n/a |
| ε in the stopping test | ✘ | ✘ | 1e-4 / 1e-6 | low |
| Inner-loop convergence tolerance | ✘ | ✘ | `inner_tol = 1e-4`, scaled by √nEl | low |
| Inner-loop iteration budget | ✘ | ✘ | 30 recorded; **shown insufficient**, §3.1 | low |
| Multiplicity tolerance | ✘ ("very small") | | `mult_tol = 1e-3` | low |
| Number of modes computed J | ✘ ("sufficiently large") | | `n_modes = 4` | med |
| Upper bound on β | ✘ (none in Eq. 25) | | `β̂ ≤ 1e6`; **not inert**, §3.4 | low |
| MMA constants a₀, a, c, d | ✘ | | 1, 0, 1e3, 1 | low |
| MMA asymptote handling across outer steps | ✘ | | reinitialised every outer step | low |
| λ̄ = cluster mean in Eq. (25c) | ✘ (paper uses individual ω_j²) | | mean substituted; identical for N=1 | high |
| **Move limit / trust region on Δρ** | ✘ **(absent from the paper)** | ✔ **hypothesised necessary, §9.1 [B]** | 0.2 in regime B | med |
| Outer step damping α | ✘ | | 0.5 in regime B | low |
| Acceptance test on the outer step | ✘ | | none on the default path | high |

---

## 2. Phase 2 — Diagnosis of the paper-literal collapse

Primary case CC. Probes in `phase2_diagnose.m`; artefacts in
`results/phase2_CC_160x20/`. Full-trajectory runs are variants V0–V3 in
Section 5.

### 2.1 The first irreversible failure is outer iteration 1

[OBS] There is no gradual degradation to locate. The very first accepted outer
step of the paper-literal regime takes ω₁ from 145.57 to 0.1143 rad/s
(160 × 20) and the design never recovers over 300 further iterations
(`results/V0_CC_160x20_i30/`). Every trigger listed in the Phase-2 brief fires
at iteration 1 simultaneously:

| Trigger | Fires at iteration 1? | Evidence |
|---|:-:|---|
| inner MMA fails to converge | **yes** | 30/30 iterations used, last change 4.00e-1 vs tolerance 5.66e-3 |
| increment saturates a large fraction of the box | **yes** | 81.5 % of variables within 1 % of a bound; ‖Δρ‖∞ = 0.499994 of a 0.4990–0.5000 box |
| linear model predicts improvement, nonlinear frequency collapses | **yes** | predicted √β = 264.87, realised ω₁ = 0.1143; ratio of realised to predicted Δλ = **−0.4327** |
| subproblem becomes numerically singular | **no** | 0 singular/RCOND warnings at iteration 1; they appear only later, on already-collapsed designs |
| volume correction destroys the MMA step | **no** | no volume correction exists; predicted volume residual −2.6e-4 |
| multiplicity logic inconsistent | **no** | N = 1 correctly, gap 1.49 |
| invalid (non-finite, out-of-bounds) increment returned | **no** | increment finite and within bounds |

### 2.2 Acceptance audit

| Question | Answer | Evidence |
|---|---|---|
| Was the inner subproblem converged? | **No** | `converged = 0`, `max_iterations`, at outer iteration 1 of *every* paper-literal and *every* Regime-B run at the recorded budget of 30 |
| Was the returned step feasible? | **Yes** | within box bounds; predicted mean density 0.49974 ≤ 0.5 |
| Was it accepted despite failed convergence? | **Yes** | the default path never consults the flag (`topopt_freq_exact.m:413-418`) |
| Was the predicted objective improvement realised? | **No** | predicted Δλ = +4.90e4, realised Δλ = −2.12e4 |
| Would a fail-closed policy have prevented the collapse? | **No — it would only have prevented the *invalid* step** | §3.1: with a budget large enough to converge, the gate passes and the collapse still occurs |

### 2.3 The inner-budget sweep

`results/phase2_CC_160x20/p1_inner_budget.csv`. Same Eq. (25) subproblem at
outer iteration 1, solved with increasing inner budgets:

| budget | paper-literal: conv | iters | ‖Δρ‖∞ | √β | **realised ω₁** | Regime-B: conv | iters | **realised ω₁** |
|---:|:-:|---:|---:|---:|---:|:-:|---:|---:|
| 1 | ✘ | 1 | 0.0074 | 146.4 | 146.91 | ✘ | 1 | 145.81 |
| 5 | ✘ | 5 | 0.0477 | 155.0 | 156.91 | ✘ | 5 | 147.57 |
| 10 | ✘ | 10 | 0.1647 | 178.2 | 184.07 | ✘ | 10 | 152.78 |
| 20 | ✘ | 20 | 0.5000 | 261.7 | **0.1585** | ✘ | 20 | 177.60 |
| **30** (recorded) | ✘ | 30 | 0.5000 | 264.9 | **0.1143** | ✘ | 30 | 177.70 |
| 60 | ✘ | 60 | 0.5000 | 265.5 | **0.1000** | ✘ | 60 | 177.32 |
| 120 | ✘ | 120 | 0.5000 | 265.6 | **0.0971** | ✘ | 120 | 177.13 |
| 300 | ✘ | 300 | 0.5000 | 265.6 | **0.0950** | **✔** | **181** | 177.06 |
| 1000 | **✔** | **312** | 0.5000 | 265.6 | **0.0953** | ✔ | 181 | 177.06 |

[OBS] The inner subproblem needs **312** MMA iterations to meet the declared
inner stopping test in the paper-literal box and **181** in the Regime-B box.
The recorded budget of 30 is an order of magnitude short. The collapse is
already complete by inner iteration 20 and converging further makes it
marginally *worse*, not better.

[OBS] **The sweep contains a transition that V1 did not exploit.** In the
paper-literal box the truncated increment grows with the budget:
‖Δρ‖∞ = 0.0074 (budget 1), 0.0477 (5), 0.1647 (10), then 0.5000 from budget 20
onwards. Realised ω₁ tracks that growth — 146.91, 156.91, **184.07**, then
collapse. At budgets 1–10 the paper-literal step *increases* ω₁; at budgets ≥ 20
it destroys the structure. Raw values in `results/phase2_CC_160x20/log.txt`
lines 9–26.

[INF] Truncating the inner solve is therefore functioning as an **undeclared and
uncalibrated restriction on step length**: a budget of 10 happens to return a
step of roughly the magnitude the linear model supports (§2.5 puts the validity
radius at t ≈ 0.3, i.e. ‖Δρ‖∞ ≈ 0.15), while the recorded budget of 30 sits past
the transition. This is a second, independent line of evidence — one that
involves no move limit, no damping and no outer bound — that what separates a
productive step from a destructive one at outer iteration 1 is its **length**.
It is used in §9.1 to address the objection that Regime B changes three step
controls at once.

[OBS] In the Regime-B box the move limit binds from budget 20 onward
(‖Δρ‖∞ = 0.2000) and realised ω₁ is flat at 177.1–177.7 across budgets 20–1000.
Additional inner iterations there change *which* elements move, not how far.

### 2.4 The subproblem is a linear program, and MMA converges to its vertex

[EV] The paper says so itself, §3.5.3: *"the suboptimization problems (25a–f) and
(26a–i) reduce to linear programming problems (see Krog and Olhoff 1999) and can
be solved using a linear programming algorithm."*

[OBS] Verified numerically. Along an arbitrary direction, the second differences
of every constraint value vanish to round-off — cluster μ₁ 4.44e-15, J-mode
4.26e-14, volume 2.22e-16 (`p2_affineness.csv`). For N = 1 the subproblem is
exactly

> max f₁₁ᵀΔρ s.t. mean(ρ+Δρ) ≤ volfrac, ρ_min−ρ ≤ Δρ ≤ 1−ρ

(the J-mode constraint is inactive by a margin of 9.5 λ̄ at iteration 1,
`p3_lp_vs_mma.csv`), whose optimum is the greedy box vertex.

| | exact LP vertex | MMA, budget 1000 |
|---|---:|---:|
| objective f ᵀΔρ | 4.938492e+04 | 4.937705e+04 (99.98 %) |
| ‖Δρ‖∞ | 0.5000 | 0.5000 |
| fraction *exactly* at a bound | **0.9997** | 0.0000 |
| fraction *within 1 %* of a bound | 1.0000 | **0.8150** |
| **realised ω₁** | **0.0638** | **0.0953** |
| predicted ω₁ | 265.66 | 265.65 |

cos(angle between the LP vertex and the converged MMA increment) = **0.9976**.

[OBS] V2 corrects a presentation error in V1, which reported 0.9997 and 0.8150
in a single "fraction at a bound" row although they are different metrics. The
exact-bound fraction of the MMA increment is 0.0000, as expected of an
interior-point-style method that approaches bounds asymptotically; its
within-1 % fraction is 0.8150 (`results/phase2_CC_160x20/log.txt` lines 33–34).
Neither number changes any conclusion.

[OBS] The LP's exact-bound fraction is 0.9997, not 1.0000 — that is
0.0003 × 3200 ≈ **one** design variable strictly between its bounds. This is the
textbook structure of a linear program whose only non-box constraint is active:
a basic optimal solution has at most one variable off a bound per active linking
constraint, here the single volume constraint. The measured value is what the
theory predicts, and V2 states the property in that exact form rather than as
"every element moves to a bound".

[CONC-preview] **At the initial uniform design, with N = 1 and the J-mode
constraint inactive**, the converged solution of the paper-explicit Eq. (25) is
the LP box vertex, to within 0.02 % in objective and 4° in direction, and both
the exact LP step and the converged MMA step collapse the structure. The
collapse at this state is therefore a property of the formulation, not of the
solver. §9.1 [A] establishes this for the initial state only; V1's extension of
the same argument to *every* outer iteration is withdrawn (§9.1).

### 2.5 Where the linear model stops being valid

`p5_step_profile.csv` — ω₁(ρ + t·Δρ) along the accepted paper-literal direction:

| t | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| realised ω₁ | 145.57 | 154.19 | 162.61 | **177.67** | 187.61 | 175.61 | 93.34 | **0.11** |
| linear model | 145.57 | 153.75 | 161.52 | 176.03 | 189.43 | 213.73 | 240.67 | 264.89 |

[OBS] The direction is good. The model is accurate to ≈ 1 % out to t = 0.2 and
still useful at t = 0.3; it fails completely beyond t ≈ 0.5. The full paper step
t = 1 corresponds to ‖Δρ‖∞ = 0.5, i.e. every element moved by half the density
range in one iteration.

[INF] **At outer iteration 1, the paper-literal failure is a step-length
failure, not a direction failure.** This probe holds the direction fixed at the
accepted paper-literal increment and varies only the scalar t, so it isolates
step length from every other control: no move limit, no damping, no outer bound
and no change to the inner solve are involved. Fractions of the *same*
destructive direction improve ω₁ by up to 29 % (145.57 → 187.61 at t = 0.3)
while the full step destroys the structure.

[OBS] Regime B's `move_lim = 0.2` lands the step at t ≈ 0.4 of the
paper-literal increment, inside the validity region, and produces ω₁ = 177.06
where the paper-literal step produces 0.095. **This is a consistency
observation, not an isolation experiment**: Regime B changes `move_lim`,
`outer_move` and `alpha` simultaneously (§5.1), so the trajectory-level
comparison V4/VR versus V0–V3 cannot attribute the difference to `move_lim`
alone. The isolation of step length rests on the t-profile above and on the
inner-budget transition of §2.3, both of which vary length with no other
control changing.

### 2.6 MMA conditioning induced by the β variable box

[OBS] `inner_loop_mma.m:112` sets `beta_max_hat = 1e6` as an "inactive large
value". It is not inactive. `mmasub` derives its asymptotes and its p/q
regularization from `xmax − xmin`, so this choice propagates:

| `beta_max_hat` | β asymptote span | max(P,Q) column-1 / column-rest ratio |
|---:|---:|---:|
| **1e6** (production) | 9.001e+03 | **5.692e+13** |
| 1e3 | 1.000e+01 | 5.692e+07 |
| 1e1 | 1.800e-01 | 5.692e+03 |
| 2 | 3.600e-02 | 2.277e+02 |

[OBS] The primal–dual system inside `subsolv` is therefore handed a 10¹³
dynamic range between the bound variable's column and the design columns. This
is the origin of the `RCOND ~ 1e-20` warnings recorded in the preceding campaign.

[INF] This is a genuine reconstruction defect, and it is *not* the cause of the
collapse: at outer iteration 1 zero singular warnings are raised, yet the
collapse is complete. It degrades the inner solve on already-degenerate designs.
It is reported, not repaired (Section 10).

### 2.7 Which reconstruction choices the first-step result depends on

Added in V2. The audit correctly observed that a report documenting eighteen
unspecified choices cannot issue a blanket verdict on the paper's procedure. The
correct response is not to weaken the first-step result but to state precisely
what it rests on — and the answer is that it rests on notably little, because
the **exact LP solution of §2.4 bypasses the inner solver entirely**.

| # | Unspecified choice (§12) | Does the first-step collapse depend on it? | Why |
|---|---|:-:|---|
| A3 | Inner convergence test and tolerance | **No** | The LP is solved exactly, without reference to any inner stopping test |
| A4 | Inner iteration budget | **No** | Same. Separately, the MMA path was swept 1 → 1000 and collapses at every budget ≥ 20 (§2.3) |
| A7 | Upper bound on β | **No** | The reduced LP of §2.4 contains no β variable; β is an MMA-side bound variable |
| A8 | MMA constants a₀, a, c, d | **No** | Same |
| A9 | MMA asymptote handling | **No** | Same |
| A5 | Multiplicity tolerance | **No** | The relative gap at the initial design is 1.494, three orders above any tolerance considered; N = 1 under all of them |
| A6 | Number of modes J computed | **No** | The J-mode constraint is inactive by a margin of 9.5 λ̄ (§2.4), so it does not enter the active LP |
| A15 | λ̄ = cluster mean | **No** | Identical to the paper's form for N = 1 |
| A10–A12 | Continuation schedule, mass model during continuation, density transfer | **No** | The collapse occurs at every p on the 1 → 3 path with a fully converged inner solve, evaluated independently at each p (§3.3) |
| A13, A14 | Move limit, outer damping | **n/a** | These are *absent* from the paper-literal regime by construction; their absence is the object of study, not an assumption |
| A1 | Mesh | **Not eliminated, but tested** | Same behaviour at 160 × 20 and 240 × 30 (§7.5). Only two meshes were tested |
| A2 | Outer stopping ε | **No** | The collapse occurs at outer iteration 1, before any stopping test is consulted |
| A16 | ρ_min = 1e-3 | **Not eliminated** | Sets the lower box bound of Eq. (25f) and therefore the LP vertex. Not varied |
| A17 | Fail-closed tolerances | **No** | Irrelevant to the ungated paper-literal path |
| A18 | Outer iteration budgets | **No** | The collapse is complete at iteration 1 |
| — | Forward model: element type, filter radius and type, mass interpolation Eq. (4b), volfrac, boundary-condition geometry | **Yes** | These define f₁₁ and hence the LP. They are inherited from the preceding mesh-resolution campaign, which verified the forward model against the paper's Fig. 2, and reproduce the published initial ω₁ to 0.4 % |

[CONC-preview] **The first-step result is independent of every MMA-related and
every multiplicity-related reconstruction choice, and of the continuation
schedule.** It depends on the forward model, on ρ_min, and on the mesh — of
which the forward model is independently verified and the mesh is tested at two
resolutions. This is why §11 states the first-step verdict as a property of the
paper-explicit formulation on this forward model, while stating every
trajectory-level verdict as a property of the tested reconstruction only.

[OBS] What §2.7 does **not** establish: that no undocumented contemporaneous
choice could change the outcome. A different ρ_min, a different filter radius, a
different volume fraction, or a step restriction — the last of which is the
subject of §9.1 — would each change the LP and could change its vertex. The
claim is scoped to the choices enumerated in §12.

---

## 3. Phase 3 — Continuation reconstructed from the paper

### 3.1 What the paper says

[EV] Du & Olhoff §2.1, on the SIMP exponent *p* of Eq. (1):

> "The power p in (1), which is termed the penalization power, is introduced
> with a view to yield distinctive '0–1' designs, and is **normally assigned
> values increasing from 1 to 3 during the optimization process**."

That is the whole of it. The expected candidate in the brief — a stiffness
penalization path p = 1 → 3 — is therefore **paper-explicit in existence** and
**entirely unspecified in schedule**. Searching the full text, the figure and
table captions, and Section 4.1 yields no stage count, no update interval, no
convergence trigger, no statement about MMA state, and no statement about the
mass model during continuation. [EV] The mass interpolation is separately fixed:
Eq. (4b) with q = 1 above ρ = 0.1 and the C¹ branch below, with the note that
the three variants (4), (4a), (4b) give "only negligible differences".

[EV] The bimaterial exponent is stated as a *constant* 3 (§2.3), and the mass
exponent r ≈ 6 is described as "much larger than the penalization power p for
the stiffness, which is kept unchanged at a value about p = 3" (§2.2) — a
sentence in tension with §2.1's "increasing from 1 to 3".

### 3.2 Provenance table for continuation

| Procedural detail | Paper explicit | Inferred | Project reconstruction | Confidence |
|---|:-:|:-:|---|:-:|
| Continuation on *p* is used at all | ✔ §2.1 | | enabled in V1/V3/V5 | high |
| Initial value of *p* | ✔ p = 1 | | 1 | high |
| Final value of *p* | ✔ p = 3 | | 3 | high |
| Number of stages | ✘ | ✘ | **5 (1, 1.5, 2, 2.5, 3)** [DEC] | low |
| Update interval / trigger | ✘ | ✘ | **fixed, 25 outer iterations per stage** [DEC] | low |
| MMA state retained or reinitialised | ✘ | ✔ | reinitialised — the production inner loop reinitialises MMA at *every* outer step regardless, so continuation changes nothing here | med |
| Mass interpolation during continuation | ✘ | ✔ | held at Eq. (4b) throughout; the paper fixes the mass model independently of *p* | med |
| Densities transferred between stages | ✘ | ✔ | **yes**, no reinitialisation (tested, T4d) | med |
| Asymptotes transferred between stages | ✘ | ✔ | not applicable (see above) | med |

[DEC] Equal-length stages spanning 1 → 3 are the simplest defensible
reconstruction of "values increasing from 1 to 3". Two alternative schedules
(coarser ladder p ∈ {1,2,3}; shorter stages, 15 iterations) are run as a
sensitivity check in Section 5.4 so that no conclusion rests on the particular
choice.

### 3.3 Continuation does not avert the collapse — at any p

`phase3_continuation_probe.m`, CC 160 × 20, uniform ρ = 0.5, inner loop run to
**full convergence** (budget 4000) at each p on the path:

| p | ω₁ at start | inner iters | ‖Δρ‖∞ | predicted √β | **realised ω₁ (paper-literal)** | same design re-evaluated at p = 3 | realised ω₁ (move-limited) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 291.14 | 84 | 0.500 | 424.05 | **97.98** | 0.141 | **326.05** |
| 1.5 | 244.82 | 77 | 0.500 | 376.96 | **20.68** | 0.154 | **279.60** |
| 2.0 | 205.87 | 201 | 0.500 | 337.01 | **3.21** | 0.118 | **240.27** |
| 2.5 | 173.11 | 248 | 0.500 | 299.94 | **0.56** | 0.107 | **206.90** |
| 3.0 | 145.57 | 312 | 0.500 | 265.65 | **0.095** | 0.095 | **177.06** |

[OBS] The paper-literal converged step collapses ω₁ at **every** p on the
continuation path. Evaluated on the common p = 3 physical model, every one of
those five designs is a mechanism (ω₁ between 0.095 and 0.154). The move-limited
step increases ω₁ at every p.

[OBS] Continuation does widen the trust radius, but nowhere near enough. Largest
step t for which the linear model is accurate to 5 %:

| p | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|
| t_max | 0.40 | 0.50 | 0.50 | 0.40 | **0.30** |

[CONC-preview] Lowering p from 3 to 1 buys a factor of ~1.3 in admissible step
length, while the paper-literal step exceeds the p = 3 validity radius by a
factor of ~3. Continuation as reconstructed cannot close that gap.

[OBS] Scope of this probe. Each row is an independent single step from the
uniform ρ = 0.5 field at the stated p; the rows are **not** a continuation
trajectory. What the probe establishes is that the full-box step is destructive
at every p, which is the relevant question — a continuation trajectory can only
ever take steps at some p on this path. It does not establish how a trajectory
that arrives at p = 2 with a non-uniform design would behave, and §5.4 addresses
that separately with three full continuation runs.

---

## 4. Phase 4 — Fail-closed inner-MMA semantics

### 4.1 Implemented policy

`recon_solve.m`, function `fc_check`. Each outer iteration calls the inner
solver, which returns the candidate increment, a convergence flag, the residual
history, the iteration count and a termination reason. The outer solver accepts
the increment **only if** all of:

1. `ih.converged` is true (the declared test `‖Δρₖ − Δρₖ₋₁‖ < inner_tol·√nEl`
   was met, not the iteration cap);
2. every constraint value is finite;
3. every component of the increment is finite;
4. `max(ρ_min−ρ, −outer_move) − tol_b ≤ Δρ ≤ min(1−ρ, +outer_move) + tol_b`;
5. `mean(ρ + Δρ) ≤ volfrac + tol_v`.

Otherwise **no outer update is applied**, the run stops with status
`INNER_FAILURE`, and the first violated condition is recorded. There is no
fallback, no partial acceptance, and — per the brief — no backtracking or step
reduction. [DEC] `tol_v = 1e-4` on mean density (0.02 % of the 0.5 volume
fraction) and `tol_b = 1e-9` absolute; both declared here and never adjusted.

[DEC] **What this policy is and is not.** Added in V2. Fail-closed acceptance is
a *declared audit convention of this campaign*, adopted so that every accepted
increment is traceable to a stated, checkable condition. It is not derived from
the paper, which is silent on inner stopping semantics, and it is not a theorem.
Specifically:

* Condition 1 is a **successive-iterate test**, not an optimality certificate.
  No KKT residual, dual residual or objective-gap bound is computed anywhere in
  this campaign (§0.0). An increment that passes condition 1 is not thereby
  proven to solve Eq. (25); an increment that fails it is not thereby proven not
  to.
* `inner_tol = 1e-4` scaled by √nEl is a reconstruction choice (§12, A3). A
  different declared tolerance moves the budget at which the gate begins to pass.
* **An inexact inner solve can be a legitimate component of an outer sequential
  method.** Nothing in this campaign shows otherwise, and §2.3 shows that
  truncation at budget 10 produces a *better* first step than convergence does.
  The report therefore does not claim that accepting truncated increments is
  mathematically invalid.

What the policy does establish is narrow, measured and unaffected by the above:
the recorded production configuration does not solve the subproblem it declares
it is solving, so its increments may not be described as solutions of Eq. (25),
and any property attributed to "the Eq. (25) iteration" on the basis of those
runs is attributed to a different iteration.

### 4.2 Regression tests

`tests/run_all_tests.m`, all 15 assertions passing:

```
[PASS] T1 A: move=Inf outer=Inf                       max|ddrho|=0 |dbeta|=0
[PASS] T1 B: move=0.2 outer=0.2                       max|ddrho|=0 |dbeta|=0
[PASS] T1 C: no J-mode                                max|ddrho|=0 |dbeta|=0
[PASS] T1 D: N=2 cluster                              max|ddrho|=0
[PASS] T2 regime A (paper-literal) bit-identical      iters 12/12  max|drho|=0
[PASS] T2 regime B (stabilized) bit-identical         iters 12/12  max|drho|=0
[PASS] T3a halts on non-converged inner, no update    status=INNER_FAILURE iters=1
[PASS] T3b without fail-closed the same step is accepted
[PASS] T3c inert when inner converges (bit-identical)
[PASS] T3d gate predicates fire correctly             vol/bound/nonfinite/clean-pass
[PASS] T4a fixed schedule realised exactly            p = [1 1 1 1 2 2 2 2 3 3 3 3]
[PASS] T4b disabled continuation holds p = cfg.penal
[PASS] T4c p=[3] schedule == continuation off (bit-identical)
[PASS] T4d densities transfer across the stage boundary
[PASS] T4e drho-trigger respects min_stage_len
```

T1 and T2 establish that the campaign's instrumented code reproduces the
production algorithm **bit-identically**; T3c establishes that the fail-closed
gate is inert whenever the inner problem does converge, so any behavioural
difference it produces is attributable solely to rejecting invalid steps.

### 4.3 What the policy reveals

[OBS] At the recorded inner budget of 30, **every** fail-closed variant halts at
outer iteration 1, in both step-control regimes, with
`inner_not_converged(max_iterations, 30 it)` and the initial uniform design
untouched (ω₁ = 145.5692, volume 0.5). See `results/V2_…_i30`, `V3_…_i30`,
`V4_…_i30`, `V5_…_i30`.

[CONC-preview] **Every outer step in every configuration documented by this
campaign and by the preceding mesh-resolution campaign — that is, every
execution of this benchmark for which a record exists — was obtained by
accepting an inner solution that had not met its own declared convergence
condition.** V1 wrote "all apparent progress the production solver has ever
made", which asserts more than the campaign can document: it covers executions
for which no record was examined. The narrowed statement is what the evidence
supports, and it covers every number this project has published for the
benchmark.

This is a defect in **inner-solve validity** as defined in §0.0 — the solver
does not solve the subproblem it declares — and it is independent of whether the
resulting trajectory looks reasonable. It is *not* a claim that the resulting
increments are mathematically illegitimate (§4.1).

---

## 5. Phase 5 — Controlled ablation matrix

### 5.1 Design of the experiment

The six variants required by the brief, plus one reference control:

| variant | continuation | fail-closed inner solve | step controls |
|---|:-:|:-:|---|
| V0 | no | no | paper-literal |
| V1 | **yes** | no | paper-literal |
| V2 | no | **yes** | paper-literal |
| V3 | **yes** | **yes** | paper-literal |
| V4 | no | **yes** | existing Regime-B step controls |
| V5 | **yes** | **yes** | existing Regime-B step controls |
| **VR** | no | no | existing Regime-B step controls |

"paper-literal" = `move_lim = Inf`, `outer_move = Inf`, `alpha = 1`,
`outer_tol = 1e-4` (`run_clamped_clamped_exact.m:11-24`). "Regime-B" =
`move_lim = 0.2`, `outer_move = 0.2`, `alpha = 0.5`, `outer_tol = 1e-6`
(`audit_optimizer_nochange.m:12-31`). Neither set is retuned.

**VR is the reference control**: it is *exactly* the configuration of the
preceding mesh-resolution campaign, including its inner budget of 30 and its
absence of any acceptance gate. It exists to quantify what accepting
non-converged inner solutions actually buys.

Because §2.3 showed the recorded inner budget of 30 to be an order of magnitude
short of inner convergence, the whole matrix is run **twice**:

* **series i30** — the recorded budget. Establishes what the on-disk
  configurations actually do.
* **series i2000** — a budget large enough that the inner subproblem reaches
  its own declared tolerance, which is what Fig. 1's inner loop requires.

[DEC] Outer iteration budgets: 300 at 160 × 20 and 150 at 240 × 30, except the
paper-literal variants of series i2000, which get 15. Their terminal
classification (`MECHANISM_COLLAPSE`) is established at outer iteration 1 and is
never reversed; the same four variants are run for the full 300 iterations in
series i30 and confirm non-recovery; and `eigs` on the resulting near-singular
stiffness matrix — not the optimizer — is what makes those runs expensive.

### 5.2 Results

`results/tables.md`, `results/classification.csv`, `results/aggregate.json`
carry the complete 19-run matrix; the continuation-schedule variants appear in
§5.4 and the 240 × 30 variants in §7.5. ω₁ is always reported at p = 3 so that
continuation and fixed-p runs are compared on one physical model.

| tag | variant | mesh | inner budget | steps | cont | FC | status | iters | ω₁ (p = 3) | % paper | final N | min g₁₂ | classification |
|---|---|---|---:|---|:-:|:-:|---|---:|---:|---:|---:|---:|---|
| V0 | V0 | 160×20 | 30 | paper-literal | n | n | MAX_ITER | 300 | 0.02 | 0.0 % | 1 | 2.45e-03 | **MECHANISM_COLLAPSE** |
| V0 | V0 | 240×30 | 30 | paper-literal | n | n | MAX_ITER | 120 | 9.08 | 2.0 % | 1 | 1.40e-02 | **MECHANISM_COLLAPSE** |
| V1 | V1 | 160×20 | 30 | paper-literal | **Y** | n | MAX_ITER | 300 | 0.02 | 0.0 % | 1 | 8.77e-01 | **MECHANISM_COLLAPSE** |
| V2 | V2 | 160×20 | 30 | paper-literal | n | **Y** | **INNER_FAILURE** | **1** | 145.57 | 31.9 % | 1 | 1.49e+00 | INNER_FAILURE |
| V3 | V3 | 160×20 | 30 | paper-literal | **Y** | **Y** | **INNER_FAILURE** | **1** | 145.57 | 31.9 % | 1 | 1.49e+00 | INNER_FAILURE |
| V4 | V4 | 160×20 | 30 | regime-B | n | **Y** | **INNER_FAILURE** | **1** | 145.57 | 31.9 % | 1 | 1.49e+00 | INNER_FAILURE |
| V5 | V5 | 160×20 | 30 | regime-B | **Y** | **Y** | **INNER_FAILURE** | **1** | 145.57 | 31.9 % | 1 | 1.49e+00 | INNER_FAILURE |
| **VR** | VR | 160×20 | 30 | regime-B | n | n | MAX_ITER | 300 | **328.55** | **72.0 %** | 1 | 4.78e-03 | OUTER_LIMIT_CYCLE |
| **VR** | VR | 240×30 | 30 | regime-B | n | n | MAX_ITER | 150 | **371.54** | **81.4 %** | 1 | 4.65e-03 | OUTER_LIMIT_CYCLE |
| V0 | V0 | 160×20 | 2000 | paper-literal | n | n | MAX_ITER | 15 | 0.04 | 0.0 % | 1 | 2.63e-05 | **MECHANISM_COLLAPSE** |
| V1 | V1 | 160×20 | 2000 | paper-literal | **Y** | n | MAX_ITER | 15 | 0.04 | 0.0 % | 1 | 1.25e+00 | **MECHANISM_COLLAPSE** |
| V2 | V2 | 160×20 | 2000 | paper-literal | n | **Y** | INNER_FAILURE | **13** | 0.02 | 0.0 % | 1 | 2.63e-05 | INNER_FAILURE |
| V3 | V3 | 160×20 | 2000 | paper-literal | **Y** | **Y** | MAX_ITER | 15 | 0.04 | 0.0 % | 1 | 1.25e+00 | **MECHANISM_COLLAPSE** |
| **V4** | V4 | 160×20 | 2000 | regime-B | n | **Y** | MAX_ITER | 300 | **300.90** | **65.9 %** | 1 | 1.48e-03 | OUTER_LIMIT_CYCLE |
| **V5** | V5 | 160×20 | 2000 | regime-B | **Y** | **Y** | MAX_ITER | 300 | **312.28** | **68.4 %** | 1 | 2.01e-02 | MAX_ITERATIONS |

Inner-solve validity, the same runs:

| tag | inner budget | inner converged / total | rejected outer steps | median inner iters | singular warnings |
|---|---:|---:|---:|---:|---:|
| V0 160×20 | 30 | **0 / 300** | 300 | 30 | 8560 |
| V0 240×30 | 30 | **0 / 120** | 120 | 30 | 2331 |
| V1 160×20 | 30 | **0 / 300** | 300 | 30 | 8293 |
| V2/V3/V4/V5 160×20 | 30 | **0 / 1** | 1 | 30 | 0 |
| **VR 160×20** | 30 | **0 / 300** | 300 | 30 | **0** |
| **VR 240×30** | 30 | **0 / 150** | 150 | 30 | **0** |
| V0 160×20 | 2000 | 14 / 15 | 1 | 113 | 2764 |
| V1 160×20 | 2000 | 15 / 15 | 0 | 65 | 1015 |
| V2 160×20 | 2000 | 12 / 13 | 1 | 129 | 2585 |
| V3 160×20 | 2000 | 15 / 15 | 0 | 65 | 1015 |
| **V4 160×20** | 2000 | **300 / 300** | **0** | 152 | **0** |
| **V5 160×20** | 2000 | **300 / 300** | **0** | 204 | **0** |

[DEC] Reading the "rejected outer steps" column. For **gated** variants (V2–V5)
it counts steps actually rejected, each of which halts the run. For **ungated**
variants (V0, V1, VR) no rejection mechanism exists and every step was applied;
the column reports how many steps the fail-closed predicate **would have
rejected** had it been active. V1 did not state this, which made the column
appear to contradict the unconditional acceptance described in §1.1 item 9.

### 5.3 What the matrix isolates

[OBS] **Effect of continuation, holding everything else fixed.**
V0 → V1 (paper-literal, no gate): collapse either way, ω₁ = 0.02 in both, at
both inner budgets. V4 → V5 (Regime-B, gated, converged inner): ω₁ 300.90 →
312.28, but the terminal behaviour *degrades* — V4 settles into a period-2-like
oscillation with a stationary objective (tail CV 1.9e-3), whereas V5 does not
become stationary within its budget (tail CV **0.259**, ω₁ ranging 98.4 – 364.0
over the last 40 iterations) and ends **disconnected** (3 raw 8-connected
components, 2 structural members, no spanning component, mid-height symmetry
0.606 against V4's 0.999). Under the tested schedules continuation does not
produce a better final state; it produces a noisier one.

[OBS] **Effect of meeting the declared inner stopping test, holding everything
else fixed.** V4 (gated, 300/300 inner solves met the test) versus VR (ungated,
0/300 met it), same step controls, same mesh: ω₁ **300.90 vs 328.55**, i.e. the
configuration whose increments do *not* solve Eq. (25) reports a **9 % higher**
frequency and a cleaner topology (1 structural component, spanning, y-symmetry
1.000, grey fraction 0.570 against V4's 0.772). Both terminate in a period-2-like
oscillation.

[INF] V2 removes V1's causal phrasing that the 9 % difference exists "*because*"
VR's solves are invalid. What is measured is that truncating MMA at 30 iterations
produces a **materially different effective update** — §2.3 shows truncation
changes both the length and the content of the increment — and that the two
different iterations terminate 9 % apart. Neither terminal value is a converged
optimum (§7.2), so the campaign cannot say which is closer to the fixed point of
the true Eq. (25) iteration, or whether that iteration has one. The defensible
statement is that the published-style figure and the figure obtained from
increments that solve the declared subproblem differ by 9 %, and that the higher
of the two comes from the configuration that does not solve it.

[OBS] **Interaction between continuation and step control.** Continuation
changes nothing in the paper-literal regime (V0 ≡ V1, V2 ≈ V3 — all collapse)
and is actively harmful under Regime-B step control (V4 → V5). There is no
combination in which continuation rescues a configuration that would otherwise
fail.

[OBS] **Singular MMA subproblems are a consequence, not a cause.** Complete
counts over all 19 runs, from `results/aggregate.json`
(`summary.n_singular_warn_total`):

| group | runs | singular/RCOND warnings |
|---|---|---:|
| collapsed paper-literal runs at the converged inner budget | V0, V1, V2, V3 @ 160 × 20 | 1015 – 2764 |
| collapsed paper-literal runs at budget 30 | V0 @ 160 × 20, V0 @ 240 × 30, V1 @ 160 × 20 | 2331 – 8560 |
| every non-collapsed run at 160 × 20 | V4, V5, V5a, V5b, VR | **0** |
| non-collapsed runs at 240 × 30 | V5, VR | **0** |
| non-collapsed run at 240 × 30 | **V4** | **7** (of 150 outer iterations) |
| runs that halt at outer iteration 1 | V2, V3, V4, V5 @ budget 30 | 0 |

V1's §5.3 stated the count as "exactly zero for V4, V5 and VR", which is correct
for the 160 × 20 comparison it was describing but not for V4 at 240 × 30, where
the aggregate records **7**. V2 reports the complete table. The conclusion is
unchanged and is if anything sharpened: collapsed designs generate warnings by
the thousand, non-collapsed designs generate at most 7 in 150 iterations, and
zero warnings are raised at outer iteration 1 where the collapse is complete
(§2.1). The warnings track the near-mechanism state, not the failure.

### 5.4 Continuation-schedule sensitivity

Because the schedule is invented (§3.2, [DEC]), two alternatives were run on the
only viable configuration (Regime-B step controls, fail-closed, converged inner
budget):

* **V5** — p ∈ {1, 1.5, 2, 2.5, 3}, 25 outer iterations per stage (primary);
* **V5a** — p ∈ {1, 2, 3}, 25 iterations per stage (coarser ladder);
* **V5b** — p ∈ {1, 1.5, 2, 2.5, 3}, 15 iterations per stage (shorter stages).

Topology is reported as *raw 8-connected components* / *structural members*
(components of area ≥ 0.5 % of the mesh); see §8 for the definitions. ω₁ (p = 3)
is the **final** value; `MECHANISM_COLLAPSE` is awarded on the run minimum.

| variant | p ladder | stage length | classification | final ω₁ (p = 3) | min ω₁ over run | min g₁₂ | tail CV of ω₁ | lag2/lag1 | final topology (raw / structural) |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| V5 | 1, 1.5, 2, 2.5, 3 | 25 | MAX_ITERATIONS | 312.28 | 15.31 | 2.01e-02 | 2.59e-01 | 1.044 | 3 / 2, not spanning, y-sym 0.606 |
| V5a | 1, 2, 3 | 25 | **MECHANISM_COLLAPSE** | **354.49** | **12.88** | 4.77e-03 | 2.58e-01 | 1.060 | 10 / 3, not spanning, y-sym 0.676 |
| V5b | 1, 1.5, 2, 2.5, 3 | 15 | MAX_ITERATIONS | 296.71 | 45.62 | 5.49e-03 | 2.81e-01 | 1.120 | 9 / 2, not spanning, y-sym 0.722; **mode 1 is 99.77 % void-localized** |

[OBS] All three tested schedules behave alike and all behave worse than no
continuation. None reaches N ≥ 2 at any iteration; all are non-stationary at the
end (tail CV of ω₁ ≈ 0.26 against V4's 0.0019); none shows period-2 structure
(lag2/lag1 ≈ 1.05 against V4's 0.19); and all end **disconnected**, against V4's
single spanning structural component. V5a dips to a minimum ω₁ = 12.88 at outer
iteration 130 and is therefore classified `MECHANISM_COLLAPSE`; V5b ends with
**99.77 % of mode 1's strain energy in ρ ≤ 0.1 elements** and so fails spectral
validity (G4), exactly as V5 does at 240 × 30.

[OBS] **The V5/V5a mechanism boundary is marginal and V2 reports it as such.**
The mechanism criterion compares the minimum ω₁ over the run against ω₁ at
outer iteration 1. For the continuation runs iteration 1 is evaluated at p = 1,
where ω₁ = 291.14, so the threshold is 14.56. V5a's minimum is 12.88 (4.42 %) and
V5's is 15.31 (5.26 %) — the two fall on opposite sides of a 5 % line by less
than one part in fifty of the reference value. The `MECHANISM_COLLAPSE` label
separating them therefore carries far less information than its name suggests,
and no conclusion in this report rests on it. Both runs are non-stationary,
disconnected and non-converged, which is the finding that matters.

[OBS] **What the schedule sensitivity does and does not establish.** V1
concluded that the result is "not an artefact of the invented schedule". That is
too strong. Three fixed schedules differing in ladder spacing and stage length
do not span the space of continuation semantics: the paper specifies no update
trigger, no per-stage convergence requirement, and no state-transfer rule, and
all three tested schedules share this campaign's choices on all of them
(§3.2, A10–A12). The supported statement is that the conclusion is **not an
artefact of the particular ladder or stage length among the three tested**, all
of which are worse than no continuation, and two of which produce spurious
localized modes. A qualitatively different continuation semantics — for example
one triggered by design change rather than iteration count, or one that also
relaxes ρ_min — remains untested. Per-gate results for these runs are in §8;
V1's summed gate scores are withdrawn (§8.3).

### 5.5 Gates for progression to 240 × 30

[DEC] Declared before the 240 × 30 runs: a variant proceeds only if it (i)
starts successfully, (ii) does not produce a near-zero-frequency mechanism, and
(iii) completes its outer budget without an `INNER_FAILURE` halt. Only **V4**,
**V5** and the reference **VR** pass. V0–V3 fail (ii) at every inner budget.

## 6. Phase 6 — Multiplicity and bimodality audit

The solver's own `N` is never trusted below. Every iteration additionally
records `g₁₂ = |ω₂ − ω₁| / max(ω₁, ε)` and the multiplicity independently
reconstructed at four tolerances; per-run files are
`multiplicity_history.csv` and `mac_history.csv`.

### 6.1 Does the trajectory ever reach N = 2?

Reference run: **V4, CC, 160 × 20, converged inner budget** — the only variant
that neither collapses nor halts.

[OBS] Over 300 outer iterations:

| quantity | value |
|---|---|
| solver-reported `N` (pre-update) | **1 at every one of the 300 iterations** |
| `N_trial` (post-update) | **1 at every one of the 300 iterations** |
| minimum post-update gap g₁₂ | **1.4766e-03**, at outer iteration 21→22 |
| iterations with g₁₂ < 0.10 | 12 |
| iterations with g₁₂ < 0.02 | **2** |
| iterations with g₁₂ < 0.005 | **1** |
| iterations with g₁₂ < 0.001 (= `mult_tol`) | **0** |
| median g₁₂ | 3.87e-01 |

[OBS] At `mult_tol = 1e-3` the code reports N = 2 at zero iterations. At the
diagnostic tolerances 1.5e-3, 2e-3, 5e-3 and 1e-2 it would report N = 2 at
**exactly one** iteration (number 22) and at 2e-2 at two (7 and 22). Loosening
the tolerance by a factor of ten therefore buys one clustered iteration, not a
clustered regime.

### 6.2 The single near-bimodal event, in detail

The critical transition, from `multiplicity_history.csv` and `mac_history.csv`
(pre-update state → accepted post-update state):

| it | N | ω₁ pre | ω₂ pre | g₁₂ pre | ω₁ post | ω₂ post | g₁₂ post | λ_J/λ̄ | Δλ realised / predicted | MAC₁₁ |
|---:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 310.37 | 375.91 | 2.11e-01 | 309.32 | 371.73 | 2.02e-01 | 1.467 | −0.011 | 0.967 |
| **21** | 1 | 309.32 | 371.73 | 2.02e-01 | **284.48** | **284.90** | **1.48e-03** | 1.444 | **−0.131** | 0.626 |
| **22** | **1** | **284.48** | **284.90** | **1.48e-03** | 294.60 | 397.18 | **3.48e-01** | **1.003** | 0.001 | **0.016** |
| 23 | 1 | 294.60 | 397.18 | 3.48e-01 | 301.26 | 383.12 | 2.72e-01 | 1.818 | 0.045 | 0.015 |

[OBS] The near-coalesced state is reached **by accident**, on a step whose
realised Δλ₁ was *negative* (−0.131 of predicted: ω₁ fell from 309.3 to 284.5),
and it is left on the very next step, which re-opens the relative gap g₁₂ from
1.48e-03 to 3.48e-01, a factor of 236.

[OBS] MAC₁₁ across that step is **0.016**. V2 attaches a caveat V1 omitted:
**individual-mode MAC is not a reliable indicator at or near a multiple
eigenvalue**, because the eigenvectors of a clustered pair are determined only
up to a rotation within their invariant subspace, so MAC₁₁ can collapse under a
basis rotation that preserves the physical modal subspace entirely. No
subspace-level MAC or other cluster-invariant diagnostic was computed in this
campaign. The retention conclusions below are therefore rested on the **relative
eigengap g₁₂**, which is basis-invariant and is measured directly, with MAC
reported as corroborating but non-decisive.

### 6.3 Is the N = 2 subproblem correctly reconstructed?

`phase6_bimodal_probe.m` takes the near-coalesced design itself
(ω = [284.4755, 284.8956, 302.0267, 391.4099], g₁₂ = 1.4766e-03) and builds the
Eq. (25) subproblem twice: once with N = 1, exactly as the solver did, and once
with the cluster forced to N = 2 (full f_sk array, J-mode moved to mode 3). Both
inner solves are run to full convergence; both accepted steps use the identical
Regime-B step controls.

| subproblem | inner iters | converged | ω₁ after | ω₂ after | **g₁₂ after** | N after | MAC₁₁ | MAC₂₂ |
|---|---:|:-:|---:|---:|---:|:-:|---:|---:|
| N = 1 (as solved) | 235 | ✔ | 294.60 | 397.18 | **3.4821e-01** | 1 | 0.0145 | 0.0000 |
| N = 2 (forced) | 159 | ✔ | 301.27 | 368.07 | **2.2172e-01** | 1 | 0.0098 | 0.0000 |

[OBS] The off-diagonal generalized gradient at this design is **as large as the
diagonal**: ‖f₁₂‖ / ‖f₁₁‖ = **1.0154**. The paper's Eq. (22) condition for
treating a cluster as a set of simple eigenvalues (vanishing off-diagonal terms)
is therefore emphatically violated, so the N = 1 treatment the solver applied at
iteration 22 is mathematically the wrong model. The generalized-gradient basis
spans 1 of the 2 modes lying within 1 % of ω₁ — it is rank-deficient with
respect to the physically clustered pair.

[OBS] Nevertheless, **engaging N = 2 does not retain the cluster**. The forced
N = 2 step is better — it re-opens the gap to 0.222 rather than 0.348, and the
increments differ (cos = 0.8932) — but both steps saturate the move limit
(‖Δρ‖∞ = 0.2000 in both cases) and both destroy mode identity (MAC ≈ 0.01).

[OBS] Retention test, eight further steps from the same design with the cluster
re-detected each iteration under a **diagnostic** `mult_tol = 1e-2` chosen
specifically so that N = 2 can engage (sensitivity diagnostic only, never used
as a primary result):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N | **2** | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| ω₁ | 301.27 | 306.09 | 306.67 | 302.82 | 310.95 | 301.95 | 314.15 | 306.03 |
| g₁₂ | 2.22e-1 | 2.78e-1 | 2.53e-1 | 3.20e-1 | 2.73e-1 | 3.34e-1 | 2.86e-1 | 3.01e-1 |
| ‖Δρ‖/√nEl | 8.46e-2 | 7.95e-2 | 7.77e-2 | 8.15e-2 | 7.80e-2 | 7.80e-2 | 7.58e-2 | 7.66e-2 |

[OBS] The cluster is left after one step regardless of the tolerance, and the
trajectory returns to the same ω₁ ≈ 305, g₁₂ ≈ 0.3 oscillation.

#### What this establishes, and what remains open

V2 substantially narrows V1's conclusion here. V1 asserted that the failure "is
not a multiplicity-detection failure and not a defect in the N = 2 subproblem:
it is a step-length failure", and supported it with the claim that
`move_lim × alpha = 0.1` is "the smallest step the procedure can take". **Both
halves of that argument were wrong or unsupported, and both are withdrawn.**

The logical error first: a move limit is an **upper** bound on the step. It does
not prevent the procedure from returning a smaller increment, so it is not a
"smallest step", and no basin width was measured against it. The correct
statement of the same observation is empirical: the accepted increment
**saturates** that upper bound at every tail iteration (‖δ‖∞ = 0.100000, §7.3),
so no *small* step is ever in fact taken — but that is a measurement of what the
iteration does, not a bound on what it could do.

[INF] There is a structural reason to expect the saturation, which V2 states as
an inference rather than as a measurement: for a simple eigenvalue the
subproblem is an LP (§2.4), and the optimum of an LP lies on the boundary of its
feasible region, which under Regime-B controls is the move-limit box for all but
the volume-linking variable. This predicts ‖Δρ‖∞ = `move_lim` whenever the LP
structure holds and the move limit is the binding box — which is what §7.3
measures. It is *not* demonstrated for the clustered subproblem, whose
constraint set is not the same LP (§9.1).

**[A] Demonstrated by this probe.** Multiplicity *detection* is not by itself
the obstacle. The forced-N = 2 experiment bypasses detection entirely — the
cluster is imposed, not found — and the resulting step still fails to retain the
cluster (g₁₂ after = 0.222). Loosening `mult_tol` by a factor of ten likewise
buys exactly one clustered iteration, not a clustered regime (§6.1). Detection
alone is therefore excluded as a sufficient explanation.

**[B] Supported.** Step length is a plausible and well-motivated contributor:
both increments saturate the move limit, the linear model is a poor predictor
here (`pred_ratio` = 0.001 across the critical step), and the step profile of
§2.5 shows the procedure has no mechanism for taking a shorter step when the
model degrades.

**Open alternatives, none excluded by this campaign.** V1 presented step length
as a diagnosis with alternatives eliminated. It is not. The following remain
live explanations for the failure to retain bimodality, individually or in
combination:

1. **The N = 2 subproblem reconstruction is not independently verified.** It was
   tested behaviourally — it runs, converges and produces a different step — but
   its generalized gradients were never checked against finite differences, its
   basis-invariance was never demonstrated, and its cluster construction was
   never validated against an independent implementation. Behavioural failure
   does not establish implementation correctness.
2. **The λ̄ substitution differs from the paper for N ≥ 2** (§10 item 7, §12
   A15). The paper writes Eq. (25c) against the individual ω_j²; the
   reconstruction uses the cluster mean. Identical for N = 1, different here.
3. **`mult_tol` and `n_modes` are reconstruction choices** (§12, A5–A6), and
   `n_modes = 4` was never varied. A larger modal window could change which
   modes enter the cluster and the J-mode constraint.
4. **The tolerance probe cannot exclude trajectory-level tolerance effects.** It
   starts from one design and takes eight descendant steps; it does not show what
   a different `mult_tol` would have done to the *earlier* trajectory that
   produced that design.
5. **Individual MAC is fragile near multiplicity** (§6.2), so the MAC ≈ 0.01
   figures do not independently establish loss of the modal subspace.
6. **The effective sample size is one.** The whole campaign contains one
   near-coalescent iterate at 160 × 20 and one solver-detected N = 2 state at
   240 × 30, and the latter sits on a badly degraded design (§6.5). No
   near-cluster at a *good* design was ever available to test.

[CONC-preview] The bounded conclusion carried forward to C4 is: **the
implemented N = 2 model, under the existing step controls, did not retain the one
near-cluster this campaign was able to test.** Detection alone is excluded;
step length is a supported contributor; the alternatives above are open.

### 6.4 Bimodality reached on collapsed designs is spurious

[OBS] The paper-literal variants **do** report N = 2 — for example V0/V2 at
outer iteration 4 of the converged-inner-budget run, with
ω₁ = 11.3021, ω₂ = 11.3024. This is a degenerate cluster of near-zero
mechanism modes on an already-destroyed structure, not the paper's bimodal
optimum. Any multiplicity statistic must therefore be read jointly with Gate G2;
`N = 2` alone is not evidence of anything.

---

### 6.5 The one N = 2 detection in the whole campaign — and what it is

[OBS] At the primary mesh the picture changes in one respect that must be
reported precisely. **V4 at 240 × 30 does report N = 2, at exactly one outer
iteration.** The context:

| it | N pre | ω₁ pre | ω₂ pre | g₁₂ pre | ω₁ post | ω₂ post | g₁₂ post | N post | Δλ real/pred | MAC₁₁ |
|---:|:-:|---:|---:|---:|---:|---:|---:|:-:|---:|---:|
| 22 | 1 | 350.87 | 370.08 | 5.48e-02 | 346.27 | 381.15 | 1.01e-01 | 1 | −0.085 | 0.891 |
| **23** | 1 | 346.27 | 381.15 | 1.01e-01 | **90.86** | **90.90** | **4.90e-04** | **2** | **−1.573** | 0.572 |
| **24** | **2** | **90.86** | **90.90** | **4.90e-04** | 315.98 | 408.51 | 2.93e-01 | 1 | 0.088 | **0.000** |
| 25 | 1 | 315.98 | 408.51 | 2.93e-01 | 349.35 | 360.37 | 3.15e-02 | 1 | 0.331 | 0.000 |

[OBS] The coalescence at iteration 23 is produced by a **destructive** step: the
subproblem predicted an increase and ω₁ fell by 74 %, from 346.27 to 90.86
(`pred_ratio = −1.573`). The resulting bimodal state sits at 20 % of the
published optimum and below the design's own initial ω₁ of 145.49. It is not a
mechanism by the declared 5 % criterion (90.86 > 7.27), so the run is not
classified `MECHANISM_COLLAPSE` — but it is plainly not the paper's optimum
either. The solver leaves it on the next step, with MAC₁₁ = **0.000**.

[CONC-preview] Across all 19 runs, `N ≥ 2` at `mult_tol = 1e-3` is reported at
**one** outer iteration of **one** run that is not an outright mechanism, and
that iteration sits at a badly degraded design. Bimodality is never reached at a
good design and is never retained for more than one iteration anywhere. Total
iterations with `N ≥ 2` at 240 × 30: V4 = 1 of 150, V5 = 0 of 150, VR = 0 of 150.

---

## 7. Phase 7 — Convergence classification

### 7.1 Declared criteria

Fixed in `analyze.py` before any run was classified, and applied identically to
every run. Tail = the last 40 outer iterations.

| criterion | definition | threshold |
|---|---|---|
| objective stationary | tail CV of ω₁ **and** \|per-iteration relative drift of ω₁\| | < 1e-2 and < 2e-4 |
| design converged | ‖ρₖ − ρₖ₋₁‖₂/√nEl at the last iteration | < `outer_tol` |
| mechanism | min ω₁ **over the run**, relative to ω₁ at iteration 1 | < 5 % |
| period-k cycle | median ‖ρₖ − ρₖ₋ₖ‖₂/√nEl over the tail, relative to the lag-1 median | < 0.25 |
| design change not decaying | slope of log₁₀‖ρₖ − ρₖ₋₁‖₂/√nEl per iteration over the tail | \|slope\| < 5e-3 |

[DEC] The status strings `OUTER_LIMIT_CYCLE` and `MECHANISM_COLLAPSE` are code
identifiers fixed in `analyze.py` before any run was classified. They are
retained unchanged so that the report matches `classification.csv`, but **the
identifier is not a claim**: `OUTER_LIMIT_CYCLE` denotes a lag-2/lag-1 ratio
below 0.25 and nothing more, and V2 does not use "limit cycle" in prose (§7.3).

Precedence: `INNER_FAILURE` → `MECHANISM_COLLAPSE` → `CONVERGED_{BIMODAL,
UNIMODAL}` → `OUTER_LIMIT_CYCLE` → `OBJECTIVE_STATIONARY_DESIGN_CHATTERING` →
`MAX_ITERATIONS`. A run is labelled `CONVERGED_*` only if the design-change
criterion is met — a stable eigenfrequency is never accepted as convergence.

#### Limits of the classifier, stated explicitly

Added in V2. The criteria above were fixed before classification and applied
uniformly, which makes them fair for their primary purpose — **rejecting**
convergence — but they are weaker instruments for assigning asymptotic
dynamical classes, and V2 uses them accordingly.

* **The rejection of convergence is robust.** Terminal design changes of
  4.2e-02 to 8.7e-02 against tolerances of 1e-6 are four to five orders of
  magnitude out. No plausible adjustment of `outer_tol`, of the tail length, or
  of the norm changes that verdict for any run.
* **`MECHANISM_COLLAPSE` is a property of the whole run, not of its end
  state.** It fires on the *minimum* ω₁ over the trajectory. A run that dips
  below the threshold and recovers keeps the label. This is deliberate — a
  trajectory that passed through a mechanism has left the regime the linear model
  describes — but the label must not be read as "ends in a mechanism". V2 marks
  every affected statement. Six of the seven runs so classified do end at
  ω₁ < 10; **V5a ends at ω₁ = 354.49** after a transient minimum of 12.88 at
  iteration 130.
* **The mechanism reference is p-dependent.** The threshold is 5 % of ω₁ at
  outer iteration 1, which for continuation runs is evaluated at p = 1
  (ω₁ = 291.14) and for fixed-p runs at p = 3 (ω₁ = 145.57). The two groups are
  therefore judged against different absolute thresholds, 14.56 and 7.28. This
  is a defect of the classifier that V2 discloses; it affects only the V5/V5a
  labelling discussed in §5.4, and no conclusion rests on it.
* **Tail length 40 and the lag-ratio threshold 0.25 are declared, not
  derived.** They were fixed in advance to prevent post-hoc tuning, but no
  independent justification is offered for either value.
* **A finite trajectory cannot prove non-convergence for unlimited iteration
  count.** Every statement below is scoped to the observed budget. "Did not
  settle within the observed budget" is used in place of V1's "wanders without
  settling".

### 7.2 Measurements required by Q3

For the three long, non-collapsing runs at 160 × 20:

| quantity | V4 (gated, inner test met) | V5 (+ continuation) | VR (reference, inner test not met) |
|---|---:|---:|---:|
| ‖ρₖ − ρₖ₋₁‖∞ (final) | **9.99999e-02** | **1.00000e-01** | **9.9997e-02** |
| ‖ρₖ − ρₖ₋₁‖₂/√nEl (final) | 8.6701e-02 | 5.8076e-02 | 5.6849e-02 |
| ‖ρₖ − ρₖ₋₂‖∞ (tail median) | **0.1360** | 0.2000 | **0.0869** |
| ‖ρₖ − ρₖ₋₂‖₂/√nEl ÷ lag-1 | **0.1895** | 1.0442 | **0.1420** |
| ‖ρₖ − ρₖ₋₃‖₂/√nEl ÷ lag-1 | 1.0029 | 1.0763 | 1.0005 |
| relative change of ω₁ (tail CV) | 1.9039e-03 | **2.5925e-01** | 1.0279e-03 |
| relative change of the subeigenvalue objective β (tail median \|Δβ/β\|) | 0.0337 | **0.7731** | 0.0282 |
| volume change per iteration | ≤ 1e-4 (volume pinned at 0.5000) | ≤ 1e-4 | ≤ 2e-4 (0.4998) |
| grayness (final) | 0.5211 | 0.4639 | 0.3982 |
| elements carrying 90 % of the squared design change | **2138 of 3200 (66.8 %)** | see `outer_history.csv` `n_top_elems_90` | see `outer_history.csv` |
| log₁₀ decay slope of the design change | 2.3045e-05 | −1.5254e-03 | −5.9383e-06 |
| **classification** | **OUTER_LIMIT_CYCLE** | **MAX_ITERATIONS** | **OUTER_LIMIT_CYCLE** |

The `outer_tol` values are 1e-6 (Regime-B). The terminal design change is
**8.7e-02**, i.e. **five orders of magnitude** above tolerance.

### 7.3 The terminal behaviour of V4 at 160 × 20 is a strong period-2-like oscillation

[OBS] From `tail_deltas.csv` (the last ten accepted increments at full spatial
resolution). **These statistics were computed for V4 at 160 × 20 only.** V1
applied the −0.982 figure to all four oscillatory runs; V2 restricts it to the
run it was measured on and reports the four-run evidence separately below.

| quantity | value (V4, 160 × 20) |
|---|---|
| ‖δₖ‖∞, every one of the last 10 iterations | **0.100000** = `move_lim` × α = 0.2 × 0.5, exactly |
| mean \|δₖ\| over elements | 0.0753 – 0.0766 (i.e. the average element moves 76 % of the maximum allowed amount, every iteration) |
| **corr(δₖ, δₖ₊₁)** | **−0.982** (mean over 9 consecutive pairs) |
| **corr(δₖ, δₖ₊₂)** | **+0.966** (mean over 8 pairs) |
| elements active (\|δ\| > 50 % of max in any tail iteration) | 2573 of 3200 (80 %) |
| spatial extent of the churn | all 20 rows, all 160 columns; activity centroid at 45 % of the half-span |

[OBS] The four-run evidence is the lag-2/lag-1 design-change ratio of §7.2 and
§7.5, which is computed identically for every run: V4 **0.1895** (160 × 20) and
**0.1901** (240 × 30), VR **0.1420** (160 × 20) and **0.0843** (240 × 30),
against a declared threshold of 0.25 and against lag-3 ratios of 1.00 for all
four. This is what supports the four-run classification.

#### The oscillation is not an *exact* period-2 cycle

[OBS] V1 described the terminal behaviour as an "exact period-2 limit cycle".
V2 withdraws "exact" and "limit cycle" as unsupported by these diagnostics, for
three reasons that are visible in the report's own numbers:

1. **The two-step return is not to the same state.** An exact period-2 orbit
   requires ρₖ₊₂ = ρₖ. The measured median ‖ρₖ − ρₖ₋₂‖∞ over the tail is
   **0.1360** for V4 at 160 × 20 and 0.0869 for VR — small relative to the lag-1
   change, but not zero, and of the same order as the move limit itself.
2. **The classifier's own threshold is 0.25, not 0.** `OUTER_LIMIT_CYCLE` is
   awarded for a lag-2/lag-1 ratio below 0.25. A ratio of 0.19 is strong
   evidence of a dominant period-2 component; it is not evidence of exact
   two-state repetition.
3. **Asymptotic language is not available from a finite budget.** "Limit cycle"
   asserts an asymptotic attractor. These are 300- and 150-iteration
   trajectories.

[CONC-preview] The supported description, used throughout V2, is: **a
persistent, move-limit-saturated period-2-like oscillation over the observed
tail**, in which the increment decomposes into a dominant sign-alternating
component (corr(δₖ, δₖ₊₁) = −0.982 for V4 at 160 × 20) and a smaller
non-repeating residual (lag-2 ratio 0.19, i.e. ~19 % of the lag-1 amplitude does
not return). It is not boundary chattering in a localized region — 80 % of
elements are active, across all 20 rows and all 160 columns — and it is not slow
drift: the log-decay slope of the design change is 2.3e-05 per iteration, i.e.
flat. ω₁ sits at 301.0 ± 1.1 throughout. The evidence is decisive for rejecting
design convergence and does not support a claim of exact or asymptotic
periodicity.

### 7.4 Why the cycle exists

[OBS] Predictive quality of the linearized subproblem, `pred_ratio` = realised
Δλ₁ ÷ predicted Δλ₁:

| run | median | 10th pct | 90th pct | # iterations with pred_ratio < 0 |
|---|---:|---:|---:|---:|
| V4 160×20 | **0.0022** | −0.011 | 0.013 | **144 / 300** |
| V5 160×20 | 0.0514 | — | — | 121 / 300 |
| VR 160×20 | 0.0024 | — | — | 145 / 300 |

[DEC] `pred_ratio` is defined as realised Δλ₁ ÷ predicted Δλ₁ and is set to
`NaN`, and excluded from every statistic, when `|predicted Δλ₁| ≤ eps`
(`recon_solve.m:318-320`). V2 records that **this never occurred**: all 300, 300,
300, 150 and 150 entries are finite in V4/V5/VR at 160 × 20 and V4/VR at
240 × 30 respectively, so no run's statistics are affected by the guard.

[INF] In the terminal regime the subproblem's linear model realises a median of
**0.2 %** of the improvement it predicts, and predicts the *wrong sign* in
roughly half of all iterations. The procedure nevertheless takes a
maximum-length step every iteration, because nothing in the tested
reconstruction — and nothing written in Du & Olhoff — reduces the step when the
model stops being predictive.

[INF] The resulting motion is best described as a **near-degenerate direction
field explored at fixed maximum amplitude**: on a plateau where the linear model
carries almost no usable information, the LP still returns a boundary point
every iteration, and the design is driven the full permitted distance towards it.
V2 reconciles this with §7.3, which V1 left in tension: the increment has a
dominant component that alternates in sign — this is the period-2 signature, and
it is what a bang-bang LP does when the active set flips between two
neighbouring vertices — plus a residual ~19 % that does not repeat, which is the
non-periodic part. Calling the whole motion a "random walk" (V1) understates the
period-2 component; calling it an "exact limit cycle" (V1, §7.3) ignores the
residual. Both descriptions are replaced by the decomposition.

### 7.5 Mesh transfer

Only the variants that passed the §5.5 progression gates were run at 240 × 30:
V4, V5 and the reference VR. Budget 150 outer iterations. V0 was additionally
run at 240 × 30 (budget 120) to confirm the paper-literal collapse transfers.

| variant | mesh | classification | ω₁ (p = 3) | % paper | min g₁₂ | ‖ρₖ−ρₖ₋₁‖₂/√nEl final | lag2/lag1 | tail CV of ω₁ | mode-1 localization |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **V4** | 160 × 20 | **OUTER_LIMIT_CYCLE** | 300.90 | 65.9 % | 1.4766e-03 | 8.6701e-02 | 0.1895 | 1.90e-03 | 1.36e-03 |
| **V4** | 240 × 30 | **OUTER_LIMIT_CYCLE** | **343.04** | **75.2 %** | **4.8995e-04** | 7.6037e-02 | 0.1901 | 2.55e-03 | 3.99e-04 |
| V5 | 160 × 20 | MAX_ITERATIONS | 312.28 | 68.4 % | 2.0090e-02 | 5.8076e-02 | 1.0442 | 2.59e-01 | 1.02e-03 |
| V5 | 240 × 30 | MAX_ITERATIONS | 248.76 | 54.5 % | 1.3455e-02 | 6.1558e-02 | 1.0922 | 2.23e-01 | **0.9972** |
| VR | 160 × 20 | **OUTER_LIMIT_CYCLE** | 328.55 | 72.0 % | 4.7808e-03 | 5.6849e-02 | 0.1420 | 1.03e-03 | 1.28e-03 |
| VR | 240 × 30 | **OUTER_LIMIT_CYCLE** | **371.54** | **81.4 %** | 4.6541e-03 | 4.2442e-02 | 0.0843 | 6.19e-03 | 6.61e-04 |
| V0 | 160 × 20 | MECHANISM_COLLAPSE | 0.02 | 0.0 % | 2.4490e-03 | — | — | — | 0.9965 |
| V0 | 240 × 30 | MECHANISM_COLLAPSE | 9.08 | 2.0 % | 1.3990e-02 | — | — | — | 1.0000 |

[OBS] **Every behaviour of the four variants run at both meshes transfers.**
Four of the nineteen runs have a mesh sibling: V0, V4, V5 and VR. For each of
them the paper-literal collapse (V0), the period-2-like oscillation with a
stationary objective (V4, VR) and the non-stationary behaviour produced by
continuation (V5) receive the same classification at both meshes. The lag-2
signature is essentially unchanged (V4 0.1895 → 0.1901). The single-step
forensics transfer too: at 240 × 30 the inner solve needs 324 iterations
paper-literal and 182 Regime-B, the converged paper-literal step gives
ω₁ = 0.1314 and the move-limited step ω₁ = 177.15 — within 1 % of the 160 × 20
figures (§2.3, §3.3). **Gate G7 passes for every variant that has a sibling
run**; it is `n/a` for the remaining fifteen.

V1's phrasing "every qualitative behaviour transfers" invited the reading that
the whole 19-run design had been replicated. It had not: V1–V3, V5a and V5b were
run at 160 × 20 only, per the §5.5 progression gates. The transfer evidence
covers the paper-literal collapse and both surviving Regime-B behaviours, which
is what the conclusions use, and nothing more.

[OBS] Refinement raises ω₁ by 12–14 % (V4 300.90 → 343.04; VR 328.55 → 371.54).
The VR figures are consistent with the preceding mesh-resolution campaign's
327.14 → 369.43 for the same configuration, which is an independent
cross-campaign check on this campaign's instrumentation.

[OBS] Refinement does **not** change the terminal classification and does not
produce a usable bimodal state. It does bring the minimum eigengap below
`mult_tol` once, at V4 iteration 24 — the campaign's only genuine `N = 2`
detection — but that state sits at ω₁ = 90.86 and is abandoned in one step
(§6.5).

[OBS] Continuation degrades under refinement rather than improving. V5 at
240 × 30 ends with **99.72 % of mode 1's strain energy in ρ ≤ 0.1 elements**: the
reported ω₁ = 248.76 is a void-localized artefact, so V5 **fails G4 at the
primary mesh** while passing it at 160 × 20.

---

## 8. Phase 8 — Acceptance gates

Full matrix in `results/gates.csv`.

| gate | criterion **exactly as implemented** in `analyze.py` | scope |
|---|---|---|
| G1 inner-solve validity | no outer step *accepted* from an inner problem that failed the declared stopping test. For fail-closed runs this holds by construction of the policy, since a violation halts before any update (`analyze.py:229-233`) | trajectory |
| G2 no mechanism collapse | min ω₁ over the run ≥ 5 % of ω₁ at iteration 1, **and** the final design has a support-to-support spanning solid component | trajectory + final |
| G3 feasibility | mean density ≤ volfrac + 1e-4 at every accepted iteration; ρ within bounds throughout | trajectory |
| G4 spectral validity | reported ω₁ equals the recomputed lowest eigenvalue **of the final design**, the run is not a mechanism, and mode 1's strain-energy fraction in ρ ≤ 0.1 elements is < 10 % | **final design only** |
| G5 multiplicity retained | `N_trial ≥ 2` at each of the last 10 iterations (`analyze.py:257-260`) | final 10 iterations |
| G6 trajectory convergence | terminal classification is `CONVERGED_BIMODAL` or `CONVERGED_UNIMODAL` | trajectory |
| G7 mesh transfer | the sibling run of the same variant at the other mesh receives the same classification; `n/a` when no sibling exists | pair |
| G8 topological plausibility | spanning, `extra_members = 0` (exactly one component of area ≥ 0.5 % of the mesh), \|mid-height symmetry\| > 0.9, grey fraction < 0.75 | final design |

[OBS] **Two corrections to V1's gate descriptions, neither of which changes any
gate result.**

1. V1 described G5 as "N ≥ 2 at each of the last 10 iterations, **or strong
   evidence bimodality is unreachable**". The second limb does not exist in the
   implementation: `analyze.py:257-260` computes G5 purely as
   `all(N_trial[-10:] >= 2)`. No subjective judgement entered any G5 result, and
   G5 evaluates to FAIL for all 19 runs on the objective criterion alone. V1's
   description was a documentation error, and it is the description that is
   corrected here, not the gate.
2. G4 is a property of the **final design**, not of a trajectory. V2 avoids
   describing any trajectory as "spectrally valid"; the phrase used is
   "spectrally valid final design".

[DEC] **Definition of "structural component".** G8 and the topology prose count
`n_members` — 8-connected components of the ρ ≥ 0.5 field whose area is at least
0.5 % of the mesh (`analyze.py:96-103`, `MEMBER_MIN_AREA = 0.005`) — while the
aggregate tables also report the raw 8-connected count `n_comp_8conn`, which
includes speckle. The two differ substantially and V1 used both without
distinguishing them: V4 at 240 × 30 has `n_comp_8conn = 3` but `n_members = 1`,
which is why it is described as a single spanning structural component and
passes G8. V4 at 160 × 20 has `n_comp_8conn = 5`, `n_members = 1`, and fails G8
on grey fraction (0.7716 against the 0.75 threshold), not on connectivity.

Read this matrix **gate by gate**. The final column is reproduced from
`results/gates.csv` for traceability only; per §8.3 it is not a score and is not
used as evidence anywhere in V2.

| tag | G1 inner | G2 no-mech | G3 feasible | G4 spectral | G5 multiplicity | G6 trajectory | G7 mesh | G8 topology | (raw count, not a score) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---:|
| V0_CC_160x20_i2000 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 1/8 |
| V0_CC_160x20_i30 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | 2/8 |
| V0_CC_240x30_i30 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | 2/8 |
| V1_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V1_CC_160x20_i30 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 1/8 |
| V2_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V2_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V3_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V3_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V4_CC_160x20_i2000 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | FAIL | 5/8 |
| V4_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V4_CC_240x30_i2000 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | 6/8 |
| V5_CC_160x20_i2000 | PASS | FAIL | PASS | PASS | FAIL | FAIL | PASS | FAIL | 4/8 |
| V5_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V5_CC_240x30_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | 3/8 |
| V5a_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V5b_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| VR_CC_160x20_i30 | FAIL | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | 5/8 |
| VR_CC_240x30_i30 | FAIL | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | 5/8 |

### 8.1 Spectral-validity evidence (G4)

Modal strain-energy fraction in ρ ≤ 0.1 elements for the final design of each
run, recomputed at p = 3 (`postprocess_final_modes.m`, `localization.json`):

| run | ω₁ | mode 1 | modes 2–4 |
|---|---:|---:|---|
| V0 160×20 (collapsed) | 0.0169 | **0.9965** | 0.9996, 0.9998, 0.9999 |
| V0 240×30 (collapsed) | 9.0826 | **1.0000** | 1.0000, 1.0000, 1.0000 |
| V1 160×20 (collapsed) | 0.0172 | **0.9966** | 0.9996, 0.9998, 0.9999 |
| V4 160×20 | 300.9043 | **0.00136** | 0.0065, 0.0068, 0.0041 |
| VR 160×20 | 328.5473 | **0.00128** | 0.0036, 0.0025, 0.0212 |
| V2/V3/V4/V5 160×20 @ budget 30 | 145.5692 | 0.0000 | 0.0000 (uniform ρ = 0.5, no low-density elements exist) |

[OBS] For every collapsed paper-literal run, essentially **100 %** of mode 1's
strain energy sits in void elements — the reported objective is a void artefact,
not a structural eigenfrequency. For V4 and VR it is ~0.13 %, a genuine global
mode. This is a clean separation and it is what G4 is testing.

### 8.2 Vacuous and structural passes

[OBS] The runs that halt at outer iteration 1 (`INNER_FAILURE`, budget 30) pass
G1, G2 and G3 **vacuously**: no step was taken, so no invalid step was accepted,
no mechanism was created and no constraint was violated. They fail G4–G8 and
their design is the untouched uniform ρ = 0.5 field (grey fraction 1.000).

[OBS] Added in V2: **G1 passes by construction for every fail-closed run**, not
by measurement. The implementation sets G1 = true whenever the fail-closed
policy is active (`analyze.py:229-231`), on the reasoning that a violating
iteration halts the run before any update can be applied. This is sound but it
means a G1 PASS on a gated run carries no information beyond "the gate was
switched on". The informative G1 results are the ungated runs V0, V1 and VR,
where G1 is genuinely measured — and fails for all three.

### 8.3 Why gate totals are not reported in V2

[DEC] V1 summed the eight gates into a per-run score and used the best score
(6/8, V4 at 240 × 30) as a headline. V2 withdraws every summed score, including
from the executive summary, §5.4 and §11. The eight gates are not a calibrated
scale and must not be added:

* they are **heterogeneous** — G3 (feasibility) is a basic sanity check that 19
  of 19 runs pass, while G6 (convergence) is the central research question;
* some passes are **vacuous** (§8.2) and some are **structural** (G1 on gated
  runs), so a pass does not uniformly denote an achievement;
* **G7 is `n/a`** for the fifteen runs without a mesh sibling, yet V1 still
  reported those runs "out of 8", which silently penalised them;
* summing implies the gates are independent, and they are not — G2, G4 and G8
  all read the same final design, and a mechanism fails all three together.

The gate matrix above is retained in full and is read **gate by gate**. The
findings that matter are stated directly: G5 and G6 fail for all 19 runs; G4
fails for every collapsed run and for the two continuation runs with
void-localized modes; G1 fails for every ungated run.

---

## 9. Phase 9 — Required scientific conclusions

### 9.1 The mechanism, stated once, with each claim at its evidence grade

**[A] Demonstrated — the initial-state subproblem and its consequence.**
At the initial uniform design, with a simple lowest eigenvalue and the J-mode
constraint inactive, Du & Olhoff's Eq. (25) reduces to a **linear program over
the full density box**. The paper says so itself (§3.5.3) and this campaign
verifies it numerically: constraints affine to 4e-15, and a converged MMA
increment within 0.02 % of the exact LP vertex objective and 4° of its direction
(§2.4). Because the volume constraint is the only active non-box constraint, a
basic optimal solution has at most one variable strictly between its bounds —
measured, 0.9997 of 3200 variables sit exactly on a bound. On the CC benchmark
that vertex is a direction of genuine ascent: the linear model is accurate to
1 % over the first 20 % of it and the design improves along it out to t ≈ 0.4
(§2.5). But the full step is roughly three times longer than the region in which
the model holds, and applying it takes ω₁ from 145.57 to 0.064 (exact LP) or
0.095 (converged MMA). §2.7 establishes that this result is independent of every
MMA-related and multiplicity-related reconstruction choice.

**Scope limit, and a withdrawal.** V1 wrote that the LP optimum is a vertex at
which "every element moves to one bound", and inferred that "without a bound,
every outer iteration produces a 0/1 design". Both are withdrawn:

* *Every element* is wrong even for the demonstrated case — one variable is
  fractional, which is precisely what LP theory predicts for a single active
  linking constraint. The corrected form is "all but one".
* *Every outer iteration* is not demonstrated. The LP analysis of §2.4 holds at
  the **initial state**, where N = 1 and the J-mode constraint is inactive by a
  margin of 9.5 λ̄. Whether a later iterate with an active J-mode constraint, or
  with N ≥ 2 and its subeigenproblem Eq. (25d), presents the same greedy-box
  structure was never tested. The campaign has no evidence either way, and the
  claim is not needed for any verdict.

**[A] Demonstrated — what the missing ingredient is not.** Continuation is
neither necessary nor sufficient, at every p on the tested 1 → 3 path and under
all three tested schedules (§3.3, §5.4). Multiplicity detection is not the
obstacle to retaining the one tested near-cluster, since forcing N = 2 bypasses
detection and still fails (§6.3). Inner-solver truncation does not explain the
collapse, since removing it makes the collapse slightly worse (§2.3).

**[B] Supported hypothesis — a finite restriction on step length is required.**
Nothing in the paper bounds the step. Four lines of evidence support the
hypothesis that some finite restriction is required for the tested
reconstruction to produce any iteration history at all:

1. **The step profile isolates length directly.** Holding the paper-literal
   direction fixed and varying only the scalar t, ω₁ rises to 187.61 at t = 0.3
   and collapses at t = 1 (§2.5). No other control varies in this probe.
2. **The inner-budget transition isolates it again, by a different route.** In
   the paper-literal box a budget of 10 returns ‖Δρ‖∞ = 0.165 and *raises* ω₁ to
   184.07, while budget 20 returns 0.500 and destroys the structure (§2.3). No
   move limit, damping or outer bound is present in either case; only the
   realised step length differs.
3. **The surviving runs are the restricted ones.** V4 and VR complete their
   budgets; V0–V3 do not (§5.2).
4. **The paper's own Fig. 4 is inconsistent with an unrestricted step.** The
   published CC history rises smoothly from ω₁ ≈ 146 to 456 over roughly 100
   iterations, whereas a full-box step changes ω₁ by three orders of magnitude
   in one iteration.

Why this is graded [B] and not [A]: line 3 is confounded, because Regime B
changes `move_lim`, `outer_move` and `alpha` together (§5.1), so it cannot
attribute survival to a move limit specifically. Lines 1 and 2 are clean but
both operate at **outer iteration 1 only**; neither shows that a finite
restriction is sufficient for a whole trajectory, and indeed §7 shows it is not.
Line 4 concerns the published computation, not the tested reconstruction.

[HYP] **Whether the historical implementation contained such a restriction is
not determined here.** Line 4 above and the observation that Svanberg's
`mmasub.m:69` exposes exactly this parameter — `move = 1.0`, its inactive value,
where standard SIMP practice uses 0.2 — are indirect evidence about
contemporaneous practice and about the published figure. They are not
implementation evidence. V1's claim that "the computation behind the paper
**must** have bounded it" is downgraded to a hypothesis that the campaign
motivates but cannot test. Establishing it would require documentary evidence
about the 2007 code.

**[C] Speculative — that the restriction must additionally contract.** V1
stated as a conclusion that "the second missing detail is that this limit must
*contract*". V2 withdraws this as a conclusion and retains it only as a
hypothesis for a separate experiment (§11), because:

* no contracting variant was implemented or run anywhere in this campaign;
* no basin width was measured, so V1's inferred contraction factor "of order 10"
  has no measurement behind it and is withdrawn entirely;
* V1's supporting premise — that `move_lim × alpha = 0.1` is "the smallest step
  the procedure can take" — is a logical error, since a move limit is an upper
  bound (§6.3);
* the observation that actually holds, that every tail step *saturates* the
  bound, is consistent with contraction being the remedy but equally consistent
  with a smaller fixed bound, with damping, with a different cluster model, or
  with another unresolved reconstruction choice. None of these was tested.

What remains, and is [A], is the negative half: **a fixed restriction at the
recorded value 0.2 is not sufficient.** V4 and VR stay alive and stay
non-converged, saturating the bound at every tail iteration while realising a
median 0.2 % of predicted improvement (§7.2–7.4).

### C1 — Is continuation necessary for avoiding the paper-literal collapse?

**No, and it is not sufficient either.**

*Measurements.* At every p on the 1 → 3 path, with the inner loop run to full
convergence, the paper-literal increment collapses ω₁: 291.14 → 97.98 (p = 1),
244.82 → 20.68, 205.87 → 3.21, 173.11 → 0.56, 145.57 → 0.095 (p = 3). Re-evaluated
on the common p = 3 model, all five resulting designs are mechanisms
(ω₁ = 0.095 – 0.154). Identical at 240 × 30 (§3.3). In the ablation, V0 and V1
are indistinguishable — ω₁ = 0.02 at budget 30, 0.04 at budget 2000.

*Interpretation.* Continuation widens the region in which the linearization
holds by a factor of ~1.3 (t_max 0.30 → 0.40); the paper-literal step overshoots
it by a factor of ~3.

*Grade.* **[A]** for "not necessary and not sufficient **as reconstructed**".
*Remaining uncertainty.* Only three reconstructed schedules were tested (§5.4),
and all share this campaign's choices for the update trigger, per-stage
convergence and state transfer, none of which the paper specifies. A
qualitatively different continuation semantics — one triggered by design change,
one that also relaxes ρ_min, or a non-monotone path — was not tried. V1's
unqualified "continuation is harmful" is replaced throughout by "harmful under
the tested reconstructed schedules".

### C2 — Is fail-closed inner MMA necessary?

**Necessary for inner-solve validity as defined in §0.0; insufficient for
viability. It is a declared audit convention, not a mathematical requirement.**

*Measurements.* At the recorded inner budget of 30 the inner subproblem
converged **0 out of 300 times** (V0, V1, VR at 160 × 20; 0/120 and 0/150 at
240 × 30). Every design update the production solver has ever applied to this
benchmark came from an increment that had not met its own declared convergence
condition. With the gate enabled at that budget, every variant halts at outer
iteration 1. At a budget where the inner solve does converge, the gate passes
and the paper-literal step is accepted — and still collapses the structure
(V2: 12/13 inner solves converged, ω₁ → 0.02; V3: 15/15 converged, ω₁ → 0.04).

*Interpretation.* Fail-closed semantics convert a silent failure of the declared
inner stopping test into a loud halt. They do not repair the formulation. The
difference they make is measurable: V4, whose increments meet the test, reports
ω₁ = 300.90 where the reference VR, whose increments do not, reports **328.55**
— 9 % higher.

*What this does and does not mean.* V1 wrote that the published-style number is
higher "*because* it rests on truncated inner solves", which asserts a causal
attribution the data do not identify. Both terminal values are samples from
non-converged oscillations (§7.2), so neither is an optimum and the campaign
cannot say which lies nearer the fixed point of the exact Eq. (25) iteration —
or whether such a fixed point exists. What is measured is that **truncation at
30 iterations produces a materially different effective update** (§2.3 shows it
changes the length and content of the increment), and that the two different
iterations terminate 9 % apart, with the higher value belonging to the
configuration that does not solve its declared subproblem.

*Grade.* **[A]** for the measurements. **Not claimed:** that the successive-
iterate test is a necessary or sufficient certificate of subproblem optimality
(none was computed, §0.0), or that an inexact inner step is illegitimate in an
outer sequential method — §2.3 shows a truncated step can be *better*.
*Remaining uncertainty.* `inner_tol = 1e-4` scaled by √nEl is itself a
reconstruction (§12, A3); a different declared tolerance would move the budget
at which the gate starts passing, though not the LP vertex the inner problem
converges to.

### C3 — Does continuation enable the transition from N = 1 to N = 2?

**No — it moves the trajectory away from coalescence.**

*Measurements.* Minimum relative eigengap over 300 iterations at 160 × 20:
V4 (no continuation) **1.4766e-03**; V5 (continuation) **2.0090e-02**, an order
of magnitude worse. Iterations with g₁₂ < 1e-2: V4 = 1, V5 = **0**, VR = 4. At
240 × 30 the same ordering holds and is sharper: min g₁₂ is 4.90e-04 for V4 —
the only value in the campaign below `mult_tol` — against 1.35e-02 for V5. Total
iterations with N ≥ 2 at 240 × 30: V4 = 1, V5 = **0**, VR = 0.

*Interpretation.* Under the three tested schedules, continuation moves the
trajectory away from the only region where coalescence occurs. The N = 2 states
that *are* reported are of two kinds, neither of them the paper's optimum:
degenerate clusters of near-zero mechanism modes on collapsed paper-literal
designs (V0/V2 at iteration 4, ω₁ = 11.3021, ω₂ = 11.3024, §6.4), and the single
V4 240 × 30 detection at ω₁ = 90.86 created by a step that cut ω₁ by 74 % (§6.5).

*Grade.* **[A]** for the measured ordering under the tested schedules.
*Remaining uncertainty.* As in C1, the result is scoped to three fixed schedules
that share unspecified continuation semantics.

### C4 — Does the first bimodal state remain stable, or does the solver leave it?

**It is reached once, by accident, and left in a single step.**

*Measurements, 160 × 20.* V4: the accepted step at outer iteration 21 lands on
ω₁ = 284.4755, ω₂ = 284.8956, g₁₂ = 1.4766e-03 — on a step whose realised Δλ₁ was
**negative** (ω₁ fell from 309.32). The next step re-opens the gap to 3.48e-01, a
factor of 236, with MAC₁₁ = **0.016**. Forcing N = 2 at that same design and
solving the full generalized-gradient subproblem to convergence gives a better
but still non-retaining step (g₁₂ after = 2.22e-01 versus 3.48e-01). Under a
diagnostic `mult_tol = 1e-2`, N = 2 engages for exactly one step and the
trajectory returns to the same ω₁ ≈ 305, g₁₂ ≈ 0.3 cycle (§6.3).

*Measurements, 240 × 30.* The same pattern, one iteration long: V4 reaches
g₁₂ = 4.90e-04 at iteration 24 — the only genuine `N = 2` detection in the
campaign — and leaves it in a single step with MAC₁₁ = 0.000, the gap re-opening
to 2.93e-01. That state was itself produced by a step that cut ω₁ from 346.27 to
90.86 (§6.5), so it is a coalescence of the wrong pair of modes at the wrong
design.

*Interpretation, restated in V2 at the strength the evidence supports.*

**[A] Demonstrated.** The cluster is not retained: the relative eigengap — a
basis-invariant quantity — re-opens by a factor of 236 at 160 × 20 and to
2.93e-01 at 240 × 30, in a single step, in every configuration tested including
the forced-N = 2 one. **Multiplicity detection alone is excluded** as the
explanation, because the forced-N = 2 probe imposes the cluster rather than
detecting it and still fails. Both increments saturate the move limit
(‖Δρ‖∞ = 0.2000 exactly). The paper's Eq. (22) condition for treating a cluster
as a set of simple eigenvalues is violated at this design
(‖f₁₂‖/‖f₁₁‖ = 1.0154), so the N = 1 treatment the solver applied was the wrong
model there.

**[B] Supported.** Step length is a plausible and well-motivated contributor:
every candidate step saturates the bound, and the procedure has no mechanism for
taking a shorter one when the model degrades.

**Withdrawn from V1.** "The smallest step the procedure can take is larger than
the basin of the bimodal state" — a move limit is an upper bound, not a smallest
step, and no basin width was measured (§6.3, §9.1). "This is not a detection
failure **and not a defect in the N = 2 subproblem**" — the second half is not
established: the N = 2 reconstruction was tested behaviourally but never verified
against derivatives or an independent implementation, so its correctness is
assumed, not shown.

**Open alternatives** — enumerated in full at §6.3: unverified N = 2
implementation; the λ̄ = cluster-mean substitution, which differs from the paper
for N ≥ 2; untested `n_modes`; trajectory-level effects of `mult_tol` that a
single-design probe cannot see; the fragility of individual-mode MAC near
multiplicity; and an effective sample size of one.

*Grade.* **[A]** for non-retention and for excluding detection alone. **[B]**
for step length as a contributor. **Not established:** that step length is the
primary cause. *Remaining uncertainty.* One near-coalescent iterate exists at
160 × 20 and one solver-detected N = 2 state at 240 × 30, the latter on a badly
degraded design; the retention test extends the sample to eight steps but from a
single starting design.

### C5 — What can and cannot be said about the residual frequency gap?

**The terminal values are not optima, so the gap is not decomposable. V2
withdraws V1's causal ranking.**

V1 answered this question with a ranked attribution headed "optimizer
non-convergence — primary". That ranking is not identifiable from these data and
is withdrawn. The reason is simple and decisive: every terminal frequency in the
campaign comes from a trajectory that has not converged (§7.2), so none of them
is the value this reconstruction would produce at convergence. Without knowing
where a converged run would land, no fraction of the 456.4 → 343.04 difference
can be assigned to non-convergence, to reconstruction ambiguity, or to anything
else. A ranking requires a counterfactual the campaign never observed.

*What is measured, and stands.*

1. **[A] The terminal values are inadmissible as optima.** Terminal design
   change 4.2e-02 to 8.7e-02 against `outer_tol` = 1e-6, four to five orders of
   magnitude out, for every non-collapsed run at both meshes. The linearized
   model realises a median 0.2 % of its predicted improvement and has the wrong
   sign in 144 of 300 iterations (V4, 160 × 20). This alone disqualifies
   343.04 rad/s, and every other figure in the campaign, from being compared to
   456.4 as optimum-to-optimum.
2. **[A] There is no large discrepancy in the initial scalar eigenfrequency.**
   The forward model reproduces the published initial value to within 0.4 %
   (145.57 at 160 × 20 and 145.49 at 240 × 30 against the paper's 146.1), and
   M-orthonormality holds to 2.9e-15. The preceding mesh-resolution campaign
   independently verified the FE model against the paper's Fig. 2.
3. **[A] Bimodality is not retained** (C4), and the published optimum is
   described as bimodal. Whatever the gap is composed of, the reconstruction
   does not reach the modal state the paper reports.
4. **[A] Eighteen procedural items are unspecified by the paper** (§12), of
   which §2.7 shows nine to be irrelevant to the first-step result and the
   remainder untested at trajectory level.

*What is explicitly not claimed.* V1's point 4 read "remaining modelling
discrepancy — not supported", which over-reads item 2 above. Agreement on one
forward state at the uniform design validates the eigensolver, the mass and
stiffness assembly and the normalization **at that state**. It does not exclude
discrepancies in the sensitivity expressions, the filter, cluster handling, the
interpolation derivatives, continuation details, or any other quantity that only
enters along the optimization path. The narrow statement — no large discrepancy
in the reported initial scalar eigenfrequency — replaces it.

*Grade.* **[A]** for items 1–4 individually. **No grade is assigned to any
causal decomposition of the gap, because none is supported.**

### C6 — Does the tested paper-literal reconstruction become viable after faithful continuation and a converged inner solve, without modern globalization?

**No, for the tested reconstruction.** V3 — continuation on the reconstructed
p = 1 → 3 path, fail-closed inner semantics, an inner budget large enough that 15
of 15 inner solves meet the declared test, and paper-literal step control —
terminates in `MECHANISM_COLLAPSE` with ω₁ = 0.04 rad/s. V2, the same without
continuation, collapses identically. The result holds at **both tested meshes**
(§3.3, 240 × 30) and at **every tested inner budget ≥ 20** (§2.3).

*Precision on scope.* V1 called this result "mesh-independent" and
"budget-independent". Both are replaced. Two meshes were tested, which supports
"consistent on both tested meshes" and not independence in general. Budgets 1,
5, 10, 20, 30, 60, 120, 300 and 1000 were tested in the sweep and 30 and 2000 in
the matrix; the collapse occurs at every tested budget **from 20 upward**, and
notably **not** at budgets 1–10, where the truncated step is short enough to
improve ω₁ (§2.3). Stating this correctly strengthens rather than weakens the
finding, because the budget at which the collapse begins is exactly the budget
at which the increment reaches full box length.

*Grade.* **[A]** for the tested reconstruction. This is the strongest negative
result of the campaign, and §2.7 shows it to be independent of nine of the
eighteen unspecified reconstruction choices — including every MMA-related one,
since the exact LP solution bypasses MMA altogether. It is a direct answer to
the brief's central question about the paper-literal regime.

**[HYP]** It is **not** a verdict on the historical 2007 implementation, which
may have contained undocumented details this reconstruction does not
reproduce — a step restriction being the leading candidate (§9.1).

### C7 — Which current project conclusions must be withdrawn, retained, or reinterpreted?

**Withdraw.**

* Any statement that an `OlhoffApproachExact` optimized frequency is a
  *converged* optimum. Every non-collapsing run in this campaign terminates
  four to five orders of magnitude above its own stopping tolerance, in a
  persistent period-2-like oscillation or without settling. This applies to the
  160 × 20 and 240 × 30 numbers of the mesh-resolution campaign (327.14, 369.43)
  and to the numbers reported here (300.90, 312.28, 328.55, 371.54) alike.
* Any statement resting on the recorded `413.869` figure for CC 40 × 5. The
  mesh-resolution campaign already could not reproduce it; the reference
  configuration re-measures at 328.55 over 300 iterations at 160 × 20.
* Any characterisation of the production solver's inner loop as "converged" at
  the recorded budget. It does not meet its own declared stopping test, in any
  run, at any mesh.

**Reinterpret.**

* The mesh-resolution campaign's bounded exception — "a residual ω₁ gap of
  roughly 20 % and the absence of bimodality survive refinement and require a
  separate, optimizer-side explanation" — is now **partly** answered. What is
  established is that the exception cannot be resolved by comparing terminal
  frequencies at all, because the trajectories that produce them do not converge
  (C5). The absence of bimodality is confirmed and characterised (C4). The
  *size* of the residual gap remains unexplained, and V1's attribution of it to
  optimizer non-convergence is withdrawn as unidentifiable.
* The recorded diagnosis "N = 1 LP bang-bang divergence" is **confirmed and
  quantified at the initial state**: the converged inner solve reproduces the
  exact LP vertex to 0.02 % in objective and cos = 0.9976 in direction. Its
  extension to every outer iteration is not established (§9.1).
* Regime B's `move_lim`/`outer_move` should be described as **a reconstruction
  of a step restriction that the tested formulation appears to require [B]**,
  rather than as ad hoc non-paper stabilization. They should **not** be
  described as a restriction that the paper's procedure is known to have
  contained [HYP]. Svanberg's `mmasub` exposes the identical parameter at its
  inactive value 1.0, and the paper's Fig. 4 is inconsistent with an
  unrestricted step; both are indirect evidence, not implementation evidence.
* The `RCOND ~ 1e-20` singular-subproblem storm is a **consequence** of the
  collapsed design, not a cause: 1015 – 8560 warnings in every collapsed run,
  zero in every non-collapsed run at 160 × 20, zero for V5 and VR at 240 × 30,
  and 7 in 150 iterations for V4 at 240 × 30 (§5.3).

**Retain.**

* The forward-model verification and the mesh-resolution campaign's H1 findings.
* That the tested paper-literal reconstruction is numerically non-viable — now
  with a demonstrated mechanism at outer iteration 1 rather than an inference.

**Restrictions on use, stated by claim class rather than by directory.** V1
wrote that `analysis/OlhoffApproachExact` "must not be used for reviewer-facing
comparisons, speed-ups, frequency gaps, convergence claims or optimality
claims". V2 keeps every item on that list and makes explicit that it is a list
of **claim classes**, not a blanket prohibition on the directory:

| Use | Status after this campaign |
|---|---|
| Any claim that a frequency from this directory is a converged optimum | **Forbidden** — no run converges |
| Comparison of a terminal frequency against the paper's 456.4 as optimum-to-optimum | **Forbidden** — C5 |
| Speed-up or iteration-count comparisons derived from these trajectories | **Forbidden** — the trajectories terminate on budget, not on convergence |
| Any description of the production inner loop as "converged" at the recorded budget | **Forbidden** — 0/300 at every mesh |
| The FE forward model, its verification against Fig. 2, and initial-state frequencies | **Usable**, independently verified |
| Topology-class observations, with the explicit caveat that the design is a sample from a non-converged oscillation | **Usable with caveat** |
| The diagnostic findings of this campaign — the LP structure at the initial state, the budget transition, the gate results | **Usable**, they are this report's contribution |

## 10. Deficiencies identified but deliberately NOT modified

Per the controlled-experiment requirement, the following were measured and are
recorded without any change being made to the production solver.

1. **The β variable's box `[0, 1e6]` is not numerically inert** (§2.6).
   `inner_loop_mma.m:112` describes it as "an inactive large value"; because
   `mmasub` derives its asymptotes and p/q regularization from `xmax − xmin`, it
   sets the bound variable's asymptote span to 9.0e3 and creates a 5.7e13 dynamic
   range between the β column and the Δρ columns of the MMA subproblem matrices.
   This is the origin of the `RCOND ~ 1e-20` warnings recorded previously.
   *Not modified: it is not the cause of the collapse (zero warnings at outer
   iteration 1, where the collapse is complete), and changing it would confound
   the reconstruction experiment.*

2. **`inner_max_iter = 30` is an order of magnitude too small** (§2.3). The
   inner subproblem needs 312–324 MMA iterations in the paper-literal box and
   181–182 in the Regime-B box. Over 300 outer iterations at the recorded
   budget, the inner problem converged **0/300 times**. *Not modified in the
   recorded regimes; a larger budget is run as a separate, labelled variant
   series so that both are reported.*

3. **The outer loop never consults `hist.inner_converged`**
   (`topopt_freq_exact.m:413-418`). The flag is computed and stored but the
   update is applied unconditionally. *Not modified; the fail-closed policy is
   implemented only in the additive campaign solver.*

4. **No move limit exists in the paper; one appears to be required by the
   tested reconstruction** ([B], §9.1). *Not modified: Regime B's recorded 0.2
   is used as-is and never retuned; no adaptive or contracting variant was
   introduced.*

5. **The move limit does not contract, and the terminal iteration saturates it
   at every step** (§7.3). Whether contraction would permit convergence or
   retention of a coalesced state is **untested** ([C], §9.1); the saturation is
   measured, the remedy is not. The production solver already contains a
   default-disabled `post_coalescence_trust_enabled` switch of that shape.
   *Not modified and not enabled: it is a proposal for a separate procedural
   experiment, not part of this diagnosis.*

6. **`mult_tol = 1e-3` is marginal at the one near-coalescent iterate**
   (g₁₂ = 1.4766e-3, §6.3). *Not modified; alternative tolerances were evaluated
   only as sensitivity diagnostics and are never used as a primary result.*

7. **λ̄ substitution in Eq. (25c).** The paper writes the cluster constraint
   against the individual ω_j²; the reconstruction uses the cluster mean. These
   are identical for N = 1 and differ at O(mult_tol) for N ≥ 2. *Not modified.*

8. **No volume projection or correction exists anywhere** (§1.2). Volume is
   enforced only inside the MMA subproblem and only up to MMA's artificial-
   variable relaxation (`c = 1e3`). Observed residuals are small
   (−2.6e-4 at outer iteration 1) but not zero. *Not modified.*

---

## 11. Final verdict

# `THE TESTED FULL-BOX RECONSTRUCTION IS NUMERICALLY NONVIABLE`

**Precise statement of the verdict.** The incremental procedure defined by
Eq. (25a–f) solved to convergence, with the box bounds of Eq. (25f) as the
**only** restriction on Δρ, and the Fig. 1 update ρ := ρ + Δρ applied in full,
destroys the clamped–clamped structure on the first outer iteration. It does so:

* at **both tested meshes**, 160 × 20 and 240 × 30;
* at **every tested point** of the penalization path, p = 1, 1.5, 2, 2.5, 3;
* at **every tested inner budget from 20 upward** — 20, 30, 60, 120, 300, 1000
  in the sweep and 2000 in the matrix — including budgets at which the inner
  subproblem provably reaches its own declared convergence condition; and
* when the subproblem is solved **exactly as a linear program**, bypassing MMA
  altogether.

The converged increment is the vertex of the linear program the paper itself
identifies in §3.5.3, reproduced here to 0.02 % in objective and cos = 0.9976 in
direction. Neither continuation nor fail-closed inner semantics — the two
candidate missing ingredients the brief nominated — changes this.

**What the verdict is a verdict on.** The subject is the **tested full-box
reconstruction**, and, for the first-step result specifically, the
**paper-explicit formulation evaluated on this verified forward model**: §2.7
shows that result to be independent of every MMA-related and every
multiplicity-related unspecified choice, and of the continuation schedule. The
subject is **not** the historical 2007 implementation. V1's headline
`PAPER-LITERAL PROCEDURE REMAINS NUMERICALLY NONVIABLE` read as a verdict on
what Du and Olhoff actually did, which this campaign cannot deliver: the paper
does not specify eighteen procedural details (§12), and an undocumented
contemporaneous choice — a step restriction being the leading candidate (§9.1)
— would not appear in the printed formulation. What the campaign shows is that
**the printed formulation, taken literally and completely, does not work on this
forward model**.

[HYP] The natural inference — that the 2007 computation therefore relied on
something not printed — is stated here as a hypothesis and not as a conclusion,
because it rests on a disjunction the campaign cannot close. If the printed
formulation collapses at outer iteration 1 while the paper reports a smooth
~100-iteration history (Fig. 4), then **either** the reconstruction's forward
model differs from the original in some way that changes the LP vertex, **or**
the original computation applied a restriction that is not in the printed
formulation. The forward model is independently verified against Fig. 2 and
reproduces the published initial frequency to 0.4 %, which makes the second
branch the more likely, but "more likely" is not "demonstrated". Discriminating
between the branches would require documentary evidence about the original
implementation, which this campaign does not have.

**Two qualifications V1 overstated, corrected here.** "Mesh-independent" becomes
"consistent on both tested meshes". "At every budget from 20 to 2000" becomes
"at every tested budget from 20 upward", and V2 adds the finding that the
collapse does **not** occur at budgets 1–10, where truncation returns a short
enough step to raise ω₁ (§2.3) — a transition that locates the failure precisely
at the point where the increment reaches full box length.

`PARTIAL RECONSTRUCTION — BIMODALITY NOT REPRODUCED` was considered and
**rejected** as the primary verdict: it would imply that a defensible
reconstruction produced a converged trajectory that merely failed to coalesce.
No variant in this campaign produced a converged trajectory. Gate G6 fails for
all 19 runs.

### Sub-verdicts

| aspect | verdict | grade | key evidence |
|---|---|:-:|---|
| **Continuation** | **NOT NECESSARY, NOT SUFFICIENT, AND HARMFUL UNDER ALL THREE TESTED SCHEDULES** | [A] | Collapse occurs at every tested p on the 1 → 3 path with a fully converged inner solve (§3.3); V0 ≡ V1 and V2 ≈ V3; under Regime-B step control continuation degrades the terminal state from a stationary period-2-like oscillation to a non-stationary one (tail CV 0.0019 → 0.2592) and the topology from connected/symmetric to disconnected (V4 vs V5), at both meshes and under all three tested schedules. Scope: the three schedules share this campaign's unspecified continuation semantics (§5.4) |
| **Inner-solve validity** | **NEVER ACHIEVED AT THE RECORDED BUDGET; ENFORCING IT IS INSUFFICIENT FOR VIABILITY** | [A] | 0 / 300 inner solves met the declared stopping test at `inner_max_iter = 30` (0 / 120 and 0 / 150 at 240 × 30); 312–324 iterations are required paper-literal, 181–182 Regime-B; with a sufficient budget the gate passes and the paper-literal step still collapses (V2, V3). "Validity" here is the operational sense of §0.0, not mathematical optimality — no KKT certificate was computed anywhere |
| **Multiplicity transition** | **NOT REPRODUCED** | [A] | Across 19 runs, `N ≥ 2` is reported at **one** outer iteration of **one** non-mechanism run (V4, 240 × 30, iteration 24) — and that state sits at ω₁ = 90.86, 20 % of the published optimum, created by a step that cut ω₁ by 74 % (`pred_ratio` = −1.573). It is abandoned in one step, the relative eigengap re-opening to 2.93e-01. At 160 × 20 the closest approach is g₁₂ = 1.4766e-03, likewise reached on a step with negative realised Δλ₁ and abandoned immediately (gap × 236). Forcing N = 2 with the full generalized-gradient array does not retain it. G5 fails for all 19 runs |
| **Why multiplicity is not retained** | **DETECTION ALONE EXCLUDED; STEP LENGTH SUPPORTED BUT NOT ISOLATED** | [A] / [B] | The forced-N = 2 probe bypasses detection and still fails, which excludes detection alone [A]. Step length is a supported contributor [B]: both candidate steps saturate the move limit. **Not established**, and open: correctness of the N = 2 reconstruction, the λ̄ substitution, `n_modes`, trajectory-level `mult_tol` effects, and an effective sample size of one (§6.3) |
| **Design convergence** | **NOT ACHIEVED WITHIN THE OBSERVED BUDGETS** | [A] | Terminal design change 4.2e-02 to 8.7e-02 against `outer_tol = 1e-6` — four to five orders of magnitude. Persistent period-2-like oscillation: lag-2/lag-1 ratios 0.19, 0.19, 0.14, 0.08 against a 0.25 threshold and lag-3 ratios of 1.00; ‖δ‖∞ = 0.100000 = move limit × α at every tail iteration. For V4 at 160 × 20 specifically, corr(δₖ, δₖ₊₁) = −0.982 and corr(δₖ, δₖ₊₂) = +0.966, with 80 % of elements active across the whole beam. **Not an exact cycle** — median lag-2 ‖·‖∞ is 0.136, not zero (§7.3). Same classification at both meshes |
| **Agreement with the published topology** | **PARTIAL** | [A] | V4 at 240 × 30 is in the published Fig. 3c morphological class — single spanning structural component (`n_members = 1`; raw 8-connected count 3), mid-height symmetry 0.971, mid-span symmetry 0.954, grey fraction 0.604 — and passes G8. So does the reference VR at both meshes. But the design is a sample from a non-converged oscillation, not a converged optimum, so the resemblance is not a reproduction |
| **Agreement with the published eigenfrequency** | **NOT ACHIEVED; THE GAP IS NOT DECOMPOSABLE** | [A] | Best result from a gated run with a spectrally valid final design: **343.04 rad/s = 75.2 %** of the published 456.4 (V4, 240 × 30). The highest number in the campaign, 371.54 = 81.4 % (VR, 240 × 30), comes from the configuration in which no inner solve met the declared test. Both are samples from non-converged oscillations, so neither is admissible as an optimum and no part of the difference can be attributed to a specific cause (C5) |

### The decisive question

> *Does a mathematically valid, feasible, converged optimization trajectory reach
> the clustered lowest-eigenvalue state described by Du and Olhoff?*

**No.** Of the 19 runs, 12 either halt on a failed acceptance gate (5) or are
classified `MECHANISM_COLLAPSE` (7). Of those 7, six end at ω₁ < 10; the seventh,
**V5a, ends at ω₁ = 354.49** and carries the label because of a transient
minimum of 12.88 at iteration 130 — the classifier fires on the minimum over the
run, not on the terminal state (§7.1). V1's statement that seven runs "end in" a
mechanism was therefore false for V5a and is corrected here.

Two further runs — V5 at 240 × 30 and V5b at 160 × 20, both continuation
variants — report an ω₁ whose mode carries over 99 % of its strain energy in void
elements, and fail G4. The remaining 5 are feasible with spectrally valid final
designs, and **none of them converges**: four terminate in a persistent
period-2-like oscillation (V4 and VR at both meshes) and one does not settle
within its budget (V5 at 160 × 20), in every case with a terminal design change
four to five orders of magnitude above their own stopping tolerance. Across all
19 runs a clustered lowest eigenvalue is held for at most one iteration.

**G5 (multiplicity retained) and G6 (trajectory converged) — the two gates that
define the question — fail for all 19 runs.** V2 reports this directly rather
than through V1's "best gate score 6/8", which summed heterogeneous, partly
vacuous and partly `n/a` criteria into a number that carried no calibrated
meaning (§8.3).

### What would have to change, stated as a hypothesis for a separate experiment

**[C] Speculative. Not implemented, not tested, and explicitly outside this
diagnostic campaign.** The campaign motivates — but does not establish — one
specific procedural addition: **a step restriction on Δρ that contracts when the
realised change in λ₁ falls short of the change the subproblem predicted.**

What the diagnosis actually supports, separated by grade:

* **[B]** A *finite* restriction is required to keep the iteration alive at all.
  Supported by the t-profile at fixed direction (§2.5) and by the inner-budget
  transition (§2.3), both of which vary step length alone; and consistent with
  V4/VR surviving where V0–V3 do not, though that comparison changes three
  controls together.
* **[A]** A fixed restriction at the recorded value 0.2 is *not* sufficient to
  converge: every terminal step saturates it while realising a median 0.2 % of
  predicted improvement.
* **[A]** The near-coalesced state is abandoned by a single saturated step.
* **[C]** That *contraction specifically* is the remedy. This does not follow
  from the three points above. A smaller fixed restriction, damping alone, a
  different cluster model, or another unresolved reconstruction choice would be
  equally consistent with them, and none was tested.

**Withdrawn from V1.** The claim that the required contraction factor is "of
order 10 on this benchmark" is withdrawn in full. It rested on treating
`move_lim × alpha` as a lower bound on the achievable step and comparing it to a
basin width that was never measured. Neither quantity supports the arithmetic.

An experiment that would settle this is well defined and small: run the V4
configuration with (i) a smaller fixed restriction, (ii) damping alone, and
(iii) a contracting restriction keyed on `pred_ratio`, and compare convergence
and cluster retention. Until it is run, contraction is a hypothesis. The
production solver already contains a default-disabled
`post_coalescence_trust_enabled` switch of the shape (iii) would require. It was
deliberately **not** enabled anywhere in this campaign.

## 12. Remaining reconstruction assumptions

Every item below is a choice this campaign had to make because Du & Olhoff
(2007) does not specify it. Each is held fixed across all variants. The final
column, added in V2, records whether the **first-step collapse result** (§2.4,
C6 — the campaign's strongest finding) depends on the choice; the derivation is
in §2.7. Trajectory-level findings depend on more of these, and are labelled
throughout as properties of the tested reconstruction.

| # | Assumption | Value used | Basis | Sensitivity tested? | First-step result depends on it? |
|---|---|---|---|---|:-:|
| A1 | Finite-element mesh | 160 × 20 (diagnostic), 240 × 30 (primary) | preceding mesh-resolution campaign | yes — both meshes reported | tested, both agree |
| A2 | Outer stopping norm and ε | RMS, ε = 1e-4 (paper-literal) / 1e-6 (Regime-B) | on-disk regimes | no — but no run ever approaches either | **no** |
| A3 | Inner convergence test | `‖Δρₖ − Δρₖ₋₁‖ < inner_tol·√nEl`, `inner_tol = 1e-4` | on-disk regimes | partially — budget swept 1…1000 | **no** (exact LP) |
| A4 | Inner iteration budget | 30 (recorded) and 2000 (converged) | both reported separately | yes — full sweep, §2.3 | **no** (exact LP) |
| A5 | Multiplicity tolerance | 1e-3 | on-disk regimes | yes — 1e-4…5e-2 as diagnostics | **no** (gap = 1.494) |
| A6 | Number of modes J computed | 4 | on-disk regimes | no | **no** (J inactive) |
| A7 | Upper bound on β | β̂ ≤ 1e6 | production `inner_loop_mma.m` | yes — conditioning probe, §2.6 | **no** (exact LP) |
| A8 | MMA constants a₀, a, c, d | 1, 0, 1e3, 1 | production `inner_loop_mma.m` | no | **no** (exact LP) |
| A9 | MMA asymptote handling across outer steps | reinitialised each outer step | production `inner_loop_mma.m` | no | **no** (exact LP) |
| A10 | Continuation stage count and length | 5 stages (1, 1.5, 2, 2.5, 3), 25 outer iterations each | simplest reading of "increasing from 1 to 3" | yes — two alternative schedules, §5.4 | **no** (collapse at every tested p) |
| A11 | Mass model during continuation | Eq. (4b) held fixed | paper fixes the mass model independently of p | no | **no** |
| A12 | Density transfer across continuation stages | transferred, no reinitialisation | simplest defensible choice | no | **no** |
| A13 | Move limit / trust region | 0.2 (Regime-B recorded value), fixed | hypothesised necessary [B], §9.1; value not retuned | partially — §2.5 profiles the validity radius | absent by construction |
| A14 | Outer damping α | 0.5 (Regime-B recorded value) | on-disk regime | no | absent by construction |
| A15 | λ̄ = cluster mean in Eq. (25c) | mean | identical for N = 1 | no | **no** (N = 1) |
| A16 | ρ_min | 1e-3 | production default | no | **yes** — sets the LP's lower box bound |
| A17 | Fail-closed tolerances | vol 1e-4, bounds 1e-9 | declared in §4.1 | no | **no** |
| A18 | Outer iteration budgets | 300 (160 × 20), 150 (240 × 30), 15 for collapsed paper-literal variants at the converged inner budget | declared in §5.1 | partially | **no** (collapse at iteration 1) |

Beyond these eighteen, the **forward model** — element formulation, filter type
and radius, mass interpolation Eq. (4b), volume fraction, support geometry — is
inherited from the preceding mesh-resolution campaign, which verified it against
the paper's Fig. 2 and reproduces the published initial ω₁ to 0.4 %. The
first-step result does depend on it, and that dependence is the principal
residual caveat on the campaign's strongest finding.

---

## Appendix A — Reproduction

```bash
cd analysis/OlhoffApproachExact/experiments/faithful_reconstruction

# 0. regression suite: equivalence to production + fail-closed + continuation
matlab -batch "cd tests; warning('off','all'); run_all_tests"

# 1. Phase 1 — narrated algorithmic trace, both regimes
matlab -batch "addpath(pwd); warning('off','all'); \
    phase1_trace(160,20,'CC','A',2); phase1_trace(160,20,'CC','B',2)"

# 2. Phase 2 — paper-literal collapse forensics, both meshes
matlab -batch "addpath(pwd); warning('off','all'); phase2_diagnose(160,20,'CC')"
matlab -batch "addpath(pwd); warning('off','all'); phase2_diagnose(240,30,'CC')"

# 3. Phase 3 — continuation probe at every p on the 1->3 path, both meshes
matlab -batch "addpath(pwd); warning('off','all'); phase3_continuation_probe(160,20,'CC')"
matlab -batch "addpath(pwd); warning('off','all'); phase3_continuation_probe(240,30,'CC')"

# 4. Phase 5 — ablation matrix at 160x20 (run the six series in parallel)
matlab -batch "warning('off','all'); drive('alphaA',160,20)"   # V0,V1 @ budget 30
matlab -batch "warning('off','all'); drive('alphaB',160,20)"   # V2,V3 @ budget 30
matlab -batch "warning('off','all'); drive('alphaC',160,20)"   # V4,V5 @ budget 30
matlab -batch "warning('off','all'); drive('betaA',160,20)"    # V0,V1 @ budget 2000
matlab -batch "warning('off','all'); drive('betaB',160,20)"    # V2,V3 @ budget 2000
matlab -batch "warning('off','all'); drive('betaC',160,20)"    # V4,V5 @ budget 2000
matlab -batch "warning('off','all'); drive('sched',160,20)"    # V5a,V5b schedule sensitivity

# 5. reference control = the preceding mesh-resolution campaign's configuration
matlab -batch "addpath(pwd); warning('off','all'); \
    run_variant('VR',160,20,'CC',struct('outer_max_iter',300,'inner_max_iter',30,'tag_suffix','_i30')); \
    run_variant('VR',240,30,'CC',struct('outer_max_iter',150,'inner_max_iter',30,'tag_suffix','_i30'))"

# 6. Phase 7 / Gate G7 — mesh transfer at 240x30
matlab -batch "addpath(pwd); warning('off','all'); \
    run_variant('V4',240,30,'CC',struct('outer_max_iter',150,'inner_max_iter',2000,'tag_suffix','_i2000'))"
matlab -batch "addpath(pwd); warning('off','all'); \
    run_variant('V5',240,30,'CC',struct('outer_max_iter',150,'inner_max_iter',2000,'tag_suffix','_i2000'))"
matlab -batch "addpath(pwd); warning('off','all'); \
    run_variant('V0',240,30,'CC',struct('outer_max_iter',120,'inner_max_iter',30,'tag_suffix','_i30'))"

# 7. Phase 6 — N=1 vs N=2 probe at the near-coalescent iterate
matlab -batch "addpath(pwd); warning('off','all'); phase6_bimodal_probe('V4_CC_160x20_i2000')"

# 8. spectral-validity post-processing (Gate G4) and aggregation
matlab -batch "addpath(pwd); warning('off','all'); postprocess_final_modes"
python3 analyze.py
python3 topology_maps.py
```

Integrity check — must print nothing:

```bash
git status --porcelain analysis/OlhoffApproachExact/Matlab tools/Matlab
shasum -a 256 -c <(grep -v '^#' results/production_sha256.txt | awk '{print $1"  "$2}')
```

**Nothing in Appendix A changed between V1 and V2.** V2 reruns no experiment and
regenerates no artefact; every number in this report is the number V1 reported,
read from the same files in `results/`.

---

## Appendix B — Response to Independent Audit

The report was reviewed by an independent scientific audit conducted in the
style of an anonymous journal review (`independent_scientific_audit_phase_b.md`,
recommendation: *major revision*). Every criticism is answered below. Decisions
are **Accepted**, **Partially accepted** or **Rejected**; a rejection is
accompanied by the reason the report should stand.

The audit was well founded. Of 39 numbered criticisms, 27 are accepted in full,
10 partially, and 2 rejected. Its central charge — that the report's strongest
mechanistic and historical claims exceeded its own evidence — was correct, and
V2 acts on it throughout. Where V2 declines to follow the audit, it is because
the report's evidence is stronger than the audit credited, not weaker.

### B.1 Critical findings

| # | Audit point | Decision | Action taken |
|---|---|---|---|
| C-1 | Final verdict exceeds the identifiable reconstruction; condemns the historical procedure | **Accepted** | §11 headline changed to `THE TESTED FULL-BOX RECONSTRUCTION IS NUMERICALLY NONVIABLE`; §0.0 added, defining paper-explicit formulation / tested reconstruction / historical implementation and forbidding conflation; C6 restated with an explicit [HYP] paragraph disclaiming any verdict on the 2007 implementation |
| C-1b | "Mesh-independent" too strong for two meshes | **Accepted** | Replaced by "consistent on both tested meshes" at every occurrence (§11, C6) |
| C-2a | "Smallest step the procedure can take" is a logical error — a move limit is an upper bound | **Accepted** | Withdrawn from §6.3, §9.1, C4 and §11. Replaced by the measured statement that every tail step *saturates* the bound, plus an [INF] LP-boundary explanation of why saturation is expected |
| C-2b | Basin width never measured; "order 10" contraction factor unsupported | **Accepted** | Both withdrawn entirely (§9.1, §11). No contraction factor is claimed anywhere in V2 |
| C-2c | Contraction is untested; grade C | **Accepted** | Demoted from [CONC] to **[C] Speculative** in §0, §9.1, §10 item 5 and §11. §11 now names the three-arm experiment that would settle it |
| C-2d | Regime B changes `move_lim`, `outer_move`, `alpha` together; cannot isolate | **Partially accepted** | Accepted for the *trajectory-level* comparison, which §2.5 and §9.1 now explicitly label a consistency observation rather than an isolation experiment. **Rejected for the first-step conclusion**: the §2.5 t-profile holds the direction fixed and varies only a scalar, and §2.3's inner-budget transition varies realised step length with no step control present at all. V2 adds the latter as a second clean isolation line (see B.5) |
| C-3a | "Fail-closed necessary for validity" is policy, not theorem | **Accepted** | §0.0 defines four distinct senses of validity; §4.1 adds an explicit [DEC] stating that the successive-iterate test is not an optimality certificate, that no KKT residual was computed, and that an inexact inner step can be legitimate in an outer sequential method; C2 retitled and restated |
| C-3b | The 9 % V4/VR difference is attributed causally to "invalidity" | **Accepted** | §5.3 and C2 rewritten: truncation produces a materially different effective update; both terminal values are non-converged; the campaign cannot say which is nearer the true fixed point |
| C-4a | Multiplicity alternatives not excluded; "not a detection failure and not an N = 2 defect" too strong | **Partially accepted** | **Accepted** for the N = 2 subproblem: its correctness is assumed, not shown, and this is now stated. **Rejected** for detection: the forced-N = 2 probe imposes the cluster rather than detecting it and still fails, which does exclude detection alone. §6.3 rewritten with a six-item list of open alternatives; C4 regraded [A] for non-retention, [B] for step length |
| C-4b | Individual MAC is fragile near multiplicity | **Accepted** | §6.2 adds the basis-rotation caveat; all retention conclusions re-rested on the basis-invariant relative eigengap, with MAC demoted to corroborating. MAC removed from the §11 sub-verdict |
| C-5 | Causal ranking of the residual frequency gap is unidentifiable | **Accepted** | C5 retitled "What can and cannot be said…"; the ranked attribution is withdrawn; four separately graded measured findings replace it; V1's "remaining modelling discrepancy — not supported" narrowed to "no large discrepancy in the reported initial scalar eigenfrequency" |
| C-6 | "Exact period-2 limit cycle" overstates the diagnostics | **Accepted** | §7.3 retitled and rewritten; "persistent, move-limit-saturated period-2-like oscillation" used throughout §0, §5.3, §7, §11. Three reasons given: non-zero lag-2 distance (0.136), a 0.25 threshold rather than 0, and the unavailability of asymptotic language from a finite budget |

### B.2 Internal consistency

| # | Audit point | Decision | Action taken |
|---|---|---|---|
| I-1 | Executive-summary cycle frequencies omit VR | **Accepted** | §0 now lists all four: V4 300.90 / 343.04, VR 328.55 / 371.54 |
| I-2 | −0.982 generalized beyond V4 at 160 × 20 | **Accepted** | §7.3 retitled to name the run; the four-run claim is now carried by the lag-2/lag-1 ratios (0.19, 0.19, 0.14, 0.08), which are computed identically for every run. No new statistic was computed — V2 reruns nothing |
| I-3 | "End in a mechanism" false for V5a | **Accepted** | Verified against `results/V5a_CC_160x20_i2000/`: minimum ω₁ = 12.876 at iteration 130, **final ω₁ = 354.488**. §7.1 now states that `MECHANISM_COLLAPSE` fires on the run minimum, not the terminal state; §11 corrected to "six of the seven end at ω₁ < 10; V5a ends at 354.49" |
| I-4 | "At every budget from 20 to 2000" too literal | **Accepted** | Replaced by "at every tested budget from 20 upward", with the tested budgets enumerated. V2 adds the finding that budgets 1–10 do **not** collapse |
| I-5 | "Every qualitative behaviour transfers" broader than tested | **Accepted** | §7.5 restated as "every behaviour of the four variants run at both meshes"; V1–V3, V5a, V5b named as 160 × 20 only |
| I-6 | Singular-warning count contradicts the aggregate | **Accepted** | Verified: V4 at 240 × 30 records **7**. §5.3 replaced with a complete six-group table over all 19 runs. Conclusion unchanged and sharpened |
| I-7 | "All progress the production solver has ever made" exceeds the campaign | **Accepted** | §4.3 narrowed to every configuration documented by this campaign and the preceding mesh-resolution campaign |
| I-8 | Gate totals are not a scientific scale | **Accepted** | New §8.3 withdraws every summed score, with four reasons; scores removed from §0, §5.4 and §11; the gate matrix is retained and read gate by gate |

### B.3 Major criticisms

| # | Audit point | Decision | Action taken |
|---|---|---|---|
| M-1 | "Exact period-2" inconsistent and not shown for all four runs | **Accepted** | See C-6, I-2 |
| M-2 | Regime B bundles three controls | **Partially accepted** | See C-2d |
| M-3 | Exact-LP result generalized to every outer iteration | **Accepted** | §9.1 withdraws "every element moves to one bound" (corrected to "all but one", with the LP-theoretic reason) and "every outer iteration produces a 0/1 design" (never tested for active-J or N ≥ 2 states). §2.4's [CONC-preview] scoped to the initial state |
| M-4 | Individual MAC used near multiplicity | **Accepted** | See C-4b |
| M-5 | Forced N = 2 implementation not independently verified | **Accepted** | Listed as open alternative 1 in §6.3 and in C4 |
| M-6 | Three schedules do not eliminate continuation ambiguity | **Accepted** | §5.4's "not an artefact of the invented schedule" replaced by "not an artefact of the particular ladder or stage length among the three tested", with trigger, stage-convergence and state-transfer semantics named as unspecified; C1 and C3 qualified likewise |
| M-7 | Classifier conflates transient with terminal mechanism | **Accepted** | New subsection in §7.1 documents the limitation; §5.4 and §11 corrected for V5a |
| M-8 | G5 contains a subjective alternative; gate totals mix categories | **Accepted, and the defect is larger than reported** | On checking `analyze.py:257-260`, the subjective limb **does not exist in the implementation**: G5 is computed purely as `all(N_trial[-10:] >= 2)`. V1's gate description was wrong, not the gate. §8 corrected; no G5 result changes. Totals: see I-8 |
| M-9 | Spectral validity is a final-state gate | **Accepted** | §8 marks G4's scope as "final design only"; "spectrally valid trajectory" removed in favour of "spectrally valid final design" |
| M-10 | Initial-frequency match does not exclude path modelling discrepancies | **Accepted** | See C-5 |
| M-11 | Claims about all historical production progress | **Accepted** | See I-7 |
| M-12 | Reconcile the seven singular warnings | **Accepted** | See I-6 |

### B.4 Minor criticisms

| # | Audit point | Decision | Action taken |
|---|---|---|---|
| m-1 | Correct executive-summary cycle frequencies | **Accepted** | See I-1 |
| m-2 | Qualify "continuation harmful" | **Accepted** | Qualified at every occurrence (§0, §5.3, §5.4, C1, C3, §11) |
| m-3 | "At every tested budget" | **Accepted** | See I-4 |
| m-4 | "Consistent on both tested meshes" | **Accepted** | See C-1b |
| m-5 | Distinguish a near-cluster above `mult_tol` from a solver-detected N = 2 state | **Accepted** | §6.1, §6.5 and §11 distinguish the 160 × 20 near-cluster (g₁₂ = 1.4766e-03, above `mult_tol`, never detected) from the single solver-detected N = 2 state at 240 × 30 |
| m-6 | Explain "structural component" versus raw component counts | **Accepted** | §8 adds the definition: `n_members` counts 8-connected components of area ≥ 0.5 % of the mesh (`MEMBER_MIN_AREA = 0.005`), while `n_comp_8conn` includes speckle. V4 at 240 × 30 has 3 raw components and 1 structural member |
| m-7 | Define prediction-ratio handling when predicted improvement is near zero | **Accepted** | §7.4 records that `pred_ratio` is `NaN` and excluded when \|predicted Δλ₁\| ≤ eps, and that this **never occurred**: all entries are finite in every reported run |
| m-8 | State whether "rejected outer steps" means "would have been rejected" | **Accepted** | §5.2 adds a [DEC] note: actual rejections for gated variants, counterfactual for ungated ones |
| m-9 | Do not use a summed gate score | **Accepted** | See I-8 |
| m-10 | Relabel conclusion previews | **Accepted** | §0.0 defines `[CONC-preview]` as a forward reference to a conclusion established with its grade in §§9–11 |

### B.5 Points on which V2 goes beyond the audit

Three items were not raised by the audit. Two are defects V2 found while
verifying the audit's claims; one is a strengthening the audit's criticism made
possible.

| # | Item | Action taken |
|---|---|---|
| X-1 | **§2.4 conflated two different metrics in one table row.** V1's "fraction at a bound" row reported 0.9997 for the LP and 0.8150 for MMA; these are the *exact-bound* and *within-1 %* fractions respectively. The MMA increment's exact-bound fraction is 0.0000 | Split into two rows with all four values (`results/phase2_CC_160x20/log.txt` lines 33–34). No conclusion affected |
| X-2 | **The mechanism threshold is p-dependent.** It is 5 % of ω₁ at iteration 1, which is evaluated at p = 1 for continuation runs (291.14) and p = 3 for fixed-p runs (145.57), giving different absolute thresholds. V5a (min 12.88, 4.42 %) and V5 (min 15.31, 5.26 %) fall on opposite sides by less than one part in fifty | Disclosed in §7.1 and §5.4; no conclusion rests on the label |
| X-3 | **The inner-budget sweep contains an unexploited transition.** In the paper-literal box, budget 10 returns ‖Δρ‖∞ = 0.165 and *raises* ω₁ to 184.07; budget 20 returns 0.500 and collapses it. Truncation is therefore acting as an undeclared step restriction | Added to §2.3 as [OBS] + [INF] and used in §9.1 as a **second, clean isolation of step length** that involves no move limit, damping or outer bound — which answers M-2/C-2d with evidence rather than by weakening the claim, and makes C6's scope statement stronger |

### B.6 Points rejected

| # | Audit point | Reason for rejection |
|---|---|---|
| R-1 | "The claim that detection is not the obstacle is unsupported" (part of C-4a) | The forced-N = 2 probe **bypasses detection entirely** — the cluster is imposed, not found — and the resulting converged step still fails to retain it (g₁₂ 1.48e-03 → 2.22e-01). Loosening `mult_tol` tenfold buys one clustered iteration, not a regime (§6.1). Detection alone is therefore excluded by direct experiment. V2 accepts the adjacent criticism, that the *N = 2 subproblem implementation* is not thereby vindicated, and states it |
| R-2 | "C7 forbids any reviewer-facing use of the directory" | V1's C7 already enumerated five claim classes rather than issuing a blanket prohibition. V2 nonetheless makes the enumeration explicit as a table separating forbidden claim classes from usable material (verified forward model, initial-state frequencies, topology-class observations with caveat, and this campaign's diagnostics), which meets the audit's substantive concern |

### B.7 What did not change

No experiment was rerun. No numerical value was altered. No run was reclassified.
No gate result changed. No conclusion supported by the evidence was weakened —
in particular, the following stand at full strength, and two of them are stated
more strongly in V2 than in V1:

* the first-step collapse of the full-box formulation, now shown in §2.7 to be
  independent of nine of the eighteen unspecified reconstruction choices;
* the exact-LP diagnosis at the initial state, now with the correct LP-theoretic
  characterisation of its vertex;
* the failure of continuation, under all three tested schedules;
* the 0/300 inner-solve record at the recorded budget;
* the failure of all 19 runs to converge, and of all 19 to retain a cluster;
* the inadmissibility of every terminal frequency in the directory as a
  converged optimum.
