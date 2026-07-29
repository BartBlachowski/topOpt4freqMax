# Faithful Reconstruction of the Du & Olhoff (2007) Optimization Procedure

Clamped–clamped beam benchmark, production mesh **160 × 20** (diagnostic) and
**240 × 30** (primary), square elements, exact mid-height supports, even `nely`.
The 40 × 5 discretization is not used for any conclusion in this report.

Date: 2026-07-29 · MATLAB R2025b · repository commit `461022f`.

**Zero production files were modified.** Evidence: `results/production_sha256.txt`
(SHA-256 of all 25 files in `analysis/OlhoffApproachExact/Matlab/` plus
`tools/Matlab/mmasub.m` and `tools/Matlab/subsolv.m`) and a clean
`git status --porcelain` over `analysis/` and `tools/`. All campaign code is
additive, in this directory. The campaign's outer loop (`recon_solve.m`) is
proven **bit-identical** to the production `topopt_freq_exact.m` on its default
path, in both step-control regimes, by `tests/run_all_tests.m` (T2).

Throughout, **[OBS]** marks a measurement, **[EV]** direct evidence quoted from
the paper or the code, **[INF]** an inferred mechanism, **[DEC]** an
implementation decision made for this campaign, and **[CONC]** a conclusion.
Conclusions appear only in Sections 9–11.

---

## 0. Executive summary

Three findings dominate everything else.

1. **The paper-literal incremental subproblem is a linear program whose optimum
   is a box vertex, and that vertex destroys the structure.** This is not an
   artefact of a truncated inner solve, of MMA conditioning, of the absence of
   continuation, or of mesh resolution. Run to full convergence (312 inner MMA
   iterations), the paper-literal Eq. (25) increment attains 99.98 % of the exact
   LP vertex objective, drives 81.5 % of elements to within 1 % of a box bound,
   and takes ω₁ from 145.57 to **0.095 rad/s** in a single accepted, feasible,
   in-bounds outer step. The linear model predicted 265.6.

2. **Fail-closed inner semantics are necessary for validity but do not prevent
   the collapse.** With the recorded inner budget of 30 iterations, *no inner
   solve in the entire campaign ever converged at outer iteration 1* — the
   production solver has always been accepting non-converged increments. Making
   the solver fail-closed therefore halts it immediately. Raising the budget
   until the inner problem genuinely converges makes the gate pass — and the
   very same step then collapses the structure anyway.

3. **No run converges.** Of 19 runs, 12 halt on an invalid inner solve or end in
   a mechanism, and 2 more report a void-localized mode. The 5 that are feasible
   and spectrally valid all fail design convergence by four to five orders of
   magnitude: four terminate in an **exact period-2 limit cycle** — at ω₁ ≈ 301
   (160 × 20) and ω₁ ≈ 343 (240 × 30), with consecutive design increments
   anti-correlating at **−0.982** and every step saturating the move limit — and
   the fifth wanders without settling. Throughout, the linearized subproblem
   realises a median **0.2 %** of the improvement it predicts and gets the sign
   wrong in 144 of 300 iterations. Across all 19 runs a bimodal state is reported
   at **one** outer iteration of one non-mechanism run — at ω₁ = 90.86, 20 % of
   the published optimum, on a step that cut ω₁ by 74 % — and is abandoned
   immediately.

The missing procedural ingredient is therefore **not** continuation, **not**
multiplicity detection, and **not** the inner solver: it is a rule that bounds
and then *contracts* the design increment as the linearization degrades. Section
9 states this precisely; Section 11 gives the verdicts.

Best gate score in the campaign: **6 / 8** (V4 at 240 × 30), failing exactly the
two gates that define the research question — G5 multiplicity and G6 trajectory
validity. The highest frequency observed, 371.54 rad/s (81.4 % of the published
456.4), comes from the reference configuration in which **no inner solve ever
converged**, and is therefore not admissible.

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
| 4 | **Multiplicity detection** | `detect_multiplicity.m` | upward scan from mode *n*; `|ω_j − ω_n| / max(ω_n, eps) ≤ mult_tol`; `J = n + N` |
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
| **Move limit / trust region on Δρ** | ✘ **(absent from the paper)** | ✔ **required, §9.1** | 0.2 in regime B | med |
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

[OBS] The inner subproblem needs **312** MMA iterations to converge in the
paper-literal box and **181** in the Regime-B box. The recorded budget of 30 is
an order of magnitude short. The collapse is already complete by iteration 20 of
the inner loop and converging further makes it marginally *worse*, not better.

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
| fraction at a bound | 0.9997 | 0.8150 within 1 % |
| **realised ω₁** | **0.0638** | **0.0953** |
| predicted ω₁ | 265.66 | 265.65 |

cos(angle between the LP vertex and the converged MMA increment) = **0.9976**.

[CONC-preview] The converged solution of the paper-literal Eq. (25) *is* the LP
box vertex, to within 0.02 % in objective and 4° in direction. The collapse is a
property of the formulation, not of the solver.

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

[INF] **The paper-literal failure is a step-length failure, not a direction
failure.** Regime B's `move_lim = 0.2` lands the step at t ≈ 0.4 of the
paper-literal increment, inside the validity region, which is exactly why it
produces ω₁ = 177.06 where the paper-literal step produces 0.095.

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
length. The paper-literal step needs a factor of ~3 more than that. Continuation
cannot close the gap.

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

[CONC-preview] **All apparent progress the production solver has ever made in
this benchmark was obtained by accepting inner solutions that had not met their
own declared convergence condition.** That is a validity defect independent of
whether the resulting trajectory looks reasonable.

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

### 5.3 What the matrix isolates

[OBS] **Effect of continuation, holding everything else fixed.**
V0 → V1 (paper-literal, no gate): collapse either way, ω₁ = 0.02 in both, at
both inner budgets. V4 → V5 (Regime-B, gated, converged inner): ω₁ 300.90 →
312.28, but the terminal behaviour *degrades* — V4 settles into a clean
period-2 cycle with a stationary objective (tail CV 1.9e-3), whereas V5 never
becomes stationary (tail CV **0.259**, ω₁ ranging 98.4 – 364.0 over the last 40
iterations) and ends **disconnected** (3 components, 1 extra structural member,
no spanning component, mid-height symmetry 0.606 against V4's 0.999). Continuation
does not produce a better final state; it produces a noisier one.

[OBS] **Effect of valid inner convergence, holding everything else fixed.**
V4 (gated, 300/300 inner solves converged) versus VR (ungated, 0/300 converged),
same step controls, same mesh: ω₁ **300.90 vs 328.55**, i.e. the *invalid*
configuration reports a **9 % higher** frequency and a cleaner topology (1
component, spanning, y-symmetry 1.000, grey fraction 0.570 against V4's 0.772).
Both terminate in a period-2 limit cycle.

[OBS] **Interaction between continuation and step control.** Continuation
changes nothing in the paper-literal regime (V0 ≡ V1, V2 ≈ V3 — all collapse)
and is actively harmful under Regime-B step control (V4 → V5). There is no
combination in which continuation rescues a configuration that would otherwise
fail.

[OBS] **Singular MMA subproblems are a consequence, not a cause.** The RCOND
warning counts are 1015 – 8560 for every collapsed paper-literal run and
**exactly zero** for V4, V5 and VR. The warnings appear only after the design
has already become a near-mechanism.

### 5.4 Continuation-schedule sensitivity

Because the schedule is invented (§3.2, [DEC]), two alternatives were run on the
only viable configuration (Regime-B step controls, fail-closed, converged inner
budget):

* **V5** — p ∈ {1, 1.5, 2, 2.5, 3}, 25 outer iterations per stage (primary);
* **V5a** — p ∈ {1, 2, 3}, 25 iterations per stage (coarser ladder);
* **V5b** — p ∈ {1, 1.5, 2, 2.5, 3}, 15 iterations per stage (shorter stages).

| variant | p ladder | stage length | classification | ω₁ (p = 3) | min g₁₂ | tail CV of ω₁ | lag2/lag1 | final topology |
|---|---|---:|---|---:|---:|---:|---:|---|
| V5 | 1, 1.5, 2, 2.5, 3 | 25 | MAX_ITERATIONS | 312.28 | 2.01e-02 | 2.59e-01 | 1.044 | 3 components, not spanning, y-sym 0.606 |
| V5a | 1, 2, 3 | 25 | **MECHANISM_COLLAPSE** | 354.49 | 4.77e-03 | 2.58e-01 | 1.060 | 10 components, 2 extra members, not spanning, y-sym 0.676 |
| V5b | 1, 1.5, 2, 2.5, 3 | 15 | MAX_ITERATIONS | 296.71 | 5.49e-03 | 2.81e-01 | 1.120 | 9 components, 1 extra member, not spanning, y-sym 0.722; **mode 1 is 99.77 % void-localized** |

[OBS] All continuation schedules behave alike and all behave worse than no
continuation. None reaches N ≥ 2 at any iteration; all are non-stationary at the
end (tail CV of ω₁ ≈ 0.26 against V4's 0.0019); none shows period-2 structure
(lag2/lag1 ≈ 1.05 against V4's 0.19); and all end **disconnected**, against V4's
single spanning component. V5a additionally dips to ω₁ = 12.88, below the 5 %
mechanism threshold, and is classified `MECHANISM_COLLAPSE`; V5b ends with
**99.77 % of mode 1's strain energy in ρ ≤ 0.1 elements** and so fails spectral
validity (G4), exactly as V5 does at 240 × 30. Gate scores: V5 4/8, V5a 2/8,
V5b 2/8, against V4's 5/8 at the same mesh and 6/8 at 240 × 30. The conclusion
that continuation does not help is therefore **not an artefact of the invented
schedule** (§3.2, [DEC]) — all three schedules are worse than no continuation,
and two of the three produce spurious localized modes.

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
and it is left on the very next step, which re-opens the gap by a factor of 236.
MAC₁₁ across that step is **0.016** — mode 1's identity is destroyed.

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

[INF] The cluster is left after one step regardless of the tolerance, and the
trajectory returns to the same ω₁ ≈ 305, g₁₂ ≈ 0.3 limit cycle. **The failure to
retain bimodality is not a multiplicity-detection failure and not a defect in
the N = 2 subproblem: it is a step-length failure.** The smallest step the
procedure can take (move limit 0.2 × α 0.5 = 0.1 per element, saturated at every
iteration) is larger than the basin of the bimodal state, so the iteration
cannot sit inside it.

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
| mechanism | min ω₁ over the run, relative to ω₁ at iteration 1 | < 5 % |
| period-k cycle | median ‖ρₖ − ρₖ₋ₖ‖₂/√nEl over the tail, relative to the lag-1 median | < 0.25 |
| design change not decaying | slope of log₁₀‖ρₖ − ρₖ₋₁‖₂/√nEl per iteration over the tail | \|slope\| < 5e-3 |

Precedence: `INNER_FAILURE` → `MECHANISM_COLLAPSE` → `CONVERGED_{BIMODAL,
UNIMODAL}` → `OUTER_LIMIT_CYCLE` → `OBJECTIVE_STATIONARY_DESIGN_CHATTERING` →
`MAX_ITERATIONS`. A run is labelled `CONVERGED_*` only if the design-change
criterion is met — a stable eigenfrequency is never accepted as convergence.

### 7.2 Measurements required by Q3

For the three long, non-collapsing runs at 160 × 20:

| quantity | V4 (gated, valid inner) | V5 (+ continuation) | VR (reference, invalid inner) |
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

### 7.3 The terminal behaviour of V4 is an exact period-2 cycle

[OBS] From `tail_deltas.csv` (the last ten accepted increments at full spatial
resolution, V4 at 160 × 20):

| quantity | value |
|---|---|
| ‖δₖ‖∞, every one of the last 10 iterations | **0.100000** = `move_lim` × α = 0.2 × 0.5, exactly |
| mean \|δₖ\| over elements | 0.0753 – 0.0766 (i.e. the average element moves 76 % of the maximum allowed amount, every iteration) |
| **corr(δₖ, δₖ₊₁)** | **−0.982** (mean over 9 consecutive pairs) |
| **corr(δₖ, δₖ₊₂)** | **+0.966** (mean over 8 pairs) |
| elements active (\|δ\| > 50 % of max in any tail iteration) | 2573 of 3200 (80 %) |
| spatial extent of the churn | all 20 rows, all 160 columns; activity centroid at 45 % of the half-span |

[CONC-preview] This is not boundary chattering in a localized region and it is
not slow drift. Consecutive increments are near-perfect mirror images of one
another and every second increment repeats: the design oscillates between two
states, over the whole beam, with every step saturating the move limit, while ω₁
sits at 301.0 ± 1.1.

### 7.4 Why the cycle exists

[OBS] Predictive quality of the linearized subproblem, `pred_ratio` = realised
Δλ₁ ÷ predicted Δλ₁:

| run | median | 10th pct | 90th pct | # iterations with pred_ratio < 0 |
|---|---:|---:|---:|---:|
| V4 160×20 | **0.0022** | −0.011 | 0.013 | **144 / 300** |
| V5 160×20 | 0.0514 | — | — | 121 / 300 |
| VR 160×20 | 0.0024 | — | — | 145 / 300 |

[INF] In the terminal regime the subproblem's linear model realises a median of
**0.2 %** of the improvement it predicts, and predicts the *wrong sign* in
roughly half of all iterations. The procedure nevertheless takes a
maximum-length step every iteration, because nothing in the reconstruction —
and nothing in Du & Olhoff — reduces the step when the model stops being
predictive. The result is a maximum-amplitude random walk on a plateau, which
manifests as a period-2 cycle.

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

[OBS] **Every qualitative behaviour transfers.** The paper-literal collapse
(V0), the period-2 limit cycle with a stationary objective (V4, VR) and the
non-stationary wander produced by continuation (V5) each receive the same
classification at both meshes. The lag-2 signature is essentially unchanged
(V4 0.1895 → 0.1901). The single-step forensics transfer too: at 240 × 30 the
inner solve needs 324 iterations paper-literal and 182 Regime-B, the converged
paper-literal step gives ω₁ = 0.1314 and the move-limited step ω₁ = 177.15 —
within 1 % of the 160 × 20 figures (§2.3, §3.3). **Gate G7 passes for every
variant run at both meshes.**

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

| gate | criterion as applied |
|---|---|
| G1 inner validity | no outer step accepted from a non-converged inner problem |
| G2 no mechanism collapse | no near-zero-frequency state, and the final design has a support-to-support spanning solid component |
| G3 feasibility | mean density ≤ volfrac + 1e-4 at every accepted iteration; ρ within bounds throughout |
| G4 spectral validity | reported ω₁ equals the recomputed lowest eigenvalue of the final design, the state is not a mechanism, and mode 1's strain-energy fraction in ρ ≤ 0.1 elements is < 10 % |
| G5 multiplicity | N ≥ 2 at each of the last 10 iterations, or strong evidence bimodality is unreachable |
| G6 trajectory validity | terminal classification is `CONVERGED_*`, i.e. not a hidden limit cycle |
| G7 mesh transfer | the 240 × 30 run of the same variant receives the same classification |
| G8 topological plausibility | one structural component, spanning, \|mid-height symmetry\| > 0.9, grey fraction < 0.75 |

| tag | G1 inner | G2 no-mech | G3 feasible | G4 spectral | G5 multiplicity | G6 trajectory | G7 mesh | G8 topology | passed |
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

### 8.2 Vacuous passes

[OBS] The runs that halt at outer iteration 1 (`INNER_FAILURE`, budget 30) pass
G1, G2 and G3 **vacuously**: no step was taken, so no invalid step was accepted,
no mechanism was created and no constraint was violated. They fail G4–G8 and
their design is the untouched uniform ρ = 0.5 field (grey fraction 1.000).
Gate counts for those runs must be read with that in mind.

---

## 9. Phase 9 — Required scientific conclusions

### 9.1 The mechanism, stated once

[CONC] Du & Olhoff's Eq. (25) is, for a simple eigenvalue, a **linear program
over the full density box** — the paper says so itself (§3.5.3) and this
campaign verifies it numerically (constraints affine to 4e-15; converged MMA
increment within 0.02 % of the exact LP vertex objective and 4° of its
direction, §2.4). The optimum of that LP is a box vertex: every element moves to
one bound. On the CC benchmark that vertex is a direction of genuine ascent —
the linear model is accurate to 1 % for the first 20 % of it (§2.5) — but the
step itself is roughly three times longer than the region in which the model
holds, and applying it destroys the structure.

[INF] Nothing in the paper bounds the step. Three independent lines of evidence
say the computation behind the paper must have bounded it anyway:

1. **The LP structure requires it.** Without a bound, every outer iteration
   produces a 0/1 design and no iteration history is possible.
2. **The paper's own Fig. 4 contradicts an unbounded step.** The published CC
   history rises smoothly from ω₁ ≈ 146 to 456 over roughly 100 iterations. A
   full-box step changes ω₁ by three orders of magnitude in one iteration.
3. **Svanberg's MMA exposes exactly this parameter, and it is at its inactive
   value.** `mmasub.m:69` sets `move = 1.0`, which makes the per-iteration
   restriction `xval ± move·(xmax − xmin)` non-binding. Standard SIMP practice
   uses 0.2. Regime B reintroduces the identical restriction externally through
   `move_lim` / `outer_move`.

[CONC] **The single most important missing procedural detail is a move limit on
Δρ.** It is not "modern globalization" and not an ad hoc stabilizer: it is a
standard, contemporaneous ingredient of SLP/MMA-based topology optimization that
Du & Olhoff do not write down.

[CONC] The **second** missing detail is that this limit must *contract*. A fixed
limit is enough to keep the iteration alive (V4, VR) but not enough to let it
converge or to hold a coalesced state (§6.3, §7.3).

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

*Confidence.* High. *Remaining uncertainty.* Only the reconstructed schedules
were tested (three of them, §5.4). A qualitatively different continuation — for
example one that also relaxes ρ_min, or a non-monotone path — was not tried.

### C2 — Is fail-closed inner MMA necessary?

**Necessary for validity; insufficient for viability.**

*Measurements.* At the recorded inner budget of 30 the inner subproblem
converged **0 out of 300 times** (V0, V1, VR at 160 × 20; 0/120 and 0/150 at
240 × 30). Every design update the production solver has ever applied to this
benchmark came from an increment that had not met its own declared convergence
condition. With the gate enabled at that budget, every variant halts at outer
iteration 1. At a budget where the inner solve does converge, the gate passes
and the paper-literal step is accepted — and still collapses the structure
(V2: 12/13 inner solves converged, ω₁ → 0.02; V3: 15/15 converged, ω₁ → 0.04).

*Interpretation.* Fail-closed semantics convert a silent invalidity into a loud
halt. They do not repair the formulation. Their cost is measurable: the valid
configuration V4 reports ω₁ = 300.90 where the invalid reference VR reports
**328.55** — the previously published-style number is 9 % higher *because* it
rests on truncated inner solves.

*Confidence.* High. *Remaining uncertainty.* `inner_tol = 1e-4` scaled by √nEl
is itself a reconstruction (§12, A3); a different declared tolerance would move
the budget at which the gate starts passing, though not the LP vertex it
converges to.

### C3 — Does continuation enable the transition from N = 1 to N = 2?

**No — it moves the trajectory away from coalescence.**

*Measurements.* Minimum relative eigengap over 300 iterations at 160 × 20:
V4 (no continuation) **1.4766e-03**; V5 (continuation) **2.0090e-02**, an order
of magnitude worse. Iterations with g₁₂ < 1e-2: V4 = 1, V5 = **0**, VR = 4. At
240 × 30 the same ordering holds and is sharper: min g₁₂ is 4.90e-04 for V4 —
the only value in the campaign below `mult_tol` — against 1.35e-02 for V5. Total
iterations with N ≥ 2 at 240 × 30: V4 = 1, V5 = **0**, VR = 0.

*Interpretation.* Continuation moves the trajectory away from the only region
where coalescence occurs. The N = 2 states that *are* reported are of two kinds,
neither of them the paper's optimum: degenerate clusters of near-zero mechanism
modes on collapsed paper-literal designs (V0/V2 at iteration 4, ω₁ = 11.3021,
ω₂ = 11.3024, §6.4), and the single V4 240 × 30 detection at ω₁ = 90.86 created
by a step that cut ω₁ by 74 % (§6.5).

*Confidence.* High.

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

*Interpretation.* Both increments saturate the move limit (‖Δρ‖∞ = 0.2000
exactly). The smallest step the procedure can take is larger than the basin of
the bimodal state. This is not a detection failure and not a defect in the N = 2
subproblem: the off-diagonal generalized gradient is correctly computed and is
in fact as large as the diagonal (‖f₁₂‖/‖f₁₁‖ = 1.0154), which is precisely why
the N = 1 treatment applied there was the wrong model.

*Confidence.* High for the mechanism. *Remaining uncertainty.* Only one
near-coalescent iterate exists in the whole campaign, so this is n = 1; the
retention test extends it to eight steps but from the same starting design.

### C5 — What primarily causes the residual frequency gap?

**Optimizer non-convergence. The failure to reach bimodality is a consequence of
it, not an independent cause.**

*Measurements, ranked.*

1. **Optimizer non-convergence — primary.** The terminal design change is
   8.7e-02 against `outer_tol = 1e-6`: five orders of magnitude. Consecutive
   increments anti-correlate at **−0.982** and every step saturates the move
   limit. The linearized model realises a median **0.2 %** of its predicted
   improvement and has the wrong sign in 144 of 300 iterations.
2. **Failure to reach bimodality — consequent.** C4 shows the state is
   reachable but not holdable at the available step size.
3. **Reconstruction ambiguity — contributory.** Eighteen items are unspecified
   by the paper (§12). Two are demonstrably decisive: the move limit (A13) and
   the inner iteration budget (A4).
4. **Remaining modelling discrepancy — not supported.** The forward model
   reproduces the published initial frequency to within 0.4 %
   (145.57 at 160 × 20 and 145.49 at 240 × 30 against the paper's 146.1), and
   M-orthonormality holds to 2.9e-15. The preceding mesh-resolution campaign
   independently verified the FE model.

*Confidence.* High for the ranking of 1 above 4; medium for the relative weight
of 3, which cannot be bounded without additional information about the original
implementation.

### C6 — Does the paper-literal algorithm become viable after faithful continuation and valid inner convergence, without modern globalization?

**No.** V3 — continuation on the reconstructed p = 1 → 3 path, fail-closed inner
semantics, an inner budget large enough that 15 of 15 inner solves converge, and
paper-literal step control — terminates in `MECHANISM_COLLAPSE` with
ω₁ = 0.04 rad/s. V2, the same without continuation, collapses identically. The
result is mesh-independent (§3.3, 240 × 30) and budget-independent (§2.3).

*Confidence.* High. This is the strongest negative result of the campaign, and
it is the direct answer to the brief's central question about the paper-literal
regime.

### C7 — Which current project conclusions must be withdrawn, retained, or reinterpreted?

**Withdraw.**

* Any statement that an `OlhoffApproachExact` optimized frequency is a
  *converged* optimum. Every non-collapsing run in this campaign terminates in a
  limit cycle five orders of magnitude above its own stopping tolerance. This
  applies to the 160 × 20 and 240 × 30 numbers of the mesh-resolution campaign
  (327.14, 369.43) and to the numbers reported here (300.90, 312.28, 328.55,
  371.54) alike.
* Any statement resting on the recorded `413.869` figure for CC 40 × 5. The
  mesh-resolution campaign already could not reproduce it; the reference
  configuration re-measures at 328.55 over 300 iterations at 160 × 20.
* Any characterisation of the production solver's inner loop as "converged". It
  is not, at the recorded budget, in any run, at any mesh.

**Reinterpret.**

* The mesh-resolution campaign's bounded exception — "a residual ω₁ gap of
  roughly 20 % and the absence of bimodality survive refinement and require a
  separate, optimizer-side explanation" — is now answered. The explanation is
  the non-convergent, move-limit-saturated outer iteration, and the absence of
  bimodality follows from it.
* The recorded diagnosis "N = 1 LP bang-bang divergence" is **confirmed and
  quantified**: the converged inner solve reproduces the exact LP vertex to
  0.02 % in objective and cos = 0.9976 in direction.
* Regime B's `move_lim`/`outer_move` should no longer be described as ad hoc
  non-paper stabilization. They are the reconstruction of a step restriction
  that the LP structure requires, that the paper's own Fig. 4 implies, and that
  Svanberg's `mmasub` exposes as its `move` parameter — currently at its
  inactive value 1.0.
* The `RCOND ~ 1e-20` singular-subproblem storm is a **consequence** of the
  collapsed design, not a cause: 1015 – 8560 warnings in every collapsed run and
  **exactly zero** in V4, V5 and VR.

**Retain.**

* The forward-model verification and the mesh-resolution campaign's H1 findings.
* That the paper-literal regime is numerically non-viable — now with a complete
  mechanism rather than an inference.
* That `analysis/OlhoffApproachExact` must not be used for reviewer-facing
  comparisons, speed-ups, frequency gaps, convergence claims or optimality
  claims. This campaign strengthens that restriction rather than lifting it.

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

4. **No move limit exists in the paper, but one is mathematically required**
   (§9.1). *Not modified: Regime B's recorded 0.2 is used as-is and never
   retuned; no adaptive or contracting variant was introduced.*

5. **The move limit does not contract near the optimum.** With a fixed limit,
   the terminal iteration saturates it at every step (§7), which is what
   prevents retention of the bimodal state (§6.3). The production solver already
   contains a default-disabled `post_coalescence_trust_enabled` switch that
   would address exactly this. *Not modified and not enabled: it is a proposal
   for a separate procedural experiment, not part of this diagnosis.*

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

# `PAPER-LITERAL PROCEDURE REMAINS NUMERICALLY NONVIABLE`

The paper-literal incremental procedure of Du & Olhoff (2007) — Eq. (25) solved
by MMA, with the box bounds of Eq. (25f) as the only restriction on Δρ, and the
Fig. 1 update ρ := ρ + Δρ — destroys the structure on the first outer iteration
and does so at every mesh (160 × 20 and 240 × 30), at every point of the
penalization continuation path (p = 1, 1.5, 2, 2.5, 3), and at every inner
iteration budget from 20 to 2000, including budgets at which the inner
subproblem provably reaches its own declared convergence condition. The
converged increment is the vertex of the linear program the paper itself
identifies in §3.5.3, reproduced here to 0.02 % in objective and cos = 0.9976 in
direction. Neither continuation nor fail-closed inner semantics — the two
candidate missing ingredients the brief nominated — changes this.

`PARTIAL RECONSTRUCTION — BIMODALITY NOT REPRODUCED` was considered and
**rejected** as the primary verdict: it would imply that a defensible
reconstruction produced a valid, converged trajectory that merely failed to
coalesce. No variant in this campaign produced a converged trajectory. Gate G6
fails for all 19 runs.

### Sub-verdicts

| aspect | verdict | key evidence |
|---|---|---|
| **Continuation** | **NOT NECESSARY, NOT SUFFICIENT, AND HARMFUL AS RECONSTRUCTED** | Collapse occurs at every p on the 1 → 3 path with a fully converged inner solve (§3.3); V0 ≡ V1 and V2 ≈ V3; under Regime-B step control continuation degrades the terminal state from a stationary period-2 cycle to a non-stationary wander (tail CV 0.0019 → 0.2592) and the topology from connected/symmetric to disconnected (V4 vs V5), at both meshes and under all three tested schedules |
| **Inner MMA convergence** | **NECESSARY FOR VALIDITY, INSUFFICIENT FOR VIABILITY; NEVER ACHIEVED AT THE RECORDED BUDGET** | 0 / 300 inner solves converged at `inner_max_iter = 30` (0 / 120 and 0 / 150 at 240 × 30); 312–324 iterations are required paper-literal, 181–182 Regime-B; with a sufficient budget the gate passes and the paper-literal step still collapses (V2, V3) |
| **Multiplicity transition** | **NOT REPRODUCED** | Across 19 runs, `N ≥ 2` is reported at **one** outer iteration of **one** non-mechanism run (V4, 240 × 30, iteration 24) — and that state sits at ω₁ = 90.86, 20 % of the published optimum, created by a step that cut ω₁ by 74 % (`pred_ratio` = −1.573). It is abandoned in one step with MAC₁₁ = 0.000. At 160 × 20 the closest approach is g₁₂ = 1.4766e-03, likewise reached on a step with negative realised Δλ₁ and abandoned immediately (gap × 236, MAC₁₁ = 0.016). Forcing N = 2 with the full generalized-gradient array does not retain it. G5 fails for all 19 runs |
| **Design convergence** | **NOT ACHIEVED** | Terminal design change 8.7e-02 against `outer_tol = 1e-6` — five orders of magnitude. Exact period-2 cycle: corr(δₖ, δₖ₊₁) = −0.982, corr(δₖ, δₖ₊₂) = +0.966, ‖δ‖∞ = 0.100000 = move limit × α at every tail iteration, 80 % of elements active, whole beam. Same classification at 240 × 30 |
| **Agreement with the published topology** | **PARTIAL** | V4 at 240 × 30 is in the published Fig. 3c morphological class — single spanning structural component, mid-height symmetry 0.971, mid-span symmetry 0.954, grey fraction 0.604 — and passes G8. So does the reference VR at both meshes. But the design is a point on a limit cycle, not a converged optimum, so the resemblance is not a reproduction |
| **Agreement with the published eigenfrequency** | **NOT ACHIEVED** | Best gated, spectrally valid result: **343.04 rad/s = 75.2 %** of the published 456.4 (V4, 240 × 30). The single highest number in the campaign, 371.54 = 81.4 % (VR, 240 × 30), comes from the configuration in which **no inner solve ever converged** and is therefore not admissible |

### The decisive question

> *Does a mathematically valid, feasible, converged optimization trajectory reach
> the clustered lowest-eigenvalue state described by Du and Olhoff?*

**No.** Of the 19 runs, 12 either halt on an invalid inner solve (5) or end in a
near-zero-frequency mechanism (7). Two more — V5 at 240 × 30 and V5b at
160 × 20, both continuation variants — report an ω₁ whose mode carries over 99 %
of its strain energy in void elements, and fail spectral validity. The remaining
5 are feasible and spectrally valid, and **none of them converges**: four
terminate in an exact period-2 limit cycle (V4 and VR at both meshes) and one in
a non-stationary wander (V5 at 160 × 20), in every case with a terminal design
change four to five orders of magnitude above their own stopping tolerance.
Across all 19 runs a clustered lowest eigenvalue is held for at most one
iteration. The best gate score is **6 / 8**
(V4 at 240 × 30), failing exactly G5 (multiplicity) and G6 (trajectory
validity) — the two gates that define the question.

### What would have to change, stated as a hypothesis for a separate experiment

[INF] Not implemented, not tested, and explicitly outside this diagnostic
campaign. The evidence assembled here points to one specific procedural
addition: **a step restriction on Δρ that contracts when the realised change in
λ₁ falls short of the change the subproblem predicted.** The diagnosis supports
this and nothing else:

* a *fixed* restriction is already necessary and sufficient to keep the
  iteration alive (V4, VR versus V0–V3);
* a fixed restriction is *not* sufficient to converge — every terminal step
  saturates it while realising a median 0.2 % of predicted improvement;
* the bimodal state is reachable but is abandoned by a single saturated step,
  so the required contraction factor is at least the ratio between the move
  limit and the basin width — of order 10 on this benchmark.

The production solver already contains a default-disabled
`post_coalescence_trust_enabled` switch of exactly this shape. It was
deliberately **not** enabled anywhere in this campaign.

## 12. Remaining reconstruction assumptions

Every item below is a choice this campaign had to make because Du & Olhoff
(2007) does not specify it. Each is held fixed across all variants.

| # | Assumption | Value used | Basis | Sensitivity tested? |
|---|---|---|---|---|
| A1 | Finite-element mesh | 160 × 20 (diagnostic), 240 × 30 (primary) | preceding mesh-resolution campaign | yes — both meshes reported |
| A2 | Outer stopping norm and ε | RMS, ε = 1e-4 (paper-literal) / 1e-6 (Regime-B) | on-disk regimes | no — but no run ever approaches either |
| A3 | Inner convergence test | `‖Δρₖ − Δρₖ₋₁‖ < inner_tol·√nEl`, `inner_tol = 1e-4` | on-disk regimes | partially — budget swept 1…1000 |
| A4 | Inner iteration budget | 30 (recorded) and 2000 (converged) | both reported separately | yes — full sweep, §2.3 |
| A5 | Multiplicity tolerance | 1e-3 | on-disk regimes | yes — 1e-4…5e-2 as diagnostics |
| A6 | Number of modes J computed | 4 | on-disk regimes | no |
| A7 | Upper bound on β | β̂ ≤ 1e6 | production `inner_loop_mma.m` | yes — conditioning probe, §2.6 |
| A8 | MMA constants a₀, a, c, d | 1, 0, 1e3, 1 | production `inner_loop_mma.m` | no |
| A9 | MMA asymptote handling across outer steps | reinitialised each outer step | production `inner_loop_mma.m` | no |
| A10 | Continuation stage count and length | 5 stages (1, 1.5, 2, 2.5, 3), 25 outer iterations each | simplest reading of "increasing from 1 to 3" | yes — two alternative schedules, §5.4 |
| A11 | Mass model during continuation | Eq. (4b) held fixed | paper fixes the mass model independently of p | no |
| A12 | Density transfer across continuation stages | transferred, no reinitialisation | simplest defensible choice | no |
| A13 | Move limit / trust region | 0.2 (Regime-B recorded value), fixed | required by the LP structure, §9.1; value not retuned | partially — §2.5 profiles the validity radius |
| A14 | Outer damping α | 0.5 (Regime-B recorded value) | on-disk regime | no |
| A15 | λ̄ = cluster mean in Eq. (25c) | mean | identical for N = 1 | no |
| A16 | ρ_min | 1e-3 | production default | no |
| A17 | Fail-closed tolerances | vol 1e-4, bounds 1e-9 | declared in §4.1 | no |
| A18 | Outer iteration budgets | 300 (160 × 20), 150 (240 × 30), 15 for collapsed paper-literal variants at the converged inner budget | declared in §5.1 | partially |

---

## Appendix — Reproduction

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
