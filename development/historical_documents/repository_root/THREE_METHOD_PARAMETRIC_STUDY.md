# Three-method parametric study of convergence, cost, quality and robustness

**Scope.** Du–Olhoff 2007 (reproduction), Yuksel 2025, and the Proposed method,
each allowed to operate in its own scientifically justified practical regime.
The purpose is to fix a defensible native practical operating profile per
method before the final performance/scaling campaign. The nine-resolution
campaign was **not** run and remains forbidden until the gates in WP24 pass.

---

## Executive summary

Under common evaluation on the calibration mesh, at each method's selected
practical profile:

| Method | Profile | Iterations | Loop time | ms/iter | ω₁ (common raw E1) | Modal state |
|---|---|---:|---:|---:|---:|---|
| Olhoff | `move=0.005` + frozen `H_balanced_v1` detector | 812 | 78.21 s | 96.3 | **169.754** | bimodal, gap 0.30% |
| Yuksel | `move=0.1`, stage tol 0.01/0.01 | 320 (N₁=168 + N₂=152) | 8.32 s | 26.0 | 159.436 | simple, gap 44.8% |
| Proposed | `move=0.2`, `tol=0.01`, OC | 236 | 6.88 s | 29.1 | 157.639 | simple, gap 47.3% |

Five findings dominate everything else:

1. **Olhoff buys a large, evaluator-robust spectral advantage at ~10× the
   cost.** Its ω₁ exceeds Yuksel's by 6.5–7.5% and Proposed's by 6.8–7.7%, and
   the sign of both differences is the same under all six common evaluator /
   representation combinations (E1/E2/E3 × raw/binary).

2. **Yuksel versus Proposed has no evaluator-robust ordering.** Raw evaluators
   put Yuksel 0.83–1.14% ahead; binary evaluators put Proposed 0.66% ahead.
   Per the R3 rule this must be reported as model-dependent, not ranked.

3. **Olhoff's `move` is a bifurcation parameter, not a rate knob.** Whether the
   reproduced bimodal pair survives alternates non-monotonically with `move`
   (0.005 ✓, 0.00625 ✓, 0.0075 ✗, 0.00875 ✗, 0.010 ✓, 0.0125 ✗, 0.015 ✓,
   0.0175 ✗, 0.020 ✗, 0.025 ✗, 0.030 ✗). Move acceleration is therefore not a
   safe route to cheaper Olhoff.

4. **All three native stopping rules are unreliable, in three different ways.**
   Olhoff's `max|Δρ| < tolOuter` never fires at any move. Yuksel's and
   Proposed's `max|Δx| < tol` fire on transient lulls — the design change
   rebounds above the firing tolerance for 16–95% of the next 100 iterations.

5. **Only two of the three profiles survive hold-out validation.** Yuksel and
   Proposed are ROBUST on all of 160×20, 320×40 and 400×50. Both Olhoff
   profiles are FAILED: the primary never fires within 1600 iterations on any
   hold-out mesh, and the fast one fires once — at 807 on 400×50 — on a
   confirmed false positive that loses bimodality afterwards. The Olhoff
   *trajectories* generalize (bimodal and healthy at every mesh); it is online
   stationarity *detection* that does not. The frozen negative result of
   `OLHOFF_NATIVE_CONVERGENCE_DETECTOR.md` therefore stands, and this study
   reproduces its hold-out diagnostics digit for digit.

The intended qualitative hierarchy is only **partially supported**: Olhoff is
confirmed as few-but-expensive iterations with the best spectral quality, but
Proposed is *not* the cheapest method at its authoritative baseline, and it is
not distinguishable from Yuksel on quality. See WP21.

---

## WP0 — Frozen repository and provenance

| item | value |
|---|---|
| branch | `benchmark-methodology-r2` |
| HEAD | `cb6353feb941f12b2aaa927e622649e1ccc926f7` ("pre-R3") |
| working tree | already dirty at start; all unrelated modifications preserved |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1, `maxNumCompThreads(1)` |
| host | Apple arm64, 10 cores (8 performance) |

`Matlab/reproduction2007/SOURCE_SHA256.txt` was verified with the documented
content-preserving migration mapping (`CLAUDE.md → SOURCE_CLAUDE.md`,
`results/* → baseline/*`): **61 of 61 files match, 0 mismatches, 0 missing.**

Numerical sources representing each method — none modified:

| Method | Authoritative implementation |
|---|---|
| Olhoff | `Matlab/reproduction2007/algo/olhoffOpt.m` (+ `innerLoopLP.m`) |
| Yuksel | `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m` |
| Proposed | `analysis/ourApproach/Matlab/topopt_freq.m` |

Everything this study added lives outside those trees:
`analysis/three_method_parametric_study/` and this report.

**One pre-existing audit file was changed, and it is a bug fix, not a retune.**
`analysis/olhoff_native_convergence/nativeConvergenceDetector.m` indexed
`omega(ix-2)` over a 40-wide window while guarding only `k ≥ q + persistence −
1`; for the frozen `H_balanced_v1` configuration (`q == window == 40`) that
raises a subscript error at `k ∈ {59, 60}`. The guard is now
`k ≥ max(q, window+2) + persistence − 1`. Every `k` at which the function
previously returned a value returns the identical value; the change only covers
`k` that previously errored. Confirmation: replaying the frozen detector on the
frozen `move=0.005` trajectory still fires at **iteration 812**, matching
`OLHOFF_NATIVE_CONVERGENCE_DETECTOR.md` exactly.

---

## WP1 — Method identities

### Olhoff (Du–Olhoff 2007 reproduction)

* Route studied: **Eq. (22) LP inner subproblem** (`innerSolver='lp'`,
  `offDiag=false`). The paper-literal MMA route is the labelled baseline of the
  reproduction, not its working configuration — it does not converge once
  N ≥ 2 — and is excluded.
* Outer iteration = one design update; inner iteration = one LP solve within an
  outer step. They are reported separately and never folded into Yuksel's
  *stage* counts.
* Multiplicity: eigenvalue multiplicity `N` detected with `tolMult = 0.05`,
  budget 4; generalized gradients of the multiple eigenvalue.
* Filtering: `filterMode='diag'`, `rminEl = 1.3` **element widths at every
  resolution** (not a fixed physical length). Not swept — WP6.
* Move control: a single move limit applied to every element each outer step.
* Stopping: native `max|Δρ| < tolOuter` (0.001). Practical stopping is the
  frozen `H_balanced_v1` detector.
* Reproduction profile: `repro2007_config('fig3a_best')`, frozen artifact
  `baseline/lp240_rmin1.3.mat`, ω = 170.4709 / 170.8659 / 285.1939.

### Yuksel (2025)

* **Two stages, preserved throughout.** Stage 1 is a compliance loop with **no
  eigensolve**; stage 2 is the inertial-frequency loop.
* Stage 1 stopping: `max|Δx| < stage1_tol` after ≥2 iterations. This is a
  **handoff, not convergence**.
* Stage 2 stopping: the same test with `stage2_tol` — this is the method's
  actual native stop.
* Every result reports N₁, N₂, N_total = N₁+N₂ and T₁, T₂, T_total. Stage-2
  cost is never reported as total method cost.
* Filter `rmin = 2.5` elements, sensitivity, symmetric; η = 0.5, β = 1
  (projection inactive); mass interpolation piecewise at cutoff 0.1
  (`g=x` above, `g=x⁶` at or below); OC update.

### Proposed

* **The R3 uncertainty is resolved, not open.** `PROPOSED_NATIVE_PROFILE_AUDIT.md`
  fixes profile `proposed_manuscript_ss_oc_a0_2026-07-14`, status
  `RESOLVED_PRE_R3_A0_A4_WITH_DISCLOSED_IMPLEMENTATION_LIMITATIONS`. This study
  did **not** re-open it and did not treat OC vs MMA as a tuning level: OC is
  the method, MMA is a different named variant and was not run.
* Objective: compliance under a frozen reference inertial load
  `F(x) = ω₀²·M(x)·Φ₀`, Φ₀ from the fully solid reference, `Φ₀ᵀM₀Φ₀ = 1`,
  refresh interval 0, load sensitivity omitted, no per-iteration renormalisation.
* Filter `rmin = 2.0` elements; OC with Lagrange-multiplier bisection.
* Stopping: native `max|x − x_old| ≤ tol` on the **raw design field**.
* Disclosed limitations carried forward unchanged: implementation design floor
  is 0.0 while the manuscript states 0.001; the filter is declared `symmetric`
  but the effective historical operator is a truncated centroid stencil.

---

## WP2 — Parameter ledger

`results/parameter_classification.csv` classifies all 62 exposed parameters.
**No parameter is class `U`**, so broad sweeps were permitted. Summary:

| Class | Count | Meaning |
|---|---:|---|
| F — fixed scientific/model | 41 | geometry, material, volume fraction, FE model, filter radii, void models, multiplicity formulation |
| N — native algorithmic | 8 | Olhoff `move`, `tolOuter`, practical detector; Yuksel `move`, `stage1_tol`, `stage2_tol`; Proposed `move`, `convergence_tol` |
| R — robustness | 3 | Olhoff `tolMult`, multiplicity budget; Proposed design lower bound |
| D — diagnostic | 10 | iteration budgets, evaluator models, inner bisection controls |

Filter radii are class **F** for all three methods and are held in **element
units at every resolution**, so no method gains or loses length scale under
mesh refinement.

---

## WP3–WP5 — Experimental frame

* Calibration mesh **240×30** for all Stage A work; hold-outs 160×20, 320×40,
  400×50. The forbidden nine-resolution set was not touched.
* Common outcome metrics recorded per run: wall time, optimization-loop time,
  measured cumulative loop time at the stop, total and method-specific
  iteration counts, per-iteration cost, eigensolve time and share (Olhoff
  telemetry), peak RAM, native ω₁₋₃ and eigengaps, multiplicity, volume and
  residual, grayness, gray fraction, four-connectivity of both the raw-0.5 and
  the volume-preserving binary topology, solver-failure counts and stop reason.
* **Evaluator separation is preserved.** Native quantities live only in
  `omega*_native_terminal` columns and are never compared across methods.
  Cross-method quality uses the R3 common evaluators E1 (SIMP linear, floor
  1e-6), E2 (Yuksel piecewise, floor 1e-9), E3 (Olhoff Eq.4, ρ_min 1e-3), each
  in a **raw** and a volume-preserving **binary** representation, on a shared
  Q4 FE model with identical supports and a deterministic eigensolver start.
  E1 is the preregistered primary; E2/E3 are always disclosed alongside.

**Timing is measured, not prorated.** Practical-stop cost is the measured
cumulative optimization-loop time at the stop iteration (Olhoff: summed
per-iteration eigensolve + gradient + inner times; Yuksel/Proposed: the
per-iteration `elapsed_s` history). This matters: Yuksel's two stages have
different per-iteration costs, so a uniform prorate would misprice every
stage-2 stop.

---

## WP6–WP7 — Olhoff Stage A

Eleven move levels were run at 240×30 as **observer-only** trajectories
(native stop suppressed, 1200 outer iterations, per-iteration density snapshots).

### The native outer test is unsalvageable

At **every one of the eleven move levels**, `min_k max|Δρ_k|` equals the move
limit exactly, and `max|Δρ| < 0.001` never fires — not once in 13 200 recorded
iterations. The frozen audit established this at `move = 0.005`; it is now
established across the whole legitimate move range. The LP subproblem always
drives at least one element to the move bound, so the statistic carries no
stopping information at any move. It is retained in the frozen profile only as
the source default and is never used.

### Move changes the attractor, not the rate

| move | ω₁ | ω₂ | ω₃ | gap₁₂ | bimodal (≤1%) | validity |
|---:|---:|---:|---:|---:|:--:|---|
| 0.00500 | 170.472 | 170.883 | 285.203 | 0.298% | ✓ | BIMODAL_VALID |
| 0.00625 | 169.007 | 169.626 | 299.766 | 0.312% | ✓ | BIMODAL_VALID |
| 0.00750 | 170.015 | 173.828 | 302.869 | 2.243% | ✗ | STATIONARY_NOT_BIMODAL |
| 0.00875 | 170.400 | 173.480 | 296.477 | 1.972% | ✗ | STATIONARY_NOT_BIMODAL |
| 0.01000 | 169.392 | 169.801 | 283.202 | 0.381% | ✓ | BIMODAL_VALID |
| 0.01250 | 170.967 | 179.067 | 283.804 | 4.830% | ✗ | STATIONARY_NOT_BIMODAL |
| 0.01500 | 169.719 | 170.691 | 261.202 | 0.553% | ✓ | BIMODAL_VALID |
| 0.01750 | 170.453 | 179.119 | 303.735 | 5.069% | ✗ | STATIONARY_NOT_BIMODAL |
| 0.02000 | 170.745 | 175.066 | 301.872 | 2.812% | ✗ | STATIONARY_NOT_BIMODAL |
| 0.02500 | 170.277 | 172.567 | 294.293 | 1.633% | ✗ | STATIONARY_NOT_BIMODAL |
| 0.03000 | 170.947 | 172.732 | 283.297 | 1.386% | ✗ | STATIONARY_NOT_BIMODAL |

All eleven runs are healthy: zero solver failures, relative volume residual
below 1.6e-15 (the evaluator's absolute residual is below 2.5e-10), a single
support-to-support connected component. The failure is
**modal**, not numerical, and it alternates: 0.0075 and 0.00875 lose the
bimodal pair while both 0.00625 and 0.010 keep it. `move` is a bifurcation
parameter of the reproduced problem.

Two cross-checks confirm the trajectories are correct, not broken:
`move = 0.02` reproduces the frozen `fig4_history` artifact
(ω = 170.744886 / 175.065616 / 301.871712) to all printed digits, and
`move = 0.005` reproduces `fig3a_best`.

**Reproduction quality also degrades away from `move = 0.005`.** Against the
paper's ω₃ = 284.9, `move = 0.005` gives +0.11%, `move = 0.010` gives −0.60%,
`move = 0.00625` gives +5.2% and `move = 0.015` gives −8.3%. Larger moves do
not merely converge differently; they converge to different local optima.

### At move ≥ 0.02 the trajectory is *exactly* period-two

For `move ∈ {0.020, 0.025, 0.030}` the lag-2 density RMS over the last 200
iterations is **identically 0.0** and the lag-2 objective recurrence is ~6e-13
— a bit-periodic two-cycle. These are the most stationary trajectories in the
study, and they are also the ones that lost bimodality. Stationarity alone is
therefore not evidence of successful Olhoff convergence, which is exactly what
the WP7 taxonomy exists to prevent.

### Detector families (observer-only)

Three families (objective-only, design-only, hybrid) × two phase lags (1, 2) ×
three tolerance levels = 198 candidate evaluations. Every candidate was
evaluated future-blind at each `k` and then labelled retrospectively at
horizons 50/100/200 with the frozen audit's label definition.

| classification | count |
|---|---:|
| NEVER_FIRES | 154 |
| TRUE_ON_TRAJECTORY | 18 |
| FALSE_POSITIVE_IMMEDIATE | 12 |
| TRUE_BUT_HORIZON_LIMITED | 8 |
| FALSE_POSITIVE_DELAYED | 6 |

* **Phase pairing is essential.** Lag-1 families produce 1 true fire out of 99;
  lag-2 families produce 17 of 99. The LP trajectory alternates, and a
  consecutive-iteration comparison cannot see through it.
* **Delayed instability is real and is caught only by long horizons.** Six
  configurations pass a 50-step look-ahead and fail at 100 and 200 — e.g. at
  `move = 0.005` the mid-tolerance objective rule fires at 446, and at
  `move = 0.010` the loose lag-1 rule fires at 434. A 50-iteration check would
  have accepted both.
* Eight candidates are `TRUE_BUT_HORIZON_LIMITED`: they fired late enough that
  the 1200-iteration window could not close the 100/200-step horizon. These are
  **censored, not passing and not false positives**, and they are excluded from
  selection rather than credited.

### The primary Olhoff detector was not tuned in this study

The `hybrid / lag 2 / strict / persistence 20` cell of the grid **is**
`H_balanced_v1` from `native_convergence_config.json`, coefficient for
coefficient (block drift 1e-4, phase recurrence 1e-4, ρ-phase RMS 1.25e-3,
topology turnover 7e-4, modal window 40, gap ≤1%, N=2, volume 1e-8, health
required). It is replayed verbatim; only `move` changes. Its behaviour:

| move | fire | H50 | H100 | H200 | terminal ω₁ loss | gap at fire |
|---:|---:|---|---|---|---:|---:|
| 0.00500 | **812** | PASS | PASS | PASS | 0.0008% | 0.296% |
| 0.00625 | **320** | PASS | PASS | PASS | −0.0002% | 0.320% |
| 0.01000 | **682** | PASS | PASS | PASS | −0.0082% | 0.306% |
| 0.01500 | 1105 | PASS | CENSORED | CENSORED | −0.0000% | 0.572% |
| all others | never | — | — | — | — | — |

The detector correctly refuses to fire on all seven non-bimodal trajectories:
its `N=2` and `gap ≤ 1%` guards block them, and the raw candidate condition is
satisfied at 0% of eligible iterations there.

---

## WP8 — Yuksel Stage A

Nine preregistered one-factor configurations plus two boundary-mapping runs.

**The first pass was censored and is retained as such.** It capped both stages
at 300 iterations, below the R3 authoritative budgets (10 000 per stage); at
that cap stage 1 itself hit the cap for `stage1_tol = 0.005`. Those eleven rows
stay in `parametric_run_ledger.csv` with `pass = stage_a` and are excluded from
selection by reason `CENSORED_FIRST_PASS`. The authoritative pass
(`stage_a_v2`) uses a 1000-iteration per-stage budget with extension enabled,
leaving 400–900 iterations of look-ahead past every native stop.

| run | move | s1 tol | s2 tol | N₁ | N₂ to stop | N_total | T_total | ω₁ (E1 raw) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.2 | 0.01 | 0.01 | 149 | 193 | 342 | 9.41 s | 159.408 |
| move_010 | 0.1 | 0.01 | 0.01 | 168 | 152 | **320** | **8.32 s** | **159.436** |
| move_030 | 0.3 | 0.01 | 0.01 | 111 | 130 | 241 | 6.08 s | 159.408 |
| tol_both_0005 | 0.2 | 0.005 | 0.005 | 339 | 271 | 610 | 15.26 s | 159.656 |
| tol_both_0020 | 0.2 | 0.02 | 0.02 | 92 | 105 | 197 | 4.97 s | 159.488 |
| tol_both_0050 | 0.2 | 0.05 | 0.05 | 59 | 38 | 97 | 2.48 s | 159.792 |
| tol_s1_0005 | 0.2 | 0.005 | 0.01 | 339 | 253 | 592 | 14.62 s | 159.658 |
| tol_s1_0020 | 0.2 | 0.02 | 0.01 | 92 | 172 | 264 | 8.84 s | 159.381 |
| tol_s2_0005 | 0.2 | 0.01 | 0.005 | 149 | 381 | 530 | 13.73 s | 159.356 |
| tol_s2_0020 | 0.2 | 0.01 | 0.02 | 149 | 113 | 262 | 6.46 s | 159.752 |
| tol_s2_0050 | 0.2 | 0.01 | 0.05 | 149 | 38 | 187 | 4.82 s | 160.222 |

Both stage tolerances materially control cost, and each acts on the stage it
names: `stage1_tol` moves N₁ over 59–339 while leaving N₂ broadly alone, and
`stage2_tol` moves N₂ over 38–381 at fixed N₁ = 149. They are the method's real
cost controls. `move` is **not** monotone — 0.3 → 241, 0.2 → 342, 0.1 → 320
total iterations — and its whole range spans less than one tolerance step.

Yuksel's 200-iteration Dynamic-Code statement was **not** used as a target for
Yuksel's own method; the comparison in WP21 is retrospective only.

---

## WP9 — Proposed Stage A

Ten configurations on the authoritative OC profile. The first pass was likewise
censored (300-iteration cap: `tol = 0.001` and `0.0005` never fired, `0.002`
fired at 298), and is retained with reason `CENSORED_FIRST_PASS`.

| run | move | tol | N to native stop | T | ω₁ (E1 raw) |
|---|---:|---:|---:|---:|---:|
| tol_00005 | 0.2 | 0.0005 | 695 | 20.09 s | 157.619 |
| base | 0.2 | 0.001 | 627 | 18.58 s | 157.617 |
| tol_00020 | 0.2 | 0.002 | 298 | 8.57 s | 157.586 |
| tol_00050 | 0.2 | 0.005 | 272 | 7.85 s | 157.599 |
| tol_00100 | 0.2 | 0.01 | **236** | **6.88 s** | **157.639** |
| tol_00200 | 0.2 | 0.02 | 46 | 1.43 s | 157.074 |
| tol_00500 | 0.2 | 0.05 | 34 | 0.98 s | 157.116 |
| move_010 | 0.1 | 0.001 | 629 | 18.48 s | 157.409 |
| move_015 | 0.15 | 0.001 | 837 | 24.46 s | 157.416 |
| move_030 | 0.3 | 0.001 | 956 | 28.24 s | 157.412 |

Two structural results:

* **The convergence tolerance is the only material cost control**, spanning
  34–695 iterations.
* **The move limit is not an accelerator.** At fixed `tol = 0.001` the native
  stop falls at 629 / 837 / 627 / 956 iterations for `move` = 0.1 / 0.15 / 0.2 /
  0.3 — non-monotone, and the largest move is the *slowest* by 52%. A larger OC
  step overshoots and takes longer to settle the `max|Δx|` test rather than
  reaching it sooner.

Proposed is not required to produce Olhoff-style bimodality — that is not part
of its objective. Its spectral compromise is quantified in WP12/WP13 instead:
its selected profile's ω₁ is 6.8–7.7% below Olhoff's under every common
evaluator.

---

## WP11 — False-convergence validation

**Nothing was terminated when a detector fired.** Every Stage A trajectory was
run past any hypothetical stop: Olhoff to 1200 outer iterations with the native
stop suppressed, Yuksel and Proposed to their safety budgets with extension
mode on. Look-ahead is measured, not assumed.

**Olhoff.** Horizons 50/100/200 with the frozen label. Results in
`false_convergence_events.csv` (132 rows: 74 PASS, 48 FAIL, 10 CENSORED). Six
delayed false positives are documented above.

**Yuksel and Proposed — the native rules are not stationarity detectors.**
After the native stop, the design-change statistic rebounds above the very
tolerance that fired it, for this fraction of the next 100 iterations:

| method | best case | worst case |
|---|---|---|
| Yuksel | 0.25 (`tol_both_0020`) | **0.95** (`move_030`) |
| Proposed | 0.16 (`move_030`) | **0.94** (`tol_00005`) |

The two Proposed runs showing zero rebound (`tol = 0.02`, `0.05`) are artifacts
of a tolerance larger than the entire post-stop range — they are the *most*
premature stops, not the most settled. An objective-progress test cannot see
any of this: every Yuksel and Proposed native stop sits at ≥99.98% of its run's
total native-objective improvement, because these objectives plateau long
before their designs do.

**The declared look-ahead is therefore outcome-based**, and is applied to all
methods: a stop is a false convergence if continuing to the safety budget moves
the reported common-evaluator ω₁ by more than 0.1% — the same objective
tolerance the frozen Olhoff label uses — or breaks connectivity. This removed
seven configurations:

| run | ω₁ drift | verdict |
|---|---:|---|
| `yuksel_tol_s2_0050` | 0.565% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `yuksel_tol_both_0050` | 0.283% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `yuksel_tol_s2_0020` | 0.269% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `yuksel_tol_s1_0005` | 0.181% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `yuksel_tol_both_0005` | 0.179% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `yuksel_tol_both_0020` | 0.101% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `proposed_tol_00200` | 0.308% | FALSE_CONVERGENCE_OMEGA1_DRIFT |
| `proposed_tol_00500` | 0.280% | FALSE_CONVERGENCE_OMEGA1_DRIFT |

The design-rebound fraction is reported for every run but is deliberately
**not** an eligibility gate: it characterises the native rules (which are not
stationarity detectors) rather than invalidating their outputs.

---

## WP12 — Cost to reach quality levels

**Within-method** progress is measured on each method's *own* native objective
as the fraction of that run's eventual total improvement, phase-smoothed over
two iterations, and requires the level to hold for the remainder of the run.

| Method | 95% | 97.5% | 99% | 99.5% | practical stop |
|---|---:|---:|---:|---:|---:|
| Olhoff (`move=0.005`) | 169 it / 19.9 s | 200 it / 22.7 s | 266 it / 28.7 s | 338 it / 35.3 s | 812 it / 78.2 s |
| Yuksel (`move=0.1`) | 170 it / 4.06 s | 170 it / 4.06 s | 170 it / 4.06 s | 170 it / 4.06 s | 320 it / 8.32 s |
| Proposed (`tol=0.01`) | 6 it / 0.24 s | 8 it / 0.29 s | 17 it / 0.55 s | 19 it / 0.60 s | 236 it / 6.88 s |

Every method spends most of its practical budget after its own objective has
reached 99.5% of its improvement — Olhoff 55%, Yuksel 51%, Proposed 91% of the
loop time. For Yuksel and Proposed this figure is close to meaningless as a
quality statement: their inertial-compliance objectives plateau within tens of
iterations while the design keeps moving, which is the same phenomenon the
rebound analysis found. **Within-method efficiency and cross-method quality
must not be conflated**, which is why the next section is separate.

**Cross-method absolute quality** at the selected profiles, common evaluators:

| | raw E1 | raw E2 | raw E3 | binary E1 | binary E2 | binary E3 |
|---|---:|---:|---:|---:|---:|---:|
| Olhoff | 169.754 | 170.156 | 170.156 | 172.442 | 172.442 | 172.442 |
| Yuksel | 159.436 | 159.492 | 159.492 | 160.399 | 160.399 | 160.399 |
| Proposed | 157.639 | 158.183 | 158.183 | 161.467 | 161.467 | 161.467 |

| pair | raw E1 | raw E2 | raw E3 | bin E1 | bin E2 | bin E3 | sign-consistent |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Olhoff − Yuksel | +6.47% | +6.69% | +6.69% | +7.51% | +7.51% | +7.51% | **yes** |
| Olhoff − Proposed | +7.69% | +7.57% | +7.57% | +6.80% | +6.80% | +6.80% | **yes** |
| Yuksel − Proposed | +1.14% | +0.83% | +0.83% | −0.66% | −0.66% | −0.66% | **no** |

Per the R3 evaluator-robust ordering rule, the Yuksel/Proposed comparison is
reported as **model-dependent, with no single raw ranking**. The Olhoff
advantage over both is robust to every evaluator and both representations.

---

## WP13 — Pareto fronts

Cost is the measured optimization-loop time to the practical stop; quality is
common raw E1 ω₁ at that stop (E2/E3 disclosed above). Non-dominated within
method:

| Method | Candidate | Cost | Quality | Quality loss | Non-dominated | Status |
|---|---|---:|---:|---:|:--:|---|
| Olhoff | `move=0.005` | 78.21 s | 169.754 | 0.000% | yes | **SELECTED_PRIMARY** |
| Olhoff | `move=0.00625` | 32.04 s | 168.568 | 0.699% | yes | CANDIDATE |
| Olhoff | `move=0.010` | 65.25 s | 168.817 | 0.552% | yes | CANDIDATE |
| Yuksel | `move=0.1` | 8.32 s | 159.436 | 0.000% | yes | **SELECTED_PRIMARY** |
| Yuksel | `move=0.3` | 6.08 s | 159.408 | 0.018% | yes | CANDIDATE |
| Proposed | `tol=0.01` | 6.88 s | 157.639 | 0.000% | yes | **SELECTED_PRIMARY** |

**The quality axis does not resolve for Yuksel or Proposed.** Across all
eligible configurations the total spread is 0.05% for Yuksel and 0.15% for
Proposed. Min–max scaling inflates those into a full unit of normalised
quality, so their "knee" is in substance cost-driven with quality noise. Any
eligible non-dominated point of those two methods is scientifically
equivalent, and this is stated rather than hidden behind the arithmetic.

Olhoff's front is genuinely two-dimensional: 0.7% of ω₁ buys a 2.4× cost
reduction between `move = 0.005` and `move = 0.00625`.

---

## WP15 — Selection rule (documented before Stage B)

Taken verbatim from `study_preregistration.json`, written before any Stage A
result was inspected. Neither `argmin T` nor `argmax ω` is used.

1. **Eligibility** — finite successful optimizer and eigensolver; relative
   volume residual ≤ 1e-3; support-to-support connected in both the raw-0.5 and
   the binary topology; no false convergence under the declared look-ahead;
   Olhoff additionally requires persistent N = 2 and relative gap₁₂ ≤ 0.01.
2. **Dominance** — within a method, no greater cost, no worse quality, no worse
   robustness, with at least one strict improvement.
3. **Knee** — the eligible non-dominated point of smallest normalised Euclidean
   distance to the cost/quality utopia point after within-method min–max
   scaling.
4. **Tie-break** — if distances differ by ≤ 0.02, take the simpler and more
   conservative profile.
5. **No hold-out retuning.**

The tie-break was exercised exactly once: at Olhoff, `move = 0.005` and
`move = 0.00625` both scored distance 1.0000 (they are the two extremes of a
three-point front), and the rule selected the conservative established
reproduction move. The rule was not adjusted to produce that outcome — it is
what a two-endpoint tie produces.

---

## WP16 — Frozen profiles

`results/profile_freeze_manifest.json`, config hash `02075494076df3de`, frozen
**before any hold-out mesh was run**. Six profiles: one primary per method plus
one secondary per method (Olhoff's fast candidate, and the published /
authoritative reference configurations of Yuksel and Proposed, which this study
did not select but which must be validated alongside so WP20 stays separable).

| key | hash | method | role |
|---|---|---|---|
| `olhoff_practical` | `d857f4ba59d75a26` | Olhoff | primary |
| `olhoff_fast` | `2a6fae929cb055bb` | Olhoff | secondary (fast candidate) |
| `yuksel_practical` | `74af830703eebfc8` | Yuksel | primary |
| `yuksel_published` | `e347b7ca8aa1b774` | Yuksel | secondary (published reference) |
| `proposed_practical` | `fab7523a8e209f3c` | Proposed | primary |
| `proposed_authoritative` | `62bda4dee5cc93b8` | Proposed | secondary (R3 reference) |

After this point no parameter was retuned. Full parameter listings are in
Table E below.

---

## WP19 — Prospective stopping validation

Run on the calibration mesh with stopping **actually enabled** — the Olhoff
detector in `detector_active_stop` mode, Yuksel and Proposed with extension
disabled — and compared against the observer-mode predictions.

| Profile | k predicted | k actual | match |
|---|---:|---:|:--:|
| `olhoff_practical` | 812 | 812 | ✓ |
| `olhoff_fast` | 320 | 320 | ✓ |
| `yuksel_practical` | 320 | 320 | ✓ |
| `yuksel_published` | 342 | 342 | ✓ |
| `proposed_practical` | 236 | 236 | ✓ |
| `proposed_authoritative` | 627 | 627 | ✓ |

**6 of 6 exact.** The prospective `olhoff_practical` run terminates at
ω = 170.4710 / 170.8666 / 285.2271, matching the frozen `fig3a_best` artifact
(170.4709 / 170.8659 / 285.1939). Offline prediction and live termination agree
for every frozen profile.

---

## WP17 — Hold-out cross-resolution validation

Frozen profiles replayed at 160×20, 320×40 and 400×50. **No parameter was
retuned.** Olhoff budget 1600 outer iterations (the frozen audit's budget, so
the two studies are directly comparable); Yuksel 1000 per stage; Proposed 2000.

| Method | Profile | Mesh | Practical stop | Observed | Loop time | ms/iter | ω₁ at stop (E1 raw) | Terminal gap₁₂ | Connected | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|---|
| Olhoff | practical | 160×20 | **never** | 1600 | 64.0 s | 40.0 | — | 0.312% | ✓ | CAP_HIT |
| Olhoff | practical | 320×40 | **never** | 1600 | 274.1 s | 171.3 | — | 0.166% | ✓ | CAP_HIT |
| Olhoff | practical | 400×50 | **never** | 1600 | 454.8 s | 284.3 | — | 0.259% | ✓ | CAP_HIT |
| Olhoff | fast | 160×20 | **never** | 1600 | 61.4 s | 38.4 | — | 0.398% | ✓ | CAP_HIT |
| Olhoff | fast | 320×40 | **never** | 1600 | 273.5 s | 170.9 | — | 0.187% | ✓ | CAP_HIT |
| Olhoff | fast | 400×50 | 807 | 1600 | 222.9 s | 259.3 | 171.817 | 2.011% | ✓ | **FALSE CONVERGENCE** |
| Yuksel | practical | 160×20 | 244 | 1121 | 3.0 s | 14.6 | 157.167 | 46.4% | ✓ | CONVERGED_NATIVE |
| Yuksel | practical | 320×40 | 572 | 1252 | 26.1 s | 47.3 | 160.690 | 38.4% | ✓ | CONVERGED_NATIVE |
| Yuksel | practical | 400×50 | 732 | 1315 | 53.5 s | 74.7 | 159.968 | 101.0% | ✓ | CONVERGED_NATIVE |
| Yuksel | published | 160×20 | 249 | 1126 | 2.9 s | 12.2 | 157.112 | 43.9% | ✓ | CONVERGED_NATIVE |
| Yuksel | published | 320×40 | 740 | 1269 | 34.2 s | 49.1 | 160.389 | 37.1% | ✓ | CONVERGED_NATIVE |
| Yuksel | published | 400×50 | 727 | 1300 | 52.5 s | 74.3 | 160.000 | 98.1% | ✓ | CONVERGED_NATIVE |
| Proposed | practical | 160×20 | 107 | 2000 | 1.6 s | 14.0 | 153.675 | 1.70% | ✓ | CONVERGED_NATIVE |
| Proposed | practical | 320×40 | 207 | 2000 | 11.2 s | 53.9 | 158.763 | 45.8% | ✓ | CONVERGED_NATIVE |
| Proposed | practical | 400×50 | 182 | 2000 | 16.4 s | 90.1 | 159.519 | 42.2% | ✓ | CONVERGED_NATIVE |
| Proposed | authoritative | 160×20 | 936 | 2000 | 12.3 s | 13.1 | 153.121 | 1.70% | ✓ | CONVERGED_NATIVE |
| Proposed | authoritative | 320×40 | 337 | 2000 | 18.2 s | 54.3 | 158.679 | 45.8% | ✓ | CONVERGED_NATIVE |
| Proposed | authoritative | 400×50 | **never** | 2000 | 181.7 s | 90.8 | — | 42.2% | ✓ | CAP_HIT |

Yuksel stage decomposition is preserved throughout — N₁ / N₂-to-stop:

| profile | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `yuksel_practical` | 121 / 123 | 252 / 320 | 315 / 417 |
| `yuksel_published` | 126 / 123 | 269 / 471 | 300 / 427 |

Stage 1 grows from 38% to 43% of the total as the mesh refines, and it does no
eigensolve, which is a material part of why Yuksel's per-iteration cost stays
below Olhoff's.

### The Olhoff practical stop does not generalize — and the failure is the detector's, not the method's

This is the study's most important negative result, and it must not be blurred.

**The Olhoff trajectories themselves generalize excellently.** At all three
hold-out meshes the terminal state is healthy, connected, bimodal (terminal
gap₁₂ = 0.312% / 0.166% / 0.259%) with persistent N = 2 and zero solver
failures. The reproduction is resolution-robust.

**What fails is online stationarity detection.** The frozen `H_balanced_v1`
detector never fires within 1600 iterations at any hold-out mesh, because those
trajectories remain dynamically active late: the last modal event occurs at
iteration 1591 / 1544 / 1505, and the eigengap reopens above 1% within the
final 200 iterations (max 1.079% / 1.158% / 1.105%). These numbers reproduce
`OLHOFF_NATIVE_CONVERGENCE_DETECTOR.md` digit for digit, which independently
confirms that this replay is faithful and that the frozen negative result is
reproducible.

**The one hold-out fire is a confirmed false positive.** `olhoff_fast` fires at
iteration 807 on 400×50 with a 0.593% eigengap, and then:

| post-fire evidence | value | label tolerance |
|---|---:|---:|
| eigengap reopens (max after fire, at iteration 1525) | 2.083% | ≤ 1% |
| modal events after the fire (first at 1263) | 38 | 0 |
| ω₁ block-mean drift, fire → terminal | 0.225% | ≤ 0.1% |
| binary topology turnover, fire → terminal | 2.805% | ≤ 0.5% |

It fails the look-ahead label on four independent terms. Had the fast profile
been allowed to terminate prospectively at 400×50 it would have returned a
design that was still 2.8% away from its own continued topology and no longer
bimodal.

**Conclusion for WP6's central question.** Changing `move` does produce
substantially earlier genuine practical stationarity *on the calibration mesh*
— 320 iterations instead of 812, a 2.5× reduction, with a validated look-ahead
and retained bimodality. It does **not** survive cross-resolution validation.
Move acceleration is therefore not a route out of the frozen negative result,
and the frozen result stands.

---

## WP18 — Robustness classification

Criteria from `study_preregistration.json`. A profile is `ROBUST` when all
hold-out meshes are valid with no solver failure, false convergence, cap hit or
connectivity failure; `RESOLUTION_SENSITIVE` when exactly one mesh loses
convergence, modal or connectivity validity; `FAILED` when two or more do.

| Profile | Method | Valid meshes | Iterations (160/320/400) | ms/iter | Class |
|---|---|:--:|---|---|---|
| `olhoff_practical` | Olhoff | 0 / 3 | cap / cap / cap | 40 / 171 / 284 | **FAILED** |
| `olhoff_fast` | Olhoff | 0 / 3 | cap / cap / 807 (false) | 38 / 171 / 259 | **FAILED** |
| `yuksel_practical` | Yuksel | 3 / 3 | 244 / 572 / 732 | 15 / 47 / 75 | **ROBUST** |
| `yuksel_published` | Yuksel | 3 / 3 | 249 / 740 / 727 | 12 / 49 / 74 | **ROBUST** |
| `proposed_practical` | Proposed | 3 / 3 | 107 / 207 / 182 | 14 / 54 / 90 | **ROBUST** |
| `proposed_authoritative` | Proposed | 2 / 3 | 936 / 337 / cap | 13 / 54 / 91 | **RESOLUTION_SENSITIVE** |

Two consequences follow, and neither was chosen — both fall out of the frozen
profiles meeting the preregistered criteria.

* **No Olhoff practical profile survives.** `FAILED` here classifies the
  *stopping profile*, not the method: the Olhoff trajectories are robust and
  bimodal at every mesh. What has no validated cross-resolution realisation is
  the practical stop.
* **The R3 authoritative Proposed profile (`tol = 0.001`) is resolution
  sensitive**: it does not reach its own native stop within its 2000-iteration
  safety budget at 400×50. The looser `tol = 0.01` practical profile does, at
  all three meshes. This is a finding about the R3 protocol's default, and it
  is reported here rather than quietly corrected.

---

## WP20 — Separate named experiments (kept distinct)

| Experiment | Definition | Where it lives | Used for the comparison? |
|---|---|---|---|
| Du–Olhoff reproduction | `fig3a_best`, move 0.005, rmin 1.3 el, 1600 outer | `Matlab/reproduction2007/baseline/lp240_rmin1.3.mat`; reproduced here at ω = 170.4710 / 170.8666 / 285.2271 | reference only |
| Yuksel Table-1 interpretation | Dynamic Code fixed at 200 outer iterations (Yuksel §6.2) | not run in this study | no — retrospective comparison only (WP21) |
| Yuksel Dynamic-Code configuration reproduction | Yuksel's stated Dynamic-Code move/filter/multiplicity settings | not performed | no |
| Yuksel published SS case | move 0.2, rmin 2.5 el, tol 0.01/0.01 (`benchmark_protocol_r3.json`) | `yuksel_published` frozen profile | validated alongside, **not** selected by this study |
| Proposed authoritative R3 profile | OC, move 0.2, tol 0.001, rmin 2.0 el | `proposed_authoritative` frozen profile | validated alongside, **not** selected by this study |
| **Native practical comparison** | the profiles this study selected | `profile_freeze_manifest.json` | **yes** |

None of these was substituted for another at any point.

---

## WP21 — Does the intended qualitative hierarchy hold?

Assessed only after the profiles were frozen and the hold-outs were run. No
profile was altered after seeing this.

| Clause | Verdict | Evidence |
|---|---|---|
| Olhoff: relatively few but expensive iterations | **PARTIALLY_SUPPORTED** | Iterations are expensive — 96 ms/iter at 240×30 vs 26–29 ms for the others, rising to 284 ms vs 75–90 ms at 400×50. But they are *not* few: 812 at 240×30, and no validated stop at all on any hold-out mesh. |
| Olhoff: high-quality multiplicity-aware bimodal optimum | **SUPPORTED** | ω₁ is 6.5–7.5% above Yuksel and 6.8–7.7% above Proposed, sign-consistent across all six evaluator/representation combinations. Bimodal (gap 0.17–0.31%) at all four meshes. |
| Yuksel: intermediate strategy and computational cost | **PARTIALLY_SUPPORTED** | Intermediate in *iteration count* on the one mesh where all three have a validated stop (240×30: Olhoff 812 > Yuksel 320 > Proposed 236), and cheapest per iteration at 3 of 4 meshes. But its total *time* is not intermediate: it exceeds Proposed at all four meshes (3.0/8.3/26.1/53.5 s vs 1.6/6.9/11.2/16.4 s) while sitting an order of magnitude below Olhoff. It shares Proposed's tier rather than lying between the tiers. |
| Proposed: low computational cost | **PARTIALLY_SUPPORTED** | At its selected practical profile it is the cheapest method at **all four** meshes, by 1.2× at 240×30 widening to 3.3× at 400×50. But at its own R3 authoritative baseline (`tol = 0.001`) it is *twice* Yuksel's cost at 240×30 (18.6 s vs 9.4 s) and fails to converge at all at 400×50. The advantage belongs to the looser practical tolerance this study selected, not to the method's documented default. |
| Proposed: accepts some spectral-quality loss | **SUPPORTED vs Olhoff, NOT_TESTED vs Yuksel** | 6.8–7.7% below Olhoff under every evaluator. Against Yuksel there is no evaluator-robust ordering: raw evaluators favour Yuksel by 0.83–1.14%, binary evaluators favour Proposed by 0.66%. |
| Proposed: lacks exact bimodality | **SUPPORTED** | Terminal gap₁₂ 42–47% at 240×30, 320×40 and 400×50. (At 160×20 it happens to be 1.70% — a property of that particular design, not of the method.) |
| Overall ordering Olhoff ≫ Yuksel > Proposed in cost | **NOT_SUPPORTED** | The measured structure is Olhoff ≫ {Yuksel ≈ Proposed}. Yuksel and Proposed are within a factor of ~3 of each other and swap places by mesh and by profile; Olhoff is 8–28× more expensive than either. |

**Summary: the hierarchy is a two-tier structure, not a three-tier one.** Olhoff
is separated from the other two by roughly an order of magnitude in cost and by
a robust ~7% in spectral quality. Yuksel and Proposed occupy the same cost tier
and are not separable on quality by evaluator-robust evidence. The data do not
support the ordering `Olhoff = few expensive / Yuksel = intermediate /
Proposed = cheapest`, and this study reports that rather than tuning toward it.

---

## WP22 — Iteration cost is trajectory dependent

`T_total / N_iter` is not an intrinsic constant for any method. Measured
optimization-loop milliseconds per iteration in 50-iteration windows
(240×30, selected profiles):

| Method | it 1–50 | last window | ratio | eigensolve share, first → last |
|---|---:|---:|---:|---|
| Olhoff (`move=0.005`) | 144.8 | 88.2 | **0.61** | 0.540 → 0.725 |
| Olhoff (`move=0.010`) | 170.2 | 89.5 | **0.53** | 0.392 → 0.707 |
| Yuksel (`move=0.1`) | 24.1 | 26.3 | 1.09 | n/a (stage 1 does no eigensolve) |
| Proposed (`tol=0.01`) | 30.0 | 28.6 | 0.95 | n/a |

**This contradicts the expectation stated in the task.** Olhoff's *early*
iterations are the expensive ones — up to 1.9× the late per-iteration cost —
not the late ones. The mechanism is visible in the telemetry: early iterations
run more inner LP work while the design is still moving, whereas the late
period-two cycle solves a single trivial inner problem per outer step
(`nInner = 1`). The eigensolve *share* rises from ~0.4–0.54 to ~0.71–0.73
precisely because the non-eigensolve work collapses.

Yuksel's per-iteration cost *rises* by 9–28% along the trajectory, which is its
stage structure: stage 1 solves no eigenproblem, stage 2 does. Proposed is
nearly flat.

Per-iteration cost against mesh (practical profiles, ms/iter):

| Method | 160×20 | 240×30 | 320×40 | 400×50 |
|---|---:|---:|---:|---:|
| Olhoff | 40.0 | 96.3 | 171.3 | 284.3 |
| Yuksel | 14.6 | 26.0 | 47.3 | 74.7 |
| Proposed | 14.0 | 29.1 | 53.9 | 90.1 |

Olhoff's per-iteration premium grows with mesh: 2.7× Yuksel at 160×20, 3.8× at
400×50. Multiplicity-aware iterations scale worse than single-mode ones.

---

## WP23 — Hardware comparability

All runtimes here were produced on one controlled configuration (Apple arm64,
MATLAB R2025b, `maxNumCompThreads(1)`, all timed runs sequential). Yuksel's
published absolute seconds were obtained on different hardware and are treated
as **contextual only**; no comparison in this report uses them. The
reproduction checks that remain valid across hardware — stated iteration
budgets, Table-1 arithmetic consistency, qualitative scaling and algorithmic
cost structure — are the ones used.

One timing caveat is disclosed: the four Stage A2 Olhoff refinement runs
(`move` = 0.00625, 0.00875, 0.0125, 0.0175) were executed after the Stage A1
batch in a separate MATLAB session. All Stage A Olhoff runs were sequential and
single-threaded, and per-iteration costs across the eleven runs span
87–129 ms, consistent with trajectory content rather than session effects.

---

## WP24 — Nine-resolution campaign status: **NOT AUTHORIZED**

| Gate | Status |
|---|---|
| 1. All three native practical profiles defined | **NO** — Olhoff has no cross-resolution-validated practical profile |
| 2. Selection rules documented before Stage B | YES — `study_preregistration.json`, WP15 |
| 3. Hold-out validation passes | **NO** — both Olhoff profiles classify FAILED |
| 4. Convergence detectors validated | **PARTIAL** — prospective validation 6/6 exact (WP19), but the Olhoff detector does not generalize |
| 5. Configuration hashes frozen | YES — `profile_freeze_manifest.json`, hash `02075494076df3de` |
| 6. R3 engineering gates pass | YES — `engineering_gates.json`, `pass: true` |

Gates 1 and 3 fail. The nine-resolution campaign was not run and must not be
run until Olhoff has a practical profile that survives cross-resolution
validation. The forbidden mesh set (480×60 … 800×100) was not touched.

---

## Table A — Parameter classification

Full ledger: `results/parameter_classification.csv` (62 rows). Swept parameters:

| Method | Parameter | Class | Native default | Swept | Levels | Justification |
|---|---|---|---|:--:|---|---|
| Olhoff | `move` | N | 0.005 | yes | 0.005, 0.00625, 0.0075, 0.00875, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030 | Primary native update control; WP6 designates it for investigation after the frozen move=0.005 detector campaign failed |
| Olhoff | `tolOuter` | N | 0.001 | characterized | — | Characterized for salvageability, not tuned; never fires at any move |
| Olhoff | practical detector | N | none | families | 3 families × 2 lags × 3 levels | Observer-only per WP6/WP11; primary is the frozen `H_balanced_v1`, not retuned |
| Olhoff | `rminEl` | F | 1.3 el | **no** | — | WP6 forbids sweeping the filter radius in the primary convergence study |
| Olhoff | `tolMult` | R | 0.05 | no | — | Defines bimodality; a robustness parameter that must not set the headline profile |
| Yuksel | `move` | N | 0.2 | yes | 0.1, 0.2, 0.3 | Native OC move limit |
| Yuksel | `stage1_tol` | N | 0.01 | yes | 0.005, 0.01, 0.02, 0.05 | Stage-1 handoff test |
| Yuksel | `stage2_tol` | N | 0.01 | yes | 0.005, 0.01, 0.02, 0.05 | Stage-2 native stop — the method's actual convergence criterion |
| Yuksel | `rmin` | F | 2.5 el | no | — | Published simply-supported case value; model parameter |
| Proposed | `move` | N | 0.2 | yes | 0.1, 0.15, 0.2, 0.3 | Native OC move limit |
| Proposed | `convergence_tol` | N | 0.001 | yes | 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05 | Native stop test on the raw design field |
| Proposed | `optimizer` | F | OC | **no** | — | Resolved by `PROPOSED_NATIVE_PROFILE_AUDIT.md`; MMA is a separate named variant, not a tuning level |
| Proposed | `rmin` | F | 2.0 el | no | — | Resolved authoritative profile value |
| all | geometry, material, volfrac, FE model, supports, `p` | F | — | no | — | Problem definition shared by all three methods |

**Held fixed and why (WP-question 2).** Everything that defines the *problem*
(geometry, material, volume fraction, supports, FE and mass formulation, target
mode, SIMP exponent) is fixed so the three methods solve the same problem.
Everything that defines a *method's identity* (Olhoff's multiplicity
formulation and Eq.(4) mass model, Yuksel's two-stage structure and piecewise
mass law, Proposed's frozen reference inertial load) is fixed so each method
stays itself. Filter radii are fixed per method and held in element units, so
no method gains length scale under refinement. Only genuinely native
algorithmic controls were swept.

---

## Table B — Stage A operating points (240×30)

| Method | Profile | Key parameters | Iterations | Runtime | Quality (E1 raw ω₁) | Robustness |
|---|---|---|---:|---:|---:|---|
| Olhoff | primary | move 0.005, rmin 1.3 el, `H_balanced_v1` | 812 | 78.21 s | 169.754 | FAILED on hold-outs |
| Olhoff | fast | move 0.00625, rmin 1.3 el, `H_balanced_v1` | 320 | 32.04 s | 168.568 | FAILED on hold-outs |
| Olhoff | (candidate) | move 0.010 | 682 | 65.25 s | 168.817 | not frozen, not validated |
| Yuksel | primary | move 0.1, tol 0.01/0.01, rmin 2.5 el | 320 (168+152) | 8.32 s | 159.436 | ROBUST |
| Yuksel | published | move 0.2, tol 0.01/0.01, rmin 2.5 el | 342 (149+193) | 9.41 s | 159.408 | ROBUST |
| Proposed | primary | OC, move 0.2, tol 0.01, rmin 2.0 el | 236 | 6.88 s | 157.639 | ROBUST |
| Proposed | authoritative | OC, move 0.2, tol 0.001, rmin 2.0 el | 627 | 18.58 s | 157.617 | RESOLUTION_SENSITIVE |

---

## Table C — Pareto candidates

| Method | Candidate | Cost | Quality loss | Robustness | Selection status |
|---|---|---:|---:|---|---|
| Olhoff | move 0.005 | 78.21 s | 0.000% | FAILED (hold-out) | SELECTED_PRIMARY → **invalidated by WP17** |
| Olhoff | move 0.00625 | 32.04 s | 0.699% | FAILED (hold-out) | CANDIDATE → **invalidated by WP17** |
| Olhoff | move 0.010 | 65.25 s | 0.552% | not validated | CANDIDATE |
| Olhoff | move 0.015 | 97.60 s | — | look-ahead censored | INELIGIBLE |
| Olhoff | 0.0075 / 0.00875 / 0.0125 / 0.0175 / 0.02 / 0.025 / 0.03 | — | — | not bimodal, detector never fires | INELIGIBLE |
| Yuksel | move 0.1 | 8.32 s | 0.000% | ROBUST | **SELECTED_PRIMARY** |
| Yuksel | move 0.3 | 6.08 s | 0.018% | not validated | CANDIDATE |
| Yuksel | tol_s2 0.05 / both 0.05 / both 0.02 / s2 0.02 / s1 0.005 / both 0.005 | — | — | ω₁ drift 0.10–0.57% | INELIGIBLE (false convergence) |
| Proposed | tol 0.01 | 6.88 s | 0.000% | ROBUST | **SELECTED_PRIMARY** |
| Proposed | tol 0.005 | 7.85 s | 0.025% | not validated | DOMINATED |
| Proposed | tol 0.002 | 8.57 s | 0.034% | not validated | DOMINATED |
| Proposed | tol 0.001 (R3) | 18.58 s | 0.014% | RESOLUTION_SENSITIVE | DOMINATED |
| Proposed | tol 0.02 / 0.05 | 1.43 / 0.98 s | 0.36 / 0.33% | — | INELIGIBLE (false convergence) |
| Proposed | move 0.1 / 0.15 / 0.3 | 18.5 / 24.5 / 28.2 s | 0.14–0.15% | not validated | DOMINATED |

---

## Table D — Hold-out validation

See the full WP17 table above. Condensed:

| Method | Profile | 160×20 | 320×40 | 400×50 | Verdict |
|---|---|---|---|---|---|
| Olhoff | practical | cap 1600 | cap 1600 | cap 1600 | FAILED |
| Olhoff | fast | cap 1600 | cap 1600 | 807 (false) | FAILED |
| Yuksel | practical | 244 / 3.0 s / 157.17 | 572 / 26.1 s / 160.69 | 732 / 53.5 s / 159.97 | ROBUST |
| Yuksel | published | 249 / 2.9 s / 157.11 | 740 / 34.2 s / 160.39 | 727 / 52.5 s / 160.00 | ROBUST |
| Proposed | practical | 107 / 1.6 s / 153.68 | 207 / 11.2 s / 158.76 | 182 / 16.4 s / 159.52 | ROBUST |
| Proposed | authoritative | 936 / 12.3 s / 153.12 | 337 / 18.2 s / 158.68 | cap 2000 | RESOLUTION_SENSITIVE |

---

## Table E — Frozen practical profiles (complete reproduction parameters)

Authoritative machine-readable source: `results/profile_freeze_manifest.json`
(config hash `02075494076df3de`). No implicit or default parameter is left
undocumented.

**Shared across all profiles.** Domain 8.0 × 1.0 × 1.0 m; E₀ = 1.0e7 Pa;
ν = 0.3; ρ₀ = 1.0; volume fraction 0.5; SIMP penalization 3.0; initial design
uniform 0.5; target mode 1; supports = uₓ and u_y fixed at both mid-height end
nodes; plane-stress Q4 with 2×2 Gauss; consistent Q4 mass; filter radii in
element widths at every resolution; `maxNumCompThreads(1)`.

**`olhoff_practical`** (`d857f4ba59d75a26`) — *not validated cross-resolution.*
`olhoffOpt.m` via `repro2007_config('fig3a_best')`; move 0.005; `rminEl` 1.3;
`tolMult` 0.05; multiplicity budget 4; `rhomin` 0.001; `innerSolver` `lp`;
`filterMode` `diag`; `offDiag` false; mass interpolation Du–Olhoff Eq.(4)
discontinuous; `tolOuter` 0.001 (never fires; retained as the source default
only); `maxOuter` 1600 (safety budget — CAP_HIT is not convergence); stopping
rule `H_balanced_v1` with phase period 2, objective block 20, window 40,
persistence 20, block-drift tol 1e-4, phase-recurrence tol 1e-4, ρ-phase RMS
tol 1.25e-3, topology turnover tol 7e-4, modal window 40, gap tol 1e-2,
required N = 2, volume tol 1e-8, LP/eigensolver/finite-state health required,
no recent modal event, earliest evaluable iteration 61.

**`olhoff_fast`** (`2a6fae929cb055bb`) — *not validated cross-resolution.*
Identical to `olhoff_practical` except move = 0.00625.

**`yuksel_practical`** (`74af830703eebfc8`) — **ROBUST.**
`top99neo_inertial_freq.m`; OC; move 0.1; `rmin` 2.5 el; sensitivity filter,
symmetric boundary; η = 0.5; β = 1.0 (projection inactive, no continuation);
`stage1_tol` 0.01; `stage2_tol` 0.01; per-stage budget 1000 (safety —
CAP_HIT is not convergence); stiffness floor ratio 1e-9; void density ratio
1e-9; mass interpolation piecewise at cutoff 0.1 (`g = x` above, `g = x⁶` at or
below); stopping rule = stage-2 `max|Δx| < stage2_tol` after ≥ 2 iterations;
stage-1 stop is a handoff, not convergence.

**`yuksel_published`** (`e347b7ca8aa1b774`) — **ROBUST.** As above with
move = 0.2. Provenance `benchmark_protocol_r3.json`; **not** selected by this
study.

**`proposed_practical`** (`fab7523a8e209f3c`) — **ROBUST.**
`topopt_freq.m`; OC with Lagrange-multiplier bisection (λ relative tolerance
1e-3, ≤ 200 bisections); move 0.2; `tol` 0.01; `rmin` 2.0 el; filter
`density_weighted_sensitivity`, declared boundary `symmetric` (effective
operator is a truncated centroid stencil — disclosed mismatch); load
`F(x) = ω₀²·M(x)·Φ₀`, mode 1, factor 1.0, fully solid reference design,
`Φ₀ᵀM₀Φ₀ = 1`, deterministic phase (largest-magnitude free DOF positive),
refresh interval 0, load sensitivity omitted, no per-iteration renormalisation;
stiffness floor ratio 1e-9; void density ratio 1e-9; design bounds [0.0, 1.0]
(implementation floor 0.0 vs manuscript 0.001 — disclosed mismatch);
`max_iters` 2000 (safety — CAP_HIT is not convergence); stopping rule
`max|x − x_old| ≤ tol` on the raw design field.

**`proposed_authoritative`** (`62bda4dee5cc93b8`) — **RESOLUTION_SENSITIVE.**
As above with `tol` = 0.001. Provenance
`proposed_manuscript_ss_oc_a0_2026-07-14`; **not** selected by this study.

---

## Answers to the required questions

1. **Which parameters genuinely control practical performance?** Olhoff: the
   move limit — but as a *bifurcation* parameter that selects which optimum is
   reached, not as a rate control; the outer tolerance controls nothing.
   Yuksel: `stage1_tol` and `stage2_tol`, each acting on its own stage
   (N₁ ∈ 59–339, N₂ ∈ 38–381); the move limit is non-monotone and weak.
   Proposed: the convergence tolerance alone (34–695 iterations); the move
   limit is non-monotone and the largest move is the slowest.
2. **What was held fixed and why?** See Table A closing paragraph.
3. **What constitutes native convergence?** Olhoff: `max|Δρ| < tolOuter`
   (0.001), which never fires. Yuksel: stage-2 `max|Δx| < stage2_tol` after ≥2
   iterations, stage 1's identical test being a handoff. Proposed:
   `max|x − x_old| ≤ tol` on the raw design field.
4. **Which stopping criteria are unreliable?** All three, in three different
   ways. Olhoff's never fires at any move (the LP always saturates at least one
   move bound). Yuksel's and Proposed's fire on transient lulls — the design
   change rebounds above the firing tolerance for 16–95% of the next 100
   iterations. And the frozen Olhoff *practical* detector, which works on the
   calibration mesh, does not generalize across resolution.
5. **Which parameter regions generate false convergence?** Yuksel `stage2_tol`
   ≥ 0.02 and both-stage tolerances ≥ 0.005 in some combinations (ω₁ drift
   0.10–0.57%); Proposed `tol` ≥ 0.02 (drift 0.28–0.31%); Olhoff loose and mid
   detector tolerances at move 0.005 and 0.010 (fires at 434–451 that pass a
   50-step look-ahead and fail at 100 and 200); and the frozen detector itself
   at move 0.00625 on the 400×50 hold-out.
6. **Which generate numerical failure?** None. Zero solver failures in all 46
   Stage A runs and all 24 validation runs. Volume residuals at every practical
   stop are within the preregistered 1e-3 gate: ≤ 6.2e-11 (Olhoff), ≤ 6.3e-4
   (Yuksel), ≤ 1.3e-4 (Proposed). Disclosed: Yuksel's volume drifts further
   during the *extension* iterations past its native stop, reaching 2.0e-3 at
   the 1000-iteration terminal state — that state is a look-ahead reference,
   not a reported result, and is not what the eligibility gate tests.
7. **Which generate topology/connectivity failure?** None. Every raw-0.5 and
   volume-preserving binary topology in the study is support-to-support
   four-connected.
8. **How sensitive is iteration count?** Very. Proposed spans 34–956 (28×)
   across its native controls; Yuksel spans 97–610 (6.3×); Olhoff's practical
   stop spans 320–1105 where it exists at all, and does not exist for 7 of 11
   move levels.
9. **How sensitive is terminal spectral quality?** Barely, within a method:
   0.05% across eligible Yuksel configurations and 0.15% across eligible
   Proposed configurations — the quality axis does not resolve. Olhoff spans
   0.70%, and its *reproduction* quality (ω₃ against the paper's 284.9) spans
   +0.11% to −8.3% across move levels, which is far more sensitive than ω₁.
10. **Cost–quality Pareto fronts?** WP13 and `method_pareto_profiles.csv`.
11. **Is there a clear knee?** Olhoff: no — its front is two endpoints tied at
    utopia distance 1.0000 plus one interior point, and neither endpoint
    survives hold-out. Yuksel and Proposed: no meaningful knee, because their
    quality axes span < 0.15%; their fronts are effectively cost-only.
12. **Which profile should represent each method?** Yuksel:
    `yuksel_practical` (move 0.1, tol 0.01/0.01) — or equivalently
    `yuksel_published`, which is statistically indistinguishable and better
    provenanced. Proposed: `proposed_practical` (tol 0.01), and **not** the R3
    authoritative `tol = 0.001`, which is resolution-sensitive. Olhoff:
    **unresolved** — no candidate survives cross-resolution validation.
13. **Do the profiles generalize without retuning?** Yuksel yes (3/3 meshes,
    both profiles). Proposed yes at `tol = 0.01` (3/3), no at `tol = 0.001`
    (2/3). Olhoff no (0/3, both profiles).
14. **Cost to reach 95 / 97.5 / 99 / 99.5% of late quality?** WP12 table.
    Olhoff 169 / 200 / 266 / 338 iterations (19.9 / 22.7 / 28.7 / 35.3 s);
    Yuksel 170 iterations / 4.06 s for all four levels; Proposed 6 / 8 / 17 /
    19 iterations (0.24 / 0.29 / 0.55 / 0.60 s). Every method then spends
    51–91% of its practical budget beyond the 99.5% point — and for Yuksel and
    Proposed the native objective is a poor quality proxy, since it plateaus
    long before the design does.
15. **Absolute cross-method quality differences?** Olhoff exceeds Yuksel by
    6.47–7.51% and Proposed by 6.80–7.69%, sign-consistent across E1/E2/E3 and
    both representations. Yuksel versus Proposed is **model-dependent**: raw
    +0.83 to +1.14% for Yuksel, binary −0.66% (i.e. Proposed ahead). No single
    ranking may be claimed for that pair.
16. **Is Olhoff's iteration count naturally O(10²)?** No. Where a validated
    practical stop exists it is 320–812 outer iterations, i.e. high 10² to low
    10³; on three of four meshes no validated stop exists within 1600.
17. **Versus Yuksel's fixed 200 Dynamic-Code iterations (retrospective only)?**
    Olhoff's validated stop of 812 at 240×30 is 4.1× that figure, and its
    hold-out behaviour is unresolved at 1600 (8×). This comparison was made
    only after freezing; no tolerance or move level was chosen with reference
    to 200.
18. **Is Yuksel computationally intermediate?** In iteration count, yes on the
    one mesh where all three methods have a validated stop (812 > 320 > 236);
    on the hold-outs Olhoff has no stop, so the question is untestable there.
    In total runtime, no — Yuksel is slower than Proposed at all four meshes
    and sits in the same tier as it, an order of magnitude below Olhoff.
19. **Is Proposed the most computationally efficient?** At its selected
    practical profile, yes at all four meshes (1.6 / 6.9 / 11.2 / 16.4 s vs
    Yuksel's 3.0 / 8.3 / 26.1 / 53.5 s). At its R3 authoritative profile, no —
    it is twice Yuksel's cost at 240×30 and does not converge at 400×50.
20. **What quality does Proposed sacrifice?** 6.8–7.7% of ω₁ against Olhoff
    under every common evaluator, plus the absence of modal coalescence (gap₁₂
    42–47%). Against Yuksel it sacrifices nothing measurable that is
    evaluator-robust.
21. **Does the evidence support the intended hierarchy?** Partially — see
    WP21. The structure is two-tier (Olhoff ≫ Yuksel ≈ Proposed), not
    three-tier.
22. **Are the profiles ready for the nine-resolution campaign?** No. Yuksel and
    Proposed are ready; Olhoff is not, and gates 1 and 3 of WP24 fail.

---

## Reproducible artifacts

Scripts in `analysis/three_method_parametric_study/`:
`study_base_config.m`, `study_evaluate_design.m`, `run_stage_a.m`,
`run_stage_a_v2.m`, `run_stage_a2_olhoff.m`, `run_stage_a2_ours.m`,
`analyze_olhoff_moves.m`, `aggregate_stage_a.m`, `export_objective_traces.m`,
`run_holdout.m`, `analyze_holdout.m`, `select_profiles.py`,
`classify_robustness.py`, `make_windows.py`, `make_plots.py`,
`run_engineering_gates.m`.

Results in `analysis/three_method_parametric_study/results/`:

| Artifact | Contents |
|---|---|
| `parameter_classification.csv` | Table A, all 62 parameters, F/N/R/D/U |
| `parametric_run_ledger.csv` | all 46 Stage A runs, both passes, native + common metrics |
| `olhoff_parametric_results.csv` | Olhoff subset |
| `yuksel_parametric_results.csv` | Yuksel subset |
| `proposed_parametric_results.csv` | Proposed subset |
| `olhoff_trajectory_summary.csv` | 11 move levels: spectra, gaps, modal events, validity |
| `olhoff_detector_grid.csv` | 198 detector candidates with look-ahead verdicts |
| `false_convergence_events.csv` | 132 horizon evaluations (74 PASS / 48 FAIL / 10 CENSORED) |
| `eligibility_ledger.csv` | every run with its eligibility verdict, rebound and ω₁ drift |
| `method_pareto_profiles.csv` | Table C, dominance and utopia distances |
| `within_method_quality_levels.csv` | WP12 cost to 95/97.5/99/99.5% |
| `objective_traces.csv` | 45 498 per-iteration objective, cost and design-change records |
| `trajectory_window_costs.csv` | WP22, 1039 fifty-iteration windows |
| `selected_profile_candidates.json` | selection output with the rule that produced it |
| `profile_freeze_manifest.json` | Table E, config hash `02075494076df3de` |
| `cross_resolution_validation.csv` | 24 hold-out and prospective runs |
| `robustness_classification.csv` | Table, WP18 classes |
| `figures/fig01…fig12` | the twelve required plots |

Raw trajectories (`.mat`, v7.3) in `raw/stage_a/`, `raw/stage_a_v2/`,
`raw/observer/`, `raw/prospective/`.

---

## Final verdict

Olhoff has no cross-resolution-validated practical operating profile: both
frozen candidates classify FAILED, one by never firing on any hold-out mesh and
one by firing on a confirmed false positive. Yuksel and Proposed each have a
validated ROBUST profile. The comparison is therefore defensible for two of the
three methods and unresolved for the third.

**PARTIAL PROFILE IDENTIFICATION — OLHOFF REMAINS UNRESOLVED**
