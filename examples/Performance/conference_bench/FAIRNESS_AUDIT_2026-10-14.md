# Fairness audit of the timed path — 2026-10-14

Scope: the work each method performs inside the reported total wall time
(`total_wall_time_s`) of `performance_comparison.m`, with the effective benchmark
configuration (frozen profiles, single thread, `stagnation_x` stop rule).  Question:
are the mechanisms that all methods repeat implemented on equal terms?

Trigger: in the 2026-10-13 Windows campaign the Yuksel code overtook the proposed
method at 720×90 and 800×100.  Iteration counts did not explain it (proposed 54–55,
Yuksel 68–81); the proposed method's time per iteration jumped 2.4× between 640×80
and 720×90, where ndof crosses 2^17 = 131072 and MATLAB `sparse()` becomes several
times slower.

## Fixed (implementation only; results unchanged up to round-off)

| # | Mechanism | Before | After | Measured effect |
|---|---|---|---|---|
| 1 | Global K, M assembly — Proposed | 64 triplets/element for K and M, then `(K+K')/2`, `(M+M')/2` every iteration | lower triangle (36 triplets) of the symmetric element matrices, mirrored (`M + tril(M,-1)'`); K kept lower | assembly 0.517 → 0.256 s at 800×100 (Mac); whole iteration 0.812 → 0.565 s; same iteration counts, designs equal to 5e-7 |
| 2 | Global K, M assembly — Du–Olhoff (`fem/assemble2D.m`, `fem/model2D.m`) | as Proposed before | as Proposed after | bitwise identical below 2^17 (160×20); ≤ 2.4e-16 relative at 720×90 |
| 3 | Linear solve — Proposed | full `K(free,free)` + `\` | `decomposition(lower K(free,free),'chol','lower')`, as the Yuksel code | 0.196 → 0.177 s at 800×100; identical solution |
| 4 | Objective — Proposed | `u'*(K*u)` (extra full mat-vec) | `u(free)'*f(free)`, as Yuksel's `F'*U` | removes one mat-vec |
| 5 | Post-solve topology eigensolve in `run_topopt_from_json` (3 modes) | ran for Proposed and Yuksel inside the timed call, duplicating the solver's own final eigensolve; Du–Olhoff has none | `postprocessing.compute_modes = 0` in the benchmark (skips it when no snapshot/plot/correlation needs it) | 0.77 s per run at 800×100 |
| 6 | Final display figure (`plotTopology`) | created unconditionally by Proposed and Yuksel even with live plotting off; Du–Olhoff none | only when live plotting is on (the saved image is rendered separately) | 1.87 s per run at 800×100 |
| 7 | Yuksel stage-1 ω₁ eigensolve between the stages | always (diagnostic; stage 2 does not use it) | `optimization.yuksel.report_stage1_omega = false` in the benchmark (default stays true for the paper-reproduction runners) | ≈ 0.5 s per run at 800×100 |
| 8 | Yuksel `localAssemble` | two `exist('fsparse')` look-ups on every call | once per session | ≈ 2 ms per call |

`fsparse` is installed neither on the Mac nor on BIO-2_HELI, so the Yuksel code uses
MATLAB `sparse()` on both — no hidden advantage.

## Verified equal

* Single thread for every measured run (`maxNumCompThreads(1)` set by the driver; Du–Olhoff also sets it itself).
* No eigensolve inside the Proposed or Yuksel loops in the benchmark (frozen semi-harmonic mode; frequency logging, mode histories, audits, history recording, live plotting all off).
* Du–Olhoff: diagnostics and verbose output off.
* One final eigensolve of the final design per method stays in the timed total (the method's own reported ω).

## Documented, not changed (method-intrinsic or scientific settings)

| Item | Detail | Size |
|---|---|---|
| Sensitivity filter | Proposed: sparse H mat-vec, radius 2 elements; Yuksel: two `imfilter` convolutions, radius 2.5 elements (5×5 kernel); Du–Olhoff: four sparse H mat-vecs, fixed physical radius 0.06 | frozen profile settings |
| OC bisection | Proposed brackets λ in [0, 1e9] with relative tolerance 1e-3 (≈ 40 bisections); Yuksel uses an adaptive bracket and 1e-4 (≈ 14) | ≈ 0.018 s per iteration at 800×100 against Proposed; changing it changes the design |
| Per-iteration log line | Proposed prints one line per iteration | ≈ 0.4 ms per iteration |
| Timing boundary | Du–Olhoff's total (`tCall` in `olhoffcurrent_run`) excludes configuration resolution and the path guard; Proposed/Yuksel totals include JSON parsing and validation in `run_topopt_from_json` | small; against Proposed/Yuksel |
| Du–Olhoff internals | one redundant `genGrad` call per outer iteration (re-computes F(:,1,1)); `eigs` with tol 1e-12 and a fixed start vector (no warm start); the nested MMA loop dominates (94–98 % of its time) | part of the reconstruction; against Du–Olhoff |
| Yuksel stage 2 | mode-tracking scalars (`modeCos`, `du`) computed every iteration although only audits use them | O(ndof), negligible |

## Consequence

Results of earlier campaigns are not comparable in time with campaigns run after
these changes.  The cancelled Windows `stagnation_x` campaign of 2026-10-14
(17 of 27 rows) is superseded.
