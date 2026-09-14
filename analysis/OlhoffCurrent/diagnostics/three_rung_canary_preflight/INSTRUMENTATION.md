# INSTRUMENTATION — Part B

Status: **specified, implemented, exercised.** The retention decision was made
*before* any run, as Part B requires, and `cp_preflight` enforces it: it
refuses to start if the worst-case budget exceeds a quarter of host RAM.

## 1. Per-outer scientific / controller record

Retained for every outer iteration, all written by the solver's own recorder
(`hist`) or derived from the retained trajectory by `cv_telemetry` — called,
never re-typed, so the canary CSV is column-for-column the validated C320's.

| quantity | source |
|---|---|
| iteration | `P.outer` |
| stage, move | `hist.stage`, `hist.move`; `moveChanged`, `descent` derived |
| exA, exB, exE | `hist.exA/exB/exE` |
| persistence counters | `hist.exNA`, `hist.exNB` |
| declaration state | `hist.exDecl`; `res.exhaustion.events`, `.eventBranch`, `.descents` |
| branch identity | `res.exhaustion.terminalBranch`, per-event `eventBranch` |
| detector internals | `hist.exCos`, `exNet`, `exMedcos`, `exMednet`, `exAmp`, `exStageStart` |
| beta | `hist.beta` — **recorded as a diagnostic only**; it holds no authority |
| ω₁, ω₂, ω₃ | `hist.omega(1:5, :)` (Jcalc = n + Nmax = 5) |
| gap12, gap23 | `hist.gap12`; gap23 derived from `hist.omega` |
| multiplicity / subspace size | `hist.N`, `hist.degen` |
| J / next-mode warning | `hist.multJ` per iteration, plus the `res.log` line |
| volume, volume error | `hist.vol`, `hist.volErr` |
| M_nd, grayness, mid fraction | `cv_telemetry` from `RHO` |
| max\|dρ\|, ‖dρ‖₂ | `hist.dxOuter`, `hist.dxNorm2` |
| inner MMA iterations, convergence, cumulative | `hist.nInner`, `hist.innerConv`, `hist.cumInner` |

## 2. Timing record

`hist.tOuter`, `hist.tEig`, `hist.tGrad`, `hist.tInner`, with
`tOther = tOuter − tEig − tGrad − tInner` derived.

These are **nondeterministic performance telemetry**. The solver writes them
and never reads them back, and they are excluded from every scientific-state
comparison, from `olhoffcurrent_config_hash` and from every evidence digest of
scientific state. They are not bitwise-reproducible and are never presented as
if they were.

## 3. Raw trajectory

`runtime.diagnostics = true` makes the solver retain `res.diag.drho{k}` for
every outer iteration. `cp_run.m` replays those increments from the uniform
initial design through the same clamp the solver applies, producing dense
`RHO` and `DRHO` (NE × nOuter), and then **proves the rebuild**:

```matlab
assert(isequal(RHO(:,end), res.rho))      % exact, not approximate
clampErr = max(max(abs(diff([rho0*ones(NE,1) RHO],1,2) - DRHO)));
```

Saved to `evidence/three_rung_canary_preflight/<tag>_trajectory.mat` (-v7.3)
together with `hist`, `cfg`, `meta`, `res.exhaustion` and `res.log`.

## 4. Retention budget — decided before the run, not after

| mesh | NE | bytes/column | worst case `RHO`+`DRHO` at cap 1600 | at the projected outer count |
|---|---|---|---|---|
| 480×60 | 28 800 | 230 kB | 0.74 GB | 0.27 GB (≈586 outer) |
| 800×100 | 80 000 | 640 kB | 2.05 GB | 1.45 GB (≈1130 outer) |

Host RAM is 68.7 GB, so full retention fits with a wide margin and **no
information is discarded at either mesh**. Both preflights confirmed
`fits_in_ram = true` at run time before the solve was permitted to start.

**As executed**, both runs finished far below the worst case, because both used
far fewer iterations than the cap:

| mesh | outer used | `RHO`+`DRHO` retained | trajectory file | SHA-256 |
|---|---|---|---|---|
| 480×60 | 386 | 0.18 GB | 155.5 MB | `a87546bc391cdc683def34a9f27678884528032f6b140e156349d2e74135ab9b` |
| 800×100 | 468 | 0.60 GB | 536.1 MB | `aef7af1c3ec669b41d6ff6799aaaea4bc35feac0d20ff739f91a532bece012eb` |

The checkpoint fallback was never needed and was never used. Both rebuilds were
proved exact, with clamp-displacement residual 5.551e−17 at both meshes.

Fallback, specified in advance and *not* used because the budget fits: had the
budget failed, the run would have retained (a) full scientific and controller
scalar history for every iteration, (b) `RHO` at every 10th iteration, (c) a
snapshot at every declaration and every stage transition, and (d) the final
full design — with the reduced policy recorded in `PREFLIGHT_MANIFEST.json`
*before* the solve. A post-hoc discard is never permitted.

## 4a. One schema gap, and how it was closed without touching a frozen artefact

`cv_export`'s 55-column CSV schema is the **validated C320 study's own**, and
this study reuses it unmodified so the canary CSVs stay column-for-column
comparable to that oracle. It carries `tOuter` but not `tEig`, `tGrad`,
`tInner`, and it carries ω₁ and ω₂ but not ω₃…ω₅ or gap23 — all of which Part B
requires.

Those quantities are retained: `hist.omega` is `Jcalc × n` (Jcalc = 5) and the
three timing channels are written every iteration. So rather than edit another
study's frozen exporter, `scripts/cp_supplement.m` reads the trajectory `.mat`
the canary already wrote and emits them as a second per-iteration file,
`runs/<tag>_supplement.csv`:

```
outer, omega1..omega5, gap23, tOuter, tEig, tGrad, tInner, tOther,
multN, multJ, degen, nInner, dxOuter, dxNorm2, move, stage, vol
```

It reads only, runs after the solve, and writes no scientific state.

## 5. Saved state for the fixed-work benchmark

`cp_run.m` writes `<tag>_state.mat` holding the terminal `rho` and the `cfg`.
`cp_fixedwork.m` reads it, re-evaluates kernels at that fixed design, and
asserts at the end that `rho` is unchanged. It writes no design and makes no
convergence claim, so it cannot contaminate a scientific record.

## 6. Host-load record (Part G)

`cp_hostprobe.m` runs immediately before and immediately after each canary,
and again before the fixed-work benchmark. It records load average, swap,
free pages, competing MATLAB processes and the top CPU consumers, then times a
fixed, mesh-independent calibration kernel (a 1200³ dense product and a
200 000-unknown tridiagonal solve, 3 timed repeats each after a discarded
warm-up). No thermal or cache control is attempted and none is claimed; the
probe exists to make a gross host-load difference between the two canaries
visible, not to certify reproducibility.

## 7. Known comparability limit

The canaries must run with `runtime.diagnostics = true`; the legacy nine-mesh
campaign ran with it `false`, and the recorder costs measurable time per outer
iteration. An equal-mesh, equal-formulation upper bound at 400×50 puts the
combined recorder-plus-controller-plus-session effect at ≈ 1.79× per outer.
Canary wall time is therefore **not** directly comparable to legacy wall time,
and `PERFORMANCE_DECOMPOSITION.md` states this before quoting any number.

This is a real constraint on the future performance campaign, not a defect of
this study: a nine-mesh series cannot simultaneously retain full trajectories
and report recorder-free timings. `CAMPAIGN_DECISION.md` §4 carries it forward.
