# B-1 — Yuksel timing replay work identity

## Root cause

Not an off-by-one. A genuine truncation with two interacting parts:

1. `+iefinal/run_timing_firewall.m` sets `prm.max_iters = c.horizon` for Yuksel, where the horizon is a **Stage-2** index (`k_enter` or `k_cert`).
2. `tools/Matlab/run_topopt_from_json.m:535` applied `stage1MaxIter = min(stage1MaxIter, maxiter)`, treating `max_iters` as a **global** budget.

So a Stage-2 timing horizon silently clamped the Stage-1 budget.

### Why 151 became 150

Stage 1 terminates on its own convergence test (`top99neo_inertial_freq.m:579`, `loop > 1 && ch < tolX`), recording `stop_reason = 'density_change_tolerance'`. With the budget at 2000 the loop converged at 151. With the budget clamped to 150 the `while loop < maxit` bound expired first, at a **non-converged** design, recording `stop_reason = 'max_iterations'`. The replay handed a different Stage-1 design to Stage 2, so the whole timed Stage-2 trajectory differed from the measured one.

The clamp also contradicted its own in-file comment, which states "Stage 1 runs to its own native convergence test … maxiter is a safety budget only" and records that an earlier `min(maxiter, 200)` cap was removed precisely because "Capping stage 1 at 200 truncated it and inflated stage 2." Line 535 was the residue of that same defect.

### Stage 2 — no analogous mismatch

Stage 2's bound is `while loop < maxit` with `maxit = maxiter`, and under `extend_beyond_native_stop = true` it records `native_stop_iter` but keeps running to `maxit`. A replay at horizon H therefore executes exactly H Stage-2 updates. Verified: `D_stage2_identity` holds at both meshes.

### Other timing replays

Proposed is single-loop: `max_iters` is its only cap, so no clamp can apply. Olhoff sets `cfg.maxOuter = horizon` directly and `olhoffOptStabilized` has no convergence break at all. A repository-wide search for further `min(..., maxiter)` clamps in the driver found only line 535 and the comment referencing the removed 200-cap. **No other timing replay silently changes native work.**

## Correction

**1. Stage-1 budget made independent — opt-in, zero collateral.**

`run_topopt_from_json.m` now honours `optimization.yuksel.stage1_budget_independent` (default **false**). With the flag absent the historical clamp is applied exactly as before, so *every* existing caller is byte-identical — including `run_stage_a.m`, the one historical script that passes `stage1_max_iters = 1000` with `max_iters = 300` and would otherwise have changed behaviour. `study_base_config.m` passes the flag through, defaulting false.

**2. Work identity derived from authoritative native evidence and asserted.**

`run_timing_firewall` now takes the measurement trajectories as a required fourth argument and, per timing case, reads the **recorded** native Stage-1 count and the **recorded** Stage-1 budget from that evidence (`run_trajectory` now stores `stage1_budget`). It replays with that same budget and the flag set, then asserts:

- Yuksel: `replay Stage-1 == recorded native Stage-1`, `replay Stage-2 == horizon`, `total == Stage-1 + Stage-2`;
- Proposed / Olhoff: `native updates == horizon`.

Any mismatch raises `iefinal:TimingWorkMismatch` and fails closed. A case with no matching native evidence raises `iefinal:TimingEvidenceMissing` rather than timing an unverifiable replay. No hard-coded compensation is used anywhere; nothing adds an iteration.

## Evidence

Measured at production-floor meshes, `maxNumCompThreads(1)`, Stage-2 cap cut to keep cost down:

| mesh | native S1 | OLD S1 | OLD S2 | OLD total | NEW S1 | NEW S2 | NEW total | A | B | D | E |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|
| 160×20 | **121** (converged) | **120** (capped) | 120 | 240 | **121** | 120 | 241 | ✅ | ✅ | ✅ | ✅ |
| 240×30 | **168** (converged) | **167** (capped) | 167 | 334 | **168** | 167 | 335 | ✅ | ✅ | ✅ | ✅ |

- **A — failure reproduced before the fix**: with the flag absent, the effective Stage-1 budget collapses to the horizon and Stage 1 stops one short, un-converged, at both meshes.
- **B — identity after the fix**: Stage-1 work equals the native Stage-1 work exactly; the effective budget is 2000, and Stage 1 converges.
- **C — different Stage-1 counts**: 121 vs 168. A compensation constant could not satisfy both; the fix is not an off-by-one patch.
- **D — Stage-2 identity**: replay Stage 2 equals the horizon at both meshes.
- **E — total identity**: `241 = 121 + 120` and `335 = 168 + 167`.
- **Independent cross-check**: measuring the native Stage-1 length a second way — clamp left active but the budget raised above the Stage-1 length — reproduced 121 and 168 (`all_confirm = 1`).

### F — timing firewall exclusions intact

Asserted in `testTimingFirewall` against the firewall source: no `evaluate_common`, no `topology_metrics`, no `scan_persistence`, no `render_*`/`exportgraphics`, `captureTrajectory=false` for Olhoff and `record_history=false` for Proposed/Yuksel. Serial single-thread, one discarded warm-up plus three retained replays, deterministic fixed horizon — all unchanged. Nested MMA inner work still stays inside native time; nothing is subtracted.

### G — untimed native results unchanged

Pre-edit (`git show HEAD:`) driver and `study_base_config` shadowed onto the path, compared against the post-edit versions on the reference path at 160×20:

| method | design bit-identical | max abs diff | ω bit-identical | iterations identical | stage counts identical |
|---|:-:|---:|:-:|:-:|:-:|
| Proposed | ✅ | **0** | ✅ | ✅ | ✅ |
| Yuksel | ✅ | **0** | ✅ | ✅ | ✅ |

## Verdict: **CLOSED**
