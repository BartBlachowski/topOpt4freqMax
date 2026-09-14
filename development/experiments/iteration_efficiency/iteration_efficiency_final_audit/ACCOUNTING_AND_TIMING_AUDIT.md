# Iteration accounting and timing firewall audit

## 1. Accounting — correct

All paper-facing accounting is produced by `+iefinal/build_rows.m::localAccounting` at the endpoint index `k = k_enter`.

| method | fields | verdict |
|---|---|---|
| Proposed | `native_iterations = k` (OC updates) | correct |
| Yuksel | `yuksel_stage1_iterations = s1`, `yuksel_stage2_iterations = k`, `yuksel_total_iterations = s1+k`, `native_iterations = s1+k` | correct — Stage 1 neither omitted nor double-counted |
| Olhoff-LP | `olhoff_outer_updates = k`, `olhoff_lp_calls = min(k, lp_calls)`, `olhoff_failed_lp_calls`, `olhoff_lp_backend_iterations = Σ finite over 1..k` | correct |
| Olhoff-MMA | `total/mean/median/p95/max` inner, `cap_hit_count/fraction`, `converged_inner_count/fraction`, all from `inner(1:k)` | correct |

**`nInner` never appears in any result row or CSV column.** The forbidden representation of `nInner = 1` as a genuine LP/simplex/HiGHS iteration count cannot occur.

**LP backend iterations are genuine.** Measured over 120 outer updates at 160×20: unique values `[1 2 3 4 5 6 8 10 11 12 13]`, sum 322 against 120 LP calls. Distinguishable from `lp_calls`; safe to publish as backend work. (Only the first ~10 outers return 1 — a short trajectory alone would have looked degenerate.)

**MMA statistics come from genuine nested-MMA iterations.** `innerLoop.m` holds `F`, `fJJ`, `lam` fixed and calls `mmasub`/`subsolv` per inner iteration; live run recorded 108/102/108/101 inner iterations across four outer updates with zero LP calls.

Residual (MODERATE, F-04): `ie2a.account_iterations` is hash-pinned in `PRODUCTION_MANIFEST.json` and verified by `preflight`, but **never executed** — `build_rows` reimplements the arithmetic inline. Its frozen guards (rejecting `nInner_as_solver_iterations`, and requiring exactly one MMA inner-work record per outer update) therefore never run in production. The inline arithmetic is substantively correct, so this is a provenance defect, not a numerical one.

## 2. Timing firewall — correct in isolation

`+iefinal/run_timing_firewall.m` runs **after** endpoints are frozen and calls only native method entry points.

| requirement | implementation | verdict |
|---|---|---|
| Candidate C excluded | `evaluate_common` absent from the timed function (asserted by the test suite) | correct |
| topology / persistence / reference excluded | absent | correct |
| trajectory disk I/O excluded | `captureTrajectory=false` (Olhoff), `record_history=false` (Proposed/Yuksel) | correct |
| rendering / figure export / post-hoc diagnostics excluded | absent | correct |
| serial single-thread | `maxNumCompThreads(1)` inside `localOnce` | correct |
| warm-up | `rep=0` flagged `discarded_warmup` and excluded from the median | correct |
| three retained replays | `cfg.timing_repetitions = 3`, loop `rep = 0:3` | correct |
| deterministic fixed horizon | `max_iters`/`maxOuter` set to the frozen `k_enter` / `k_cert` | correct |
| clean replay state | fresh config per call; no reuse of the measurement run | correct |
| MMA inner work inside native time | `olhoffOpt` timed whole; nothing subtracted | correct — expensive MMA work is **not** removed |

Deduplication by `method|variant|mesh|horizon` avoids re-timing equal horizons. Yuksel stage times are retained in `timing_replay_samples.csv`.

## 3. Finding F-01 (CRITICAL) — the Yuksel timing replay does not reproduce the measured computation

`run_timing_firewall.localOnce` sets, for Yuksel:

```matlab
prm = struct(..., 'max_iters', c.horizon);   % horizon = k_enter or k_cert (a Stage-2 index)
prm.stage1_max_iters = 2000;
```

but `tools/Matlab/run_topopt_from_json.m:535` applies

```matlab
stage1MaxIter = min(stage1MaxIter, maxiter);
```

so the Stage-1 budget is silently clamped to the **Stage-2** horizon.

### Measured

96×12 Yuksel, `maxNumCompThreads(1)`:

| path | `max_iters` | stage1 requested | stage1 **effective** | stage1 iterations | stage2 iterations |
|---|---:|---:|---:|---:|---:|
| reference / measurement | 3200 | 2000 | 2000 | **151** | 3200 |
| timing replay at horizon 150 | 150 | 2000 | **150** | **150** | 150 |

`timing_replay_reproduces_stage1 = false`.

### Consequence

The timed run is a **different optimization** from the measured one: a truncated Stage 1 hands a different design to Stage 2, so the Stage-2 iterations being timed are not the Stage-2 iterations whose `k_enter`/`k_cert` are reported, and Stage-1 work is under-counted. This propagates into `native_total_time`, `native_total_time_to_enter`, `native_total_time_to_cert`, and `mean_native_iteration_time = te / (s1 + k)` — which divides a clamped-Stage-1 replay time by the *un*clamped trajectory's `s1`.

This is not a corner case. `ITERATION_ACCOUNTING_SPEC.md` and `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` record that Yuksel Stage 1 hit its cap at **640×80, 720×90 and 800×100** in the frozen campaign — precisely the three meshes where timing matters most, and where any `k_enter` below ~1000 truncates Stage 1 severely.

Nothing detects it: `localOnce` returns `native_iterations` but no code compares it against the expected count, and no test covers a case where Stage 1 exceeds `k_enter`.

It corrupts the computational-performance half of the study (§16 A/B), so it blocks.

### Minimum correction

1. Make the Stage-1 budget independent of the Stage-2 horizon on the timing path. The narrowest change is to drop or condition the `min(stage1MaxIter, maxiter)` clamp at `run_topopt_from_json.m:535`; it is behaviour-neutral for the reference/measurement path, where `maxiter=3200 ≥ stage1_max=2000` leaves `min()` a no-op.
2. Add a hard assertion in `run_timing_firewall.localOnce` that the replay's Stage-1 and Stage-2 counts equal the recorded trajectory's, failing closed on mismatch.

### Minimum re-verification

- One matched Yuksel reference-vs-timing pair at a mesh where Stage 1 > `k_enter`, showing identical Stage-1 and Stage-2 counts.
- A Proposed + Yuksel reference-run behaviour-neutrality check before and after the clamp change (bit-identical final design), since `run_topopt_from_json.m` drives both principal non-Olhoff methods.

## 4. Residual timing issues (non-blocking)

- **F-10 (MINOR)**: only `total/stage1/stage2` seconds are recorded and only the median reported; the contract's `timing.components` decomposition (`T_init`, `T_native_finalize`, …) and `summary = [median, range, MAD]` are not fully realised. Raw samples are retained, so range and MAD are derivable without re-running.
