# COST_REPORTING — Part 17

Infrastructure only. **No campaign was run**, and nothing here is a performance result.

## Why per-outer cost is needed

The committed R = 0.06 sweep of the Pedersen formulation has **non-monotone** outer counts: 121, 111, 101, 93, 112, 130, 156, 204, 246 from 160×20 to 800×100. A total-time power law mixes per-iteration cost with that count. Both must therefore be reported.

## What is reported now

From `olhoffcurrent_run` accounting, built from the solver's per-iteration timers (`hist.tOuter`, `tInner`, `tEig`, `tGrad`; `tOuter` is supplied by upstream 253069):

| field | definition | status |
|---|---|---|
| `total_wall_time_s` | caller-side tic/toc around `olhoffSolve` | kept |
| `outer_time_excluding_inner_s`, `inner_time_total_s`, `overhead_time_s`, `eigen_time_s`, `gradient_time_s`, `outer_bookkeeping_time_s` | as before | kept |
| `inner_iterations_per_outer_mean`, `inner_time_per_outer_mean_s`, `inner_time_per_inner_iteration_mean_s`, `inner_time_share_pct` | as before | kept |
| `outer_time_total_s` | Σ tOuter | **new** |
| `outer_time_per_outer_mean_s` | Σ tOuter / N_outer | **new** |
| `outer_time_per_outer_median_s` | median tOuter | **new** |
| `outer_time_excluding_inner_per_outer_mean_s` | (Σ tOuter − Σ tInner) / N_outer | **new** |
| `eigen_time_per_outer_mean_s` | Σ tEig / N_outer (FE assembly + eigensolve) | **new** |
| `gradient_time_per_outer_mean_s` | Σ tGrad / N_outer | **new** |
| `total_wall_time_per_outer_s` | total_wall_time_s / N_outer | **new** |

`N_outer + N_inner` is still never reported as an iteration count. The nesting assertions (`tInner ≤ tOuter`, `tEig + tGrad + tInner ≤ tOuter`, `Σ tOuter ≤ call`) are unchanged.

For the adaptive-box preset, `stopping.final_move_limit` is documented as the largest per-element box (`final_move_limit_meaning`), and `stopping.final_move_mean` gives the mean box.

## Harness

| artifact | per-outer content |
|---|---|
| `rec.times` (`confbench_run_case`) | all new fields |
| `conference_performance_detailed.csv` | `olhoff_preset`, `olhoff_outer_time_total_s`, `…_per_outer_mean_s`, `…_per_outer_median_s`, `…_excluding_inner_per_outer_mean_s`, `olhoff_eigen_time_per_outer_mean_s`, `olhoff_gradient_time_per_outer_mean_s`, `olhoff_total_wall_time_per_outer_s` (Olhoff-gated) |
| `BENCHMARK_NOTES.md` | "Cost per outer iteration" table: outer, total, total/outer, outer-excl-inner/outer, eig/outer, inner/outer, inner MMA per outer, per inner iteration |
| `scaling.per_outer` (`confbench_scaling_fit`) | power-law fits T/N_outer(Ne) for total/outer, outer-excl-inner/outer, eig/outer, inner/outer, per inner iteration. Same `ok` rows and the same refusal rules as the total-time fit; only for methods whose records carry an outer count. |
| `timing_schema.json` | definitions of every per-outer field |
| `performance_comparison.m` | prints the per-outer fits |

## Tests

| test | checks | result |
|---|---|---|
| `tests/test_cost_reporting` (160×20, cap cut to 3) | every total and per-outer field present and finite; per-outer means × N_outer reproduce the totals (rel 1e-12); outer loop nested in the call; overhead = total − Σ tOuter; status CAP_HIT; preset/role/upstream commit recorded; adaptive move-limit meaning | **8/8 PASS** |
| `confbench_selftest` T14 (synthetic records) | per-outer fit uses the same 4 ok rows (CAP_HIT excluded); exponent equals an independent least-squares slope (< 1e-10); a method without outer counts gets no per-outer fit | **PASS** |
| harness smoke (`harness_check.json`) | per-outer columns in the detailed CSV, per-outer table in the notes | **PASS** |
