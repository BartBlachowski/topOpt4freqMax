# TELEMETRY_READINESS — Step 5D

**Result: READY.** No code or algorithm was changed to add telemetry.

**Sources.**
- Record assembly: `examples/Performance/conference_bench/confbench_run_case.m` (`runOlhoff`) and `analysis/OlhoffCurrent/olhoffcurrent_run.m` (`accounting`, `stopping`).
- Export: `confbench_export.m` and `confbench_scaling_fit.m`.
- Verified live in the post-merge cap-3 harness smoke (`evidence/POST_harness.json`: record fields, times, counts, export flags).

| required | where it is recorded per run |
|---|---|
| total wall time | `rec.times.total_wall_time_s` |
| tOuter | `rec.times.outer_time_total_s` = Σ tOuter; `outer_time_per_outer_mean_s`, `outer_time_per_outer_median_s`. `olhoffcurrent_run` asserts `numel(hist.tOuter)` = outer count and that phases nest in tOuter. |
| eigensolve per outer | `rec.times.eigen_time_s`, `eigen_time_per_outer_mean_s` (includes FE assembly) |
| inner work | `rec.times.inner_time_total_s`, `inner_time_per_outer_mean_s`, `inner_time_per_inner_iteration_mean_s` |
| inner sub-iterations | `rec.counts.inner_iterations_total`, `inner_iterations_per_outer_mean`; `rec.stopping.n_inner_not_converged` (enforced as SOLVER_FAILURE) |
| ω₁, ω₂ | `rec.omega` (ω₁–ω₃), `rec.omega1_native` |
| spectral gap | `rec.stopping.gap12_pct` |
| M_nd | `rec.stopping.final_grayness` = mean 4ρ(1−ρ) (M_nd as a fraction) |
| gray fraction | computed exactly from the final design `rec.x` saved in `benchmark_records.mat` (mean(0.1 < ρ < 0.9), as in `mig_compare`); not a separate field |
| terminal criterion | `rec.status`, `rec.status_note`, `rec.stopping.{stop_rule, converged, final_l2_density_change, eps_l2, eps_rms, final_move_limit, final_move_mean, outer_iterations, max_outer}` |
| topology output | `rec.x` plus `confbench_topology_images` (one image per method per mesh), called by `performance_comparison.m` |
| total-time scaling | `confbench_scaling_fit` total-time power law (existing) |
| per-outer scaling | `confbench_scaling_fit → scaling.per_outer`: power-law fits of total/outer, outer-excluding-inner/outer, eig/outer, inner/outer and per-inner-iteration time; the detailed CSV and notes carry the "Cost per outer iteration" table (export flags verified true) |

**Limitation (disclosed, not blocking).** Records keep tOuter aggregates (total, mean, median), not the per-iteration vector. Both required scaling analyses are supported by the aggregates.
