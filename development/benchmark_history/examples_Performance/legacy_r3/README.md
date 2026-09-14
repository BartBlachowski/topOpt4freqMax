# Superseded benchmark driver (R3 / "final campaign" era)

`performance_comparison_r3.m` in this folder is the **previous**
`examples/Performance/performance_comparison.m`, preserved verbatim as
historical evidence.  It is **not** the conference benchmark driver and must
not be used to produce conference numbers.

Why it was replaced:

- Its Olhoff column dispatched
  `analysis/olhoff_stabilization_audit/run_stabilization_case.m` --- the
  fixed-1600-outer-iteration S1 profile, which is **superseded**.  See
  `analysis/OLHOFF_IMPLEMENTATION_STATUS.md`.
- The active mesh list was read from a manifest
  (`campaignManifest.mesh_sequence`) behind an `isFinalCampaign` branch, so the
  set of meshes that would actually run was not visible or editable in the
  script.
- Behaviour depended on environment variables (`TOPOPT_BENCHMARK_MODE`,
  `TOPOPT_BENCHMARK_MESHES`) that survive for a whole MATLAB session.
- It forced three architecturally different methods into one generic iteration
  count, and reported a `MaxRAM_MB` column measured by a 10 Hz `ps` sampler
  that ran *inside* the timed optimization loop.

The replacement is `examples/Performance/performance_comparison.m` with helpers
in `examples/Performance/conference_bench/`.  Its artifacts go to
`examples/Performance/conference_benchmark/`, so nothing here or in
`examples/Performance/final_campaign/` is overwritten.

The helper files this script used --- `final_campaign_config.m`,
`final_campaign_run_case.m`, `final_campaign_preflight.m`,
`performance_benchmark_profile.m`, `olhoff_preflight.m` and the rest --- are
still in `examples/Performance/` and are untouched; only the driver moved.
