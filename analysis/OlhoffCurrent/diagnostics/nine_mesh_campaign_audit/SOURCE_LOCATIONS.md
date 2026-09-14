# Load-bearing source locations

- [Production wrapper selects old preset](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/olhoffcurrent_config.m:70)
- [Actual old policy defaults](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/+impl/architecture/+olh/+presets/duOlhoffFrozenM4.m:56)
- [Frozen A/B detector](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/+impl/architecture/+olh/+move/exhaustion.m:1)
- [Solver switches and history retention](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m:121)
- [Simple-J warning and unchanged continuation](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m:223)
- [Wrapper discards history, retains aggregates](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/olhoffcurrent_run.m:107)
- [Benchmark copies aggregate record](/Users/piotrek/Programming/topOpt4freqMax/examples/Performance/conference_bench/confbench_run_case.m:113)
- [JSON removes x/config/telemetry](/Users/piotrek/Programming/topOpt4freqMax/examples/Performance/conference_bench/confbench_export.m:180)
- [Prior promotion explicitly blocked](/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics/three_rung_promotion_closure/REPORT.md:1)

Campaign JSON selection: `runs[*].method_key == "olhoff"`. Raw MAT selection: `records[*].method_key == "olhoff"`; configuration is `effective_config`, density is `x`. Manifest configuration is `method_configurations[*].canonical`. Exact measured configurations and hashes are retained in this audit.
