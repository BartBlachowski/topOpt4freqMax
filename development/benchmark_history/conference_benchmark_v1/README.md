# `examples/Performance/conference_benchmark/`

Artifacts produced by `examples/Performance/performance_comparison.m`, the
**current** conference benchmark driver. One subdirectory per run, named by the
run label the driver derives from the configuration:

| subdirectory | what it is |
|---|---|
| `smoke/` | mechanics-only runs: sub-160×20 meshes and/or a truncated outer budget. `scientific_evidence = false`, `performance_campaign = false`. **Not citable as a result.** |
| `preflight_160x20/` | the single-mesh scientific preflight at the project's mesh-resolution floor. Scientific evidence; not a campaign, so no scaling exponent is fitted. |
| `campaign_9mesh/` | the full nine-resolution campaign, when it is run. |

Nothing here overwrites the earlier evidence in `examples/Performance/` or in
`examples/Performance/final_campaign/`; the preflight refuses to start if the
output directory would land on either.

Each run directory contains:

```
conference_performance_table.csv      the primary, method-native table
conference_performance_table.tex      the same table for the slide
conference_performance_detailed.csv   explicit method-specific field names
benchmark_results.json                every record, full precision
benchmark_manifest.json               exactly what ran, with hashes
timing_schema.json                    what each count and time means
BENCHMARK_NOTES.md                    the caveats, ready to paste
benchmark_records.mat                 the raw records, including design fields
```

**There is no memory column anywhere.** Reliable, method-independent
peak-memory measurement was not available in the MATLAB environment; memory was
omitted rather than reported with inconsistent semantics.

Read `BENCHMARK_NOTES.md` before quoting any number: the count and time columns
are method-native and are not mathematically identical across methods.
