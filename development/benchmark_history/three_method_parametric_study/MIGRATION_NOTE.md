# Migration note (2026-09-14)

This study was `analysis/three_method_parametric_study` until the repository cleanup.

Three of its files were still read by the current performance runner and were moved,
byte-for-byte, next to it:

| was | now |
|---|---|
| `study_base_config.m` | `examples/Performance/benchmark_profile/study_base_config.m` |
| `study_evaluate_design.m` | `examples/Performance/benchmark_profile/study_evaluate_design.m` |
| `results/profile_freeze_manifest.json` | `examples/Performance/benchmark_profile/profile_freeze_manifest.json` |

Scripts here that call `study_base_config` / `study_evaluate_design` or read the freeze
manifest therefore refer to files that are no longer in this directory. To run the study
as it ran, check out the commit before the cleanup. Full mapping:
`development/repository_cleanup/MIGRATION_MANIFEST.tsv`.
