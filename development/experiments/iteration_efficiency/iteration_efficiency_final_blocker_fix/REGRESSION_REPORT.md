# Regression and collateral-change audit

## Files modified

**Tracked (2)** — 24 insertions, 2 deletions:

| file | change |
|---|---|
| `tools/Matlab/run_topopt_from_json.m` | opt-in `stage1_budget_independent`; the historical clamp remains the default |
| `analysis/three_method_parametric_study/study_base_config.m` | passes the flag through, default `false` |

**Untracked harness (15)** — all inside `analysis/iteration_efficiency_final/`:

`+iefinal/`: `run.m`, `run_timing_firewall.m`, `run_trajectory.m`, `config.m`, `fit_scaling_table.m`, `generate_scaling_outputs.m`, `synthetic_scaling_validation.m`, `build_rows.m`, `validate_results.m`, `render_topologies.m`, plus new `empty_result_row.m`, `build_error_rows.m`, `classify_cell_failure.m`; and `RESULT_SCHEMA.json`, `run_final_integration_tests.m`.

`SHA256SUMS.txt` regenerated (39 entries, all verify). `analysis/iteration_efficiency_final_audit/` was **not** modified (14/14 still verify).

- Native optimizer mathematics changed? **NO.**
- Frozen methodology changed? **NO.**

## Collateral-change analysis

The only shared code touched is `run_topopt_from_json.m`, which drives Proposed and Yuksel. Seven callers pass `stage1_max_iters`; one (`run_stage_a.m`, a completed frozen study) passes `1000` with `max_iters = 300` and *would* have changed behaviour had the clamp simply been deleted. Making the change opt-in leaves it, and every other caller, byte-identical. Confirmed by direct comparison of the pre-edit and post-edit drivers at 160×20: both methods produce bit-identical designs (max abs difference **0**), identical ω, identical iteration and stage counts.

`RESULT_SCHEMA.json` gained `error_identifier` and `error_message`. Every row template emits them, so all rows — successful and failed — remain schema-valid under `additionalProperties: false`.

## Latent defect found and fixed while raising the mesh floor

Raising the smoke mesh to 160×20 exposed a pre-existing fault: the shared renderer returns live figure/axes/image handles per cell and then closes the figure, so `run_smoke`'s summary failed to serialise with `jsonencode: Invalid or deleted object`. It had been masked at 16×2, where every cell was unavailable and carried no handles.

Production was never affected — `localProduction` discards the renderer's return value — so this was not a production blocker. Fixed on the harness side (`render_topologies` keeps only serialisable per-cell provenance); the frozen renderer is untouched.

## Test suite — 14 / 14 pass

| test | result |
|---|---|
| testManifestAndPreflight | PASS |
| testCandidateAnchors | PASS |
| testTopologyReferencePersistence | PASS |
| testAccounting | PASS |
| testResultSchema | PASS |
| testTimingFirewall | PASS |
| testSelectorEvidence | PASS |
| testDoubleStorageEvidence | PASS |
| testScalingAndRendering | PASS |
| testStaleAndProductionLocks | PASS |
| testOutputIsolation | PASS |
| **testB1YukselTimingWorkIdentity** | PASS |
| **testB2CommonSupport** | PASS |
| **testB3FailureContainment** | PASS |

## Required re-verifications

| item | evidence | result |
|---|---|---|
| Candidate-C anchors | 480×60 k=194 → ordinals `[1 7 13]`, schedule `[3 6 12 24]`, 3 escalations; solid 160×20 block → ordinal 1 | ✅ |
| Lossless Olhoff double trajectory identity | `reference_length_replay` — dtype `double`, stored terminal state equals optimizer state, exact-count binary and hard-gate identity at checkpoints `[80 252 453 552 2100 3200]` | ✅ |
| `b_ref = 2100` | reference replay over frozen 96×12 H3200 evidence | ✅ |
| `B_meas = 3200` | `min(max(3200, 2100+99), 3200)` | ✅ |
| `k_enter`/`k_cert` at P=100 | q=.98 → **229/328**, q=.99 → **309/408**, q=.995 → **453/552** | ✅ |
| P = 50 / 200 sensitivity | all nine pairs reproduced | ✅ |
| Proposed accounting | `native_iterations = k` | ✅ |
| Yuksel Stage-1/Stage-2/total | `s1`, `k`, `s1 + k` | ✅ |
| Olhoff-LP accounting | outer, LP calls, failed calls, genuine backend iterations; `nInner` absent from all rows | ✅ |
| Olhoff-MMA accounting | total/mean/median/p95/max inner, cap and converged counts | ✅ |
| LP / MMA / both selector | three smokes at 160×20, correct method sets, MMA zero LP calls | ✅ |
| Timing firewall | exclusions asserted against source; serial, 1 warm-up + 3 replays, fixed horizon | ✅ |
| Topology rendering | shared renderer; failed/unavailable cells drawn empty, never fabricated | ✅ |
| Result schema | 53 required fields; N/A semantics; route-specific rejection | ✅ |
| Output isolation | mode/selector/timestamp hierarchy, collision refused | ✅ |
| Negative controls / hashes | stale evaluator, wrong contract, stale topology, float32 policy, stale mesh list, self-authorization all rejected; production still locked | ✅ |

The `b_ref` / `B_meas` / endpoint anchors are replays of the frozen immutable Phase-2I 96×12 H=3200 capture, not new sub-floor verification runs.

## Static checks

- `git diff --check` → clean (exit 0).
- `checkcode` on all 14 modified/new files → no errors; only pre-existing style notes (stale suppression markers, alignment hints, `caxis` deprecation).
- `analysis/iteration_efficiency_final/SHA256SUMS.txt` → 39/39 verify.
- `analysis/iteration_efficiency_final_audit/SHA256SUMS.txt` → 14/14 verify (untouched).

## Mesh floor

No verification below 160×20. Harness smoke moved 16×2 → 160×20; `testCandidateAnchors` 8×2 → 160×20; `testTimingFirewall` 8×2 → 160×20; `testB1…` at 160×20; B-1 proof at 160×20 and 240×30; B-3 containment at 160×20, 240×30, 320×40. Cost is controlled by cutting iteration caps.
