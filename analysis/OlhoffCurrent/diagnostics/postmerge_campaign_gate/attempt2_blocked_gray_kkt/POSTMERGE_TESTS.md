# POSTMERGE_TESTS — normal checkout at `b21483b`

```
POSTMERGE_TEST_SUITE_FAIL
```

**One new failure, caused by the merge:** the untracked local study `gray_kkt_forensic_audit` newly fails the finalization gate. It is not a code regression, but under the owner's rule ("only previously demonstrated pre-existing failures may remain; any new regression blocks") it cannot be waived. Every other result is 0 failures or a verified pre-existing failure.

**Setup.** MATLAB 25.2.0.2998904 (R2025b), single thread. The gate block ran alone, since two suites perturb `+impl`. The per-block logs are in `logs/POST_*.log.txt` and the results in `evidence/POST_*.json`.

**Baseline.** The same blocks were run **before** the merge in this same checkout, with the same untracked content (`logs/BASE_*`).

## Suites (MIGRATION_HANDOFF list, plus upstream suites, self-test and harness)

| block | suite | failures | classification |
|---|---|---|---|
| gates | `test_path_isolation` | 0 | — |
| gates | `test_currentness` | 0 | — |
| gates | `test_source_integrity` | 0 | — (the pre-merge baseline had 4, A–D "→ CURRENT", which the promotion resolved) |
| gates | `test_evidence_retention` | 0 | — |
| gates | `test_finalization_gate` | **2** | H `move_activity_400`: **VERIFIED PRE-EXISTING**. I: see below |
| presets | `test_preset_identity` | 0 | — |
| presets | `test_pedersen_adaptive_units` | 0 | — |
| presets | `test_cost_reporting` | 0 | — |
| solves160 | `test_preset_equivalence` (historical, 160×20) | 0 | — |
| solves160 | `test_named_preset_reproduction('pedersen')` (160×20) | 0 | — |
| solves160 | `test_named_preset_reproduction('stageExhaustion')` (160×20) | 0 | — |
| upstream (253069 archive, SHA-256 `f9112403…`, against `+impl`) | `test_config`, `test_mass`, `test_modules`, `test_preset_resolution_unchanged`, `test_stage_exhaustion`, `test_outer_timing` | 0 each | — |
| selftest | `confbench_selftest` | 1 (T2) | **VERIFIED PRE-EXISTING**: pristine `013cc48` and this checkout's pre-merge baseline give the same message |
| harness | `confbench_preflight` 160×20 (no solve) | 0 of 38 | — |
| harness | Olhoff cap-3 smoke + manifest + export | pass | CAP_HIT, production preset recorded, per-outer columns, formulation labels |

Inside `test_finalization_gate`, everything else passes:
- A–G and D2;
- all 39 J probes, plus "all 39 ran" and "repository untouched";
- H for `beta_transition_mechanism` and `two_branch_controller_validation`;
- both H2 checks (the seven lines at `bba45e7`/`1438aa3`; statuses separate).

## `test_finalization_gate` I — per-study classification

Source: `evidence/POST_studies.json` compared with `logs/BASE_studies.json`.

| study | tracked | before merge | after merge | classification |
|---|---|---|---|---|
| controller_architecture_offline, move_activity_400, move_ladder_necessity, three_rung_architecture, three_rung_promotion_closure, three_rung_promotion_validation_retry1, two_rung_architecture | yes | FAIL | FAIL, identical G1–G5 | **VERIFIED PRE-EXISTING** (the seven, proven on pristine `013cc48` in attempt 1) |
| frozen_inner_solver_study, frozen_problem25_reference, scientific_delta_olhoff_migration | **no** | FAIL (`10111`) | FAIL, identical G1–G5 | **VERIFIED PRE-EXISTING**: untracked local studies, failing identically in this checkout before the merge (MIGRATION_HANDOFF item 4) |
| **gray_kkt_forensic_audit** | **no** | **PASS** | **FAIL (G2)** | **NEW, merge-induced** (below) |
| the 8 legacy ledger studies | yes | FAIL | FAIL, identical | legacy ledger (not counted by I) |
| every other study | — | PASS | PASS | — |

G6 (`CURRENT_SOURCE_HASH_VERIFIED`) passes for all 30 studies.

## The new failure: `gray_kkt_forensic_audit`

- **What.** The study is untracked, a local user study last modified 2026-09-12. Its `EVIDENCE.json` declares `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` as a **REQUIRED** artifact with SHA-256 `431e7b30…`, the pre-migration manifest (tree `edbfe47e…`). The merge legitimately replaced that file (now `aee44aa9…`, tree `4ba9a3ae…`). `olhoffcurrent_evidence_gate` therefore reports `REQUIRED_HASH_MISMATCH`, so G2 and the study fail. The other 57 required artifacts match.
- **Not caused by the gate repair.** The evidence gate code is unchanged by both commits, so the same failure would occur with `9b30ec4`'s gate. It stayed invisible in the migration's own tests because the worktree does not contain the untracked studies.
- **Not an implementation or committed-repository regression.** `+impl` is byte-identical to 253069, G6 passes, both anchors reproduce, and a clean clone of `b21483b` does not contain this study.
- **Why it is not tolerated here.** It was not demonstrated before the merge. The historical-source exception cannot apply either: the study's evidence is not committed, so there is no freeze commit, and the exception covers `FINAL_SHA256.txt` lines, not EVIDENCE.json artifacts. Editing the study's evidence would rewrite user evidence, and changing the gate again needs a new authorization.

Its classification under the four categories, as argued above:

| category | applies? | reason |
|---|---|---|
| VERIFIED PRE-EXISTING | no | it passed before the merge |
| TEST DEFECT | no | the gate reports correctly |
| ENVIRONMENTAL | arguable only | it depends on uncommitted local content |
| **NEW REGRESSION** | **yes** | the strict reading, applied here |

The owner decides whether to reclassify it (CAMPAIGN_AUTHORIZATION.md).
