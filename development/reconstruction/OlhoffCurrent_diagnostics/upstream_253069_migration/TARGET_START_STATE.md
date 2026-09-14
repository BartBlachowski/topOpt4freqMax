# TARGET_START_STATE — Part 0

Captured 2026-09-13T19:12:42+0200, before any file was modified. Machine-readable: `evidence/target_start_state.json`.

| item | value |
|---|---|
| repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| branch | `benchmark-methodology-r2` |
| HEAD | `013cc48451d33bed61c5c4eea174bbd898d548a2` ("Nine test passed") |
| tracked changes | **none** |
| untracked | 8 directories under `analysis/OlhoffCurrent/diagnostics/`: `c480_socp_causal_run`, `filtered_subproblem_integrability_audit`, `frozen_inner_solver_study`, `frozen_problem25_reference`, `gray_kkt_forensic_audit`, `nine_mesh_campaign_audit`, `scientific_delta_olhoff_migration`, `three_rung_canary_preflight` |
| other activity | MATLAB desktop session attached (VS Code MATLAB extension); several diagnostics modified earlier the same day |
| `+impl` (live, recomputed independently of MATLAB) | 75 files, tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`; one `.DS_Store` artifact skipped |
| `SOURCE_MANIFEST.json` | 75 files, same tree, generated 2026-09-08; **matches live tree** |
| git tree of `+impl` at HEAD | `ef4cb5f7cb771bf35b3bc04d4cf244af57afa5fa` |
| `PROVENANCE.md` | stale: claimed 74 files, one adaptation (actual: 75 files, seven differences from 695f03b since commit `1438aa3`) |
| `PROVENANCE.json` | source `695f03b` on `architecture/canonical-config`; production preset `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |
| presets (target `olh.presets.list`) | 10: `duOlhoffFrozenM4`, `duOlhoffMatureM4`, `restorationLadderGuard`, `noDescentFixedMove`, `pContinuationCoupled`, `pContinuationDecoupled`, `pMassCompatible`, `projectionIdentity`, `projected`, `legacyBinaryDiagonal` |
| OlhoffCurrent preset | one unnamed production entry `olhoffcurrent_preset()` |
| configuration schema | **81 rows** |

## Historical configuration hashes (81-row schema), reproduced on the untouched tree

All 14 recorded hashes were re-resolved on the pre-migration worktree (identical to HEAD) and reproduced exactly (`evidence/config_transition.json`, `preHashEqualsRecorded`).

| configuration | mesh | recorded hash | recorded in |
|---|---|---|---|
| production β-stall (campaign 2026-09-11) | 160×20 | `28756d22aacb59726be9f37583deca89fcfcecc93f9b867d46a49223ed1db697` | campaign_9mesh_r2 |
| ″ | 240×30 | `0856b13d02e3f1065c0ab145213bb57afb1b98570e64c9b023851f696141196c` | ″ |
| ″ | 320×40 | `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` | ″ |
| ″ | 400×50 | `cec3cd6b89ad67bc5386093a0caaf1b05dc172ddb75f488c52088c69d55d0c20` | ″ |
| ″ | 480×60 | `a49417d0571d3c2406d030cbab1fda58a1dad8d334991c5aaf22fa17f1d7f601` | ″ |
| ″ | 560×70 | `3478f34841ce743457aac7d4ba1755f2989453a9f0d79ccfc4f961f89fb7fd64` | ″ |
| ″ | 640×80 | `e5d868133d52a510797e6847b9f113ab3d5424849f8473f23c7827485bff300b` | ″ |
| ″ | 720×90 | `7efe1ee908be54cefd2b9e63f911e337809b92870056c7b1854b46bd41a91078` | ″ |
| ″ | 800×100 | `9321858983a3d7d33ca1a7b5bed9136fb7967e637b7ac5b40dadfad5ffdd4132` | ″ |
| three-rung stage exhaustion | 320×40 | `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab` | three_rung_promotion_validation_retry1 |
| ″ | 480×60 | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` | three_rung_canary_preflight (untracked) |
| ″ | 800×100 | `7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe` | ″ |
| four-rung stage exhaustion | 160×20 | `31d2ef382746a942a4036d07dd6a1012742432cba0497d2de5ec24b51d2b5904` | two_branch_controller_validation |
| ″ | 320×40 | `2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4` | ″ |

## Isolation decision

Because of the unrelated untracked work and the attached MATLAB session, the migration ran in a **dedicated worktree**:

- path `/Users/piotrek/Programming/topOpt4freqMax-migration-253069`;
- new branch `migration/olhoffcurrent-upstream-253069` at `013cc48`.

The primary checkout was not switched, cleaned, stashed or edited. Git-ignored historical evidence the tests need (`benchmark_records.mat`, three `evidence/<study>` directories) is symlinked from the primary checkout into the worktree's ignored locations, read-only. After the work, the primary checkout's `git status` is unchanged.
