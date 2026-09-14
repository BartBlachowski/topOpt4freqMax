# MERGE_RECORD

```
OLHOFF_MIGRATION_MERGE_FAIL        (NOT EXECUTED — stopped before merge at Step 1)
```

No merge, cherry-pick, rebase, checkout, stash or clean was run in either checkout.

## Step 0 — state recorded before any action (2026-09-13)

| | normal checkout `/Users/piotrek/Programming/topOpt4freqMax` | migration worktree `/Users/piotrek/Programming/topOpt4freqMax-migration-253069` |
|---|---|---|
| branch | `benchmark-methodology-r2` | `migration/olhoffcurrent-upstream-253069` |
| HEAD | `013cc48451d33bed61c5c4eea174bbd898d548a2` | `9b30ec45b038fb36e7cf20d57679b71cfd099fb3` |
| tracked modifications | none (staged and unstaged) | none |
| untracked | 8 directories (listed below) | none |
| `+impl` files / tree hash | 75 / `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` | 79 / `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` |
| `SOURCE_MANIFEST.json` SHA-256 | `431e7b3084507cd8b53862847b828cfd9f8948172e349bbae0c72749e8134aeb` | `aee44aa9172a8105314bbb9475e28bed8a53d4d154824a8748abf64a208b1836` |

The normal checkout's eight untracked directories, all under `analysis/OlhoffCurrent/diagnostics/`:

- `c480_socp_causal_run`
- `filtered_subproblem_integrability_audit`
- `frozen_inner_solver_study`
- `frozen_problem25_reference`
- `gray_kkt_forensic_audit`
- `nine_mesh_campaign_audit`
- `scientific_delta_olhoff_migration`
- `three_rung_canary_preflight`

Tree hashes were recomputed from git blobs with the `olhoffcurrent_source_manifest` algorithm, and each equals its manifest's `tree_sha256`. The normal checkout's `analysis/OlhoffCurrent` working tree equals HEAD. The git-ignored raw evidence copy `analysis/OlhoffCurrent/evidence/upstream_253069_migration/` (43 MB) was already present from the migration task.

**The brief's premise was wrong.** It said the normal checkout was already on the migration branch at `9b30ec4`. It is actually on `benchmark-methodology-r2` @ `013cc48`, and the migration lives in a separate worktree, as MIGRATION_HANDOFF.md states. That state was safe to handle, and since the gate stopped before merge, neither checkout needed to change.

## What the merge would have been (for the next attempt; not performed)

| | |
|---|---|
| target branch | `benchmark-methodology-r2` (MIGRATION_HANDOFF step 2) |
| pre-merge target HEAD | `013cc48451d33bed61c5c4eea174bbd898d548a2` |
| relationship | target is the migration commit's sole parent, so a fast-forward is possible; no conflicts are possible |
| overlap with untracked/ignored files | none of the 121 changed paths is among the eight untracked directories |
