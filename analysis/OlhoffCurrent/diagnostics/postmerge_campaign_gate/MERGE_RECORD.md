# MERGE_RECORD

```
OLHOFF_MIGRATION_MERGE_PASS
```

## Part A — state before any change (attempt 2)

| | normal checkout `/Users/piotrek/Programming/topOpt4freqMax` | migration worktree `…/topOpt4freqMax-migration-253069` |
|---|---|---|
| branch | `benchmark-methodology-r2` | `migration/olhoffcurrent-upstream-253069` |
| HEAD | `013cc48451d33bed61c5c4eea174bbd898d548a2` | `9b30ec45b038fb36e7cf20d57679b71cfd099fb3` |
| tracked changes | none | none |
| untracked | 9 directories (listed below) | none |
| relation to origin | 0 behind / 1 ahead of `origin/benchmark-methodology-r2` (`bba45e7`) | — |

The 9 untracked directories, all under `analysis/OlhoffCurrent/diagnostics/`:
- the 8 pre-existing user studies: `c480_socp_causal_run`, `filtered_subproblem_integrability_audit`, `frozen_inner_solver_study`, `frozen_problem25_reference`, `gray_kkt_forensic_audit`, `nine_mesh_campaign_audit`, `scientific_delta_olhoff_migration`, `three_rung_canary_preflight`;
- `postmerge_campaign_gate` (attempt 1).

## Repair commit (worktree)

| | |
|---|---|
| commit | `b21483b158f58e05e7b56957f2fbe8e1d2891395` "Harden historical source provenance verification" |
| parent | `9b30ec45b038fb36e7cf20d57679b71cfd099fb3` (not amended) |
| tree | `64fc72d80ee7e505825febc4ca540e0d6e3339ef` |
| changed files | 30 (`provenance_gate_hardening/REPORT.md` §6) |
| worktree after commit | clean |
| superseded | `adf86a3f8126…`, the first repair; amended into `b21483b` before any merge (ADDENDUM_1) |

## Step 3 — merge

| | |
|---|---|
| target HEAD re-verified immediately before merge | `013cc48451d33bed61c5c4eea174bbd898d548a2`, no tracked changes |
| merged | `migration/olhoffcurrent-upstream-253069` (`9b30ec4` + `b21483b`) |
| strategy | `git merge --ff-only` (fast-forward was possible; MIGRATION_HANDOFF sets no other requirement) |
| merge commit | none (fast-forward) |
| conflicts | none |
| post-merge HEAD | `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| changed-file set `013cc48..HEAD` | 146 files (+17467 / −537): 121 from `9b30ec4` and 30 from `b21483b` (some paths appear in both) |
| status after merge | the same 9 untracked directories; nothing else |
| origin | not touched: 0 behind / **3 ahead** (not pushed) |
