# REPOSITORY_INTEGRITY — Step 5E

**Result: PASS.**

## Final state

| | normal checkout | migration worktree |
|---|---|---|
| branch | `benchmark-methodology-r2` | `migration/olhoffcurrent-upstream-253069` |
| HEAD | `b21483b158f58e05e7b56957f2fbe8e1d2891395` | `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| tracked changes | none | none |
| untracked | the 8 pre-existing user studies + `postmerge_campaign_gate` (this record, with attempt 1 in `attempt1_blocked/`) | none |
| origin | 0 behind / 3 ahead (not pushed) | — |
| replace refs | 0 | — |

**Untracked-state comparison.** Part A had the same 9 directories.
- **7 user studies not modified:** 0 files newer than the session start in each of `c480_socp_causal_run`, `filtered_subproblem_integrability_audit`, `frozen_inner_solver_study`, `frozen_problem25_reference`, `nine_mesh_campaign_audit`, `scientific_delta_olhoff_migration` and `three_rung_canary_preflight`.
- **`gray_kkt_forensic_audit` was modified on 2026-09-14, by the owner's decision (option 2).** Its record is EVIDENCE_AMENDMENT_1 and RESOLUTION_GRAY_KKT.md.
  - Changed: `EVIDENCE.json`, one artifact (class and annotations), plus an amendment record appended; `FINAL_SHA256.txt`, the EVIDENCE.json line plus 3 appended lines.
  - Added: `EVIDENCE_AMENDMENT_1.md`, and `evidence_record_history/` holding byte-identical originals.
  - Nothing else in that study changed.

**Git-ignored evidence added.** `analysis/OlhoffCurrent/evidence/postmerge_campaign_gate/ANCHOR_{HISTORICAL,PEDERSEN}_S160.mat`.

**Throwaway material (outside both checkouts).** Probe clones in the system temporary directory, which the harness removes, and the upstream archive plus scripts in the session scratchpad.

## Scope lock

| item | status |
|---|---|
| Proposed / Yuksel files | unchanged (0 paths in `013cc48..HEAD`) |
| evaluator | unchanged |
| global filter policy, other methods' radii | unchanged; every path is under `analysis/OlhoffCurrent/**` or Olhoff-specific `examples/Performance/**` |
| Phase 6 | not entered |
| p-continuation logging defect D1 | still separate: `olhoffSolve.m` byte-identical to upstream 253069 |
| C480 SOCP work | not incorporated (0 paths; the study stays untracked and unmodified) |
| `+impl`, presets, material laws, filter, MMA, controller, multiplicity | no change beyond the reviewed `9b30ec4` promotion; the repair commit touched none of them |
| solves | only 160×20: the two anchors, the three 160×20 reproduction suites, the capped smoke and suite wiring solves. The nine-mesh preview resolved configurations without solving. |
| **anything above 160×20 solved** | **no** |
