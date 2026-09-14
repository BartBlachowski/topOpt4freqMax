BOTTOM LINE

**The definitive nine-mesh campaign remains blocked, and for a single reason.**
- **What.** The untracked local study `gray_kkt_forensic_audit` pins the pre-migration `SOURCE_MANIFEST.json` as required evidence. The merge replaced that manifest, so the study now fails the finalization gate.
- **Why it blocks.** Under your rule, only failures demonstrated before the merge may remain.
- **What it is not.** It is not a code, science or committed-repository regression. It is one owner decision away from resolution (CAMPAIGN_AUTHORIZATION.md).

**Everything else passed:**
- **Previous attempt.** The first authorization attempt correctly failed (`attempt1_blocked/`, preserved byte-identical).
- **Gate hardened.** One focused repair commit, `b21483b`.
- **Probes rerun.** From committed code: 39/39 as preregistered. The original rule is fail-open on 24 of them, and my own superseded first repair on 6; attempt 1's original P5/P9/P10 attacks now fail.
- **Legitimate history.** The seven superseded lines still verify.
- **Reconsidered from Step 1:**
  - amendment and migration reviews pass;
  - the fast-forward merge to `b21483b` passes;
  - `+impl` is byte-identical to upstream 253069;
  - both 160×20 anchors reproduce bitwise;
  - the production identity is frozen;
  - all nine configurations resolve consistently;
  - telemetry is ready;
  - the repository is intact.

Nothing above 160×20 was solved.

```
PROVENANCE_GATE_HARDENING_PASS
COMMITTED_PROVENANCE_SELF_CONTAINED_PASS
GATE_REPAIR_SCOPE_PASS

PREREGISTRATION_AMENDMENT_REVIEW_PASS
FINALIZATION_GATE_PROVENANCE_LOGIC_PASS
MIGRATION_COMMIT_REVIEW_PASS
OLHOFF_MIGRATION_MERGE_PASS
POSTMERGE_IMPLEMENTATION_IDENTITY_PASS
POSTMERGE_TEST_SUITE_FAIL
POSTMERGE_HISTORICAL_S160_PASS
POSTMERGE_PEDERSEN_S160_PASS
NINE_MESH_CONFIG_PREVIEW_PASS

OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED
```

> Blocker: `analysis/OlhoffCurrent/diagnostics/gray_kkt_forensic_audit` (untracked) declares `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` @ `431e7b30…` as REQUIRED evidence. After the merge it fails G2 (`REQUIRED_HASH_MISMATCH`), a new failure in `test_finalization_gate` I that passed before the merge.

## Answers

1. **What was wrong with Amendment A2?** The `SUPERSEDED_PRODUCTION_SOURCE` exception of `9b30ec4` was not fail-closed:
   - it aggregated digests per path, so a later genuine line rescued a fabricated one (P5);
   - it hashed `git show | shasum`, whose exit status hid missing objects, so an empty digest passed (P9);
   - it trusted the editable manifest as "current source" (P10);
   - it resolved source lines study-local first, and searched any ancestor commit.
2. **How was P5 fixed?** Every digested line gets its own verdict, checked against its own digest; duplicates are each validated and reported.
3. **How was P9 fixed?** Existence is proved with `git cat-file -e` and type `blob` before `cat-file blob > file`, each exit status checked. There is no pipeline. A missing object is `HISTORICAL_PATH_ABSENT_AT_FREEZE_COMMIT`, while a genuinely empty file still validates (P19).
4. **How was P10 fixed?** A new gate G6 runs for every study:
   - HEAD's `+impl` blobs are compared with the working tree by raw SHA-256, never through the index;
   - no extra files and no symlinks;
   - the manifest must equal HEAD's blob, rows and tree, and the PROVENANCE.md tree row must match;
   - git runs with `--no-replace-objects` and a scrubbed `GIT_*` environment (P36/P37).
5. **Does every line validate independently?** Yes.
6. **Is existence checked before hashing?** Yes.
7. **Is current source tied to HEAD?** Yes, to committed HEAD objects; replace refs and `GIT_DIR` redirection are neutralized.
8. **Can an edited `+impl` with a regenerated manifest still pass?** No: P10, P26, P28 (skip-worktree), P36 and P37 all fail.
9. **Can a fabricated duplicate line still pass?** No (P5, P6).
10. **Can a nonexistent historical file validate as empty?** No (P9, P18).
11. **Do the seven legitimate superseded hashes still pass?** Yes. Same paths, exact digests, freeze commit `bba45e7`, source commit `1438aa3`, declared tree `edbfe47e…` equal to the freeze tree — in the worktree, in a clean clone of the tip, and in the merged checkout (H2).
12. **Is the historical/current distinction explicit?** Yes: `HISTORICAL_SOURCE_HASH_VERIFIED` and `CURRENT_SOURCE_HASH_VERIFIED` are separate statuses with per-line records, and P8/P10/P36/P37 show them diverging.
13. **Is committed migration provenance self-contained?** Yes. PROVENANCE.md and .json cite only committed evidence and commit identities, and the three local folders are marked supplementary. The disclosed evidence strings in the preset file were not edited.
14. **What repair commit was created?** `b21483b158f58e05e7b56957f2fbe8e1d2891395` "Harden historical source provenance verification": parent `9b30ec4`, tree `64fc72d8…`, 30 files. The first attempt `adf86a3` was amended into it before any merge (ADDENDUM_1).
15. **Did the repair touch `+impl` or scientific config?** No.
16. **Was A2 accepted on re-review?** Yes (AMENDMENT_REVIEW.md).
17. **Was the branch merged?** Yes, by fast-forward into `benchmark-methodology-r2`, with no conflicts.
18. **Final merged HEAD?** `b21483b158f58e05e7b56957f2fbe8e1d2891395`.
19. **Does `+impl` still match upstream 253069 byte for byte?** Yes: 79/79 files, tree `4ba9a3ae…`.
20. **Which pre-existing failures remain?**
    - `test_finalization_gate` H `move_activity_400`;
    - its I-set: the tracked seven, plus three untracked studies failing identically before the merge (`frozen_inner_solver_study`, `frozen_problem25_reference`, `scientific_delta_olhoff_migration`);
    - `confbench_selftest` T2.
21. **Were new failures introduced?** One: `gray_kkt_forensic_audit` (untracked, G2), caused by the merge replacing the manifest it pins. It is not a code regression, but it is new.
22. **Did the historical S160 anchor reproduce?** Yes:
    - bitwise equal to the migration run (including cfg) and the upstream run;
    - equal to the frozen campaign record in ρ, ω₁, volume, 91 and 2241;
    - ω₁ 169.495227021538, and the 81-row hash `28756d22…` reconstructed exactly.
23. **Did the Pedersen S160 anchor reproduce?** Yes:
    - identical to committed `S160x20` at the established level (only `stop.rule`/`verbose`/`name` config rows differ);
    - bitwise equal to the migration and upstream runs;
    - 121/2369, ω₁ 169.210576386275.
24. **Frozen production preset and config hash?** `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, 160×20 hash `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4`.
25. **Do all nine future configs resolve correctly?** Yes:
    - only `nelx`, `nely`, the rule-derived `stop.tolerance` and `runtime.name` vary;
    - R = 0.06·b is fixed;
    - Pedersen + linear mass, adaptive box, design-change stop, maxOuter 400;
    - no stage exhaustion, eq. (4b), p continuation, projection or SOCP;
    - all nine hashes equal PROVENANCE.json's recorded ones.
26. **Is telemetry ready?** Yes: total and per-outer time, tOuter aggregates, eig/outer, inner time and sub-iterations, ω₁/ω₂/gap, M_nd, final design (for gray fraction), terminal criterion, topology images.
27. **Was anything above 160×20 solved?** No.
28. **Is the nine-mesh campaign authorized?** No: **`OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED`**, for the `gray_kkt_forensic_audit` blocker only.
29. **What HEAD and config identity must the campaign use?** HEAD `b21483b158f58e05e7b56957f2fbe8e1d2891395`, `+impl` tree `4ba9a3ae…`, preset `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, and the nine hashes in `CAMPAIGN_IDENTITY.json` (160×20 `b1a5744d…` … 800×100 `f9138743…`).
30. **What is the only authorized next action?** Your decision on `gray_kkt_forensic_audit` (CAMPAIGN_AUTHORIZATION.md options 1–3). No scientific run is authorized until then, and the campaign itself stays a separate task.

## Files

| file | what it holds |
|---|---|
| GATE_PREREGISTRATION.md | criteria and the preservation of attempt 1 |
| AMENDMENT_REVIEW.md, FINALIZATION_GATE_REVIEW.md, MIGRATION_COMMIT_REVIEW.md | Steps 1–2 |
| MERGE_RECORD.md | Part A, the commit and the merge |
| POSTMERGE_IDENTITY.md, POSTMERGE_TESTS.md | Step 4 |
| HISTORICAL_S160.md, PEDERSEN_S160.md | Steps 5A/5B |
| CAMPAIGN_IDENTITY.json, NINE_MESH_CONFIG_PREVIEW.md, NINE_MESH_CONFIGS.json | Step 5C |
| TELEMETRY_READINESS.md, REPOSITORY_INTEGRITY.md | Steps 5D/5E |
| CAMPAIGN_AUTHORIZATION.md | the decision and how to lift the block |
| METRICS.json, EVIDENCE.json, FINAL_SHA256.txt | machine-readable record |
| `evidence/`, `logs/`, `scripts/` | raw evidence, logs, scripts |
| repair audit | `diagnostics/provenance_gate_hardening/` (committed) |
| attempt 1 | `attempt1_blocked/` |
