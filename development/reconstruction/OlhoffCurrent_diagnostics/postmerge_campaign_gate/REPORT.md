BOTTOM LINE

**The definitive nine-mesh Olhoff campaign is authorized. It has not been run.**

How the gate got here:
1. **Attempt 1 correctly failed.** The provenance exception was not fail-closed (`attempt1_blocked/`).
2. **The gate was hardened** in one focused commit, `b21483b`.
3. **The adversarial probes were re-run** from committed code: 39/39 as preregistered. The original rule is fail-open on 24 of them, and a superseded first repair on 6.
4. **Authorization was reconsidered from Step 1.** Amendment and branch reviews pass; the fast-forward merge to `b21483b` passes; `+impl` is byte-identical to upstream 253069; both 160×20 anchors reproduce bitwise.
5. **The one merge-induced test failure was resolved** by the owner's decision on 2026-09-14 (`gray_kkt_forensic_audit` now marks its superseded manifest pin as historical). The re-run leaves only verified pre-existing failures (RESOLUTION_GRAY_KKT.md; the blocked state is preserved in `attempt2_blocked_gray_kkt/`).
6. **The production identity is frozen**, all nine configurations are prevalidated, telemetry is ready, and the repository is intact.

> The migrated OlhoffCurrent implementation has been reviewed, merged and independently revalidated at 160×20 for both historical and new production formulations. The exact Pedersen/adaptive production identity is frozen, all nine future configurations are prevalidated, and the definitive nine-mesh campaign is authorized.

```
PROVENANCE_GATE_HARDENING_PASS
COMMITTED_PROVENANCE_SELF_CONTAINED_PASS
GATE_REPAIR_SCOPE_PASS

PREREGISTRATION_AMENDMENT_REVIEW_PASS
FINALIZATION_GATE_PROVENANCE_LOGIC_PASS
MIGRATION_COMMIT_REVIEW_PASS
OLHOFF_MIGRATION_MERGE_PASS
POSTMERGE_IMPLEMENTATION_IDENTITY_PASS
POSTMERGE_TEST_SUITE_PASS          (after RESOLUTION_GRAY_KKT; FAIL before it, preserved)
POSTMERGE_HISTORICAL_S160_PASS
POSTMERGE_PEDERSEN_S160_PASS
NINE_MESH_CONFIG_PREVIEW_PASS

OLHOFF_NINE_MESH_CAMPAIGN_AUTHORIZED
```

## Answers

1. **What was wrong with Amendment A2?** The `SUPERSEDED_PRODUCTION_SOURCE` exception of `9b30ec4` was not fail-closed:
   - it aggregated digests per path, so a later genuine line rescued a fabricated one (P5);
   - it hashed `git show | shasum`, which hid missing objects, so an empty digest passed (P9);
   - it trusted the editable manifest as "current source" (P10);
   - it resolved source lines study-local first, and searched any ancestor commit.
2. **How was P5 fixed?** Every digested line gets its own verdict against its own digest; duplicates are each validated and reported.
3. **How was P9 fixed?** Existence is proved with `git cat-file -e` and type `blob` before `cat-file blob > file`, each exit status checked, with no pipeline. A genuinely empty file still validates (P19).
4. **How was P10 fixed?** New gate G6, for every study:
   - HEAD `+impl` blobs compared with the working tree by raw SHA-256, never through the index;
   - no extra files and no symlinks;
   - manifest equal to HEAD's blob, rows and tree; PROVENANCE.md tree row equal to HEAD's tree;
   - `--no-replace-objects` and a scrubbed `GIT_*` environment.
5. **Does every line validate independently?** Yes.
6. **Is existence checked before hashing?** Yes.
7. **Is current source tied to HEAD?** Yes, to committed HEAD objects; replace refs and `GIT_DIR` redirection are neutralized.
8. **Can an edited `+impl` with a regenerated manifest still pass?** No (P10, P26, P28, P36, P37).
9. **Can a fabricated duplicate line still pass?** No (P5, P6).
10. **Can a nonexistent historical file validate as empty?** No (P9, P18).
11. **Do the seven legitimate superseded hashes still pass?** Yes: same paths, exact digests, freeze commit `bba45e7`, source commit `1438aa3`, declared tree `edbfe47e…` equal to the freeze tree (H2 in the merged checkout).
12. **Is the historical/current distinction explicit?** Yes: `HISTORICAL_SOURCE_HASH_VERIFIED` and `CURRENT_SOURCE_HASH_VERIFIED` are separate statuses with per-line records.
13. **Is committed migration provenance self-contained?** Yes; it cites only committed evidence and commit identities, with local folders marked supplementary.
14. **What repair commit was created?** `b21483b158f58e05e7b56957f2fbe8e1d2891395`: parent `9b30ec4`, tree `64fc72d8…`, 30 files. The superseded first attempt `adf86a3` was amended before any merge.
15. **Did the repair touch `+impl` or scientific config?** No.
16. **Was A2 accepted on re-review?** Yes.
17. **Was the branch merged?** Yes, by fast-forward into `benchmark-methodology-r2`, with no conflicts.
18. **Final merged HEAD?** `b21483b158f58e05e7b56957f2fbe8e1d2891395` (not pushed).
19. **Does `+impl` still match upstream 253069 byte for byte?** Yes: 79/79, tree `4ba9a3ae…`.
20. **Which pre-existing failures remain?**
    - `test_finalization_gate` H `move_activity_400`;
    - its I-set: the tracked seven, plus three untracked studies failing identically before the merge (`frozen_inner_solver_study`, `frozen_problem25_reference`, `scientific_delta_olhoff_migration`);
    - `confbench_selftest` T2.
21. **Were new failures introduced?** One, merge-induced: `gray_kkt_forensic_audit` (untracked) pinned the superseded manifest as required evidence. By the owner's decision it was resolved in that study's own record (EVIDENCE_AMENDMENT_1: the pin marked historical, the digest kept, the originals preserved). None remain.
22. **Did the historical S160 anchor reproduce?** Yes:
    - bitwise equal to the migration and upstream runs;
    - frozen campaign record matched in ρ, ω₁, volume, 91 and 2241;
    - ω₁ 169.495227021538, and the 81-row hash `28756d22…` reconstructed exactly.
23. **Did the Pedersen S160 anchor reproduce?** Yes:
    - identical to committed `S160x20` at the established level;
    - bitwise equal to the migration and upstream runs;
    - 121/2369, ω₁ 169.210576386275.
24. **Frozen production preset and config hash?** `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`; 160×20 hash `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4`.
25. **Do all nine future configs resolve correctly?** Yes: only mesh-bound rows vary, R = 0.06·b is fixed, and all hashes equal the ones recorded in PROVENANCE.json.
26. **Is telemetry ready?** Yes, for both total and per-outer scaling.
27. **Was anything above 160×20 solved?** No.
28. **Is the nine-mesh campaign authorized?** **Yes: `OLHOFF_NINE_MESH_CAMPAIGN_AUTHORIZED`.**
29. **What HEAD and config identity must the campaign use?** `benchmark-methodology-r2` @ `b21483b158f58e05e7b56957f2fbe8e1d2891395`, `+impl` tree `4ba9a3ae…`, preset `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, and the nine config hashes in `CAMPAIGN_IDENTITY.json`.
30. **What is the only authorized next scientific action?** The definitive nine-mesh campaign, as its own separate task, on exactly that identity. Nothing else.

## Files

| file | what it holds |
|---|---|
| GATE_PREREGISTRATION.md | criteria and attempt-1 preservation |
| AMENDMENT_REVIEW.md, FINALIZATION_GATE_REVIEW.md, MIGRATION_COMMIT_REVIEW.md | Steps 1–2 |
| MERGE_RECORD.md | the commit and the merge |
| POSTMERGE_IDENTITY.md, POSTMERGE_TESTS.md, RESOLUTION_GRAY_KKT.md | Step 4 |
| HISTORICAL_S160.md, PEDERSEN_S160.md | Steps 5A/5B |
| CAMPAIGN_IDENTITY.json, NINE_MESH_CONFIG_PREVIEW.md, NINE_MESH_CONFIGS.json | Step 5C |
| TELEMETRY_READINESS.md, REPOSITORY_INTEGRITY.md | Steps 5D/5E |
| CAMPAIGN_AUTHORIZATION.md | the decision |
| METRICS.json, EVIDENCE.json, FINAL_SHA256.txt | machine-readable record |
| `evidence/`, `logs/`, `scripts/` | raw evidence, logs, scripts |
| repair audit | `diagnostics/provenance_gate_hardening/` (committed) |
| earlier blocked states | `attempt1_blocked/`, `attempt2_blocked_gray_kkt/` |
