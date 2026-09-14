BOTTOM LINE

**The definitive nine-mesh campaign remains blocked.** Migration commit `9b30ec4` added a finalization-gate rule (`SUPERSEDED_PRODUCTION_SOURCE`, PREREGISTRATION_AMENDMENT_1 §A2) that is not strictly fail-closed.

- **P5:** a fabricated production-source digest passes when a later genuine line names the same path.
- **P9:** the rule checks the exit status of `git show … | shasum`, which is `shasum`'s. When `git show` fails, the empty-content digest passes.
- **P10:** a local `+impl` edit plus a regenerated `SOURCE_MANIFEST.json` satisfies the "current source" condition.

All three were reproduced end to end in a throwaway clone. The amendment review therefore fails. As the brief requires, the gate **stopped before merge**: no merge, no test suite, no 160×20 anchor, no production freeze, no config preview, and **no solve of any mesh**.

Everything else reviewed is sound:

- `+impl` at `9b30ec4` is 79/79 blob-identical to upstream `253069`.
- The pre-existing failures (A1) are proven identical on `013cc48`.
- The comparison-script fix (A3) is bookkeeping only.
- The one real use of the rule (seven `two_branch_controller_validation` lines → `1438aa3`) is legitimate.

The remaining blocker is a small defect in the gate's evidence bookkeeping, outside `+impl`. It does not affect any solver result.

```
PREREGISTRATION_AMENDMENT_REVIEW_FAIL
FINALIZATION_GATE_PROVENANCE_LOGIC_FAIL
MIGRATION_COMMIT_REVIEW_FAIL
OLHOFF_MIGRATION_MERGE_FAIL                 (not executed)
POSTMERGE_IMPLEMENTATION_IDENTITY_FAIL      (not executed)
POSTMERGE_TEST_SUITE_FAIL                   (not executed)
POSTMERGE_HISTORICAL_S160_FAIL              (not executed)
POSTMERGE_PEDERSEN_S160_FAIL                (not executed)
NINE_MESH_CONFIG_PREVIEW_FAIL               (not executed)

OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED
```

> The definitive nine-mesh campaign remains blocked because the finalization-gate `SUPERSEDED_PRODUCTION_SOURCE` rule in migration commit `9b30ec4` is not strictly fail-closed (fabricated-digest shadowing, masked `git show` failure, manifest-regeneration bypass), so PREREGISTRATION_AMENDMENT_1 cannot be accepted and the merge was not performed.

## Answers

1. **Was PREREGISTRATION_AMENDMENT_1 accepted?** No.
2. **Why?** A1 (pre-existing failures) and A3 (comparison-script fix) are acceptable, and A4/A5 are harmless. A2, the gate rule, fails the brief's strict fail-closed requirement: three inputs pass that must fail (FINALIZATION_GATE_REVIEW.md).
3. **Is the historical-hash rule strictly fail-closed?** No.
4. **Does a fabricated hash fail?** Only sometimes.
   - A single fabricated line fails (P2).
   - A fabricated line followed by a genuine line for the same path passes (P5).
   - The fabricated empty-content digest passes for a path with a deletion commit in history (P9).
   - Both P5 and P9 are latent in today's repository.
5. **Does the gate distinguish historical-source from current-source verification?** Yes, in an equivalent form. Superseded lines are listed and printed as `SUPERSEDED_PRODUCTION_SOURCE`, separately from lines verified against the working tree, and it does not claim they match current source. However, its "current source" check is only `+impl` versus the mutable working-tree manifest (P10).
6. **Was 9b30ec45… reviewed and accepted?** Reviewed, not accepted. Every structural criterion passes: parentage, scope, byte identity, manifest, Proposed/Yuksel/evaluator/Phase 6 untouched, p-continuation defect unfixed. The one failure is that the commit's claims about the gate rule are contradicted by the probes.
7. **What branch was it merged into?** None. The intended target is `benchmark-methodology-r2`, and a fast-forward is possible.
8. **Post-merge HEAD?** None. HEAD is unchanged at `013cc48451d33bed61c5c4eea174bbd898d548a2`.
9. **Merge conflicts?** None; no merge was attempted. None are possible: the target is the commit's sole parent, and none of the 77 added paths exists on disk.
10. **Is `+impl` byte-identical to upstream 253069?** At `9b30ec4`, yes: 79/79 blob ids equal. In the normal checkout, no — it is still the pre-migration 75-file tree `edbfe47e…`, because nothing was merged.
11. **Private solver/controller differences left?** None at `9b30ec4`.
12. **Did the post-merge suite pass?** Not run.
13. **Verified pre-existing failures:** `test_finalization_gate` H (`move_activity_400`) and I (the same seven studies), and `confbench_selftest` T2. They are identical on pristine `013cc48`, checked against the raw logs.
14. **New regressions?** None observed in the migration's own logs. The post-merge suite was not run. The gate-rule defect is a new defect, not a test regression.
15. **Historical 160×20 anchor reproduced?** Not run.
16. **Its final ω₁?** Not measured here. The frozen expectation is 169.495227021538.
17. **Pedersen 160×20 anchor reproduced?** Not run.
18. **Its final ω₁?** Not measured here. The expectation is 169.210576386275.
19. **Did old 81-row hashes reconstruct?** Not re-verified here. The migration reports that all 14 reconstruct (`test_preset_identity`).
20. **New production config hash?** None frozen.
21. **Frozen implementation hash?** None frozen. The candidate is `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` (at `9b30ec4`).
22. **Preset authorized for production?** None. `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` is recorded as production in `9b30ec4`'s PROVENANCE.json but is not authorized by this gate.
23. **Nine future configs consistent?** Not resolved.
24. **Radius policy fixed and consistent?** Not verified here.
25. **Total and per-outer telemetry ready?** Not assessed on a merged tree.
26. **Proposed/Yuksel/evaluator untouched?** Yes, both by this task and in `9b30ec4`.
27. **Anything above 160×20 solved?** No. Nothing was solved at any size.
28. **Ready for the nine-mesh campaign?** No.
29. **What HEAD and config identity must the campaign use?** None yet. It must be a future merged `benchmark-methodology-r2` head containing a repaired gate, with the identity frozen by a re-run of this gate.
30. **Only authorized next scientific action?** None. The next action is the owner's decision: authorize the gate repair (CAMPAIGN_AUTHORIZATION.md) or waive A2 in writing, then re-run this gate from Step 1.

## Other findings for the owner

- **Premise mismatch.** The normal checkout was on `benchmark-methodology-r2` @ `013cc48`, not on the migration branch. The migration is in worktree `…-migration-253069`.
- **Provenance points at uncommitted material.** `PROVENANCE.md` at `9b30ec4` cites the untracked `diagnostics/scientific_delta_olhoff_migration` as acceptance evidence.
- **Unpushed commit.** Local `benchmark-methodology-r2` is one commit ahead of `origin`.

## Files

| file | what it holds |
|---|---|
| FINALIZATION_GATE_REVIEW.md | the blocker, the probe table and the remedy |
| AMENDMENT_REVIEW.md, MIGRATION_COMMIT_REVIEW.md | reviews of Steps 1 and 2 |
| MERGE_RECORD.md | Step 0 state |
| REPOSITORY_INTEGRITY.md, CAMPAIGN_AUTHORIZATION.md | integrity and the final verdict |
| `scripts/gate_probe.m`, `scripts/gate_probe_p10.m` | probe scripts |
| `evidence/gate_probe.log`, `evidence/gate_probe_results.json`, `evidence/gate_probe_p10_result.json` | raw probe evidence |
| stub files | records for the steps that were not executed |
