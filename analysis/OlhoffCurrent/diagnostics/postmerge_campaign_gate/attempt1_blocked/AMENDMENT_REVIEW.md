# AMENDMENT_REVIEW — PREREGISTRATION_AMENDMENT_1

```
PREREGISTRATION_AMENDMENT_REVIEW_FAIL
```

**Reviewed:** `upstream_253069_migration/PREREGISTRATION_AMENDMENT_1.md` at `9b30ec4`, together with its REPORT, TEST_REPORT, MIGRATION_HANDOFF, CAMPAIGN_GATE and KNOWN_PREEXISTING_DEFECTS. The raw logs are in the git-ignored copy `analysis/OlhoffCurrent/evidence/upstream_253069_migration/`.

| amendment | decision | basis |
|---|---|---|
| **A1** — three pre-existing failures | **ACCEPTABLE** | see below |
| **A2** — finalization-gate `SUPERSEDED_PRODUCTION_SOURCE` rule | **NOT ACCEPTABLE as implemented** | not strictly fail-closed; three reproduced fail-open inputs (FINALIZATION_GATE_REVIEW.md) |
| **A3** — comparison-script correction | **ACCEPTABLE** | see below |
| A4 — added evidence | acceptable | adds checks only |
| A5 — errors in new tests and scripts | acceptable | each is logged, including `T_pedersen_adaptive_units.run1.log`; no code expectation changed |

One unacceptable amendment makes the review FAIL. The brief says to stop before merge in that case, so **no merge was performed**.

## A1 — pre-existing failures (verified from raw logs, not the report)

| check | pristine `013cc48` (`logs/T_baseline_finalization.log`, `harness_selftest_BASELINE_013cc48.json`) | migrated (`logs/T_gates3.log`, `harness_selftest.json`) |
|---|---|---|
| finalization H `move_activity_400` | `[FAIL]` | `[FAIL]` (same) |
| finalization I new-failing set | controller_architecture_offline, move_activity_400, move_ladder_necessity, three_rung_architecture, three_rung_promotion_closure, three_rung_promotion_validation_retry1, two_rung_architecture | **the same seven** |
| self-test T2 | FAIL, `olhoffcurrent_paths must be called with an output argument…` | FAIL, same message |

- **Same failures on the clean tree:** yes.
- **Worse after migration:** no. `T_gates.log`, before the A2 rule, also failed H for `two_branch_controller_validation`; A2 removed that failure.
- **Scientific impact:** none. The checks concern other studies' evidence bookkeeping and a self-test probe that discards the path guard.
- **Baseline-only failures:** R5a/R5/R6 of `test_evidence_retention` failed on the baseline only because `git archive` output is not a repository. They pass in the worktree.

**Caveat carried forward:** A1's acceptance depends on the gate suite. Once A2 is repaired, the suite has to be re-run, and the acceptance holds only if the failure set is still exactly these three.

## A3 — comparison-script correction

- The original run is kept: `evidence/comparisons.run1.json` and `logs/compare.log`.
- Run 1 and run 2 have the same `differing: []` value-difference lists for every section.
- Run 1 marked `beta_post_vs_pre` and `ex3_post_vs_pre` as failing only because of the preregistered allowances: `res.aux` added, `hist.ex*` dropped, and the 6 added schema rows.
- Run 2 encodes those allowances and adds two stricter checks: `droppedFieldsAllEmptyExTrace` and `cfgRowsDifferingOnlyAddedRows`.
- The change is bookkeeping only. No scientific result depends on hiding the run-1 output.
