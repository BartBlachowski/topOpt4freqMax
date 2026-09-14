# AMENDMENT_REVIEW — PREREGISTRATION_AMENDMENT_1, re-reviewed

```
PREREGISTRATION_AMENDMENT_REVIEW_PASS
```

Attempt 1 rejected A2 (`attempt1_blocked/AMENDMENT_REVIEW.md`). This review covers the amendment as it stands after the owner-authorised repair commit `b21483b`.

| amendment | decision | basis |
|---|---|---|
| **A1** pre-existing failures | **ACCEPTABLE** | attempt 1 proved from raw logs that `test_finalization_gate` H (`move_activity_400`), the seven-study I-set and self-test T2 fail identically on pristine `013cc48`. The repair does not change this: the worktree suites after the repair show exactly H `move_activity_400` plus the same seven (`provenance_gate_hardening/evidence/tests.json`). |
| **A2** superseded-production-source rule | **ACCEPTABLE (as repaired)** | FINALIZATION_GATE_REVIEW.md: C1–C7 are met; 39/39 adversarial probes pass from **committed** code; the original attack scripts now fail closed; the seven legitimate lines verify. |
| **A3** comparison-script correction | **ACCEPTABLE** | unchanged since attempt 1: run 1 is retained, value-difference lists are identical, and the change is bookkeeping only |
| A4 / A5 | acceptable | additions and test-construction errors, all logged |

## Disclosures that come with the repair

- **The first repair commit, `adf86a3`, was superseded before any merge.** Self-review found normalization-variant, replace-ref and `GIT_DIR` bypasses. PREREGISTRATION_ADDENDUM_1 was frozen before the fix, and the commit was amended into `b21483b`. The `adf86a3` gate is kept in `provenance_gate_hardening/selfreview/` and fails 6 of the 39 probes.
- **Stricter by design:** a study whose `FINAL_SHA256.txt` or `EVIDENCE.json` is **not committed** can no longer claim superseded production source. The original attempt-1 probe P1, an uncommitted sandbox copying J1, therefore now fails. That is intended (PREREGISTRATION §3 S3.1).
