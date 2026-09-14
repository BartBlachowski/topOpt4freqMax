# POSTMERGE_TESTS

```
POSTMERGE_TEST_SUITE_FAIL        (NOT EXECUTED — campaign gate stopped before merge at Step 1)
```

No suite was run in the normal checkout. The suite list for the next attempt is MIGRATION_HANDOFF.md step 3. After an A2 repair, `test_finalization_gate` must again show exactly the three pre-existing failures (H `move_activity_400`, the seven-study I-set, self-test T2) plus passing destructive tests for the P5/P9/P10 cases.

Blocker: FINALIZATION_GATE_REVIEW.md, AMENDMENT_REVIEW.md.
