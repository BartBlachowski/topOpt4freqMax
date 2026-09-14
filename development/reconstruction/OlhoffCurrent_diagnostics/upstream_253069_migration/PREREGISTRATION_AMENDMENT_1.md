# PREREGISTRATION_AMENDMENT_1 — POST HOC

Written **after** the runs and tests, not before. The preregistration (`MIGRATION_PREREGISTRATION.md`, SHA-256 `8854fdb7d4b0ea793c99ccaaa44914c3df7494ab50d0f96b9fd30d61ec72a414`, frozen copy `evidence/MIGRATION_PREREGISTRATION.frozen`) is unchanged. Each deviation below is listed with what was preregistered, what happened, and why the interpretation is what it is. A reviewer who rejects any of them should read the corresponding gate as FAIL.

## A1. `RELEVANT_TEST_SUITE_PASS` with pre-existing failures

- **Preregistered:** "all OlhoffCurrent suites and new migration tests: 0 failures".
- **Observed:** three checks fail after migration: `test_finalization_gate` H (`move_activity_400`) and I (seven historical studies), and `confbench_selftest` T2.
- **Evidence:** all three fail identically on a pristine `git archive 013cc48`, run the same way with the same linked evidence (TEST_REPORT.md). None concerns the migrated implementation; they are evidence-bookkeeping states of other studies and a defect in a harness self-test.
- **Interpretation applied:** zero failures *attributable to the migration*, with every pre-existing failure proved identical on the baseline. Under the literal wording this gate could not pass for *any* migration without repairing unrelated historical evidence, which the brief puts out of scope.

## A2. A migration-caused test failure removed by a gate refinement

- **Observed:** `test_finalization_gate` H also failed for `two_branch_controller_validation`. Its `FINAL_SHA256.txt` pins seven production source files that the promotion replaced.
- **Action:** `olhoffcurrent_finalization_gate` now accepts such lines only as `SUPERSEDED_PRODUCTION_SOURCE`, when the digest is that path's content in a commit reachable from HEAD and `+impl` is manifest-verified. Destructive tests J1–J3 were added. The historical hash file was not edited.
- **Why an amendment:** this changes gate semantics (in `analysis/OlhoffCurrent`, i.e. in scope), and the preregistration did not foresee it. The alternative — editing the study's hash file — would have rewritten historical evidence.

## A3. Comparison-script defect

- **Observed:** `mig_compare.m` run 1 flagged the preregistered §5/§6 allowances as failures and omitted the emptiness check for dropped `hist.ex*` fields.
- **Action:** the script now implements §5/§6 exactly, including the emptiness check, and was re-run. Run 1 is kept (`evidence/comparisons.run1.json`); its value-difference lists were already empty.

## A4. Additional evidence beyond §4

These additions only add evidence; no preregistered check was dropped.

- The upstream root-independent suites were run against the migrated tree and against the snapshot.
- `test_pedersen_adaptive_units` was added after a search showed no existing test exercises the adaptive rule directly.
- The harness self-test and finalization tests were run on a pristine baseline.

## A5. Test-construction errors in new tests and scripts (not code)

- `test_pedersen_adaptive_units` P2 tolerance.
- Harness-check script `cfg.runWarmup` / `mesh`.
- The `git` pager hang.

All are recorded in TEST_REPORT.md. None changed an expectation about the promoted implementation.
