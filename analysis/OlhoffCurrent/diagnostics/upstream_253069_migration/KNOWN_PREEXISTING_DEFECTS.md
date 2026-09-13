# KNOWN_PREEXISTING_DEFECTS — Part 19

Defects present **before** this migration, observed during it, deliberately **not repaired**. The migration adds no new defect of these kinds; evidence for that is given per item.

## D1. p-continuation `hist.move` logged before the p-controller reset

| | |
|---|---|
| id | `PCONT_MOVE_HISTORY_LOGGED_BEFORE_RESET` |
| introduced | upstream `6b08708` (present in `253069`, therefore now in OlhoffCurrent `+impl`) |
| affected presets | `pContinuationDecoupled`, `pMassCompatible` — exactly those with `material.stiffness.continuation.driver = ownCounter` |
| not affected | all three OlhoffCurrent presets (none uses p continuation); every other upstream preset |

### Exact logging issue

In `architecture/olhoffSolve.m` the outer loop:

1. sets `mvMax = max(mvNow)` right after `olh.move.limit`;
2. then the p-controller's stall interception resets `mvNow = moveLevels(1)`;
3. `hist.move(outer) = mvMax` therefore stores the **pre-reset** move level at p-event iterations.

Downstream of `hist.move`, the log text and `transitions.moveLevels` inherit the wrong value. The move actually applied to the design is the post-reset one.

### Evidence that the density trajectory is unaffected (this migration, target tree)

The anchors were run on the migrated OlhoffCurrent (legacy `olhoffOpt` route, diagnostics on, 250 outer, CAP_HIT). Source: `evidence/comparisons.json → pcont`.

| anchor | target science digest = upstream candidate (253069-equivalent) digest | ρ = committed reference | ω = reference | every Δρ = reference | `hist` fields differing from reference | where `hist.move` differs | p-events |
|---|---|---|---|---|---|---|---|
| A6_pdecoupled160 | **yes** (`94f1751b…`; log shape and text too) | bitwise | bitwise | bitwise | `move` only | iterations 86, 97 | 86, 97 |
| A7_massp160 | **yes** (`635404f7…`) | bitwise | bitwise | bitwise | `move` only | iterations 61, 72 | 61, 72 |

The differences are confined to `hist.move` at the p-event iterations, and the migrated target behaves byte-for-byte like upstream. This is the defect the upstream audit localized (`DEFAULT_NONINTERFERENCE.md`), unchanged; no additional regression was introduced.

### Why it is excluded from this migration

- Repairing it changes `olhoffSolve.m`. `+impl` would then be a private fork, not byte-identical to an accepted upstream commit, which defeats the purpose of this migration.
- Mixing a logging repair into the promotion would blur the clean boundary: "the trajectory changed because of the promotion" versus "the record changed because of the repair".
- No production or historical OlhoffCurrent preset uses p continuation.

### Recommended separate repair

Upstream-first, in the Olhoff repository, as its own commit and audit:

1. Record the move level **after** the p-controller's stall interception: assign `mvMax` (or `hist.move`) after the reset.
2. Regenerate the A6/A7 anchor references, or prove them against the corrected `hist.move` with ρ/ω/Δρ bitwise unchanged.
3. Show that every other anchor and S160x20 are science-digest-identical.
4. Promote to OlhoffCurrent by byte copy with its own provenance event.

## D2. `confbench_selftest` T2 calls `olhoffcurrent_paths()` without an output

| | |
|---|---|
| where | `examples/Performance/conference_bench/confbench_selftest.m`, test T2 |
| issue | T2 calls `olhoffcurrent_paths();` with no output argument inside its probe. The guard's `nargout` check (introduced in `cf1b71d`) raises `olhoffcurrent_paths:GuardDiscarded` *before* the dispatch check, so T2 never sees `PathContaminated` and reports FAIL. |
| pre-existing? | **yes**: the same T2 failure on a pristine `git archive` of `013cc48` (`evidence/harness_selftest_BASELINE_013cc48.json`) |
| effect | the harness self-test's T2 verdict is uninformative. The fail-closed property itself is covered and passing in `test_path_isolation` and `test_source_integrity` (tests I, J). |
| recommended repair | separate change: `g0 = olhoffcurrent_paths(); %#ok<NASGU>` inside the T2 probe |

`confbench_selftest` also computes `report.pass` before appending T8 onward; the written `pass` field therefore ignores T8+. This is pre-existing as well, so `scripts/mig_harness_check.m` recomputes the verdict over every test.

## D3. Other studies failing the finalization gate

`test_finalization_gate` fails on the pristine baseline with exactly these two checks, and fails identically after migration:

- H: `move_activity_400` — required trajectory hash mismatches;
- I: seven studies fail the gate — `controller_architecture_offline`, `move_activity_400`, `move_ladder_necessity`, `three_rung_architecture`, `three_rung_promotion_closure`, `three_rung_promotion_validation_retry1`, `two_rung_architecture`.

These are evidence-bookkeeping states of historical studies, unrelated to the implementation. Repairing them requires the owners' evidence decisions; out of scope.
