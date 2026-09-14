# REPORT — three_rung_promotion_validation_retry1

## Verdicts

```
DEPENDENCY_SPECIFIC_SCIENTIFIC_PROVENANCE_PASS
THREE_RUNG_SINGLE_FACTOR_PASS
C320_THREE_RUNG_PREFIX_EQUIVALENCE_PASS
C320_THREE_RUNG_TERMINATION_PASS
THREE_RUNG_PRODUCTION_POLICY_VALIDATED
PROMOTION_PROVENANCE_BLOCKED
PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED
NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```

and, since promotion did not occur, the two Part-J verdicts are issued as
not-assessable rather than as a pass or a fail:

```
PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED
THREE_RUNG_PROMOTION_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED
```

## In one paragraph

The three-rung production controller is **validated at 320×40**, with one
scientific run. It reproduced the four-rung oracle **bitwise** through the
terminal declaration at iteration 352 — `RHO` and `DRHO` identical over all
4 505 600 entries, 52 telemetry columns and 36 `hist` fields with zero differing
elements, and S1/S2/S3 at 274/A, 313/B, 352/B exactly — then **converged there**
instead of descending to `move = 0.005`, at a terminal state equal to the oracle
in all 13 recorded quantities. That eliminated 1248 outer iterations (78.00 %)
and 70 034 inner MMA iterations (91.51 %), confirming the prior prediction to the
digit. **Nothing was promoted:** the original C240 trajectory is not on this
host, so promotion-level provenance is blocked on grounds fixed before the run.
The validation is kept and C320 need never be re-run.

## Phases NOT_REACHED or not performed

| Part | State |
|---|---|
| I — conditional promotion | **NOT PERFORMED** — condition (validated ∧ provenance) false |
| J — promotion equivalence | **PARTIAL** — the code-dispatch half discharged and PASSES; the config half not assessable, there being no promoted config |
| K — production freeze | **NOT PERFORMED** — nothing to freeze |
| L — nine-mesh campaign | **NOT RUN**, zero runs, `BLOCKED` |

No misleading empty artifact was created: each of `PROMOTION.md`,
`PROMOTION_EQUIVALENCE.md`, `PRODUCTION_FREEZE.md` and
`PERFORMANCE_READINESS.md` states plainly that its phase was not performed, why,
and what the step would consist of.

## Answers to the 40 required questions

1. **Why did attempt 1 stop?** Its mandatory gates V2/V22 require the three
   load-bearing studies to pass finalization G1–G5. On this host all three fail
   G2 and `test_finalization_gate` reports the same regression. The repository
   owner directed the stop. The cause was evidence locality plus one stale hash
   file, not scientific damage.
2. **Were scientific runs in attempt 1 exactly zero?** **Yes.**
3. **Is `C320_ORACLE.md` intact?** **Yes** — `d09579b80c0d187b3800e71e22538a1c482e60d79505d44457ffc9d57e8b82d9`, matching that study's own `FINAL_SHA256.txt`. All seven of its files match.
4. **Were its RHO and omega prefix hashes re-confirmed?** **Yes, both, exactly** — `RHO[:,1:352] = b8c0f18d…d78b5ba3`, `omega(1:2,1:352) = fd636083…7e4c488d`, recomputed from the raw container.
5. **Is the C320 oracle locally valid despite the container-digest failures?** **Yes.** The failures are `.mat` **container** digests (Class A); every scientific digest committed to git reproduces exactly, including the final `rho` `0348b288…`. The oracle arm's configuration also reconstructs to the committed `cfgHash 2359a111…`.
6. **What exact A/B rule was recovered?** `A = med20 cos < 0 ∧ med20 net < 0.5 ∧ amp ≥ tol`; `B = med20 cos > 0 ∧ amp < tol`; `E = A ∨ B`; `W = 20`, `P = 20`, `Wnp = 10`, `tol = 0.05·sqrt(NE/3200)` (= 0.1 here). Declared at the first iteration where either counter reaches P; counters reset to 0 on any false; descent resets the window wholly to the new stage. Source: `exhaustion.m`, consistent with `tb_branches.m` and with the preregistration it cites by digest (`62748225…` — the file hashes to exactly that). `CONTROLLER_RECOVERY.md`.
7. **Does architecture tracked/frozen evidence remain intact?** **Yes.** All four verdicts still asserted, and `git status` over `three_rung_architecture`, `three_rung_resolution_240`, `two_branch_maturity_240` and `+impl` is empty — everything clean at HEAD.
8. **Is C240 raw evidence locally present?** **No.**
9. **If absent, is it classified as remote-not-transferred rather than lost?** **Yes** — `REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL`. Not re-run, not called loss.
10. **What genuinely stale hash bookkeeping remains?** Three entries in `two_branch_controller_validation/FINAL_SHA256.txt` (`PROVENANCE.md`, `BASELINES.md`, `evidence/baselines.json`), all disagreeing with committed, clean files — machine-independent. `three_rung_architecture/EVIDENCE.json` carries the same stale `baselines.json` digest.
11. **What is the status of `tOuter` drift?** Diagnosed and settled for this study without changing any scientific state: excluded from acceptance in advance, and routed around entirely by taking the oracle CSV from git HEAD. The two dirty working-tree paths were left as found — that is the owner's call to commit or revert.
12. **Did dependency-specific scientific provenance PASS?** **Yes.**
13. **Was retry preregistration frozen before C320?** **Yes** — `FROZEN_BEFORE.txt` records its digest `6b0638d9…` plus proof that `runs/` held 0 entries and the candidate evidence directory did not exist.
14. **Did single-factor PASS?** **Yes** — one computational difference across all 81 schema leaves (`move.levels`), plus `runtime.name`, which the config hash excludes.
15. **Did software validation PASS?** **Yes** — 29/29 in the retry's suite; 5 of 6 repository test files clean (`test_finalization_gate` fails 4, cases H/I only, for the known provenance classes).
16. **Was exactly one scientific run executed?** **Yes — exactly one**, enforced by hard-coding the mesh in `tr_run.m`.
17. **Was it exactly 320×40?** **Yes**, NE = 12800, tol = 0.1, cap 1600.
18. **Did `RHO[:,1:352]` match the frozen hash exactly?** **Yes.**
19. **Did `omega(1:2,1:352)` match exactly?** **Yes.** `RHO` and `DRHO` are also bitwise identical to the oracle element by element.
20. **Were S1 = 274/A, S2 = 313/B, S3 = 352/B reproduced?** **Yes, all three exactly**, with `declBegin` 255 / 294 / 333.
21. **Did beta cause any transition?** **No.** Both descents carry a recorded branch and persistence window; `betaStallRel` / `betaStallFires` were computed and are bitwise identical to the oracle's, with no authority.
22. **Did the candidate ever enter 0.005?** **No.** Moves visited: exactly `{0.04, 0.02, 0.01}`; `max(stage) = 3`.
23. **Did it report CONVERGED at S3?** **Yes** — `CONVERGED`, `nOuter = 352`. No iteration 353 executed.
24. **Did terminal state equal the oracle?** **Yes — 13 of 13**, tested with `==`, not a tolerance.
25. **Did any inner solve fail?** **No** — `innerNonConv = 0`, `innerMax = 40`.
26. **Did it hit a cap?** **No** — `CONVERGED` at 352 against a cap of 1600.
27. **Scientific policy-validation verdict?** **`THREE_RUNG_PRODUCTION_POLICY_VALIDATED`.**
28. **Was C240 original evidence transferred before promotion?** **No** — and it cannot be from this host.
29. **Was the stale `FINAL_SHA256` repaired?** **No — prepared and verified, not applied.** `evidence/PROPOSED_two_branch_FINAL_SHA256.txt` (`a7ef91f9…`) changes exactly 3 of 143 lines. Not applied because H1 blocks promotion regardless and the target is a committed historical study.
30. **Did all load-bearing studies pass repository G1–G5 before promotion?** **No** — all three FAIL G2; `two_branch_controller_validation` also fails G4.
31. **Did `test_finalization_gate` pass?** **No** — 4 failures, cases H and I. Its own self-tests A–G pass; the gate is operational and fails closed.
32. **Was promotion provenance PASS?** **No — `PROMOTION_PROVENANCE_BLOCKED`.**
33. **Was production promoted?** **No.**
34. **Is the promoted config exactly the validated candidate?** n/a — nothing was promoted. The delta a promotion would apply is recorded: three fields (`move.levels`, `move.continuation.signal`, `stop.rule`).
35. **Does promoted production use the exact tested controller code?** No promotion occurred, but the code half is discharged and **passes**: `olh.move.exhaustion`, `olh.move.limit` and `olhoffSolve` all resolve under production `+impl/`, and `olhoffcurrent_assert_dispatch` passes. The tree hash is identical to the `meta.implTree` recorded inside both trajectories.
36. **Final production hashes?** `+impl` tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`; production `cfgHash` at 320×40 `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad`. Both unchanged from task start.
37. **Did production freeze PASS?** **Not performed** — there was nothing to freeze.
38. **Is the nine-mesh campaign authorized?** **No — `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.**
39. **Were zero nine-mesh runs executed?** **Yes, zero.**
40. **Was any outcome-driven scientific repair performed?** **No.** One methodological disclosure belongs here rather than being buried: my preregistration's exclusion list was enumerated against the 55-column CSV and so named only `tOuter`, missing three further wall-clock timers that live in `res.hist` (`tEig`, `tGrad`, `tInner`). Those three differ; nothing else does. Both readings are computed and reported — literal FAIL, category PASS — with a mechanical check confirming every field separating them is a timer. The governing rule is the brief's own category language ("`tOuter` / nondeterministic timing telemetry"; "do not use wall-clock timing as a bitwise reproducibility requirement"), it was fixed before the run, and it changes no scientific, controller, optimization or inner-work quantity — all of which are bitwise identical. `PREREGISTRATION.md` was **not** rewritten. Full account: `C320_PREFIX_EQUIVALENCE.md` §4.

## The result

| | four-rung oracle | three-rung candidate |
|---|---|---|
| status | `CAP_HIT @1600` | **`CONVERGED @352`** |
| final stage / move | 4 / 0.005 | **3 / 0.01** |
| cumulative inner MMA | 76 532 | **6 498** |
| design at iteration 352 | — | **bitwise identical** |
| outer iterations eliminated | — | **1248 — 78.00 %** |
| inner MMA eliminated | — | **70 034 — 91.51 %** |

The fourth rung did not merely cost 1248 iterations: it **never terminated**
within the preregistered cap, while producing no change at all to the design the
third rung had already reached. That is the case for removing it — scientifically
immaterial, operationally pathological — and it is narrower than a claim of
harm, and narrower still than a mesh law. **One mesh was run.**

## State at task end

| | |
|---|---|
| HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` — **no commit created** |
| `+impl` tree | `edbfe47e…` — unchanged, `CURRENT`, 75 files |
| production config | unchanged |
| prior stopped study | unchanged — all 7 files match its `FINAL_SHA256.txt` |
| pre-existing dirty paths | unchanged — the same two, left as found |
| new | this directory, and one git-ignored trajectory under `evidence/three_rung_promotion_validation_retry1/` |

## What would unblock promotion

No compute. A **file transfer and two hash-file edits**:

1. copy `C240x30_trajectory.mat` from the machine that produced it, verify
   against `183d7ce60d512fc2…`;
2. apply `evidence/PROPOSED_two_branch_FINAL_SHA256.txt`, and the matching
   `three_rung_architecture/EVIDENCE.json` edit;
3. commit or revert the two `tOuter`-dirty paths;
4. decide the Class A container question (transfer or explicitly re-declare);
5. re-run the gate and `test_finalization_gate` to green;
6. promote configuration-only and discharge Part J.

**C320 must not be re-run.**
