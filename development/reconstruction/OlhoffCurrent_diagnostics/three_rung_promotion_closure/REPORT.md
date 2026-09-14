# REPORT — three_rung_promotion_closure

## Final verdicts

```
ORIGINAL_C240_EVIDENCE_TRANSFER_FAIL
HISTORICAL_FINAL_SHA256_REPAIR_PASS
TIMING_ONLY_DRIFT_SETTLED_PASS
PROMOTION_PROVENANCE_BLOCKED

THREE_RUNG_PRODUCTION_POLICY_VALIDATED  [REUSED — NO NEW SCIENTIFIC RUN]

PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED
PRODUCTION_FREEZE_FAIL
NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```

Because nothing was promoted, the two promotion-equivalence verdicts cannot be
issued as a pass or a fail and are reported as not-assessable:

```
PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED
THREE_RUNG_PROMOTION_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED
```

## In one paragraph

The task's Phase 2 precondition failed: **the original C240 trajectory was never
transferred to this host**, confirmed by four independent search methods
including digest elimination of every candidate on the machine. Two of the three
repairs that *were* possible here were completed and verified — the stale
`FINAL_SHA256.txt` and the timing-only drift — clearing five entries from the
gate's mismatch lists. But promotion stays blocked, and by more than C240:
**six** git-ignored container artifacts are in the transfer class, not one, so a
C240 copy alone will not open the gate. Zero candidate scientific runs were
executed. Nothing was promoted, nothing frozen, no campaign authorized. The
validated three-rung result is untouched and reusable.

## Answers to the 58 required questions

1. **Branch / starting HEAD?** `benchmark-methodology-r2` / `60f5b72519aeba942b650d5408339f7ffe6b978b`.
2. **Initial dirty state?** 4 paths: 2 modified (`runs/C320x40_iterations.csv`, `runs/C320x40_record.json`) and 2 untracked (`three_rung_promotion_validation/`, `three_rung_promotion_validation_retry1/`).
3. **Which pre-existed?** **All four.** None was created by this task.
4. **Both prior promotion studies preserved unchanged?** **Yes.** Each verifies against its own `FINAL_SHA256.txt` — the first study's 7 files, the retry's 48 study files plus its git-ignored trajectory.
5. **Is retry1's C320 validation still intact?** **Yes.** It passes G1–G5 on this host (`ok=1`), its trajectory matches `fb29817e…ca5bbaffe`, and all frozen verdicts were recovered from its `METRICS.json`. `VALIDATED_C320_EVIDENCE_REUSE_PASS`.
6. **Were zero scientific optimization runs executed?** **Zero candidate runs** — no C320, C160, C240, C400, nine-mesh case, fixed-move arm or production baseline. One disclosed deviation: the pre-existing repository fixture `test_preset_equivalence` performed a 160×20 solve of the **production** preset once, before the skip rule was applied. It produced no scientific claim, none is made from it, and it affected no verdict. `SOFTWARE_VALIDATION.md` §0.
7. **Was C240 copied rather than regenerated?** **Neither.** It was not copied — no transfer occurred — and it was explicitly **not** regenerated.
8. **Expected full SHA-256?** `183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d` (131 203 128 bytes), recovered from `three_rung_resolution_240/EVIDENCE.json`, not from the `183d7ce6…` prefix.
9. **Actual SHA-256?** *n/a — the file does not exist on this host; the parent directory does not exist.*
10. **Did they match exactly?** **No.**
11. **Did `three_rung_resolution_240` return to G1–G5 PASS?** **No** — Phase 3 was unreachable. It fails G2 on the missing C240 trajectory *and* on three container digest mismatches.
12. **What was wrong with the historical `FINAL_SHA256`?** Three digests disagreed with the study's own committed, clean files. The documents were edited after the hash file was written and both states committed together — machine-independent staleness, not scientific damage.
13. **Which exact entries were repaired?** `PROVENANCE.md` (`42037f81…`→`0e987a9b…`), `BASELINES.md` (`74629a16…`→`2212a66a…`), `evidence/baselines.json` (`08b48346…`→`a4b55671…`). 3 of 143 lines; 68 of 71 entries untouched.
14. **Were underlying scientific files unchanged?** **Yes** — each is byte-identical to `git show HEAD:<path>`. Only digests changed.
15. **Old / new `FINAL_SHA256` hashes?** old `2fb9fc731d816e5c47c9835d5f777f8bb3be2fe61773ca00119049ed2bf5e0e0`, new `a7ef91f9cc279b46288f25f6f453495c5d358d96e1ff5ad33a8d87e61c73bd3e`. Old archived at `evidence/two_branch_FINAL_SHA256.BEFORE.txt`.
16. **The two timing-only dirty paths?** `two_branch_controller_validation/runs/C320x40_iterations.csv` and `…/C320x40_record.json`.
17. **Scientific or `tOuter`-only?** The CSV is strictly **`tOuter`-only** — 55 columns × 1600 rows compared cell by cell; the 54-column scientific projection is byte-identical (`5f9f1896…8f6bb53d`). The JSON is **not** strictly tOuter-only and is not described as such: exactly 3 leaves differ — `wall_s` (timing), `trajectoryBytes` (container size), `matlab` (toolchain version). No scientific state differs in either.
18. **How were they settled?** Working copies **archived** to `evidence/preserved_working_copies/`, then `git checkout --` restored the authoritative versions. Both now equal the digests `FINAL_SHA256.txt` and `EVIDENCE.json` already recorded.
19. **Is timing telemetry explicitly excluded?** **Yes** — `tOuter`, `tEig`, `tGrad`, `tInner`, all `toc`-derived, none read back by the optimizer. Frozen in `PREREGISTRATION.md` §3 and `TIMING_TELEMETRY_AUDIT.md` §1. Both retry readings preserved: literal enumerated-list → mismatch; category/scientific-state → PASS. The retry's preregistration was **not** rewritten.
20. **Do all load-bearing studies pass G1–G5?** **No — 1 of 4.** Only `three_rung_promotion_validation_retry1`. The other three fail G2; `two_branch_controller_validation` also fails G4.
21. **Does `test_finalization_gate` pass?** **No — 4 failures** (cases H and I). Self-tests A–G pass; the gate is operational and fails closed.
22. **Is promotion provenance PASS?** **No — `PROMOTION_PROVENANCE_BLOCKED`.**
23. **What exact validated candidate config was recovered?** `[0.04 0.02 0.01]`, `move.continuation.signal = stageExhaustion`, `stop.rule = stageExhaustion`, `move.policy = ladder`, `stop.norm = l2`, `stop.toleranceRule = meshScaled` (0.1 at 320×40), frozen `E = A OR B` with `W = 20`, `P = 20`, `Wnp = 10`. Re-resolved from the retry's own builder and **hash-matched** to the validated run: `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`. `VALIDATED_POLICY_RECOVERY.md`.
24. **Canonical production config before promotion?** `[0.04 0.02 0.01 0.005]`, `boundVariable`, `designChange`; `cfgHash 2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` at 320×40.
25. **What fields changed during promotion?** **None — no promotion occurred.** The delta a promotion *would* apply is 3 computational fields, 0 unexpected (`PROMOTION_DIFF_PLAN.md`).
26. **Was A changed?** **No.**
27. **Was B changed?** **No.**
28. **Was persistence changed?** **No.**
29. **Was tolerance scaling changed?** **No.**
30. **Was 0.005 simply removed from canonical production?** **No** — canonical production still carries `[0.04 0.02 0.01 0.005]`.
31. **Did canonical continuation change to `stageExhaustion`?** **No** — still `boundVariable`.
32. **Does beta retain only a mathematical/diagnostic role?** In the **validated candidate**, yes — it appears in neither the descent guard nor the terminal admission, and software tests confirm it can neither descend nor terminate under `stageExhaustion`. In **canonical production today** beta still holds continuation authority, because promotion did not occur.
33. **Is promoted config exactly the validated candidate?** *n/a — nothing promoted.*
34. **Does config-equivalence PASS?** **Not assessable** — no promoted configuration exists to compare. Against production *as it stands*, the only differences are the 3 intended policy fields plus 3 harness/label fields; 18 of 18 scientific formulation fields are already identical and there are **0 unexpected** computational differences.
35. **Does production dispatch to the exact tested implementation?** **Yes.** `olhoffcurrent_assert_dispatch` passes; `olh.move.exhaustion`, `olh.move.limit` and `olhoffSolve` all resolve under production `+impl/`; the `+impl` tree hash equals the `implTree` recorded inside the validated run.
36. **Does promotion-equivalence PASS?** **Not assessable** — nothing promoted. The code half is discharged and passes (Q35).
37. **Is legacy beta mode available only explicitly?** Not applicable yet — beta continuation is currently the **canonical default**, not a legacy opt-in. Making it explicit-only is part of the promotion that did not happen.
38. **Can canonical production accidentally resolve to legacy mode?** Today it resolves there **by design**, since production has not been promoted. After promotion this must be re-asked; `olh.config.validate` already couples `stop.rule` and `move.continuation.signal`, so a half-applied promotion is refused at resolve time.
39. **Final `+impl` hash?** `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` — unchanged, 75 files, `CURRENT`.
40. **Final production config/preset hash?** `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` (320×40); preset `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` (`6ed3624cd19b3569f55b23ac…`). Unchanged.
41. **A/B implementation hash?** `exhaustion.m` `17b37a384b1aa5d987d9c861…`, `limit.m` `61fa923d430121ead764a229…`, `olhoffSolve.m` `1e5a114cbf91717e01e5592e…`.
42. **Was a focused commit created?** **No.**
43. **If yes, what commit?** *n/a.*
44. **Final HEAD?** `60f5b72519aeba942b650d5408339f7ffe6b978b` — unchanged.
45. **Final dirty state?** 1 modified tracked file (`two_branch_controller_validation/FINAL_SHA256.txt`, the Phase 4 repair) and 3 untracked directories (the two prior promotion studies and this one). The two timing paths are now **clean**.
46. **Are all production changes accounted for?** **Yes — there are none.** No canonical production path was touched; `+impl` is clean at HEAD and hashes to its manifest.
47. **Did relevant software tests pass?** 4 of 5 repository test files clean; `test_finalization_gate` 4 failures (the container blockers); controller mechanics **29/29 PASS**.
48. **Were any scientific optimization tests/runs accidentally executed?** **Yes — one, and it is disclosed rather than buried.** `test_preset_equivalence` performs a 160×20 optimization and ran once before the skip rule was applied. It is a pre-existing repository fixture exercising the **production** preset and four-rung ladder, not the candidate; it produced no scientific claim, no retained artifact and no effect on any verdict. Every later invocation skips it explicitly, with the reason recorded in `evidence/software_tests.json`.
49. **Did production freeze pass?** **No — `PRODUCTION_FREEZE_FAIL`**, phase not reached. No defect was found in the production state; the freeze simply did not happen.
50. **Did closure finalization pass?** This study declares its evidence, manifests and `FINAL_SHA256.txt`, and passes its own G1–G5. Repository-wide finalization does **not** pass, for the container blockers.
51. **Is any provenance blocker unresolved?** **Yes — six git-ignored container artifacts.** `C240x30_trajectory.mat` (absent) plus `C160x20`, `C320x40`, `C400x50`, `F400_400x50`, `P400_400x50` (present, digests mismatched). Plus one analysed Class C item deliberately deferred.
52. **Is any controller/scientific blocker unresolved?** **No. None.** The controller is validated, unambiguous, dispatch-verified, and the promotion delta is 3 configuration fields.
53. **Is the nine-mesh campaign authorized?** **No — `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.**
54. **Were zero nine-mesh runs executed here?** **Yes, zero.**
55. **Was any outcome-driven scientific repair performed?** **No.** No scientific outcome was produced to repair. The two provenance repairs are bookkeeping, verified before application, fully reversible and archived.
56. **Does anything here justify projection?** **No.** Nothing observed bears on it, and projection is refused outright under stage exhaustion.
57. **Does anything justify changing R?** **No.**
58. **Does anything justify changing p / mass / q?** **No.**

## Compact summary

| | |
|---|---|
| branch | `benchmark-methodology-r2` |
| starting HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` |
| final HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` (no commit) |
| starting dirty state | 2 modified (timing drift) + 2 untracked (prior studies) — all pre-existing |
| final dirty state | 1 modified (`FINAL_SHA256.txt` repair) + 3 untracked (prior studies + this one) |
| **scientific runs** | **0** |
| C240 transfer digest result | **FAIL** — expected `183d7ce6…1fd50b0d`, file absent on this host |
| historical hash repair | **PASS** — `2fb9fc73…` → `a7ef91f9…`, 3 of 143 lines |
| timing-drift settlement | **PASS** — archived, then restored to `ff570d6e…` / `3fc68c05…` |
| promotion provenance | **BLOCKED** — 6 container artifacts in the transfer class |
| reused policy validation | `THREE_RUNG_PRODUCTION_POLICY_VALIDATED [REUSED — NO NEW SCIENTIFIC RUN]` |
| starting `+impl` hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| final `+impl` hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| production config hash | `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` |
| A/B hash | `exhaustion.m 17b37a38…`, `limit.m 61fa923d…`, `olhoffSolve.m 1e5a114c…` |
| tests | 4/5 repository files clean; `test_finalization_gate` 4 failures; mechanics 29/29 |
| finalization | this study G1–G5 PASS; 3 of 4 load-bearing studies FAIL |
| promotion | **NOT PROMOTED** |
| production freeze | **FAIL** (not reached) |
| campaign readiness | **BLOCKED** |

## What unblocks the next attempt — no compute

1. Transfer **six** container artifacts from the machine(s) that produced them
   and verify each against its declared digest — *or* make and document an
   explicit owner decision to re-declare the five locally regenerated ones.
2. Optionally apply the analysed Class C repair to
   `three_rung_architecture/EVIDENCE.json` (+ cascade).
3. Re-run the gate and `test_finalization_gate` to green.
4. Promote configuration-only: 3 fields, via the OlhoffCurrent config layer so
   `+impl` stays byte-identical (`PROMOTION_DIFF_PLAN.md` §5).
5. Discharge config and dispatch equivalence field-wise (**not** by whole-config
   hash — `CONFIG_EQUIVALENCE.md` §3), freeze, authorize.

**C320 must not be re-run**, nor C160, C240 or C400.
