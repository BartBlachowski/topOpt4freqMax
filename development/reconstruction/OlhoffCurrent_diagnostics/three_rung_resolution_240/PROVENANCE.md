# PROVENANCE — C240×30 resolution run

Everything measured, not asserted. Machine-readable:
`evidence/provenance_start.json`, `evidence/provenance_final.json`,
`evidence/single_factor.json`.

## Repository state

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` — re-read at task start, **not** inherited from a previous brief |
| Working tree at start | **dirty, 22 paths** (four prior studies' uncommitted deliverables + two top-level gate files); **23** once this study's directory was created. **None under `+impl/`.** |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files), start **and** end |
| Currentness | `CURRENT` |
| Source integrity | `PASS` — 0 mismatched, 0 missing, 0 extra |
| MATLAB | `25.2.0.2998904 (R2025b)` |
| Threads | `maxNumCompThreads(1)`, asserted in the driver before the solve |

## Frozen controller sources, hashed at task start

```
+impl/architecture/+olh/+move/exhaustion.m  17b37a384b1aa5d987d9c861e16071d1140af130f92406ccc11cec4518bcae0c
+impl/architecture/+olh/+move/limit.m       61fa923d430121ead764a229b4f48dfd77617540b4ab6db883ccd9d5904b1c1e
+impl/architecture/olhoffSolve.m            1e5a114cbf91717e01e5592e86203f401ed5cdb3e5f5ce9a727b8163cdf2fba3
```

Unchanged by this task: `git status` reports **0** modifications under `+impl/`.

## Inherited preregistrations, verified live

| study | SHA-256 |
|---|---|
| `two_branch_maturity_240` (the frozen `A`/`B` rule) | `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` |
| `two_branch_controller_validation` (controller + gates) | `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf` |
| `two_rung_architecture` | `b50455fbd3d97dcb72093fc13738f236d4d0f2e37a47caba0f515a16aaf7ca04` |
| `three_rung_architecture` (the study this one resolves) | `12c4bb960eeb6169521ea4b2e01b084a7bb3983d5387e94121e1729c2d71075c` |

**This study's own preregistration:** `f86d022e5259beb0936072d204e54e3761f14b2fd903c1794a6d4ccb2c5652cc`,
frozen `2026-09-09T17:44:21Z` — **before the run started at 19:45**.

## Phase-0 gate — `C240_PROVENANCE_PASS`

| requirement | result |
|---|---|
| currentness `CURRENT` | PASS |
| source integrity `PASS` | PASS |
| published MMA resolves correctly (`+impl/mma_published/mmasub.m`) | PASS |
| sensitivity filter resolves correctly | PASS |
| forbidden Olhoff paths absent | PASS |
| four-rung controller definition unambiguous (`[0.04 0.02 0.01 0.005]`) | PASS |
| tolerance law identical at 160/240/320/400 | PASS |
| inherited preregistration hashes valid (4/4) | PASS |
| required raw evidence present + hashed (27/27) | PASS |
| controller-study finalization gate | PASS |
| two-rung-study finalization gate | PASS |
| **three-rung-study finalization gate** | **PASS** |
| test suite | PASS (see below) |

## Phase 1 — prior three-rung study sealed *before* execution

Verified independently, before the run began:

* `FINAL_SHA256.txt` — **43/43** digests re-verified with `shasum -c`, zero failures;
* `DATA_MANIFEST.json` — 43 artifacts, identical file set;
* `EVIDENCE.json` — **11/11** declared artifacts present and hash-valid;
* recorded verdicts intact: `THREE_RUNG_COUNTERFACTUAL_EXACT` ·
  `THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED` · `MORE_THREE_RUNG_EVIDENCE_REQUIRED`.

**No prior evidence was modified and no prior verdict was rewritten.**

## Repository test suite

| test | start | end |
|---|---|---|
| `test_currentness` | PASS | PASS |
| `test_evidence_retention` | PASS | PASS |
| `test_finalization_gate` | **FAIL (self-referential)** | PASS |
| `test_path_isolation` | PASS | PASS |
| `test_preset_equivalence` | PASS | PASS |
| `test_source_integrity` | PASS | PASS |

The start-of-task failure is created by this task and cured by finishing it:
`test_finalization_gate` asserts that no *new* study directory fails the
finalization gate, and at Phase 0 this study's directory existed with no
deliverables. That is the fail-closed mechanism working. It was accepted only
under an explicit `testsPassOrSelfReference` flag valid at tag `start`; the
**final** provenance record is required to show `testsPass` true outright.

## The one scientific run

| | |
|---|---|
| Runs executed | **exactly 1** |
| Mesh | **240 × 30** (`NE = 7200`) |
| Controller | frozen four-rung `A OR B`, `[0.04 0.02 0.01 0.005]` |
| Resolved config hash | `33833323efa08facaa5849c24fe32d6c9c47b5924f88c00d34fb65f7140d54d6` — asserted in the driver before the solve |
| Cap | **1600**, inherited from `cv_config.m` |
| `stop.tolerance` | `0.075` = `0.05·√(7200/3200)` |
| Status | **CONVERGED @1358** (cap not reached) |
| Wall | 14 160 s (≈ 3.93 h), 1 thread |
| Inner MMA total | 44 181; non-converged inner solves **0** |
| Trajectory rebuild | exact — clamp displacement max `5.55e-17` |
| Raw evidence | `evidence/three_rung_resolution_240/C240x30_trajectory.mat`, 131.2 MB, `RHO` 7200 × 1358 |

**The controller was recovered by CALL, not by copy.** `cv_config('C',240,30)`,
`cv_telemetry` and `cv_export` from `two_branch_controller_validation` are
invoked unchanged, so the controller and telemetry are identical *by reference*
to those that produced the 160×20 / 320×40 / 400×50 evidence.

`cv_run.m` hard-refuses any mesh outside its three authorized ones — a guard from
that study's preregistration and a hashed artifact of a sealed study. **It was
not edited.** `scripts/r240_run.m` mirrors its call sequence and authorizes only
240×30; a diff of the two call sequences shows the new driver adds **six extra
asserts** (config hash, move policy, continuation signal, stop rule, cap, thread
count) and removes none.

## What was not done

No second optimization run of any kind. No rerun of 160×20, 320×40 or 400×50. No
modification of `A`, `B`, `E`, the persistence, the windows, the thresholds, the
move levels or the terminal admission. No Branch C. No alternative ladder. No
promotion. Production's preset still resolves to `move.levels = [0.04 0.02 0.01 0.005]`
with `move.continuation.signal = 'boundVariable'`.
