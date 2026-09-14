# PROMOTION_PROVENANCE — Phases 3 and 6

# `PROMOTION_PROVENANCE_BLOCKED`

## 1. The frozen criteria, scored

| # | Criterion | Result |
|---|---|:--:|
| 1 | `VALIDATED_C320_EVIDENCE_REUSE_PASS` | ✅ **PASS** |
| 2 | `ORIGINAL_C240_EVIDENCE_TRANSFER_PASS` | ❌ **FAIL** |
| 3 | `C240_FINALIZATION_RESTORED_PASS` | ❌ **NOT REACHED** |
| 4 | `HISTORICAL_FINAL_SHA256_REPAIR_PASS` | ✅ **PASS** |
| 5 | `TIMING_ONLY_DRIFT_SETTLED_PASS` | ✅ **PASS** |
| 6 | all load-bearing studies G1–G5 PASS | ❌ **FAIL** (3 of 4) |
| 7 | `test_finalization_gate` 0 failures | ❌ **FAIL** (4 failures) |

Two of the three repairs this task could perform were performed. The blockers
that remain are, without exception, **file-transfer class**.

## 2. Load-bearing study gates, after the Phase 4 and 5 repairs

| Study | G1 | G2 | G3 | G4 | G5 | result |
|---|:--:|:--:|:--:|:--:|:--:|---|
| `two_branch_controller_validation` | ✅ | ❌ | ✅ | ❌ | ✅ | **FAIL** |
| `three_rung_architecture` | ✅ | ❌ | ✅ | ✅ | ✅ | **FAIL** |
| `three_rung_resolution_240` | ✅ | ❌ | ✅ | ✅ | ✅ | **FAIL** |
| `three_rung_promotion_validation_retry1` | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** |

Run with the repository's own `olhoffcurrent_finalization_gate`, which resolves
the mixed study-relative / repo-relative manifest convention. These are
**content** failures, not path-convention artifacts. Machine-readable:
`evidence/blockers.json`.

`three_rung_resolution_240` cannot reach Phase 3 at all: its required
trajectory is absent, so `C240_FINALIZATION_RESTORED_PASS` is not merely
unproven but unreachable on this host.

## 3. The remaining blockers — **six** container files, not one

This is the finding the phase list did not anticipate, and it should not be
discovered later: **transferring C240 alone does not unblock promotion.**

### 3a. `REQUIRED_MISSING` — 1 artifact

```
analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat
  sha256 183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d
  bytes  131 203 128
```

### 3b. `REQUIRED_HASH_MISMATCH` on git-ignored containers — 3 artifacts, required by 3 studies

```
analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory.mat
   recorded 4d11a2fdc1c985b15dea74cc234ed9d1f658960c1d42bfd8092770cc1b658832
   on disk  81244cf570e1f5382f3840f54bc5dec3cbd51fd803ed6f518b73b498fc39c4e1
analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat
   recorded c9d4d76649de605690acecad185875f400dedea6c287cd91d4ee8b644ea328c0
   on disk  4892c10ef17bca921063832d3edf1d08d15db10f739164ea9d388df143e52bf0
analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C400x50_trajectory.mat
   recorded fa0e714c712889a1442c8063d1ec809244bc42701883cf74366efb0a2019045e
   on disk  673be8c7e2ba868b56e18777f8edbdbb528b4c4872d56cf827594af8e6a27bb8
```

These break **G2 for all three** historical studies, because all three declare
them as required evidence.

### 3c. `FINAL_SHA256` self-verify, git-ignored containers — 2 further artifacts

```
analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat
analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat
```

named by `two_branch_controller_validation/FINAL_SHA256.txt`, breaking its G4.

### 3d. One Class C item, analysed and deliberately left

`three_rung_architecture/EVIDENCE.json` records the stale `baselines.json`
digest. Proven repairable and harmless (purely additive change, no value
modified) but **not applied**: the brief scopes Phase 4 to
*"ONLY FINAL_SHA256.txt"*, and repairing it would not change that study's gate,
which fails on §3b regardless. `HISTORICAL_HASH_REPAIR.md` §5.

## 4. Why the containers cannot be resolved here

`analysis/OlhoffCurrent/evidence/` is git-ignored wholesale, so these `.mat`
files never travelled with the repository; the copies on this host were
regenerated locally, and a `.mat` container is not byte-reproducible across
writes. Their **scientific** content is verified — every final-`rho` digest
matches the values committed in git-tracked `runs/*_record.json`, and the retry
independently re-verified C320 down to `RHO[:,1:352]` and `omega(1:2,1:352)` —
but the containers themselves cannot be made to match from here.

Two honest routes, and both are the owner's decision:

- **(a) transfer the originals** from the machine that produced them, so the
  recorded digests match again; or
- **(b) explicitly re-declare** the local containers, recording that the
  science was verified unchanged against the committed `rho_sha256` values.

**Route (b) was not taken unilaterally, and should not be.** Re-declaring a
digest to make a gate pass is precisely the move the finalization gate exists
to prevent; it is defensible only as a deliberate, documented decision by the
repository owner, never as a side effect of a promotion task.

## 5. `test_finalization_gate`

```
test_currentness           0 failures
test_source_integrity      0 failures
test_path_isolation        0 failures
test_preset_equivalence    0 failures
test_evidence_retention    0 failures
test_finalization_gate     4 failures
```

Self-tests **A–G all pass** — the gate is operational and fails closed. Cases
**H** and **I** fail, on the container classes above. Phase 6 requires 0
failures.

## 6. Verdict and consequence

```
PROMOTION_PROVENANCE_BLOCKED
PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED
NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```

STOP, per the frozen stop conditions. No scientific run was executed to work
around any of this, and none may be: every remaining blocker is a **file
transfer or an owner decision**, and compute cannot substitute for either.

The validated scientific result is untouched and remains
`THREE_RUNG_PRODUCTION_POLICY_VALIDATED [REUSED — NO NEW SCIENTIFIC RUN]`.
