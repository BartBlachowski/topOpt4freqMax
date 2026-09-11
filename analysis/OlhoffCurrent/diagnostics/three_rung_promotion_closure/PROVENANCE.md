# PROVENANCE — three_rung_promotion_closure

## 1. Phase 0 inventory — recorded before anything was changed

| Field | Value |
|---|---|
| branch | `benchmark-methodology-r2` |
| starting HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` |
| final HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` — **no commit created** |
| MATLAB | `25.2.0.3042426 (R2025b) Update 1` |
| platform | `MACA64` (Darwin 25.6.0, arm64) |
| `maxNumCompThreads` default | 10 (production runs force 1; no run occurred here) |
| `+impl` tree hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| `+impl` source files | 75, manifest **verified** |
| currentness | **`CURRENT`** |
| production preset | `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |
| production `cfgHash` (320×40) | `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` |
| production policy | `[0.04 0.02 0.01 0.005]`, `boundVariable`, `designChange` |

Machine-readable: `evidence/inventory.json`.

### Starting dirty state — 4 paths, all pre-existing

```
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_record.json
?? analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation/
?? analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation_retry1/
```

**All four pre-existed this task.** The two modified paths are the known
timing-drift pair; the two untracked directories are the prior promotion
studies.

### Final dirty state

```
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/FINAL_SHA256.txt
?? analysis/OlhoffCurrent/diagnostics/three_rung_promotion_closure/
?? analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation/
?? analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation_retry1/
```

The two timing-drifted paths are **clean** (restored, Phase 5). One tracked file
is modified: the repaired hash manifest (Phase 4). One new untracked directory:
this study.

### Both prior promotion studies preserved unchanged

| Study | Files | Self-verify |
|---|---|---|
| `three_rung_promotion_validation` | 7 | all match its own `FINAL_SHA256.txt` |
| `three_rung_promotion_validation_retry1` | 48 study files + 1 git-ignored trajectory | all match its own `FINAL_SHA256.txt`; **G1–G5 PASS** |

Neither was modified, rewritten or re-run.

## 2. Phase 1 — validated evidence reuse

# `VALIDATED_C320_EVIDENCE_REUSE_PASS`

`three_rung_promotion_validation_retry1` passes its own finalization gate on
this host:

```
ok=1  G1=1 G2=1 G3=1 G4=1 G5=1
```

Its raw trajectory verifies against the digest recorded in its own manifest
(`fb29817e…ca5bbaffe`, 61 179 192 bytes), and all 48 study files verify.

Frozen verdicts recovered and reused, **not re-derived**:

```
DEPENDENCY_SPECIFIC_SCIENTIFIC_PROVENANCE_PASS
THREE_RUNG_SINGLE_FACTOR_PASS
THREE_RUNG_SOFTWARE_VALIDATION_PASS
C320_THREE_RUNG_PREFIX_EQUIVALENCE_PASS
C320_THREE_RUNG_TERMINATION_PASS
THREE_RUNG_PRODUCTION_POLICY_VALIDATED
```

with `scientific_runs = 1`, `CONVERGED @352`, `innerTotal = 6498`,
candidate `cfgHash afad9ea4…`, `implTree edbfe47e…`.

**No scientific run was executed to re-verify any of this.** Only stored hashes
were checked.

## 3. Phases 2–6 — the provenance work

| Phase | Verdict | Document |
|---|---|---|
| 2 — C240 transfer | **`ORIGINAL_C240_EVIDENCE_TRANSFER_FAIL`** | `C240_TRANSFER_VERIFICATION.md` |
| 3 — C240 finalization | **NOT REACHED** | — |
| 4 — historical hash repair | **`HISTORICAL_FINAL_SHA256_REPAIR_PASS`** | `HISTORICAL_HASH_REPAIR.md` |
| 5 — timing drift | **`TIMING_ONLY_DRIFT_SETTLED_PASS`** | `TIMING_TELEMETRY_AUDIT.md` |
| 6 — promotion provenance | **`PROMOTION_PROVENANCE_BLOCKED`** | `PROMOTION_PROVENANCE.md` |

Two of the three repairs available on this host were completed. The blocker is
not repairable here: it is a set of **six git-ignored container artifacts** that
must be transferred from the machine(s) that produced them, or an explicit owner
decision to re-declare them.

## 4. The zero-run lock

```
CANDIDATE SCIENTIFIC OPTIMIZATION RUNS IN THIS TASK: 0
NINE-MESH CAMPAIGN RUNS:                             0
C320 / C160 / C240 / C400 RE-RUNS:                   0
```

One deviation is disclosed rather than buried: `test_preset_equivalence`, a
pre-existing repository fixture that performs a 160×20 optimization of the
**production** preset, ran once before the skip rule was applied. It produced no
scientific claim, none is made from it, no artifact was retained, and it had no
effect on any verdict. Every later invocation skips it explicitly.
`SOFTWARE_VALIDATION.md` §0.

## 5. What was not touched

- **The controller** — `exhaustion.m`, `limit.m`, `olhoffSolve.m`; `+impl` tree
  hash identical at task start and end, manifest-verified.
- **A, B, persistence, windows, reset semantics, thresholds, normalization,
  tolerance scaling** — unchanged; the detector takes no ladder argument and is
  provably ladder-blind.
- **The scientific formulation** — all 18 audited fields identical between
  production and the validated candidate.
- **Production** — no canonical production path modified; production still
  resolves to the four-rung beta-continuation policy.
- **Both prior promotion studies** — verified intact.
- **The `provenance_start.json` / `provenance_final.json` snapshots** in several
  studies carry the same stale digests found in Phase 4. They are **historical
  records of state at run time** and were deliberately left alone; editing them
  would falsify history.
