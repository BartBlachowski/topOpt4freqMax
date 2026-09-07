# PROVENANCE — beta transition mechanism audit

**Mechanism audit. No optimisation was run. Nothing in production was changed.**

## Repository state at task start

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **Starting HEAD** | **`cb6c0eae31a25521f7c5fed1c4a89564ed63344e`** ("Measure design-activity scaling at a third mesh (400x50)") |
| `git status` at start | **clean** |
| Date | 2026-09-08 |
| MATLAB | `25.2.0.2998904 (R2025b)` |
| Threads | `maxNumCompThreads = 10` available; **no solver was invoked**, so thread count did not affect any result |
| Python | 3, with `numpy 2.3.4`, `h5py 3.16.0`, `matplotlib 3.10.7` |

HEAD was recorded before any other action, as required, and is **not** an
inherited value from an earlier task.

## Integrity gate — all requirements met

| requirement | result |
|---|---|
| OlhoffCurrent currentness | **`CURRENT`** (`localOk = 1`) |
| canonical source integrity | **PASS** — manifest `ok = 1`, **74/74** files |
| `+impl` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` |
| dispatch gate | **PASS** — all owned symbols resolve inside `+impl/` |
| sensitivity filter wins | **yes** — `applyFilter`, `prepFilter` → `+impl/filter/` |
| published MMA wins | **yes** — `mmasub`, `subsolv` → `+impl/mma_published/` |
| forbidden Olhoff implementations on path | **0** |
| `move_activity_400` evidence gate | **PASS** — 2 required, 2 present and hash-matching, 0 missing, 0 mismatched |

Verdict: the audit proceeded. `MECHANISM_AUDIT_PROVENANCE_FAIL` was not
triggered.

Also confirmed for completeness: `olhoffSolve` → `+impl/architecture/`,
`innerLoop` → `+impl/algo/`, `moveControl` → `+impl/algo/`.

## Scope compliance

Nothing under `+impl/` was read-modified: the tree hash above is the same value
recorded by `move_transition`, `move_activity_offline` and `move_activity_400`.
No configuration, preset, controller, tolerance, window, ladder, filter,
projection, `p`, mass law, multiplicity setting, eigensolver or FE routine was
altered. No new controller was written. No optimisation was run, so brief §14's
default (no new run) holds; §14's authorisation criterion was tested explicitly
and found not to apply — see `DATA_INVENTORY.md`, "What does NOT exist".

## Evidence used, and its integrity

This audit reads — and therefore **depends on** — the durable raw trajectories
produced by `move_activity_400`. They are declared `required` in this study's
own `EVIDENCE.json` so that `olhoffcurrent_evidence_gate` fails **this** study
too if they are ever lost:

| artifact | bytes | role here |
|---|---:|---|
| `evidence/move_activity_400/F400_400x50_trajectory.mat` | 102,057,786 | bound-active population (not derivable from the CSVs) |
| `evidence/move_activity_400/P400_400x50_trajectory.mat` | 36,576,273 | production-arm cross-check |

No new raw evidence was generated, so this study's own `evidence/` directory
holds only a pointer README; the shared artifacts live in the single durable
evidence root rather than being duplicated.

Committed scalar telemetry read (unmodified) from `move_stop`,
`move_transition` and `move_activity_400` is listed in `DATA_INVENTORY.md`.

## Predicate-replay validation

Because an audit of a rule is worthless if it audits a paraphrase, the stall
predicate was re-implemented from `+impl/architecture/+olh/+move/limit.m` and
replayed against the recorded `beta` histories. It reproduces **every actual
production descent at every mesh**:

| mesh | actual descents | replayed | match |
|---|---|---|---|
| 160x20 | 79, 90, 101 | 79, 90, 101 | **yes** |
| 320x40 | 130, 141, 152 | 130, 141, 152 | **yes** |
| 400x50 | 138 | 138 | **yes** |

## History

No history was rewritten. No previous diagnostic directory was modified; all
five prior studies re-verify against their own `FINAL_SHA256.txt`.
