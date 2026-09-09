# PROVENANCE — two-rung architecture audit

Everything here is measured, not asserted. The machine-readable form is
`evidence/provenance_start.json` (and `provenance_final.json`).

## Repository state

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| Working tree at start | **dirty**, 21 paths — the uncommitted deliverables of `two_branch_controller_validation` and `move_ladder_necessity`, plus this study's new directory. None of them is under `+impl/`. |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| Currentness | `CURRENT` |
| Source integrity | `PASS` — 0 mismatches, 0 missing, 0 extra |

The `+impl/` tree hash is **byte-identical** to the one recorded by
`move_ladder_necessity/METRICS.json`. The canonical implementation was not
touched by that study and is not touched by this one.

## MATLAB

| | |
|---|---|
| Available on this machine now | `25.2.0.2998904 (R2025b)` |
| Build that produced the C-arm trajectories | `25.2.0.3042426 (R2025b) Update 1` |

**This difference cannot affect any scientific number in this audit.** Zero
optimization runs are executed. MATLAB is used only to hash files, resolve the
production configuration for the tolerance-law identity check, and run the
provenance and finalization gates — all deterministic file operations. The
numerical analysis is Python, reading trajectories that were produced under
Update 1 and whose SHA-256 digests are recorded and verified.

## Dispatch and scientific lock

| check | result |
|---|---|
| `mmasub` resolves to | `analysis/OlhoffCurrent/+impl/mma_published/mmasub.m` — **published MMA** |
| Forbidden Olhoff trees on path | **none** |
| Filter type | `sensitivity` (wins) |
| Projection | off |
| `move.levels` (production) | `[0.04 0.02 0.01 0.005]` — **unchanged by this task** |
| Tolerance law | `stop.tolerance == 0.05·√(NE/3200)` verified identical at 160×20, 240×30, 320×40, 400×50 |

## Frozen preregistrations this audit inherits from

| study | live SHA-256 | recorded | match |
|---|---|---|---|
| `two_branch_maturity_240` (the frozen `A`/`B` rule) | `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` | quoted in `exhaustion.m` | ✅ |
| `two_branch_controller_validation` (acceptance gates §11) | `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf` | `evidence/PREREGISTRATION.sha256` | ✅ |
| `move_ladder_necessity` (materiality thresholds §6) | `a08b879b9d4893bc55dd9e5dfc2c26f897cb9d49355f0557ccc8644c7fc6ca3e` | `evidence/PREREGISTRATION.sha256` | ✅ |

**This study's own preregistration:** SHA-256
`b50455fbd3d97dcb72093fc13738f236d4d0f2e37a47caba0f515a16aaf7ca04`,
frozen `2026-09-09T15:57:27Z`, before any S2 endpoint value was read.

## Prior studies' finalization gates

| study | gate |
|---|---|
| `two_branch_controller_validation` | **PASS** |
| `move_ladder_necessity` | **PASS** |
| `two_branch_maturity_240` | **FAIL** — known, and pre-existing. Its raw 240×30 `.mat` artifacts are among the losses that motivated the finalization gate. This audit uses only that study's `PREREGISTRATION.md`, which is hash-valid, as the provenance anchor of the frozen rule; it uses **none** of its numeric evidence. |

## Phase-0 gate

| requirement | result |
|---|---|
| currentness `CURRENT` | PASS |
| source integrity `PASS` | PASS |
| published MMA resolution correct | PASS |
| forbidden Olhoff paths absent | PASS |
| sensitivity filter wins | PASS |
| tolerance law identical | PASS |
| four-rung production levels intact | PASS |
| inherited preregistration hashes valid | PASS |
| required raw evidence present + hashed | PASS (14/14) |
| controller-study finalization gate | PASS |

**`TWO_RUNG_EVIDENCE_GATE_PASS`.**

## Scientific runs

**Zero.** No `olhoffSolve` call, no `olhoffcurrent_run`, no arm of any mesh. The
only MATLAB executed is `tr_provenance.m` and the finalization gate, both of
which read and hash files. Everything else is Python reading `.mat` and `.csv`
files that already existed.
