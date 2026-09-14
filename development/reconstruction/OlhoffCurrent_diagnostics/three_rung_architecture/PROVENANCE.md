# PROVENANCE — three-rung architecture audit

Everything here is measured, not asserted. Machine-readable form:
`evidence/provenance_start.json`, `evidence/provenance_final.json`,
`evidence/config_audit.json`.

## Repository state

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` — re-read at task start, **not** inherited from a previous brief |
| Working tree at start | **dirty**, 22 paths: the uncommitted deliverables of `two_branch_controller_validation`, `move_ladder_necessity` and `two_rung_architecture`, the two uncommitted top-level gate files, and this study's new directory. **None is under `+impl/`.** |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| Currentness | `CURRENT` |
| Source integrity | `PASS` — 0 mismatched, 0 missing, 0 extra |

The `+impl/` tree hash is byte-identical to the values recorded by
`move_ladder_necessity` and `two_rung_architecture`. The canonical implementation
is not touched by this study.

## MATLAB

| | |
|---|---|
| Available on this machine now | `25.2.0.2998904 (R2025b)` |
| Build that produced the C-arm trajectories | `25.2.0.3042426 (R2025b) Update 1` |

**This difference cannot affect any scientific number here.** Zero optimization
runs are executed. MATLAB is used only to hash files, resolve configurations for
the static audit, and run the provenance / finalization gates and the test suite
— all deterministic. The numerical analysis is Python, over trajectories produced
under Update 1 whose SHA-256 digests are recorded and verified.

## Dispatch and scientific lock

| check | result |
|---|---|
| `mmasub` resolves to | `+impl/mma_published/mmasub.m` — **published MMA** |
| Forbidden Olhoff trees on path | **none** |
| Filter type | `sensitivity` (wins) |
| Projection | off |
| `move.levels` (production) | `[0.04 0.02 0.01 0.005]` — **unchanged by this task** |
| `move.continuation.signal` (production) | `boundVariable` — **unchanged** |
| Tolerance law | `stop.tolerance == 0.05·√(NE/3200)` verified identical at 160×20, 240×30, 320×40, 400×50 |

## Static configuration audit (Phase 3 input)

Resolved candidate configuration, all three meshes, identical:

| flag | value | consequence |
|---|---|---|
| `move.policy` | `ladder` | the ladder path is taken |
| `move.levels` | `[0.04 0.02 0.01 0.005]` | four-rung recorded arm |
| `move.continuation.signal` | `stageExhaustion` | `exhaustMove = true` |
| `stop.rule` | `stageExhaustion` | `exhaustStop = true` |
| `stop.guards.ladderExhausted` | **false** | |
| `stop.guards.maxDesignChange` | **false** | ⇒ **`anyStopGuard = false`** |
| `projection.enabled` | false | ⇒ projection block never runs |
| `material.stiffness.continuation.enabled` | false | ⇒ **`pOwnCounter = false`** |

Consequences, per mesh, all true:

* `olhoffSolve.m:485` — the **only** site whose value depends on the *identity*
  of the final rung — **never executes**, for two independent reasons
  (`anyStopGuard = false` *and* `exhaustStop = true`);
* `olhoffSolve.m:312` never executes (`pOwnCounter = false`);
* the projection guard block never executes.

The three-rung ladder `[0.04 0.02 0.01]` **resolves to a legal configuration**
(`schema.m` permits length `[1 Inf]`; `validate.m` requires non-increasing levels
and the paired signal/stop rule — all satisfied), and differs from the four-rung
resolution in exactly two fields: `move.levels`, and the recorded override list
under `provenance` (metadata, not science).

## Frozen preregistrations this audit inherits from

| study | live SHA-256 | match |
|---|---|---|
| `two_branch_maturity_240` (the frozen `A`/`B` rule) | `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` | ✅ (quoted in `exhaustion.m`) |
| `two_branch_controller_validation` (acceptance gates §11) | `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf` | ✅ |
| `move_ladder_necessity` (materiality thresholds §6) | `a08b879b9d4893bc55dd9e5dfc2c26f897cb9d49355f0557ccc8644c7fc6ca3e` | ✅ |
| `two_rung_architecture` (the audit this one follows) | `b50455fbd3d97dcb72093fc13738f236d4d0f2e37a47caba0f515a16aaf7ca04` | ✅ |

**This study's own preregistration:** SHA-256
`12c4bb960eeb6169521ea4b2e01b084a7bb3983d5387e94121e1729c2d71075c`,
frozen `2026-09-09T16:43:33Z`, before any S3 scientific value was read.

## Prior studies' finalization gates

| study | gate |
|---|---|
| `two_branch_controller_validation` | **PASS** |
| `move_ladder_necessity` | **PASS** |
| `two_rung_architecture` | **PASS** |
| `two_branch_maturity_240` | **FAIL** — known and pre-existing; its raw 240×30 `.mat` artifacts are among the losses that motivated the finalization gate. This audit uses only that study's hash-valid `PREREGISTRATION.md` as the provenance anchor of the frozen rule, and **none** of its numeric evidence. |

## Repository test suite

Run at task start and again at the end (`analysis/OlhoffCurrent/tests/`):

| test | start | end |
|---|---|---|
| `test_currentness` | PASS | PASS |
| `test_evidence_retention` | PASS | PASS |
| `test_finalization_gate` | **FAIL (self-referential)** | PASS |
| `test_path_isolation` | PASS | PASS |
| `test_preset_equivalence` | PASS | PASS |
| `test_source_integrity` | PASS | PASS |

**The start-of-task failure is created by this task and cured by finishing it.**
`test_finalization_gate` asserts that no *new* study directory fails the
finalization gate. At Phase 0 this study's directory existed with no deliverables
in it, so it failed — which is the fail-closed mechanism working as designed, not
a defect in the inherited evidence. This is recorded rather than excused: the
Phase-0 gate accepted it only under an explicit `testsPassOrSelfReference` flag
valid at tag `start` alone, and the **final** provenance record is required to
show `testsPass` true outright.

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
| inherited preregistration hashes valid (4/4) | PASS |
| required raw evidence present + hashed (17/17) | PASS |
| controller-study finalization gate | PASS |
| two-rung-study finalization gate | PASS |
| test suite | PASS (with the self-reference recorded above) |

**`THREE_RUNG_EVIDENCE_GATE_PASS`.**

## Scientific runs

**Zero.** No `olhoffSolve` call, no `olhoffcurrent_run`, no arm of any mesh. The
MATLAB executed is `tr3_provenance.m`, `tr3_configaudit.m`, `tr3_finalize.m` and
the test suite — reading files, hashing them, and resolving configurations.
Everything else is Python reading `.mat` and `.csv` files that already existed.
