# SHARED_IMPLEMENTATION_EQUIVALENCE — Part 12

```
SHARED_IMPLEMENTATION_EQUIVALENCE_PASS
```

## Byte level

79/79 files under the migrated `+impl` are byte-identical to the verified `git archive 253069…` snapshot (BYTE_IDENTITY.md). The shared implementation is therefore the upstream implementation, not an equivalent rewrite.

## Execution level

The same resolved configuration was run through the two trees independently.

- **Target (POST).** The worktree's OlhoffCurrent behind its fail-closed gate. `olhoffSolve` resolved to `…/topOpt4freqMax-migration-253069/analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m`, with `mmasub` proved to be the published copy.
- **Upstream (UP).** The read-only snapshot on a restored default path. `olhoffSolve` resolved to `…/scratchpad/mig/up253069/architecture/olhoffSolve.m`. Every solver symbol was asserted inside the snapshot, and no target-repository directory was on the path.

| configuration | target resolved via | upstream resolved via | configuration rows | result fields | values | verdict |
|---|---|---|---|---|---|---|
| historical β-stall (stage exhaustion **off**) | `olhoffcurrent_config(…,'Preset','duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered')` | `olh.config.resolve('duOlhoffFrozenM4', …)` | 87/87 equal | identical sets (top level and `hist`) | bitwise | **strict pass** |
| historical stage exhaustion (three rungs, **on**, diagnostics) | `…'Preset','duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered','Diagnostics',true` | `olh.config.resolve('duOlhoffFrozenM4', …, 'move.levels',[0.04 0.02 0.01], 'move.continuation.signal','stageExhaustion', 'stop.rule','stageExhaustion', cap 1600, diagnostics)` | 87/87 | identical, incl. the 12 `hist.ex*`, `res.exhaustion`, `res.diag` (all Δρ) | bitwise | **strict pass** |
| Pedersen adaptive box | `…'Preset','duOlhoffPedersenAdaptiveBoxSensitivityFiltered'` | `olh.config.resolve('duOlhoffAdaptivePedersen', …)` | 87/87 | identical, incl. `res.aux` | bitwise | **strict pass** |

"Strict" means no differing value, no field on one side only, and no differing configuration row (`passStrict`).

## Configuration layer (no solve)

For all 14 historical configurations and the Pedersen preset at nine meshes, the migrated wrapper's resolution equals upstream's on all 87 leaves (`evidence/config_transition.json`).

## Controller and suite level

The six root-independent upstream suites ran against the migrated `+impl`, under OlhoffCurrent's own gate with `olh.move.limit` asserted inside `+impl`: `test_config`, `test_mass`, `test_modules`, `test_preset_resolution_unchanged`, `test_stage_exhaustion` (fixtures, config rules and 160×20 wiring solves) and `test_outer_timing`. **0 failures**, identical to the same suites on the snapshot.

The p-continuation anchors A6/A7 (legacy `olhoffOpt` route, 250 outer): the target's science digest equals the upstream candidate's (KNOWN_PREEXISTING_DEFECTS.md).

**Conclusion.** Once a preset is resolved into a scientific configuration, OlhoffCurrent and upstream 253069 execute the same trajectory bit for bit. The wrapper layer adds naming, provenance and gating only.
