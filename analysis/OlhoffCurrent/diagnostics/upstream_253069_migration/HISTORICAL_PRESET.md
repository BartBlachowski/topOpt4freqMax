# HISTORICAL_PRESET — Part 5

```
HISTORICAL_PRESET_PRESERVED_PASS
```

## What "the historical OlhoffCurrent method" was — two configurations, kept apart

The prior audits use "historical OlhoffCurrent" for two different configurations, so the migration preserves each under its own name.

| | A. production formulation | A′. stage-exhaustion controller |
|---|---|---|
| canonical preset | `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` | `duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered` |
| what it was | `olhoffcurrent_preset()` → `duOlhoffFrozenM4` from 2026-09-07; the preset of the 2026-09-11 nine-mesh campaign | `duOlhoffFrozenM4` + three policy overrides, run by the C320/C480/C800 studies; validated, **never promoted** (`three_rung_promotion_closure/PROMOTION.md`: NOT PERFORMED) |
| resolves as | `olh.config.resolve('duOlhoffFrozenM4', mesh, runtime)` — unchanged | `olh.config.resolve('duOlhoffFrozenM4', mesh, 'move.levels',[0.04 0.02 0.01], 'move.continuation.signal','stageExhaustion', 'stop.rule','stageExhaustion', runtime)` — exactly the override set of `PROMOTION_DIFF_PLAN.md` |
| default cap | 400 | 1600, the preregistered study cap of every three-rung run |
| compatibility alias | `duOlhoffFixedPenaltySensitivityFiltered` | — |
| production eligible | yes (not selected) | **no** |

`duOlhoffFrozenM4.m` is byte-identical at 695f03b, 6b08708 and 253069; the migration did not touch it.

## Every historically frozen scientific field

The field-by-field assertions of `test_preset_identity` (§3) and the historical-branch assertions of `confbench_olhoff_assertions` pin A at 160×20:

| field group | A value |
|---|---|
| stiffness | `simp`, p = 3, continuation off (`material.stiffness.linearBelow` present but inert under simp) |
| mass | `eq4b`, r = 6, cut-off 0.1, continuation off |
| filter | sensitivity, applyTo all, R_phys = 0.06, R_el empty |
| projection | off |
| multiplicity | subspace, N = 2, offsets and off-diagonals on, tol 0.05 |
| eigen | eigs, maxCluster 4, tol 1e-12 |
| inner optimizer | MMA published, increment, tol 0.05, 5…500, asymptoteHistory `inner` |
| move | ladder [0.04 0.02 0.01 0.005], initial 0.04, signal `boundVariable`, window 10, tol 5e-3 |
| stop | rule `designChange`, l2, ε = 0.05·√(NE/3200), guards settledMove on (window 1), boxInactiveFraction 0, ladderExhausted off, maxDesignChange off |
| design | ρ₀ 0.5, ρ_min 1e-3, V 0.5 |

A′ differs from A in exactly `move.levels`, `move.continuation.signal`, `stop.rule` and the runtime cap.

## Preservation proof (configuration level; no solve above 160×20)

`evidence/config_transition.json` covers every recorded historical configuration: nine campaign meshes for A, C320/C480/C800 for A′, C160/C320 for the four-rung override.

- All 81 pre-migration schema leaves are equal by value, pre vs post: **14/14**.
- The 81-row hash recomputed from the migrated configuration reproduces the recorded hash: **14/14**.
- The six added schema rows take the values that select the pre-migration code path: **14/14**.
- The migrated wrapper resolution equals upstream 253069's resolution on all 87 leaves: **14/14**.

Scientific meaning unchanged. Reproduction at the solve level is in HISTORICAL_REPRODUCTION.md.

## What was deliberately not done

- No historical preset name was repointed.
- No material law, ρ_min, maxCluster, filter, MMA or multiplicity value changed.
- The three-rung controller was not promoted to production: it keeps its historical-diagnostic status.
- No historical evidence file was edited. `two_branch_controller_validation/FINAL_SHA256.txt` in particular was left as is (see TEST_REPORT.md).
