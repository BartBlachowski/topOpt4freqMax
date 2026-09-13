# PEDERSEN_PRESET — Part 6

**`duOlhoffPedersenAdaptiveBoxSensitivityFiltered` is a DISTINCT SCIENTIFIC METHOD VARIANT.** It is not a bug-fixed version of the eq. (4b) presets.

The scientific-delta audit showed why. Keeping the source controller but putting the target's SIMP + eq. (4b) law in (run M1 at 480×60) gave localized-mode collapses as soon as ρ crossed 0.1, and a false natural stop inside a spike (ω₁ 34.4). The Pedersen law is what makes the adaptive controller and its stop behave.

## Definition

The OlhoffCurrent preset delegates to the promoted upstream preset `olh.presets.duOlhoffAdaptivePedersen` with **no overrides**. Runtime defaults are those of every production run: cap 400, single thread, diagnostics off, verbose off. The upstream preset file is byte-identical at 6b08708 and 253069.

| field | value | upstream source line |
|---|---|---|
| `material.stiffness.model` / `linearBelow` | `pedersen` / 0.1 | `duOlhoffAdaptivePedersen.m` |
| `material.mass.model` | `eq2` (linear, q = 1) | ″ |
| `move.initial` | 0.10 | ″ |
| `filter.radiusPhysical` / `radiusElements` | 0.06 / [] | ″ (fixed physical radius) |
| `move.policy` | `adaptive` | parent `duOlhoffAdaptiveMove.m` |
| `move.minimum`, `move.adaptive.grow`, `move.adaptive.shrink` | 0.002, 1.2, 0.7 | ″ |
| `stop.guards.settledMove`, `stop.guards.boxInactiveFraction` | false, 0 | ″ |
| `stop.rule`, `move.continuation.signal` | `designChange`, `boundVariable` (inert under adaptive) — **stage exhaustion off** | schema defaults (253069) |
| everything else (p = 3 fixed, sensitivity filter on all f_sk, subspace N = 2 + offsets + off-diagonals, published MMA on the increment tol 0.05 5…500, ρ_min 1e-3, maxCluster 4, ε rule) | as `duOlhoffFrozenM4` | grandparent |

**Stage exhaustion is not inherited.** `validate.m` forbids the stage-exhaustion signal on a non-ladder policy (`olh:config:exhaustionNeedsLadder`). The preflight also asserts `stop.rule = designChange` and signal ≠ `stageExhaustion` for this preset.

## Fidelity to the committed source configuration

| check | result |
|---|---|
| resolved leaves vs the committed `repro/results/S*x*/res.mat` cfg, all nine sweep meshes | equal except `stop.rule` (absent at 6b08708; 253069 default `designChange` is the behaviour 6b08708 hard-wired), `runtime.verbose` (true in the sweep) and `runtime.name` |
| resolved leaves vs upstream 253069 `olh.config.resolve('duOlhoffAdaptivePedersen', …)`, nine meshes | all 87 leaves equal |
| source multiplicity behaviour | shared `+olh/+multi`, `algo/multRule.m`, `deltaLambda.m`: byte-identical to 253069 |
| source MMA behaviour | `algo/innerLoop.m`, `mma_published/*`: byte-identical |
| 160×20 solve | bitwise equal to committed S160x20 (PEDERSEN_S160_REPRODUCTION.md) |

## Epistemic status (carried by `olhoffcurrent_caveat`)

- It is a reconstruction. The controller and radius are class C; the material law is class B/D. Pedersen stiffness is the alternative Du & Olhoff §2.2 name, not their choice.
- Its natural stop is a heuristic design-change test, not a KKT certificate. The source endpoint was not shown to be more stationary than the historical one.
- Terminal bimodality is not expected beyond coarse meshes. In the committed R = 0.06 sweep, native gap₁₂ is 0.70 % at 160×20 and 11.8–24.5 % from 240×30 to 800×100.
- Native frequencies belong to the Pedersen/linear-mass model. Any cross-method table must name its evaluator model.
- Must not be labelled "Olhoff 2007".
