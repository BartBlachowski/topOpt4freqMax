# COUNTERFACTUAL VALIDITY — S3 at 240×30 is exact

Phase 11. The claim:

> The state of the new C240×30 four-rung trajectory at its **stage-3 exhaustion
> declaration** is, exactly, the terminal state the three-rung policy
> `[0.04, 0.02, 0.01]` would produce at this mesh.

The brief required this be re-derived under the current effective configuration
rather than inherited from the 160/320/400 argument. It was — twice: statically
before the run, and against the realized trajectory after it.

---

## 1. Static audit, under the resolved 240×30 configuration (pre-run)

Every site in `+impl/` that reads `move.levels`, with its activation state
measured from the resolved 240×30 config (`evidence/single_factor.json`):

| site | expression | depends on | active at 240×30? |
|---|---|---|---|
| `olhoffSolve.m:104` | `moveLevels = g('move.levels')` | — | yes (read only) |
| `olhoffSolve.m:312` | `mvNow = moveLevels(1)` | — | **no** — `pOwnCounter = false` |
| `olhoffSolve.m:485` | `~any(moveLevels(stage+1:end) > epsRMS)` | **ladder tail / final-rung identity** | **no** — `anyStopGuard = false` **and** `exhaustStop = true` |
| `olhoffSolve.m:509` | `atLastLevel = stage >= numel(levels)` | length only | yes |
| `limit.m:109` | `stage < numel(levels)` | length only | yes |
| `limit.m:122` | `mv = levels(stage)` | — | yes (lookup; equal for stages 1–3) |
| `limit.m:148,152` | β-stall descent branch | length | **no** — the `stageExhaustion` branch returns first |

Measured flags at 240×30: `stop.guards.ladderExhausted = false`,
`stop.guards.maxDesignChange = false` ⇒ **`anyStopGuard = false`**;
`material.stiffness.continuation.enabled = false` ⇒ **`pOwnCounter = false`**;
`projection.enabled = false`.

**`olhoffSolve.m:485` is the only site whose value depends on which rung is
last, and it is inert here for two independent reasons.** This is the check that
could not be inherited: had a stop guard been enabled at this mesh, the
three-rung and four-rung runs could have differed *before* `kE(3)`.

The remaining two active sites depend on the ladder's **length** only, and both
are consumed exclusively in conjunction with `ex.declared`, which is false at
every iteration strictly between declarations and is reset to false at each
descent. Hence:

| iteration range | stage | `stage<numel` (3 vs 4) | `stage>=numel` (3 vs 4) | `ex.declared` | both ladders |
|---|---|---|---|---|---|
| `1 … 205` | 1 | T / T | F / F | false | identical |
| `206` | 1 | T / T | F / F | **true** | descend to 0.02 |
| `207 … 244` | 2 | T / T | F / F | false | identical |
| `245` | 2 | T / T | F / F | **true** | descend to 0.01 |
| `246 … 283` | 3 | **F / T** | **T / F** | false | identical — both ANDed with `declared = false` |
| **`284`** | 3 | F / T | T / F | **true** | **three-rung STOPS; four-rung descends to 0.005** |

Also confirmed: `move.levels = [0.04 0.02 0.01]` **resolves to a legal
configuration** at 240×30, differing from the four-rung resolution in
`move.levels` alone plus provenance metadata.

## 2. The ten checks against the realized trajectory (post-run)

`scripts/r240_verify.py`, output `evidence/event_verification.json`:

| # | check | result |
|---|---|---|
| 1 | same initialization, `design.initial = 0.5` | ✅ |
| 2 | `move = 0.04`, `stage = 1` for all `k ≤ 206` | ✅ |
| 3 | exactly **two** descents in `[1, 284]`, at 207 and 246 | ✅ |
| 4 | first descent is to `move = 0.02` | ✅ |
| 5 | `move = 0.02`, `stage = 2` throughout `[207, 245]` | ✅ |
| 6 | second descent is to `move = 0.01` | ✅ |
| 7 | `move = 0.01`, `stage = 3` throughout `[246, 284]` | ✅ |
| 8 | no `move ≤ 0.005` at or before 284 | ✅ |
| 9 | reset semantics: `exStageStart = 207` in stage 2, `246` in stage 3 | ✅ |
| 10 | divergence only at 284 (stage 3 / move 0.01; next iteration is move 0.005) | ✅ |

## 3. Frozen-rule replay

`scripts/r240_frozen.py` is **code-identical** to
`three_rung_architecture/scripts/tr_frozen.py` (verified by `diff`), which is
itself a re-use of `move_ladder_necessity/scripts/ml_frozen.py`. It recomputes
the rule from the **raw** trajectory (`RHO`, `hist.dxNorm2`) rather than reading
the controller's own log.

Required: element-wise agreement of `A`, `B`, `E`, `nA`, `nB` in **every** stage,
and exact agreement of every declaration.

| stage | move | range | declaration | trace `A`/`B`/`E`/`nA`/`nB` |
|---|---|---|---|---|
| 1 | 0.04 | 1–206 | **206** (B) | ✅ all five |
| 2 | 0.02 | 207–245 | **245** (B) | ✅ all five |
| 3 | 0.01 | 246–284 | **284** (B) | ✅ all five |
| 4 | 0.005 | 285–1358 | **1358** (B) | ✅ all five |

`replay_all_match = true`.

## 4. Index convention — one convention, off-by-one resolved explicitly

`kE(s)` is the **declaration** iteration (`nA` or `nB` first reaches 20 in that
stage). The descent is applied at `kE(s)+1`. A policy whose last level is stage
`s` terminates with the row at `kE(s)`.

| | stage start | first evaluable | earliest possible | **declaration** | window | descent applied |
|---|---|---|---|---|---|---|
| S1 | 1 | 20 | 39 | **206** | [187, 206] | 207 |
| S2 | 207 | 226 | 245 | **245** | [226, 245] | 246 |
| S3 | 246 | 265 | 284 | **284** | [265, 284] | 285 |
| F | 285 | 304 | 323 | **1358** | [1339, 1358] | — (terminal) |

**Cross-convention check.** The earlier `two_branch_maturity_240` study reported
events as the iteration at which the sustained window *begins*. Its fixed-move
240×30 arm recorded **187**. This run's stage-1 window begins at **187** —
identical — and declares at 206. The two conventions differ by exactly `P − 1 = 19`,
as expected, and the agreement is exact despite that arm running under a
different `+impl` tree and MATLAB build.

## 5. Verdict

Static audit clean, all ten checks pass, replay matches element-wise in all four
stages, and the independent cross-convention check on S1 agrees exactly.

**`C240_THREE_RUNG_COUNTERFACTUAL_EXACT`**
