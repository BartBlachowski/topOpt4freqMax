# COUNTERFACTUAL VALIDITY — why S3 needs no new run

The claim established here:

> The state of the existing four-rung candidate trajectory at its **stage-3
> exhaustion declaration** is, exactly and not approximately, the terminal state
> the three-rung policy `move levels = [0.04, 0.02, 0.01]` would produce.

**The two-rung proof is not assumed to extend.** It is re-derived for three
rungs, a stronger independent fact is established by static audit, and all ten
preregistered checks are run against the recorded traces.

---

## 1. Complete inventory of ladder dependence in the source

Every site in `+impl/` (excluding `legacy/`) that reads `move.levels`, with its
activation state in the resolved candidate configuration:

| # | site | expression | active? | on **length**? | on **final-rung identity**? |
|---|---|---|---|---|---|
| 1 | `olhoffSolve.m:104` | `moveLevels = g('move.levels')` | yes | no | no |
| 2 | `olhoffSolve.m:312` | `mvNow = moveLevels(1)` | **no** | no | no |
| 3 | `olhoffSolve.m:485` | `~any(moveLevels(stage+1:end) > epsRMS)` | **no** | yes | **yes** |
| 4 | `olhoffSolve.m:509` | `atLastLevel = stage >= numel(moveLevels)` | yes | **yes** | no |
| 5 | `limit.m:109` | `state.stage < numel(cfg.move.levels)` | yes | **yes** | no |
| 6 | `limit.m:122` | `mv = cfg.move.levels(state.stage)` | yes | no | no |
| 7 | `limit.m:148,152` | β-stall descent branch | **no** | yes | no |
| 8 | `validate.m:164` | `any(diff(levels) > 0)` → error | config-time | shape only | no |
| 9 | `describe.m:131` | display string | display only | — | — |

**Site 3 is the only place the identity of the final rung can ever enter the
computation.** It is the ladder-restoration stop guard, and it is inert here for
**two independent reasons**, both measured from the resolved configuration
(`evidence/config_audit.json`, identical on all three meshes):

```
stop.guards.ladderExhausted = false
stop.guards.maxDesignChange = false      =>  anyStopGuard = false
stop.rule                   = 'stageExhaustion'  =>  exhaustStop  = true

    if anyStopGuard && ~exhaustStop      %  false && false  ->  never entered
```

This is a **stronger** result than the two-rung audit needed: there, site 3 was
argued inert because `exhaustStop` is true; here it is additionally inert because
no stop guard is enabled at all. Had `anyStopGuard` been true, site 3 would have
read the *tail* of the ladder and the three-rung and four-rung runs could have
differed before `kE(3)` — which is precisely why this brief required the check
rather than an inherited argument.

Site 2 never runs: `material.stiffness.continuation.enabled = false`, so
`pOwnCounter = false`. Site 7 never runs: the `stageExhaustion` branch of
`olh.move.limit` returns before reaching it. `state.mv` initialises to
`cfg.move.initial` — a config key independent of `move.levels` — and is
overwritten by site 6 on the first call.

## 2. The prefix argument

Sites 4 and 5 depend on the ladder's **length** only, and both are consumed
exclusively in conjunction with `ex.declared`:

```
limit.m:108     if ~isempty(state.ex) && state.ex.declared && stage < numel(levels)
olhoffSolve:510 convOuter = mvState.ex.declared && atLastLevel
```

`ex.declared` is false at every iteration strictly between two declarations, and
is reset to false at each descent. Therefore:

| iteration range | stage | `stage < numel` | `stage >= numel` | `ex.declared` | outcome in **both** ladders |
|---|---|---|---|---|---|
| `1 … kE(1)−1` | 1 | `1<3` = `1<4` = T | F / F | false | no descent, no convergence |
| `kE(1)` | 1 | T / T | F / F | **true** | descend to 0.02 in both |
| `kE(1)+1 … kE(2)−1` | 2 | T / T | F / F | false | identical |
| `kE(2)` | 2 | `2<3`=T, `2<4`=T | F / F | **true** | descend to 0.01 in both |
| `kE(2)+1 … kE(3)−1` | 3 | `3<3`=F, `3<4`=T | `3>=3`=T, `3>=4`=F | **false** | both give no descent **and** `convOuter = false` |
| **`kE(3)`** | 3 | F / **T** | **T** / F | **true** | **three-rung STOPS; four-rung descends to 0.005** |

The row that matters is the fifth: the two predicates take *different* values
there, but both are ANDed with `ex.declared = false`, so both ladders produce the
same result. The first genuine divergence is at `kE(3)`.

**Therefore the two runs are bitwise identical for every outer iteration
`k ≤ kE(3)`, and `S3 = row(kE(3))` of the existing trajectory, exactly.**

The same argument with `numel(levels) = 2` gives `S2 = row(kE(2))`, and with
`numel(levels) = 1` gives `S1 = row(kE(1))` — reproducing the two prior audits.

## 3. The three-rung ladder is a legal configuration

Resolving `move.levels = [0.04 0.02 0.01]` against the same preset and overrides
succeeds: `schema.m` permits a vector of length `[1 Inf]`; `validate.m` requires
non-increasing levels (satisfied) and requires `move.policy = 'ladder'` with
matching `stop.rule` and `move.continuation.signal` (satisfied). Field-for-field
diff against the four-rung resolution:

```
move.levels:          [0.04 0.02 0.01 0.005] -> [0.04 0.02 0.01]
provenance.overrides: cell -> cell            (the recorded override list; metadata)
```

Nothing else differs. No solver flag, tolerance, guard, filter, material,
multiplicity or optimizer setting changes.

## 4. The ten preregistered checks (PREREGISTRATION §7)

`scripts/tr3_verify.py`, output in `evidence/event_verification.json`.

| # | check | 160×20 | 320×40 | 400×50 |
|---|---|---|---|---|
| 1 | same initialization (`design.initial` = 0.5) | ✅ | ✅ | ✅ |
| 2 | `move = 0.04`, `stage = 1` for all `k ≤ kE(1)` | ✅ | ✅ | ✅ |
| 3 | exactly **two** descents in `[1, kE(3)]`, at `kE(1)+1` and `kE(2)+1` | ✅ | ✅ | ✅ |
| 4 | first descent is to `move = 0.02` | ✅ | ✅ | ✅ |
| 5 | `move = 0.02`, `stage = 2` throughout `[kE(1)+1, kE(2)]` | ✅ | ✅ | ✅ |
| 6 | second descent is to `move = 0.01` | ✅ | ✅ | ✅ |
| 7 | `move = 0.01`, `stage = 3` throughout `[kE(2)+1, kE(3)]` | ✅ | ✅ | ✅ |
| 8 | no `move ≤ 0.005` at or before `kE(3)` | ✅ | ✅ | ✅ |
| 9 | reset semantics: `exStageStart = kE(1)+1` in stage 2 and `kE(2)+1` in stage 3 | ✅ | ✅ | ✅ |
| 10 | divergence only at `kE(3)` (state there is stage 3 / move 0.01; next iteration is move 0.005) | ✅ | ✅ | ✅ |

All ten hold on all three meshes.

## 5. The frozen-rule replay (PREREGISTRATION §5)

`scripts/tr3_frozen.py` is **code-identical** to
`two_rung_architecture/scripts/tr_frozen.py` (verified by diff), itself a re-use
of `move_ladder_necessity/scripts/ml_frozen.py`, with the stage-local window and
reset semantics transcribed from `exhaustion.m`. It recomputes the rule from the
**raw** trajectory (`RHO`, `hist.dxNorm2`), not from the controller's log.

Required: element-wise agreement of `A`, `B`, `E`, `nA`, `nB` in every stage, and
exact agreement of every declaration — **including S1 and S2**.

| mesh | stages | `A`/`B`/`E`/`nA`/`nB` element-wise | declarations |
|---|---|---|---|
| 160×20 | 4 | ✅ all four | 102, 141, 180, 219 — all match |
| 320×40 | 4 | ✅ all four | 274, 313, 352, **none in stage 4** — all match |
| 400×50 | 4 | ✅ all four | 388, 427, 466, 505 — all match |

Cross-check against the two-rung audit's recorded events: `kE(1)` and `kE(2)`
reproduced exactly, with matching branch identities, on all three meshes.

## 6. Index convention (PREREGISTRATION §6) — one convention, stated once

`kE(s)` is the iteration at which the detector **declares** (`nA` or `nB` first
reaches 20 in that stage). The descent is applied at `kE(s)+1`. The terminal
state of a policy whose last level is stage `s` is the row at `kE(s)`, because
`olhoffSolve` sets `convOuter` inside that same iteration.

| mesh | `kE(1)` S1 | descent | `kE(2)` S2 | descent | stage-3 start | `kE(3)` S3 | offset | descent |
|---|---|---|---|---|---|---|---|---|
| 160×20 | **102** (A) | 103 | **141** (A) | 142 | 142 | **180** (B) | 38 | 181 |
| 320×40 | **274** (A) | 275 | **313** (B) | 314 | 314 | **352** (B) | 38 | 353 |
| 400×50 | **388** (B) | 389 | **427** (B) | 428 | 428 | **466** (B) | 38 | 467 |

No `+1` ambiguity survives: the declaration index is used for every state, every
comparison and every cost figure in this study.

## 7. Verdict

All ten checks pass on all three meshes; the static audit shows no active
dependence on final-rung identity; the replay reproduces every recorded event.

**`THREE_RUNG_COUNTERFACTUAL_EXACT`**

## 8. What this does *not* establish

The prefix argument concerns the three-rung policy versus the four-rung policy,
which share a solver, a detector and a configuration in every respect but the
ladder's length. It says nothing about production, whose β-stall ladder takes a
different path from its first descent onward; every `P` comparison in this study
is against production's own recorded endpoint, never a replayed predicate.
