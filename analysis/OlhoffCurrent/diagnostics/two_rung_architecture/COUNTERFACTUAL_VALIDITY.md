# COUNTERFACTUAL VALIDITY — why S2 needs no new run

The claim this document establishes:

> The state of the existing four-rung candidate trajectory at its **stage-2
> exhaustion declaration** is, exactly and not approximately, the terminal state
> the two-rung policy `move levels = [0.04, 0.02]` would produce.

It is argued from the source, then **checked** against the recorded traces.

---

## 1. The argument from the source

Under the two switches the candidate arm sets —
`move.continuation.signal = 'stageExhaustion'` and `stop.rule = 'stageExhaustion'` —
the entire dependence of the solver on `numel(cfg.move.levels)` is two lines:

**`+impl/architecture/+olh/+move/limit.m:107–124`**

```matlab
if strcmp(cfg.move.continuation.signal, 'stageExhaustion')
    if ~isempty(state.ex) && state.ex.declared && ...
            state.stage < numel(cfg.move.levels)          % <-- (i)
        ... descend: stage = stage+1, ex.stageStart = outer, cntA = cntB = 0 ...
    end
    mv = cfg.move.levels(state.stage);
    state.mv = mv;
    return
end
```

**`+impl/architecture/olhoffSolve.m:508–510`**

```matlab
if exhaustStop
    atLastLevel = hist.stage(outer) >= numel(moveLevels);   % <-- (ii)
    convOuter   = mvState.ex.declared && atLastLevel;
```

Nothing else on the active path reads the ladder's length:

* the move value at a stage is `cfg.move.levels(state.stage)` — a lookup by
  stage index, identical for stages 1 and 2 in both ladders;
* the stop guards at `olhoffSolve.m:477` are gated on `anyStopGuard && ~exhaustStop`
  and are therefore **inert** in this configuration, so the ladder-restoration
  guard (the one other place `moveLevels` appears, line 485) never executes;
* projection continuation is off (`projection.enabled = false`, and the solver
  refuses `stageExhaustion` under projection at all, line 141);
* `p` continuation is off, so the `pOwnCounter` block at line 305 — the only
  other reader of `moveLevels` — never runs;
* the detector `olh.move.exhaustion` is a pure observer of `rho` and `drho` and
  has no knowledge of the ladder;
* MMA, the sensitivities, the objective, the constraints, the eigensolver and
  the admission arithmetic are all functions of the design and the current move
  value alone.

Now compare `levels = [0.04, 0.02]` against `levels = [0.04, 0.02, 0.01, 0.005]`:

| iteration range | stage | predicate (i) `stage < numel` | predicate (ii) `stage ≥ numel` |
|---|---|---|---|
| `1 … kE(1)` | 1 | `1<2` = `1<4` = **true** | `1≥2` = `1≥4` = **false** |
| `kE(1)+1 … kE(2)−1` | 2 | `2<2`=false, `2<4`=true — but `ex.declared` is **false** here, so the branch is not taken either way | `2≥2`=true, `2≥4`=false — but `convOuter = declared && atLastLevel` and `declared` is **false**, so both give `convOuter = false` |
| `kE(2)` | 2 | first genuine divergence | first genuine divergence |

Both predicates are consumed only in conjunction with `ex.declared`. `ex.declared`
is false at every iteration strictly between the two declarations. Hence the two
runs are **bitwise identical for every iteration `k ≤ kE(2)`**, and first differ
at `kE(2)` itself: the four-rung run descends to `0.01`, the two-rung run sets
`convOuter = true` and stops with the design `ρ_{kE(2)}` that iteration produced.

**Therefore `S2 = row(kE(2))` of the existing trajectory, exactly.**

The same argument, with `numel(levels) = 1`, gives `S1 = row(kE(1))` — which is
how `move_ladder_necessity` obtained its single-stage endpoint.

## 2. The checks (PREREGISTRATION §7), run on the recorded traces

`scripts/tr_verify.py`, output in `evidence/event_verification.json`.

| check | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| 1 — stage 1 holds `move = 0.04` for all `k ≤ kE(1)`, `stage = 1` | ✅ | ✅ | ✅ |
| 2 — exactly one descent in `[1, kE(2)]`, at `kE(1)+1`, to `move = 0.02` | ✅ | ✅ | ✅ |
| 3 — stage 2 holds `move = 0.02` for all `k ∈ [kE(1)+1, kE(2)]`, `stage = 2` | ✅ | ✅ | ✅ |
| 4 — no `move ≤ 0.01` iteration at or before `kE(2)` | ✅ | ✅ | ✅ |
| 5 — `hist.exStageStart = kE(1)+1` throughout stage 2 (reset semantics identical) | ✅ | ✅ | ✅ |

All five hold on all three meshes.

## 3. The frozen-rule replay (PREREGISTRATION §5)

`scripts/tr_frozen.py` re-implements the frozen rule independently, with the
stage-local window and reset semantics transcribed from `exhaustion.m`, and
recomputes it from the **raw** trajectory (`RHO`, `hist.dxNorm2`) rather than
reading the controller's log back.

Required: element-wise agreement of `A`, `B`, `E`, `nA`, `nB` **in every stage**,
and exact agreement of every declaration.

| mesh | stages | `A`/`B`/`E`/`nA`/`nB` element-wise | declarations |
|---|---|---|---|
| 160×20 | 4 | ✅ all four stages | 102, 141, 180, 219 — all match |
| 320×40 | 4 | ✅ all four stages | 274, 313, 352, **none in stage 4** — all match |
| 400×50 | 4 | ✅ all four stages | 388, 427, 466, 505 — all match |

Nothing was tuned. `W = 20`, `P = 20`, `W_np = 10`, `tol = 0.05·√(NE/3200)`.

## 4. Index convention (PREREGISTRATION §6), stated once and used throughout

`kE(s)` is the iteration at which the detector **declares** — the iteration at
which `nA` or `nB` first reaches 20 in that stage. The descent is applied at
`kE(s)+1`. The terminal state of a policy whose last level is stage `s` is the
row at `kE(s)`.

| mesh | `kE(1)` (S1) | descent applied | `kE(2)` (S2) | descent applied | prior brief's "approximately" |
|---|---|---|---|---|---|
| 160×20 | **102** (A) | 103 | **141** (A) | 142 | "~102/103" ✅ |
| 320×40 | **274** (A) | 275 | **313** (B) | 314 | "~274/275" ✅ |
| 400×50 | **388** (B) | 389 | **427** (B) | 428 | "~388/389" ✅ |

The off-by-one in the earlier briefs is exactly this declaration/descent pair.
This study uses the **declaration** index everywhere, because that is the
iteration at which `olhoffSolve` sets `convOuter` and returns.

## 5. What this does *not* establish

The prefix argument is about the **two-rung policy versus the four-rung policy**,
which share a solver, a detector and a configuration in every respect except the
ladder's length. It says nothing about production, whose β-stall ladder takes a
different path from its first descent onward; every `P` comparison in this study
is against production's own recorded endpoint, not against a replayed predicate.
