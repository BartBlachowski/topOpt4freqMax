# CONTROLLER_RECOVERY — the exact frozen A/B rule

Recovered from authoritative repository artifacts, **not** from the task brief's
prose. Recorded here so a re-attempt need not repeat the recovery.

## 1. Authoritative sources (consistent with each other)

| Role | File | SHA-256 |
|---|---|---|
| offline (retrospective) predicates | `diagnostics/two_branch_maturity_240/scripts/tb_branches.m` | see `FINAL_SHA256.txt` |
| **online (executed) detector** | `+impl/architecture/+olh/+move/exhaustion.m` | member of `+impl` tree `edbfe47e…` |
| ladder descent + reset | `+impl/architecture/+olh/+move/limit.m` | member of `+impl` tree `edbfe47e…` |
| terminal admission | `+impl/architecture/olhoffSolve.m` | member of `+impl` tree `edbfe47e…` |
| preregistration | `diagnostics/two_branch_maturity_240/PREREGISTRATION.md` | `6274822525…` (cited by `exhaustion.m`) |

`exhaustion.m` is the frozen rule of `two_branch_maturity_240/PREREGISTRATION.md`
sections 2–7 evaluated **forwards** in time; `tb_branches.m` is the same rule
evaluated retrospectively. **Consistency verified — no ambiguity.**
Status: `THREE_RUNG_CONTROLLER_DEFINITION_AMBIGUOUS` does **not** apply.

## 2. Constants

```
W   = 20     median window
P   = 20     persistence (consecutive iterations)
Wnp = 10     net/path window
tol = cfg.stop.tolerance = 0.05*sqrt(NE/3200)      (mesh-scaled)
     at 320x40, NE = 12800  ->  tol = 0.1          (confirmed: exTol column)
```

## 3. Signals

With `d_k = rho_k − rho_{k−1}` (designs actually visited) and `n2(v) = ‖v‖/sqrt(NE)`:

```
cos(k)   = <d_k, d_{k−1}> / (‖d_k‖ ‖d_{k−1}‖)
net(k)   = n2(rho_k − rho_{k−10}) / sum_{j=k−9..k} n2(d_j)
amp(k)   = ‖drho_k‖            <- the increment the INNER sub-problem returned,
                                  the inherited native measure (hist.dxNorm2),
                                  NOT the realized design difference d_k
med20 x  = median over [k−19, k], 'omitnan'
```

`amp` uses `drho` (sub-problem output); `cos`/`net` use `d` (realized design
motion). These differ under bound saturation and the distinction is load-bearing.

## 4. The two branches

```
A(k) = med20 cos(k) < 0   AND  med20 net(k) < 0.5  AND  amp(k) >= tol
B(k) = med20 cos(k) > 0   AND                           amp(k) <  tol
E(k) = A(k) OR B(k)
```

NaN handling: `A` requires both medians non-NaN; `B` requires `medcos` non-NaN.
A and B are **mutually exclusive** at any single k (`amp >= tol` vs `amp < tol`),
so at most one persistence counter runs at a time.

## 5. Declaration, persistence, indexing

- `cntA`/`cntB` increment on a true predicate and **reset to 0 on any false**.
- `E` is DECLARED at the first iteration `t` where either counter reaches `P = 20`.
- `declIter = t`, `declBegin = t − P + 1` (the window start `tb_branches` reports).
- Branch A is tested before B when both could declare on the same iteration.

Distinct indices to keep separate (they are NOT the same number):

| Index | Meaning | C320 S1 example |
|---|---|---|
| `declBegin` | first iteration of the sustaining window | 255 |
| `declIter` | iteration at which E is declared | 274 |
| move-transition | iteration at which the new move is APPLIED | 275 |

## 6. Stage locality and reset semantics

`s = ex.stageStart` is the first outer iteration executed at the current move level.

- `amp(j)` defined for `j >= s`; `cos(j)` for `j >= s+1`; `net(j)` for `j >= s+9`.
- Medians defined only once the trailing 20-window lies **wholly inside** the
  stage: `j >= s + W − 1`.
- **ANCHOR exception:** `net` at `j = s+9` uses `rho_{s−1}`, the design at the
  moment the stage began. Only the anchor is pre-stage; every *step* is stage-local.

On descent (`limit.m`), atomically: push `descents` row `[outer, stageFrom,
declIter, declBegin]`; `stage += 1`; `lastStage = outer`; append `stageStarts`;
`ex.stageStart = outer`; `cntA = cntB = 0`; `declared = false`; append
`events`/`eventBranch`; clear `declIter`/`declBegin`/`declBranch`.

The reset can only **delay** a descent, never advance one. It exists because a
mechanically halved amplitude across a move change would otherwise satisfy
Branch B and trigger a spurious descent.

## 7. Continuation authority and terminal admission

`olhoffSolve.m` exposes two independent switches, both defaulting to historical
behaviour:

```
move.continuation.signal == 'stageExhaustion'   ladder descends on frozen E
stop.rule               == 'stageExhaustion'    terminal admission on frozen E
```

Descent guard (`limit.m`):     `ex.declared && state.stage < numel(cfg.move.levels)`
Terminal admission (`olhoffSolve.m:509`):
```
atLastLevel = hist.stage(outer) >= numel(moveLevels);
convOuter   = mvState.ex.declared && atLastLevel;
```

`beta` appears in **neither** branch. Under `stop.rule == 'stageExhaustion'` the
sec. 3.5.1 design-change test, `stop.guards.settledMove` and the restoration
guards are **replaced wholesale**, not combined with (they remain computed and
recorded as the counterfactual: `prodStopRaw`, `prodSettled`, `prodStopAdmit`).

Projection is refused under exhaustion (`olh:stop:exhaustionUnderProjection`).

## 8. Consequence for the three-rung candidate — a CONFIGURATION-ONLY change

Every read of `move.levels` in the executed path:

| Site | Expression | Depends on ladder length? |
|---|---|---|
| `limit.m:122,152` | `cfg.move.levels(state.stage)` | No — values identical for stages 1–3 |
| `limit.m:109` | `state.stage < numel(cfg.move.levels)` | Only once `stage == 3` |
| `limit.m:148` | `min(stage+1, numel(levels))` | `boundVariable` path only — not used |
| `olhoffSolve.m:312` | `moveLevels(1)` | p-continuation restoration only — disabled |
| `olhoffSolve.m:485` | `~any(moveLevels(stage+1:end) > epsRMS)` | inside `~exhaustStop` — **not executed** |
| `olhoffSolve.m:509` | `hist.stage(outer) >= numel(moveLevels)` | Only once `stage == 3` |

Therefore, with `stop.rule = 'stageExhaustion'` and p-continuation off, ladder
length is **provably inert until `stage` reaches 3**. Prefix equivalence through
S3 is a structural property of the code, not merely an empirical expectation.

The candidate would have been, in full:

```matlab
olh.config.resolve('duOlhoffFrozenM4', ...
    'domain.mesh.nelx', 320, 'domain.mesh.nely', 40, ...
    'runtime.maxOuter', 1600, 'runtime.singleThread', true, ...
    'runtime.diagnostics', true, 'runtime.verbose', false, ...
    'move.continuation.signal', 'stageExhaustion', ...   % as validated C arm
    'stop.rule',                'stageExhaustion', ...   % as validated C arm
    'move.levels',              [0.04 0.02 0.01]);       % <- the ONLY new factor
```

i.e. **exactly the already-validated `cv_config('C', …)` arm plus one field.**
No new controller, no duplicate implementation, no source change.
