# CONTROLLER_RECOVERY — the exact frozen A/B rule (retry1)

Recovered in this retry from **executable and frozen repository sources**, not
from the task brief's prose, and cross-checked against the recovery the stopped
attempt performed (`three_rung_promotion_validation/CONTROLLER_RECOVERY.md`,
SHA-256 `119082321bf5ba56ffc39a36c76f7608e18a9b2c645a033da70a76475261bdac`,
verified unchanged). The two recoveries agree.

## 1. Authoritative sources

| Role | File | Status |
|---|---|---|
| **online (executed) detector** | `+impl/architecture/+olh/+move/exhaustion.m` | member of `+impl` tree `edbfe47e…`, manifest-verified |
| ladder descent + reset | `+impl/architecture/+olh/+move/limit.m` | same tree |
| terminal admission | `+impl/architecture/olhoffSolve.m` | same tree |
| retrospective predicates | `diagnostics/two_branch_maturity_240/scripts/tb_branches.m` | clean at HEAD |
| preregistration | `diagnostics/two_branch_maturity_240/PREREGISTRATION.md` | `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` |

The preregistration digest is the decisive link: `exhaustion.m`'s own PROVENANCE
comment cites *"SHA-256 6274822525…"*, and the file on disk hashes to exactly
that. The executed detector names the frozen document it implements, and the
document has not moved.

## 2. Constants — identical in both forms

```
W   = 20     median window
P   = 20     persistence, consecutive iterations
Wnp = 10     net/path window
tol = cfg.stop.tolerance = 0.05*sqrt(NE/3200)      mesh-scaled
     at 320x40, NE = 12800  ->  tol = 0.1          (confirmed: exh.tol = 0.1)
```

## 3. Signals — and the one documented divergence

With `d_k = rho_k − rho_{k−1}` (the designs actually visited) and
`n2(v) = ‖v‖/sqrt(NE)`:

```
cos(k)   = <d_k, d_{k−1}> / (‖d_k‖ ‖d_{k−1}‖)
net(k)   = n2(rho_k − rho_{k−10}) / sum_{j=k−9..k} n2(d_j)
med20 x  = median over [k−19, k], 'omitnan'
amp(k)   = the step amplitude
```

`amp` is the single place where the online and retrospective forms read a
different operand, and it is recorded here rather than smoothed over:

| Form | `amp` source |
|---|---|
| `exhaustion.m` (online, **executed**) | `norm(drho)` — the increment the inner sub-problem returned |
| `tb_branches.m` (retrospective) | `per.l2` — the realized design difference `‖d_k‖` |

These differ under box saturation, where the returned increment is clipped
before it becomes a realized step. The preregistration writes the clause as
`‖Δρ_j‖₂ < tol(NE)` — the update vector — so the **online reading is the
literal one**.

This creates **no ambiguity for this retry.** Both the oracle and the candidate
are produced by the *same* online detector, running from the *same* `+impl`
tree; the quantity under validation is that detector's behaviour, not a
retrospective re-scoring of it. No threshold, window, persistence length,
normalization or tolerance differs between the two forms.

## 4. The two branches

```
A(k) = med20 cos(k) < 0   AND  med20 net(k) < 0.5  AND  amp(k) >= tol
B(k) = med20 cos(k) > 0   AND                           amp(k) <  tol
E(k) = A(k) OR B(k)
```

NaN handling: `A` requires both medians non-NaN, `B` requires `medcos` non-NaN.
A and B are **mutually exclusive** at any single `k` (`amp >= tol` versus
`amp < tol`), so at most one persistence counter is ever running.

## 5. Declaration, persistence, indexing

- `cntA` / `cntB` increment on a true predicate and **reset to 0 on any false**.
- `E` is DECLARED at the first iteration `t` at which either counter reaches
  `P = 20`.
- `declIter = t`; `declBegin = t − P + 1`.
- Branch A is tested before B when both could declare on the same iteration
  (they cannot, being mutually exclusive, but the order is fixed regardless).

Three indices that are **not** the same number:

| Index | Meaning | C320 S1 |
|---|---|---|
| `declBegin` | first iteration of the sustaining window | 255 |
| **`declIter`** | iteration at which E is declared — **the oracle** | **274** |
| move-transition | iteration at which the new move is APPLIED | 275 |

## 6. Stage locality and reset

`s = ex.stageStart` is the first outer iteration executed at the current level.

- `amp(j)` defined for `j >= s`; `cos(j)` for `j >= s+1`; `net(j)` for `j >= s+9`.
- Medians defined only once the trailing 20-window lies **wholly inside** the
  stage: `j >= s + W − 1`.
- **ANCHOR exception:** `net` at `j = s+9` uses `rho_{s−1}`, the design at the
  moment the stage began. Only the anchor is pre-stage; every *step* is
  stage-local.

On descent, `limit.m` performs atomically: push `descents` row
`[outer, stageFrom, declIter, declBegin]`; `stage += 1`; `lastStage = outer`;
append `stageStarts`; `ex.stageStart = outer`; `cntA = cntB = 0`;
`declared = false`; append `events` / `eventBranch`; clear
`declIter` / `declBegin` / `declBranch`.

The reset can only **delay** a descent, never advance one. Verified as software
mechanics in `SOFTWARE_VALIDATION.md` (test: *descent resets the detector window
wholly to the new stage*).

## 7. Continuation authority and terminal admission

```
move.continuation.signal == 'stageExhaustion'   ladder descends on frozen E
stop.rule               == 'stageExhaustion'    terminal admission on frozen E
```

`olh.config.validate` **couples** them: `stop.rule='stageExhaustion'` requires
the matching move signal, *"or a declaration below the last rung is never
consumed."* Asserted as a test.

```
descent guard      (limit.m:109)        ex.declared && state.stage < numel(cfg.move.levels)
terminal admission (olhoffSolve.m:509)  atLastLevel = hist.stage(outer) >= numel(moveLevels);
                                        convOuter   = mvState.ex.declared && atLastLevel;
```

`beta` appears in **neither** branch. Under `stop.rule = 'stageExhaustion'` the
sec. 3.5.1 design-change test, `stop.guards.settledMove` and the restoration
guards are **replaced wholesale** — they remain computed and are recorded as the
counterfactual (`prodStopRaw`, `prodSettled`, `prodStopAdmit`). Projection is
refused under exhaustion (`olh:stop:exhaustionUnderProjection`).

That beta genuinely has no authority is not asserted from reading alone: a beta
history engineered to be a textbook stall is shown to descend the ladder under
the production `boundVariable` signal and to be **inert** under
`stageExhaustion` — see `SOFTWARE_VALIDATION.md`, tests 15–17.

## 8. The three-rung candidate is a CONFIGURATION-ONLY change

Complete audit of every `move.levels` read, and the proof of prefix inertness,
are in `SINGLE_FACTOR_AUDIT.md` §2. The candidate, in full:

```matlab
% = cv_config('C', 320, 40) -- the exact arm that produced the oracle -- plus one field
cfg = olh.config.resolve('duOlhoffFrozenM4', ov{:}, ...
        'move.levels',  [0.04 0.02 0.01], ...   % <- the ONLY new factor
        'runtime.name', 'TR3_C_320x40');        % <- label only, excluded from the config hash
```

where `ov` is `cv_config('C',320,40)`'s **own recorded override list**, replayed
rather than re-typed. Proof that the replay is exact: the resulting four-rung
configuration hashes to `2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4`,
which is the `cfgHash` committed in `two_branch_controller_validation/runs/C320x40_record.json`.

No new controller. No duplicate implementation. No source change.
