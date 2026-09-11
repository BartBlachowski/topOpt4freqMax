# SINGLE_FACTOR_AUDIT — the candidate differs in exactly one computational field

## 1. Mechanical diff of the two resolved configurations

`scripts/tr_singlefactor.m` resolves both arms and compares **every one of the
81 schema leaves**. It does not trust `tr_config`'s claim; it reads the resolved
structs. Output: `evidence/singlefactor.json`.

| Class | Path | four-rung | three-rung |
|---|---|---|---|
| **COMPUTATIONAL** | `move.levels` | `[0.04 0.02 0.01 0.005]` | `[0.04 0.02 0.01]` |
| LABEL_ONLY | `runtime.name` | `CV_C_320x40` | `TR3_C_320x40` |

Nothing else differs. `runtime.name` is declared by the schema as
*"free-text run label; never read by solver mathematics"* and is **excluded from
`olhoffcurrent_config_hash`**, so the two config hashes differ by `move.levels`
alone:

```
four-rung   2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4
three-rung  afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
```

The four-rung hash is **exactly the `cfgHash` committed in
`two_branch_controller_validation/runs/C320x40_record.json`** — proof that the
baseline in this diff is the configuration that actually produced the oracle,
not a re-typed approximation.

### The locked scope, asserted field by field on the candidate

All 21 asserted and passing: `material.stiffness.p = 3`, p-continuation off,
`material.mass.model = eq4b`, `q = 1`, sensitivity filter applied to `all`,
`filter.radiusPhysical = 0.06`, projection off, subspace multiplicity of size 2
with diagonal offsets and off-diagonal terms, published MMA variant,
`move.policy = ladder`, `move.continuation.signal = stageExhaustion`,
`stop.rule = stageExhaustion`, `stop.tolerance = 0.05·sqrt(NE/3200)`,
`runtime.maxOuter = 1600`, `runtime.singleThread`, identical `design.initial`,
and `move.levels = [0.04 0.02 0.01]`.

Unchanged, therefore: **A, B, persistence, history/reset, p, mass
interpolation, q, filter, R, projection, multiplicity, MMA, FE, eigensolver,
objective, volume, initialization, deterministic/thread policy.**

# `THREE_RUNG_SINGLE_FACTOR_PASS`

## 2. Static prefix proof — every read of the ladder, audited

Complete enumeration over the promoted tree
(`grep -rn "move\.levels\|moveLevels" +impl --include='*.m'`, comments excluded),
17 sites:

### Sites in the executed solve path

| Site | Expression | Depends on ladder **length**? |
|---|---|---|
| `olhoffSolve.m:104` | `moveLevels = g('move.levels')` | no — a read |
| `olhoffSolve.m:312` | `mvNow = moveLevels(1)` | no — `= 0.04` for both; and the enclosing `if pOwnCounter` is **dead**, p-continuation is off |
| `olhoffSolve.m:485` | `~any(moveLevels(stage+1:end) > epsRMS)` | **not executed** — the enclosing block is `if anyStopGuard && ~exhaustStop`, and `exhaustStop` is true |
| **`olhoffSolve.m:509`** | `atLastLevel = hist.stage(outer) >= numel(moveLevels)` | **yes — only once `stage == 3`** |
| **`limit.m:109`** | `ex.declared && state.stage < numel(cfg.move.levels)` | **yes — only once `stage == 3`** |
| `limit.m:122` | `mv = cfg.move.levels(state.stage)` | no — stages 1–3 index `0.04 / 0.02 / 0.01` in both |
| `limit.m:148,152` | `min(stage+1, numel(levels))`, `levels(stage)` | **not reached** — the `stageExhaustion` branch `return`s at line 124, before the `boundVariable` / `designRms` code |

### Sites outside the executed path

`duOlhoffFrozenM4.m:55` (the preset default, overridden);
`fromLegacy.m:179`, `toLegacy.m:69` (legacy conversion, unused);
`validate.m:114` (requires `move.policy='ladder'` — satisfied);
`validate.m:164` (rejects a non-increasing ladder — `[0.04 0.02 0.01]` passes);
`schema.m:100` (bounds `[1 Inf]` — length 3 valid);
`describe.m:131` (printing).

### The argument

Only two sites can tell a three-rung ladder from a four-rung one, and both
reduce to the same comparison of `stage` against `numel(levels)`.

- **Stages 1 and 2.** `1 < 3` and `1 < 4` are both true; `1 >= 3` and `1 >= 4`
  are both false; likewise for 2. The two policies are indistinguishable.
- **Stage 3, before declaration** (iterations 314…351). `limit.m:109` is
  conjoined with `ex.declared`, which is false, so neither policy descends.
  `olhoffSolve.m:510` computes `convOuter = ex.declared && atLastLevel`, and
  `ex.declared` false makes it false **regardless of `atLastLevel`**. The two
  policies are *still* indistinguishable — the ladder length is inert even at
  the terminal stage, as long as the detector has not declared.
- **Iteration 352, the declaration.** `ex.declared` becomes true and the
  policies part for the first time:

  | | `atLastLevel` | `convOuter` | next |
  |---|---|---|---|
  | four-rung | `3 >= 4` = **false** | false | iteration 353 descends to `move = 0.005`, `stage = 4` |
  | three-rung | `3 >= 3` = **true** | **true** | **`CONVERGED @352`**, `limit.m` never called again |

**Ordering matters and is favourable.** The terminal-admission test sits at
`olhoffSolve.m:508–518`, *after* the design update, after the detector advance,
after every guard and after `hist.tOuter(outer)`; the `break` is at line 560.
So iteration 352's entire scientific state — `RHO(:,352)`, `omega(:,352)`, the
inner work, every telemetry column — is computed **before** the two policies can
differ. The difference is confined to what happens *next*.

# `THREE_RUNG_PREFIX_STATIC_PROOF_PASS`

Prefix equivalence through S3 is therefore a **structural property of the
code**, not merely an empirical expectation. Part E tests it empirically anyway,
because a static proof that is never confronted with a run is only half an
argument.

## 3. One derived-telemetry caveat, recorded in advance

`cv_telemetry.m` computes two **post-hoc counterfactual** columns,
`prodStageShadow` and `prodMoveShadow`, by replaying production's beta-stall
ladder over the realized trajectory. That replay reads `move.levels`, so those
two columns are parameterized by the ladder under test and **may legitimately
differ between the arms** without any solver difference.

They are diagnostics *about* the discarded production policy, not solver state.
They are therefore excluded from bitwise scientific equivalence alongside
`tOuter`, this exclusion is declared **before** the run (see
`PREREGISTRATION.md` §6), and `C320_PREFIX_EQUIVALENCE.md` reports whether they
actually differed.
