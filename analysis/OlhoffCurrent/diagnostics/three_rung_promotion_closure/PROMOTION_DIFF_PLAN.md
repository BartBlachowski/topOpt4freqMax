# PROMOTION_DIFF_PLAN — Phase 8

The complete, audited change set that promotion **would** apply. It was
produced **before** any promotion was attempted, and — because promotion is
blocked — no part of it was applied. Machine-readable:
`evidence/policy_recovery.json`.

## 1. Method

Every one of the 81 schema leaves was compared between canonical production at
320×40 and the recovered validated candidate, and each difference classified.
The audit does not assume the expected three fields; it enumerates and then
checks that nothing else appears.

## 2. The complete delta — 6 fields, 3 computational

| Class | Field | production | validated |
|---|---|---|---|
| **POLICY** | `move.levels` | `[0.04 0.02 0.01 0.005]` | **`[0.04 0.02 0.01]`** |
| **POLICY** | `move.continuation.signal` | `boundVariable` | **`stageExhaustion`** |
| **POLICY** | `stop.rule` | `designChange` | **`stageExhaustion`** |
| harness | `runtime.maxOuter` | `400` | `1600` |
| harness | `runtime.diagnostics` | `false` | `true` |
| label | `runtime.name` | `OLHOFF_CURRENT_320x40` | `TR3_C_320x40` |

```
UNEXPECTED computational differences: 0
minimal promotion: yes
```

The three POLICY fields are exactly the three the brief anticipated. **No
fourth computational field is required**, so no extra audit is owed.

The three non-policy differences are study-harness settings, not policy, and
would **not** be promoted: `runtime.maxOuter = 1600` was the preregistered study
cap, `runtime.diagnostics = true` was needed to rebuild the raw trajectory, and
`runtime.name` is declared by the schema as *"free-text run label; never read by
solver mathematics"* and is excluded from `olhoffcurrent_config_hash`.

## 3. Configuration-only — no code change

The validated controller code already exists in the promoted tree and is
already what production dispatches to (`DISPATCH_EQUIVALENCE.md`). Therefore:

- **no** controller implementation change;
- **no** change to A, B, persistence, windows, reset semantics, thresholds,
  normalization or tolerance scaling;
- **no** duplicate or "cleaned up" reimplementation;
- **no** rename that could alter dispatch;
- **no** additional stop rule.

## 4. Coupling that constrains how the change must be made

`olh.config.validate` enforces:

```
stop.rule = 'stageExhaustion'  REQUIRES  move.continuation.signal = 'stageExhaustion'
    "the same frozen rule must govern both the stage transition and the terminal
     admission, or a declaration below the last rung is never consumed."
```

So fields 2 and 3 must move **together or not at all**. A promotion that set one
without the other would be refused at resolve time — a useful property, and one
the software tests assert.

`validate` also requires `move.policy = 'ladder'` (already satisfied) and a
non-increasing ladder (`[0.04 0.02 0.01]` passes).

## 5. Where the change would be made

The canonical production realization delegates to the promoted upstream preset
`olh.presets.duOlhoffFrozenM4`, which currently declares
`'move.levels', [0.04 0.02 0.01 0.005]` at
`+impl/architecture/+olh/+presets/duOlhoffFrozenM4.m:55`.

A future promotion must decide, deliberately, between:

- **(a)** changing the **upstream preset** — which alters `+impl`, and therefore
  the tree hash `edbfe47e…` that every existing piece of evidence records; or
- **(b)** overriding in the **OlhoffCurrent production config layer**
  (`olhoffcurrent_config` / `olhoffcurrent_preset`), leaving `+impl` and its
  tree hash untouched.

**(b) is the better route** and is recommended: it keeps `+impl` byte-identical
to the tree the validated run executed, so dispatch equivalence stays trivially
provable and no prior evidence is invalidated. It also keeps the historical
beta-continuation path reachable as an explicit opt-in without a code change.

This choice is recorded as a recommendation, not executed — promotion is
blocked (`PROMOTION_PROVENANCE.md`).

## 6. Expected post-promotion hashes

Under route (b), after promotion the production configuration at 320×40 should
resolve to the validated candidate on every computational field, with
`runtime.*` harness settings at their production values. `+impl` tree must
remain `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`.

The production `cfgHash` would change from
`2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` to a new
value reflecting the three policy fields; it would **not** equal the candidate's
`afad9ea4…` because the harness fields `runtime.maxOuter` and
`runtime.diagnostics` legitimately differ. Config equivalence must therefore be
asserted **field-wise over the computational set**, not by comparing whole-config
hashes — see `CONFIG_EQUIVALENCE.md` §3.
