# DISPATCH_EQUIVALENCE — Phase 12

# `THREE_RUNG_PROMOTION_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED`

Phase 12 proves that **promoted** production dispatches to the exact controller
code the validated C320 run executed. There is no promoted production, so the
verdict cannot be issued as a pass or a fail.

The *code* half of the question, however, is fully answerable today and is
answered: the controller production resolves to is already, byte for byte, the
controller the validated run used. When promotion happens by configuration only
— the recommended route — this remains true by construction.

## 1. Dispatch audit — PASS

`olhoffcurrent_assert_dispatch()` passes: the published MMA wins, the
sensitivity filter wins, and no forbidden Olhoff tree resolves.

Each controller function resolved by its **qualified** name — a bare-name
`which` cannot see inside a `+package` and would report a false negative:

| Qualified name | Resolves to | Under production `+impl/`? |
|---|---|:--:|
| `olh.move.exhaustion` | `analysis/OlhoffCurrent/+impl/architecture/+olh/+move/exhaustion.m` | ✅ |
| `olh.move.limit` | `analysis/OlhoffCurrent/+impl/architecture/+olh/+move/limit.m` | ✅ |
| `olhoffSolve` | `analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m` | ✅ |

## 2. Tree identity with the validated run — PASS

```
+impl tree hash now                       edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
implTree recorded inside the C320 run     edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
MATCH                                     yes
```

The tree hash is also recorded inside the four-rung oracle trajectory, so the
oracle, the validated candidate and today's production all share one
implementation tree. Source manifest verifies: 75 files, `ok = 1`, currentness
`CURRENT`.

Individual A/B implementation hashes:

| File | SHA-256 |
|---|---|
| `+impl/architecture/+olh/+move/exhaustion.m` | `17b37a384b1aa5d987d9c861…` |
| `+impl/architecture/+olh/+move/limit.m` | `61fa923d430121ead764a229…` |
| `+impl/architecture/olhoffSolve.m` | `1e5a114cbf91717e01e5592e…` |

## 3. Why configuration-only promotion keeps this true

The three-rung change is provably inert in the code path: it alters no source
file, only two comparisons of `stage` against `numel(move.levels)` at
`limit.m:109` and `olhoffSolve.m:509`. Promoting via the OlhoffCurrent config
layer (route (b) of `PROMOTION_DIFF_PLAN.md` §5) leaves `+impl` byte-identical,
so §1 and §2 continue to hold without re-derivation.

Promoting instead by editing the upstream preset would change `+impl` and
therefore the tree hash that every existing piece of evidence records — which is
precisely why route (b) is recommended.

## 4. What remains for a future Phase 12

1. apply the three-field configuration change;
2. re-run `scripts/tr_policy_recovery.m` and confirm §1 and §2 unchanged;
3. confirm the resolved production config matches the validated candidate
   field-wise (`CONFIG_EQUIVALENCE.md` §3);
4. re-run the software-mechanics suite against the promoted config.

**No scientific run is required at any step, and C320 must not be re-run.**
