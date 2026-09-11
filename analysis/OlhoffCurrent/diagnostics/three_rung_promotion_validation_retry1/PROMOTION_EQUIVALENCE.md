# PROMOTION_EQUIVALENCE — Part J

Part J proves that a promoted production configuration would be the validated
candidate. **No promotion occurred**, so no promoted configuration exists to
compare against, and the two Part-J verdicts cannot be issued as passes.

What *can* be established without a promotion, and is, is the half of Part J
that concerns the **code**: whether canonical production already dispatches to
the exact controller the candidate was tested with. It does.

C320 was **not** re-run for any of this. Everything below is a static audit.
Evidence: `evidence/dispatch.json`.

## 1. Production dispatches to the exact tested controller code — **PASS**

`olhoffcurrent_assert_dispatch()` passes: the published MMA wins, the
sensitivity filter wins, and no forbidden Olhoff tree resolves. Resolving each
controller file by its **qualified** name (a bare-name `which` cannot see inside
a `+package` and would report a false negative):

| Qualified name | Resolves to | Under production `+impl/`? | SHA-256 |
|---|---|:--:|---|
| `olh.move.exhaustion` | `+impl/architecture/+olh/+move/exhaustion.m` | ✅ | `17b37a384b1aa5d9…` |
| `olh.move.limit` | `+impl/architecture/+olh/+move/limit.m` | ✅ | `61fa923d430121ea…` |
| `olhoffSolve` | `+impl/architecture/olhoffSolve.m` | ✅ | `1e5a114cbf91717e…` |

`+impl` tree hash `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`,
identical to `meta.implTree` recorded **inside both trajectories** — the
candidate's and the oracle's. The code that would serve a promoted production
config is, byte for byte, the code that produced the validated result.

`PROMOTED_PRODUCTION_CODE_DISPATCH_PASS` — the code half of Part J.

## 2. Effective-configuration equivalence — **NOT ASSESSABLE**

The comparison Part J asks for is *promoted production config* versus
*validated candidate config*. Production has not been promoted, so the left-hand
side does not exist. What exists is production **as it stands today**, which
differs from the candidate by design:

```
production cfgHash  2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad
candidate  cfgHash  afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
```

Six schema differences at 320×40, of which three are policy and three are study
harness settings:

| Field | production | candidate | class |
|---|---|---|---|
| `move.levels` | `[0.04 0.02 0.01 0.005]` | `[0.04 0.02 0.01]` | **policy** |
| `move.continuation.signal` | `boundVariable` | `stageExhaustion` | **policy** |
| `stop.rule` | `designChange` | `stageExhaustion` | **policy** |
| `runtime.maxOuter` | 400 | 1600 | harness — the preregistered study cap |
| `runtime.diagnostics` | false | true | harness — required to rebuild the raw trajectory |
| `runtime.name` | `OLHOFF_CURRENT_320x40` | `TR3_C_320x40` | label; excluded from the config hash |

Reporting this as a pass would be a category error: it would assert equivalence
between the candidate and a configuration that does not yet exist. Reporting it
as a fail would be equally wrong — nothing failed; the step was not taken.

# `PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED`
# `THREE_RUNG_PROMOTION_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED`

## 3. What remains for Part J when provenance closes

The three policy fields above are the entire delta, and §1 has already
discharged the code half. A future Part J therefore reduces to:

1. apply the three-field configuration change;
2. re-resolve production and assert its `cfgHash` equals
   `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`
   on the scope-relevant fields, allowing only the declared harness settings;
3. re-run `tr_dispatch` and confirm §1 still holds;
4. re-run the software mechanics suite against the promoted config.

**No scientific run is required**, and C320 must not be repeated.
