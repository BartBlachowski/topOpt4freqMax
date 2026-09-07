# ARCHITECTURE RECOMMENDATION — a semantic move-controller schema

**Proposal only. Nothing here is implemented, and this task does not implement
it.** Brief sec. 15 asks for the design; sec. 20 forbids building it.

## The defect, stated precisely

The `move_transition` study could not express its experimental factor as
configuration. `cfg.move.continuation.signal` admits exactly two values,
`'boundVariable'` and `'designRms'` — both of which are *stall detectors on a
scalar history*. The candidate under test was a different kind of rule
altogether: a **persistence count on a distributional statistic of the design
increment**. There was no field for it.

The study therefore copied the solver. `mt_olhoffSolveT.m` is a 554-line
duplicate of `+impl/architecture/olhoffSolve.m` differing in two lines (the
signature, and the one call to the move controller), with the real experimental
logic in a parallel `mt_moveLimit.m`.

The visible consequence is in the study's own `CONFIG_DIFF.json`:

```json
"diff_armP_vs_armU": [ { "field": "runtime.name", "a": "MT_ARMP_160x20", "b": "MT_ARMU_160x20" } ]
```

**The two arms differ only in their name.** Both resolve to `cfgHash`
`0c9482af…`, and the study had to add a prose note conceding that "the
experimental factor is the move-transition controller, not a configuration
field". A configuration diff that cannot show what was varied is not a
configuration diff. As experimental evidence, backed by a passing bitwise
baseline gate, that was acceptable. As architecture it is not: the provenance
chain from "what we changed" to "what the config records" is broken.

A second, quieter defect: the ladder's transition rule is *fused to the ladder
policy*. `olh.move.limit`'s `case 'ladder'` contains the stall detector inline.
Policy (what levels exist) and transition (when to move between them) are
independent concerns that cannot currently vary independently.

## Proposed schema

Separate the three concerns the current design fuses — **levels**, **transition
rule**, **transition statistic** — and make the transition rule a discriminated
union on an explicit `type`.

```
move.policy                        enum {fixed, geometric, ladder, trustRatio}
move.levels                        vector, descending          [ladder only]
move.initial / minimum / geometric.* / trust.*                 [unchanged]

move.transition.type               enum {never, signalStall, designActivity}
move.transition.dwell              int   -- min iterations at a level before any descent

# type == 'signalStall'   (this is what production does today)
move.transition.signal.name        enum {boundVariable, designRms}
move.transition.signal.window      int
move.transition.signal.tolerance   double

# type == 'designActivity'
move.transition.activity.statistic       enum {maxUtilization, activeCount,
                                               participationNumber}
move.transition.activity.incrementThreshold  double  -- tau on |drho_e|; unused by maxUtilization
move.transition.activity.normalisation       enum {none, elementCount, meshPower}
move.transition.activity.normalisationExponent double -- meshPower only
move.transition.activity.threshold           double
move.transition.activity.persistence         int
```

### Production is exactly representable

Today's behaviour maps one-to-one, with no change in meaning:

| today | proposed |
|---|---|
| `move.policy = 'ladder'` | `move.policy = 'ladder'` |
| `move.levels = [0.04 0.02 0.01 0.005]` | unchanged |
| `move.continuation.signal = 'boundVariable'` | `move.transition.type = 'signalStall'`, `signal.name = 'boundVariable'` |
| `move.continuation.window = 10` | `move.transition.signal.window = 10` |
| `move.continuation.tolerance = 5e-3` | `move.transition.signal.tolerance = 5e-3` |
| the implicit `(outer - state.lastStage) > W` guard | `move.transition.dwell = 10`, now **explicit** |

That last row is worth flagging: the dwell requirement currently exists only as
an inline condition in `olh.move.limit`, invisible to configuration, to the
config hash, and to any diff. Promoting it makes an active part of production
behaviour auditable for the first time.

### The previous experiment is exactly representable

ARM U of `move_transition` becomes, with no solver copy:

```
move.transition.type                    = 'designActivity'
move.transition.activity.statistic      = 'maxUtilization'
move.transition.activity.threshold      = 0.5
move.transition.activity.persistence    = 10
move.transition.activity.normalisation  = 'none'
```

and `diff(armP, armU)` would then have shown five substantive fields instead of
`runtime.name`.

### The candidate this study narrows to is representable

```
move.transition.type                           = 'designActivity'
move.transition.activity.statistic             = 'activeCount'
move.transition.activity.incrementThreshold    = 1e-3
move.transition.activity.normalisation         = 'meshPower'
move.transition.activity.normalisationExponent = <undetermined; needs a third mesh>
move.transition.activity.threshold             = <undetermined>
move.transition.activity.persistence           = 10
```

The schema deliberately makes the two undetermined quantities *visible as named
fields* rather than burying them in code. A rule whose calibration is unresolved
should look unresolved in the configuration.

## Requirements from sec. 15, and how the design meets them

**No opaque names.** `move.continuation.signal` is not opaque, but the study
names around it are (`ARM U`, `M4`, `S2`, `R2`). Public config takes only
semantic values — `maxUtilization`, `activeCount`, `boundVariable`. Arm labels
stay in `runtime.name`, which is metadata and already excluded from scientific
comparison.

**No solver copies.** `olh.move.limit` gains a `transition` dispatch alongside
its existing `policy` dispatch, and receives the per-iteration statistics it
needs. `olhoffSolve` already stores `dxOuter`/`dxNorm2`/`move` in `hist` before
the controller's next call, so `maxUtilization` and `participationNumber` need
no new plumbing at all. `activeCount` is the one that does: it needs
`sum(|drho| > tau)` computed at line ~341 and stored in `hist` — one line, in the
production solver, recorded for every run whether or not the rule is active.

**Validation of incompatible fields.** The union must be enforced, not
documented. Setting `activity.*` while `type = 'signalStall'` is an error, not a
silently ignored field — silent ignoring is how an experiment ends up believing
it varied something it did not. Likewise `normalisationExponent` is an error
unless `normalisation = 'meshPower'`, and `move.levels` is an error unless
`policy = 'ladder'`. The existing schema table already carries a domain column,
so this is a `requires`/`forbids` predicate per row.

**Provenance of every field.** The schema's existing A/B/C/D classification
extends unchanged. Every field above is **C — pure reconstruction**: Du & Olhoff
(2007) place no bound on the increment beyond the box (25f), so neither the
ladder, nor its levels, nor any transition rule is attributable to the authors.
`olh.move.limit`'s header already says this; the new fields inherit it verbatim.
Nothing here should ever be described as the authors'.

**Resolved config diff exposes the experimental factor.** With the transition
rule as schema fields, `cfgHash` covers it and the existing field-by-field diff
machinery (`mt_configDiff.m`) reports it with no change. The `CONFIG_DIFF.json`
failure above becomes structurally impossible.

## What this does not fix

The schema makes the transition rule *expressible and auditable*. It does not
make it *correct*: the calibration gap identified in `CANDIDATE_STATISTICS.md`
(the `NE^alpha` exponent is unpinned by two meshes) is a physics question, not a
configuration question, and no schema resolves it.

One caution for whoever implements this. `move.transition.activity.threshold`
carries a mesh-dependent meaning under `normalisation = 'none'` or
`'elementCount'`. If the exponent turns out to be genuinely non-integer, a
configuration API that lets a user set a bare threshold without stating a
normalisation is a trap — the same number silently means different things at
different meshes. Making `normalisation` a required field with no default, so
that the choice must be stated explicitly at every use, is the cheap defence.
