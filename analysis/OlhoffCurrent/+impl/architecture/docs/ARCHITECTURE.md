# ARCHITECTURE

The post-conference architecture of the Du–Olhoff eigenfrequency
topology-optimization solver.

---

## 1. The question the architecture must answer

> *What mathematical formulation am I running?*

Before: you read the runner, then the audit report that named the runner, then
the solver branch the runner's label reached, then the `.mat` blob the runner
loaded. Four places, none of them the configuration.

After: you print one object.

```matlab
cfg = olh.config.resolve('duOlhoffFrozenM4', 'domain.mesh.nelx', 320, ...
                                             'domain.mesh.nely', 40);
res = olhoffSolve(cfg);
```

`cfg` states the formulation completely. `res.cfg` is that same object, recorded
with the result.

---

## 2. Layers

```
        olh.presets.*          named compositions.  NO mathematics.
               |
        olh.config.resolve     defaults -> preset -> overrides -> derived
               |               -> validate -> effective cfg (immutable)
               v
          olhoffSolve          the main loop of Fig. 1.
               |               Reads canonical fields, branches on SCIENCE.
               |               Never writes to its configuration.
      +--------+--------+--------------+---------------+
      v        v        v              v               v
 olh.material  olh.multi  olh.move   fem/, filter/   mma_published/
   mass and    multiplicity  move     unchanged        Svanberg,
   stiffness   and (25d)   policies   numerical        untouched
                                      kernels
```

and, to one side:

```
      olhoffOpt(flatCfg)   compatibility shim.  Translates the legacy flat
                           configuration and calls olhoffSolve.  Contains no
                           mathematics.  Every historical runner still works.
```

### 2.1 Why the numerical kernels were not rewritten

Phase 18 of the brief demands bitwise reproduction of every historical
trajectory. The cheapest way to guarantee that is not to touch a single
floating-point expression. So `fem/`, `filter/`, `mma*/`, `algo/innerLoop.m`,
`algo/genGrad.m` and `algo/deltaLambda.m` keep their arithmetic exactly, and the
refactor changes only:

* how a value reaches them (configuration), and
* how a branch decides which of them to call (predicates).

Every numeric expression in `olhoffSolve` is character-identical to the one it
replaces in `algo/olhoffOpt.m` at baseline `2029baa`.

---

## 3. Module boundaries

| Module | Responsibility | Knows about |
|---|---|---|
| `olh.config.schema` | every field, its type, domain, default and provenance class | nothing |
| `olh.config.defaults` | the single source of defaults | schema |
| `olh.config.validate` | refusals and warnings | schema |
| `olh.config.fromLegacy` / `toLegacy` | the only code that knows legacy field names | schema |
| `olh.config.resolve` | the resolution order | schema, presets |
| `olh.presets.*` | named realizations | canonical field names only |
| `olh.material.massInterpolation` | the printed mass laws, value **and** derivative | `cfg.material.mass` |
| `olh.multi.detect` | how N is chosen | `cfg.multiplicity` |
| `olh.move.limit` | move policy and its stall controller | `cfg.move`, `cfg.domain.mesh` |
| `olhoffSolve` | the outer loop of Fig. 1 | canonical cfg |
| `fem/`, `filter/`, `mma*/` | numerics | flat scalars and arrays |

**No module below `olh.config` knows any experiment identifier.** `R1`, `R2`,
`S0`–`S3`, `M0`–`M4`, `P1`, `PD1`, `PM1`, `T160`, `D160` appear in the solver
only inside comments and in one log string, never in a branch.

---

## 4. Configuration resolution — the one order

```
canonical defaults          olh.config.defaults, built from the schema
      |
      v
preset                      olh.presets.<name>, canonical fields only
      |
      v
explicit overrides          resolve('name','path.to.field',value,...)
      |                     an unknown path is an ERROR, not a new field
      v
derived rules               stop.tolerance recomputed here when
      |                     stop.toleranceRule = 'meshScaled', which is why a
      |                     mesh override automatically rescales epsilon
      v
validation                  refusals with named identifiers; warnings for
      |                     valid-but-suspicious; NEVER coercion
      v
effective cfg (immutable)   recorded in cfg.provenance and echoed in res.cfg
```

There is deliberately no other place a default may enter. In particular:

* the five `isfield` fallbacks that used to sit at the top of `olhoffOpt.m` are
  gone;
* the solver no longer writes `cfg.mmasubPath` or `cfg.rminEl` into its own input;
* `algo/defaultCfg.m` is **superseded** — it differed from the frozen
  realization in eleven scientific fields and was a second source of truth.

---

## 5. Design variable, filtered density, physical density

The projection work introduced three fields where the frozen realization has
one. Architecturally they are now distinct:

| Field | Symbol | Definition | Without projection |
|---|---|---|---|
| design variable | `z` | the optimizer's variable | *is* the density |
| filtered density | `z̃` | `(H z)/Hs` | does not exist |
| physical density | `ρ_phys` | `ρ_min + (1−ρ_min)·P(z̃)` | `= z` |

* `res.rho` is **always** the physical density the FE model used.
* Under projection `res.z`, `res.zTilde`, `res.rhoPhys` are recorded separately.
* `cfg.stop.field = 'designVariable'` states, in the configuration, which field
  the stopping rule monitors — faithful to §3.5.1, which monitors Δρ where ρ *is*
  the design variable, but **not** the physical change when projection is on.
  The physical change is recorded as `hist.dxPhys2`.

---

## 6. Continuation: three independent controllers

Move, penalization and projection continuation are distinct concepts and are
represented independently. There is deliberately **no** generic continuation
framework: the three differ in what they watch, what they advance and whether
they can veto convergence, and an abstraction over them would hide exactly those
differences.

| Controller | State | Transition criterion | Can block convergence? |
|---|---|---|---|
| move | `move.levels` index, re-arm clock | signal stalled over a window | no |
| penalization | schedule index | the move controller's stall event | **yes**, until p is final |
| projection | `beta.levels` index | the outer convergence event itself | **yes**, it consumes it |

Each records its state and its transitions in `hist`: `stage`/`move`,
`pStage`/`pEvent`/`pPen`, `projStage`/`projEvent`/`projBeta`.

The penalization controller has two drivers — `moveLadderStage` shares the move
ladder's index, `ownCounter` keeps its own while consuming the same event. That
is a *scientific* distinction (P1 vs PD1) and is a field, not two algorithms.

---

## 7. Stopping: separated concerns and explicit precedence

| Concern | Where |
|---|---|
| convergence metric | `cfg.stop.norm` over the design increment |
| threshold | `cfg.stop.tolerance` (+ `toleranceRule`) |
| guards | `cfg.stop.guards.{settledMove, ladderExhausted, maxDesignChange}` |
| continuation veto | the penalization and projection controllers |
| iteration cap | `cfg.runtime.maxOuter` |
| solver failure | inner-loop failure, recorded in `res.log` |

Status precedence is explicit in `res.status`:

```
SOLVER_FAILURE  >  CONVERGED  >  CAP_HIT  >  STOPPED_OTHER
```

A failure is reported even if the run also reached the cap; a run that met the
criterion is `CONVERGED` and one that never did but exhausted its budget is
`CAP_HIT`. The two are never conflated.

The three guards are independent booleans with semantic names. The historical
`restorationGuard ∈ {'R1','R2'}` — an experiment identifier inside solver
mathematics — is translated once, in `olh.config.fromLegacy`, and appears
nowhere else.

---

## 8. Filtering and projection, decoupled

`cfg.filter.type` and `cfg.projection.enabled` are independent:

| filter.type | projection | Meaning |
|---|---|---|
| `sensitivity` | off | **the printed choice**: Sigmund (1997) applied to the sensitivities |
| `density` | off | the identity control (D160): the filter switch alone |
| `density` | on | the projected realization |
| `sensitivity` | on | **refused** — the chain rule replaces the sensitivity filter, so this combination is not implemented, and validation says so |

Previously the density filter was reachable *only* by switching projection on,
so the departure from the paper's printed choice was invisible in the
configuration. Now it is a field.

---

## 9. Evidence and compatibility

* Every `audit_*/` directory is immutable evidence and is untouched.
* Every historical runner still works: it passes a legacy flat struct to
  `olhoffOpt`, which translates and forwards. `res.cfg` keeps its exact legacy
  contract, including the three mutations the old solver performed on its input.
* The pre-canonical solver is preserved verbatim at
  `architecture/legacy/olhoffOpt_PRE_CANONICAL.m`.
* The frozen conference reconstruction lives in the *other* repository
  (`topOpt4freqMax/analysis/OlhoffM4Reconstruction/+frozen/`) and is not on this
  tree's path at all. It cannot be reached by this refactor.

---

## 10. What the architecture deliberately does not do

* It does not fix the move ladder's stopping defect. `duOlhoffFrozenM4` keeps it,
  because that preset *is* the conference result. A corrected realization is a
  new preset, not an edit to this one.
* It does not choose between fixed p=3 and p continuation. Both are presets;
  `SCIENTIFIC_CONFIG_PROVENANCE.md` records that the paper's stated practice is
  the *continuation*, and the reconstruction's fixed p=3 is a ruling made on
  numerical evidence.
* It does not reconcile the paper's figure with the paper's numbers on support
  placement. `domain.boundary.support` records the choice and the provenance
  file records the contradiction.
* It does not generalize the printed mass coefficients. `eq4a`/`eq4b` refuse a
  cut-off or exponent their printed constants do not fit, rather than silently
  producing a differently-shaped law under a published name.
