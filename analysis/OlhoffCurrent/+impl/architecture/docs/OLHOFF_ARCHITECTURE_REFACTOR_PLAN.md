# OLHOFF_ARCHITECTURE_REFACTOR_PLAN

Written before any solver edit, as required by Phase 8. Baseline
`2029baa` / tag `post-conference-baseline`.

---

## 1. Objective, restated as an acceptance test

After this work it must be possible to answer

> *What mathematical formulation am I running?*

by printing **one** effective configuration object — without reading audit
names, runner names, hidden branches or historical patches.

And every behavioural anchor listed in §7 must reproduce its pre-refactor
trajectory bitwise.

## 2. Module boundaries

```
+olh/
  +config/     schema, defaults, validation, legacy adapters, resolution
  +presets/    named compositions of canonical cfg — NO mathematics
  +material/   stiffness and mass interpolation (value + derivative)
  +filterx/    filter construction and the two filtering formulations
  +project/    density transform (tanh projection) and its chain rule
  +multi/      multiplicity method and the subeigenvalue problem
  +move/       move-limit policies and their continuation controller
  +stop/       convergence metric, guards, status precedence
olhoffSolve.m  canonical entry point: takes a canonical cfg
algo/, fem/,   numerical kernels, unchanged arithmetic
filter/, mma*/
```

`+filterx` rather than `+filter` because `filter` is a MATLAB built-in and a
package of that name would shadow it inside the package scope.

**The numerical kernels keep their present arithmetic byte for byte.** The
refactor changes *how a decision is expressed*, never *what is computed*.

## 3. Canonical cfg schema

Nested struct, one branch per scientific concept. Field names are semantic; no
experiment identifier appears anywhere.

```
cfg.domain.a|b|thickness
cfg.domain.mesh.nelx|nely
cfg.domain.boundary.condition        simplySupported | clampedSimple | clamped
cfg.domain.boundary.support          midHeight | corner | face
cfg.domain.boundary.axialRestraint   oneEnd | bothEnds
cfg.domain.element.type              Q4 | Q6
cfg.domain.element.massMatrix        consistent | lumped

cfg.material.solid.E|nu|density
cfg.material.stiffness.model         simp
cfg.material.stiffness.p
cfg.material.stiffness.continuation.enabled
cfg.material.stiffness.continuation.schedule
cfg.material.stiffness.continuation.driver        moveLadderStage | ownCounter
cfg.material.stiffness.continuation.blockStopUntilFinal
cfg.material.mass.model              eq2 | eq4 | eq4a | eq4b
cfg.material.mass.q
cfg.material.mass.lowDensityExponent
cfg.material.mass.cutoff
cfg.material.mass.continuation.enabled
cfg.material.mass.continuation.lowPModel          eq2 | eq4 | eq4a | eq4b

cfg.design.initial|minimum|volumeFraction

cfg.filter.type                      sensitivity | density | none
cfg.filter.radiusPhysical            [] or > 0
cfg.filter.radiusElements
cfg.filter.applyTo                   diagonal | all        (sensitivity only)

cfg.projection.enabled
cfg.projection.eta
cfg.projection.beta.levels
cfg.projection.continuation.trigger  outerConvergence

cfg.eigen.targetMode                 (n)
cfg.eigen.maxCluster                 (Nmax)
cfg.eigen.solver                     eigs | dense
cfg.eigen.tolerance|maxIterations|krylovFactor

cfg.multiplicity.method              binary | latch | hysteresis | subspace
cfg.multiplicity.tolerance
cfg.multiplicity.enterTolerance|exitTolerance
cfg.multiplicity.subspaceSize
cfg.multiplicity.diagonalOffsets     logical   <-- decoupled from .method
cfg.multiplicity.offDiagonal         logical

cfg.optimizer.inner.type             mma | lp
cfg.optimizer.inner.variable         increment | design
cfg.optimizer.inner.variant          published | asfound
cfg.optimizer.inner.tolerance|minIterations|maxIterations

cfg.move.policy                      fixed | geometric | ladder | trustRatio
cfg.move.initial|minimum|levels
cfg.move.geometric.ratio|afterCoalescence
cfg.move.trust.loRatio|hiRatio|shrink|grow
cfg.move.continuation.signal         boundVariable | designRms
cfg.move.continuation.window|tolerance

cfg.stop.norm                        l2 | max
cfg.stop.tolerance
cfg.stop.field                       designVariable        (documented, fixed)
cfg.stop.guards.settledMove          logical
cfg.stop.guards.ladderExhausted      logical
cfg.stop.guards.maxDesignChange      logical

cfg.runtime.maxOuter|singleThread|diagnostics|verbose|name
cfg.provenance.preset|overrides|resolvedAt|solverHashes
```

### 3.1 Couplings deliberately broken

| Was | Becomes | Why |
|---|---|---|
| `useOff = strcmp(multRule,'subspace')` | `cfg.multiplicity.diagonalOffsets` | Two independent choices: *how N is chosen* and *which subeigenvalue problem is solved*. |
| density filter reachable only via projection | `cfg.filter.type` and `cfg.projection.enabled` independent | The paper filtered **sensitivities**; substituting a density filter must be visible in the configuration. |
| `restorationGuard ∈ {R1,R2}` | three independent boolean guards | Experiment IDs out of solver mathematics. |
| `tolOuter` scaling retyped in 6 runners | `olh.config.epsilonForMesh` | One home for a scientific policy. |

### 3.2 Couplings deliberately kept

Mass continuation stays tied to the p schedule; p continuation may still be
driven by the ladder stall event; projection continuation still consumes the
outer convergence event. Each of these *defines* an experiment, and decoupling
them would change the science. They become named policy fields, not accidents.

## 4. Preset architecture

A preset is a pure function `cfg = preset(cfg)` that sets canonical fields and
nothing else. Presets may not contain mathematics, may not read run state, and
may not call the solver. `olh.config.resolve` records the preset name and the
explicit overrides in `cfg.provenance`.

Resolution order, and the only one:

```
canonical defaults  ->  preset  ->  explicit overrides  ->  validate  ->  effective cfg (immutable)
```

There is **no** other place a default may be supplied. In particular the five
`isfield` fallbacks currently inside `olhoffOpt.m` are removed, and the solver
never writes to its own configuration.

## 5. Validation architecture

`olh.config.validate` walks the schema and rejects:

* unknown fields at any depth (typo protection — the highest-value check here,
  because a mistyped `cfg.restorationGuard` silently disabled a guard);
* unknown enum values;
* wrong types, shapes, or out-of-range numbers;
* **incompatible combinations**, each with a named error identifier:
  * `projection.enabled` with `filter.type='sensitivity'`
  * `filter.type='density'` without projection **is allowed** (that is exactly
    the D160 identity control) but warns
  * `move.policy≠'ladder'` with `stop.guards.ladderExhausted`
  * `stiffness.continuation.driver='moveLadderStage'` with `move.policy≠'ladder'`
  * `mass.continuation.enabled` without `stiffness.continuation.enabled`
  * non-monotone `projection.beta.levels` or `stiffness.continuation.schedule`
  * `filter.radiusPhysical` and `filter.radiusElements` both empty
  * `subspaceSize` exceeding `maxCluster`
  * `optimizer.inner.type='lp'` with `projection.enabled`

Warnings — never coercion — for configurations that are valid but
scientifically suspicious (e.g. `multiplicity.tolerance > 0.1`,
`move.policy='ladder'` without `stop.guards.settledMove`).

**A configuration is never silently coerced into another realization.**

## 6. Compatibility layer

`olh.config.fromLegacy(flat)` maps every legacy field to its canonical home.
`olh.config.toLegacy(cfg)` is its inverse. Requirements:

1. `toLegacy(fromLegacy(x))` equals `x` for all twelve anchor configurations and
   for every stored `*.config.mat` in the audit tree.
2. `olhoffOpt` accepts either form: a flat struct is passed through
   `fromLegacy` first. Existing runners keep working untouched.
3. The adapter is *lossless in both directions* or it errors — it never guesses.

This is what makes it safe to leave every audit runner exactly as it is, which
Phase 23 requires.

## 7. Behavioural regression strategy

Twelve anchors at **160×20**, chosen to exercise every independent code path,
each with a saved pre-refactor reference artifact and a SHA-256 digest.

| Anchor | Exercises |
|---|---|
| `A1_frozen160` | the frozen conference realization verbatim |
| `A2_mature160` | R2 guard |
| `A3_r1ladder160` | R1 guard (ladder-dependent) |
| `A4_nodescent160` | S0, no move descent |
| `A5_pcont160` | p continuation coupled to the ladder |
| `A6_pdecoupled160` | p continuation decoupled, stall interception |
| `A7_massp160` | printed-mass continuation |
| `A8_projidentity160` | density filter + identity projection + chain rule + volFun |
| `A9_projection160` | full tanh projection and its continuation |
| `A10_binarydiag160` | binary multiplicity, diagonal-only filtering, fixed move |
| `A11_maxnorm160` | max-norm stopping |
| `A12_rhovar160` | `innerVar='rho'`, persistent MMA state |

**Equality standard.** Two digests are computed:

* `science` — densities, design variables, eigenfrequencies, volume, the
  complete non-timing history, every per-iteration `drho`, all continuation and
  multiplicity transitions, status and failure counters.
  **Must be bitwise identical. Anything else is a refactor failure.**
* `presentation` — log message text. Renaming `R2` to `maxDesignChange`
  necessarily rewords a message without changing a trajectory, so text is
  reported separately, together with `logShape`: the number of log events and
  the kind of each, which **must** stay identical.

Validation of the harness: `A1_frozen160` was verified before any edit to
reproduce the archived `TMA_160x20.mat` — the conference result itself — bitwise
in `rho`, `omega`, the full non-timing history and every per-iteration `drho`.

**Timing fields are excluded and only timing fields are excluded.**

Existing large evidence (`T320`, `T800`, `PD1_320x40`, `PM1_320x40`,
`B0/R1/R2_320x40`) is used as documentation. Phase 7 forbids launching an 800×100
regression unnecessarily and none is planned.

## 8. Migration order

Each step ends with the full anchor suite green before the next begins.

1. Canonical schema, defaults, validation. *No solver contact.*
2. Legacy adapters + round-trip test over all stored configs.
3. Presets.
4. Solver: resolve configuration once, up front; remove the five `isfield`
   fallbacks and the self-mutation; keep every branch predicate legacy-valued.
5. Mass and stiffness interpolation dispatch.
6. Filter dispatch: `sensitivity` vs `density` made explicit.
7. Multiplicity: split `method` from `diagonalOffsets`.
8. Move policy and its continuation controller.
9. Stopping: three named guards, explicit status precedence.
10. Projection continuation named.
11. Preset-driven runners.
12. Cleanup and classification of superseded files.

## 9. Files to delete eventually — none, yet

Phase 19 forbids deletion before equivalence. The classification is prepared but
not executed:

| Class | Files |
|---|---|
| `RETAIN_EVIDENCE` | every `audit_*/` tree, including its solver snapshots and `*_PRE_*.m` copies; `results/`, `runs/*.mat` |
| `RETAIN_COMPATIBILITY` | `algo/olhoffOpt.m` entry point, `algo/defaultCfg.m` (marked superseded), all audit runners |
| `ARCHIVE` | `top88.m` (byte-identical duplicate of `filter/top88_reference.m`) |
| `DELETE` | nothing in this pass |

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Reordering a floating-point expression changes the last bit | **high** | Numeric expressions are not touched. Only predicates and configuration plumbing change. |
| Splitting `R1`/`R2` into independent booleans changes semantics when both are set | medium | Legacy adapter sets exactly one; validation rejects `ladderExhausted` without a ladder. |
| Splitting `diagonalOffsets` from `method` changes the frozen path | medium | Adapter derives `diagonalOffsets = (method=='subspace')`, reproducing the coupling exactly for every legacy config. |
| Log rewording read as a trajectory change | medium | Separate `presentation` digest and a `logShape` invariant. |
| `useMMA` mutates global path state | medium | Left exactly as is; it is recorded in provenance. Out of scope. |
| Anchors too weak to catch a regression | medium | Digest covers every per-iteration `drho`, not just endpoints. |
| MATLAB licence unavailable mid-run | low | Occurred before (error −15.2); currently clear. Anchors are re-runnable and cheap at 160×20. |

## 11. Rollback

Every step is a separate commit on `architecture/canonical-config`. The
pre-refactor state is `main` at `2029baa`, additionally pinned by the tag
`post-conference-baseline` and by `EVIDENCE_MANIFEST.sha256` covering all 1507
files. Rollback is `git checkout post-conference-baseline`, or reverting a
single step's commit. No audit directory is written to at any point, so no
rollback can lose evidence.

## 12. Stop condition

Phase 8 requires stopping here if the proposed refactor cannot preserve the
historical realizations without changing their mathematics.

**It can.** Every historical realization is a point in the canonical schema:

| Realization | Canonical representation |
|---|---|
| frozen conference (TMA/B0/REG160) | defaults + `duOlhoffFrozenM4` |
| Bmature / R2 | + `stop.guards.maxDesignChange` |
| R1 | + `stop.guards.ladderExhausted` |
| no-descent | + `move.policy='fixed'` |
| P1 | + stiffness continuation, driver `moveLadderStage` |
| PD1 | + driver `ownCounter` |
| PM1 | + mass continuation `eq2` |
| D160 | + `filter.type='density'`, projection at β=0 |
| T160/240/320/800 | + projection β levels `[1 2 4 8]` |

No realization requires a branch that another realization cannot express as a
field value. **Proceed.**
