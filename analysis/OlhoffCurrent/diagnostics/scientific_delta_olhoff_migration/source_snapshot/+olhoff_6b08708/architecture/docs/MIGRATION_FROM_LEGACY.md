# MIGRATION_FROM_LEGACY

How the old flat configuration maps onto the canonical one, and what still works
unchanged.

---

## 1. Nothing you already have is broken

Every historical runner keeps working with **no edit**. `olhoffOpt(flatCfg)` is
now a compatibility shim: it translates through `olh.config.fromLegacy` and calls
`olhoffSolve`. It preserves the legacy `res` contract exactly, including
`res.cfg` with the three mutations the old solver performed on its own input
(the five `isfield` defaults, `cfg.mmasubPath`, and `cfg.rminEl` recomputed from
`cfg.rminPhys`) — audit code reads `res.cfg.rminEl`.

Verified: all 35 legacy configurations in this tree round-trip losslessly
(`architecture/tests/test_legacy_roundtrip.m`).

Additive, and nothing legacy reads them: `res.cfgCanonical`, `res.cfgWarnings`,
`res.status`.

## 2. Field map

| Legacy flat | Canonical | Note |
|---|---|---|
| `a`, `b`, `t` | `domain.a`, `domain.b`, `domain.thickness` | |
| `nelx`, `nely` | `domain.mesh.nelx/nely` | |
| `bc` `'a'\|'b'\|'c'` | `domain.boundary.condition` `simplySupported\|clampedSimple\|clamped` | |
| `support` `'mid'` | `domain.boundary.support` `'midHeight'` | |
| `axial` `'one'\|'both'` | `domain.boundary.axialRestraint` `oneEnd\|bothEnds` | |
| `elemType`, `massType` | `domain.element.type`, `domain.element.massMatrix` | |
| `E`, `nu`, `rhom` | `material.solid.E/nu/density` | |
| `p` | `material.stiffness.p` | |
| `pSchedule` present | `material.stiffness.continuation.enabled` + `.schedule` | presence became a flag |
| `pDecouple` | `material.stiffness.continuation.driver` `ownCounter` vs `moveLadderStage` | |
| *(implicit)* | `material.stiffness.continuation.blockStopUntilFinal` | was unconditional |
| `massInterp` `'lin'\|'4'\|'4a'\|'4b'` | `material.mass.model` `eq2\|eq4\|eq4a\|eq4b` | see TERMINOLOGY §1 |
| `massLowP` present | `material.mass.continuation.enabled` + `.lowPModel` | |
| *(hard-coded)* | `material.mass.q`, `.lowDensityExponent`, `.cutoff` | now visible |
| `rho0`, `rhomin`, `volfrac` | `design.initial/minimum/volumeFraction` | |
| `filterMode` `'diag'\|'all'` | `filter.applyTo` `diagonal\|all` | |
| `filterMode` `'none'` | `filter.type` `'none'` | type, not scope |
| `rminPhys`, `rminEl` | `filter.radiusPhysical`, `filter.radiusElements` | |
| `projection.on` | `projection.enabled` **and** `filter.type='density'` | **see §3** |
| `projection.betaSchedule` | `projection.beta.levels` | |
| `projection.eta` | `projection.eta` | |
| `n`, `Nmax` | `eigen.targetMode`, `eigen.maxCluster` | |
| `solver` | `eigen.solver` | |
| `multRule` `'hyst'` | `multiplicity.method` `'hysteresis'` | spelled out |
| `multRule=='subspace'` | **also** `multiplicity.diagonalOffsets` | **see §3** |
| `tolMult`, `tolEnter`, `tolExit`, `subN` | `multiplicity.tolerance/enterTolerance/exitTolerance/subspaceSize` | |
| `offDiag` | `multiplicity.offDiagonal` | |
| `innerSolver`, `mmaVariant` | `optimizer.inner.type`, `.variant` | |
| `innerVar` `'drho'\|'rho'` | `optimizer.inner.variable` `increment\|design` | |
| `tolInner`, `minInner`, `maxInner` | `optimizer.inner.tolerance/minIterations/maxIterations` | |
| `moveFamily` `S0\|S1\|S2\|S3` | `move.policy` `fixed\|geometric\|ladder\|trustRatio` | |
| `move`, `moveMin`, `s2Levels` | `move.initial`, `.minimum`, `.levels` | |
| `s1Gamma`, `s1AfterCoal` | `move.geometric.ratio`, `.afterCoalescence` | |
| `s2Window`, `s2Tol` | `move.continuation.window`, `.tolerance` | |
| `s2Signal` `'beta'\|'drms'` | `move.continuation.signal` `boundVariable\|designRms` | |
| `s3Lo/Hi/Down/Up` | `move.trust.loRatio/hiRatio/shrink/grow` | |
| `outerNorm`, `tolOuter` | `stop.norm`, `stop.tolerance` | |
| `outerGuard` `'settledmove'` | `stop.guards.settledMove` (logical) | |
| `restorationGuard` `'R1'` | `stop.guards.ladderExhausted` | **see §3** |
| `restorationGuard` `'R2'` | `stop.guards.maxDesignChange` | **see §3** |
| `maxOuter`, `threads`, `diag`, `verbose`, `name` | `runtime.*` | |
| `mmasubPath` | *not configuration* | it was solver output written back into its input |

## 3. The three couplings the adapter resolves

These are the only places where one legacy field became two canonical ones. Each
is resolved in exactly one function, `olh.config.fromLegacy`, so every historical
run maps to its existing behaviour.

**`multRule=='subspace'` → `multiplicity.diagonalOffsets = true`.**
The old solver derived the diagonal-offset form of (25d) from the multiplicity
*rule name*. Two independent scientific choices — how N is chosen, and which
subeigenvalue problem is solved — shared one switch. `binary` + offsets, and
`subspace` + printed (25d), were both meaningful and both unreachable. They are
reachable now; `legacyBinaryDiagonal` uses the latter.

**`projection.on` → `filter.type = 'density'`.**
Turning projection on silently replaced the Sigmund **sensitivity** filter with a
**density** filter plus a chain rule. §1 of the paper says the filter was applied
to the sensitivities, so that substitution is a departure from a printed choice —
and it was invisible in the configuration. It is now a field, and
`projectionIdentity` isolates it.

**`restorationGuard ∈ {'R1','R2'}` → two independent booleans.**
Experiment identifiers inside solver mathematics. Validation rejects
`ladderExhausted` without a ladder, so the combination the old `switch` could not
express is refused rather than silently mis-run.

## 4. Fields the old configs omitted

Three fields the frozen configs predate. Emitting them is safe only because the
legacy code supplied exactly the same value, which the test checks rather than
assumes (`test_presets_match_history.m`):

| Field | Why it is safe |
|---|---|
| `s2Signal` | `moveControl.m:115` — absent or empty means `'beta'` |
| `tolEnter`, `tolExit` | `multRule.m:60,62` — read only by `multRule 'hyst'`, which no historical run used |

## 5. Writing new code

```matlab
% the old way -- still works, still supported
cfg = defaultCfg();  cfg.massInterp = '4b';  cfg.multRule = 'subspace';
res = olhoffOpt(cfg);

% the way to write new work
cfg = olh.config.resolve('duOlhoffFrozenM4', 'domain.mesh.nelx', 320, ...
                                             'domain.mesh.nely', 40);
res = olhoffSolve(cfg);
```

`algo/defaultCfg.m` is **superseded**. It is not any realization: it differs from
the frozen configuration in eleven scientific fields, including `multRule`,
`filterMode`, `moveFamily` and `massInterp`. Anything built from it is a
different algorithm from anything built from `CFG(k)`. It is retained only so
that historical scripts keep resolving.

## 6. Discovering what exists

```matlab
olh.presets.list()        % presets, classification, historical labels
olh.config.schema()       % every field, its domain, default and provenance class
olh.config.defaults()     % the canonical defaults
cfg.provenance            % preset, overrides and warnings recorded with the run
```
and `architecture/docs/CONFIG_REFERENCE.md`, which is generated from the schema.
