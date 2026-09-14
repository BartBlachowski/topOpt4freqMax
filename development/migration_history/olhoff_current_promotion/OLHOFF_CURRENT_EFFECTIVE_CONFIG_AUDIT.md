# OLHOFF_CURRENT_EFFECTIVE_CONFIG_AUDIT

What exact scientific configuration the current production preset in
`analysis/OlhoffCurrent` resolves to at runtime — resolved through the real
production path, not inferred from names or historical reports.

**Verification only. No file was modified.**

---

## 1. Source provenance

| | |
|---|---|
| Audit date | 2026-09-07 |
| Branch | `benchmark-methodology-r2` |
| HEAD | `cf1b71df808140871648c2a50d0ca47337fbe70a` |
| `git status` | clean except four pre-existing untracked historical trees (`OlhoffApproach`, `OlhoffApproachExact`, `OlhoffRegularized`, `OlhoffReproduced2007`) — unchanged by this audit |
| MATLAB | R2025b Update 1 (25.2.0.3042426) |
| Promoted from | `/Users/piotrek/Programming/Matlab/Olhoff`, branch `architecture/canonical-config`, commit `695f03bdac20c423a4e1d389cf9db9187597bcc3` |
| Promotion date | 2026-09-07 |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files) |
| Currentness | `CURRENT` (integrity PASS, 0 commits ahead upstream) |

### File hashes recorded before anything was read further

| File | SHA-256 |
|---|---|
| `OlhoffCurrent/olhoffcurrent_preset.m` | `301349abf03190768f51a3b66f75cd894e73e677025bee1a3d386044af3cdadf` |
| `OlhoffCurrent/olhoffcurrent_config.m` | `16b431039ec47a9131a5b9365d1e4e28628507cc88e986684bf982c28e462b50` |
| `+impl/architecture/+olh/+presets/duOlhoffFrozenM4.m` | `6ed3624cd19b3569f55b23acad1d5a1038d9c625a7fae9935e801ffb765ac40e` |
| `+impl/architecture/olhoffSolve.m` (canonical solver entry) | `5d4abd37c8b186a42d1a7ef5b7066c8bc2429d33ee91f20b0b844bace772324b` |
| `+impl/architecture/+olh/+config/resolve.m` | `d77d43fd0bcbefffc66524f2b35add5384c98e30fced1e068ceb4723d283717d` |
| `+impl/architecture/+olh/+config/schema.m` | `3ec54d8fae85c1526b6ae4ad3d7abbeee04b32d7c40c09f739bd678ae8fc544e` |
| `+impl/architecture/+olh/+config/fromLegacy.m` | `0883c6e08354c1cca45e52e017424c7648613abf97d84d2603acb0c873eac7c6` |
| **`+impl/filter/applyFilter.m`** (sensitivity filter) | `461ec7c7950b4d08e20c04d348d9d95674d3e42f1f167b05d3804c94abe057a7` |
| **`+impl/filter/prepFilter.m`** (filter weights) | `18ceb629c567fafcd90830c4eed3a4809fc2dda682d54cef1176b82bd6d47e1a` |
| `+impl/filter/projectDensity.m` | `817a0db22efdfe8ee0f8cace56a13d5c52a28cb4e875bd49880392745919c320` |
| `+impl/filter/projChain.m` | `1392d0030d51aa43a84b4a4a3b344741274008b316aa448915d362bb659ca655` |
| `+impl/filter/projDensityField.m` | `cc62399dee265b56a7ac27d669857361b340365b86d1336fa227566696a4b5fb` |
| `+impl/fem/assemble2D.m` | `7b2f1e10228d57237d962689909297351c879c9b613f311500f23dd37bd00878` |
| `OlhoffM4Reconstruction/olhoffm4_config.m` (frozen reference) | `49106f8448c010aff87122e391a1a668ec1bb13c2bab940ab98f1accd423393e` |

---

## 2. The production preset, and how production reaches it

Traced from `examples/Performance/performance_comparison.m`:

```
performance_comparison.m
  addpath(<repo>/analysis/OlhoffCurrent)
  olhoffcurrent_scrub_forbidden_paths(repoRoot)
      |
      +-- confbench_method_config('olhoff', nelx, nely, outDir)
      |       olhoffcurrent_paths()          <- fail-closed gate
      |       olhoffcurrent_preset()          <- the named preset
      |       olhoffcurrent_config(nelx,nely) <- olh.config.resolve(...)
      |
      +-- confbench_run_case('olhoff', mcfg, runOpts)
              runOlhoff -> olhoffcurrent_run(nelx, nely)
                             olhoffcurrent_paths()
                             olhoffcurrent_config(nelx, nely)
                             olhoffSolve(cfg)
```

| | |
|---|---|
| **Production preset name** | **`duOlhoffFixedPenaltySensitivityFiltered`** |
| Canonical preset function | `analysis/OlhoffCurrent/olhoffcurrent_preset.m`, which delegates to `+impl/architecture/+olh/+presets/duOlhoffFrozenM4.m` |
| Preset classification | `SCIENTIFIC_PRESET` |
| Historical aliases (provenance only) | `M4`, `TMA`, `B0`, `REG160`, `duOlhoffFrozenM4` |

### Runtime overrides applied by `performance_comparison`

`cfg.maxOuterOverride = []` by default, so `runOpts.max_outer_override` is
**not set** in a normal production run and no `MaxOuter` argument reaches
`olhoffcurrent_config`. The seven overrides recorded in `cfg.provenance` are
therefore all that a normal 160×20 run applies:

| Override | Value | Kind |
|---|---|---|
| `domain.mesh.nelx` | 160 | mesh |
| `domain.mesh.nely` | 20 | mesh |
| `runtime.maxOuter` | 400 | runtime (the preset's own value, restated) |
| `runtime.singleThread` | true | runtime |
| `runtime.diagnostics` | false | runtime |
| `runtime.verbose` | false | runtime |
| `runtime.name` | `'OLHOFF_CURRENT_160x20'` | label, never read by the mathematics |

**Not one scientific field is overridden.** Validation warnings: **0**.

### Does any production script reconstruct the preset manually?

**No.** `olhoffcurrent_config.m` contains no `'material.*'`, `'filter.*'`,
`'projection.*'`, `'multiplicity.*'`, `'move.*'`, `'stop.*'` or
`'optimizer.*'` override — only `domain.mesh.*` and `runtime.*`. There is
exactly one `olh.config.resolve` call in the whole production chain. The
architecture the brief asks for is the architecture in force:

```
production script -> named production preset -> explicit runtime-only overrides
                  -> resolve/validate -> solver
```

**One reporting observation, not a mismatch.** `printMethodSettings` in
`performance_comparison.m` (console output only, lines 576–588) prints the
operator-facing summary of the Olhoff row in **historical shorthand**: "M4
multiplicity", "continuation: S2 ladder", "signal beta (legacy)". It reads the
genuine resolved flat view, so nothing it prints is wrong — but it surfaces
exactly the codes §7 of the brief warns about, and it does **not** state the
filter *type* at all (it prints `mode=all`, which is `filter.applyTo`, not
`filter.type`). Cosmetic; **not changed**, per §10.

---

## 3. Effective configuration at 160×20

Resolved by `olhoffcurrent_config(160, 20)` through the production path.
Provenance classes are read from `olh.config.schema`:
**A** specified by a Du–Olhoff source · **B** implied by one ·
**C** under-specified reconstruction choice · **D** later experimental modification.

### MATERIAL

| Field | Value | Class |
|---|---|---|
| `material.stiffness.model` | `'simp'` | A |
| **`material.stiffness.p`** | **3** | A |
| `material.stiffness.continuation.enabled` | **false** | A |
| `material.stiffness.continuation.schedule` | `[]` | B |
| `material.stiffness.continuation.driver` | `'moveLadderStage'` (inert; continuation off) | C |
| **`material.mass.model`** | **`'eq4b'`** | A |
| **`material.mass.q`** | **1** | A |
| `material.mass.lowDensityExponent` (r) | 6 | A |
| `material.mass.cutoff` | 0.1 | A |
| `material.mass.continuation.enabled` | **false** | D |
| `material.mass.continuation.lowPModel` | `'eq2'` (inert; continuation off) | D |
| `material.solid.E` | 1e7 | A |
| `material.solid.nu` | 0.3 | A |
| `material.solid.density` | 1 | A |
| `design.initial` | 0.5 | A |
| `design.minimum` | 1e-3 | A |
| `design.volumeFraction` | 0.5 | A |

### FILTER

| Field | Value | Class |
|---|---|---|
| **`filter.type`** | **`'sensitivity'`** | **A** |
| `filter.radiusPhysical` | 0.06 | C |
| `filter.radiusElements` | `[]` (stored); **1.2 derived at 160×20** | C |
| `filter.applyTo` | `'all'` (every `f_sk`, off-diagonals included) | C |

Radius convention: **physical**, `R = 0.06·b`, converted at run time as
`rminEl = radiusPhysical / (b/nely) = 0.06 / 0.05 = 1.2` elements. Mesh-independent
by construction.

### PROJECTION

| Field | Value | Class |
|---|---|---|
| **`projection.enabled`** | **false** | D |
| `projection.type` | **not a schema field** — the form is not configurable; it is the tanh Heaviside map hard-coded in `projectDensity`/`projDensityField` | — |
| `projection.eta` | 0.5 (inert) | D |
| `projection.beta.levels` | `[]` (inert) | D |
| `projection.continuation.trigger` | `'outerConvergence'` (inert) | D |

### MULTIPLICITY

| Field | Value | Class |
|---|---|---|
| `multiplicity.method` | `'subspace'` — fixed subspace, **no threshold classifier** | C |
| `multiplicity.tolerance` | 0.05 | A |
| `multiplicity.subspaceSize` | 2 | C |
| `multiplicity.offDiagonal` | true (full determinant) | A |
| `multiplicity.diagonalOffsets` | true — `diag(λ_j − λ_n)` retained in (25d) | C |

### OPTIMIZER

| Field | Value | Class |
|---|---|---|
| `optimizer.inner.type` | `'mma'` | A |
| **`optimizer.inner.variant`** | **`'published'`** (Svanberg Sept-2007 constants) | B |
| `optimizer.inner.variable` | `'increment'` (Δρ) | C |
| `optimizer.inner.tolerance` | 0.05 | C |
| `optimizer.inner.minIterations` | 5 | C |
| `optimizer.inner.maxIterations` | 500 | C |

### MOVE / CONTINUATION

| Field | Value | Class |
|---|---|---|
| `move.policy` | **`'ladder'`** | C |
| `move.initial` | 0.04 | C |
| `move.levels` | `[0.04 0.02 0.01 0.005]` | C |
| `move.minimum` | 0.002 | C |
| `move.continuation.enabled` | **not a schema field** — continuation is implied by `move.policy`; `ladder` continues, `fixed` does not | — |
| `move.continuation.signal` | **`'boundVariable'`** (β of 25a) | C |
| `move.continuation.window` | 10 | C |
| `move.continuation.tolerance` | 5e-3 | C |

### STOPPING

| Field | Value | Class |
|---|---|---|
| `stop.field` | **`'designVariable'`** | A |
| `stop.norm` | **`'l2'`** | B |
| `stop.tolerance` | 0.05 at 160×20 | C |
| `stop.toleranceRule` | `'meshScaled'` → `0.05·√(NE/3200)`, re-derived after mesh overrides | C |
| `stop.guards.settledMove` | **true** | C |
| `stop.guards.ladderExhausted` | false | D |
| `stop.guards.maxDesignChange` | false | D |

Implied per-element RMS threshold: `0.05/√3200 = 8.8388347648318442e-04`,
constant across meshes.

### EIGEN

| Field | Value | Class |
|---|---|---|
| `eigen.targetMode` (n) | 1 | A |
| `eigen.maxCluster` (Nmax) | 4 → J = 5 modes extracted | C |
| `eigen.solver` | `'eigs'` | C |
| `eigen.tolerance` / `maxIterations` / `krylovFactor` | 1e-12 / 5000 / 4 — **present in schema but not wired**; `olhoffSolve` passes only the solver name to `eigSolve`, where these remain hard-coded (recorded upstream as technical debt) | C |

### RUNTIME / DOMAIN

`runtime.maxOuter` 400 [C] · `runtime.singleThread` true [C] ·
`runtime.diagnostics` false [C] · `runtime.verbose` false [C].
Domain 8 × 1 × 1, 160 × 20 [A/C]; `simplySupported`, `midHeight` support [C],
axial restraint `bothEnds` [C]; Q4 elements [C], `consistent` mass matrix [C].

**Effective configuration hash:** `ca9a5c90d5c412759b43ca5c44f8d6b3d3e13ca70bfde6d8f6e9cead0da7cde6`

---

## 4. THE FILTER QUESTION — settled directly

### Answer

```
filter.type = 'sensitivity'
```

**Not** `density`. This **agrees** with Du & Olhoff (2007) §1, which states the
Sigmund mesh-independent filter was applied to the **sensitivities** of the
objective functions. The schema classes this field **A** — specified by a
Du–Olhoff source — and `olh.config.describe` annotates it
`Sigmund (1997) SENSITIVITY filter [A]  <- the printed choice`.

### Config-level evidence: the solver's own branch predicates

Evaluated on the actual resolved production configuration, using the identical
expressions `olhoffSolve` uses (lines 93–95, 90):

```
useDensityFilter = strcmp(filter.type,'density')     -> 0
useSensFilter    = strcmp(filter.type,'sensitivity') -> 1
filterAll        = strcmp(filter.applyTo,'all')      -> 1
useProj          = projection.enabled                -> 0
```

### Execution-path evidence: what the solver *actually called*

A configuration value alone cannot exclude a solver that ignores it, so the
execution path was observed directly. **Execution-path probe, not a scientific
run:** `runtime.maxOuter` forced to 1, MATLAB profiler on, result discarded —
one outer iteration cannot converge and produces no optimization outcome. No
file was modified and no scientific number was taken from it.

| Function | Calls in one outer iteration | Resolved file |
|---|---|---|
| **`applyFilter`** | **4** | `<repo>/analysis/OlhoffCurrent/+impl/filter/applyFilter.m` |
| `prepFilter` | 1 | `<repo>/…/+impl/filter/prepFilter.m` |
| **`projChain`** | **0** | *never called* |
| **`projDensityField`** | **0** | *never called* |
| **`projectDensity`** | **0** | *never called* |
| `assemble2D` | 2 | `<repo>/…/+impl/fem/assemble2D.m` |
| `massScale` | 6 | `<repo>/…/+impl/fem/massScale.m` |
| `mmasub` / `subsolv` | 22 / 22 | `<repo>/…/+impl/mma_published/…` |
| `genGrad` | 4 | `<repo>/…/+impl/algo/genGrad.m` |

The count of **4** is exactly what the sensitivity branch predicts and the
density branch cannot produce: with `filterAll` and `N = 2`, the loop
`for s=1:N, for k=s:N` gives 3 calls, plus 1 for `f_JJ`.

### Which quantity is filtered

`applyFilter` is the top88 `ft == 1` **sensitivity** filter:

```matlab
function df = applyFilter(flt, rho, df)
rho = rho(:);
den = flt.Hs .* max(1e-3, rho);
for c = 1:size(df,2)
    df(:,c) = (flt.H * (rho .* df(:,c))) ./ den;
end
end
```

It **returns `df`** — the filtered sensitivity. `rho` enters only as the
weighting field in numerator and denominator; the function never returns a
modified density, and the caller never assigns its result to `rho`. In
`olhoffSolve` the call sites are `F(:,s,k) = applyFilter(flt, rho, F(:,s,k))`
and `fJJ = applyFilter(flt, rho, fJJ)` — generalized gradients, produced by
`genGrad` immediately above, filtered *after* the FE solve.

### Are densities filtered before FE assembly? — No

The only routine that maps a design variable to a filtered physical density is
`projDensityField`, and in `olhoffSolve` it is reachable **only** inside
`if useProj` (line 174). With `useProj = 0`:

* `projDensityField` was called **0 times**;
* `z` is never initialised (`if useProj, z = rho; end`, line 132);
* `rho` is never reassigned, so `assemble2D(mdl, rho, pNow, massNowCfg)`
  receives the **design variable itself**;
* `olh.config.describe` states it plainly: `design variable rho, and it IS the
  physical density`.

The two filter treatments are a single mutually exclusive `if useDensityFilter
… elseif useSensFilter … end` (lines 223–255), so both cannot run.

**It is therefore not possible for the configuration to say `sensitivity` while
the solver executes density filtering:** the density-filter helpers were
observed to execute zero times, and the sensitivity filter was observed to
execute on the gradient vectors the expected number of times.

---

## 5. Projection state

```
projection.enabled = false
```

Proof that the physical FE density is not subjected to a Heaviside projection:

| Evidence | Observed |
|---|---|
| `projDensityField` calls (the tanh three-field map) | **0** |
| `projectDensity` calls | **0** |
| `projChain` calls (the projection chain rule) | **0** |
| `res.z` present (set only inside `if useProj`) | **no** |
| `res.zTilde` present (set only inside `if useProj`) | **no** |
| `res.rhoPhys` present (set only inside `if useProj`) | **no** |
| `res.projFinalBeta` present | **no** |
| `hist.projBeta(1)` (sharpness in force) | **0** |
| `hist.projStage(1)` | 1 (never advanced) |
| `projection.beta.levels` | `[]` |

No scientific mismatch. Projection is class **D** — a later experimental
modification — and it is **off**, which is correct for the baseline
reconstruction.

---

## 6. Mass model

```
material.mass.model = 'eq4b'
```

**Yes — eq. (4b) is active**, the C¹ mass interpolation. Not `eq2`, not `eq4`,
not `eq4a`. Reported by its semantic name; the historical alias for the whole
realization is "M4", which is **not** the mass-model name and is not used here.

```
material.stiffness.p           = 3      (FIXED, no continuation)
material.mass.q                = 1
material.mass.lowDensityExponent (r) = 6
material.mass.cutoff           = 0.1
material.mass.continuation.enabled = false
```

Runtime confirmation from the probe: `hist.pPen(1) = 3` (the p actually used
for assembly) and `hist.massLow(1) = 0` (the terminal model, i.e. eq4b, in
force — not the low-p substitute).

---

## 7. Move-control and stopping policies, in semantic names

```
move.policy                   = ladder
move.initial                  = 0.04
move.levels                   = [0.04 0.02 0.01 0.005]
move.minimum                  = 0.002
move.continuation.signal      = boundVariable
move.continuation.window      = 10
move.continuation.tolerance   = 0.005

stop.field                    = designVariable
stop.norm                     = l2
stop.tolerance                = 0.05          (at 160x20)
stop.toleranceRule            = meshScaled
stop.guards.settledMove       = true
stop.guards.ladderExhausted   = false
stop.guards.maxDesignChange   = false
```

Runtime confirmation: `hist.move(1) = 0.04`, `hist.N(1) = 2`.

`S2` and `R2` appear nowhere in these values. They survive **only** as
historical aliases in `olhoffcurrent_preset.m`'s alias list and in the
console-summary text noted in §2.

**Known deficiency carried deliberately** (documented in the preset, not a
defect of this audit): under a move ladder the measured step reports the
schedule rather than the design on any iteration where the move limit changes.
`settledMove` suppresses the symptom; it does not remove the cause.

---

## 8. Comparison with the frozen conference realization

Method: `olhoffm4_config(160,20)` produces the frozen flat configuration; it is
translated into canonical terms by the audited adapter `olh.config.fromLegacy`
(covered by `test_legacy_roundtrip` over 35 configurations, the three frozen TMA
configs included) and compared field-by-field against the resolved production
configuration. This compares *scientific content*, not profile IDs or hashes.

**All 80 canonical schema fields compared. 2 differ; neither is scientific.**

| Field | OlhoffCurrent production | Frozen conference | Match? |
|---|---|---|---|
| stiffness `p` | 3 | 3 | **YES** |
| p continuation | false | false | **YES** |
| mass model | `eq4b` | `eq4b` | **YES** |
| mass `q` | 1 | 1 | **YES** |
| mass low-density exponent `r` | 6 | 6 | **YES** |
| mass cutoff | 0.1 | 0.1 | **YES** |
| mass continuation | false | false | **YES** |
| **filter type** | **`sensitivity`** | **`sensitivity`** | **YES** |
| filter radius (physical) | 0.06 | 0.06 | **YES** |
| filter radius (elements) | `[]` (derived 1.2) | `[]` (derived 1.2) | **YES** |
| filter scope | `all` | `all` | **YES** |
| **projection enabled** | **false** | **false** | **YES** |
| multiplicity method | `subspace` | `subspace` | **YES** |
| multiplicity tolerance | 0.05 | 0.05 | **YES** |
| subspace size | 2 | 2 | **YES** |
| off-diagonal terms | true | true | **YES** |
| (25d) diagonal offsets | true | true | **YES** |
| inner optimizer | `mma` | `mma` | **YES** |
| MMA variant | `published` | `published` | **YES** |
| inner variable | `increment` | `increment` | **YES** |
| inner tolerance | 0.05 | 0.05 | **YES** |
| move policy | `ladder` | `ladder` | **YES** |
| move levels | `[0.04 0.02 0.01 0.005]` | same | **YES** |
| move stall signal | `boundVariable` | `boundVariable` | **YES** |
| stopping field | `designVariable` | `designVariable` | **YES** |
| stopping norm | `l2` | `l2` | **YES** |
| stopping tolerance | 0.05 | 0.05 | **YES** |
| guard `settledMove` | true | true | **YES** |
| guard `ladderExhausted` | false | false | **YES** |
| guard `maxDesignChange` | false | false | **YES** |
| `maxOuter` | 400 | 400 | **YES** |
| `stop.toleranceRule` | `meshScaled` | `explicit` | **representational — see below** |
| `runtime.name` | `OLHOFF_CURRENT_160x20` | `OLHOFF_M4_160x20` | label only, never read by the mathematics |

### `stop.toleranceRule` — investigated, not waived

The **tolerance itself is 0.05 in both**. The difference is in how it is carried,
and it is an artifact of the translation, by design:

* `olh.config.fromLegacy` line 199 sets `'explicit'` unconditionally, with the
  stated reason *"A legacy config carries a NUMBER, not a rule. Preserve it
  verbatim: several audit runners computed it with the mesh-scaling law and
  several did not, and re-deriving it here could change a stored tolerance in
  its last bit."*
* `olhoffm4_config` computes `cfg.tolOuter = 0.05*sqrt(NE/3200)` itself and
  stores the resulting number; the canonical configuration carries the rule and
  lets `resolve` derive the number after mesh overrides. **Same law, two
  encodings.**
* **`stop.toleranceRule` is never read by `olhoffSolve`** (verified by grep). It
  is consumed only by `olh.config.resolve` at build time and by `describe`. It
  cannot affect a trajectory.

Verified rather than argued — both routes were evaluated at five meshes:

| Mesh | OlhoffCurrent (`meshScaled`) | Frozen (stored number) | Bitwise |
|---|---|---|---|
| 160×20 | 0.050000000000000003 | 0.050000000000000003 | **YES** |
| 240×30 | 0.075000000000000011 | 0.075000000000000011 | **YES** |
| 320×40 | 0.10000000000000001 | 0.10000000000000001 | **YES** |
| 480×60 | 0.15000000000000002 | 0.15000000000000002 | **YES** |
| 800×100 | 0.25 | 0.25 | **YES** |

Bitwise identical at every mesh, 800×100 included. **Not a scientific
difference.**

### Independent corroboration

This config-level comparison is consistent with the trajectory-level evidence
already on record from the promotion (`OLHOFF_CURRENT_PROMOTION_REPORT.md`):
`PROMOTION_EQUIVALENCE_PASS`, with 28 shared result fields bitwise identical at
both 160×20 and 320×40 against a freshly executed frozen-M4 run **and** the
saved nine-mesh conference record (`ω₁ = 169.49522702153845` and
`165.95078925220545`; outer 91/131; inner 2241/2614). No new optimization was
run for this audit.

---

## 9. Path-isolation check

Run under the production path setup, with `tools/Matlab`,
`three_method_parametric_study` and `conference_bench` added as
`performance_comparison` adds them.

| Check | Result |
|---|---|
| Gate `ok` | **1** — 0 blockers |
| Owned symbols verified | **32**, all resolving inside `+impl` |
| `analysis/OlhoffCurrent` visible | **yes** |
| `analysis/OlhoffM4Reconstruction` executable | **no** |
| `/Users/piotrek/Programming/Matlab/Olhoff` executable | **no** |
| Any other historical Olhoff tree executable | **no** — none of the 14 forbidden roots is on the path |
| Correct MMA variant wins | **yes** — `mmasub` → `+impl/mma_published/mmasub.m` |
| Correct filter implementation wins | **yes** — `applyFilter`, `prepFilter` → `+impl/filter/` |

`which -all` on the decisive symbols:

```
olhoffSolve       <repo>/analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m
olhoffOpt         <repo>/analysis/OlhoffCurrent/+impl/algo/olhoffOpt.m
applyFilter       <repo>/analysis/OlhoffCurrent/+impl/filter/applyFilter.m
                    ALSO: <matlabroot>/toolbox/shared/slcoverage/.../@cvdata/applyFilter.p
prepFilter        <repo>/analysis/OlhoffCurrent/+impl/filter/prepFilter.m
projChain         <repo>/analysis/OlhoffCurrent/+impl/filter/projChain.m
projDensityField  <repo>/analysis/OlhoffCurrent/+impl/filter/projDensityField.m
projectDensity    <repo>/analysis/OlhoffCurrent/+impl/filter/projectDensity.m
massScale         <repo>/analysis/OlhoffCurrent/+impl/fem/massScale.m
mmasub            <repo>/analysis/OlhoffCurrent/+impl/mma_published/mmasub.m
                    ALSO: <repo>/tools/Matlab/mmasub.m
subsolv           <repo>/analysis/OlhoffCurrent/+impl/mma_published/subsolv.m
                    ALSO: <repo>/tools/Matlab/subsolv.m
useMMA            <repo>/analysis/OlhoffCurrent/+impl/algo/useMMA.m
```

Two warnings, both **declared** in `olhoffcurrent_known_collisions.m` and both
losing the resolution to the production copy: `tools/Matlab/mmasub.m` (which is
byte-identical to the `asfound` variant, not `published` — it must never win,
and does not) and `tools/Matlab/subsolv.m`. The `@cvdata/applyFilter.p` entry is
a MATLAB class method and cannot take part in bare-name resolution.

---

## 16. Mismatches

**No scientific configuration mismatch was found.** The effective production
configuration is internally consistent and matches the frozen conference
realization field for field.

**But one operational blocker was discovered, and it currently stops production
from starting.** It is unrelated to the configuration and is reported here
rather than repaired, per §10.

### BLOCKER — the integrity gate is tripped by macOS `.DS_Store` files

`olhoffcurrent_source_manifest()` walks **every file** under `+impl/`. macOS
created two `.DS_Store` files there after the promotion commit:

```
./+impl/.DS_Store                 2026-09-07 10:31:04
./+impl/architecture/.DS_Store    2026-09-07 10:30:58
promotion commit cf1b71d          2026-09-07 10:11:07
```

They are `.gitignore`d (`.gitignore:69`), so **`git status` shows nothing** —
the working tree looks clean while the integrity manifest does not.

The promoted source itself is **provably untouched**:

```
mismatches (recorded file whose CONTENT changed): 0
missing    (recorded file now absent)           : 0
extra      (on disk, not recorded)              : 2   <- both .DS_Store
```

All 74 recorded source files are bit-identical. Nothing scientific changed.

**The consequence is nevertheless a hard stop.** Demonstrated, not predicted —
`confbench_preflight` run at 160×20 in the current state:

```
PREFLIGHT pass = 0  (27 checks)
  [FAIL] promoted Olhoff source hashes to SOURCE_MANIFEST.json
         .DS_Store; architecture/.DS_Store
  [FAIL] promoted Olhoff source is not modified in place
         currentness state = LOCAL_MODIFIED
```

`olhoffcurrent_currentness()` reports **`LOCAL_MODIFIED`** — its most serious
state, meaning "production source has been edited in place" — for a cause that
is neither an edit nor source. A Finder window or a Spotlight pass over the
directory is enough to block a production benchmark run.

**This is a defect in the gate introduced by the promotion, not in the
science.** The manifest should exclude filesystem artifacts (`.DS_Store`,
`Thumbs.db`, `*.asv`, `*.m~`) from the walk, or distinguish "recorded file
changed" (a real blocker) from "unrecorded artifact appeared" (at most a
warning). The current design conflates them.

**Not repaired here.** This audit is verification only; the fix is a code change
to `olhoffcurrent_source_manifest.m` and possibly to the preflight's
classification of `extra` files, and it is the user's call. Deleting the two
files would clear the symptom, mask the defect, and leave the next Finder visit
to reproduce it — so they were deliberately left in place.

### Two further non-scientific observations, neither acted upon

1. **`stop.toleranceRule` representational difference** — investigated in §8,
   proved bitwise-equivalent at five meshes, never read by the solver.
2. **Console summary uses historical shorthand** — `printMethodSettings` prints
   "M4 multiplicity" / "S2 ladder" / "signal beta (legacy)" for the Olhoff row,
   and does not print `filter.type` at all. Accurate but legacy-labelled; a
   candidate for a future terminology pass.

---

## 17. Did this audit change any file?

**No.** No file under `analysis/OlhoffCurrent`, `examples/Performance` or any
Olhoff tree was created, modified or deleted by this audit. The only file it
adds to the repository is this report.

The single-iteration execution-path probe of §4 ran from a scratch script
outside the repository, wrote nothing, and its result was discarded.

**Correction to an earlier draft of this section.** It initially stated that
`+impl/` still hashes to
`c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` and that
`olhoffcurrent_currentness()` still reports `CURRENT`. That is **no longer
true**, and the cause is the `.DS_Store` finding above — macOS artifacts created
at 10:30, twenty minutes after the promotion commit and independently of
anything this audit ran. The live tree hash over all files under `+impl/` is now
`8c2bde072e8bcc0fe76c49008539c04d7e0d7fba93e3f23a1c8483c8c2d5923e` (76 files)
and the state is `LOCAL_MODIFIED`. The **74 recorded source files remain
bit-identical**, so the promoted implementation is unchanged; it is the manifest's
scope that is too wide.

---

# OLHOFF_CURRENT_CONFIG_VERIFIED

The verdict is about the **configuration**, and the configuration is verified:
`filter.type = 'sensitivity'`, projection off, eq4b, p = 3, q = 1, and every
scientific field matching the frozen conference realization.

It is **not** a statement that a production run would start today. It would not:
the integrity gate is currently failing on two macOS `.DS_Store` files (§16).
That is an operational defect in the gate, deliberately left unrepaired.
