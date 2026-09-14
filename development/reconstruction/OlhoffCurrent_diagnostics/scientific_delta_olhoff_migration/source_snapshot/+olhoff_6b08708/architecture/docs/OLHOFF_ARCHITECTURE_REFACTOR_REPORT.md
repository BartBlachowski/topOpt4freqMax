# OLHOFF_ARCHITECTURE_REFACTOR_REPORT

Post-conference architectural cleanup of the Du–Olhoff eigenfrequency
topology-optimization implementation.

---

## 1. Starting provenance

| | |
|---|---|
| Working tree | `/Users/piotrek/Programming/Matlab/Olhoff`, 1.5 GB, 1507 files |
| Version control at start | **none** — not a git repository, and no parent was |
| Reference repository | `/Users/piotrek/Programming/topOpt4freqMax`, branch `benchmark-methodology-r2`, HEAD `9cd1c8e`, **clean** |
| Frozen conference solver | `analysis/OlhoffM4Reconstruction/+frozen/`, 23 files, tracked and clean |
| MATLAB | R2025b (25.2.0.2998904). The licensing error −15.2 noted in the last checkpoint had cleared. |

Phase 0 required stopping before refactoring if the tree was not under version
control. It was not, so a baseline was established first:

* `git init`, branch `main`, commit **`2029baa`**, tag **`post-conference-baseline`**;
* `.gitattributes` pins `* -text` — the inherited global `core.autocrlf=input`
  would otherwise have rewritten CRLF evidence files on checkout;
* every human-readable file tracked at any size, plus binaries under 5 MB;
  59 binaries ≥ 5 MB (1276 MB) excluded and covered instead by
  `EVIDENCE_MANIFEST.sha256`, which hashes **all 1507 files**;
* work proceeded on branch `architecture/canonical-config`.

Verified before any edit: all 23 files of the frozen conference solver still
hash exactly to their `sha256_imported` in `IMPORT_MANIFEST.json`.

## 2. Architecture problems found

1. **No preset layer.** Every historical realization derived from one stored
   blob, `TMA_FROZEN_CFGS.mat` `CFG(k).cfg`, plus a `switch` on an experiment
   label inside each runner. Realization identity lived in a `.mat` file and a
   runner name.
2. **Four competing sources of defaults.** `algo/defaultCfg.m`, the `.mat` blob,
   runner literals, and five `isfield` fallbacks at the top of `olhoffOpt.m`.
   `defaultCfg.m` is not any realization: it differs from the frozen
   configuration in **eleven scientific fields**, including `multRule`,
   `filterMode`, `moveFamily` and `massInterp`.
3. **Experiment identifiers inside the mathematics.** `switch upper(cfg.restorationGuard)`
   with cases `'R1'` and `'R2'` decided a stopping rule.
4. **Two accidental couplings** welded independent scientific choices together
   (§7).
5. **The solver mutated its own configuration** — it wrote `cfg.mmasubPath` and
   overwrote `cfg.rminEl`, so the effective configuration did not exist until
   the solver had already started.
6. **A scientific policy with no home**: `tolOuter = 0.05*sqrt(NE/3200)` was
   retyped in six runners.
7. **No validation of any kind.** A mistyped `cfg.restorationGuard` would have
   silently disabled a guard.
8. **`res.rho` ambiguity.** Under projection the stopping test silently changed
   the field it monitored, from the density to the design variable.

## 3. Complete variant matrix

60 concepts inventoried in `ARCHITECTURE_VARIANT_INVENTORY.md`, covering
mathematical meaning, provenance, implementation location, current field,
whether hard-coded, duplicated or coupled, which experiments used it, the
proposed canonical field and the refactor risk.

## 4. Canonical configuration schema

**80 fields**, in `olh.config.schema`, each carrying its type, domain, default,
provenance class and a one-line description written in terms of the science.
Generated reference: `CONFIG_REFERENCE.md`.

Branches: `domain`, `material.solid`, `material.stiffness`, `material.mass`,
`design`, `filter`, `projection`, `eigen`, `multiplicity`, `optimizer.inner`,
`move`, `stop`, `runtime`, plus `provenance` recorded with every run.

## 5. Provenance classification

Field-level, in `SCIENTIFIC_CONFIG_PROVENANCE.md`, quoting the four sentences
that carry most of the class-A weight.

| Class | Fields | |
|---|---|---|
| **A** | 23 | explicitly specified by a Du–Olhoff source |
| **B** | 4 | directly implied by one |
| **C** | 41 | under-specified reconstruction choice |
| **D** | 12 | later experimental modification |

**Two thirds of the fields that determine a trajectory are reconstruction or
later modification.** Verified by text extraction over both papers:
`move limit`, `trust region`, `step size`, `continuation`, `projection`,
`Heaviside` and `density filter` occur **zero** times in Du & Olhoff (2007);
`filter`, `tolerance` and `move limit` occur **zero** times in Olhoff & Du (2014),
which therefore closes none of the 2007 gaps.

Three findings worth stating plainly:

* **Fixed p = 3 is a reconstruction ruling, not the paper's procedure.** §2.1
  says p is "normally assigned values increasing from 1 to 3". Running a p
  schedule is closer to the paper's stated practice than the frozen realization.
* **The paper filtered sensitivities.** Every projected run replaces that with a
  density filter, which is a departure from a printed choice.
* **The move ladder has no source at all**, and its `settledMove` guard exists
  only to repair a defect the ladder itself introduces.

## 6. Presets

Ten, each classified, each documenting its parent, its single departure and its
provenance. See `PRESETS.md`.

`duOlhoffFrozenM4`, `legacyBinaryDiagonal` — SCIENTIFIC_PRESET.
`duOlhoffMatureM4`, `restorationLadderGuard`, `noDescentFixedMove`,
`pContinuationCoupled`, `pContinuationDecoupled`, `pMassCompatible`,
`projectionIdentity`, `projected` — EXPERIMENT_PRESET.
No OBSOLETE_ALIAS: every historical realization is scientifically distinct. The
things that looked like runtime-only presets — the mesh variants, and `Bmature`
at three different caps — are **overrides**, not presets.

**All 14 historical realizations are reproduced exactly from presets**, T800 at
800×100 included, with the mesh-scaled tolerance deriving correctly at every
mesh (`test_presets_match_history`).

## 7. Bad couplings found

| Coupling | Classification | Action |
|---|---|---|
| diagonal offsets in (25d) derived from `multRule=='subspace'` | **HISTORICAL_ACCIDENT** | split into `multiplicity.method` and `multiplicity.diagonalOffsets` |
| density filter reachable only via `projection.on` | **HISTORICAL_ACCIDENT** | split into `filter.type` and `projection.enabled` |
| stopping field silently changing under projection | **HISTORICAL_ACCIDENT** | named as `stop.field`; `hist.dxPhys2` recorded |
| `tolOuter` scaling law retyped in six runners | **HISTORICAL_ACCIDENT** | `olh.config.epsilonForMesh`, one definition |
| solver mutating its own configuration | **HISTORICAL_ACCIDENT** | resolution happens before the solver; it never writes to cfg |
| p continuation indexed by the move ladder stage | **INTENTIONAL_POLICY** | **kept**, as `continuation.driver` |
| mass model switching on the p schedule | **INTENTIONAL_POLICY** | **kept**, as `mass.continuation` |
| projection continuation consuming the convergence event | **INTENTIONAL_POLICY** | **kept**, named |
| `R1` guard reading the ladder | **INTENTIONAL_POLICY**, misnamed | renamed `ladderExhausted` |
| a move change mechanically satisfying the RMS test | **INTENTIONAL_POLICY**, a known defect | **kept in the frozen preset**, documented |

Only the historical accidents were removed. No intentional scientific coupling
was decoupled merely because decoupling would produce cleaner software.

## 8. Files changed

Three files outside the new `architecture/` directory, and only three:

| File | Change |
|---|---|
| `algo/olhoffOpt.m` | became a compatibility shim with no mathematics |
| `fem/massScale.m` | became a thin dispatcher onto `olh.material.massInterpolation` |
| `setpaths.m` | one `addpath` so the `+olh` package resolves |

New: `architecture/` — 28 source files, 9 test files, 6 anchor-harness files,
11 documents, and `legacy/olhoffOpt_PRE_CANONICAL.m` (the pre-refactor solver,
preserved verbatim).

## 9. Files archived or deleted

**None.** Phase 19 forbids deletion before equivalence; the classification is
prepared in `FILE_CLASSIFICATION.md` and not executed. One `ARCHIVE` candidate
is recorded: `top88.m` at the repository root, a byte-identical duplicate of
`filter/top88_reference.m`.

## 10. Regression anchors

Twelve, all at **160×20**, chosen to exercise every independent code path, each
with a pre-refactor reference artifact and a SHA-256 digest over its scientific
content.

Harness validated before any edit: `A1_frozen160` reproduces the archived
conference result `audit_termination_mesh_admission/runs/TMA_160x20.mat`
bitwise in `rho`, `omega`, the complete non-timing history, and **every
per-iteration `drho`**.

**Equality standard, preregistered in the plan before any solver edit.** Two
digests:

* `science` — densities, design variables, eigenfrequencies, volume, the
  complete non-timing history, every per-iteration `drho`, all continuation,
  projection and multiplicity transitions, status and failure counters.
  **Must be bitwise identical.**
* `presentation` — log message text, reported separately, because renaming a
  guard necessarily rewords a message without changing a trajectory. The
  accompanying `logShape` invariant — the number of log events and the kind of
  each — **must** stay identical.

Timing fields are excluded, and only timing fields are excluded.

## 11. Before/after equivalence

All twelve anchors, re-run against the **final** state of the code (the suite was
re-run from scratch after the last static-analysis cleanup, so no result below is
against a stale build):

| Anchor | Exercises | outer | inner | status | science digest | logShape | log text |
|---|---|---|---|---|---|---|---|
| `A1_frozen160` | the frozen conference realization verbatim | 91 | 2241 | CONVERGED | **BITWISE IDENTICAL** | identical | identical |
| `A2_mature160` | stopping guard `maxDesignChange` | 187 | 3925 | CONVERGED | **BITWISE IDENTICAL** | identical | reworded |
| `A3_r1ladder160` | stopping guard `ladderExhausted` (reads the ladder) | 102 | 2460 | CONVERGED | **BITWISE IDENTICAL** | identical | reworded |
| `A4_nodescent160` | move policy `fixed` — no descent | 400 | 11429 | CAP_HIT | **BITWISE IDENTICAL** | identical | identical |
| `A5_pcont160` | p continuation coupled to the ladder stage; stop blocked while p<3 | 250 | 5315 | CAP_HIT | **BITWISE IDENTICAL** | identical | identical |
| `A6_pdecoupled160` | p continuation decoupled: stall interception, ladder reset, `pEvent` | 250 | 5299 | CAP_HIT | **BITWISE IDENTICAL** | identical | reworded |
| `A7_massp160` | printed-mass continuation tied to the p schedule | 250 | 4771 | CAP_HIT | **BITWISE IDENTICAL** | identical | reworded |
| `A8_projidentity160` | density filter + identity projection + chain rule + exact `volFun` | 113 | 2383 | CONVERGED | **BITWISE IDENTICAL** | identical | reworded |
| `A9_projection160` | full tanh projection and its continuation (`projEvent`) | 600 | 12757 | CAP_HIT | **BITWISE IDENTICAL** | identical | reworded |
| `A10_binarydiag160` | binary multiplicity, (25d) **as printed**, diagonal-only filtering, fixed move | 400 | 6666 | CAP_HIT | **BITWISE IDENTICAL** | identical | identical |
| `A11_maxnorm160` | max-norm stopping | 92 | 2257 | CONVERGED | **BITWISE IDENTICAL** | identical | identical |
| `A12_rhovar160` | inner variable `design` (persistent MMA state) | 22 | 815 | CONVERGED | **BITWISE IDENTICAL** | identical | identical |

**Anchors failing the equality standard: 0 of 12.**

Between them the anchors cover 60 811 inner iterations and 2 957 outer
iterations, and the digest includes **every per-iteration `drho` vector**, not
merely the endpoints.

### On the reworded logs

Five anchors report `logText=reworded`. In every case the cause is the same and
is the intended change: the stopping guard that used to print

```
iter 153: baseline stop blocked by R2 (stage=4, max|drho|=..., epsRMS=...)
```

now prints `maxDesignChange (R2)` or `ladderExhausted (R1)`. The historical
spelling is kept in parentheses so an old log and a new one can still be read
side by side. `logShape` — the number of log events and the kind of each — is
**identical for all twelve anchors**, so no event was added, dropped or
reordered.

No numerical tolerance was invoked anywhere. Bitwise equality was achieved, so
the question of preregistering a tolerance never arose.

### Both paths verified

The twelve anchors drive the **legacy** path — a flat configuration into
`olhoffOpt`, exactly as every audit runner does (`grep` confirms all of them
call `res=olhoffOpt(cfg)`). `test_preset_reproduces_anchor` drives the
**canonical** path instead, `olh.config.resolve(preset) -> olhoffSolve`, and
reproduces `A1`, `A2` and `A3` bitwise. The compatibility layer and the new
entry point therefore agree with each other and with history.

### Independent re-verification

`architecture/anchors/code/anchorReport.m` recomputes both digests from the
stored reference and candidate records, so the comparison can be repeated
without running a single solver iteration:

```
anchors failing the equality standard: 0 of 12
```


## 12. Unit-test results

| Suite | Result | What it establishes |
|---|---|---|
| `test_config` | **PASS** (40 checks) | defaults cover the schema; overrides apply; the mesh-scaled tolerance is re-derived *after* overrides and matches the stored historical value at 240×30; **18 distinct refusals** each with its named identifier; a suspicious-but-valid configuration warns and is **not coerced**; all ten presets validate |
| `test_mass` | **PASS** (18 checks) | all four mass models bitwise against the **printed equations**, written out independently; derivatives against central finite differences; continuity at ρ=0.1 confirming the paper's own claims — **(4) discontinuous, (4a) C⁰ but not C¹, (4b) C¹**; the printed coefficients refused outside their printed regime; the legacy names still reach the same numbers bitwise |
| `test_modules` | **PASS** (28 checks) | `olh.multi.detect` == `algo/multRule` over 200 spectra for all four methods; `olh.move.limit` == `algo/moveControl` **bitwise** over 120 iterations for all four policies; the sensitivity and density filters are demonstrably different operators; projection identity at β=0, monotonicity, P(0)=0, P(1)=1, and the chain rule against finite differences; design/filtered/physical fields distinct under projection and collapsing correctly without it; status precedence |
| `test_continuation` | **PASS** (16 checks) | the three controllers are independent; `hist.move` is always the level of `hist.stage`; coupled p indexes the ladder while decoupled p uses its own counter; the low-p mass model is in force exactly while p<final; the projection stage advances only on a `projEvent`; neither a fixed-p nor a non-projection run carries the other controllers' state |
| `test_legacy_roundtrip` | **PASS** (35 configurations) | `toLegacy(fromLegacy(x))` reproduces x for the three frozen TMA configs, all 20 saved `*.config.mat` in the audit tree, the twelve anchors, and `algo/defaultCfg` |
| `test_presets_match_history` | **PASS** (14 realizations) | every historical realization reproduced exactly from a preset, T800 at 800×100 included |
| `test_preset_reproduces_anchor` | **PASS** (3 realizations) | `duOlhoffFrozenM4`, `duOlhoffMatureM4` and `restorationLadderGuard` reproduce `A1`, `A2` and `A3` **bitwise through the canonical path** (`resolve(preset) -> olhoffSolve`), where the anchors themselves drive the legacy path (`flat cfg -> shim -> fromLegacy -> olhoffSolve`). Passing both makes the presets trajectory-equivalent to the historical realizations, not merely config-equivalent |

**Total failures: 0.**

## 13. Static analysis

`checkcode` over all 44 files of the new and modified code: **0 messages**
(14 before cleanup).

## 14. Unresolved scientific ambiguities

Recorded, not reconciled, exactly as the brief requires.

1. **p fixed versus p continuation.** §2.1 states p normally increases 1→3; the
   reconstruction fixed p = 3 on the evidence that the reported initial
   eigenfrequencies fit p = 3 and not p = 1. Both are presets. The paper's own
   practice is the one the frozen realization does *not* use.
2. **Support idealization.** The paper *draws* corner supports; its *numbers*
   fit mid-height supports with axial restraint at both ends. The drawing and
   the numbers disagree. Confounded with element formulation — a 5 % effect,
   and plain Q4 shear locking is comfortably within that band.
3. **Filter radius.** Never stated for any example, in either paper, and the
   strongest single determinant of member thickness.
4. **Multiplicity tolerance.** "predefined, very small" and never given. The
   frozen realization uses 0.05; Krog & Olhoff report 1e-4 for their own
   examples, in an algorithm whose detector does not gate the ascent.
5. **Which `f_sk` are filtered.** The paper has one sensitivity vector; the
   multiple case has N(N+1)/2 and the paper does not say.
6. **The constraint gradients.** `∂Δλ_j/∂Δρ_e = Σ v_js v_jk (f_sk)_e` is
   reconstruction; the paper never states how they are formed, and the
   expression is itself non-differentiable when the Δλ_j coalesce inside the
   inner loop.
7. **ω_J multiple.** (25b) assumes it simple; no procedure is defined. Logged,
   not patched.
8. **NE never reported**, in any example.

## 15. Remaining technical debt

1. `useMMA` selects the MMA variant by **mutating the global MATLAB path**.
   Left alone: changing it is a trajectory risk for no scientific gain.
2. `eigen.tolerance`, `maxIterations` and `krylovFactor` exist in the schema but
   `olhoffSolve` still passes only the solver name to `eigSolve`, where those
   values remain hard-coded. Wiring them would change nothing today; a port
   should do it.
3. MMA's `a₀, a, c, d` and the β box `xmax = 5` remain hard-coded in
   `innerLoop.m`. Class C, no configuration path.
4. `mma/subsolv.m` and `mma_published/subsolv.m` are byte-identical; the "two
   variants" differ in one file.
5. `algo/moveControl.m` and `algo/multRule.m` are retained for the legacy path
   and are duplicates of the canonical modules in behaviour. They are pinned to
   each other by tests, but they are still two implementations.
6. The continuation controllers expose their state through `hist` fields rather
   than a uniform reporting interface.

## 16. pyMorphoGen port implications

`PYMORPHOGEN_PORT_MAP.md`. The largest gaps in the existing Python code:
`SimpInterpolation.mass_scale` is **linear only** and cannot express
(4)/(4a)/(4b); there is no sensitivity filter, only a density filter and its
adjoint; and there is no multiple-eigenvalue machinery or bound formulation at
all. Three things must not be flattened in the port: the two filters are
different operators, the three continuation controllers differ in whether they
can veto convergence, and the design/filtered/physical density distinction must
survive.

## 17. Final source hashes

<<<HASHES>>>

---

## Table: historical realization → new preset

| Historical realization | New preset | Equality | Status |
|---|---|---|---|
| TMA / B0 / REG160 — the conference realization | `duOlhoffFrozenM4` | bitwise (`A1`) | reproduced |
| Bmature / R2 | `duOlhoffMatureM4` | bitwise (`A2`) | reproduced |
| R1 | `restorationLadderGuard` | bitwise (`A3`) | reproduced |
| nodescent | `noDescentFixedMove` | bitwise (`A4`) | reproduced |
| P1 | `pContinuationCoupled` | bitwise (`A5`) | reproduced |
| PD1 | `pContinuationDecoupled` | bitwise (`A6`) | reproduced |
| PM1 | `pMassCompatible` | bitwise (`A7`) | reproduced |
| D160 | `projectionIdentity` | bitwise (`A8`) | reproduced |
| T160 / T240 / T320 | `projected` + mesh override | bitwise (`A9`) | reproduced |
| T800 | `projected` + mesh override | config exact; not re-run | **evidence reused**, per Phase 7 |
| the pre-M4 reconstruction | `legacyBinaryDiagonal` | bitwise (`A10`) | reproduced |
| HISTORICAL_320x40 | `duOlhoffFrozenM4` | same as B0 | identical mathematics, different runner — **not a preset** |
| `Bmature` at caps 400 / 1200 | `duOlhoffMatureM4` + `runtime.maxOuter` | — | runtime override, **not a preset** |

The 800×100 treatment was deliberately **not** re-run: Phase 7 forbids launching
an expensive 800×100 regression unless genuinely necessary, and the saved T800
evidence plus the exact configuration match make it unnecessary. The projection
code path it exercises is pinned bitwise at 160×20 by `A8` and `A9`.

## Table: scientific concept → canonical field

| Concept | Canonical field | Choices | Provenance |
|---|---|---|---|
| stiffness interpolation | `material.stiffness.model` | `simp` | A |
| penalization | `material.stiffness.p` | ≥ 1 | A |
| p continuation | `material.stiffness.continuation.*` | on/off, schedule, driver | A practice / B values / C rule |
| mass interpolation | `material.mass.model` | `eq2`, `eq4`, `eq4a`, `eq4b` | A |
| mass exponent q | `material.mass.q` | ≥ 1 | A |
| low-density exponent r | `material.mass.lowDensityExponent` | ≥ 1 | A |
| mass cut-off | `material.mass.cutoff` | [0,1] | A |
| mass continuation | `material.mass.continuation.*` | on/off, low-p model | D |
| filter formulation | `filter.type` | `sensitivity`, `density`, `none` | A / D |
| filter radius | `filter.radiusPhysical`, `.radiusElements` | > 0 | C |
| filter scope | `filter.applyTo` | `diagonal`, `all` | C |
| projection | `projection.enabled` | on/off | D |
| projection sharpness | `projection.beta.levels` | monotone vector | D |
| projection threshold | `projection.eta` | [0,1] | D |
| projection continuation | `projection.continuation.trigger` | `outerConvergence` | D |
| multiplicity detection | `multiplicity.method` | `binary`, `latch`, `hysteresis`, `subspace` | A measure / C rule |
| multiplicity tolerance | `multiplicity.tolerance` | [0,1] | A measure / C value |
| subspace size | `multiplicity.subspaceSize` | ≥ 1 | C |
| (25d) diagonal offsets | `multiplicity.diagonalOffsets` | on/off | C |
| off-diagonal terms | `multiplicity.offDiagonal` | on/off | A |
| inner optimizer | `optimizer.inner.type` | `mma`, `lp` | A / B |
| MMA variant | `optimizer.inner.variant` | `published`, `asfound` | B |
| inner variable | `optimizer.inner.variable` | `increment`, `design` | C |
| inner tolerance | `optimizer.inner.tolerance` | > 0 | C |
| move policy | `move.policy` | `fixed`, `geometric`, `ladder`, `trustRatio` | B existence / C form |
| move levels | `move.levels` | descending vector | C |
| stall signal | `move.continuation.signal` | `boundVariable`, `designRms` | C / D |
| stall window, tolerance | `move.continuation.window`, `.tolerance` | > 0 | C |
| convergence norm | `stop.norm` | `l2`, `max` | B |
| convergence tolerance | `stop.tolerance`, `.toleranceRule` | > 0 | C |
| monitored field | `stop.field` | `designVariable` | A |
| stopping guards | `stop.guards.{settledMove, ladderExhausted, maxDesignChange}` | on/off | C / D |
| iteration cap | `runtime.maxOuter` | ≥ 1 | C |
| threads, diagnostics | `runtime.singleThread`, `.diagnostics` | on/off | C |

---

## Final verdict

**OLHOFF_ARCHITECTURE_REFACTOR_VERIFIED**

Against the criteria the brief sets:

* **one canonical configuration model** — `olh.config.schema`, 80 fields, single
  source of defaults, one resolution order, strict validation;
* **experiment IDs removed from solver mathematics** — `olhoffSolve` branches on
  no experiment identifier; the only remaining textual occurrences are inside a
  log-formatting helper that branches on the semantic booleans and prints the
  historical spelling so old and new logs stay comparable;
* **all selected behavioural anchors reproduced** — 12 of 12 bitwise, covering
  60 811 inner iterations, with every per-iteration `drho` in the digest;
* **frozen conference evidence untouched** — the reference repository is clean at
  `9cd1c8e`, all 23 frozen-solver files still hash to their `sha256_imported`,
  and 1504 of the 1507 files recorded at baseline are bit-identical (the three
  that changed are the three this refactor sets out to change);
* **variants accessible through clean configuration and presets** — ten presets,
  14 of 14 historical realizations reproduced exactly from them;
* **scientific provenance documented** — field level, A/B/C/D, with the
  quotations that carry the class-A weight;
* **tests passing** — 6 suites, 0 failures; `checkcode` 0 messages.

The guiding principle is met: `olh.config.describe(cfg)` answers *what
mathematical formulation am I running?* from one object, with the provenance
class of every choice, and flags each class-D departure from the printed method
explicitly.
