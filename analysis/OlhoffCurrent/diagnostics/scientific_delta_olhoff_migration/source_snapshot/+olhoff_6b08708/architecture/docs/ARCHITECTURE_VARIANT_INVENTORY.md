# ARCHITECTURE_VARIANT_INVENTORY

Read-only audit of the active Du–Olhoff solver at
`/Users/piotrek/Programming/Matlab/Olhoff`, taken at the post-conference
baseline commit `2029baa` (tag `post-conference-baseline`).

Scope: `algo/`, `fem/`, `filter/`, `mma/`, `mma_published/`, `setpaths.m`, plus
every runner under `runs/` and `audit_*/code/` that constructs a configuration.

Nothing in this document changes behaviour. It records what exists.

---

## 0. The one fact that shapes everything below

Every historical realization in this tree — the frozen conference run, the
restoration guards, both p-continuation variants, the mass continuation and both
projection campaigns — is built from **one** stored configuration:

```
audit_m4_topology_restoration/baseline/tree/
    audit_termination_mesh_admission/runs/TMA_FROZEN_CFGS.mat   →  CFG(k).cfg
```

`CFG(1)`, `CFG(2)`, `CFG(3)` are 160×20, 240×30 and 320×40. They differ in
**exactly four fields**: `nelx`, `nely`, `tolOuter`, `name`. All 47 scientific
fields are identical across the three.

Each audit then applies between one and four overrides on top. There is no
preset layer, no schema and no validation: the realization identity lives in a
`.mat` blob plus a `switch` on an experiment label inside each runner.

`algo/defaultCfg.m` is **not** the frozen realization and never was. It differs
from `CFG(k)` in eleven scientific fields (§3). Anything built from
`defaultCfg` is a different algorithm from anything built from `CFG(k)`.

---

## 1. Variant inventory

Legend for *Hard-coded?* — **yes** = the value or rule is a literal in solver
code with no configuration path. *Refactor risk* is the risk that touching the
item changes a trajectory.

| # | Concept | Published choices (Du–Olhoff) | Reconstruction choices | Current implementation | Current config field | Hard-coded? | Duplicated? | Coupled to? | Historical users | Proposed canonical field | Refactor risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Stiffness interpolation | Eq. (1) `ρ^p Ee` | none — used verbatim | `assemble2D.m:18` `mdl.K0(:)*(rho.^p)'` | — | law is hard-coded | no | p | all | `cfg.material.stiffness.model='simp'` | low |
| 2 | Penalization p | Eq. (1); "normally assigned values increasing from 1 to 3" | fixed p=3 chosen because reported ω⁰ only fit p=3 | `olhoffOpt.m:118` `pNow=cfg.p` | `cfg.p` | no | no | pSchedule | all | `cfg.material.stiffness.p` | low |
| 3 | p continuation | §2.1 states increasing 1→3 as normal practice; **no schedule, no transition rule** | schedule `[1 2 3]`; transition = S2 stall event | `olhoffOpt.m:118-132` | `cfg.pSchedule` | transition rule is hard-coded | no | **move ladder stage** (§2.1) | P1, PD1, PM1 | `cfg.material.stiffness.continuation.*` | **high** |
| 4 | p continuation coupling mode | none | coupled to ladder index (P1) vs own counter (PD1) | `olhoffOpt.m:120-131`, `:270-285` | `cfg.pDecouple` | asserts `moveFamily=='S2'` | no | move ladder | PD1, PM1 | `cfg.material.stiffness.continuation.driver` | **high** |
| 5 | Mass interpolation model | Eqs. (2), (4), (4a), (4b) — all four printed and all four reported as used | none — coded verbatim | `fem/massScale.m` | `cfg.massInterp` `'lin'\|'4'\|'4a'\|'4b'` | no | no | — | all (`4b`); `lin` in PM1 | `cfg.material.mass.model='eq2'\|'eq4'\|'eq4a'\|'eq4b'` | low |
| 6 | Mass exponent q | Eq. (2) `q≥1`, "normally q=1" | q=1 only | `massScale.m:13` `g=rho` | — | **yes** | no | — | all | `cfg.material.mass.q` | low |
| 7 | Low-density mass exponent r | Eq. (4) "r chosen to be about r=6" | r=6 | `massScale.m:15` `rho.^6` | — | **yes** | no | — | all | `cfg.material.mass.r` | low |
| 8 | Mass cutoff density | Eqs. (4),(4a),(4b) all use ρ≤0.1 | 0.1 | `massScale.m:6` `lo = rho<=0.1` | — | **yes** | no | — | all | `cfg.material.mass.cutoff` | low |
| 9 | Mass continuity coefficients | c0=1e5; c1=6e5, c2=−5e6 | none | `massScale.m:17,20` | — | **yes** | no | model | all | derived from cutoff+r, or explicit | low |
| 10 | Mass model continuation | **none** | printed low-p model while p<pEnd | `olhoffOpt.m:145-151` | `cfg.massLowP` | switch rule hard-coded | no | **p schedule** (§2.2) | PM1 | `cfg.material.mass.continuation.*` | **high** |
| 11 | Sensitivity filter | Sigmund (1997) "applied to **the sensitivities** of the objective functions" | top88 `ft=1` form | `filter/applyFilter.m` | `cfg.filterMode` | `max(1e-3,ρ)` guard | no | projection (§2.5) | all non-projection | `cfg.filter.type='sensitivity'` | low |
| 12 | Filter application scope | not stated (paper has one sensitivity vector; multiple case has N(N+1)/2) | `diag` \| `all` \| `none` | `olhoffOpt.m:234-250` | `cfg.filterMode` | no | no | projection | `all` (frozen), `diag` (early) | `cfg.filter.applyTo` | medium |
| 13 | Filter radius, element units | **never stated for any example** | swept | `prepFilter.m` | `cfg.rminEl` | no | no | rminPhys | early runs | `cfg.filter.radiusElements` | low |
| 14 | Filter radius, physical | never stated | R=0.06 frozen; `rminEl=R/(b/nely)` | `olhoffOpt.m:25-28` **mutates cfg** | `cfg.rminPhys` | conversion hard-coded | **yes** — also in `run_800.m:14` | mesh | all frozen | `cfg.filter.radiusPhysical` | low |
| 15 | Density filter | **not published** | reachable **only** via projection | `projDensityField.m:19` `(H*z)./Hs` | *none* | **yes** | no | **projection** (§2.5) | D160, T160/240/320, T800 | `cfg.filter.type='density'` | **high** |
| 16 | Projection enable | not published (Class D) | tanh Heaviside | `olhoffOpt.m:88-102` | `cfg.projection.on` | no | no | filter, stopping | D160, T*, T800 | `cfg.projection.enabled` | medium |
| 17 | Projection β schedule | not published | `[1 2 4 8]`; `0` = identity control | `olhoffOpt.m:156-159` | `cfg.projection.betaSchedule` | no | no | stopping (§2.3) | T160/240/320, T800 | `cfg.projection.beta.levels` | medium |
| 18 | Projection η | not published | 0.5 | `projectDensity.m` | `cfg.projection.eta` | no | no | — | T*, D160 | `cfg.projection.eta` | low |
| 19 | Projection continuation trigger | not published | the outer convergence event itself | `olhoffOpt.m:446-453` | *none* | **yes** | no | **stopping** (§2.3) | T*, T800 | `cfg.projection.continuation.trigger` | **high** |
| 20 | Multiplicity detection rule | §3.5.1 measure only: relative frequency difference within "a predefined, very small tolerance" | `binary`\|`latch`\|`hyst`\|`subspace` | `algo/multRule.m` | `cfg.multRule` | no | no | **off-diagonal offsets** (§2.4) | `subspace` (frozen), `binary` (early) | `cfg.multiplicity.method` | medium |
| 21 | Multiplicity tolerance | value never given, 2007 or 2014 | 0.05 frozen; 0.02 default; 1e-4 tested (Krog & Olhoff) | `multRule.m:80` | `cfg.tolMult` | no | no | — | all | `cfg.multiplicity.tolerance` | low |
| 22 | Hysteresis thresholds | none | `tolEnter`/`tolExit` | `multRule.m:57-64` | `cfg.tolEnter/tolExit` | no | no | method | none run | `cfg.multiplicity.hysteresis.*` | low |
| 23 | Subspace size | none | `subN=2` | `multRule.m:66-68` | `cfg.subN` | no | no | method | frozen, all M4 | `cfg.multiplicity.subspaceSize` | low |
| 24 | Off-diagonal coupling in (25d) | Eq. (25d) full determinant (erratum form) | may be forced to vanish → Krog & Olhoff LP route | `innerLoop.m:77-86,122-132` | `cfg.offDiag` | no | no | innerSolver | all (`true`) | `cfg.multiplicity.offDiagonal` | low |
| 25 | Diagonal eigenvalue offsets `dOff` | **not in any source** — (25d) assumes exact degeneracy | retain actual separation on the diagonal | `olhoffOpt.m:196-205`, `deltaLambda.m:47-58` | *none* | **yes — derived from `multRule=='subspace'`** | no | **multiplicity method** (§2.4) | frozen, all M4 | `cfg.multiplicity.diagonalOffsets` | **high** |
| 26 | λ̃ in eq. (19) | §3.5.1 "we set λ̃ = ω_n²" | first eigenvalue of cluster, not the mean | `olhoffOpt.m:189` | *none* | **yes** | no | — | all | `cfg.multiplicity.lambdaTilde` | low |
| 27 | J = n + N, Nmax | Eq. (25b) `J=n+N` | `Jcalc = n + Nmax` modes computed | `olhoffOpt.m:31-33` | `cfg.Nmax` | no | no | — | all | `cfg.multiplicity.maxCluster` | low |
| 28 | Inner optimizer | §3.5.3 "the MMA method (Svanberg 1987) has been used" | `mma` \| `lp` (Krog & Olhoff route) | `olhoffOpt.m:306-315` | `cfg.innerSolver` | no | no | offDiag, projection | all (`mma`) | `cfg.optimizer.inner.type` | low |
| 29 | MMA source variant | Svanberg 1987 cited; constants not stated | `published` (Sept-2007) vs `asfound` | `algo/useMMA.m` — **mutates the MATLAB path** | `cfg.mmaVariant` | no | no | — | all (`published`) | `cfg.optimizer.inner.variant` | low |
| 30 | Inner design variable | §3.5.2 independent variables are β and Δρ_e | `drho` (frozen) \| `rho` (persistent MMA) | `innerLoop.m` / `innerLoopRho.m` | `cfg.innerVar` | no | no | projection (asserts `drho`) | all (`drho`) | `cfg.optimizer.inner.variable` | low |
| 31 | Inner convergence test | Fig. 1 "Increments Δρ_e converged?" — no criterion | `max|dx|/max|Δρ| < tolInner`, `minInner` floor | `innerLoop.m:145-149` | `cfg.tolInner`, `minInner` | form hard-coded | no | move limit | all | `cfg.optimizer.inner.tolerance/minIterations` | low |
| 32 | Inner iteration cap | none | 500 frozen / 300 default | `innerLoop.m:72` | `cfg.maxInner` | no | no | — | all | `cfg.optimizer.inner.maxIterations` | low |
| 33 | MMA hyper-constants | none | `a0=1, a=0, c=1000, d=0`; `xmax(β)=5` | `innerLoop.m:50,61` | *none* | **yes** | **yes** — also `innerLoopRho.m` | — | all | `cfg.optimizer.inner.mma.*` | medium |
| 34 | β scaling by `lamref` | none | divide constraints by λ_n | `innerLoop.m:43,95-101` | *none* | **yes** | **yes** | — | all | keep internal | low |
| 35 | Move-limit policy | **none** — only the box (25f) | `S0`\|`S1`\|`S2`\|`S3` | `algo/moveControl.m` | `cfg.moveFamily` | no | no | p continuation, R1 guard | `S2` (frozen), `S0` (no-descent) | `cfg.move.policy` | medium |
| 36 | Initial move | none | 0.04 frozen; 0.05 default | `moveControl.m:73` | `cfg.move` | no | no | s2Levels(1) | all | `cfg.move.initial` | low |
| 37 | Move floor | none | 0.002 | `moveControl.m:84,160` | `cfg.moveMin` | no | no | — | S1/S3 only | `cfg.move.minimum` | low |
| 38 | S2 ladder levels | none | `[0.04 0.02 0.01 0.005]` | `moveControl.m:143` | `cfg.s2Levels` | no | no | **R1 guard, pDecouple** | all frozen | `cfg.move.levels` | medium |
| 39 | S2 stall window / tol | none | `W=10`, `tol=5e-3` | `moveControl.m:130-141` | `cfg.s2Window`, `s2Tol` | mean-vs-mean form hard-coded | no | — | all frozen | `cfg.move.continuation.window/tolerance` | medium |
| 40 | S2 stall signal | none | `beta` (legacy) \| `drms` | `moveControl.m:115-129` | `cfg.s2Signal` | default `'beta'` inside the function | no | — | `beta` everywhere; `drms` tested, **not adopted** | `cfg.move.continuation.signal` | medium |
| 41 | S2 re-arm dwell | none | `(outer-lastStage) > W` | `moveControl.m:130` | *none* | **yes** | no | window | all frozen | `cfg.move.continuation.rearm` | medium |
| 42 | S3 gain band | none | `0.30/0.70/0.7/1.1` | `moveControl.m:145-165` | `cfg.s3*` | no | no | — | none run | `cfg.move.trust.*` | low |
| 43 | Outer convergence norm | §3.5.1 "the norm of the vector Δρ … less than a small, predefined value ε" — norm unqualified | `l2` (natural reading) \| `max` (legacy) | `olhoffOpt.m:391-395` | `cfg.outerNorm` | no | no | — | `l2` everywhere | `cfg.stop.norm` | low |
| 44 | Outer tolerance ε | value never given | `0.05·√(NE/3200)` mesh scaling | `olhoffOpt.m:392` reads `cfg.tolOuter`; **the scaling law lives in the runners** | `cfg.tolOuter` | no | **yes — 6 runners** | mesh | all | `cfg.stop.tolerance` (+ explicit scaling helper) | medium |
| 45 | Stop guard: settled move | none | assert only when `move(k)==move(k-1)` | `olhoffOpt.m:403-416` | `cfg.outerGuard` | no | no | **move ladder** (§2.8) | all frozen | `cfg.stop.guards.settledMove` | medium |
| 46 | Stop guard: R1 | none | no remaining ladder level exceeds ε_RMS | `olhoffOpt.m:423-426` | `cfg.restorationGuard='R1'` | **experiment ID in solver** | no | **s2Levels, hist.stage** | R1_320x40 | `cfg.stop.guards.ladderExhausted` | **high** |
| 47 | Stop guard: R2 | none | `max\|Δρ\| < ε_RMS` | `olhoffOpt.m:427` | `cfg.restorationGuard='R2'` | **experiment ID in solver** | no | — | R2, Bmature, all later | `cfg.stop.guards.maxDesignChange` | **high** |
| 48 | ε_RMS definition | none | `tolOuter/√NE` | `olhoffOpt.m:421` | *none* | **yes** | no | tolOuter | R1, R2 | `cfg.stop.guards.rmsTolerance` | medium |
| 49 | Stop block while p < p_end | none | forced by the experiment definition | `olhoffOpt.m:458-463` | *none* | **yes** | no | **p schedule** | P1, PD1, PM1 | `cfg.material.stiffness.continuation.blockStop` | medium |
| 50 | Outer iteration cap | none | 200 default / 400 / 600 / 1200 | `olhoffOpt.m:105` | `cfg.maxOuter` | no | no | — | all | `cfg.runtime.maxOuter` | low |
| 51 | Convergence field monitored | §3.5.1 monitors **Δρ**, the design increment | `drho` in both formulations — under projection `drho` **is `dz`**, so the monitored field silently becomes the design variable, not the density | `olhoffOpt.m:332-333` | *none* | **yes** | no | **projection** (§2.6) | all | `cfg.stop.field` | **high** |
| 52 | Eigen solver | not stated | `dense` \| `eigs` | `fem/eigSolve.m` | `cfg.solver` | no | no | — | `eigs` frozen | `cfg.runtime.eigen.solver` | low |
| 53 | `eigs` start vector / tol / p | not stated | fixed deterministic `sin(...)`, tol 1e-12, maxit 5000 | `eigSolve.m:34-37` | *none* | **yes** | no | — | all | `cfg.runtime.eigen.*` | medium |
| 54 | Threads | not stated | 1 | `olhoffOpt.m:8` | `cfg.threads` | no | **yes** — runners also call `maxNumCompThreads(1)` | — | all | `cfg.runtime.singleThread` | low |
| 55 | Diagnostics | none | per-iteration `dg` record | `olhoffOpt.m:45,335-356` | `cfg.diag` | no | no | — | all M4 | `cfg.runtime.diagnostics` | low |
| 56 | Geometry / material | §4 `a=8,b=1,E=1e7,ν=0.3,ρ_m=1` | none | `model2D.m`, `elemMats2D.m` | `cfg.a,b,t,E,nu,rhom` | no | no | — | all | `cfg.domain.*`, `cfg.material.solid.*` | low |
| 57 | Mesh | **NE never reported** | 160×20 … 800×100 | `cfg.nelx/nely` | `cfg.nelx/nely` | no | no | tolOuter, rminEl | all | `cfg.domain.mesh` | low |
| 58 | Support idealization | "plane stress elements"; drawing and numbers disagree | `mid` (adopted) \| `corner` \| `face`; `axial='both'` | `model2D.m:52-98` | `cfg.support`, `cfg.axial`, `cfg.bc` | no | no | — | all | `cfg.domain.boundary.*` | low |
| 59 | Element / mass type | "plane stress elements" | `Q4`/`Q6`, `consistent`/`lumped` | `elemMats2D.m` | `cfg.elemType`, `cfg.massType` | no | no | — | all `Q4`/`consistent` | `cfg.domain.element.*` | low |
| 60 | Effective-config recording | — | solver **writes into its own cfg**: `mmasubPath`, `rminEl`, and five `isfield` defaults | `olhoffOpt.m:11-16,25-28` | — | **yes** | no | — | all | resolve before the solver; pass immutable | medium |

---

## 2. Couplings found (Phase 5 classification)

### 2.1 p continuation ↔ move ladder — INTENTIONAL_POLICY, not mathematically required
`olhoffOpt.m:127-131` indexes `cfg.pSchedule` by `mvState.stage`. The audit's own
rationale is explicit: reuse the ladder's stall event so that **no new numerical
constant enters**. That is a defensible policy, and P1 *is* that realization —
so the coupling must remain reproducible. But it is not mathematics: PD1 later
broke it (`cfg.pDecouple`) while keeping the same stall event. Canonically these
are two values of one field, `continuation.driver ∈ {ladderStage, ownCounter}`.

### 2.2 Mass model ↔ p schedule — INTENTIONAL_POLICY
`cfg.massLowP` activates only while `pNow < cfg.pSchedule(end)`. This *defines*
the PM1 experiment ("printed low-p mass model during the low-p phase"), and the
tie to the p schedule is what avoids introducing a threshold. Retain as an
explicit policy field; do not decouple.

### 2.3 Projection continuation ↔ stopping — INTENTIONAL_POLICY
`olhoffOpt.m:446-453` consumes the outer convergence event to advance β_proj.
This is the standard converge–sharpen–reconverge scheme and is deliberate.
Retain, but name it: the trigger is currently unnamed and unreachable.

### 2.4 Diagonal offsets `dOff` ↔ multiplicity method — **HISTORICAL_ACCIDENT**
`olhoffOpt.m:196`: `useOff = strcmpi(cfg.multRule,'subspace')`.
Two independent scientific choices are welded together:
* **how N is chosen** (classifier vs fixed window), and
* **which subeigenvalue problem is solved** — printed (25d), which assumes exact
  degeneracy, versus the reconstruction that retains `diag(λ_j − λ_n)`.
Nothing mathematical requires them to agree. `binary` + offsets, and `subspace`
+ printed (25d), are both meaningful and both currently unreachable. Split into
`cfg.multiplicity.method` and `cfg.multiplicity.diagonalOffsets`, defaulting so
that every historical run maps to its existing behaviour.

### 2.5 Filter type ↔ projection — **HISTORICAL_ACCIDENT**
The density filter exists only inside `projDensityField.m`. Turning projection on
silently *replaces* the Sigmund sensitivity filter with a density filter plus a
chain rule (`olhoffOpt.m:210-231`). Two independent choices — *which filter* and
*whether to project* — share one switch, and the departure from the paper's
printed sensitivity filtering is invisible in the configuration. Phase 11 target:
`cfg.filter.type ∈ {sensitivity, density}` and `cfg.projection.enabled`
independently, with validation rejecting `sensitivity + projection`.

### 2.6 Design variable vs physical density in the stopping test — **HISTORICAL_ACCIDENT**
`dxOuter`/`dxNorm2` are computed from `drho`, which is `Δρ` without projection and
`Δz` with it. The stopping rule therefore silently changes the field it monitors.
`hist.dxPhys2` was added as a "cross-reference only" but nothing reads it. The
architecture must name the monitored field; the historical presets must keep
monitoring what they monitored.

### 2.7 Restoration guard R1 ↔ move ladder — INTENTIONAL_POLICY (but misnamed)
R1 is *defined* as "no remaining ladder level exceeds ε_RMS", so it legitimately
reads `cfg.s2Levels` and `hist.stage`. The defect is naming, not coupling: `'R1'`
and `'R2'` are experiment IDs living inside solver mathematics, exactly the
anti-pattern Phase 3 names. Rename to `ladderExhausted` / `maxDesignChange`.

### 2.8 Move change ↔ RMS stopping test — INTENTIONAL_POLICY, a known defect
Under a ladder `‖Δρ‖_∞ ≤ mv_k`, so an iteration that *reduces* the move limit
mechanically reduces the measured step with no change in the design. The frozen
conference realization carries `outerGuard='settledmove'` precisely to suppress
this. **This deficiency is part of the frozen preset and must be preserved.**

### 2.9 Mesh-scaled ε duplicated across runners — HISTORICAL_ACCIDENT
`tolOuter = 0.05*sqrt(nelx*nely/3200)` is retyped in six runners and in
`run_800.m`. It is a scientific policy with no home in the configuration.

### 2.10 Solver mutates its own configuration — HISTORICAL_ACCIDENT
`olhoffOpt.m:11-16` supplies five defaults by `isfield`, `:16` writes
`cfg.mmasubPath`, `:25-28` overwrites `cfg.rminEl`. The *effective* configuration
therefore does not exist until the solver has already started, and
`useMMA` additionally mutates global MATLAB path state.

---

## 3. `defaultCfg.m` is not any realization

Fields where `algo/defaultCfg.m` differs from the frozen `CFG(k).cfg`:

| Field | `defaultCfg.m` | Frozen `CFG(k)` | Consequence |
|---|---|---|---|
| `massInterp` | `'4'` | `'4b'` | different mass law |
| `tolMult` | `0.02` | `0.05` | different multiplicity detection |
| `multRule` | `'binary'` | `'subspace'` | **different sensitivity formulation** |
| `filterMode` | `'diag'` | `'all'` | different filtering |
| `moveFamily` | `'S0'` | `'S2'` | fixed move vs ladder |
| `move` | `0.05` | `0.04` | different step |
| `s2Levels` | `[0.05 …]` | `[0.04 …]` | different ladder |
| `maxInner` | `300` | `500` | different inner cap |
| `tolInner` | `1e-2` | `0.05` | different inner exit |
| `outerGuard` | `'none'` | `'settledmove'` | different stopping |
| `rminPhys` | `[]` | `0.06` | element- vs physical-radius filter |

Eleven differences, several of which change the algorithm rather than a
tolerance. This is the Phase 15 defect in its clearest form: the file named
"defaults" is a fourth source of truth alongside the `.mat` blob, the runner
literals and the solver's own `isfield` fallbacks.

---

## 4. Experiment identifiers found inside solver mathematics

| Location | Construct | Meaning it should carry |
|---|---|---|
| `olhoffOpt.m:422-430` | `switch upper(cfg.restorationGuard)`, cases `'R1'`,`'R2'` | `stop.guards.ladderExhausted` / `stop.guards.maxDesignChange` |
| `olhoffOpt.m:196` | `useOff = strcmp(cfg.multRule,'subspace')` | `multiplicity.diagonalOffsets` |
| `olhoffOpt.m:146-150` | `cfg.massLowP` + implicit p-schedule dependency | `material.mass.continuation.*` |
| `olhoffOpt.m:120` | `pDecoupled` + assert on `moveFamily` | `stiffness.continuation.driver` |
| `moveControl.m:64` | families `'S0'…'S3'` | `move.policy ∈ {fixed, geometric, ladder, trustRatio}` |
| `moveControl.m:115-129` | `s2Signal` defaulted inside the controller | `move.continuation.signal` |

None of these is a preset name in the strict sense, but `R1`/`R2` are audit
labels, and `S0…S3` and `M0…M4` are candidate indices from the audit
programme — identifiers whose meaning is only recoverable by reading audit
reports.

---

## 5. Duplication

* `mma/subsolv.m` and `mma_published/subsolv.m` are **byte-identical**
  (`130033335ac5…`); only `mmasub.m` differs. The "two MMA variants" are one
  file's worth of difference, selected by mutating the MATLAB path.
* `top88.m` (root) and `filter/top88_reference.m` are byte-identical
  (`eb8613bdb46e…`).
* Five complete solver snapshots exist under `audit_*/` (`baseline/tree/`,
  `audit_projection_800x100/solver/`), plus `*_PRE_*.m` point-in-time copies of
  `olhoffOpt`, `innerLoop`, `deltaLambda`, `defaultCfg`, `moveControl`. These are
  evidence, correctly so, and must stay immutable.
* The mesh-scaled `tolOuter` law appears in six runners (§2.9).

---

## 6. What is *not* broken

Worth recording, because a refactor should not disturb it:

* `fem/massScale.m` already returns **value and derivative together**, dispatches
  on a semantic model name, and cites the printed equation for each branch. This
  is close to the Phase 10 target already.
* `algo/multRule.m` and `algo/moveControl.m` carry explicit A/B/C evidence
  classifications in their headers.
* `filter/projectDensity.m` documents that its `betaProj` is unrelated to the
  bound-formulation `β` of eq. (25a) — a real naming hazard, already handled.
* Every optional path is genuinely default-off, and each audit proved its
  default-off inertness bitwise before use.
