# CONFIG_REFERENCE

Every field of the canonical configuration.

**Generated from `olh.config.schema` by `architecture/tests/gen_config_reference.m`.**
Do not edit by hand: regenerate it.

Provenance classes: **A** specified by a Du-Olhoff source; **B** implied by one;
**C** under-specified reconstruction choice; **D** later experimental modification.
The evidence for each letter is in `SCIENTIFIC_CONFIG_PROVENANCE.md`.

| Class | Fields |
|---|---|
| **A** | 23 |
| **B** | 4 |
| **C** | 41 |
| **D** | 12 |
| total | 80 |


## cfg.domain

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `domain.a` | double | `8` | [0, Inf] | A | beam length |
| `domain.b` | double | `1` | [0, Inf] | A | beam height |
| `domain.thickness` | double | `1` | [0, Inf] | A | out-of-plane thickness |
| `domain.mesh.nelx` | int | `160` | [1, Inf] | C | elements along the length; NE is never reported in the paper |
| `domain.mesh.nely` | int | `20` | [1, Inf] | C | elements through the height |
| `domain.boundary.condition` | enum | `'simplySupported'` | `simplySupported`, `clampedSimple`, `clamped` | A | Fig. 2(a-c) support cases |
| `domain.boundary.support` | enum | `'midHeight'` | `midHeight`, `corner`, `face` | C | idealization of a SIMPLE support; the paper draws corner, its numbers fit midHeight |
| `domain.boundary.axialRestraint` | enum | `'bothEnds'` | `oneEnd`, `bothEnds` | C | which ends carry ux restraint |
| `domain.element.type` | enum | `'Q4'` | `Q4`, `Q6` | C | the paper says only "plane stress elements" |
| `domain.element.massMatrix` | enum | `'consistent'` | `consistent`, `lumped` | C | element mass matrix form |

## cfg.material

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `material.solid.E` | double | `1e+07` | [0, Inf] | A | Young modulus |
| `material.solid.nu` | double | `0.3` | [-1, 0.5] | A | Poisson ratio |
| `material.solid.density` | double | `1` | [0, Inf] | A | solid mass density rho_m |
| `material.stiffness.model` | enum | `'simp'` | `simp` | A | eq. (1): Ee = rho^p * Ee |
| `material.stiffness.p` | double | `3` | [1, Inf] | A | penalization power of eq. (1) |
| `material.stiffness.continuation.enabled` | logical | `false` | `true`, `false` | A | sec. 2.1: p is "normally assigned values increasing from 1 to 3" |
| `material.stiffness.continuation.schedule` | vector | `[]` | numeric vector or `[]` | B | successive values of p; endpoints 1 and 3 are printed, the schedule is not |
| `material.stiffness.continuation.driver` | enum | `'moveLadderStage'` | `moveLadderStage`, `ownCounter` | C | what advances p: the move ladder stage, or p own counter consuming the same stall event |
| `material.stiffness.continuation.blockStopUntilFinal` | logical | `true` | `true`, `false` | B | a p=3 problem may not be declared converged while p<3 |
| `material.mass.model` | enum | `'eq4b'` | `eq2`, `eq4`, `eq4a`, `eq4b` | A | printed mass interpolation: (2) linear, (4) discontinuous, (4a) C0, (4b) C1 |
| `material.mass.q` | double | `1` | [1, Inf] | A | eq. (2) exponent; "normally, q=1 is chosen" |
| `material.mass.lowDensityExponent` | double | `6` | [1, Inf] | A | eq. (4) exponent r; "r is chosen to be about r=6" |
| `material.mass.cutoff` | double | `0.1` | [0, 1] | A | density below which the low-density branch applies |
| `material.mass.continuation.enabled` | logical | `false` | `true`, `false` | D | switch mass model while p is below its final value |
| `material.mass.continuation.lowPModel` | enum | `'eq2'` | `eq2`, `eq4`, `eq4a`, `eq4b` | D | model in force during the low-p phase |

## cfg.design

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `design.initial` | double | `0.5` | [0, 1] | A | uniform initial density |
| `design.minimum` | double | `0.001` | [0, 1] | A | rho_min of the box (7e)/(25f) |
| `design.volumeFraction` | double | `0.5` | [0, 1] | A | alpha of the volume constraint (25e) |

## cfg.filter

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `filter.type` | enum | `'sensitivity'` | `sensitivity`, `density`, `none` | A | sec. 1: Sigmund (1997) filter "applied to the sensitivities"; density filtering is NOT published |
| `filter.radiusPhysical` | any | `0.06` | see validation | C | filter radius in physical units; never stated in the paper. Overrides radiusElements when non-empty |
| `filter.radiusElements` | any | `[]` | see validation | C | filter radius in element units |
| `filter.applyTo` | enum | `'all'` | `diagonal`, `all` | C | sensitivity filtering: only f_jj, or every f_sk. The paper has one sensitivity vector and does not say |

## cfg.projection

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `projection.enabled` | logical | `false` | `true`, `false` | D | tanh Heaviside projection; absent from every source |
| `projection.eta` | double | `0.5` | [0, 1] | D | projection threshold |
| `projection.beta.levels` | vector | `[]` | numeric vector or `[]` | D | monotone non-decreasing projection sharpness levels |
| `projection.continuation.trigger` | enum | `'outerConvergence'` | `outerConvergence` | D | what advances the sharpness level |

## cfg.eigen

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `eigen.targetMode` | int | `1` | [1, Inf] | A | n: which eigenfrequency is maximized |
| `eigen.maxCluster` | int | `4` | [1, Inf] | C | Nmax; J = n + Nmax modes are extracted |
| `eigen.solver` | enum | `'eigs'` | `eigs`, `dense` | C | generalized eigensolver |
| `eigen.tolerance` | double | `1e-12` | [0, Inf] | C | eigs convergence tolerance |
| `eigen.maxIterations` | int | `5000` | [1, Inf] | C | eigs iteration cap |
| `eigen.krylovFactor` | int | `4` | [1, Inf] | C | eigs subspace size factor: p = max(20, factor*J) |

## cfg.multiplicity

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `multiplicity.method` | enum | `'subspace'` | `binary`, `latch`, `hysteresis`, `subspace` | C | how the multiplicity N of omega_n is decided each outer iteration |
| `multiplicity.tolerance` | double | `0.05` | [0, 1] | A | sec. 3.5.1 measure: relative frequency difference. The VALUE is never given |
| `multiplicity.enterTolerance` | double | `0.01` | [0, 1] | C | hysteresis entry threshold |
| `multiplicity.exitTolerance` | double | `0.05` | [0, 1] | C | hysteresis exit threshold |
| `multiplicity.subspaceSize` | int | `2` | [1, Inf] | C | fixed cluster size when method is subspace |
| `multiplicity.diagonalOffsets` | logical | `true` | `true`, `false` | C | retain diag(lambda_j - lambda_n) in (25d). Printed (25d) assumes EXACT degeneracy; this is reconstruction |
| `multiplicity.offDiagonal` | logical | `true` | `true`, `false` | A | true = full (25d) determinant; false = force f_sk'drho=0, the Krog & Olhoff LP route |

## cfg.optimizer

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `optimizer.inner.type` | enum | `'mma'` | `mma`, `lp` | A | sec. 3.5.3: "the MMA method (Svanberg 1987) has been used" |
| `optimizer.inner.variable` | enum | `'increment'` | `increment`, `design` | C | increment: MMA state reset each outer iteration; design: asymptotes persist |
| `optimizer.inner.variant` | enum | `'published'` | `published`, `asfound` | B | published = Svanberg Sept-2007 constants; asfound = local lineage copy |
| `optimizer.inner.tolerance` | double | `0.05` | [0, Inf] | C | relative inner step test; Fig. 1 gives no criterion |
| `optimizer.inner.minIterations` | int | `5` | [0, Inf] | C | sub-iterates always taken |
| `optimizer.inner.maxIterations` | int | `500` | [1, Inf] | C | inner iteration cap |

## cfg.move

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `move.policy` | enum | `'ladder'` | `fixed`, `geometric`, `ladder`, `trustRatio` | C | the paper places NO bound on drho other than the box (25f) |
| `move.initial` | double | `0.04` | [0, 1] | C | move limit, and the starting value for geometric/trustRatio |
| `move.minimum` | double | `0.002` | [0, 1] | C | floor for geometric/trustRatio |
| `move.levels` | vector | `[0.04 0.02 0.01 0.005]` | numeric vector or `[]` | C | descending ladder levels |
| `move.geometric.ratio` | double | `0.97` | [0, 1] | C | geometric contraction ratio |
| `move.geometric.afterCoalescence` | logical | `true` | `true`, `false` | C | start contracting only once N>=2 is first seen |
| `move.trust.loRatio` | double | `0.3` | [0, Inf] | D | shrink below this realized/predicted gain ratio |
| `move.trust.hiRatio` | double | `0.7` | [0, Inf] | D | grow above this ratio |
| `move.trust.shrink` | double | `0.7` | [0, 1] | D | contraction factor |
| `move.trust.grow` | double | `1.1` | [1, Inf] | D | expansion factor |
| `move.continuation.signal` | enum | `'boundVariable'` | `boundVariable`, `designRms` | C | what the ladder stall detector watches: the bound variable beta of (25a), or ||drho||/sqrt(NE) |
| `move.continuation.window` | int | `10` | [1, Inf] | C | stall detector window W |
| `move.continuation.tolerance` | double | `0.005` | [0, Inf] | C | relative-progress threshold below which a stall is declared |

## cfg.stop

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `stop.norm` | enum | `'l2'` | `l2`, `max` | B | sec. 3.5.1 writes "the norm" unqualified; l2 is the natural reading |
| `stop.tolerance` | double | `0.05` | [0, Inf] | C | epsilon of Fig. 1; never given in the paper |
| `stop.toleranceRule` | enum | `'meshScaled'` | `explicit`, `meshScaled` | C | meshScaled recomputes stop.tolerance as 0.05*sqrt(NE/3200) AFTER mesh overrides, so eps means the same RMS density change at every resolution |
| `stop.field` | enum | `'designVariable'` | `designVariable` | A | sec. 3.5.1 monitors the DESIGN increment. Under projection that is dz, not d(rho_phys) |
| `stop.guards.settledMove` | logical | `true` | `true`, `false` | C | assert convergence only when the move limit is unchanged from the previous iteration |
| `stop.guards.ladderExhausted` | logical | `false` | `true`, `false` | D | assert convergence only when no remaining ladder level exceeds epsilon/sqrt(NE) |
| `stop.guards.maxDesignChange` | logical | `false` | `true`, `false` | D | assert convergence only when max|d(design)| < epsilon/sqrt(NE) |

## cfg.runtime

| Field | Type | Default | Admissible | Class | Meaning |
|---|---|---|---|---|---|
| `runtime.maxOuter` | int | `400` | [1, Inf] | C | outer iteration cap; reaching it is CAP_HIT, not convergence |
| `runtime.singleThread` | logical | `true` | `true`, `false` | C | required for meaningful complexity measurement |
| `runtime.diagnostics` | logical | `false` | `true`, `false` | C | per-iteration diagnostic record; provably inert |
| `runtime.verbose` | logical | `false` | `true`, `false` | C | per-iteration console table |
| `runtime.name` | char | `''` | text | C | free-text run label; never read by solver mathematics |

