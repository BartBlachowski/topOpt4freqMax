# CONFIG_ASSERTIONS — the field-by-field requirement table

**Status: RUNTIME-VERIFIED.** Every "runtime" value below was read back out of
the configuration `olh.config.resolve` actually returned — after defaults,
preset, overrides, derived rules and validation — using `olh.config.getPath`,
by `scripts/cp_preflight.m`, **before** any optimization ran. 47 of 47 checks
pass at each mesh, 0 blockers.

Records: `evidence/preflight_480x60.json`, `evidence/preflight_800x100.json`.

## The single scalar that stands for the whole table

| mesh | frozen expected (written before MATLAB was available) | runtime measured | match |
|---|---|---|---|
| 480×60 | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` | identical | ✓ |
| 800×100 | `7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe` | identical | ✓ |

The expected hashes were frozen in `PREFLIGHT_MANIFEST.json` before MATLAB could
be started on this host, using an offline reconstruction of
`olhoffcurrent_config_hash` validated against ten independently recorded
hashes — the nine legacy campaign configs (9/9) and the validated three-rung
C320 config `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`.
The runtime resolution reproduced them bit for bit. This is a pre-run
commitment that was met, not a post-run value declared to match.

A four-rung variant of the same construction produces a different hash
(`2359a111…50f4`), so the check is sensitive to the single factor under test.

## 1. The controller under test

| field | required | 480×60 runtime | 800×100 runtime | ok |
|---|---|---|---|---|
| `move.levels` | `[0.040000000000000001 0.02 0.01]` | `[0.040000000000000001 0.02 0.01]` | `[0.040000000000000001 0.02 0.01]` | ✓ |
| `move.continuation.signal` | `stageExhaustion` | `stageExhaustion` | `stageExhaustion` | ✓ |
| `stop.rule` | `stageExhaustion` | `stageExhaustion` | `stageExhaustion` | ✓ |
| `move.policy` | `ladder` | `ladder` | `ladder` | ✓ |

## 2. The scientific locks

| field | required | 480×60 runtime | 800×100 runtime | ok |
|---|---|---|---|---|
| `material.stiffness.p` | `3` | `3` | `3` | ✓ |
| `material.stiffness.continuation.enabled` | `false` | `false` | `false` | ✓ |
| `material.mass.model` | `eq4b` | `eq4b` | `eq4b` | ✓ |
| `material.mass.q` | `1` | `1` | `1` | ✓ |
| `material.mass.continuation.enabled` | `false` | `false` | `false` | ✓ |
| `filter.type` | `sensitivity` | `sensitivity` | `sensitivity` | ✓ |
| `filter.applyTo` | `all` | `all` | `all` | ✓ |
| `filter.radiusPhysical` | `0.059999999999999998` | `0.059999999999999998` | `0.059999999999999998` | ✓ |
| `projection.enabled` | `false` | `false` | `false` | ✓ |
| `multiplicity.method` | `subspace` | `subspace` | `subspace` | ✓ |
| `multiplicity.subspaceSize` | `2` | `2` | `2` | ✓ |
| `multiplicity.diagonalOffsets` | `true` | `true` | `true` | ✓ |
| `multiplicity.offDiagonal` | `true` | `true` | `true` | ✓ |
| `multiplicity.tolerance` | `0.050000000000000003` | `0.050000000000000003` | `0.050000000000000003` | ✓ |
| `optimizer.inner.type` | `mma` | `mma` | `mma` | ✓ |
| `optimizer.inner.variant` | `published` | `published` | `published` | ✓ |
| `optimizer.inner.variable` | `increment` | `increment` | `increment` | ✓ |
| `optimizer.inner.tolerance` | `0.050000000000000003` | `0.050000000000000003` | `0.050000000000000003` | ✓ |
| `optimizer.inner.minIterations` | `5` | `5` | `5` | ✓ |
| `optimizer.inner.maxIterations` | `500` | `500` | `500` | ✓ |
| `eigen.solver` | `eigs` | `eigs` | `eigs` | ✓ |
| `eigen.targetMode` | `1` | `1` | `1` | ✓ |
| `eigen.maxCluster` | `4` | `4` | `4` | ✓ |
| `eigen.tolerance` | `9.9999999999999998e-13` | `9.9999999999999998e-13` | `9.9999999999999998e-13` | ✓ |
| `eigen.krylovFactor` | `4` | `4` | `4` | ✓ |
| `design.initial` | `0.5` | `0.5` | `0.5` | ✓ |
| `design.minimum` | `0.001` | `0.001` | `0.001` | ✓ |
| `design.volumeFraction` | `0.5` | `0.5` | `0.5` | ✓ |
| `domain.a` | `8` | `8` | `8` | ✓ |
| `domain.b` | `1` | `1` | `1` | ✓ |
| `domain.boundary.condition` | `simplySupported` | `simplySupported` | `simplySupported` | ✓ |
| `domain.boundary.support` | `midHeight` | `midHeight` | `midHeight` | ✓ |
| `domain.element.type` | `Q4` | `Q4` | `Q4` | ✓ |
| `domain.element.massMatrix` | `consistent` | `consistent` | `consistent` | ✓ |
| `stop.norm` | `l2` | `l2` | `l2` | ✓ |
| `stop.field` | `designVariable` | `designVariable` | `designVariable` | ✓ |
| `stop.toleranceRule` | `meshScaled` | `meshScaled` | `meshScaled` | ✓ |

## 3. Mesh, tolerance, cap, thread policy, telemetry

| field | required | 480×60 runtime | 800×100 runtime | ok |
|---|---|---|---|---|
| `domain.mesh.nelx` | `480` | `480` | `800` | ✓ |
| `domain.mesh.nely` | `60` | `60` | `100` | ✓ |
| `stop.tolerance` | `0.15000000000000002` | `0.15000000000000002` | `0.25` | ✓ |
| `runtime.maxOuter` | `1600` | `1600` | `1600` | ✓ |
| `runtime.singleThread` | `true` | `true` | `true` | ✓ |
| `runtime.diagnostics` | `true` | `true` | `true` | ✓ |

## 4. Structural checks beyond the field table

| check | 480×60 | 800×100 |
|---|---|---|
| `olhoffcurrent_assert_dispatch` — exactly one Olhoff implementation visible, no helper shadowing | OK | OK |
| `+impl` tree hash | `edbfe47e…52cb`, 75/75 files verified | same |
| tree identical to the `implTree` recorded inside the validated C320 run | yes | yes |
| beta holds continuation authority | **false** | **false** |
| beta holds stop authority | **false** | **false** |
| `move.levels` still contains the removed 0.005 rung | **no** | **no** |
| required `hist` telemetry fields missing | **none** | **none** |
| retention budget within a quarter of RAM | 0.74 GB — yes | 2.05 GB — yes |

Beta's lack of authority is asserted **structurally**, not by name: the check is
that `move.continuation.signal ≠ boundVariable` (so `olh.move.limit` returns
from its `stageExhaustion` branch before the bound-variable window is ever
formed) and `stop.rule ≠ designChange` (so `olhoffSolve` replaces the sec. 3.5.1
admission wholesale rather than combining with it).

`olh.config.validate` gives a second, independent guarantee: it refuses to
resolve `stop.rule = stageExhaustion` without the matching continuation signal,
so a half-promoted controller cannot reach the solver at all.

## 5. Mesh-derived quantities, cross-checked against independent records

| quantity | 480×60 | 800×100 | agrees with |
|---|---|---|---|
| NE | 28 800 | 80 000 | — |
| free DOF = 2(nelx+1)(nely+1) − 4 | 58 678 | 161 798 | recorded legacy campaign values |
| `stop.tolerance` = 0.05·√(NE/3200) | 0.15 | 0.25 | recorded legacy `eps_l2` |
| element-space filter radius R/(b/nely) | 3.6 | 6.0 | recorded legacy `rminEl` |

## 6. What is NOT asserted, and why

The MATLAB build is **recorded, not asserted**. This host reports
`25.2.0.2998904 (R2025b)`; the validated C320 run recorded
`25.2.0.3042426 (R2025b) Update 1`. No expected toolchain version could be
frozen honestly, because none had ever been observed on this host when the
manifest was written. The difference is disclosed in `DEPLOYMENT_PREFLIGHT.md`
and matters only to a floating-point-exact comparison against the validated
C320 endpoint — a comparison this study does not make, since the canaries sit
at meshes the validated runs never visited.
