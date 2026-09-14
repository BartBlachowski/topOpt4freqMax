# CONFIG_COMPARISON — effective configurations at 480×60

Four configurations flattened leaf by leaf (`scripts/sd_config_dump.m`, `evaluations/effective_configs_480.json`):

- **S480** — source committed run `repro/results/S480x60` (stored cfg), preset `duOlhoffAdaptivePedersen`.
- **M1** — this audit's single run: source code, preset `duOlhoffAdaptiveMove` + `move.initial = 0.10`.
- **C480** — target three-rung canary (stored cfg, hash `03097a28…` re-verified).
- **PROD** — target canonical production resolved now by `olhoffcurrent_config(480,60)` (hash `a49417d0…`).

Leaves: 96; identical in all four: **70**; differing: 26.

| leaf | S480 | M1 | C480 | PROD | kind |
|---|---|---|---|---|---|
| `material.mass.model` | eq2 | eq4b | eq4b | eq4b | FORMULATION |
| `material.stiffness.linearBelow` | 0.10000000000000001 | 0.10000000000000001 | <absent> | <absent> | FORMULATION |
| `material.stiffness.model` | pedersen | simp | simp | simp | FORMULATION |
| `move.adaptive.grow` | 1.2 | 1.2 | <absent> | <absent> | CONTROLLER |
| `move.adaptive.shrink` | 0.69999999999999996 | 0.69999999999999996 | <absent> | <absent> | CONTROLLER |
| `move.continuation.signal` | boundVariable | boundVariable | stageExhaustion | boundVariable | CONTROLLER |
| `move.initial` | 0.10000000000000001 | 0.10000000000000001 | 0.040000000000000001 | 0.040000000000000001 | CONTROLLER |
| `move.levels` | [0.040000000000000001 0.02 0.01 0.0050000000000000001] | [0.040000000000000001 0.02 0.01 0.0050000000000000001] | [0.040000000000000001 0.02 0.01] | [0.040000000000000001 0.02 0.01 0.0050000000000000001] | CONTROLLER |
| `move.policy` | adaptive | adaptive | ladder | ladder | CONTROLLER |
| `optimizer.inner.asymptoteHistory` | inner | inner | <absent> | <absent> | INNER (inert on executed path) |
| `provenance.historicalAliases` | <absent> | <absent> | <absent> | M4 / TMA / B0 / REG160 / duOlhoffFrozenM4 | METADATA |
| `provenance.implementation` | <absent> | <absent> | analysis/OlhoffCurrent | analysis/OlhoffCurrent | METADATA |
| `provenance.overrides` | domain.mesh.nelx / 480 / domain.mesh.nely / 60 / runtime.nam | domain.mesh.nelx / 480 / domain.mesh.nely / 60 / runtime.nam | domain.mesh.nelx / 480 / domain.mesh.nely / 60 / runtime.max | domain.mesh.nelx / 480 / domain.mesh.nely / 60 / runtime.max | METADATA |
| `provenance.preset` | duOlhoffAdaptivePedersen | duOlhoffAdaptiveMove | duOlhoffFrozenM4 | duOlhoffFrozenM4 | METADATA |
| `provenance.productionPreset` | <absent> | <absent> | duOlhoffFixedPenaltySensitivityFiltered | duOlhoffFixedPenaltySensitivityFiltered | METADATA |
| `provenance.resolvedAt` | 2026-09-13T11:51:29 | 2026-09-13T16:44:09 | 2026-09-12T11:40:12 | 2026-09-13T16:56:02 | METADATA |
| `provenance.schemaRows` | 86 | 86 | 81 | 81 | METADATA |
| `provenance.upstreamPreset` | <absent> | <absent> | <absent> | duOlhoffFrozenM4 | METADATA |
| `runtime.diagnostics` | false | true | true | false | RUNTIME |
| `runtime.maxOuter` | 400 | 400 | 1600 | 400 | RUNTIME |
| `runtime.name` | S480x60 | M1_480x60_simp4b_adaptive | CAN3_480x60 | OLHOFF_CURRENT_480x60 | RUNTIME |
| `runtime.verbose` | true | true | false | false | RUNTIME |
| `stop.guards.boxInactiveFraction` | 0 | 0 | <absent> | <absent> | STOPPING |
| `stop.guards.settledMove` | false | false | true | true | STOPPING |
| `stop.guards.settledWindow` | 1 | 1 | <absent> | <absent> | STOPPING |
| `stop.rule` | <absent> | <absent> | stageExhaustion | designChange | STOPPING |

## Reading

- **S480 vs M1:** only `material.stiffness.model` and `material.mass.model` among scientific leaves (M1 preflight, Amendment 1).
- **M1 vs C480:** only controller (`move.policy`, `move.initial`, adaptive factors, continuation signal), stopping (`stop.rule`, guards) and runtime leaves. Every problem/physics leaf (material, filter, design, eigen, multiplicity, inner optimizer) is equal → D0 passes under M1.
- **C480 vs PROD:** `move.levels` (3 vs 4 rungs), `move.continuation.signal`, `stop.rule`, `runtime.maxOuter`, diagnostics — the three-rung candidate has never been promoted (three_rung_promotion_closure: PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED).
- Identical in all four (selection): `filter.radiusPhysical = 0.06`, `filter.type = sensitivity`, `filter.applyTo = all`, `stop.tolerance = 0.15`, `stop.norm = l2`, `design.initial = 0.5`, `design.minimum = 1e-3`, `design.volumeFraction = 0.5`, `material.stiffness.p = 3`, `multiplicity.method = subspace`, `subspaceSize = 2`, `diagonalOffsets = offDiagonal = true`, `optimizer.inner.{type=mma, variant=published, variable=increment, tolerance=0.05, minIterations=5, maxIterations=500}`, `eigen.{solver=eigs, tolerance=1e-12, maxIterations=5000, krylovFactor=4, maxCluster=4}`. Full list in `evaluations/config_comparison.json`.
