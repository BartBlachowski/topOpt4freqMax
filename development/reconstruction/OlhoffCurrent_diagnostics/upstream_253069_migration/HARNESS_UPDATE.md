# HARNESS_UPDATE — Parts 13, 14, 15, 16

```
BENCHMARK_PREFLIGHT_PASS
```

Evidence: `evidence/harness_check.json` (`scripts/mig_harness_check.m`).

- `confbench_preflight` at 160×20 on the campaign method set, **no solve**: **38/38 checks pass**.
- Mechanics self-test: T1–T14 → 13 pass. T2 fails, identically on a pristine `013cc48` (pre-existing, KNOWN_PREEXISTING_DEFECTS.md D2).
- Olhoff column smoke through `confbench_method_config → confbench_run_case`: 160×20, outer cap cut to 3; software only.
- `confbench_manifest` + `confbench_export` into a scratch directory.

## Production preset decision (Part 13)

**Selected production preset: `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`.**
**Historical preset retained: `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered`** (and the stage-exhaustion diagnostic), unchanged and resolvable by name.

Rationale. It rests on the prior audits, not on topology appearance and not on this migration's runs.

1. **It is a distinct, scientifically defensible reconstruction** (scientific_delta_olhoff_migration). Pedersen (2000) is the remedy Du & Olhoff §2.2 name; linear mass eq. (2) is printed; the adaptive box is a class C reconstruction of the same standing as the ladder it replaces.
2. **Its success is primarily formulation stability.** The committed nine-mesh R = 0.06 sweep terminates naturally on every mesh without localized-mode spikes. The same controller under SIMP + eq. (4b) collapsed into localized modes and stopped falsely (M1, 480×60).
3. **The historical formulation's cross-mesh termination was not credible as benchmark evidence** (nine_mesh_campaign_audit: all nine campaign stops one iteration after a move reduction, `TERMINATION_CROSS_MESH_NOT_CREDIBLE`). Its intended successor, the three-rung controller, was never promoted, and its cross-mesh behaviour is `INCONCLUSIVE`.
4. **It is not claimed to be more KKT-stationary.** The source endpoint is not demonstrably closer to physical stationarity. The natural stop is a heuristic; the caveat says so.
5. **The historical formulation stays reproducible**, bitwise, under its own name (HISTORICAL_REPRODUCTION.md). Nothing about it is lost by the selection.

The selection is a recorded provenance event (event 2 of `PROVENANCE.json → production_preset_events`, 2026-09-13), not an edit. The benchmark names the preset explicitly in one place (`confbench_olhoff_preset.m`), and the preflight refuses to run unless that name **is** the recorded production preset.

## Changes, by file (all Olhoff-specific)

| file | change | Proposed / Yuksel affected? |
|---|---|---|
| `conference_bench/confbench_olhoff_preset.m` (new) | the single place the benchmark names its Olhoff preset | no |
| `conference_bench/confbench_olhoff_assertions.m` (new) | per-preset field assertions. Pedersen set: Pedersen stiffness 0.1, eq2 mass, adaptive box 0.10/0.002/×1.2/×0.7, stage exhaustion off (move and stop), natural stop with no guards, cap 400, inner asymptotes reset, p = 3, ρ_min 1e-3, maxCluster 4, R = 0.06 physical on all f_sk, subspace N = 2, published MMA on the increment, tolInner 0.05 5…500, single thread, diagnostics off. β-stall set: the historical S2-ladder / settledmove assertions retained verbatim plus SIMP + eq4b. Any other preset: refused. | no |
| `conference_bench/confbench_preflight.m` | §5 notes use the recorded branch/head and production event; new §5a (benchmark preset canonical, eligible, = recorded production); §5b per-preset assertions via `confbench_olhoff_assertions` (replaces the hard-coded "S2 ladder", "settledmove" and "M4 multiplicity" checks); §11 timing-schema key | no (Proposed/Yuksel checks untouched) |
| `conference_bench/confbench_method_config.m` | Olhoff branch resolves the NAMED preset; `mcfg.olhoff_preset`; profile carries display name, role, upstream commit, formulation, aliases, distinct-from, per-preset caveat | no (their branches untouched) |
| `conference_bench/confbench_run_case.m` | `runOlhoff` passes `'Preset'`; `rec.times` gains the per-outer cost fields | no (shared record field set unchanged; T3 passes) |
| `conference_bench/confbench_display_name.m` | Olhoff label = preset display name: "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" instead of "(M4)" | no |
| `conference_bench/confbench_caveats.m` | Olhoff caveat, label and iteration-count text per preset (adaptive-box, non-monotone counts); scaling caveat names the per-outer fit | no (shared caption/memory strings unchanged) |
| `conference_bench/confbench_timing_schema.m` | key `DuOlhoffReconstructionM4` → `DuOlhoffReconstruction` with `preset`, `upstream_*`; per-outer-cost definitions | no |
| `conference_bench/confbench_export.m` | Olhoff-gated per-outer columns in the detailed CSV (+ `olhoff_preset`); notes: formulation-bearing header and a "Cost per outer iteration" table; per-outer fit table | no (gated with `isO`) |
| `conference_bench/confbench_scaling_fit.m` | `scaling.per_outer` fits for methods whose records carry `counts.outer_iterations` (Olhoff only); total-time fit unchanged | no (T12 unchanged and passing; T14 proves Yuksel gets none) |
| `conference_bench/confbench_manifest.m` | hashes the new preset files; records benchmark/production preset, event date, registry, parent commit | no |
| `conference_bench/confbench_selftest.m` | T6 names the preset; new T13 (assertions formulation-specific), T14 (per-outer fit) | no |
| `conference_bench/confbench_frozen_budget.m` | error text points to the preset registry | no |
| `performance_comparison.m` | Olhoff `printMethodSettings` prints preset, stiffness/mass law, move policy (adaptive box or ladder + signal), stop rule; per-outer fits printed | no |

Not changed: the Proposed and Yuksel solvers, `profile_freeze_manifest.json`, `study_base_config.m`, `study_evaluate_design.m`, `run_topopt_from_json.m`, any filter radius, the evaluator policy, the committed `campaign_9mesh_r2` outputs.

## Stale assumptions removed

| assumption | where it was | now |
|---|---|---|
| "S2 continuation realization as frozen" asserted for production | preflight §5b | asserted only under the β-stall preset |
| `outerGuard == 'settledmove'` for production | preflight §5b | Pedersen requires `none`; β-stall keeps `settledmove` |
| "M4 multiplicity treatment" label | preflight, `printMethodSettings` | "fixed subspace multiplicity, subN = 2" |
| method label "Du-Olhoff reconstruction (M4)" | display name, notes, interpretation, LaTeX comment, timing schema | the preset's display name |
| "outer count depends on a move-limit continuation schedule" | caveats | per-preset text |
| `olhoffcurrent_config(nelx, nely)` without a preset | method config, preflight, selftest T6 | named preset everywhere |

## Preflight behaviour

| case | outcome | how established |
|---|---|---|
| actual production configuration at 160×20 | 38/38 pass | run (`harness_check.json`) |
| Pedersen assertions on the β-stall configuration | 5 checks fail | run (self-test T13) |
| β-stall assertions on the Pedersen configuration | 4 checks fail | run (T13) |
| stage-exhaustion diagnostic | its assertion set refuses it | run (T13) |
| stage-exhaustion diagnostic named as the benchmark preset | §5a "canonical, production-eligible" check fails (`productionEligible = false`) | by construction; not run as a separate probe |
| benchmark preset ≠ recorded production preset | §5a "is the recorded production preset" check fails | by construction; not run as a separate probe |

## Smoke output checks (`evidence/harness_check.json → smoke, export`)

- status `CAP_HIT`, `ok = false` (a cut cap is never convergence);
- `rec.production_preset = duOlhoffPedersenAdaptiveBoxSensitivityFiltered`; `rec.method` = the formulation label;
- all per-outer fields present;
- detailed CSV has the per-outer columns and the preset cell;
- BENCHMARK_NOTES header is the formulation label and has the per-outer table; no "(M4)" label in the new outputs;
- timing schema key `DuOlhoffReconstruction` with the preset;
- manifest records benchmark preset = production preset.
