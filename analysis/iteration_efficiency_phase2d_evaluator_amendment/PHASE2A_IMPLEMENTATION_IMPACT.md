# WP20 — Phase-2A implementation impact
OFFLINE_AMENDMENT_VALIDATION — NO_NEW_OPTIMIZATION — NO PRODUCTION AUTHORIZATION

## A — changes made NOW, to validate the amendment offline

| artifact | change |
|---|---|
| `+ie2d/study_evaluate_design_eq4a.m` | created, isolated; two functional lines differ from the frozen evaluator |
| `QUALITY_EFFORT_SPEC.md`, `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` | native-identity claims retired/qualified (they are now false as written) |

Nothing else. No Phase-2A implementation artifact was modified.

## B — changes required AFTER independent delta audit and refreeze

| artifact | required change | why deferred |
|---|---|---|
| `analysis/three_method_parametric_study/study_evaluate_design.m` | apply the two-line Eq. (4a) change | it is a `protected_numerical_sources` entry and the contract pins its SHA-256; changing it now would break Phase-2A/2B/2C provenance verification |
| `iteration_efficiency_contract.json` → `quality.evaluators[E2].mass`, `[E3].mass`, both `identity` fields | state Eq. (4a) and the retired native identity | contract hash is checked by `production_preflight` `contract_hash` against the hard-coded `frozen_contract_sha256()` |
| `iteration_efficiency_contract.json` → `quality.source_sha256` | new evaluator hash | checked by `production_preflight` `evaluator_hash` |
| `+ie2a/frozen_contract_sha256.m` | new frozen contract hash literal | refreeze action, not an author action |
| `+ie2a/run_negative_controls.m` | any control asserting the old E2/E3 mass strings | must track the contract |
| `METHODOLOGY_FREEZE_RECORD.md` | amendment entry | refreeze action; editing it now is the self-refreeze WP22 forbids |
| Olhoff precision-qualification artifact / staleness logic | a NEW qualification is required (WP19); the Phase-2B artifact was never created, so nothing stale exists to invalidate | after refreeze |

**Preflight consequence, stated plainly:** the moment the contract and evaluator are amended,
`production_preflight` fails on `contract_hash` and `evaluator_hash` until
`frozen_contract_sha256()` and `quality.source_sha256` are updated as part of the refreeze.
That is the gate working as designed. It must be done by the refreeze, not by this phase.

## C — changes NOT required at all

`+ie2a/reference_phase.m`, `scan_persistence.m`, `measurement_budget.m`,
`exact_count_binary.m`, `topology_metrics.m`, `account_iterations.m`,
`timing_replay_plan.m`, `run_timing_replays.m`, `fit_power_law.m`, `generate_scaling.m`,
`classify_status.m` — all consume evaluator *outputs* or densities and contain no
interpolation law. They were reused verbatim throughout this validation.

Every native optimizer source, `massScale.m`, `defaultCfg.m`, `mass_interp.m` — unchanged
by design; all hashes verified before and after.

## Production

Not authorized. No token set, no campaign started, `production_preflight.m` unmodified.
