# PRESET_IDENTITY — Part 7

```
PEDERSEN_PRESET_DISTINCT_IDENTITY_PASS
```

Registry: `analysis/OlhoffCurrent/olhoffcurrent_presets.m`. Lookup: `olhoffcurrent_preset(name)`; a name is required. Production: `olhoffcurrent_production_preset()`, which reads the latest `PROVENANCE.json → production_preset_events` entry.

## The table

| canonical preset name | compatibility alias (resolvable) | historical / provenance aliases (refused as names) | stiffness law | mass law | controller | stop rule | intended purpose | evidence provenance | production status |
|---|---|---|---|---|---|---|---|---|---|
| **`duOlhoffPedersenAdaptiveBoxSensitivityFiltered`** | — | `duOlhoffAdaptivePedersen`, S160x20 … S800x100 | SIMP ρ³ above ρ₀ = 0.1; Pedersen (2000) eq. (5) linear ρ·ρ₀^(p−1) below | eq. (2) linear, q = 1 | per-element adaptive box: 0.10 initial/ceiling, 0.002 floor, ×1.2 monotone / ×0.7 reversal | ‖Δρ‖₂ < 0.05·√(NE/3200), no guards, no persistence (`designChange`, settledMove off) | the successful source formulation; benchmark production | Olhoff `6b08708` sweep `repro/results/S*` (SWEEP_R06); upstream audit T2; this study PEDERSEN_S160_REPRODUCTION | **PRODUCTION** (event 2, 2026-09-13) |
| **`duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered`** | `duOlhoffFixedPenaltySensitivityFiltered` | M4, TMA, B0, REG160, `duOlhoffFrozenM4`, A1_frozen160 | SIMP ρ³ at every density | eq. (4b) C¹ cut-off below 0.1, r = 6 | global ladder [0.04 0.02 0.01 0.005], β-stall (window 10, tol 5e-3) | ‖Δρ‖₂ < ε only on a settled move (`designChange`, settledMove) | historical OlhoffCurrent formulation; reproducibility of all earlier evidence | frozen conference realization; campaign_9mesh_r2; anchor A1 | **HISTORICAL_FORMULATION** — production 2026-09-07 … 2026-09-13; eligible, not selected |
| **`duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered`** | — | TR3_C, TR3_C_320x40, CAN3_480x60, CAN3_800x100, EX3_160 | SIMP ρ³ at every density | eq. (4b) | global ladder [0.04 0.02 0.01], one rung per stage-exhaustion declaration E = A ∨ B (W = P = 20, Wnp = 10) | terminal E declaration at the last rung (`stageExhaustion`) | historical diagnostic controller of the C160/C320/C480/C800 studies | three_rung_promotion_validation_retry1 (C320), three_rung_canary_preflight (C480/C800), upstream audit EX3_160 | **HISTORICAL_DIAGNOSTIC** — never promoted; **not production-eligible** |

Shared by all three and unchanged by this migration: p = 3 fixed; Sigmund sensitivity filter on every f_sk at fixed physical R = 0.06; fixed subspace N = 2 with diagonal offsets and off-diagonals; published MMA on the increment (tol 0.05, 5…500, state reset per outer); ρ_min = 1e-3; maxCluster = 4; ρ₀ = 0.5; V = 0.5. `test_preset_identity` asserts this field by field.

## Naming

Each name carries the axes that separate the formulations: material law, controller, filter.

- **SimpEq4b vs Pedersen.** This is the scientific split the delta audit measured: low-density stiffness and mass law.
- **BetaStallLadder vs ThreeRungStageExhaustion vs AdaptiveBox.** The outer controller and its stop.
- **SensitivityFiltered.** Sigmund sensitivity filter; common to all three.

I did not use "FixedPenalty" as the distinguishing axis because all three presets are fixed-penalty. That is why the old name `duOlhoffFixedPenaltySensitivityFiltered` became ambiguous once a second formulation existed. It survives only as a compatibility alias of the formulation it always named.

I also did not name the historical β-stall preset after stage exhaustion, even though the brief suggested `duOlhoffEq4bStageExhaustion`. Its resolved configuration has `move.continuation.signal = boundVariable` and `stop.rule = designChange`, so that name would have been misleading. The stage-exhaustion controller is a separate historical preset with its own name.

## Alias rules (enforced, `test_preset_identity`)

1. The only compatibility alias is `duOlhoffFixedPenaltySensitivityFiltered`. It resolves to the β-stall preset with a configuration hash identical to the canonical name's, and to nothing else.
2. Provenance aliases are refused with `olhoffcurrent_preset:ProvenanceAlias`, which names the owning preset. Tested with M4, S160x20, `duOlhoffFrozenM4`, `duOlhoffAdaptivePedersen` and TR3_C.
3. No alias is shared between presets, and no alias equals a canonical name.
4. Unnamed lookup, configuration, run and caveat are all refused (`…:NameRequired` / `…:PresetRequired`).

**No alias silently maps one scientific formulation to the other.**

## `olhoffcurrent_preset.m` decision (Part 7)

`olhoffcurrent_preset()` was the convenience production entry. It is **neither kept on the historical method nor repointed.** It now requires a name. Production is read explicitly with `olhoffcurrent_production_preset()`, whose answer comes from a recorded provenance event (option 2 of the brief, with the event recorded).

This removes the semantic overloading of an unnamed call. The historical preset stays directly accessible by its canonical name and by its compatibility alias.

## Distinct identity evidence

- Resolved configurations differ in `material.stiffness.model` (simp / pedersen), `material.mass.model` (eq4b / eq2) and `move.policy` (ladder / adaptive). The 87-row hashes also differ: 160×20 `c0fe56ce…` vs `b1a5744d…`.
- `olh.config.describe` names Pedersen only for the Pedersen preset.
- `olhoffcurrent_caveat` differs per preset. The Pedersen caveat states that it is a DISTINCT formulation and "not a bug fix".
- The harness assertion sets are formulation-specific: the Pedersen set fails 5 checks on the β-stall configuration, and the β-stall set fails 4 on the Pedersen configuration (`confbench_selftest` T13).
