# Globalization/Stabilization Experiment Report

Generated: `2026-07-08 12:27:12`

## Scope

CC 40x5 Du & Olhoff benchmark. FE, interpolation, sensitivities, filters, boundary conditions, and generalized gradients were not changed. Only outer-update acceptance/globalization and post-coalescence optimizer move limits were varied.

Base numerical layer: `move_lim=0.2`, `outer_move=0.2`, `alpha=0.5`, `inner_max_iter=30`, low-mode MAC threshold `0.25`, `alpha_min=0.0078125`.

Coalesced basin: `omega1 >= 450.0` and `abs(omega2-omega1)/omega1 <= 0.005`.
Paper-like guard: `abs(omega1 - 456.4)/456.4 <= 0.02` and `abs(omega2-omega1)/omega1 <= 0.005`. Paper reproduction is not claimed unless this stricter guard holds for `20` consecutive outer iterations.

## Summary

| variant | basin entry | basin exit | max basin streak | paper-like streak | final omega1 | final omega2 | final N | rejected outer | rejected trials | alpha min/median | retained paper-like? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `A_baseline` | 24 | 27 | 3 | 1 | 153.6920 | 309.3188 | 1 | 0 | 0 | 0.5 / 0.5 | no |
| `B_monotone_cluster` | 15 | 24 | 9 | 0 | 1330.2740 | 1345.1000 | 1 | 70 | 614 | 0 / 0 | no |
| `C_low_mode_guard` | NA | NA | 0 | 0 | 204.2731 | 398.5410 | 1 | 0 | 0 | 1 / 1 | no |
| `D_trust_0p5` | 24 | 25 | 1 | 0 | 256.3804 | 361.2148 | 1 | 0 | 0 | 0.5 / 0.5 | no |
| `D_trust_0p25` | 24 | 37 | 13 | 1 | 390.2537 | 450.9455 | 1 | 0 | 0 | 0.5 / 0.5 | no |
| `D_trust_0p1` | 25 | NA | 76 | 2 | 488.9747 | 489.3673 | 2 | 0 | 0 | 0.5 / 0.5 | no |
| `E_combined_0p5` | 16 | 17 | 2 | 0 | 994.1633 | 999.4843 | 1 | 74 | 639 | 0 / 0 | no |
| `E_combined_0p25` | 17 | NA | 84 | 0 | 687.3682 | 687.4501 | 2 | 75 | 641 | 0 / 0 | no |
| `E_combined_0p1` | 20 | 36 | 16 | 0 | 691.9462 | 695.9430 | 1 | 64 | 550 | 0 / 0 | no |

## Topology Outputs

- `A_baseline`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/A_baseline/A_baseline_topology.png`
- `B_monotone_cluster`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/B_monotone_cluster/B_monotone_cluster_topology.png`
- `C_low_mode_guard`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/C_low_mode_guard/C_low_mode_guard_topology.png`
- `D_trust_0p5`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/D_trust_0p5/D_trust_0p5_topology.png`
- `D_trust_0p25`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/D_trust_0p25/D_trust_0p25_topology.png`
- `D_trust_0p1`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/D_trust_0p1/D_trust_0p1_topology.png`
- `E_combined_0p5`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/E_combined_0p5/E_combined_0p5_topology.png`
- `E_combined_0p25`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/E_combined_0p25/E_combined_0p25_topology.png`
- `E_combined_0p1`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/globalization_stabilization_results/E_combined_0p1/E_combined_0p1_topology.png`

## Interpretation Guard

No paper-reproduction claim is allowed: no variant stayed inside the strict paper-like guard for `20` consecutive outer iterations.

## Evidence Files

- `globalization_stabilization_results/globalization_summary.csv`
- `globalization_stabilization_results/<variant>/<variant>_iterations.csv`
- `globalization_stabilization_results/<variant>/<variant>_rho_final.csv`
- `globalization_stabilization_results/<variant>/<variant>_topology.png`
- `globalization_stabilization_results/<variant>/<variant>_result.mat`
