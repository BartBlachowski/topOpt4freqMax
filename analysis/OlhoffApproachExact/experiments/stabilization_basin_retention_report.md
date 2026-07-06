# Stabilization Basin-Retention Report

Generated: `2026-07-06 09:17:42 +0000`

## Scope

Targeted Du & Olhoff 2007 clamped-clamped benchmark test. The FE formulation, mass interpolation, sensitivities, boundary conditions, and objective are unchanged. The controlled numerical-layer settings are MMA state persistence and inner MMA iteration budget only.

Base benchmark: `40x5`, `volfrac=0.5`, `mass_mode=du2007_c1`, `rmin_elem=2.5`, `mult_tol=1.0e-03`, `alpha=0.5`, `move_lim=0.2`, `outer_move=0.2`.

Near-paper basin definition: `omega_1 >= 450.0` and `abs(omega_2 - omega_1)/omega_1 <= 0.005`. Retention requires `10` consecutive outer iterations. Published CC target: `omega_1 = 456.4`, bimodal optimum.

## Variant Comparison

| Variant | Persistent MMA | Inner max | Outer iters | Inner converged | Inner cap hits | Basin entry | Basin exit | Retained 10? | Any N=2 freq tol | Any N=2 lambda tol | Best omega1 | Best gap f | Final omega1 | Final gap f | Final N | Final support-connected | Paper reproduction claim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `baseline` | 0 | 30 | 120 | 0/120 | 120 | 24 | 27 | 0 | 1 | 1 | 574.1635 | 0.01943 | 81.4882 | 1.546 | 1 | no | not allowed |
| `persistent_mma` | 1 | 30 | 120 | 0/120 | 120 | NA | NA | 0 | 0 | 0 | 488.5042 | 0.1126 | 138.6645 | 0.1118 | 1 | no | not allowed |
| `converged_inner` | 0 | 300 | 120 | 98/120 | 22 | 14 | 15 | 0 | 1 | 1 | 550.9243 | 0.04304 | 277.8998 | 0.2251 | 1 | no | not allowed |
| `combined` | 1 | 300 | 120 | 103/120 | 17 | 15 | 16 | 0 | 1 | 1 | 534.8715 | 0.0003717 | 228.6516 | 0.4759 | 1 | no | not allowed |

## Basin Entry And Exit

| Variant | Entry iter | Exit iter | Retain window | Basin iter count | N=2 freq first | N=2 lambda first |
|---|---:|---:|---|---:|---:|---:|
| `baseline` | 24 | 27 | none | 5 | 25 | 25 |
| `persistent_mma` | NA | NA | none | 0 | NA | NA |
| `converged_inner` | 14 | 15 | none | 1 | 13 | 13 |
| `combined` | 15 | 16 | none | 1 | 13 | 14 |

## Best Snapshot Vs Final Snapshot

| Variant | Best iter | Best omega1 | Best omega2 | Best freq gap | Best lambda gap | Best N | Best support-connected | Final iter | Final omega1 | Final omega2 | Final freq gap | Final lambda gap | Final N | Final support-connected |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 33 | 574.1635 | 585.3197 | 0.01943 | 0.03924 | 1 | no | 120 | 81.4882 | 207.4819 | 1.546 | 5.483 | 1 | no |
| `persistent_mma` | 99 | 488.5042 | 543.5251 | 0.1126 | 0.2379 | 1 | no | 120 | 138.6645 | 154.1677 | 0.1118 | 0.2361 | 1 | no |
| `converged_inner` | 15 | 550.9243 | 574.6361 | 0.04304 | 0.08793 | 1 | no | 120 | 277.8998 | 340.4554 | 0.2251 | 0.5009 | 1 | no |
| `combined` | 15 | 534.8715 | 535.0703 | 0.0003717 | 0.0007435 | 2 | no | 120 | 228.6516 | 337.4563 | 0.4759 | 1.178 | 1 | no |

## Transition Notes

- `baseline`: enters at iteration 24 but exits at iteration 27; inspect that transition.
- `persistent_mma`: never enters the near-paper basin.
- `converged_inner`: enters at iteration 14 but exits at iteration 15; inspect that transition.
- `combined`: enters at iteration 15 but exits at iteration 16; inspect that transition.

## Transition Detail

| Variant | Entry iter | Entry omega1 | Entry omega2 | Entry freq gap | Exit iter | Exit omega1 | Exit omega2 | Exit freq gap | Exit reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 24 | 462.4208 | 463.1639 | 0.001607 | 27 | 11.7026 | 20.7156 | 0.7702 | omega1 11.7026 < 450.0; gap 0.7702 > 0.005 |
| `persistent_mma` | NA | NA | NA | NA | NA | NA | NA | NA | no basin entry |
| `converged_inner` | 14 | 495.4878 | 495.5053 | 3.529e-05 | 15 | 550.9243 | 574.6361 | 0.04304 | gap 0.04304 > 0.005 |
| `combined` | 15 | 534.8715 | 535.0703 | 0.0003717 | 16 | 5.3110 | 8.6705 | 0.6326 | omega1 5.3110 < 450.0; gap 0.6326 > 0.005 |

## Attribution

At least one variant enters the near-paper basin but no variant retains it for 10 consecutive iterations. Under the requested acceptance logic, this remains unresolved path instability; the transition-detail table identifies the exit iteration and failed basin condition for each entry.

## Evidence Files

- `baseline`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/stabilization_basin_retention_results/baseline`
- `persistent_mma`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/stabilization_basin_retention_results/persistent_mma`
- `converged_inner`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/stabilization_basin_retention_results/converged_inner`
- `combined`: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/stabilization_basin_retention_results/combined`

Each variant directory contains the MAT result, per-iteration CSV, full rho-by-iteration CSV, and topology PNGs for the initial, first basin-entry when present, best feasible, and final designs.

## Reproduction Guard

No paper-reproduction claim is allowed unless a final or retained design is support-connected, bimodal under the declared tolerance, and within 2.0% of `omega_1 = 456.4`. The table above reports this as a separate guard rather than assuming it from basin entry alone.
