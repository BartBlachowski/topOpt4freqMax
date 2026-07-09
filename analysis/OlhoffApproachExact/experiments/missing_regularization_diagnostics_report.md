# Missing Regularization / Benchmark-Assumption Diagnostics

Generated: `2026-07-09 15:46:38`

Success guard: support-connected topology, `omega1 = 456.4 +/- 2.0%`, `gap12 <= 0.005`, `N >= 2`, and support-connected component carrying at least 50% mean kinetic/strain energy in modes 1 and 2.

## Summary

| variant | omega1 | omega2 | gap | N | support | mode1/2 support energy | components | isolated frac | success | classification |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| `baseline_sensitivity_full_edges` | 489.188 | 489.198 | 2.037e-05 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `density_filter_r2p5` | 972.438 | 972.559 | 0.0001235 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `density_filter_r5p0` | 730.409 | 730.423 | 1.837e-05 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `density_filter_r7p5` | 547.297 | 547.329 | 5.945e-05 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `projection_cont_r2p5_sym_both` | 2551.931 | 2551.931 | 1.049e-09 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `projection_cont_r5p0_sym_both` | 2574.063 | 2574.063 | 3.212e-09 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `projection_cont_r7p5_sym_both` | 1959.640 | 1959.640 | 2.048e-07 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `support_path_sensitivity_r2p5` | 243.411 | 395.542 | 0.625 | 1 | yes | 0.857 / 0.778 | 1 | 0.000 | no | unsuccessful |
| `support_path055_sensitivity_r2p5` | 285.250 | 400.549 | 0.4042 | 1 | yes | 0.800 / 0.772 | 1 | 0.000 | no | unsuccessful |
| `support_path055_sensitivity_r3p5` | 277.685 | 361.553 | 0.302 | 1 | yes | 0.762 / 0.719 | 1 | 0.000 | no | unsuccessful |
| `support_path055_sensitivity_alpha1` | 277.512 | 405.668 | 0.4618 | 1 | yes | 0.822 / 0.824 | 1 | 0.000 | no | unsuccessful |
| `support_path055_sensitivity_move0p5` | 222.856 | 302.027 | 0.3553 | 1 | yes | 0.663 / 0.674 | 1 | 0.000 | no | unsuccessful |
| `support_path055_sensitivity_alpha1_move0p5` | 129.690 | 213.546 | 0.6466 | 1 | yes | 0.959 / 0.947 | 1 | 0.000 | no | unsuccessful |
| `support_path055_density_r7p5` | 245.737 | 352.127 | 0.4329 | 1 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `support_path_density_r5p0` | 237.998 | 404.250 | 0.6985 | 1 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `support_path_projection_cont_r5p0` | 672.571 | 674.773 | 0.003274 | 1 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `mass_lumped_sensitivity` | 563.119 | 563.507 | 0.0006888 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `mass_lumped_density_r5p0` | 720.511 | 720.514 | 3.265e-06 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `support_corner_clamps` | 373.923 | 373.969 | 0.0001227 | 2 | no | 0.000 / 0.000 | 2 | 1.000 | no | unsuccessful |
| `support_midheight_clamps` | 143.202 | 192.544 | 0.3446 | 1 | yes | 0.951 / 0.892 | 1 | 0.000 | no | unsuccessful |

## Findings

- No tested diagnostic variant recovered a support-connected bimodal CC topology near `omega = 456.4` under the success guard.
- Variants with passive support-path material are connectivity diagnostics only and are not paper-faithful constraints.
- Density filtering, Heaviside projection, lumped mass, and altered support placement are treated as extra or ambiguous benchmark assumptions, not confirmed Du-Olhoff 2007 reproduction details.

## Answer

Within this controlled matrix, the missing ingredient was not identified as a simple density filter, projection continuation, larger filter radius, lumped mass, tested support-placement alternative, or minimum support-path heuristic. The variants split into two failure modes: unconstrained filter/projection/mass/support alternatives remain coalesced but disconnected, while explicit support-path or weak-support variants can be support-connected structural designs but lose the bimodal near-456 target.

## Evidence Files

- `missing_regularization_diagnostics_results/missing_regularization_diagnostics_summary.csv`
- `missing_regularization_diagnostics_results/<variant>/<variant>_components.csv`
- `missing_regularization_diagnostics_results/<variant>/<variant>_mode_component_energy.csv`
- `missing_regularization_diagnostics_results/<variant>/<variant>_topology.png`
- `missing_regularization_diagnostics_results/<variant>/<variant>_audit.mat`
