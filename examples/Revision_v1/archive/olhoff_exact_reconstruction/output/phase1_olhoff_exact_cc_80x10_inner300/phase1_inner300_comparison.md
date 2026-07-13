# Phase 1 Comparison: inner_max_iter 30 vs 300

| category | metric | inner=30 alpha=0.5 | inner=300 alpha=0.5 |
|---|---|---:|---:|
| inner solver | average iterations | 30 | 155.315 |
| inner solver | max iterations | 30 | 300 |
| inner solver | cap fraction | 1 | 0.015 |
| outer convergence | omega parity gap | 6.96163 | 3.70072 |
| outer convergence | beta parity gap | 7.5174 | 15.6703 |
| outer convergence | final design change | 0.0763754 | 0.0914339 |
| S1 | localized modes | 0 | 0 |
| S1 | mode 1 ld_strain_frac | 0.00292671 | 0.00165011 |
| topology | support-connected components | 0 | 0 |

Decision: **Inner solver still reaches the cap.**.
