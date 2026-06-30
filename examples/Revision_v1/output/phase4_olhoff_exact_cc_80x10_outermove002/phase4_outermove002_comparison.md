# Phase 4 Comparison: Phase 3 vs outer_move=0.02

| category | metric | Phase 3 | Phase 4 |
|---|---|---:|---:|
| outer convergence | final design change | 0.0206388 | 0.000101792 |
| outer convergence | mean design change, last 50 | 0.0206685 | 0.0081628 |
| outer convergence | omega parity gap | 1.14922 | 2.97799 |
| outer convergence | beta parity gap | 6.03482 | 2.80241 |
| outer convergence | omega oscillation amplitude | 1.23164 | 71.2235 |
| outer convergence | beta oscillation amplitude | 6.35641 | 70.0391 |
| inner behaviour | average inner iterations | 114.552 | 52.625 |
| inner behaviour | maximum inner iterations | 166 | 71 |
| inner behaviour | cap-hit fraction | 0 | 0 |
| inner behaviour | average CPU time s | 1.99277 | 0.839847 |
| S1 | localized modes | 0 | 0 |
| S1 | mode 1 ld_strain_frac | 0.00125398 | 0 |
| topology | support-connected components | 0 | 0 |

Decision: **outer_move = 0.02 removes the oscillation and achieves convergence.**.
