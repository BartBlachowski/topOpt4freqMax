# Phase 3 Comparison: Phase 2 vs outer_move=0.05

| category | metric | Phase 2 | Phase 3 |
|---|---|---:|---:|
| outer convergence | final design change | 0.0921098 | 0.0206388 |
| outer convergence | mean design change, last 50 | 0.0915786 | 0.0206685 |
| outer convergence | omega parity gap | 1.63382 | 1.14922 |
| outer convergence | beta parity gap | 18.4989 | 6.03482 |
| outer convergence | omega oscillation amplitude | 3.92221 | 1.23164 |
| outer convergence | beta oscillation amplitude | 25.2246 | 6.35641 |
| inner behaviour | average inner iterations | 162.308 | 114.552 |
| inner behaviour | cap-hit fraction | 0.01 | 0 |
| inner behaviour | average CPU time s | 1.12935 | 1.99277 |
| S1 | localized modes | 0 | 0 |
| S1 | mode 1 ld_strain_frac | 0.00119814 | 0.00125398 |
| topology | support-connected components | 0 | 0 |

Decision: **Smaller outer_move substantially reduces the 2-cycle.**.
