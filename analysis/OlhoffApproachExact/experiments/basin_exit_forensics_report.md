# Basin Exit Forensics Report

Generated: `2026-07-08 12:02:34`

## Scope

Diagnostic-only run of `topopt_freq_exact` for the Du & Olhoff CC beam. FE, interpolation, sensitivities, filters, and update logic were not changed. Forensic tracing started after the first post-update state with `abs(omega2-omega1)/omega1 < 0.005`.

Config: `nelx=40`, `nely=5`, `volfrac=0.5`, `mass_mode=du2007_c1`, `rmin_elem=2.5`, `mult_tol=0.001`, `move_lim=0.2`, `outer_move=0.2`, `alpha=0.5`, `inner_max_iter=30`.

## Trigger

First trigger: outer iteration `23`, post-update `omega1=431.514`, `omega2=432.348`, relative gap `0.00193225`, volume `0.495942`.

## Per-Iteration Evidence

| iter | N pre | omega pre | beta omega | predicted cluster omega | omega after rho+Delta | accepted omega | drho norm/max | active constraints | MAC diag min | pred improve + real decrease |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 24 | 1 | 431.5143 / 432.3481 | 532.9743 | 533.0384 / 0.0000 | 65.0351 / 66.0722 | 462.4208 / 463.1639 | 0.1073 / 0.2 | none | 9.598e-14 | 1 |
| 25 | 1 | 462.4208 / 463.1639 | 536.4688 | 536.5388 / 0.0000 | 466.0991 / 475.2448 | 503.5452 / 503.7968 | 0.08213 / 0.2 | none | 0.01417 | 0 |
| 26 | 2 | 503.5452 / 503.7968 | 607.8956 | 607.9004 / 607.9295 | 587.9970 / 592.5975 | 552.0413 / 552.5253 | 0.1055 / 0.2 | cluster_1, volume | 0.0002637 | 0 |
| 27 | 2 | 552.0413 / 552.5253 | 624.5580 | 624.5586 / 625.2760 | 0.5388 / 1.0372 | 11.7026 / 20.7156 | 0.1071 / 0.2 | cluster_1 | 4.03e-14 | 1 |
| 28 | 1 | 11.7026 / 20.7156 | 124.6586 | 124.6586 / 0.0000 | 177.0288 / 232.0792 | 212.3854 / 230.1112 | 0.159 / 0.2 | cluster_1 | 4.576e-13 | 0 |
| 29 | 1 | 212.3854 / 230.1112 | 398.3265 | 398.3287 / 0.0000 | 32.0939 / 56.0703 | 323.9922 / 375.2889 | 0.1572 / 0.2 | cluster_1 | 4.928e-09 | 1 |
| 30 | 1 | 323.9922 / 375.2889 | 472.3291 | 472.4257 / 0.0000 | 528.4532 / 574.2046 | 407.9111 / 419.7862 | 0.1349 / 0.2 | none | 2.823e-06 | 0 |
| 31 | 1 | 407.9111 / 419.7862 | 484.4781 | 484.6268 / 0.0000 | 405.2858 / 430.1299 | 438.8073 / 451.9054 | 0.09293 / 0.2 | none | 0.5828 | 1 |
| 32 | 1 | 438.8073 / 451.9054 | 546.0948 | 546.1397 / 0.0000 | 589.5558 / 601.0631 | 503.3446 / 505.4770 | 0.1074 / 0.2 | J_mode, volume | 9.041e-07 | 0 |
| 33 | 1 | 503.3446 / 505.4770 | 584.9201 | 584.9635 / 0.0000 | 594.4142 / 655.7294 | 574.1635 / 585.3197 | 0.09515 / 0.2 | none | 5.375e-12 | 0 |
| 34 | 1 | 574.1635 / 585.3197 | 699.4705 | 699.4712 / 0.0000 | 0.7575 / 2.9935 | 11.1927 / 29.7170 | 0.08595 / 0.2 | cluster_1 | 1.931e-15 | 1 |
| 35 | 1 | 11.1927 / 29.7170 | 91.7191 | 91.7191 / 0.0000 | 251.2090 / 255.0265 | 231.5692 / 290.7379 | 0.1461 / 0.2 | cluster_1 | 1.893e-13 | 0 |
| 36 | 1 | 231.5692 / 290.7379 | 445.7866 | 446.3904 / 0.0000 | 291.0762 / 386.1880 | 362.9686 / 380.4461 | 0.1465 / 0.2 | none | 0.0003202 | 0 |
| 37 | 1 | 362.9686 / 380.4461 | 492.1338 | 492.3673 / 0.0000 | 475.4797 / 567.9331 | 435.2121 / 443.8980 | 0.1051 / 0.2 | none | 6.495e-16 | 0 |
| 38 | 1 | 435.2121 / 443.8980 | 513.1204 | 513.1211 / 0.0000 | 1.9707 / 4.0832 | 328.6798 / 499.6873 | 0.101 / 0.2 | cluster_1, volume | 2.357e-12 | 1 |
| 39 | 1 | 328.6798 / 499.6873 | 1304.3489 | 1304.3493 / 0.0000 | 1.7961 / 18.5402 | 84.9009 / 249.6830 | 0.1436 / 0.2 | cluster_1 | 0.0008659 | 1 |
| 40 | 1 | 84.9009 / 249.6830 | 224.7721 | 224.7721 / 0.0000 | 2.9105 / 7.1614 | 179.7385 / 322.4455 | 0.1619 / 0.2 | cluster_1 | 7.874e-10 | 1 |

## Findings

A) MMA linearization: yes. The first predicted-improvement/real-decrease step is iteration `24` for the full proposed step, and the accepted basin exit is iteration `27`. At the accepted exit, beta predicts `omega=624.558`, the cluster model predicts `omega=624.559 / 625.276`, the accepted update gives `omega=11.7026 / 20.7156`, and the full `rho+Delta rho` proposal gives `omega=0.538758 / 1.03718`.

B) Multiple-eigenvalue constraint: not inactive at the decisive exit. At iteration `27`, `N_pre=2` and active model constraints were `cluster_1`. The multiple constraint was present, but its local model was not protective for the accepted/proposed density change.

C) Mode tracking/multiplicity: not the primary failure. On the decisive step, the maximum MAC between any pre-update mode and the two collapsed post-update modes is `0.246031`, so the low modes are newly introduced by the density step rather than a simple swap of the coalesced pair. The previous `N=2` step may show expected mode mixing inside the nearly multiple subspace; the exit itself is not resolved by renumbering modes.

D) Step size: yes. At the accepted exit, the proposed step has `drho_norm=0.10715` and `drho_max=0.199998` at the imposed cap; the accepted half-step still has `drho_norm=0.0535748` and `drho_max=0.0999992`. The accepted update leaves the basin at `omega1=11.7026`, while the full proposed update collapses further to `omega1=0.538758`. This is a step-length failure coupled to the bad local model.

## Evidence Files

- `basin_exit_forensics_results/basin_exit_forensics_result.mat`: full rho, Delta rho, constraints, and MAC matrices.
- `basin_exit_forensics_results/basin_exit_forensics_table.csv`: compact scalar evidence.
- `basin_exit_forensics_results/basin_exit_rho_drho_vectors.csv`: density and proposed increment vectors by recorded iteration.
- `basin_exit_forensics_results/mac_*_iter_*.csv`: mode MAC matrices.
