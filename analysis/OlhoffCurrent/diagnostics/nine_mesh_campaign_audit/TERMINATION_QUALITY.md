# Termination quality

**TERMINATION_CROSS_MESH_NOT_CREDIBLE** as a scientific maturity claim for the actual legacy campaign. The requested vocabulary has no inconclusive termination code; this verdict concerns its affirmative convergence claims, not the unexecuted three-rung policy.

All nine satisfy their actual programmed design-change test. None is a CAP_HIT disguised as convergence. But all stop exactly one iteration after a move halving, and their final stage contains only two updates. That is strong evidence that the one-step settledMove guard still measures schedule-induced contraction rather than demonstrating a mature trajectory. It does not alone prove what later evolution would be on each finer mesh.

| Mesh | max|drho| | L2 | RMS | L2/tol | Move | Mnd % | Relative gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160x20 | 0.00581044 | 0.0235826 | 0.000416885 | 0.471651 | 0.01 | 13.4025 | 0.0145418 |
| 240x30 | 0.00364849 | 0.0429341 | 0.000505983 | 0.572454 | 0.01 | 15.6036 | 0.161837 |
| 320x40 | 0.00818858 | 0.0900805 | 0.000796207 | 0.900805 | 0.02 | 23.3596 | 0.107375 |
| 400x50 | 0.00666917 | 0.0936602 | 0.000662278 | 0.749282 | 0.02 | 32.3283 | 0.0773963 |
| 480x60 | 0.00893536 | 0.0968165 | 0.000570497 | 0.645443 | 0.02 | 34.6717 | 0.0696574 |
| 560x70 | 0.0129207 | 0.110601 | 0.000558621 | 0.632007 | 0.02 | 37.014 | 0.0289662 |
| 640x80 | 0.00632049 | 0.0805639 | 0.000356045 | 0.402819 | 0.02 | 39.7204 | 0.0089898 |
| 720x90 | 0.00614536 | 0.0968426 | 0.000380433 | 0.430411 | 0.02 | 41.2418 | 0.00922841 |
| 800x100 | 0.00435752 | 0.0647593 | 0.000228959 | 0.259037 | 0.02 | 50.6561 | 0.00600634 |


Final 20/50/100 changes in omega1 and Mnd, directional coherence, cancellation, stagewise inner distributions, and multiplicity trajectories are unavailable for **all nine actual campaign runs**. Master cells are blank. No estimate substitutes for absent history. Terminal small RMS updates coexist with 13–51% grayness and large thresholded topology changes. Small step size alone does not distinguish a poor stationary point, prematurely suppressed motion, or a nearly mature design.

Historical legacy CSVs match campaign endpoint outer/inner counts, terminal density norm and Mnd at 160/320/400; they are **supporting historical analogues**, not restored campaign histories. Their last-20 pre-update omega changes are +0.105%, +0.466%, +0.471%; Mnd changes are −0.709, −2.782, −1.862 percentage points. Last-50 changes are +6.659%, +3.816%, +3.549% and −26.073, −13.274, −9.785 Mnd points. All these windows cross move transitions; they are not steady-stage estimates. See [HISTORICAL_LEGACY_WINDOWS.csv](HISTORICAL_LEGACY_WINDOWS.csv). Matched final numbers do not prove bitwise history identity across runs.

For historical E-controller S3 designs, stored density paths do support low residual evolution. Final 20-step net density L1 is 0.000458/0.000243/0.000156/0.000155 for 160/240/320/400. Their Mnd changes are −0.0203/−0.00156/+0.00808/−0.0141 points. The 50/100 windows cross stages because S3 lasts 39 updates. Full measured values and frequency conventions are in [HISTORICAL_TERMINAL_WINDOWS.csv](HISTORICAL_TERMINAL_WINDOWS.csv). The subsequent 0.005 continuation barely changes those designs, strengthening maturity evidence at these four meshes only.

An indexing issue in earlier summaries matters for honest comparison: `hist.omega(:,k)` belongs to the **pre-update** state rho_(k−1), while RHO(:,k) and Mnd are post-update. This audit obtains exact post-update S3 frequencies from the next stored modal evaluation `hist.omega(:,S3+1)`, before any new design update; fixed p, mass and filter make this valid. C320 is 166.4263044135, matching the separate validated candidate record, rather than the older S3 table's 166.4272692776. The difference is tiny but conventions are not silently mixed.

No available nine-mesh evidence proves inner KKT stationarity; “inner converged” means the configured relative-step criterion succeeded. Reported inner failure count is zero, but per-outer diagnostics were discarded.
