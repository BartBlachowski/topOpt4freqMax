# RESULTS_TABLE — nine-mesh Pedersen/adaptive production campaign

Nine rows, one per preregistered mesh, in run order; nothing is omitted. The source is `evidence/EXTRACT.json`: tapped solver results, cross-checked bit for bit against the runner records in `benchmark_records.mat`.

Frequencies are the **native** Pedersen/linear-mass eigenfrequencies of the final design. The gap is (ω₂−ω₁)/ω₁ of the final analysis. Times come from the runner's own timers: total = caller-side `tic/toc` around `olhoffSolve`; eig = assembly + `eigs` per outer iteration; inner = nested MMA per outer iteration.

| mesh | elements | DOFs | outer | inner | ω₁ | ω₂ | gap % | volume | M_nd | gray frac. | status | total wall [s] | wall/outer [s] | mean tOuter [s] | eig/outer [s] | ρ SHA-256 | config hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 3200 | 6762 (6758 free) | 121 | 2369 | 169.2106 | 170.4000 | 0.70 | 0.499999 | 0.1146 | 0.1419 | NATIVE_CONVERGED | 223.9 | 1.851 | 1.850 | 0.0324 | `2a1c0d0afa18e763…` | `b1a5744df798ad62…` |
| 240x30 | 7200 | 14942 (14938 free) | 111 | 2077 | 167.3424 | 187.1602 | 11.84 | 0.499999 | 0.1227 | 0.1447 | NATIVE_CONVERGED | 354.9 | 3.198 | 3.197 | 0.0626 | `bc8fa77962f9e45a…` | `2fac1384239527b1…` |
| 320x40 | 12800 | 26322 (26318 free) | 101 | 2001 | 165.8568 | 195.1400 | 17.66 | 0.500000 | 0.1406 | 0.1656 | NATIVE_CONVERGED | 427.2 | 4.230 | 4.228 | 0.1086 | `f246a9436ec9fd72…` | `a1203d43efb24030…` |
| 400x50 | 20000 | 40902 (40898 free) | 93 | 1913 | 166.4552 | 198.1318 | 19.03 | 0.500000 | 0.1216 | 0.1397 | NATIVE_CONVERGED | 570.1 | 6.130 | 6.128 | 0.1764 | `34c618a3cd085e26…` | `e40c6ba16c9addbb…` |
| 480x60 | 28800 | 58682 (58678 free) | 112 | 2137 | 166.0093 | 203.4412 | 22.55 | 0.500000 | 0.1307 | 0.1509 | NATIVE_CONVERGED | 981.4 | 8.762 | 8.759 | 0.2725 | `acf85ed42f2241b2…` | `8e55d18251152ec2…` |
| 560x70 | 39200 | 79662 (79658 free) | 130 | 2371 | 165.8101 | 206.3760 | 24.47 | 0.500000 | 0.1329 | 0.1534 | NATIVE_CONVERGED | 1425.0 | 10.962 | 10.958 | 0.3737 | `b69314dfdd7de5ea…` | `8b2533a3caf8432d…` |
| 640x80 | 51200 | 103842 (103838 free) | 156 | 2868 | 165.6500 | 205.9730 | 24.34 | 0.500000 | 0.1330 | 0.1529 | NATIVE_CONVERGED | 2215.8 | 14.204 | 14.200 | 0.4939 | `65eedc9596ef7c62…` | `cbe8dcf070d3e318…` |
| 720x90 | 64800 | 131222 (131218 free) | 204 | 3752 | 165.4234 | 202.4626 | 22.39 | 0.500000 | 0.1617 | 0.1863 | NATIVE_CONVERGED | 3537.5 | 17.341 | 17.336 | 0.8925 | `857c95da6a3e9b41…` | `2893ad47136fc9a7…` |
| 800x100 | 80000 | 161802 (161798 free) | 246 | 4650 | 165.4322 | 195.7393 | 18.32 | 0.500000 | 0.1645 | 0.1878 | NATIVE_CONVERGED | 4901.0 | 19.923 | 19.918 | 1.1143 | `b89af0554b020c24…` | `f9138743067afd0e…` |

The full hashes and the remaining columns are in `RESULTS_TABLE.csv`: ω₃, minimum gap in the history, volume error, final ‖Δρ‖₂, ε, final largest box, median tOuter, inner time per outer, and eigensolve count.
