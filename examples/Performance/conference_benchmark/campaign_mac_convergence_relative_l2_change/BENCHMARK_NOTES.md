# Conference performance benchmark -- notes

Generated 2026-10-09T06:01:59+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_convergence_relative_l2_change`
- scientific evidence: **true**
- performance campaign: **true**
- resolutions: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100
- threads: 1
- timing schema: `conference_benchmark_timing/2`

## How to read the table

Count/time columns represent method-native computational stages and are not mathematically identical across methods. Total wall time is the common performance quantity.

Proposed: Count 1 = reference eigenanalysis solves (always 1, not an optimization iteration), Count 2 = SIMP iterations, Time 1 = that single eigenanalysis (K0/M0 assembly and the eigensolve, nothing else; solver preparation is in Other), Time 2 = SIMP. Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 iteration counts, Time 1 and Time 2 the corresponding stage times. Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box): Count 1 = outer iterations, Count 2 = cumulative nested MMA iterations, Time 1 = outer work excluding the nested MMA solve (FE assembly, the eigenproblem, sensitivities, filtering, the design update), Time 2 = nested MMA total. The two counts are never added.

Stage time [s] = Time 1 + Time 2, computed before rounding. Other [s] is overhead_time_s: everything inside the timed solve but outside the two named stages, including setup, final modal analysis and result assembly. Other includes computations too. Total wall time [s] = Stage time + Other. Method-specific configuration dispatch is included only where it lies inside the recorded timer; Du-Olhoff configuration resolution precedes its timer.

omega_1 native is the first eigenfrequency of the converged design under the solver's own material model, which differs per method (Proposed: SIMP p = 3, E_min = 1e-9 E_0, linear mass, rho_min = 1e-9, design floor 0; Yuksel: SIMP p = 3, E_min = 1e-9 E_0, mass linear above rho = 0.1 and proportional to the sixth power of rho below it; Du-Olhoff: Pedersen stiffness, linear mass). omega_1 E1 is the same design under the frozen common evaluator E1 (SIMP p = 3, E_min = 1e-6 E_0, linear mass), classifier-selected structural mode, computed outside every timer; it is the only omega_1 that is comparable across methods. A dagger marks a native value that deviates from E1 by more than 5%: the native model then returns a localized mode of near-void elements, not the structural frequency.

### Proposed Stage 1: eigenanalysis versus preparation

Proposed Stage 1 preparation (initialization minus the reference eigenanalysis) is one-off setup and is reported in Other, never in Time 1. It is a fixed cost that does not scale with the mesh; the first measured row of a session also carries that session's first-call costs.

| Mesh | Initialization incl. eigenanalysis [s] | Time 1 = reference eigenanalysis [s] | Preparation [s] (in Other) | Other [s] | Total [s] |
|---|---|---|---|---|---|
| 160x20 | 0.132 | 0.031 | 0.101 | 0.279 | 0.725 |
| 240x30 | 0.157 | 0.059 | 0.097 | 0.386 | 1.409 |
| 320x40 | 0.212 | 0.108 | 0.104 | 0.569 | 2.369 |
| 400x50 | 0.294 | 0.187 | 0.107 | 0.858 | 3.559 |
| 480x60 | 0.371 | 0.256 | 0.115 | 0.800 | 6.710 |
| 560x70 | 0.476 | 0.358 | 0.119 | 0.963 | 10.497 |
| 640x80 | 0.613 | 0.479 | 0.133 | 1.781 | 12.262 |
| 720x90 | 1.090 | 0.948 | 0.141 | 2.547 | 28.450 |
| 800x100 | 1.427 | 1.273 | 0.154 | 3.343 | 44.644 |

The largest preparation value, 0.154 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.097 and 0.141 s, independent of the mesh.

### Native omega_1 values flagged

6 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 83.85, 83.85, 108.44 | 153.69 | 45.4% | 0.009 |
| Proposed | 240x30 | 86.63, 86.63, 106.92 | 157.09 | 44.9% | 0.009 |
| Proposed | 320x40 | 105.20, 105.20, 115.23 | 159.09 | 33.9% | 0.005 |
| Proposed | 400x50 | 110.81, 110.81, 117.45 | 160.38 | 30.9% | 0.004 |
| Proposed | 720x90 | 87.30, 87.51, 88.49 | 161.57 | 46.0% | 0.002 |
| Proposed | 800x100 | 140.13, 140.58, 141.79 | 161.63 | 13.3% | 0.002 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 122 | 220.541 | 1.8077 | 0.0320 | 0.0298 | 1.7754 | 19.50 | 0.0910 | NATIVE_CONVERGED |
| 240x30 | 114 | 359.889 | 3.1569 | 0.0652 | 0.0613 | 3.0911 | 18.59 | 0.1663 | NATIVE_CONVERGED |
| 320x40 | 128 | 567.261 | 4.4317 | 0.1128 | 0.1064 | 4.3180 | 19.20 | 0.2249 | NATIVE_CONVERGED |
| 400x50 | 105 | 658.779 | 6.2741 | 0.1836 | 0.1742 | 6.0886 | 20.22 | 0.3011 | NATIVE_CONVERGED |
| 480x60 | 126 | 1139.541 | 9.0440 | 0.2830 | 0.2691 | 8.7586 | 18.96 | 0.4619 | NATIVE_CONVERGED |
| 560x70 | 148 | 1704.639 | 11.5178 | 0.3882 | 0.3684 | 11.1266 | 18.21 | 0.6110 | NATIVE_CONVERGED |
| 640x80 | 170 | 2526.063 | 14.8592 | 0.5135 | 0.4882 | 14.3422 | 18.35 | 0.7815 | NATIVE_CONVERGED |
| 720x90 | 335 | 6573.957 | 19.6238 | 0.9163 | 0.8817 | 18.7043 | 17.85 | 1.0476 | NATIVE_CONVERGED |
| 800x100 | 256 | 5182.157 | 20.2428 | 1.1460 | 1.1025 | 19.0917 | 18.83 | 1.0140 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 2.172852e-05 | 1.2470 | 0.9550 | 9 |
| Yuksel | 9.856206e-07 | 1.6956 | 0.9758 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 3.309469e-02 | 1.0451 | 0.9178 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 3.004338e-03 | 0.7820 | 0.9922 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 3.481701e-06 | 1.1092 | 0.9862 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.023335e-06 | 1.1180 | 0.9860 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 3.231815e-03 | 0.7713 | 0.9924 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.428570e-04 | 0.7895 | 0.9869 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0130 | 0.0097 | 0.0119 | 0.0298 | 0.0910 |
| 240x30 | 14942 | 0.0268 | 0.0202 | 0.0263 | 0.0613 | 0.1663 |
| 320x40 | 26322 | 0.0483 | 0.0369 | 0.0470 | 0.1064 | 0.2249 |
| 400x50 | 40902 | 0.0785 | 0.0599 | 0.0751 | 0.1742 | 0.3011 |
| 480x60 | 58682 | 0.1131 | 0.0886 | 0.1075 | 0.2691 | 0.4619 |
| 560x70 | 79662 | 0.1610 | 0.1224 | 0.1495 | 0.3684 | 0.6110 |
| 640x80 | 103842 | 0.2128 | 0.1634 | 0.1992 | 0.4882 | 0.7815 |
| 720x90 | 131222 | 0.5545 | 0.2895 | 0.4068 | 0.8817 | 1.0476 |
| 800x100 | 161802 | 0.7148 | 0.3580 | 0.5053 | 1.1025 | 1.0140 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 32 | 0.031 | 0.416 | 0.446 | 0.279 | 0.725 | 83.8478 † | 153.6910 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 56 | 50 | 0.542 | 0.597 | 1.138 | 0.248 | 1.387 | 157.4248 | 157.2184 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 122 | 2379 | 3.901 | 216.598 | 220.499 | 0.042 | 220.541 | 169.2119 | 169.1985 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 36 | 0.059 | 0.964 | 1.023 | 0.386 | 1.409 | 86.6332 † | 157.0915 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 54 | 35 | 1.090 | 0.922 | 2.012 | 0.372 | 2.384 | 159.4154 | 159.2583 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 114 | 2119 | 7.429 | 352.387 | 359.816 | 0.073 | 359.889 | 167.3455 | 167.3368 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 35 | 0.108 | 1.691 | 1.800 | 0.569 | 2.369 | 105.2046 † | 159.0873 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 72 | 97 | 2.655 | 4.561 | 7.216 | 0.510 | 7.726 | 160.9092 | 160.8408 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 128 | 2458 | 14.434 | 552.704 | 567.138 | 0.123 | 567.261 | 165.8285 | 165.8192 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 32 | 0.187 | 2.513 | 2.701 | 0.858 | 3.559 | 110.8097 † | 160.3766 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 128 | 128 | 7.670 | 9.611 | 17.282 | 0.781 | 18.063 | 159.9431 | 159.8050 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 105 | 2123 | 19.277 | 639.302 | 658.579 | 0.199 | 658.779 | 166.4147 | 166.4054 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 50 | 0.256 | 5.654 | 5.910 | 0.800 | 6.710 | 160.2877 | 160.2878 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 136 | 170 | 12.055 | 18.279 | 30.334 | 1.136 | 31.470 | 160.2625 | 160.1911 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 126 | 2389 | 35.654 | 1103.579 | 1139.232 | 0.309 | 1139.541 | 165.9569 | 165.9472 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 57 | 0.358 | 9.175 | 9.533 | 0.963 | 10.497 | 160.7771 | 160.7772 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 195 | 187 | 23.862 | 27.950 | 51.811 | 1.340 | 53.151 | 160.1637 | 160.0986 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 148 | 2695 | 57.450 | 1646.730 | 1704.179 | 0.460 | 1704.639 | 165.7539 | 165.7436 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 47 | 0.479 | 10.001 | 10.480 | 1.781 | 12.262 | 161.0925 | 161.0929 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 180 | 225 | 29.408 | 44.810 | 74.218 | 1.759 | 75.977 | 160.5510 | 160.5098 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 170 | 3120 | 87.298 | 2438.175 | 2525.473 | 0.590 | 2526.063 | 165.6177 | 165.6070 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 45 | 0.948 | 24.954 | 25.902 | 2.547 | 28.450 | 87.3049 † | 161.5701 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 252 | 300 | 72.948 | 122.027 | 194.976 | 2.877 | 197.853 | 160.8630 | 160.8257 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 335 | 5981 | 306.976 | 6265.939 | 6572.915 | 1.042 | 6573.957 | 165.4679 | 165.4566 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 56 | 1.273 | 40.029 | 41.301 | 3.343 | 44.644 | 140.1271 † | 161.6303 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 251 | 344 | 89.858 | 173.816 | 263.674 | 3.555 | 267.229 | 161.0426 | 161.0230 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 256 | 4820 | 293.383 | 4887.473 | 5180.856 | 1.301 | 5182.157 | 165.4289 | 165.4159 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

