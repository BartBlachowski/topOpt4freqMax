# Conference performance benchmark -- notes

Generated 2026-10-09T16:36:46+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_convergence_relative_l2_change_2`
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
| 160x20 | 0.124 | 0.029 | 0.096 | 0.240 | 0.692 |
| 240x30 | 0.171 | 0.061 | 0.110 | 0.360 | 1.376 |
| 320x40 | 0.270 | 0.109 | 0.161 | 0.595 | 2.418 |
| 400x50 | 0.291 | 0.182 | 0.109 | 0.820 | 3.526 |
| 480x60 | 0.372 | 0.253 | 0.119 | 0.774 | 6.718 |
| 560x70 | 0.479 | 0.353 | 0.126 | 0.929 | 10.553 |
| 640x80 | 0.611 | 0.477 | 0.134 | 1.732 | 12.291 |
| 720x90 | 1.086 | 0.944 | 0.142 | 2.537 | 28.584 |
| 800x100 | 1.415 | 1.264 | 0.151 | 3.341 | 44.802 |

The largest preparation value, 0.161 s at 320x40, is the first measured row of its session; the other 8 rows lie between 0.096 and 0.151 s, independent of the mesh.

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
| 160x20 | 143 | 265.348 | 1.8556 | 0.0315 | 0.0295 | 1.8237 | 18.87 | 0.0966 | NATIVE_CONVERGED |
| 240x30 | 402 | 1466.449 | 3.6479 | 0.0627 | 0.0592 | 3.5849 | 17.39 | 0.2062 | NATIVE_CONVERGED |
| 320x40 | 405 | 1989.181 | 4.9116 | 0.1101 | 0.1040 | 4.8012 | 17.61 | 0.2727 | NATIVE_CONVERGED |
| 400x50 | 181 | 1317.469 | 7.2788 | 0.1818 | 0.1728 | 7.0959 | 19.28 | 0.3680 | NATIVE_CONVERGED |
| 480x60 | 329 | 3417.446 | 10.3874 | 0.2764 | 0.2632 | 10.1100 | 18.22 | 0.5549 | NATIVE_CONVERGED |
| 560x70 | 407 | 5723.408 | 14.0624 | 0.3828 | 0.3639 | 13.6785 | 18.05 | 0.7580 | NATIVE_CONVERGED |
| 640x80 | 230 | 3836.823 | 16.6818 | 0.5134 | 0.4878 | 16.1659 | 18.26 | 0.8853 | NATIVE_CONVERGED |
| 720x90 | 372 | 7469.698 | 20.0798 | 0.9182 | 0.8832 | 19.1588 | 17.79 | 1.0771 | NATIVE_CONVERGED |
| 800x100 | 311 | 6592.439 | 21.1976 | 1.1497 | 1.1055 | 20.0436 | 18.68 | 1.0729 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 1.887535e-05 | 1.2606 | 0.9578 | 9 |
| Yuksel | 8.328299e-07 | 1.7112 | 0.9778 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 2.438187e-01 | 0.9196 | 0.8762 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 3.290065e-03 | 0.7834 | 0.9947 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 3.139871e-06 | 1.1181 | 0.9853 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 2.768378e-06 | 1.1256 | 0.9851 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 3.535584e-03 | 0.7732 | 0.9940 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.939179e-04 | 0.7732 | 0.9904 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0132 | 0.0090 | 0.0119 | 0.0295 | 0.0966 |
| 240x30 | 14942 | 0.0265 | 0.0199 | 0.0261 | 0.0592 | 0.2062 |
| 320x40 | 26322 | 0.0490 | 0.0367 | 0.0469 | 0.1040 | 0.2727 |
| 400x50 | 40902 | 0.0789 | 0.0597 | 0.0750 | 0.1728 | 0.3680 |
| 480x60 | 58682 | 0.1138 | 0.0874 | 0.1080 | 0.2632 | 0.5549 |
| 560x70 | 79662 | 0.1627 | 0.1225 | 0.1496 | 0.3639 | 0.7580 |
| 640x80 | 103842 | 0.2145 | 0.1635 | 0.1991 | 0.4878 | 0.8853 |
| 720x90 | 131222 | 0.5578 | 0.2877 | 0.4051 | 0.8832 | 1.0771 |
| 800x100 | 161802 | 0.7178 | 0.3581 | 0.5041 | 1.1055 | 1.0729 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 32 | 0.029 | 0.423 | 0.452 | 0.240 | 0.692 | 83.8478 † | 153.6910 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 56 | 50 | 0.502 | 0.594 | 1.096 | 0.207 | 1.303 | 157.4248 | 157.2184 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 143 | 2699 | 4.504 | 260.786 | 265.290 | 0.057 | 265.348 | 169.1802 | 169.1671 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 36 | 0.061 | 0.955 | 1.016 | 0.360 | 1.376 | 86.6332 † | 157.0915 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 54 | 35 | 1.076 | 0.914 | 1.990 | 0.344 | 2.334 | 159.4154 | 159.2583 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 402 | 6990 | 25.212 | 1441.147 | 1466.358 | 0.090 | 1466.449 | 167.3216 | 167.3133 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 35 | 0.109 | 1.714 | 1.823 | 0.595 | 2.418 | 105.2046 † | 159.0873 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 72 | 97 | 2.640 | 4.554 | 7.193 | 0.481 | 7.675 | 160.9092 | 160.8408 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 405 | 7131 | 44.571 | 1944.481 | 1989.052 | 0.129 | 1989.181 | 165.6647 | 165.6554 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 32 | 0.182 | 2.524 | 2.706 | 0.820 | 3.526 | 110.8097 † | 160.3766 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 128 | 128 | 7.640 | 9.596 | 17.236 | 0.750 | 17.986 | 159.9431 | 159.8050 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 181 | 3490 | 32.910 | 1284.353 | 1317.263 | 0.206 | 1317.469 | 166.3542 | 166.3452 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 50 | 0.253 | 5.691 | 5.944 | 0.774 | 6.718 | 160.2877 | 160.2878 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 136 | 170 | 11.882 | 18.357 | 30.239 | 1.089 | 31.328 | 160.2625 | 160.1911 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 329 | 5994 | 90.943 | 3326.193 | 3417.136 | 0.310 | 3417.446 | 165.7849 | 165.7753 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 57 | 0.353 | 9.271 | 9.624 | 0.929 | 10.553 | 160.7771 | 160.7772 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 195 | 187 | 23.893 | 27.971 | 51.864 | 1.316 | 53.180 | 160.1637 | 160.0986 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 407 | 7345 | 155.787 | 5567.170 | 5722.957 | 0.451 | 5723.408 | 165.5007 | 165.4905 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 47 | 0.477 | 10.083 | 10.560 | 1.732 | 12.291 | 161.0925 | 161.0929 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 180 | 225 | 29.430 | 44.791 | 74.221 | 1.729 | 75.950 | 160.5510 | 160.5098 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 230 | 4200 | 118.073 | 3718.153 | 3836.226 | 0.598 | 3836.823 | 165.5418 | 165.5313 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 45 | 0.944 | 25.102 | 26.047 | 2.537 | 28.584 | 87.3049 † | 161.5701 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 252 | 300 | 72.512 | 121.536 | 194.047 | 2.846 | 196.894 | 160.8630 | 160.8257 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 372 | 6617 | 341.552 | 7127.079 | 7468.632 | 1.067 | 7469.698 | 165.5049 | 165.4939 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 56 | 1.264 | 40.197 | 41.461 | 3.341 | 44.802 | 140.1271 † | 161.6303 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 251 | 344 | 89.893 | 173.426 | 263.318 | 3.515 | 266.833 | 161.0426 | 161.0230 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 311 | 5810 | 357.547 | 6233.575 | 6591.122 | 1.317 | 6592.439 | 165.3403 | 165.3277 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

