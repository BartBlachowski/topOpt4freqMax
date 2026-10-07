# Conference performance benchmark -- notes

Generated 2026-10-07T23:13:08+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_convergence_corrected_olhoff_08`
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
| 160x20 | 0.473 | 0.031 | 0.441 | 0.619 | 1.092 |
| 240x30 | 0.201 | 0.060 | 0.141 | 0.425 | 1.478 |
| 320x40 | 0.253 | 0.108 | 0.145 | 0.620 | 2.441 |
| 400x50 | 0.339 | 0.184 | 0.155 | 0.921 | 3.667 |
| 480x60 | 0.420 | 0.256 | 0.164 | 0.859 | 6.906 |
| 560x70 | 0.581 | 0.356 | 0.225 | 1.076 | 10.909 |
| 640x80 | 0.652 | 0.475 | 0.176 | 1.815 | 12.506 |
| 720x90 | 1.126 | 0.939 | 0.187 | 2.591 | 28.550 |
| 800x100 | 1.453 | 1.258 | 0.195 | 3.379 | 45.021 |

The largest preparation value, 0.441 s at 160x20, is the first measured row of its session; the other 8 rows lie between 0.141 and 0.225 s, independent of the mesh.

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
| 160x20 | 74 | 129.382 | 1.7484 | 0.0327 | 0.0304 | 1.7145 | 20.84 | 0.0823 | NATIVE_CONVERGED |
| 240x30 | 75 | 212.437 | 2.8325 | 0.0666 | 0.0626 | 2.7643 | 19.53 | 0.1415 | NATIVE_CONVERGED |
| 320x40 | 83 | 336.174 | 4.0503 | 0.1156 | 0.1088 | 3.9326 | 20.28 | 0.1939 | NATIVE_CONVERGED |
| 400x50 | 78 | 447.378 | 5.7356 | 0.1872 | 0.1774 | 5.5452 | 21.06 | 0.2633 | NATIVE_CONVERGED |
| 480x60 | 92 | 745.121 | 8.0991 | 0.2874 | 0.2734 | 7.8079 | 19.34 | 0.4038 | NATIVE_CONVERGED |
| 560x70 | 108 | 1067.941 | 9.8883 | 0.3890 | 0.3693 | 9.4948 | 18.32 | 0.5182 | NATIVE_CONVERGED |
| 640x80 | 138 | 1808.306 | 13.1037 | 0.5139 | 0.4871 | 12.5851 | 18.53 | 0.6792 | NATIVE_CONVERGED |
| 720x90 | 181 | 2981.456 | 16.4721 | 0.9221 | 0.8864 | 15.5440 | 18.57 | 0.8371 | NATIVE_CONVERGED |
| 800x100 | 235 | 4605.421 | 19.5975 | 1.1426 | 1.0993 | 18.4491 | 18.99 | 0.9714 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 5.729887e-05 | 1.1573 | 0.9271 | 9 |
| Yuksel | 2.660852e-06 | 1.6054 | 0.9674 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 1.300851e-02 | 1.0927 | 0.9402 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 3.278094e-03 | 0.7631 | 0.9912 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 3.831034e-06 | 1.1008 | 0.9864 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.311797e-06 | 1.1100 | 0.9861 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 3.555189e-03 | 0.7512 | 0.9917 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.279185e-04 | 0.7867 | 0.9888 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0138 | 0.0125 | 0.0173 | 0.0304 | 0.0823 |
| 240x30 | 14942 | 0.0276 | 0.0229 | 0.0319 | 0.0626 | 0.1415 |
| 320x40 | 26322 | 0.0489 | 0.0397 | 0.0527 | 0.1088 | 0.1939 |
| 400x50 | 40902 | 0.0801 | 0.0634 | 0.0813 | 0.1774 | 0.2633 |
| 480x60 | 58682 | 0.1158 | 0.0920 | 0.1147 | 0.2734 | 0.4038 |
| 560x70 | 79662 | 0.1663 | 0.1254 | 0.1555 | 0.3693 | 0.5182 |
| 640x80 | 103842 | 0.2174 | 0.1661 | 0.2040 | 0.4871 | 0.6792 |
| 720x90 | 131222 | 0.5560 | 0.2919 | 0.4124 | 0.8864 | 0.8371 |
| 800x100 | 161802 | 0.7211 | 0.3610 | 0.5103 | 1.0993 | 0.9714 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 32 | 0.031 | 0.441 | 0.472 | 0.619 | 1.092 | 83.8478 † | 153.6910 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 56 | 50 | 0.698 | 0.867 | 1.565 | 0.363 | 1.928 | 157.4248 | 157.2184 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 74 | 1542 | 2.418 | 126.873 | 129.291 | 0.092 | 129.382 | 168.8287 | 168.7781 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 36 | 0.060 | 0.993 | 1.053 | 0.425 | 1.478 | 86.6332 † | 157.0915 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 54 | 35 | 1.237 | 1.116 | 2.353 | 0.438 | 2.791 | 159.4154 | 159.2583 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 75 | 1465 | 4.993 | 207.323 | 212.315 | 0.122 | 212.437 | 167.3632 | 167.3539 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 35 | 0.108 | 1.712 | 1.820 | 0.620 | 2.441 | 105.2046 † | 159.0873 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 72 | 97 | 2.859 | 5.114 | 7.973 | 0.570 | 8.543 | 160.9092 | 160.8408 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 83 | 1683 | 9.598 | 326.403 | 336.001 | 0.173 | 336.174 | 165.8921 | 165.8827 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 32 | 0.184 | 2.562 | 2.746 | 0.921 | 3.667 | 110.8097 † | 160.3766 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 128 | 128 | 8.111 | 10.412 | 18.523 | 0.862 | 19.385 | 159.9431 | 159.8050 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 78 | 1643 | 14.601 | 432.529 | 447.130 | 0.248 | 447.378 | 166.5398 | 166.5301 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 50 | 0.256 | 5.792 | 6.048 | 0.859 | 6.906 | 160.2877 | 160.2878 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 136 | 170 | 12.507 | 19.501 | 32.008 | 1.186 | 33.193 | 160.2625 | 160.1911 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 92 | 1779 | 26.441 | 718.327 | 744.767 | 0.353 | 745.121 | 166.1366 | 166.1265 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 57 | 0.356 | 9.477 | 9.833 | 1.076 | 10.909 | 160.7771 | 160.7772 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 195 | 187 | 24.460 | 29.071 | 53.530 | 1.407 | 54.937 | 160.1637 | 160.0986 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 108 | 1979 | 42.007 | 1025.443 | 1067.449 | 0.491 | 1067.941 | 165.9030 | 165.8921 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 47 | 0.475 | 10.216 | 10.691 | 1.815 | 12.506 | 161.0925 | 161.0929 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 180 | 225 | 29.889 | 45.911 | 75.800 | 1.814 | 77.615 | 160.5510 | 160.5098 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 138 | 2557 | 70.918 | 1736.750 | 1807.668 | 0.638 | 1808.306 | 165.6946 | 165.6833 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 45 | 0.939 | 25.020 | 25.959 | 2.591 | 28.550 | 87.3049 † | 161.5701 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 252 | 300 | 73.554 | 123.724 | 197.278 | 2.926 | 200.204 | 160.8630 | 160.8257 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 181 | 3361 | 166.905 | 2813.458 | 2980.363 | 1.093 | 2981.456 | 165.4075 | 165.3952 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 56 | 1.258 | 40.384 | 41.642 | 3.379 | 45.021 | 140.1271 † | 161.6303 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 251 | 344 | 90.606 | 175.527 | 266.133 | 3.589 | 269.722 | 161.0426 | 161.0230 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 235 | 4463 | 268.516 | 4335.544 | 4604.060 | 1.361 | 4605.421 | 165.4043 | 165.3908 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

