# Conference performance benchmark -- notes

Generated 2026-10-13T02:28:11+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_windows_convergence_stagnation_r1`
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
| 160x20 | 0.162 | 0.066 | 0.097 | 0.416 | 1.546 |
| 240x30 | 0.209 | 0.110 | 0.099 | 0.749 | 3.421 |
| 320x40 | 0.272 | 0.195 | 0.077 | 0.825 | 6.552 |
| 400x50 | 0.471 | 0.347 | 0.124 | 0.918 | 16.494 |
| 480x60 | 0.633 | 0.504 | 0.129 | 1.307 | 16.669 |
| 560x70 | 1.014 | 0.853 | 0.161 | 1.943 | 24.642 |
| 640x80 | 1.201 | 1.039 | 0.162 | 2.543 | 33.947 |
| 720x90 | 2.187 | 2.008 | 0.179 | 4.348 | 96.950 |
| 800x100 | 2.825 | 2.627 | 0.199 | 5.556 | 148.445 |

The largest preparation value, 0.199 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.077 and 0.179 s, independent of the mesh.

### Native omega_1 values flagged

3 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 80.62, 80.62, 108.05 | 153.82 | 47.6% | 0.009 |
| Proposed | 240x30 | 104.87, 104.87, 107.23 | 157.07 | 33.2% | 0.008 |
| Proposed | 320x40 | 130.86, 131.66, 133.66 | 158.88 | 17.6% | 0.003 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 63 | 175.440 | 2.7848 | 0.0547 | 0.0502 | 2.7287 | 19.83 | 0.1376 | NATIVE_CONVERGED |
| 240x30 | 55 | 299.634 | 5.4479 | 0.1300 | 0.1217 | 5.3151 | 20.89 | 0.2544 | NATIVE_CONVERGED |
| 320x40 | 93 | 875.748 | 9.4166 | 0.2277 | 0.2129 | 9.1864 | 19.73 | 0.4656 | NATIVE_CONVERGED |
| 400x50 | 80 | 1154.335 | 14.4292 | 0.4822 | 0.4438 | 13.9418 | 20.11 | 0.6932 | NATIVE_CONVERGED |
| 480x60 | 82 | 1431.134 | 17.4528 | 0.5837 | 0.5352 | 16.8615 | 20.05 | 0.8410 | NATIVE_CONVERGED |
| 560x70 | 116 | 3357.809 | 28.9466 | 0.9806 | 0.9038 | 27.9582 | 18.89 | 1.4802 | NATIVE_CONVERGED |
| 640x80 | 134 | 5470.309 | 40.8232 | 1.2203 | 1.1224 | 39.5933 | 18.84 | 2.1012 | NATIVE_CONVERGED |
| 720x90 | 201 | 10462.554 | 52.0525 | 2.1640 | 2.0282 | 49.8773 | 18.50 | 2.6964 | NATIVE_CONVERGED |
| 800x100 | 227 | 15386.867 | 67.7836 | 2.6005 | 2.4341 | 65.1708 | 18.89 | 3.4508 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 2.246446e-05 | 1.3478 | 0.9515 | 9 |
| Yuksel | 8.919451e-07 | 1.8432 | 0.9851 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 1.526953e-03 | 1.3937 | 0.9545 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 8.212183e-04 | 0.9923 | 0.9901 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 2.973692e-06 | 1.2033 | 0.9913 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 2.704909e-06 | 1.2052 | 0.9909 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 8.461350e-04 | 0.9862 | 0.9899 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 3.277695e-05 | 1.0141 | 0.9866 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0237 | 0.0141 | 0.0185 | 0.0502 | 0.1376 |
| 240x30 | 14942 | 0.0545 | 0.0359 | 0.0466 | 0.1217 | 0.2544 |
| 320x40 | 26322 | 0.1006 | 0.0715 | 0.0900 | 0.2129 | 0.4656 |
| 400x50 | 40902 | 0.1655 | 0.1161 | 0.1464 | 0.4438 | 0.6932 |
| 480x60 | 58682 | 0.2358 | 0.1718 | 0.2138 | 0.5352 | 0.8410 |
| 560x70 | 79662 | 0.3361 | 0.2426 | 0.2946 | 0.9038 | 1.4802 |
| 640x80 | 103842 | 0.4465 | 0.3512 | 0.5347 | 1.1224 | 2.1012 |
| 720x90 | 131222 | 1.0915 | 0.5632 | 0.7935 | 2.0282 | 2.6964 |
| 800x100 | 161802 | 1.4168 | 0.7129 | 1.0036 | 2.4341 | 3.4508 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 45 | 0.066 | 1.065 | 1.131 | 0.416 | 1.546 | 80.6186 † | 153.8243 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 101 | 1.711 | 1.868 | 3.579 | 0.334 | 3.913 | 157.4175 | 157.2593 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 63 | 1249 | 3.448 | 171.910 | 175.359 | 0.081 | 175.440 | 168.3462 | 168.3309 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 47 | 0.110 | 2.562 | 2.672 | 0.749 | 3.421 | 104.8684 † | 157.0737 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 41 | 6.039 | 1.911 | 7.951 | 0.521 | 8.472 | 160.1553 | 160.0577 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 55 | 1149 | 7.149 | 292.332 | 299.481 | 0.153 | 299.634 | 167.4267 | 167.4177 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 55 | 0.195 | 5.532 | 5.727 | 0.825 | 6.552 | 130.8576 † | 158.8833 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 113 | 18.028 | 10.175 | 28.203 | 0.754 | 28.957 | 160.9653 | 160.8961 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 93 | 1835 | 21.178 | 854.334 | 875.512 | 0.236 | 875.748 | 166.1703 | 166.1614 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 92 | 0.347 | 15.230 | 15.576 | 0.918 | 16.494 | 159.6588 | 159.6590 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 133 | 36.575 | 19.475 | 56.050 | 1.428 | 57.478 | 159.5747 | 159.4903 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 80 | 1609 | 38.575 | 1115.341 | 1153.917 | 0.419 | 1154.335 | 166.4025 | 166.3932 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 63 | 0.504 | 14.858 | 15.362 | 1.307 | 16.669 | 160.2540 | 160.2541 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 586 | 133 | 100.696 | 28.430 | 129.126 | 1.940 | 131.066 | 159.1423 | 159.0790 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 82 | 1644 | 47.863 | 1382.646 | 1430.508 | 0.625 | 1431.134 | 166.1806 | 166.1705 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 65 | 0.853 | 21.847 | 22.700 | 1.943 | 24.642 | 160.7651 | 160.7652 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 833 | 178 | 202.052 | 52.437 | 254.489 | 2.633 | 257.122 | 159.0228 | 158.9843 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 116 | 2191 | 113.755 | 3243.150 | 3356.905 | 0.904 | 3357.809 | 165.8449 | 165.8342 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 68 | 1.039 | 30.365 | 31.404 | 2.543 | 33.947 | 160.9023 | 160.9023 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 1311 | 160 | 460.391 | 85.552 | 545.943 | 3.718 | 549.661 | 158.8421 | 158.8090 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 134 | 2525 | 163.524 | 5305.505 | 5469.029 | 1.280 | 5470.309 | 165.7290 | 165.7176 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 83 | 2.008 | 90.594 | 92.602 | 4.348 | 96.950 | 161.0322 | 161.0322 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 1162 | 164 | 654.403 | 130.138 | 784.541 | 5.902 | 790.443 | 159.2647 | 159.2328 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 201 | 3718 | 434.971 | 10025.329 | 10460.300 | 2.254 | 10462.554 | 165.5406 | 165.5290 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 99 | 2.627 | 140.262 | 142.889 | 5.556 | 148.445 | 161.2201 | 161.2202 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 1150 | 143 | 819.816 | 143.516 | 963.332 | 7.396 | 970.729 | 159.3088 | 159.2799 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 227 | 4287 | 590.323 | 14793.774 | 15384.096 | 2.771 | 15386.867 | 165.2606 | 165.2477 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

