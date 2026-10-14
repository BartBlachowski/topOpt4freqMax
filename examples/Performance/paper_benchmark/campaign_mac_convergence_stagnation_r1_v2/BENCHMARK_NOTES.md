# Conference performance benchmark -- notes

Generated 2026-10-13T19:47:46+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_convergence_stagnation_r1_v2`
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
| 160x20 | 0.428 | 0.029 | 0.399 | 0.574 | 1.117 |
| 240x30 | 0.206 | 0.060 | 0.146 | 0.465 | 1.581 |
| 320x40 | 0.263 | 0.110 | 0.153 | 0.691 | 2.846 |
| 400x50 | 0.347 | 0.185 | 0.162 | 1.151 | 4.922 |
| 480x60 | 0.410 | 0.254 | 0.156 | 1.077 | 6.902 |
| 560x70 | 0.527 | 0.357 | 0.170 | 1.142 | 9.530 |
| 640x80 | 0.649 | 0.475 | 0.175 | 1.283 | 12.968 |
| 720x90 | 1.120 | 0.933 | 0.187 | 2.631 | 33.629 |
| 800x100 | 1.459 | 1.262 | 0.197 | 3.385 | 44.337 |

The largest preparation value, 0.399 s at 160x20, is the first measured row of its session; the other 8 rows lie between 0.146 and 0.197 s, independent of the mesh.

### Native omega_1 values flagged

6 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 79.99, 79.99, 108.30 | 153.84 | 48.0% | 0.008 |
| Proposed | 240x30 | 90.49, 90.49, 104.98 | 157.08 | 42.4% | 0.009 |
| Proposed | 320x40 | 116.39, 116.39, 119.02 | 158.93 | 26.8% | 0.004 |
| Proposed | 400x50 | 148.43, 150.07, 153.01 | 159.95 | 7.2% | 0.003 |
| Proposed | 720x90 | 141.50, 141.91, 142.91 | 161.27 | 12.3% | 0.002 |
| Proposed | 800x100 | 122.18, 122.63, 123.96 | 161.66 | 24.4% | 0.002 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 59 | 98.928 | 1.6767 | 0.0337 | 0.0307 | 1.6413 | 20.95 | 0.0783 | NATIVE_CONVERGED |
| 240x30 | 46 | 102.967 | 2.2384 | 0.0695 | 0.0648 | 2.1658 | 21.20 | 0.1022 | NATIVE_CONVERGED |
| 320x40 | 70 | 274.652 | 3.9236 | 0.1182 | 0.1103 | 3.8028 | 20.91 | 0.1818 | NATIVE_CONVERGED |
| 400x50 | 64 | 327.960 | 5.1244 | 0.1877 | 0.1767 | 4.9328 | 21.64 | 0.2279 | NATIVE_CONVERGED |
| 480x60 | 82 | 628.827 | 7.6686 | 0.2905 | 0.2749 | 7.3738 | 19.51 | 0.3779 | NATIVE_CONVERGED |
| 560x70 | 98 | 910.700 | 9.2929 | 0.3908 | 0.3705 | 8.8970 | 18.42 | 0.4830 | NATIVE_CONVERGED |
| 640x80 | 131 | 1656.983 | 12.6487 | 0.5158 | 0.4890 | 12.1280 | 18.61 | 0.6517 | NATIVE_CONVERGED |
| 720x90 | 118 | 1694.428 | 14.3596 | 0.9274 | 0.8919 | 13.4230 | 19.37 | 0.6929 | NATIVE_CONVERGED |
| 800x100 | 119 | 1931.418 | 16.2304 | 1.1567 | 1.1120 | 15.0623 | 20.67 | 0.7286 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 7.176597e-05 | 1.1413 | 0.9327 | 9 |
| Yuksel | 1.060837e-03 | 0.8940 | 0.9403 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 1.293057e-02 | 1.0561 | 0.9500 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 3.175341e-03 | 0.7563 | 0.9860 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 4.323813e-06 | 1.0903 | 0.9854 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.503792e-06 | 1.1053 | 0.9855 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 3.485947e-03 | 0.7427 | 0.9853 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.256497e-04 | 0.7748 | 0.9763 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0129 | 0.0125 | 0.0189 | 0.0307 | 0.0783 |
| 240x30 | 14942 | 0.0264 | 0.0234 | 0.0327 | 0.0648 | 0.1022 |
| 320x40 | 26322 | 0.0499 | 0.0397 | 0.0532 | 0.1103 | 0.1818 |
| 400x50 | 40902 | 0.0815 | 0.0624 | 0.0825 | 0.1767 | 0.2279 |
| 480x60 | 58682 | 0.1185 | 0.0894 | 0.1183 | 0.2749 | 0.3779 |
| 560x70 | 79662 | 0.1639 | 0.1231 | 0.1621 | 0.3705 | 0.4830 |
| 640x80 | 103842 | 0.2198 | 0.1652 | 0.2104 | 0.4890 | 0.6517 |
| 720x90 | 131222 | 0.5567 | 0.2920 | 0.4269 | 0.8919 | 0.6929 |
| 800x100 | 161802 | 0.7216 | 0.3680 | 0.5242 | 1.1120 | 0.7286 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 40 | 0.029 | 0.514 | 0.543 | 0.574 | 1.117 | 79.9875 † | 153.8395 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 78 | 34 | 0.977 | 0.644 | 1.621 | 0.344 | 1.965 | 157.8837 | 157.6615 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 59 | 1236 | 1.990 | 96.834 | 98.823 | 0.104 | 98.928 | 168.6582 | 168.6230 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 40 | 0.060 | 1.056 | 1.116 | 0.465 | 1.581 | 90.4948 † | 157.0766 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 57 | 23 | 1.331 | 0.753 | 2.084 | 0.416 | 2.501 | 159.9319 | 159.6670 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 46 | 975 | 3.199 | 99.629 | 102.827 | 0.140 | 102.967 | 167.5633 | 167.5543 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 41 | 0.110 | 2.045 | 2.155 | 0.691 | 2.846 | 116.3935 † | 158.9330 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 51 | 48 | 2.026 | 2.555 | 4.581 | 0.564 | 5.145 | 161.1381 | 160.9929 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 70 | 1464 | 8.275 | 266.199 | 274.474 | 0.178 | 274.652 | 165.9930 | 165.9831 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 44 | 0.185 | 3.586 | 3.770 | 1.151 | 4.922 | 148.4338 † | 159.9452 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 52 | 25 | 3.244 | 2.063 | 5.307 | 0.952 | 6.259 | 159.4157 | 159.2338 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 64 | 1385 | 12.011 | 315.700 | 327.710 | 0.250 | 327.960 | 166.8131 | 166.8030 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 47 | 0.254 | 5.571 | 5.825 | 1.077 | 6.902 | 160.3111 | 160.3118 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 47 | 23 | 4.202 | 2.721 | 6.923 | 1.416 | 8.339 | 159.5337 | 159.3807 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 82 | 1600 | 23.824 | 604.649 | 628.473 | 0.354 | 628.827 | 166.2009 | 166.1907 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 49 | 0.357 | 8.030 | 8.387 | 1.142 | 9.530 | 160.8480 | 160.8481 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 43 | 24 | 5.295 | 3.890 | 9.186 | 1.862 | 11.048 | 159.9082 | 159.7794 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 98 | 1805 | 38.303 | 871.905 | 910.208 | 0.492 | 910.700 | 165.9476 | 165.9364 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 51 | 0.475 | 11.210 | 11.685 | 1.283 | 12.968 | 161.0083 | 161.0083 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 45 | 23 | 7.434 | 4.839 | 12.273 | 2.116 | 14.389 | 159.9804 | 159.8795 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 131 | 2438 | 67.574 | 1588.773 | 1656.347 | 0.636 | 1656.983 | 165.7042 | 165.6928 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 54 | 0.933 | 30.064 | 30.998 | 2.631 | 33.629 | 141.5015 † | 161.2738 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 44 | 24 | 12.847 | 10.245 | 23.092 | 3.316 | 26.408 | 160.2044 | 160.1040 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 118 | 2286 | 109.430 | 1583.918 | 1693.349 | 1.079 | 1694.428 | 164.4823 | 164.4694 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 55 | 1.262 | 39.689 | 40.952 | 3.385 | 44.337 | 122.1839 † | 161.6556 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 55 | 26 | 20.242 | 13.630 | 33.872 | 3.658 | 37.530 | 160.0071 | 159.9356 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 119 | 2460 | 137.645 | 1792.419 | 1930.065 | 1.354 | 1931.418 | 163.5091 | 163.4935 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

