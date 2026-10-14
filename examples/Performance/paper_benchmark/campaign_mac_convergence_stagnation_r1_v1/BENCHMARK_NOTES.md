# Conference performance benchmark -- notes

Generated 2026-10-14T01:14:07+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_convergence_stagnation_r1_v1`
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
| 160x20 | 0.420 | 0.028 | 0.392 | 0.543 | 1.165 |
| 240x30 | 0.208 | 0.059 | 0.149 | 0.467 | 1.783 |
| 320x40 | 0.265 | 0.109 | 0.156 | 0.567 | 3.396 |
| 400x50 | 0.339 | 0.185 | 0.154 | 0.564 | 8.300 |
| 480x60 | 0.414 | 0.254 | 0.160 | 0.796 | 8.281 |
| 560x70 | 0.519 | 0.355 | 0.165 | 0.992 | 12.126 |
| 640x80 | 0.652 | 0.477 | 0.175 | 1.257 | 16.518 |
| 720x90 | 1.122 | 0.937 | 0.185 | 2.106 | 49.222 |
| 800x100 | 1.460 | 1.265 | 0.195 | 2.749 | 75.844 |

The largest preparation value, 0.392 s at 160x20, is the first measured row of its session; the other 8 rows lie between 0.149 and 0.195 s, independent of the mesh.

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
| 160x20 | 67 | 117.491 | 1.7536 | 0.0326 | 0.0304 | 1.7196 | 21.18 | 0.0812 | NATIVE_CONVERGED |
| 240x30 | 58 | 147.467 | 2.5425 | 0.0692 | 0.0652 | 2.4712 | 20.28 | 0.1219 | NATIVE_CONVERGED |
| 320x40 | 79 | 311.549 | 3.9437 | 0.1148 | 0.1081 | 3.8266 | 20.44 | 0.1872 | NATIVE_CONVERGED |
| 400x50 | 79 | 452.742 | 5.7309 | 0.1850 | 0.1755 | 5.5428 | 21.03 | 0.2636 | NATIVE_CONVERGED |
| 480x60 | 86 | 671.039 | 7.8028 | 0.2850 | 0.2716 | 7.5137 | 19.43 | 0.3867 | NATIVE_CONVERGED |
| 560x70 | 105 | 1018.134 | 9.6965 | 0.3868 | 0.3680 | 9.3050 | 18.36 | 0.5068 | NATIVE_CONVERGED |
| 640x80 | 136 | 1763.916 | 12.9700 | 0.5124 | 0.4870 | 12.4529 | 18.55 | 0.6713 | NATIVE_CONVERGED |
| 720x90 | 187 | 3132.166 | 16.7496 | 0.9221 | 0.8873 | 15.8216 | 18.52 | 0.8544 | NATIVE_CONVERGED |
| 800x100 | 242 | 4808.219 | 19.8687 | 1.1486 | 1.1044 | 18.7145 | 18.93 | 0.9884 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 3.049923e-05 | 1.2536 | 0.9241 | 9 |
| Yuksel | 1.307381e-03 | 0.9524 | 0.9475 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 5.715646e-03 | 1.1678 | 0.9360 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 2.754213e-03 | 0.7786 | 0.9868 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 3.992434e-06 | 1.0968 | 0.9855 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.464112e-06 | 1.1057 | 0.9853 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 2.966186e-03 | 0.7673 | 0.9871 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 9.679336e-05 | 0.8118 | 0.9864 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0132 | 0.0122 | 0.0177 | 0.0304 | 0.0812 |
| 240x30 | 14942 | 0.0267 | 0.0237 | 0.0328 | 0.0652 | 0.1219 |
| 320x40 | 26322 | 0.0495 | 0.0406 | 0.0533 | 0.1081 | 0.1872 |
| 400x50 | 40902 | 0.0821 | 0.0628 | 0.0788 | 0.1755 | 0.2636 |
| 480x60 | 58682 | 0.1148 | 0.0876 | 0.1138 | 0.2716 | 0.3867 |
| 560x70 | 79662 | 0.1658 | 0.1210 | 0.1557 | 0.3680 | 0.5068 |
| 640x80 | 103842 | 0.2174 | 0.1611 | 0.2046 | 0.4870 | 0.6713 |
| 720x90 | 131222 | 0.5564 | 0.2854 | 0.4177 | 0.8873 | 0.8544 |
| 800x100 | 161802 | 0.7256 | 0.3596 | 0.5147 | 1.1044 | 0.9884 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 45 | 0.028 | 0.594 | 0.622 | 0.543 | 1.165 | 80.6186 † | 153.8243 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 174 | 84 | 2.129 | 1.490 | 3.620 | 0.334 | 3.954 | 157.6655 | 157.5117 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 67 | 1419 | 2.186 | 115.212 | 117.398 | 0.093 | 117.491 | 168.7712 | 168.7447 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 47 | 0.059 | 1.256 | 1.316 | 0.467 | 1.783 | 104.8684 † | 157.0737 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 100 | 72 | 2.372 | 2.360 | 4.733 | 0.368 | 5.101 | 159.5170 | 159.4112 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 58 | 1176 | 4.013 | 143.331 | 147.344 | 0.122 | 147.467 | 167.3843 | 167.3753 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 55 | 0.109 | 2.720 | 2.829 | 0.567 | 3.396 | 130.8576 † | 158.8833 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 102 | 90 | 4.146 | 4.795 | 8.941 | 0.547 | 9.488 | 160.8034 | 160.7151 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 79 | 1615 | 9.070 | 302.305 | 311.375 | 0.174 | 311.549 | 165.8955 | 165.8857 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 92 | 0.185 | 7.550 | 7.736 | 0.564 | 8.300 | 159.6588 | 159.6590 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 85 | 99 | 5.335 | 7.804 | 13.140 | 0.819 | 13.959 | 159.9425 | 159.7953 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 79 | 1661 | 14.611 | 437.880 | 452.491 | 0.251 | 452.742 | 166.5309 | 166.5212 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 63 | 0.254 | 7.231 | 7.485 | 0.796 | 8.281 | 160.2540 | 160.2541 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 81 | 146 | 7.093 | 16.612 | 23.705 | 1.139 | 24.844 | 160.3266 | 160.2441 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 86 | 1671 | 24.511 | 646.176 | 670.687 | 0.352 | 671.039 | 166.1630 | 166.1528 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 65 | 0.355 | 10.779 | 11.134 | 0.992 | 12.126 | 160.7651 | 160.7652 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 83 | 71 | 10.041 | 11.055 | 21.096 | 1.572 | 22.669 | 160.0461 | 159.9506 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 105 | 1928 | 40.611 | 977.023 | 1017.634 | 0.500 | 1018.134 | 165.9175 | 165.9066 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 68 | 0.477 | 14.784 | 15.261 | 1.257 | 16.518 | 160.9023 | 160.9023 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 84 | 90 | 13.532 | 18.417 | 31.949 | 1.759 | 33.708 | 160.4446 | 160.3876 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 136 | 2523 | 69.692 | 1693.591 | 1763.283 | 0.633 | 1763.916 | 165.6984 | 165.6870 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 83 | 0.937 | 46.179 | 47.116 | 2.106 | 49.222 | 161.0322 | 161.0322 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 88 | 92 | 25.115 | 38.428 | 63.543 | 2.888 | 66.431 | 160.8026 | 160.7435 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 187 | 3463 | 172.429 | 2958.643 | 3131.072 | 1.095 | 3132.166 | 165.4250 | 165.4128 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 99 | 1.265 | 71.830 | 73.095 | 2.749 | 75.844 | 161.2201 | 161.2202 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 95 | 81 | 34.164 | 41.692 | 75.856 | 3.578 | 79.434 | 160.7860 | 160.7341 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 242 | 4582 | 277.961 | 4528.907 | 4806.868 | 1.351 | 4808.219 | 165.4282 | 165.4149 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

