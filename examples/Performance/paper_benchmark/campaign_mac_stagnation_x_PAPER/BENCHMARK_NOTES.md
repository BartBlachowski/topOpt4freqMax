# Conference performance benchmark -- notes

Generated 2026-10-14T19:13:59+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_stagnation_x_w10_rel1e-2_gray1e-2_audited`
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
| 160x20 | 0.519 | 0.035 | 0.484 | 0.536 | 1.129 |
| 240x30 | 0.272 | 0.060 | 0.211 | 0.348 | 1.521 |
| 320x40 | 0.313 | 0.108 | 0.205 | 0.470 | 2.504 |
| 400x50 | 0.393 | 0.181 | 0.212 | 0.752 | 3.748 |
| 480x60 | 0.462 | 0.249 | 0.213 | 0.633 | 5.688 |
| 560x70 | 0.578 | 0.350 | 0.228 | 0.840 | 7.496 |
| 640x80 | 0.717 | 0.476 | 0.241 | 1.088 | 10.925 |
| 720x90 | 1.181 | 0.943 | 0.238 | 1.410 | 20.152 |
| 800x100 | 1.529 | 1.275 | 0.254 | 1.781 | 28.220 |

The largest preparation value, 0.484 s at 160x20, is the first measured row of its session; the other 8 rows lie between 0.205 and 0.254 s, independent of the mesh.

### Native omega_1 values flagged

6 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 80.11, 80.11, 108.23 | 153.86 | 47.9% | 0.009 |
| Proposed | 240x30 | 98.22, 98.22, 106.14 | 157.07 | 37.5% | 0.009 |
| Proposed | 320x40 | 124.82, 124.82, 125.57 | 158.90 | 21.4% | 0.004 |
| Proposed | 400x50 | 131.91, 131.91, 131.99 | 160.00 | 17.6% | 0.003 |
| Proposed | 720x90 | 88.86, 89.15, 90.34 | 161.45 | 45.0% | 0.002 |
| Proposed | 800x100 | 97.52, 97.96, 99.59 | 161.71 | 39.7% | 0.002 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 64 | 111.259 | 1.7384 | 0.0341 | 0.0303 | 1.7026 | 21.25 | 0.0801 | NATIVE_CONVERGED |
| 240x30 | 58 | 167.413 | 2.8864 | 0.0630 | 0.0584 | 2.8211 | 21.38 | 0.1320 | NATIVE_CONVERGED |
| 320x40 | 71 | 275.620 | 3.8820 | 0.1059 | 0.0981 | 3.7736 | 20.86 | 0.1809 | NATIVE_CONVERGED |
| 400x50 | 62 | 328.914 | 5.3051 | 0.1756 | 0.1644 | 5.1256 | 22.79 | 0.2249 | NATIVE_CONVERGED |
| 480x60 | 77 | 567.029 | 7.3640 | 0.2625 | 0.2465 | 7.0969 | 20.35 | 0.3487 | NATIVE_CONVERGED |
| 560x70 | 99 | 952.362 | 9.6198 | 0.3549 | 0.3343 | 9.2602 | 19.26 | 0.4807 | NATIVE_CONVERGED |
| 640x80 | 131 | 1648.041 | 12.5805 | 0.4713 | 0.4443 | 12.1045 | 18.61 | 0.6504 | NATIVE_CONVERGED |
| 720x90 | 109 | 1542.227 | 14.1489 | 0.8019 | 0.7666 | 13.3380 | 19.06 | 0.6996 | NATIVE_CONVERGED |
| 800x100 | 126 | 2166.105 | 17.1913 | 0.9884 | 0.9437 | 16.1932 | 20.72 | 0.7814 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 2.293566e-04 | 1.0035 | 0.9406 | 9 |
| Yuksel | 1.622124e-04 | 1.0737 | 0.9591 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 3.140723e-02 | 0.9740 | 0.9534 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 4.592517e-03 | 0.7234 | 0.9909 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 5.611973e-06 | 1.0549 | 0.9850 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 4.324364e-06 | 1.0741 | 0.9860 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 4.969370e-03 | 0.7118 | 0.9911 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.717349e-04 | 0.7464 | 0.9812 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0130 | 0.0092 | 0.0116 | 0.0303 | 0.0801 |
| 240x30 | 14942 | 0.0253 | 0.0201 | 0.0263 | 0.0584 | 0.1320 |
| 320x40 | 26322 | 0.0438 | 0.0361 | 0.0468 | 0.0981 | 0.1809 |
| 400x50 | 40902 | 0.0687 | 0.0585 | 0.0744 | 0.1644 | 0.2249 |
| 480x60 | 58682 | 0.1001 | 0.0869 | 0.1105 | 0.2465 | 0.3487 |
| 560x70 | 79662 | 0.1342 | 0.1222 | 0.1548 | 0.3343 | 0.4807 |
| 640x80 | 103842 | 0.1950 | 0.1631 | 0.2039 | 0.4443 | 0.6504 |
| 720x90 | 131222 | 0.3708 | 0.2891 | 0.4174 | 0.7666 | 0.6996 |
| 800x100 | 161802 | 0.4748 | 0.3644 | 0.5178 | 0.9437 | 0.7814 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 43 | 0.035 | 0.557 | 0.592 | 0.536 | 1.129 | 80.1109 † | 153.8584 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 78 | 32 | 0.715 | 0.370 | 1.086 | 0.268 | 1.354 | 157.9278 | 157.7054 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 64 | 1360 | 2.180 | 108.966 | 111.146 | 0.113 | 111.259 | 168.7271 | 168.7000 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 44 | 0.060 | 1.113 | 1.173 | 0.348 | 1.521 | 98.2247 † | 157.0739 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 57 | 26 | 1.144 | 0.684 | 1.828 | 0.253 | 2.082 | 159.7974 | 159.5674 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 58 | 1240 | 3.653 | 163.621 | 167.274 | 0.139 | 167.413 | 167.5508 | 167.5418 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 44 | 0.108 | 1.926 | 2.034 | 0.470 | 2.504 | 124.8182 † | 158.8966 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 51 | 25 | 1.841 | 1.170 | 3.011 | 0.296 | 3.308 | 158.9864 | 158.7672 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 71 | 1481 | 7.520 | 267.926 | 275.446 | 0.174 | 275.620 | 165.9710 | 165.9612 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 41 | 0.181 | 2.815 | 2.997 | 0.752 | 3.748 | 131.9135 † | 159.9995 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 55 | 26 | 3.218 | 1.935 | 5.153 | 0.364 | 5.517 | 159.4140 | 159.2400 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 62 | 1413 | 10.885 | 317.788 | 328.673 | 0.241 | 328.914 | 166.8098 | 166.8000 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 48 | 0.249 | 4.805 | 5.054 | 0.633 | 5.688 | 160.3069 | 160.3072 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 53 | 27 | 4.605 | 2.984 | 7.589 | 0.474 | 8.063 | 159.5540 | 159.4042 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 77 | 1567 | 20.216 | 546.465 | 566.681 | 0.347 | 567.029 | 166.3250 | 166.3145 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 47 | 0.350 | 6.306 | 6.656 | 0.840 | 7.496 | 160.8840 | 160.8842 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 58 | 27 | 7.088 | 4.180 | 11.268 | 0.594 | 11.862 | 159.7795 | 159.6701 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 99 | 1907 | 35.137 | 916.756 | 951.893 | 0.469 | 952.362 | 165.9075 | 165.8964 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 48 | 0.476 | 9.361 | 9.836 | 1.088 | 10.925 | 161.0677 | 161.0680 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 63 | 33 | 10.274 | 6.728 | 17.003 | 0.764 | 17.767 | 160.1130 | 160.0131 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 131 | 2438 | 61.744 | 1585.688 | 1647.432 | 0.609 | 1648.041 | 165.7042 | 165.6928 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 48 | 0.943 | 17.799 | 18.742 | 1.410 | 20.152 | 88.8634 † | 161.4470 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 58 | 32 | 16.767 | 13.357 | 30.125 | 1.055 | 31.180 | 160.2709 | 160.1802 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 109 | 2078 | 87.410 | 1453.841 | 1541.251 | 0.976 | 1542.227 | 164.1365 | 164.1234 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 53 | 1.275 | 25.164 | 26.439 | 1.781 | 28.220 | 97.5155 † | 161.7104 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 62 | 30 | 22.591 | 15.533 | 38.124 | 1.235 | 39.359 | 160.0754 | 160.0074 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 126 | 2611 | 124.541 | 2040.338 | 2164.879 | 1.226 | 2166.105 | 163.2323 | 163.2175 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

