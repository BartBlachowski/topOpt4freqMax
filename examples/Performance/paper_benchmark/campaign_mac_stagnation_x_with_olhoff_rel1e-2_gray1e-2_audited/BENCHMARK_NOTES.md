# Conference performance benchmark -- notes

Generated 2026-10-15T14:31:33+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_stagnation_x_with_olhoff_rel1e-2_gray1e-2_audited`
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
| 160x20 | 0.110 | 0.031 | 0.080 | 0.131 | 0.637 |
| 240x30 | 0.153 | 0.069 | 0.084 | 0.221 | 1.407 |
| 320x40 | 0.226 | 0.133 | 0.092 | 0.370 | 2.304 |
| 400x50 | 0.317 | 0.220 | 0.097 | 0.584 | 3.509 |
| 480x60 | 0.358 | 0.262 | 0.096 | 0.683 | 5.100 |
| 560x70 | 0.529 | 0.368 | 0.162 | 0.977 | 7.523 |
| 640x80 | 0.680 | 0.508 | 0.172 | 1.064 | 9.452 |
| 720x90 | 1.173 | 1.010 | 0.163 | 1.611 | 18.264 |
| 800x100 | 1.519 | 1.361 | 0.158 | 2.160 | 25.564 |

The largest preparation value, 0.172 s at 640x80, is the first measured row of its session; the other 8 rows lie between 0.080 and 0.163 s, independent of the mesh.

### Native omega_1 values flagged

8 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 79.82, 79.82, 108.31 | 153.84 | 48.1% | 0.008 |
| Proposed | 240x30 | 90.49, 90.49, 104.98 | 157.08 | 42.4% | 0.009 |
| Proposed | 320x40 | 116.39, 116.39, 119.02 | 158.93 | 26.8% | 0.004 |
| Proposed | 400x50 | 121.37, 121.37, 122.45 | 160.07 | 24.2% | 0.003 |
| Proposed | 480x60 | 149.26, 150.10, 152.71 | 160.41 | 7.0% | 0.003 |
| Proposed | 640x80 | 112.13, 112.30, 113.57 | 161.38 | 30.5% | 0.002 |
| Proposed | 720x90 | 87.18, 87.30, 88.11 | 161.85 | 46.1% | 0.002 |
| Proposed | 800x100 | 72.70, 72.91, 74.29 | 162.06 | 55.1% | 0.002 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 48 | 90.611 | 1.8877 | 0.0369 | 0.0311 | 1.8494 | 21.60 | 0.0856 | NATIVE_CONVERGED |
| 240x30 | 49 | 148.920 | 3.0392 | 0.0669 | 0.0606 | 2.9700 | 22.41 | 0.1325 | NATIVE_CONVERGED |
| 320x40 | 68 | 289.553 | 4.2581 | 0.1132 | 0.1044 | 4.1431 | 21.03 | 0.1970 | NATIVE_CONVERGED |
| 400x50 | 61 | 339.370 | 5.5634 | 0.1831 | 0.1706 | 5.3772 | 22.87 | 0.2351 | NATIVE_CONVERGED |
| 480x60 | 74 | 572.408 | 7.7352 | 0.2799 | 0.2629 | 7.4506 | 20.45 | 0.3644 | NATIVE_CONVERGED |
| 560x70 | 95 | 937.624 | 9.8697 | 0.3789 | 0.3556 | 9.4860 | 19.36 | 0.4900 | NATIVE_CONVERGED |
| 640x80 | 83 | 925.234 | 11.1474 | 0.4957 | 0.4648 | 10.6446 | 19.47 | 0.5467 | NATIVE_CONVERGED |
| 720x90 | 86 | 1182.971 | 13.7555 | 0.8298 | 0.7895 | 12.9144 | 19.62 | 0.6583 | NATIVE_CONVERGED |
| 800x100 | 101 | 1690.947 | 16.7421 | 1.0273 | 0.9776 | 15.7018 | 21.59 | 0.7271 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 7.381724e-05 | 1.1026 | 0.9754 | 9 |
| Yuksel | 1.547726e-04 | 1.0530 | 0.9644 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 4.915214e-02 | 0.9146 | 0.9827 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 7.380064e-03 | 0.6781 | 0.9945 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 6.607122e-06 | 1.0442 | 0.9853 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 4.415041e-06 | 1.0764 | 0.9882 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 8.127130e-03 | 0.6645 | 0.9950 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 2.834472e-04 | 0.6960 | 0.9885 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0122 | 0.0090 | 0.0120 | 0.0311 | 0.0856 |
| 240x30 | 14942 | 0.0279 | 0.0209 | 0.0260 | 0.0606 | 0.1325 |
| 320x40 | 26322 | 0.0439 | 0.0361 | 0.0481 | 0.1044 | 0.1970 |
| 400x50 | 40902 | 0.0712 | 0.0605 | 0.0762 | 0.1706 | 0.2351 |
| 480x60 | 58682 | 0.1014 | 0.0845 | 0.1092 | 0.2629 | 0.3644 |
| 560x70 | 79662 | 0.1404 | 0.1289 | 0.1622 | 0.3556 | 0.4900 |
| 640x80 | 103842 | 0.1970 | 0.1629 | 0.2034 | 0.4648 | 0.5467 |
| 720x90 | 131222 | 0.3911 | 0.2988 | 0.4340 | 0.7895 | 0.6583 |
| 800x100 | 161802 | 0.5010 | 0.3654 | 0.5334 | 0.9776 | 0.7271 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 39 | 0.031 | 0.475 | 0.506 | 0.131 | 0.637 | 79.8243 † | 153.8423 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 64 | 28 | 0.574 | 0.337 | 0.911 | 0.103 | 1.014 | 157.8816 | 157.6126 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 48 | 1037 | 1.772 | 88.773 | 90.545 | 0.067 | 90.611 | 168.2586 | 168.2412 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 40 | 0.069 | 1.117 | 1.186 | 0.221 | 1.407 | 90.4948 † | 157.0766 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 49 | 23 | 1.026 | 0.597 | 1.623 | 0.138 | 1.761 | 159.7251 | 159.5143 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 49 | 1098 | 3.276 | 145.532 | 148.808 | 0.112 | 148.920 | 167.8893 | 167.8798 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 41 | 0.133 | 1.801 | 1.934 | 0.370 | 2.304 | 116.3935 † | 158.9330 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 42 | 20 | 1.516 | 0.962 | 2.477 | 0.195 | 2.673 | 158.7335 | 158.5167 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 68 | 1430 | 7.700 | 281.728 | 289.428 | 0.125 | 289.553 | 166.0497 | 166.0398 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 38 | 0.220 | 2.706 | 2.925 | 0.584 | 3.509 | 121.3734 † | 160.0720 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 42 | 21 | 2.540 | 1.600 | 4.139 | 0.259 | 4.399 | 159.3528 | 159.1097 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 61 | 1395 | 11.167 | 328.011 | 339.179 | 0.192 | 339.370 | 166.8367 | 166.8268 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 41 | 0.262 | 4.156 | 4.418 | 0.683 | 5.100 | 149.2628 † | 160.4125 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 43 | 21 | 3.634 | 2.294 | 5.927 | 0.361 | 6.288 | 159.5433 | 159.3841 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 74 | 1513 | 20.711 | 551.344 | 572.056 | 0.352 | 572.408 | 166.3448 | 166.3342 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 44 | 0.368 | 6.178 | 6.546 | 0.977 | 7.523 | 160.9545 | 160.9561 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 43 | 21 | 5.544 | 3.407 | 8.951 | 0.520 | 9.471 | 159.8295 | 159.7018 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 95 | 1839 | 35.996 | 901.170 | 937.166 | 0.458 | 937.624 | 165.9167 | 165.9055 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 40 | 0.508 | 7.880 | 8.388 | 1.064 | 9.452 | 112.1350 † | 161.3759 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 45 | 21 | 7.332 | 4.271 | 11.603 | 0.708 | 12.311 | 159.9037 | 159.7995 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 83 | 1616 | 41.141 | 883.505 | 924.645 | 0.589 | 925.234 | 164.8214 | 164.8077 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 40 | 1.010 | 15.643 | 16.653 | 1.611 | 18.264 | 87.1815 † | 161.8539 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 44 | 22 | 13.148 | 9.548 | 22.697 | 1.103 | 23.800 | 160.1387 | 160.0330 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 86 | 1687 | 71.359 | 1110.635 | 1181.995 | 0.976 | 1182.971 | 163.7022 | 163.6873 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 44 | 1.361 | 22.043 | 23.404 | 2.160 | 25.564 | 72.6993 † | 162.0649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 44 | 23 | 16.079 | 12.268 | 28.347 | 1.207 | 29.555 | 160.8509 | 160.7558 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 101 | 2181 | 103.762 | 1585.886 | 1689.649 | 1.299 | 1690.947 | 162.4788 | 162.4616 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

