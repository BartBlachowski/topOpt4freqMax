# Conference performance benchmark -- notes

Generated 2026-10-07T13:20:13+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_convergence_corrected`
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
| 160x20 | 0.452 | 0.029 | 0.424 | 0.597 | 1.072 |
| 240x30 | 0.202 | 0.057 | 0.145 | 0.418 | 1.466 |
| 320x40 | 0.263 | 0.113 | 0.150 | 0.611 | 2.484 |
| 400x50 | 0.407 | 0.220 | 0.187 | 1.035 | 4.267 |
| 480x60 | 0.454 | 0.276 | 0.177 | 0.936 | 7.531 |
| 560x70 | 0.568 | 0.379 | 0.189 | 1.069 | 11.695 |
| 640x80 | 0.723 | 0.484 | 0.240 | 1.907 | 12.948 |
| 720x90 | 1.163 | 0.967 | 0.196 | 2.577 | 29.722 |
| 800x100 | 1.619 | 1.339 | 0.280 | 3.557 | 47.417 |

The largest preparation value, 0.424 s at 160x20, is the first measured row of its session; the other 8 rows lie between 0.145 and 0.280 s, independent of the mesh.

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
| 160x20 | 58 | 103.801 | 1.7897 | 0.0339 | 0.0313 | 1.7540 | 20.88 | 0.0840 | NATIVE_CONVERGED |
| 240x30 | 52 | 138.408 | 2.6617 | 0.0721 | 0.0674 | 2.5870 | 20.65 | 0.1253 | NATIVE_CONVERGED |
| 320x40 | 67 | 319.337 | 4.7662 | 0.1320 | 0.1236 | 4.6314 | 21.09 | 0.2196 | NATIVE_CONVERGED |
| 400x50 | 62 | 359.931 | 5.8053 | 0.2072 | 0.1966 | 5.5934 | 21.76 | 0.2571 | NATIVE_CONVERGED |
| 480x60 | 76 | 616.932 | 8.1175 | 0.3182 | 0.3027 | 7.7943 | 19.63 | 0.3970 | NATIVE_CONVERGED |
| 560x70 | 90 | 879.302 | 9.7700 | 0.4231 | 0.4022 | 9.3412 | 18.54 | 0.5037 | NATIVE_CONVERGED |
| 640x80 | 38 | 302.010 | 7.9476 | 0.5288 | 0.5017 | 7.4018 | 20.84 | 0.3551 | NATIVE_CONVERGED |
| 720x90 | 33 | 339.899 | 10.3000 | 1.0348 | 0.9955 | 9.2309 | 21.55 | 0.4284 | NATIVE_CONVERGED |
| 800x100 | 37 | 440.746 | 11.9120 | 1.2173 | 1.1699 | 10.6563 | 22.05 | 0.4832 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 4.799538e-05 | 1.1796 | 0.9374 | 9 |
| Yuksel | 4.231612e-06 | 1.5681 | 0.9577 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 3.045897e+00 | 0.4651 | 0.5619 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 1.594605e-02 | 0.5907 | 0.9592 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 4.008209e-06 | 1.1046 | 0.9867 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.361258e-06 | 1.1166 | 0.9866 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 1.980293e-02 | 0.5637 | 0.9464 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 9.786755e-04 | 0.5612 | 0.9171 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0139 | 0.0166 | 0.0184 | 0.0313 | 0.0840 |
| 240x30 | 14942 | 0.0275 | 0.0242 | 0.0332 | 0.0674 | 0.1253 |
| 320x40 | 26322 | 0.0503 | 0.0431 | 0.0592 | 0.1236 | 0.2196 |
| 400x50 | 40902 | 0.0941 | 0.0717 | 0.0924 | 0.1966 | 0.2571 |
| 480x60 | 58682 | 0.1264 | 0.0974 | 0.1264 | 0.3027 | 0.3970 |
| 560x70 | 79662 | 0.1798 | 0.1329 | 0.1659 | 0.4022 | 0.5037 |
| 640x80 | 103842 | 0.2246 | 0.1704 | 0.2124 | 0.5017 | 0.3551 |
| 720x90 | 131222 | 0.5817 | 0.3070 | 0.4472 | 0.9955 | 0.4284 |
| 800x100 | 161802 | 0.7593 | 0.3762 | 0.5473 | 1.1699 | 0.4832 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 32 | 0.029 | 0.446 | 0.475 | 0.597 | 1.072 | 83.8478 † | 153.6910 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 56 | 50 | 0.930 | 0.922 | 1.853 | 0.573 | 2.425 | 157.4248 | 157.2184 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 58 | 1211 | 1.966 | 101.735 | 103.701 | 0.100 | 103.801 | 168.6555 | 168.6253 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 36 | 0.057 | 0.991 | 1.048 | 0.418 | 1.466 | 86.6332 † | 157.0915 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 54 | 35 | 1.305 | 1.162 | 2.467 | 0.434 | 2.901 | 159.4154 | 159.2583 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 52 | 1074 | 3.747 | 134.525 | 138.271 | 0.136 | 138.408 | 167.4134 | 167.4044 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 35 | 0.113 | 1.761 | 1.873 | 0.611 | 2.484 | 105.2046 † | 159.0873 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 72 | 97 | 3.103 | 5.741 | 8.844 | 0.582 | 9.425 | 160.9092 | 160.8408 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 67 | 1413 | 8.842 | 310.302 | 319.144 | 0.192 | 319.337 | 166.0838 | 166.0739 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 32 | 0.220 | 3.012 | 3.232 | 1.035 | 4.267 | 110.8097 † | 160.3766 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 128 | 128 | 9.178 | 11.832 | 21.010 | 0.967 | 21.977 | 159.9431 | 159.8050 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 62 | 1349 | 12.848 | 346.791 | 359.639 | 0.292 | 359.931 | 166.8450 | 166.8348 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 50 | 0.276 | 6.318 | 6.595 | 0.936 | 7.531 | 160.2877 | 160.2878 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 136 | 170 | 13.249 | 21.486 | 34.735 | 1.298 | 36.033 | 160.2625 | 160.1911 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 76 | 1492 | 24.186 | 592.369 | 616.554 | 0.378 | 616.932 | 166.2460 | 166.2356 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 57 | 0.379 | 10.246 | 10.625 | 1.069 | 11.695 | 160.7771 | 160.7772 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 195 | 187 | 25.912 | 31.017 | 56.929 | 1.470 | 58.400 | 160.1637 | 160.0986 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 90 | 1669 | 38.082 | 840.706 | 878.788 | 0.515 | 879.302 | 165.9822 | 165.9708 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 47 | 0.484 | 10.558 | 11.041 | 1.907 | 12.948 | 161.0925 | 161.0929 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 180 | 225 | 30.673 | 47.785 | 78.458 | 1.884 | 80.342 | 160.5510 | 160.5098 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 38 | 792 | 20.095 | 281.270 | 301.366 | 0.645 | 302.010 | 156.2801 | 156.2656 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 45 | 0.967 | 26.178 | 27.145 | 2.577 | 29.722 | 87.3049 † | 161.5701 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 252 | 300 | 77.372 | 134.162 | 211.534 | 3.239 | 214.773 | 160.8630 | 160.8257 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 33 | 711 | 34.147 | 304.620 | 338.768 | 1.131 | 339.899 | 154.3480 | 154.3293 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 56 | 1.339 | 42.521 | 43.860 | 3.557 | 47.417 | 140.1271 † | 161.6303 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 251 | 344 | 94.416 | 188.287 | 282.703 | 3.890 | 286.593 | 161.0426 | 161.0230 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 37 | 816 | 45.039 | 394.285 | 439.323 | 1.423 | 440.746 | 153.2031 | 153.1844 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

