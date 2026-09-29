# Conference performance benchmark -- notes

Generated 2026-09-29T01:04:25+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_9mesh_r2_recompute_bio2heli`
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
| 160x20 | 0.170 | 0.075 | 0.095 | 0.379 | 2.945 |
| 240x30 | 0.212 | 0.112 | 0.100 | 0.572 | 14.234 |
| 320x40 | 0.274 | 0.196 | 0.078 | 0.725 | 22.940 |
| 400x50 | 0.461 | 0.342 | 0.119 | 0.914 | 32.644 |
| 480x60 | 0.635 | 0.508 | 0.127 | 1.311 | 55.108 |
| 560x70 | 0.890 | 0.751 | 0.139 | 1.929 | 90.363 |
| 640x80 | 1.186 | 1.030 | 0.155 | 2.435 | 140.055 |
| 720x90 | 2.141 | 1.972 | 0.169 | 4.252 | 324.563 |
| 800x100 | 2.777 | 2.582 | 0.195 | 5.448 | 468.419 |

The largest preparation value, 0.195 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.078 and 0.169 s, independent of the mesh.

### Native omega_1 values flagged

2 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 109.05, 109.49, 112.92 | 153.68 | 29.0% | 0.009 |
| Proposed | 240x30 | 108.78, 109.92, 117.83 | 157.64 | 31.0% | 0.007 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 112 | 376.623 | 3.3627 | 0.0566 | 0.0524 | 3.3054 | 21.22 | 0.1557 | NATIVE_CONVERGED |
| 240x30 | 104 | 690.332 | 6.6378 | 0.1277 | 0.1197 | 6.5089 | 18.93 | 0.3438 | NATIVE_CONVERGED |
| 320x40 | 116 | 1144.584 | 9.8671 | 0.2194 | 0.2050 | 9.6457 | 19.25 | 0.5011 | NATIVE_CONVERGED |
| 400x50 | 92 | 1245.350 | 13.5364 | 0.3796 | 0.3453 | 13.1521 | 19.84 | 0.6630 | NATIVE_CONVERGED |
| 480x60 | 106 | 2084.676 | 19.6668 | 0.5821 | 0.5326 | 19.0790 | 19.58 | 0.9742 | NATIVE_CONVERGED |
| 560x70 | 132 | 3609.694 | 27.3462 | 0.8411 | 0.7734 | 26.4982 | 18.66 | 1.4201 | NATIVE_CONVERGED |
| 640x80 | 151 | 6112.633 | 40.4810 | 1.1414 | 1.0495 | 39.3316 | 18.64 | 2.1105 | NATIVE_CONVERGED |
| 720x90 | 205 | 10150.421 | 49.5142 | 2.0271 | 1.9007 | 47.4761 | 18.47 | 2.5707 | NATIVE_CONVERGED |
| 800x100 | 238 | 16033.390 | 67.3672 | 2.5356 | 2.3746 | 64.8199 | 18.80 | 3.4482 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 2.248807e-05 | 1.4614 | 0.9692 | 9 |
| Yuksel | 3.856296e-07 | 1.9800 | 0.9926 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 2.787443e-02 | 1.1322 | 0.9285 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 1.813399e-03 | 0.9175 | 0.9861 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 3.569212e-06 | 1.1788 | 0.9888 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.322901e-06 | 1.1783 | 0.9875 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 1.890888e-03 | 0.9105 | 0.9860 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 7.170092e-05 | 0.9418 | 0.9856 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0233 | 0.0142 | 0.0187 | 0.0524 | 0.1557 |
| 240x30 | 14942 | 0.0574 | 0.0366 | 0.0479 | 0.1197 | 0.3438 |
| 320x40 | 26322 | 0.1064 | 0.0740 | 0.0939 | 0.2050 | 0.5011 |
| 400x50 | 40902 | 0.1725 | 0.1210 | 0.1492 | 0.3453 | 0.6630 |
| 480x60 | 58682 | 0.2433 | 0.1755 | 0.2189 | 0.5326 | 0.9742 |
| 560x70 | 79662 | 0.3425 | 0.2499 | 0.2961 | 0.7734 | 1.4201 |
| 640x80 | 103842 | 0.4420 | 0.3205 | 0.3900 | 1.0495 | 2.1105 |
| 720x90 | 131222 | 1.0719 | 0.5570 | 0.7855 | 1.9007 | 2.5707 |
| 800x100 | 161802 | 1.3951 | 0.7056 | 0.9893 | 2.3746 | 3.4482 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 107 | 0.075 | 2.490 | 2.565 | 0.379 | 2.945 | 109.0501 † | 153.6752 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 123 | 1.722 | 2.299 | 4.020 | 0.348 | 4.368 | 157.2784 | 157.1668 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 112 | 2377 | 6.339 | 370.203 | 376.542 | 0.081 | 376.623 | 168.3572 | 168.3205 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 236 | 0.112 | 13.550 | 13.662 | 0.572 | 14.234 | 108.7822 † | 157.6392 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 152 | 6.150 | 7.280 | 13.430 | 0.490 | 13.920 | 159.4915 | 159.4358 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 104 | 1969 | 13.276 | 676.923 | 690.200 | 0.133 | 690.332 | 167.3641 | 167.3555 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 207 | 0.196 | 22.018 | 22.215 | 0.725 | 22.940 | 158.7628 | 158.7632 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 320 | 18.653 | 30.048 | 48.701 | 0.857 | 49.558 | 160.7459 | 160.6896 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 116 | 2233 | 25.445 | 1118.902 | 1144.347 | 0.237 | 1144.584 | 166.2467 | 166.2378 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 182 | 0.342 | 31.388 | 31.730 | 0.914 | 32.644 | 159.5184 | 159.5186 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 417 | 38.116 | 62.214 | 100.330 | 1.324 | 101.653 | 160.0551 | 159.9684 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 92 | 1825 | 34.927 | 1209.997 | 1244.924 | 0.426 | 1245.350 | 166.3316 | 166.3225 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 219 | 0.508 | 53.289 | 53.797 | 1.311 | 55.108 | 160.2542 | 160.2544 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 586 | 615 | 102.851 | 134.606 | 237.456 | 1.804 | 239.261 | 160.5983 | 160.5506 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 106 | 2076 | 61.700 | 2022.371 | 2084.071 | 0.606 | 2084.676 | 166.0151 | 166.0053 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 256 | 0.751 | 87.682 | 88.434 | 1.929 | 90.363 | 160.7224 | 160.7225 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 833 | 771 | 208.125 | 228.301 | 436.426 | 2.634 | 439.060 | 160.3923 | 160.3432 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 132 | 2463 | 111.029 | 3497.765 | 3608.794 | 0.900 | 3609.694 | 165.8062 | 165.7958 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 309 | 1.030 | 136.589 | 137.620 | 2.435 | 140.055 | 160.8517 | 160.8518 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 1311 | 1763 | 420.124 | 687.486 | 1107.609 | 3.574 | 1111.183 | 160.8379 | 160.8160 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 151 | 2814 | 172.359 | 5939.064 | 6111.423 | 1.211 | 6112.633 | 165.6808 | 165.6698 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 297 | 1.972 | 318.339 | 320.311 | 4.252 | 324.563 | 161.0688 | 161.0688 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 1162 | 907 | 647.222 | 712.484 | 1359.707 | 5.834 | 1365.540 | 160.6227 | 160.5978 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 205 | 3786 | 415.546 | 9732.597 | 10148.144 | 2.277 | 10150.421 | 165.5386 | 165.5270 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 330 | 2.582 | 460.388 | 462.971 | 5.448 | 468.419 | 161.3649 | 161.3649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 1150 | 1171 | 811.423 | 1158.459 | 1969.882 | 7.298 | 1977.180 | 160.8856 | 160.8644 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 238 | 4474 | 603.473 | 15427.145 | 16030.618 | 2.772 | 16033.390 | 165.3011 | 165.2883 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

