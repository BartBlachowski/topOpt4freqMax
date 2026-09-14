# Conference performance benchmark -- notes

Generated 2026-09-14T17:03:53+02:00 by `examples/Performance/compose_nine_mesh_comparison.m` from two recorded campaigns.

## Composition

Rows are composed from two recorded campaigns run on the same host (PMS.local, MATLAB 2025b, Apple Accelerate BLAS (ILP64), 1 thread): Proposed and Yuksel from campaign_9mesh_r2 (generated 2026-09-11T23:51:04+02:00), the Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) from nine_mesh_pedersen_b21483b (generated 2026-09-14T05:34:28+02:00). Nothing was re-solved or re-timed. The Proposed/Yuksel rows and the Du-Olhoff rows were timed in separate MATLAB sessions, not interleaved mesh by mesh in one session. The historical "Du-Olhoff reconstruction (M4)" rows of campaign_9mesh_r2 are not used. The Proposed rows are re-accounted to timing schema 2 from their recorded timers: Time 1 is the reference eigenanalysis alone and the solver preparation is in Other, as for the other two methods.

| Method(s) | Source campaign | Campaign generated | Repository HEAD | Records SHA-256 |
|---|---|---|---|---|
| Proposed; Yuksel | `examples/Performance/conference_benchmark/campaign_9mesh_r2` | 2026-09-11T23:51:04+02:00 | `bba45e7` (dirty) | `873125858df02664d9a7d37bd09e4e9b0b1ccff5c37940d088f9b1e424921fae` |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b` | 2026-09-14T05:34:28+02:00 | `b21483b` (dirty) | `261bf8fc94a1efb7da223aae64ac72da336a4d8f48341f0d8bf79b0a4d2abe82` |


- run label: `nine_mesh_comparison_pedersen_b21483b`
- scientific evidence: **true**
- performance campaign: **true**
- resolutions: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100
- threads: 1
- timing schema: `conference_benchmark_timing/2`

## How to read the table

Count/time columns represent method-native computational stages and are not mathematically identical across methods. Total wall time is the common performance quantity.

Proposed: Count 1 = reference eigenanalysis solves (always 1, not an optimization iteration), Count 2 = SIMP iterations, Time 1 = that single eigenanalysis (K0/M0 assembly and the eigensolve, nothing else; solver preparation is in Other), Time 2 = SIMP. Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 iteration counts, Time 1 and Time 2 the corresponding stage times. Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box): Count 1 = outer iterations, Count 2 = cumulative nested MMA iterations, Time 1 = outer work excluding the nested MMA solve (FE assembly, the eigenproblem, sensitivities, filtering, the design update), Time 2 = nested MMA total. The two counts are never added.

Other [s] is overhead_time_s: everything inside the timed solve but outside the two named stages -- configuration dispatch, model build, filter preparation, the final modal analysis of the converged design and result assembly. It is defined identically for the three methods, and on every row Time 1 + Time 2 + Other = Total.

omega_1 native is the first eigenfrequency of the converged design under the solver's own material model, which differs per method (Proposed: SIMP p = 3, E_min = 1e-9 E_0, linear mass, rho_min = 1e-9, design floor 0; Yuksel: SIMP p = 3, E_min = 1e-9 E_0, mass linear above rho = 0.1 and proportional to the sixth power of rho below it; Du-Olhoff: Pedersen stiffness, linear mass). omega_1 E1 is the same design under the frozen common evaluator E1 (SIMP p = 3, E_min = 1e-6 E_0, linear mass), classifier-selected structural mode, computed outside every timer; it is the only omega_1 that is comparable across methods. A dagger marks a native value that deviates from E1 by more than 5%: the native model then returns a localized mode of near-void elements, not the structural frequency.

### Proposed Stage 1: eigenanalysis versus preparation

Proposed Stage 1 preparation (initialization minus the reference eigenanalysis) is one-off setup and is reported in Other, never in Time 1. It is a fixed cost that does not scale with the mesh; the first measured row of a session also carries that session's first-call costs.

| Mesh | Initialization incl. eigenanalysis [s] | Time 1 = reference eigenanalysis [s] | Preparation [s] (in Other) | Other [s] | Total [s] |
|---|---|---|---|---|---|
| 160x20 | 0.434 | 0.028 | 0.406 | 0.581 | 1.951 |
| 240x30 | 0.259 | 0.061 | 0.198 | 0.449 | 6.934 |
| 320x40 | 0.267 | 0.109 | 0.158 | 0.551 | 11.304 |
| 400x50 | 0.340 | 0.187 | 0.153 | 0.607 | 15.734 |
| 480x60 | 0.430 | 0.263 | 0.166 | 0.806 | 27.111 |
| 560x70 | 0.530 | 0.362 | 0.168 | 1.041 | 44.268 |
| 640x80 | 0.677 | 0.495 | 0.182 | 1.319 | 70.696 |
| 720x90 | 1.124 | 0.938 | 0.186 | 2.142 | 166.853 |
| 800x100 | 1.459 | 1.262 | 0.197 | 2.765 | 243.223 |

The largest preparation value, 0.406 s at 160x20, is the first measured row of its session; the other 8 rows lie between 0.153 and 0.198 s, independent of the mesh.

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
| 160x20 | 121 | 223.920 | 1.8506 | 0.0354 | 0.0324 | 1.8147 | 19.58 | 0.0927 | NATIVE_CONVERGED |
| 240x30 | 111 | 354.944 | 3.1977 | 0.0670 | 0.0626 | 3.1300 | 18.71 | 0.1673 | NATIVE_CONVERGED |
| 320x40 | 101 | 427.204 | 4.2297 | 0.1159 | 0.1086 | 4.1126 | 19.81 | 0.2076 | NATIVE_CONVERGED |
| 400x50 | 93 | 570.098 | 6.1301 | 0.1863 | 0.1764 | 5.9416 | 20.57 | 0.2888 | NATIVE_CONVERGED |
| 480x60 | 112 | 981.385 | 8.7624 | 0.2867 | 0.2725 | 8.4728 | 19.08 | 0.4441 | NATIVE_CONVERGED |
| 560x70 | 130 | 1425.047 | 10.9619 | 0.3930 | 0.3737 | 10.5654 | 18.24 | 0.5793 | NATIVE_CONVERGED |
| 640x80 | 156 | 2215.820 | 14.2040 | 0.5206 | 0.4939 | 13.6795 | 18.38 | 0.7441 | NATIVE_CONVERGED |
| 720x90 | 204 | 3537.510 | 17.3407 | 0.9283 | 0.8925 | 16.4074 | 18.39 | 0.8921 | NATIVE_CONVERGED |
| 800x100 | 246 | 4901.034 | 19.9229 | 1.1584 | 1.1143 | 18.7591 | 18.90 | 0.9924 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 1.870042e-05 | 1.4134 | 0.9616 | 9 |
| Yuksel | 1.175524e-06 | 1.8157 | 0.9775 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 6.488706e-02 | 0.9608 | 0.9202 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 3.811912e-03 | 0.7549 | 0.9922 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 4.473465e-06 | 1.0868 | 0.9831 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.672673e-06 | 1.1008 | 0.9835 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 4.128173e-03 | 0.7434 | 0.9926 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.810479e-04 | 0.7613 | 0.9863 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0125 | 0.0122 | 0.0192 | 0.0324 | 0.0927 |
| 240x30 | 14942 | 0.0272 | 0.0242 | 0.0325 | 0.0626 | 0.1673 |
| 320x40 | 26322 | 0.0514 | 0.0412 | 0.0520 | 0.1086 | 0.2076 |
| 400x50 | 40902 | 0.0821 | 0.0624 | 0.0796 | 0.1764 | 0.2888 |
| 480x60 | 58682 | 0.1189 | 0.0898 | 0.1122 | 0.2725 | 0.4441 |
| 560x70 | 79662 | 0.1674 | 0.1252 | 0.1526 | 0.3737 | 0.5793 |
| 640x80 | 103842 | 0.2229 | 0.1666 | 0.2001 | 0.4939 | 0.7441 |
| 720x90 | 131222 | 0.5514 | 0.2917 | 0.4092 | 0.8925 | 0.8921 |
| 800x100 | 161802 | 0.7248 | 0.3633 | 0.5103 | 1.1143 | 0.9924 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Other [s] | Total [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 107 | 0.028 | 1.342 | 0.581 | 1.951 | 109.0501 † | 153.6752 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 123 | 1.475 | 2.360 | 0.454 | 4.289 | 157.2784 | 157.1668 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 121 | 2369 | 4.289 | 219.581 | 0.050 | 223.920 | 169.2106 | 169.1972 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 236 | 0.061 | 6.424 | 0.449 | 6.934 | 108.7822 † | 157.6392 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 152 | 4.065 | 4.935 | 0.393 | 9.393 | 159.4915 | 159.4358 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 111 | 2077 | 7.438 | 347.426 | 0.080 | 354.944 | 167.3424 | 167.3335 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 207 | 0.109 | 10.644 | 0.551 | 11.304 | 158.7628 | 158.7632 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 320 | 10.376 | 16.639 | 0.560 | 27.574 | 160.7459 | 160.6896 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 101 | 2001 | 11.701 | 415.372 | 0.131 | 427.204 | 165.8568 | 165.8475 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 182 | 0.187 | 14.940 | 0.607 | 15.734 | 159.5184 | 159.5186 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 417 | 19.654 | 33.192 | 0.817 | 53.663 | 160.0551 | 159.9684 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 93 | 1913 | 17.327 | 552.569 | 0.203 | 570.098 | 166.4552 | 166.4458 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 219 | 0.263 | 26.042 | 0.806 | 27.111 | 160.2542 | 160.2544 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 586 | 615 | 52.611 | 69.020 | 1.047 | 122.678 | 160.5983 | 160.5506 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 112 | 2137 | 32.112 | 948.950 | 0.323 | 981.385 | 166.0093 | 165.9994 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 256 | 0.362 | 42.866 | 1.041 | 44.268 | 160.7224 | 160.7225 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 833 | 771 | 104.307 | 117.670 | 1.567 | 223.543 | 160.3923 | 160.3432 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 130 | 2371 | 51.095 | 1373.502 | 0.450 | 1425.047 | 165.8101 | 165.7996 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 309 | 0.495 | 68.882 | 1.319 | 70.696 | 160.8517 | 160.8518 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 1542 | 2000 | 256.822 | 400.200 | 1.897 | 658.919 | 160.8751 | 160.8534 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 156 | 2868 | 81.212 | 2133.997 | 0.610 | 2215.820 | 165.6500 | 165.6390 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 297 | 0.938 | 163.773 | 2.142 | 166.853 | 161.0688 | 161.0688 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 1162 | 907 | 339.007 | 371.117 | 2.925 | 713.049 | 160.6227 | 160.5978 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 204 | 3752 | 189.373 | 3347.104 | 1.032 | 3537.510 | 165.4234 | 165.4113 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 330 | 1.262 | 239.197 | 2.765 | 243.223 | 161.3649 | 161.3649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 1150 | 1171 | 417.753 | 597.601 | 3.584 | 1018.938 | 160.8856 | 160.8644 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 246 | 4650 | 284.958 | 4614.751 | 1.325 | 4901.034 | 165.4322 | 165.4190 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

