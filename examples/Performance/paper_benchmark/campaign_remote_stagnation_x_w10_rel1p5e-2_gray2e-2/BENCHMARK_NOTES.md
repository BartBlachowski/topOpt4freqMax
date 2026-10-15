# Conference performance benchmark -- notes

Generated 2026-10-15T11:23:16+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w10_rel1p5e-2_gray2e-2`
- scientific evidence: **true**
- performance campaign: **true**
- resolutions: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100
- threads: 1
- timing schema: `conference_benchmark_timing/2`

## How to read the table

Count/time columns represent method-native computational stages and are not mathematically identical across methods. Total wall time is the common performance quantity.

Proposed: Count 1 = reference eigenanalysis solves (always 1, not an optimization iteration), Count 2 = SIMP iterations, Time 1 = that single eigenanalysis (K0/M0 assembly and the eigensolve, nothing else; solver preparation is in Other), Time 2 = SIMP. Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 iteration counts, Time 1 and Time 2 the corresponding stage times.

Stage time [s] = Time 1 + Time 2, computed before rounding. Other [s] is overhead_time_s: everything inside the timed solve but outside the two named stages, including setup, final modal analysis and result assembly. Other includes computations too. Total wall time [s] = Stage time + Other. Method-specific configuration dispatch is included only where it lies inside the recorded timer; Du-Olhoff configuration resolution precedes its timer.

omega_1 native is the first eigenfrequency of the converged design under the solver's own material model, which differs per method (Proposed: SIMP p = 3, E_min = 1e-9 E_0, linear mass, rho_min = 1e-9, design floor 0; Yuksel: SIMP p = 3, E_min = 1e-9 E_0, mass linear above rho = 0.1 and proportional to the sixth power of rho below it; Du-Olhoff: Pedersen stiffness, linear mass). omega_1 E1 is the same design under the frozen common evaluator E1 (SIMP p = 3, E_min = 1e-6 E_0, linear mass), classifier-selected structural mode, computed outside every timer; it is the only omega_1 that is comparable across methods. A dagger marks a native value that deviates from E1 by more than 5%: the native model then returns a localized mode of near-void elements, not the structural frequency.

### Proposed Stage 1: eigenanalysis versus preparation

Proposed Stage 1 preparation (initialization minus the reference eigenanalysis) is one-off setup and is reported in Other, never in Time 1. It is a fixed cost that does not scale with the mesh; the first measured row of a session also carries that session's first-call costs.

| Mesh | Initialization incl. eigenanalysis [s] | Time 1 = reference eigenanalysis [s] | Preparation [s] (in Other) | Other [s] | Total [s] |
|---|---|---|---|---|---|
| 160x20 | 0.146 | 0.059 | 0.088 | 0.208 | 0.876 |
| 240x30 | 0.212 | 0.112 | 0.100 | 0.348 | 2.116 |
| 320x40 | 0.273 | 0.191 | 0.081 | 0.570 | 3.978 |
| 400x50 | 0.471 | 0.341 | 0.130 | 0.886 | 6.105 |
| 480x60 | 0.604 | 0.492 | 0.112 | 1.386 | 9.527 |
| 560x70 | 0.860 | 0.731 | 0.129 | 1.687 | 13.598 |
| 640x80 | 1.180 | 1.028 | 0.152 | 2.001 | 18.481 |
| 720x90 | 2.147 | 1.972 | 0.175 | 2.663 | 35.629 |
| 800x100 | 2.790 | 2.584 | 0.206 | 3.479 | 48.004 |

The largest preparation value, 0.206 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.081 and 0.175 s, independent of the mesh.

### Native omega_1 values flagged

7 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 79.99, 79.99, 108.30 | 153.84 | 48.0% | 0.008 |
| Proposed | 240x30 | 92.07, 92.07, 105.01 | 157.07 | 41.4% | 0.009 |
| Proposed | 320x40 | 118.64, 118.64, 120.55 | 158.92 | 25.3% | 0.004 |
| Proposed | 400x50 | 124.01, 124.01, 124.68 | 160.05 | 22.5% | 0.003 |
| Proposed | 640x80 | 132.30, 132.64, 133.98 | 161.19 | 17.9% | 0.002 |
| Proposed | 720x90 | 87.03, 87.22, 88.17 | 161.62 | 46.2% | 0.002 |
| Proposed | 800x100 | 74.29, 74.58, 76.07 | 161.92 | 54.1% | 0.002 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 4.421823e-05 | 1.2091 | 0.9845 | 9 |
| Yuksel | 1.031567e-04 | 1.1580 | 0.9778 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0152 | 0.0136 | 0.0180 | N/A | N/A |
| 240x30 | 14942 | 0.0404 | 0.0341 | 0.0454 | N/A | N/A |
| 320x40 | 26322 | 0.0766 | 0.0677 | 0.0887 | N/A | N/A |
| 400x50 | 40902 | 0.1251 | 0.1110 | 0.1442 | N/A | N/A |
| 480x60 | 58682 | 0.1779 | 0.1655 | 0.2138 | N/A | N/A |
| 560x70 | 79662 | 0.2541 | 0.2354 | 0.2967 | N/A | N/A |
| 640x80 | 103842 | 0.3512 | 0.3124 | 0.3959 | N/A | N/A |
| 720x90 | 131222 | 0.7044 | 0.5491 | 0.7888 | N/A | N/A |
| 800x100 | 161802 | 0.8923 | 0.6983 | 0.9872 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 40 | 0.059 | 0.609 | 0.667 | 0.208 | 0.876 | 79.9875 † | 153.8395 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 64 | 29 | 0.870 | 0.521 | 1.392 | 0.111 | 1.503 | 157.8053 | 157.5836 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 41 | 0.112 | 1.655 | 1.767 | 0.348 | 2.116 | 92.0651 † | 157.0694 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 49 | 24 | 1.670 | 1.090 | 2.761 | 0.176 | 2.937 | 159.6723 | 159.4727 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 42 | 0.191 | 3.216 | 3.407 | 0.570 | 3.978 | 118.6431 † | 158.9164 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 44 | 21 | 2.980 | 1.862 | 4.843 | 0.269 | 5.112 | 158.7847 | 158.5616 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 39 | 0.341 | 4.878 | 5.219 | 0.886 | 6.105 | 124.0135 † | 160.0473 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 46 | 21 | 5.108 | 3.027 | 8.135 | 0.453 | 8.588 | 159.3314 | 159.1097 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 43 | 0.492 | 7.649 | 8.141 | 1.386 | 9.527 | 158.8525 | 160.3641 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 47 | 22 | 7.779 | 4.705 | 12.484 | 0.620 | 13.104 | 159.4871 | 159.3475 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 44 | 0.731 | 11.180 | 11.911 | 1.687 | 13.598 | 160.9545 | 160.9561 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 48 | 23 | 11.298 | 6.823 | 18.121 | 0.892 | 19.012 | 159.8153 | 159.6927 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 44 | 1.028 | 15.453 | 16.481 | 2.001 | 18.481 | 132.3045 † | 161.1894 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 47 | 23 | 14.683 | 9.105 | 23.788 | 1.185 | 24.974 | 159.9620 | 159.8554 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 44 | 1.972 | 30.993 | 32.966 | 2.663 | 35.629 | 87.0269 † | 161.6187 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 47 | 24 | 25.806 | 18.931 | 44.737 | 1.891 | 46.628 | 160.1720 | 160.0711 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 47 | 2.584 | 41.940 | 44.524 | 3.479 | 48.004 | 74.2852 † | 161.9159 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 57 | 24 | 39.801 | 23.694 | 63.494 | 2.323 | 65.817 | 159.9347 | 159.8660 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

