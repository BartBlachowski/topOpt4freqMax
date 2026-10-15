# Conference performance benchmark -- notes

Generated 2026-10-15T10:23:35+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w15_rel2e-2_gray2e-2`
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
| 160x20 | 0.147 | 0.058 | 0.088 | 0.210 | 0.947 |
| 240x30 | 0.217 | 0.115 | 0.102 | 0.439 | 2.420 |
| 320x40 | 0.272 | 0.193 | 0.080 | 0.927 | 4.772 |
| 400x50 | 0.480 | 0.351 | 0.128 | 1.979 | 7.876 |
| 480x60 | 0.631 | 0.516 | 0.115 | 1.414 | 10.472 |
| 560x70 | 0.895 | 0.758 | 0.136 | 1.168 | 14.910 |
| 640x80 | 1.199 | 1.043 | 0.155 | 2.057 | 19.658 |
| 720x90 | 2.199 | 2.020 | 0.178 | 2.716 | 37.373 |
| 800x100 | 2.866 | 2.657 | 0.209 | 3.549 | 51.304 |

The largest preparation value, 0.209 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.080 and 0.178 s, independent of the mesh.

### Native omega_1 values flagged

7 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 80.20, 80.20, 108.23 | 153.85 | 47.9% | 0.009 |
| Proposed | 240x30 | 99.85, 99.85, 106.94 | 157.07 | 36.4% | 0.008 |
| Proposed | 320x40 | 132.60, 133.04, 133.04 | 158.89 | 16.5% | 0.004 |
| Proposed | 400x50 | 145.06, 145.06, 145.32 | 159.96 | 9.3% | 0.003 |
| Proposed | 640x80 | 141.83, 142.21, 143.50 | 161.15 | 12.0% | 0.002 |
| Proposed | 720x90 | 87.30, 87.51, 88.49 | 161.57 | 46.0% | 0.002 |
| Proposed | 800x100 | 76.73, 77.07, 78.61 | 161.84 | 52.6% | 0.002 |

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
| Proposed | 6.139580e-05 | 1.1876 | 0.9870 | 9 |
| Yuksel | 1.643851e-04 | 1.1232 | 0.9819 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0154 | 0.0138 | 0.0181 | N/A | N/A |
| 240x30 | 14942 | 0.0415 | 0.0350 | 0.0464 | N/A | N/A |
| 320x40 | 26322 | 0.0794 | 0.0701 | 0.0922 | N/A | N/A |
| 400x50 | 40902 | 0.1290 | 0.1161 | 0.1507 | N/A | N/A |
| 480x60 | 58682 | 0.1857 | 0.1714 | 0.2215 | N/A | N/A |
| 560x70 | 79662 | 0.2650 | 0.2414 | 0.3093 | N/A | N/A |
| 640x80 | 103842 | 0.3680 | 0.3286 | 0.4146 | N/A | N/A |
| 720x90 | 131222 | 0.7253 | 0.5674 | 0.8160 | N/A | N/A |
| 800x100 | 161802 | 0.9204 | 0.7205 | 1.0252 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 44 | 0.058 | 0.678 | 0.737 | 0.210 | 0.947 | 80.2031 † | 153.8534 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 75 | 33 | 1.033 | 0.597 | 1.629 | 0.114 | 1.743 | 157.8827 | 157.6466 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 45 | 0.115 | 1.866 | 1.981 | 0.439 | 2.420 | 99.8515 † | 157.0733 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 57 | 28 | 1.998 | 1.299 | 3.296 | 0.178 | 3.475 | 159.6790 | 159.5003 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 46 | 0.193 | 3.652 | 3.845 | 0.927 | 4.772 | 132.6011 † | 158.8875 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 49 | 25 | 3.435 | 2.304 | 5.739 | 0.272 | 6.011 | 158.9897 | 158.7837 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 43 | 0.351 | 5.546 | 5.897 | 1.979 | 7.876 | 145.0574 † | 159.9601 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 48 | 26 | 5.572 | 3.917 | 9.489 | 0.453 | 9.942 | 159.4699 | 159.2719 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 46 | 0.516 | 8.542 | 9.058 | 1.414 | 10.472 | 160.3236 | 160.3248 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 49 | 26 | 8.398 | 5.758 | 14.156 | 0.634 | 14.790 | 159.5763 | 159.4195 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 49 | 0.758 | 12.983 | 13.742 | 1.168 | 14.910 | 160.8480 | 160.8481 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 48 | 26 | 11.588 | 8.043 | 19.630 | 0.889 | 20.519 | 159.8809 | 159.7599 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 45 | 1.043 | 16.558 | 17.601 | 2.057 | 19.658 | 141.8267 † | 161.1510 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 50 | 26 | 16.428 | 10.780 | 27.208 | 1.212 | 28.420 | 160.0578 | 159.9427 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 45 | 2.020 | 32.637 | 34.657 | 2.716 | 37.373 | 87.3049 † | 161.5701 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 49 | 26 | 27.804 | 21.217 | 49.020 | 1.906 | 50.927 | 160.2222 | 160.1166 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 49 | 2.657 | 45.097 | 47.755 | 3.549 | 51.304 | 76.7318 † | 161.8404 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 51 | 27 | 36.744 | 27.681 | 64.425 | 2.410 | 66.835 | 160.0750 | 159.9962 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

