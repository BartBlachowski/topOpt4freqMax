# Conference performance benchmark -- notes

Generated 2026-10-15T10:32:22+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w20_rel2e-2_gray2e-2`
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
| 160x20 | 0.147 | 0.058 | 0.089 | 0.224 | 1.028 |
| 240x30 | 0.214 | 0.112 | 0.101 | 0.432 | 2.610 |
| 320x40 | 0.283 | 0.200 | 0.082 | 0.593 | 4.815 |
| 400x50 | 0.465 | 0.338 | 0.127 | 0.868 | 7.357 |
| 480x60 | 0.616 | 0.504 | 0.112 | 0.782 | 10.664 |
| 560x70 | 0.941 | 0.784 | 0.157 | 0.955 | 15.958 |
| 640x80 | 1.191 | 1.044 | 0.147 | 1.950 | 21.248 |
| 720x90 | 2.191 | 2.007 | 0.184 | 2.693 | 41.003 |
| 800x100 | 2.859 | 2.632 | 0.226 | 3.574 | 55.742 |

The largest preparation value, 0.226 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.082 and 0.184 s, independent of the mesh.

### Native omega_1 values flagged

6 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 89.78, 89.78, 102.63 | 153.86 | 41.6% | 0.009 |
| Proposed | 240x30 | 107.24, 107.27, 111.92 | 157.07 | 31.7% | 0.008 |
| Proposed | 320x40 | 140.14, 141.43, 144.01 | 158.88 | 11.8% | 0.004 |
| Proposed | 400x50 | 151.79, 155.10, 155.42 | 159.90 | 5.1% | 0.003 |
| Proposed | 720x90 | 94.29, 94.65, 96.07 | 161.38 | 41.6% | 0.002 |
| Proposed | 800x100 | 108.59, 109.04, 110.52 | 161.68 | 32.8% | 0.002 |

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
| Proposed | 6.060838e-05 | 1.1938 | 0.9837 | 9 |
| Yuksel | 1.833255e-04 | 1.1250 | 0.9836 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0152 | 0.0137 | 0.0180 | N/A | N/A |
| 240x30 | 14942 | 0.0413 | 0.0348 | 0.0459 | N/A | N/A |
| 320x40 | 26322 | 0.0789 | 0.0696 | 0.0912 | N/A | N/A |
| 400x50 | 40902 | 0.1282 | 0.1167 | 0.1489 | N/A | N/A |
| 480x60 | 58682 | 0.1839 | 0.1712 | 0.2275 | N/A | N/A |
| 560x70 | 79662 | 0.2633 | 0.2397 | 0.3053 | N/A | N/A |
| 640x80 | 103842 | 0.3651 | 0.3246 | 0.4093 | N/A | N/A |
| 720x90 | 131222 | 0.7261 | 0.5656 | 0.8097 | N/A | N/A |
| 800x100 | 161802 | 0.9173 | 0.7159 | 1.0125 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 49 | 0.058 | 0.746 | 0.804 | 0.224 | 1.028 | 89.7845 † | 153.8557 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 84 | 37 | 1.148 | 0.665 | 1.813 | 0.113 | 1.926 | 157.8644 | 157.6701 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 50 | 0.112 | 2.066 | 2.178 | 0.432 | 2.610 | 107.2437 † | 157.0695 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 63 | 34 | 2.192 | 1.560 | 3.752 | 0.186 | 3.938 | 159.5695 | 159.4308 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 51 | 0.200 | 4.022 | 4.223 | 0.593 | 4.815 | 140.1380 † | 158.8754 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 57 | 30 | 3.969 | 2.735 | 6.704 | 0.266 | 6.970 | 159.2568 | 159.0402 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 48 | 0.338 | 6.152 | 6.490 | 0.868 | 7.357 | 151.7944 † | 159.8983 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 55 | 31 | 6.418 | 4.617 | 11.036 | 0.489 | 11.524 | 159.4853 | 159.3326 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 51 | 0.504 | 9.379 | 9.882 | 0.782 | 10.664 | 160.2841 | 160.2843 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 55 | 31 | 9.417 | 7.054 | 16.470 | 0.630 | 17.100 | 159.6054 | 159.4618 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 54 | 0.784 | 14.220 | 15.004 | 0.955 | 15.958 | 160.7932 | 160.7933 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 53 | 31 | 12.704 | 9.466 | 22.170 | 0.882 | 23.052 | 159.9037 | 159.7957 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 50 | 1.044 | 18.254 | 19.298 | 1.950 | 21.248 | 161.0242 | 161.0243 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 55 | 31 | 17.853 | 12.688 | 30.542 | 1.202 | 31.743 | 160.1282 | 160.0334 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 50 | 2.007 | 36.304 | 38.311 | 2.693 | 41.003 | 94.2876 † | 161.3786 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 54 | 31 | 30.542 | 25.100 | 55.642 | 1.909 | 57.551 | 160.2837 | 160.1915 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 54 | 2.632 | 49.535 | 52.168 | 3.574 | 55.742 | 108.5895 † | 161.6818 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 57 | 32 | 40.807 | 32.401 | 73.208 | 2.378 | 75.586 | 160.1640 | 160.0898 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

