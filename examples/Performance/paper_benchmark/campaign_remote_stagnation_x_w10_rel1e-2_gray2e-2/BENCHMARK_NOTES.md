# Conference performance benchmark -- notes

Generated 2026-10-15T11:08:41+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w10_rel1e-2_gray2e-2`
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
| 160x20 | 0.147 | 0.059 | 0.088 | 0.212 | 0.930 |
| 240x30 | 0.212 | 0.113 | 0.098 | 0.349 | 2.300 |
| 320x40 | 0.272 | 0.193 | 0.079 | 0.538 | 4.232 |
| 400x50 | 0.469 | 0.338 | 0.131 | 1.071 | 6.676 |
| 480x60 | 0.616 | 0.502 | 0.115 | 0.941 | 10.280 |
| 560x70 | 0.983 | 0.828 | 0.155 | 1.455 | 14.680 |
| 640x80 | 1.207 | 1.045 | 0.163 | 2.015 | 20.607 |
| 720x90 | 2.204 | 2.018 | 0.186 | 2.738 | 39.548 |
| 800x100 | 2.974 | 2.714 | 0.260 | 3.573 | 54.777 |

The largest preparation value, 0.260 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.079 and 0.186 s, independent of the mesh.

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

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 4.074830e-05 | 1.2260 | 0.9833 | 9 |
| Yuksel | 6.721577e-05 | 1.2189 | 0.9855 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0153 | 0.0137 | 0.0181 | N/A | N/A |
| 240x30 | 14942 | 0.0418 | 0.0351 | 0.0465 | N/A | N/A |
| 320x40 | 26322 | 0.0796 | 0.0703 | 0.0916 | N/A | N/A |
| 400x50 | 40902 | 0.1285 | 0.1154 | 0.1491 | N/A | N/A |
| 480x60 | 58682 | 0.1841 | 0.1708 | 0.2206 | N/A | N/A |
| 560x70 | 79662 | 0.2638 | 0.2406 | 0.3047 | N/A | N/A |
| 640x80 | 103842 | 0.3656 | 0.3252 | 0.4103 | N/A | N/A |
| 720x90 | 131222 | 0.7248 | 0.5660 | 0.8112 | N/A | N/A |
| 800x100 | 161802 | 0.9149 | 0.7186 | 0.9867 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 43 | 0.059 | 0.659 | 0.718 | 0.212 | 0.930 | 80.1109 † | 153.8584 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 64 | 32 | 0.879 | 0.579 | 1.459 | 0.113 | 1.571 | 157.6959 | 157.4969 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 44 | 0.113 | 1.837 | 1.951 | 0.349 | 2.300 | 98.2247 † | 157.0739 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 49 | 28 | 1.720 | 1.302 | 3.021 | 0.182 | 3.203 | 159.5223 | 159.3340 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 44 | 0.193 | 3.501 | 3.694 | 0.538 | 4.232 | 124.8182 † | 158.8966 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 50 | 25 | 3.515 | 2.291 | 5.806 | 0.273 | 6.078 | 158.9901 | 158.7726 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 41 | 0.338 | 5.267 | 5.605 | 1.071 | 6.676 | 131.9135 † | 159.9995 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 55 | 26 | 6.345 | 3.877 | 10.222 | 0.456 | 10.678 | 159.4140 | 159.2400 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 48 | 0.502 | 8.837 | 9.338 | 0.941 | 10.280 | 160.3069 | 160.3072 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 53 | 27 | 9.052 | 5.957 | 15.010 | 0.627 | 15.636 | 159.5540 | 159.4042 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 47 | 0.828 | 12.397 | 13.225 | 1.455 | 14.680 | 160.8840 | 160.8842 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 58 | 27 | 13.955 | 8.227 | 22.181 | 0.888 | 23.069 | 159.7795 | 159.6701 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 48 | 1.045 | 17.547 | 18.592 | 2.015 | 20.607 | 161.0677 | 161.0680 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 63 | 33 | 20.489 | 13.538 | 34.028 | 1.194 | 35.221 | 160.1130 | 160.0131 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 48 | 2.018 | 34.791 | 36.810 | 2.738 | 39.548 | 88.8634 † | 161.4470 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 58 | 32 | 32.828 | 25.958 | 58.786 | 1.909 | 60.695 | 160.2709 | 160.1802 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 53 | 2.714 | 48.490 | 51.204 | 3.573 | 54.777 | 97.5155 † | 161.7104 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 62 | 30 | 44.553 | 29.601 | 74.154 | 2.308 | 76.462 | 160.0754 | 160.0074 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

