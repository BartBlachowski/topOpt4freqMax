# Conference performance benchmark -- notes

Generated 2026-10-15T11:15:36+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w10_rel3e-2_gray2e-2`
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
| 160x20 | 0.147 | 0.059 | 0.088 | 0.209 | 0.860 |
| 240x30 | 0.213 | 0.112 | 0.101 | 0.352 | 2.033 |
| 320x40 | 0.270 | 0.191 | 0.079 | 0.474 | 3.649 |
| 400x50 | 0.470 | 0.339 | 0.131 | 0.807 | 5.629 |
| 480x60 | 0.609 | 0.495 | 0.114 | 1.247 | 8.882 |
| 560x70 | 0.864 | 0.734 | 0.130 | 1.925 | 13.518 |
| 640x80 | 1.181 | 1.034 | 0.147 | 1.990 | 16.690 |
| 720x90 | 2.154 | 1.977 | 0.177 | 3.245 | 31.141 |
| 800x100 | 2.786 | 2.580 | 0.206 | 4.130 | 41.359 |

The largest preparation value, 0.206 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.079 and 0.177 s, independent of the mesh.

### Native omega_1 values flagged

9 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 79.82, 79.82, 108.31 | 153.84 | 48.1% | 0.008 |
| Proposed | 240x30 | 88.82, 88.82, 105.30 | 157.08 | 43.5% | 0.008 |
| Proposed | 320x40 | 112.28, 112.28, 116.70 | 158.97 | 29.4% | 0.004 |
| Proposed | 400x50 | 117.40, 117.40, 119.61 | 160.15 | 26.7% | 0.003 |
| Proposed | 480x60 | 145.12, 145.82, 148.46 | 160.44 | 9.6% | 0.003 |
| Proposed | 560x70 | 147.34, 147.66, 149.62 | 161.02 | 8.5% | 0.003 |
| Proposed | 640x80 | 110.07, 110.22, 111.42 | 161.44 | 31.8% | 0.002 |
| Proposed | 720x90 | 85.35, 85.44, 86.21 | 162.09 | 47.3% | 0.003 |
| Proposed | 800x100 | 45.41, 45.41, 46.45 | 162.33 | 72.0% | 0.002 |

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
| Proposed | 5.582206e-05 | 1.1783 | 0.9870 | 9 |
| Yuksel | 1.687137e-04 | 1.0984 | 0.9822 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0152 | 0.0135 | 0.0179 | N/A | N/A |
| 240x30 | 14942 | 0.0402 | 0.0341 | 0.0452 | N/A | N/A |
| 320x40 | 26322 | 0.0765 | 0.0677 | 0.0895 | N/A | N/A |
| 400x50 | 40902 | 0.1245 | 0.1104 | 0.1446 | N/A | N/A |
| 480x60 | 58682 | 0.1785 | 0.1649 | 0.2141 | N/A | N/A |
| 560x70 | 79662 | 0.2585 | 0.2298 | 0.2971 | N/A | N/A |
| 640x80 | 103842 | 0.3504 | 0.3148 | 0.3970 | N/A | N/A |
| 720x90 | 131222 | 0.7005 | 0.5475 | 0.7880 | N/A | N/A |
| 800x100 | 161802 | 0.8884 | 0.6923 | 0.9960 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 39 | 0.059 | 0.592 | 0.651 | 0.209 | 0.860 | 79.8243 † | 153.8423 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 64 | 27 | 0.867 | 0.483 | 1.350 | 0.110 | 1.460 | 157.9114 | 157.6419 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 39 | 0.112 | 1.569 | 1.681 | 0.352 | 2.033 | 88.8167 † | 157.0760 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 49 | 22 | 1.669 | 0.995 | 2.663 | 0.174 | 2.838 | 159.7809 | 159.5534 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 39 | 0.191 | 2.984 | 3.175 | 0.474 | 3.649 | 112.2762 † | 158.9665 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 42 | 20 | 2.844 | 1.790 | 4.634 | 0.267 | 4.901 | 158.7335 | 158.5167 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 36 | 0.339 | 4.483 | 4.821 | 0.807 | 5.629 | 117.3958 † | 160.1471 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 42 | 20 | 4.635 | 2.893 | 7.528 | 0.459 | 7.987 | 159.2966 | 159.0627 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 40 | 0.495 | 7.140 | 7.634 | 1.247 | 8.882 | 145.1214 † | 160.4445 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 43 | 20 | 7.093 | 4.283 | 11.376 | 0.606 | 11.981 | 159.5075 | 159.3500 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 42 | 0.734 | 10.858 | 11.592 | 1.925 | 13.518 | 147.3423 † | 161.0230 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 41 | 20 | 9.421 | 5.942 | 15.363 | 0.870 | 16.233 | 159.8290 | 159.6892 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 39 | 1.034 | 13.666 | 14.700 | 1.990 | 16.690 | 110.0736 † | 161.4391 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 43 | 19 | 13.537 | 7.543 | 21.080 | 1.204 | 22.284 | 159.8601 | 159.7506 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 37 | 1.977 | 25.920 | 27.897 | 3.245 | 31.141 | 85.3506 † | 162.0912 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 41 | 20 | 22.447 | 15.761 | 38.208 | 1.892 | 40.100 | 160.1061 | 159.9948 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 39 | 2.580 | 34.649 | 37.229 | 4.130 | 41.359 | 45.4118 † | 162.3330 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 41 | 20 | 28.385 | 19.920 | 48.306 | 2.378 | 50.684 | 160.7860 | 160.6899 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

