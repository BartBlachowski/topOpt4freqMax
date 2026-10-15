# Conference performance benchmark -- notes

Generated 2026-10-15T15:15:54+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_relative_l2_change_no_olhoff_rel1e-2_gray1e-2`
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
| 160x20 | 0.098 | 0.026 | 0.072 | 0.123 | 0.504 |
| 240x30 | 0.130 | 0.057 | 0.073 | 0.175 | 1.031 |
| 320x40 | 0.196 | 0.109 | 0.088 | 0.344 | 1.960 |
| 400x50 | 0.285 | 0.189 | 0.097 | 0.578 | 2.915 |
| 480x60 | 0.352 | 0.258 | 0.094 | 0.773 | 4.809 |
| 560x70 | 0.485 | 0.380 | 0.105 | 1.447 | 10.384 |
| 640x80 | 0.639 | 0.519 | 0.120 | 1.175 | 9.032 |
| 720x90 | 1.273 | 1.146 | 0.127 | 1.535 | 17.729 |
| 800x100 | 1.478 | 1.318 | 0.160 | 2.027 | 25.410 |

The largest preparation value, 0.160 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.072 and 0.127 s, independent of the mesh.

### Native omega_1 values flagged

9 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 82.85, 82.85, 108.59 | 153.74 | 46.1% | 0.009 |
| Proposed | 240x30 | 89.26, 89.26, 107.03 | 157.12 | 43.2% | 0.008 |
| Proposed | 320x40 | 103.31, 103.31, 115.35 | 159.14 | 35.1% | 0.005 |
| Proposed | 400x50 | 109.03, 109.03, 117.57 | 160.47 | 32.1% | 0.004 |
| Proposed | 480x60 | 138.53, 138.93, 141.66 | 160.54 | 13.7% | 0.003 |
| Proposed | 560x70 | 126.76, 126.87, 128.32 | 161.31 | 21.4% | 0.003 |
| Proposed | 640x80 | 107.80, 107.93, 109.05 | 161.51 | 33.3% | 0.002 |
| Proposed | 720x90 | 86.51, 86.61, 87.38 | 162.00 | 46.6% | 0.002 |
| Proposed | 800x100 | 72.52, 72.71, 73.99 | 162.12 | 55.3% | 0.002 |

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
| Proposed | 2.231058e-05 | 1.2139 | 0.9755 | 9 |
| Yuksel | 9.617713e-05 | 1.1259 | 0.9721 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0107 | 0.0087 | 0.0115 | N/A | N/A |
| 240x30 | 14942 | 0.0235 | 0.0202 | 0.0259 | N/A | N/A |
| 320x40 | 26322 | 0.0443 | 0.0377 | 0.0488 | N/A | N/A |
| 400x50 | 40902 | 0.0693 | 0.0687 | 0.0730 | N/A | N/A |
| 480x60 | 58682 | 0.0994 | 0.0873 | 0.1111 | N/A | N/A |
| 560x70 | 79662 | 0.2313 | 0.1264 | 0.1557 | N/A | N/A |
| 640x80 | 103842 | 0.1931 | 0.1696 | 0.2323 | N/A | N/A |
| 720x90 | 131222 | 0.3960 | 0.3008 | 0.4213 | N/A | N/A |
| 800x100 | 161802 | 0.5131 | 0.3757 | 0.5325 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 33 | 0.026 | 0.355 | 0.381 | 0.123 | 0.504 | 82.8529 † | 153.7425 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 78 | 32 | 0.676 | 0.370 | 1.046 | 0.104 | 1.149 | 157.9278 | 157.7054 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 34 | 0.057 | 0.800 | 0.857 | 0.175 | 1.031 | 89.2612 † | 157.1164 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 57 | 26 | 1.153 | 0.674 | 1.827 | 0.135 | 1.962 | 159.7974 | 159.5674 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 34 | 0.109 | 1.507 | 1.615 | 0.344 | 1.960 | 103.3145 † | 159.1382 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 51 | 25 | 1.923 | 1.219 | 3.142 | 0.268 | 3.410 | 158.9864 | 158.7672 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 31 | 0.189 | 2.148 | 2.337 | 0.578 | 2.915 | 109.0343 † | 160.4677 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 55 | 26 | 3.781 | 1.898 | 5.679 | 0.296 | 5.975 | 159.4140 | 159.2400 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 38 | 0.258 | 3.778 | 4.036 | 0.773 | 4.809 | 138.5254 † | 160.5435 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 53 | 27 | 4.629 | 3.000 | 7.629 | 0.376 | 8.005 | 159.5540 | 159.4042 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 37 | 0.380 | 8.558 | 8.937 | 1.447 | 10.384 | 126.7556 † | 161.3081 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 58 | 27 | 7.333 | 4.203 | 11.536 | 0.503 | 12.039 | 159.7795 | 159.6701 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 38 | 0.519 | 7.337 | 7.856 | 1.175 | 9.032 | 107.7960 † | 161.5077 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 63 | 33 | 10.687 | 7.666 | 18.353 | 0.650 | 19.003 | 160.1130 | 160.0131 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 38 | 1.146 | 15.048 | 16.193 | 1.535 | 17.729 | 86.5099 † | 162.0034 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 58 | 32 | 17.446 | 13.480 | 30.926 | 0.968 | 31.894 | 160.2709 | 160.1802 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 43 | 1.318 | 22.065 | 23.383 | 2.027 | 25.410 | 72.5225 † | 162.1238 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 62 | 30 | 23.294 | 15.975 | 39.269 | 1.173 | 40.442 | 160.0754 | 160.0074 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

