# Conference performance benchmark -- notes

Generated 2026-10-15T09:38:41+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_stagnation_x_no_olhoff_rel1e-2_gray1e-2_audited_4`
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
| 160x20 | 0.126 | 0.030 | 0.096 | 0.146 | 0.649 |
| 240x30 | 0.134 | 0.062 | 0.072 | 0.227 | 1.233 |
| 320x40 | 0.189 | 0.110 | 0.080 | 0.343 | 2.241 |
| 400x50 | 0.282 | 0.193 | 0.089 | 0.548 | 3.442 |
| 480x60 | 0.402 | 0.288 | 0.115 | 0.761 | 5.884 |
| 560x70 | 0.484 | 0.375 | 0.108 | 0.924 | 7.438 |
| 640x80 | 0.617 | 0.497 | 0.120 | 1.010 | 9.395 |
| 720x90 | 1.121 | 0.986 | 0.135 | 1.617 | 18.201 |
| 800x100 | 1.923 | 1.747 | 0.176 | 2.076 | 26.111 |

The largest preparation value, 0.176 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.072 and 0.135 s, independent of the mesh.

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

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 6.060673e-05 | 1.1221 | 0.9761 | 9 |
| Yuksel | 1.349003e-04 | 1.0659 | 0.9636 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0121 | 0.0087 | 0.0114 | N/A | N/A |
| 240x30 | 14942 | 0.0236 | 0.0195 | 0.0250 | N/A | N/A |
| 320x40 | 26322 | 0.0436 | 0.0357 | 0.0463 | N/A | N/A |
| 400x50 | 40902 | 0.0711 | 0.0632 | 0.0901 | N/A | N/A |
| 480x60 | 58682 | 0.1179 | 0.0861 | 0.1086 | N/A | N/A |
| 560x70 | 79662 | 0.1395 | 0.1164 | 0.1535 | N/A | N/A |
| 640x80 | 103842 | 0.1972 | 0.1596 | 0.2037 | N/A | N/A |
| 720x90 | 131222 | 0.3900 | 0.3024 | 0.4700 | N/A | N/A |
| 800x100 | 161802 | 0.5065 | 0.3633 | 0.5325 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 39 | 0.030 | 0.473 | 0.503 | 0.146 | 0.649 | 79.8243 † | 153.8423 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 64 | 28 | 0.556 | 0.319 | 0.875 | 0.103 | 0.978 | 157.8816 | 157.6126 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 40 | 0.062 | 0.944 | 1.006 | 0.227 | 1.233 | 90.4948 † | 157.0766 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 49 | 23 | 0.956 | 0.576 | 1.532 | 0.134 | 1.666 | 159.7251 | 159.5143 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 41 | 0.110 | 1.789 | 1.898 | 0.343 | 2.241 | 116.3935 † | 158.9330 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 42 | 20 | 1.499 | 0.927 | 2.426 | 0.198 | 2.624 | 158.7335 | 158.5167 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 38 | 0.193 | 2.702 | 2.895 | 0.548 | 3.442 | 121.3734 † | 160.0720 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 42 | 21 | 2.655 | 1.892 | 4.546 | 0.258 | 4.804 | 159.3528 | 159.1097 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 41 | 0.288 | 4.835 | 5.123 | 0.761 | 5.884 | 149.2628 † | 160.4125 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 43 | 21 | 3.701 | 2.280 | 5.981 | 0.365 | 6.346 | 159.5433 | 159.3841 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 44 | 0.375 | 6.139 | 6.514 | 0.924 | 7.438 | 160.9545 | 160.9561 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 43 | 21 | 5.004 | 3.223 | 8.227 | 0.497 | 8.723 | 159.8295 | 159.7018 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 40 | 0.497 | 7.888 | 8.385 | 1.010 | 9.395 | 112.1350 † | 161.3759 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 45 | 21 | 7.180 | 4.278 | 11.458 | 0.649 | 12.106 | 159.9037 | 159.7995 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 40 | 0.986 | 15.599 | 16.585 | 1.617 | 18.201 | 87.1815 † | 161.8539 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 44 | 22 | 13.307 | 10.340 | 23.647 | 1.218 | 24.865 | 160.1387 | 160.0330 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 44 | 1.747 | 22.287 | 24.035 | 2.076 | 26.111 | 72.6993 † | 162.0649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 44 | 23 | 15.986 | 12.248 | 28.234 | 1.211 | 29.444 | 160.8509 | 160.7558 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

