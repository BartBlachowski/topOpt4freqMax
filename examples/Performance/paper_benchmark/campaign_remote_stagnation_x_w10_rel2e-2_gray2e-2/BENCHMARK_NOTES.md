# Conference performance benchmark -- notes

Generated 2026-10-15T10:15:16+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w10_rel2e-2_gray2e-2`
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
| 160x20 | 0.160 | 0.072 | 0.088 | 0.209 | 0.884 |
| 240x30 | 0.213 | 0.115 | 0.098 | 0.350 | 2.103 |
| 320x40 | 0.283 | 0.198 | 0.085 | 0.537 | 3.925 |
| 400x50 | 0.498 | 0.367 | 0.131 | 1.002 | 6.191 |
| 480x60 | 0.627 | 0.515 | 0.111 | 1.248 | 9.155 |
| 560x70 | 0.890 | 0.764 | 0.126 | 1.766 | 13.933 |
| 640x80 | 1.193 | 1.036 | 0.157 | 2.039 | 17.627 |
| 720x90 | 2.233 | 2.033 | 0.200 | 3.223 | 33.711 |
| 800x100 | 2.812 | 2.605 | 0.207 | 4.140 | 46.577 |

The largest preparation value, 0.207 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.085 and 0.200 s, independent of the mesh.

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
| Proposed | 5.046674e-05 | 1.1944 | 0.9852 | 9 |
| Yuksel | 1.387278e-04 | 1.1232 | 0.9794 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0155 | 0.0137 | 0.0182 | N/A | N/A |
| 240x30 | 14942 | 0.0410 | 0.0345 | 0.0460 | N/A | N/A |
| 320x40 | 26322 | 0.0778 | 0.0686 | 0.0909 | N/A | N/A |
| 400x50 | 40902 | 0.1269 | 0.1116 | 0.1461 | N/A | N/A |
| 480x60 | 58682 | 0.1803 | 0.1666 | 0.2182 | N/A | N/A |
| 560x70 | 79662 | 0.2592 | 0.2354 | 0.3063 | N/A | N/A |
| 640x80 | 103842 | 0.3638 | 0.3159 | 0.4051 | N/A | N/A |
| 720x90 | 131222 | 0.7114 | 0.5524 | 0.8031 | N/A | N/A |
| 800x100 | 161802 | 0.9053 | 0.7030 | 1.0139 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 39 | 0.072 | 0.603 | 0.675 | 0.209 | 0.884 | 79.8243 † | 153.8423 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 64 | 28 | 0.875 | 0.510 | 1.385 | 0.117 | 1.501 | 157.8816 | 157.6126 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 40 | 0.115 | 1.639 | 1.753 | 0.350 | 2.103 | 90.4948 † | 157.0766 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 49 | 23 | 1.691 | 1.058 | 2.749 | 0.183 | 2.932 | 159.7251 | 159.5143 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 41 | 0.198 | 3.191 | 3.388 | 0.537 | 3.925 | 116.3935 † | 158.9330 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 42 | 20 | 2.881 | 1.817 | 4.699 | 0.275 | 4.973 | 158.7335 | 158.5167 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 38 | 0.367 | 4.822 | 5.189 | 1.002 | 6.191 | 121.3734 † | 160.0720 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 42 | 21 | 4.686 | 3.068 | 7.754 | 0.469 | 8.223 | 159.3528 | 159.1097 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 41 | 0.515 | 7.392 | 7.907 | 1.248 | 9.155 | 149.2628 † | 160.4125 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 43 | 21 | 7.165 | 4.582 | 11.748 | 0.610 | 12.358 | 159.5433 | 159.3841 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 44 | 0.764 | 11.404 | 12.167 | 1.766 | 13.933 | 160.9545 | 160.9561 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 43 | 21 | 10.122 | 6.432 | 16.555 | 0.896 | 17.450 | 159.8295 | 159.7018 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 40 | 1.036 | 14.552 | 15.588 | 2.039 | 17.627 | 112.1350 † | 161.3759 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 45 | 21 | 14.217 | 8.506 | 22.724 | 1.195 | 23.918 | 159.9037 | 159.7995 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 40 | 2.033 | 28.455 | 30.488 | 3.223 | 33.711 | 87.1815 † | 161.8539 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 44 | 22 | 24.305 | 17.669 | 41.973 | 1.906 | 43.879 | 160.1387 | 160.0330 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 44 | 2.605 | 39.833 | 42.438 | 4.140 | 46.577 | 72.6993 † | 162.0649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 44 | 23 | 30.934 | 23.319 | 54.252 | 2.338 | 56.590 | 160.8509 | 160.7558 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

