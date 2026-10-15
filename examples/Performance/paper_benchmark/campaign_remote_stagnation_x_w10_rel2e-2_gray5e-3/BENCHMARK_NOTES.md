# Conference performance benchmark -- notes

Generated 2026-10-15T11:59:40+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w10_rel2e-2_gray5e-3`
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
| 160x20 | 0.146 | 0.059 | 0.087 | 0.209 | 0.904 |
| 240x30 | 0.212 | 0.111 | 0.100 | 0.351 | 2.074 |
| 320x40 | 0.274 | 0.190 | 0.084 | 0.590 | 3.990 |
| 400x50 | 0.464 | 0.339 | 0.126 | 0.872 | 5.939 |
| 480x60 | 0.601 | 0.492 | 0.109 | 1.412 | 9.579 |
| 560x70 | 0.864 | 0.738 | 0.126 | 1.678 | 13.616 |
| 640x80 | 1.171 | 1.021 | 0.150 | 1.992 | 16.999 |
| 720x90 | 2.162 | 1.985 | 0.176 | 3.171 | 33.248 |
| 800x100 | 2.895 | 2.651 | 0.244 | 4.146 | 46.088 |

The largest preparation value, 0.244 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.084 and 0.176 s, independent of the mesh.

### Native omega_1 values flagged

7 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 80.04, 80.04, 108.09 | 153.84 | 48.0% | 0.009 |
| Proposed | 240x30 | 90.49, 90.49, 104.98 | 157.08 | 42.4% | 0.009 |
| Proposed | 320x40 | 118.64, 118.64, 120.55 | 158.92 | 25.3% | 0.004 |
| Proposed | 400x50 | 121.37, 121.37, 122.45 | 160.07 | 24.2% | 0.003 |
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
| Proposed | 5.532026e-05 | 1.1846 | 0.9840 | 9 |
| Yuksel | 4.347732e-04 | 1.0289 | 0.9731 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0152 | 0.0137 | 0.0181 | N/A | N/A |
| 240x30 | 14942 | 0.0403 | 0.0345 | 0.0454 | N/A | N/A |
| 320x40 | 26322 | 0.0764 | 0.0689 | 0.0879 | N/A | N/A |
| 400x50 | 40902 | 0.1244 | 0.1129 | 0.1441 | N/A | N/A |
| 480x60 | 58682 | 0.1785 | 0.1659 | 0.2132 | N/A | N/A |
| 560x70 | 79662 | 0.2545 | 0.2323 | 0.2958 | N/A | N/A |
| 640x80 | 103842 | 0.3496 | 0.3120 | 0.3972 | N/A | N/A |
| 720x90 | 131222 | 0.7023 | 0.5504 | 0.7926 | N/A | N/A |
| 800x100 | 161802 | 0.8930 | 0.6996 | 0.9852 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 42 | 0.059 | 0.637 | 0.695 | 0.209 | 0.904 | 80.0403 † | 153.8357 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 101 | 27 | 1.383 | 0.490 | 1.872 | 0.114 | 1.986 | 158.3209 | 158.0774 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 40 | 0.111 | 1.612 | 1.723 | 0.351 | 2.074 | 90.4948 † | 157.0766 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 70 | 24 | 2.414 | 1.088 | 3.503 | 0.177 | 3.679 | 160.0178 | 159.8131 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 42 | 0.190 | 3.210 | 3.400 | 0.590 | 3.990 | 118.6431 † | 158.9164 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 82 | 28 | 5.649 | 2.461 | 8.110 | 0.266 | 8.376 | 159.0564 | 158.8605 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 38 | 0.339 | 4.728 | 5.067 | 0.872 | 5.939 | 121.3734 † | 160.0720 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 63 | 24 | 7.113 | 3.457 | 10.570 | 0.465 | 11.035 | 159.2761 | 159.0963 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 43 | 0.492 | 7.674 | 8.166 | 1.412 | 9.579 | 158.8525 | 160.3641 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 54 | 24 | 8.957 | 5.116 | 14.072 | 0.605 | 14.677 | 159.4435 | 159.3064 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 44 | 0.738 | 11.200 | 11.938 | 1.678 | 13.616 | 160.9545 | 160.9561 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 49 | 23 | 11.382 | 6.803 | 18.184 | 0.875 | 19.059 | 159.8010 | 159.6809 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 40 | 1.021 | 13.986 | 15.007 | 1.992 | 16.999 | 112.1350 † | 161.3759 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 46 | 21 | 14.350 | 8.341 | 22.691 | 1.328 | 24.018 | 159.8923 | 159.7857 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 40 | 1.985 | 28.092 | 30.077 | 3.171 | 33.248 | 87.1815 † | 161.8539 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 47 | 22 | 25.868 | 17.436 | 43.304 | 1.913 | 45.217 | 160.1033 | 159.9943 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 44 | 2.651 | 39.292 | 41.942 | 4.146 | 46.088 | 72.6993 † | 162.0649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 57 | 23 | 39.874 | 22.660 | 62.535 | 2.324 | 64.858 | 159.9142 | 159.8402 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

