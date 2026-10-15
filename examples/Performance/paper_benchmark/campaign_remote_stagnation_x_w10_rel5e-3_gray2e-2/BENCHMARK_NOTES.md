# Conference performance benchmark -- notes

Generated 2026-10-15T11:35:42+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_remote_stagnation_x_w10_rel5e-3_gray2e-2`
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
| 160x20 | 0.147 | 0.059 | 0.088 | 0.236 | 1.126 |
| 240x30 | 0.214 | 0.112 | 0.102 | 0.300 | 2.555 |
| 320x40 | 0.276 | 0.193 | 0.082 | 0.573 | 4.692 |
| 400x50 | 0.458 | 0.336 | 0.122 | 0.804 | 7.134 |
| 480x60 | 0.601 | 0.492 | 0.109 | 0.621 | 12.009 |
| 560x70 | 0.866 | 0.735 | 0.131 | 0.900 | 17.653 |
| 640x80 | 1.180 | 1.027 | 0.152 | 1.228 | 22.498 |
| 720x90 | 2.149 | 1.971 | 0.178 | 2.114 | 46.678 |
| 800x100 | 2.819 | 2.612 | 0.207 | 2.766 | 64.208 |

The largest preparation value, 0.207 s at 800x100, is the first measured row of its session; the other 8 rows lie between 0.082 and 0.178 s, independent of the mesh.

### Native omega_1 values flagged

4 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 98.51, 98.51, 105.50 | 153.84 | 36.0% | 0.009 |
| Proposed | 240x30 | 107.13, 107.16, 112.63 | 157.05 | 31.8% | 0.008 |
| Proposed | 320x40 | 140.14, 141.43, 144.01 | 158.88 | 11.8% | 0.004 |
| Proposed | 400x50 | 151.79, 155.10, 155.42 | 159.90 | 5.1% | 0.003 |

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
| Proposed | 4.631269e-05 | 1.2268 | 0.9786 | 9 |
| Yuksel | 3.321955e-05 | 1.3394 | 0.9786 | 9 |

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0151 | 0.0136 | 0.0178 | N/A | N/A |
| 240x30 | 14942 | 0.0404 | 0.0346 | 0.0456 | N/A | N/A |
| 320x40 | 26322 | 0.0770 | 0.0687 | 0.0881 | N/A | N/A |
| 400x50 | 40902 | 0.1249 | 0.1130 | 0.1435 | N/A | N/A |
| 480x60 | 58682 | 0.1786 | 0.1718 | 0.2171 | N/A | N/A |
| 560x70 | 79662 | 0.2584 | 0.2388 | 0.2938 | N/A | N/A |
| 640x80 | 103842 | 0.3551 | 0.3160 | 0.3910 | N/A | N/A |
| 720x90 | 131222 | 0.7099 | 0.5557 | 0.7901 | N/A | N/A |
| 800x100 | 161802 | 0.9051 | 0.7022 | 0.9852 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 55 | 0.059 | 0.831 | 0.889 | 0.236 | 1.126 | 98.5119 † | 153.8421 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 66 | 51 | 0.898 | 0.906 | 1.804 | 0.114 | 1.918 | 157.5050 | 157.3064 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 53 | 0.112 | 2.143 | 2.255 | 0.300 | 2.555 | 107.1321 † | 157.0525 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 62 | 45 | 2.145 | 2.053 | 4.199 | 0.183 | 4.382 | 159.4208 | 159.2778 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 51 | 0.193 | 3.925 | 4.118 | 0.573 | 4.692 | 140.1380 † | 158.8754 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 79 | 85 | 5.429 | 7.489 | 12.918 | 0.272 | 13.190 | 161.0284 | 160.9408 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 48 | 0.336 | 5.993 | 6.330 | 0.804 | 7.134 | 151.7944 † | 159.8983 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 80 | 37 | 9.044 | 5.311 | 14.355 | 0.459 | 14.814 | 159.3783 | 159.2421 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 61 | 0.492 | 10.896 | 11.388 | 0.621 | 12.009 | 160.2563 | 160.2564 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 78 | 62 | 13.401 | 13.461 | 26.862 | 0.641 | 27.503 | 159.6897 | 159.5479 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 62 | 0.735 | 16.018 | 16.753 | 0.900 | 17.653 | 160.7659 | 160.7660 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 85 | 59 | 20.295 | 17.332 | 37.627 | 0.903 | 38.530 | 160.0010 | 159.9105 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 57 | 1.027 | 20.243 | 21.270 | 1.228 | 22.498 | 160.9393 | 160.9393 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 94 | 66 | 29.703 | 25.805 | 55.508 | 1.169 | 56.677 | 160.2561 | 160.1873 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 60 | 1.971 | 42.592 | 44.563 | 2.114 | 46.678 | 161.1667 | 161.1668 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 89 | 90 | 49.456 | 71.108 | 120.563 | 1.907 | 122.470 | 160.7916 | 160.7300 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 65 | 2.612 | 58.830 | 61.442 | 2.766 | 64.208 | 161.4589 | 161.4589 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 101 | 79 | 70.921 | 77.833 | 148.754 | 2.327 | 151.081 | 160.7729 | 160.7158 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

