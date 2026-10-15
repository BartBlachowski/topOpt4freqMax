# Conference performance benchmark -- notes

Generated 2026-10-15T09:19:48+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_mac_stagnation_x_no_olhoff_rel1e-2_gray1e-2_audited`
- scientific evidence: **true**
- performance campaign: **false**
- resolutions: 160x20, 240x30, 320x40, 400x50
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
| 160x20 | 0.112 | 0.029 | 0.083 | 0.135 | 0.646 |
| 240x30 | 0.141 | 0.061 | 0.081 | 0.212 | 1.348 |
| 320x40 | 0.201 | 0.122 | 0.080 | 0.350 | 2.396 |
| 400x50 | 0.276 | 0.187 | 0.089 | 0.633 | 3.655 |

The largest preparation value, 0.089 s at 400x50, is the first measured row of its session; the other 3 rows lie between 0.080 and 0.083 s, independent of the mesh.

### Native omega_1 values flagged

4 row(s) carry a native omega_1 that deviates from E1 by more than 5%. For each, the first three native modes and the E1 selected structural mode are listed; a cluster of native modes within a few percent of each other, and an E1 mode whose kinetic energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of localized near-void modes in the native model, not of a different structure.

| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |
|---|---|---|---|---|---|
| Proposed | 160x20 | 80.11, 80.11, 108.23 | 153.86 | 47.9% | 0.009 |
| Proposed | 240x30 | 98.22, 98.22, 106.14 | 157.07 | 37.5% | 0.009 |
| Proposed | 320x40 | 124.82, 124.82, 125.57 | 158.90 | 21.4% | 0.004 |
| Proposed | 400x50 | 131.91, 131.91, 131.99 | 160.00 | 17.6% | 0.003 |

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/Olhoff at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/Olhoff/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

_No scaling fit was performed for this run: this run is not a complete performance campaign; a scaling exponent must not be fitted to smoke or preflight data_

### Per-iteration cost across meshes

Per-iteration cost of all three methods steps up between 640x80 and 720x90 while the Du-Olhoff inner MMA, which does no sparse assembly, does not. The step coincides with the number of DOFs crossing 2^17 = 131072 (103842 at 640x80, 131222 at 720x90). All three solvers assemble K and M with MATLAB sparse(i,j,v,n,n) every iteration, and in isolation that call measured about 4x slower for n >= 131072 (MATLAB R2025b, Apple silicon, 1 thread, 2026-09-14: 0.086 s at n = 131000, 0.382 s at n = 131072, 4e6 triplets), whereas backslash, decomposition and eigs scale smoothly across the same sizes. Exponents fitted across this boundary include the step: they describe the assembly primitive as much as the methods.

| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |
|---|---|---|---|---|---|---|
| 160x20 | 6762 | 0.0112 | 0.0089 | 0.0116 | N/A | N/A |
| 240x30 | 14942 | 0.0244 | 0.0202 | 0.0259 | N/A | N/A |
| 320x40 | 26322 | 0.0437 | 0.0361 | 0.0460 | N/A | N/A |
| 400x50 | 40902 | 0.0691 | 0.0583 | 0.0720 | N/A | N/A |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 43 | 0.029 | 0.483 | 0.512 | 0.135 | 0.646 | 80.1109 † | 153.8584 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 78 | 32 | 0.693 | 0.370 | 1.063 | 0.099 | 1.162 | 157.9278 | 157.7054 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 44 | 0.061 | 1.076 | 1.136 | 0.212 | 1.348 | 98.2247 † | 157.0739 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 57 | 26 | 1.151 | 0.673 | 1.825 | 0.131 | 1.956 | 159.7974 | 159.5674 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 44 | 0.122 | 1.925 | 2.046 | 0.350 | 2.396 | 124.8182 † | 158.8966 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 51 | 25 | 1.842 | 1.149 | 2.992 | 0.186 | 3.177 | 158.9864 | 158.7672 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 41 | 0.187 | 2.835 | 3.021 | 0.633 | 3.655 | 131.9135 † | 159.9995 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 55 | 26 | 3.207 | 1.872 | 5.079 | 0.253 | 5.332 | 159.4140 | 159.2400 | NATIVE_CONVERGED |

† native omega1 deviates from E1 by more than 5% (see "Native omega_1 values flagged").

