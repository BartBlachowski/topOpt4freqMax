# Conference performance benchmark -- notes

Generated 2026-09-14T05:34:28+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `nine_mesh_pedersen_b21483b`
- scientific evidence: **true**
- performance campaign: **true**
- resolutions: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100
- threads: 1

## How to read the table

Count/time columns represent method-native computational stages and are not mathematically identical across methods. Total wall time is the common performance quantity.

Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box): Count 1 = outer iterations, Count 2 = cumulative nested MMA iterations, Time 1 = outer work excluding the nested MMA solve, Time 2 = nested MMA total. The two counts are never added.

## Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)

The Olhoff column is labelled "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)" and must not be labelled "Olhoff 2007". It is produced by analysis/OlhoffCurrent at the named preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered (upstream preset duOlhoffAdaptivePedersen @ 2530692). It is NOT the historical SIMP + eq. (4b) realization that earlier campaigns labelled "Du-Olhoff reconstruction (M4)" (preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered): the two differ in the low-density material law and the outer controller.

> Du-Olhoff RECONSTRUCTION, Pedersen/adaptive-box formulation (class C controller and filter radius, class B/D material law): SIMP p = 3 with the Pedersen (2000) linearized low-density stiffness (rho*rho0^(p-1) below rho0 = 0.1) and LINEAR mass, eq. (2); a per-element ADAPTIVE move box (initial 0.10, floor 0.002, x1.2 on monotone / x0.7 on reversing outer steps); NATURAL termination by ||drho||_2 < 0.05*sqrt(NE/3200) with no guards and no persistence, a heuristic design-change stop and not a KKT certificate; Sigmund sensitivity filter on all f_sk at a FIXED PHYSICAL radius R = 0.06 (1.2 elements at 160x20, 6 at 800x100). This is a DISTINCT formulation from the historical SIMP + eq. (4b) OlhoffCurrent reconstruction (duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered), not a bug fix of it; Pedersen stiffness is the alternative Du & Olhoff sec. 2.2 name, not their choice. Terminal bimodality is NOT expected beyond coarse meshes: in the committed R = 0.06 sweep the optimum is bimodal only at 160x20 (native gap12 0.7 %) and native gap12 is 11.8-24.5 % from 240x30 to 800x100. Native eigenfrequencies are those of the Pedersen/linear-mass model; any cross-method table must name its evaluator model. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/OlhoffCurrent/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction is set by its per-element adaptive move box and its mesh-scaled design-change tolerance, neither of which the original publication specifies, and it is NOT monotone in the mesh (committed R = 0.06 sweep: 121, 111, 101, 93, 112, 130, 156, 204, 246 outer iterations from 160x20 to 800x100). Per-outer-iteration cost is therefore reported next to total wall time.

### Cost per outer iteration

Total wall time and per-outer-iteration cost, from the same per-iteration timers. eig/outer includes FE assembly.

| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 121 | 223.920 | 1.8506 | 0.0354 | 0.0324 | 1.8147 | 19.58 | 0.0927 | NATIVE_CONVERGED |
| 240x30 | 111 | 354.944 | 3.1977 | 0.0670 | 0.0626 | 3.1300 | 18.71 | 0.1673 | NATIVE_CONVERGED |
| 320x40 | 101 | 427.204 | 4.2297 | 0.1159 | 0.1086 | 4.1126 | 19.81 | 0.2076 | NATIVE_CONVERGED |
| 400x50 | 93 | 570.098 | 6.1301 | 0.1863 | 0.1764 | 5.9416 | 20.57 | 0.2888 | NATIVE_CONVERGED |
| 480x60 | 112 | 981.385 | 8.7624 | 0.2867 | 0.2725 | 8.4728 | 19.08 | 0.4441 | NATIVE_CONVERGED |
| 560x70 | 130 | 1425.047 | 10.9619 | 0.3930 | 0.3737 | 10.5654 | 18.24 | 0.5793 | NATIVE_CONVERGED |
| 640x80 | 156 | 2215.820 | 14.2040 | 0.5206 | 0.4939 | 13.6795 | 18.38 | 0.7441 | NATIVE_CONVERGED |
| 720x90 | 204 | 3537.510 | 17.3407 | 0.9283 | 0.8925 | 16.4074 | 18.39 | 0.8921 | NATIVE_CONVERGED |
| 800x100 | 246 | 4901.034 | 19.9229 | 1.1584 | 1.1143 | 18.7591 | 18.90 | 0.9924 | NATIVE_CONVERGED |

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the outer iteration count is a property of the reconstruction's controller and stop, not of the published method, and need not be monotone in the mesh; a total-time exponent mixes per-iteration cost with that count, so the per-outer-iteration cost fit is reported alongside it. Either exponent describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 6.488706e-02 | 0.9608 | 0.9202 | 9 |

Per-outer-iteration cost, T/N_outer (Ne) = C * Ne^p:

| Method | Quantity | C | p | R^2 | points |
|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | total_wall_time_per_outer_s | 3.811912e-03 | 0.7549 | 0.9922 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | outer_time_excluding_inner_per_outer_mean_s | 4.473465e-06 | 1.0868 | 0.9831 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | eigen_time_per_outer_mean_s | 3.672673e-06 | 1.1008 | 0.9835 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_outer_mean_s | 4.128173e-03 | 0.7434 | 0.9926 | 9 |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | inner_time_per_inner_iteration_mean_s | 1.810479e-04 | 0.7613 | 0.9863 | 9 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | omega1 | Status |
|---|---|---|---|---|---|---|---|---|
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 160x20 | 121 | 2369 | 4.289 | 219.581 | 223.920 | 169.2106 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 240x30 | 111 | 2077 | 7.438 | 347.426 | 354.944 | 167.3424 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 320x40 | 101 | 2001 | 11.701 | 415.372 | 427.204 | 165.8568 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 400x50 | 93 | 1913 | 17.327 | 552.569 | 570.098 | 166.4552 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 480x60 | 112 | 2137 | 32.112 | 948.950 | 981.385 | 166.0093 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 560x70 | 130 | 2371 | 51.095 | 1373.502 | 1425.047 | 165.8101 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 640x80 | 156 | 2868 | 81.212 | 2133.997 | 2215.820 | 165.6500 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 720x90 | 204 | 3752 | 189.373 | 3347.104 | 3537.510 | 165.4234 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box) | 800x100 | 246 | 4650 | 284.958 | 4614.751 | 4901.034 | 165.4322 | NATIVE_CONVERGED |

