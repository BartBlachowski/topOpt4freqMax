# Conference performance benchmark -- notes

Generated 2026-09-11T23:51:04+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_9mesh_r2`
- scientific evidence: **true**
- performance campaign: **true**
- resolutions: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100
- threads: 1

## How to read the table

Count/time columns represent method-native computational stages and are not mathematically identical across methods. Total wall time is the common performance quantity.

Proposed: Count 1 = reference eigenanalysis solves (always 1, not an optimization iteration), Count 2 = SIMP iterations, Time 1 = eigenanalysis and preparation, Time 2 = SIMP. Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 iteration counts, Time 1 and Time 2 the corresponding stage times. Du-Olhoff reconstruction (M4): Count 1 = outer iterations, Count 2 = cumulative nested MMA iterations, Time 1 = outer work excluding the nested MMA solve, Time 2 = nested MMA total. The two counts are never added.

## Du-Olhoff reconstruction (M4)

The Olhoff column is labelled "Du-Olhoff reconstruction, fixed penalty, sensitivity filtered" and must not be labelled "Olhoff 2007". It is produced by analysis/OlhoffCurrent at its named production preset duOlhoffFixedPenaltySensitivityFiltered; the historical audit code for the same realization is M4.

> Du-Olhoff RECONSTRUCTION (class C): internally coherent and fully documented, but not a claimed historical implementation. Numerical continuation, filter radius, multiplicity tolerance and inner-loop convergence are not specified in Du & Olhoff (2007) and are reconstruction choices here. Must NOT be labelled "Olhoff 2007". Field-level provenance (A/B/C/D) is documented in analysis/OlhoffCurrent/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md and is printed per-run by olh.config.describe.

The outer iteration count of the Du-Olhoff reconstruction depends on a move-limit continuation schedule that the original publication does not specify. The frozen schedule used here is documented; a different admissible schedule changes the count without changing the physics.

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the source audit records that its outer iteration count is an artifact of the continuation trigger, so a fitted exponent for that method describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 1.870042e-05 | 1.4134 | 0.9616 | 9 |
| Yuksel | 1.175524e-06 | 1.8157 | 0.9775 | 9 |
| Du-Olhoff reconstruction (M4) | 3.701857e-02 | 0.9825 | 0.9800 | 9 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | omega1 | Status |
|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 107 | 0.434 | 1.342 | 1.951 | 109.0501 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 123 | 1.475 | 2.360 | 4.289 | 157.2784 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 160x20 | 91 | 2241 | 2.784 | 116.884 | 119.761 | 169.4952 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 236 | 0.259 | 6.424 | 6.934 | 108.7822 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 152 | 4.065 | 4.935 | 9.393 | 159.4915 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 240x30 | 104 | 2334 | 6.642 | 207.642 | 214.408 | 167.0704 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 207 | 0.267 | 10.644 | 11.304 | 158.7628 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 320 | 10.376 | 16.639 | 27.574 | 160.7459 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 320x40 | 131 | 2614 | 14.889 | 320.650 | 335.713 | 165.9508 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 182 | 0.340 | 14.940 | 15.734 | 159.5184 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 417 | 19.654 | 33.192 | 53.663 | 160.0551 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 400x50 | 139 | 2918 | 25.876 | 517.458 | 543.591 | 162.8888 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 219 | 0.430 | 26.042 | 27.111 | 160.2542 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 586 | 615 | 52.611 | 69.020 | 122.678 | 160.5983 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 480x60 | 164 | 3463 | 46.429 | 851.471 | 898.260 | 161.9055 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 256 | 0.530 | 42.866 | 44.268 | 160.7224 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 833 | 771 | 104.307 | 117.670 | 223.543 | 160.3923 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 560x70 | 190 | 3922 | 73.265 | 1261.160 | 1334.929 | 161.0335 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 309 | 0.677 | 68.882 | 70.696 | 160.8517 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 1542 | 2000 | 256.822 | 400.200 | 658.919 | 160.8751 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 640x80 | 199 | 4324 | 106.191 | 1666.952 | 1773.787 | 159.7253 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 297 | 1.124 | 163.773 | 166.853 | 161.0688 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 1162 | 907 | 339.007 | 371.117 | 713.049 | 160.6227 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 720x90 | 223 | 4831 | 211.456 | 2185.029 | 2397.571 | 159.0860 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 330 | 1.459 | 239.197 | 243.223 | 161.3649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 1150 | 1171 | 417.753 | 597.601 | 1018.938 | 160.8856 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 800x100 | 170 | 3713 | 195.607 | 1790.086 | 1987.064 | 153.3020 | NATIVE_CONVERGED |

