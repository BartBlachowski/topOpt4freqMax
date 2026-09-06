# Conference performance benchmark -- notes

Generated 2026-09-06T21:07:00+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_9mesh_r2`
- scientific evidence: **true**
- performance campaign: **true**
- resolutions: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100
- threads: 1

## How to read the table

Count/time columns represent method-native computational stages and are not mathematically identical across methods. Total wall time is the common performance quantity.

Proposed: Count 1 = reference eigenanalysis solves (always 1, not an optimization iteration), Count 2 = SIMP iterations, Time 1 = eigenanalysis and preparation, Time 2 = SIMP. Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 iteration counts, Time 1 and Time 2 the corresponding stage times. Du-Olhoff reconstruction (M4): Count 1 = outer iterations, Count 2 = cumulative nested MMA iterations, Time 1 = outer work excluding the nested MMA solve, Time 2 = nested MMA total. The two counts are never added.

## Du-Olhoff reconstruction (M4)

The Olhoff column is labelled "Du-Olhoff reconstruction (M4)" and must not be labelled "Olhoff 2007".

> Du-Olhoff timings and iteration counts refer to the frozen reconstruction used in this study. Some continuation and inner-solver details are not uniquely specified by the original publication; therefore these values should be interpreted as representative measurements of this reconstruction rather than exact historical implementation timings.

The outer iteration count of the Du-Olhoff reconstruction depends on a move-limit continuation schedule that the original publication does not specify. The frozen schedule used here is documented; a different admissible schedule changes the count without changing the physics.

## Memory

Reliable, method-independent peak-memory measurement was not available in the MATLAB environment; memory was omitted rather than reported with inconsistent semantics.

## Scaling

Scaling exponents may be fitted only to complete campaign data and never to smoke or preflight runs. For the Du-Olhoff reconstruction the source audit records that its outer iteration count is an artifact of the continuation trigger, so a fitted exponent for that method describes this reconstruction, not the published method.

| Method | C | p | R^2 | points |
|---|---|---|---|---|
| Proposed | 2.024053e-05 | 1.4051 | 0.9594 | 9 |
| Yuksel | 1.279804e-06 | 1.8096 | 0.9779 | 9 |
| Du-Olhoff reconstruction (M4) | 3.568613e-02 | 0.9857 | 0.9812 | 9 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | omega1 | Status |
|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 107 | 0.470 | 1.420 | 2.079 | 109.0501 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 123 | 1.707 | 2.342 | 4.440 | 157.2784 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 160x20 | 91 | 2241 | 2.935 | 114.650 | 117.632 | 169.4952 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 236 | 0.202 | 6.207 | 6.664 | 108.7822 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 152 | 4.032 | 4.921 | 9.351 | 159.4915 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 240x30 | 104 | 2334 | 6.635 | 204.921 | 211.633 | 167.0704 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 207 | 0.327 | 10.225 | 10.949 | 158.7628 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 320 | 10.793 | 17.749 | 29.126 | 160.7459 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 320x40 | 131 | 2614 | 14.984 | 317.479 | 332.592 | 165.9508 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 182 | 0.352 | 15.001 | 15.824 | 159.5184 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 417 | 20.485 | 34.655 | 55.985 | 160.0551 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 400x50 | 139 | 2918 | 26.347 | 529.800 | 556.357 | 162.8888 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 219 | 0.454 | 25.741 | 26.845 | 160.2542 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 586 | 615 | 55.396 | 72.647 | 129.133 | 160.5983 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 480x60 | 164 | 3463 | 46.688 | 852.281 | 899.280 | 161.9055 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 256 | 0.535 | 42.125 | 43.525 | 160.7224 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 833 | 771 | 105.366 | 118.699 | 225.521 | 160.3923 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 560x70 | 190 | 3922 | 73.456 | 1248.656 | 1322.574 | 161.0335 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 309 | 0.676 | 67.561 | 69.382 | 160.8517 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 1542 | 2000 | 257.580 | 409.617 | 669.086 | 160.8751 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 640x80 | 199 | 4324 | 107.121 | 1646.166 | 1753.887 | 159.7253 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 297 | 1.149 | 164.395 | 167.524 | 161.0688 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 1162 | 907 | 339.038 | 375.221 | 717.278 | 160.6227 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 720x90 | 223 | 4831 | 213.664 | 2173.659 | 2388.358 | 159.0860 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 330 | 1.512 | 240.638 | 244.728 | 161.3649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 1150 | 1171 | 420.287 | 603.446 | 1027.407 | 160.8856 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 800x100 | 170 | 3713 | 197.122 | 1779.240 | 1977.665 | 153.3020 | NATIVE_CONVERGED |

