# Conference performance benchmark -- notes

Generated 2026-09-04T22:53:41+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `campaign_9mesh`
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
| Proposed | 1.154552e-05 | 1.4585 | 0.9665 | 9 |
| Yuksel | 2.342802e-06 | 1.7404 | 0.9870 | 7 |
| Du-Olhoff reconstruction (M4) | 3.961009e-02 | 0.9878 | 0.9817 | 9 |

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | omega1 | Status |
|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 107 | 0.172 | 1.292 | 1.631 | 109.0501 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 123 | 1.482 | 2.214 | 4.005 | 157.2784 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 160x20 | 91 | 2241 | 2.668 | 129.120 | 131.829 | 169.4952 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 236 | 0.205 | 6.190 | 6.643 | 108.7822 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 152 | 3.952 | 4.922 | 9.248 | 159.4915 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 240x30 | 104 | 2334 | 6.484 | 234.539 | 241.097 | 167.0704 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 207 | 0.251 | 10.293 | 10.936 | 158.7628 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 320 | 10.172 | 16.881 | 27.590 | 160.7459 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 320x40 | 131 | 2614 | 14.731 | 367.219 | 382.077 | 165.9508 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 182 | 0.340 | 15.182 | 15.957 | 159.5184 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 417 | 19.289 | 54.478 | 74.560 | 160.0551 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 400x50 | 139 | 2918 | 25.914 | 593.034 | 619.154 | 162.8888 | NATIVE_CONVERGED |
| Proposed | 480x60 | 1 | 219 | 0.435 | 25.733 | 26.797 | 160.2542 | NATIVE_CONVERGED |
| Yuksel | 480x60 | 586 | 615 | 53.230 | 70.233 | 124.521 | 160.5983 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 480x60 | 164 | 3463 | 46.322 | 983.931 | 1030.569 | 161.9055 | NATIVE_CONVERGED |
| Proposed | 560x70 | 1 | 256 | 0.531 | 42.665 | 44.061 | 160.7224 | NATIVE_CONVERGED |
| Yuksel | 560x70 | 833 | 771 | 104.043 | 117.223 | 222.672 | 160.3923 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 560x70 | 190 | 3922 | 73.344 | 1432.974 | 1506.777 | 161.0335 | NATIVE_CONVERGED |
| Proposed | 640x80 | 1 | 309 | 0.666 | 68.502 | 70.281 | 160.8517 | NATIVE_CONVERGED |
| Yuksel | 640x80 | 1000 | 1000 | 166.231 | 201.044 | 369.093 | 160.7206 | CAP_HIT |
| Du-Olhoff reconstruction (M4) | 640x80 | 199 | 4324 | 106.943 | 1882.669 | 1990.213 | 159.7253 | NATIVE_CONVERGED |
| Proposed | 720x90 | 1 | 297 | 1.196 | 163.745 | 166.906 | 161.0688 | NATIVE_CONVERGED |
| Yuksel | 720x90 | 1000 | 966 | 290.660 | 397.906 | 691.542 | 160.7024 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 720x90 | 223 | 4831 | 214.004 | 2492.621 | 2707.682 | 159.0860 | NATIVE_CONVERGED |
| Proposed | 800x100 | 1 | 330 | 1.519 | 240.195 | 244.244 | 161.3649 | NATIVE_CONVERGED |
| Yuksel | 800x100 | 1000 | 1000 | 364.689 | 512.203 | 880.536 | 160.8791 | CAP_HIT |
| Du-Olhoff reconstruction (M4) | 800x100 | 170 | 3713 | 197.963 | 2046.916 | 2246.199 | 153.3020 | NATIVE_CONVERGED |

