# Conference performance benchmark -- notes

Generated 2026-09-04T19:00:55+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `partial_4mesh`
- scientific evidence: **true**
- performance campaign: **false**
- resolutions: 160x20, 240x30, 320x40, 400x50
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

_No scaling fit was performed for this run: this run is not a complete performance campaign; a scaling exponent must not be fitted to smoke or preflight data_

## Results

| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | omega1 | Status |
|---|---|---|---|---|---|---|---|---|
| Proposed | 160x20 | 1 | 107 | 0.178 | 1.300 | 1.670 | 109.0501 | NATIVE_CONVERGED |
| Yuksel | 160x20 | 121 | 123 | 1.482 | 2.220 | 4.022 | 157.2784 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 160x20 | 91 | 2241 | 2.832 | 115.346 | 118.220 | 169.4952 | NATIVE_CONVERGED |
| Proposed | 240x30 | 1 | 236 | 0.216 | 6.180 | 6.644 | 108.7822 | NATIVE_CONVERGED |
| Yuksel | 240x30 | 168 | 152 | 4.043 | 4.943 | 9.377 | 159.4915 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 240x30 | 104 | 2334 | 6.677 | 207.415 | 214.168 | 167.0704 | NATIVE_CONVERGED |
| Proposed | 320x40 | 1 | 207 | 0.263 | 10.462 | 11.109 | 158.7628 | NATIVE_CONVERGED |
| Yuksel | 320x40 | 252 | 320 | 10.436 | 16.796 | 27.781 | 160.7459 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 320x40 | 131 | 2614 | 14.991 | 324.843 | 339.964 | 165.9508 | NATIVE_CONVERGED |
| Proposed | 400x50 | 1 | 182 | 0.343 | 15.521 | 16.313 | 159.5184 | NATIVE_CONVERGED |
| Yuksel | 400x50 | 315 | 417 | 19.822 | 33.105 | 53.751 | 160.0551 | NATIVE_CONVERGED |
| Du-Olhoff reconstruction (M4) | 400x50 | 139 | 2918 | 25.904 | 522.210 | 548.320 | 162.8888 | NATIVE_CONVERGED |

