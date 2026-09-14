# Conference performance benchmark -- notes

Generated 2026-09-04T14:13:00+02:00 from `examples/Performance/performance_comparison.m`.

- run label: `smoke`
- scientific evidence: **false**
- performance campaign: **false**
- resolutions: 40x6, 60x8
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
| Proposed | 40x6 | 1 | 6 | 0.126 | 0.023 | 0.263 | 116.0481 | CAP_HIT |
| Yuksel | 40x6 | 6 | 6 | 0.016 | 0.021 | 0.258 | 156.0727 | CAP_HIT |
| Du-Olhoff reconstruction (M4) | 40x6 | 6 | 97 | 0.051 | 0.370 | 0.442 | 102.0078 | CAP_HIT |
| Proposed | 60x8 | 1 | 6 | 0.149 | 0.025 | 0.287 | 127.1741 | CAP_HIT |
| Yuksel | 60x8 | 6 | 6 | 0.024 | 0.025 | 0.281 | 135.7079 | CAP_HIT |
| Du-Olhoff reconstruction (M4) | 60x8 | 6 | 112 | 0.058 | 0.669 | 0.740 | 104.1692 | CAP_HIT |

