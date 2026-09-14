# Terminal-direction audit

Generated 2026-07-31 15:01:11 with MATLAB. The filtered LP direction is solved once at the saved terminal density and move floor, then held fixed over `t = [0 6.1035e-05 0.00012207 0.00024414 0.00048828 0.00097656 0.0019531 0.0039062 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1]`. This is a directional diagnostic, not an optimization sweep.

| case | filtered dLambda | raw dLambda | FD slope | rel. raw error | classification |
|---|---:|---:|---:|---:|---|
| ss_n1 | +1.680085e+00 | -6.265501e+00 | -4.918387e-01 | 9.215e-01 | FILTERED_DIRECTION_IS_TRUE_DESCENT |
| cs_n1 | +2.390570e+00 | +1.090741e+00 | +1.474858e+00 | 2.604e-01 | TRUE_ASCENT_EXISTS_BELOW_FLOOR |
| cc_n1 | +1.089586e+01 | -1.333101e+01 | -1.004026e+01 | 2.468e-01 | FILTERED_DIRECTION_IS_TRUE_DESCENT |
| ss_n2 | +2.856314e+01 | -1.112624e+03 | +2.803281e+01 | 1.025e+00 | TRUE_ASCENT_BELOW_FLOOR_RAW_MODEL_MISMATCH |
| cs_n2 | +5.115170e+01 | -4.191794e+01 | -3.423127e+01 | 1.834e-01 | FILTERED_DIRECTION_IS_TRUE_DESCENT |
| cc_n2 | +6.330338e+01 | -1.132995e+01 | -5.054282e+00 | 5.539e-01 | FILTERED_DIRECTION_IS_TRUE_DESCENT |
| cc_gap23 | +1.117722e+02 | -1.840931e+01 | -1.528460e+01 | 1.697e-01 | FILTERED_DIRECTION_IS_TRUE_DESCENT |

## Result

2 of 7 terminal states have a true improving physical step at the nominal move floor. The solver therefore stopped before testing an available ascent step. 5 of 7 filtered LP directions are physical descent directions as `t -> 0`, confirming that the filtered subproblem is not a consistent local model of the physical objective.

Interpretation is limited to the tested filtered direction. A non-ascent result does not certify KKT stationarity or exclude other feasible ascent directions.
