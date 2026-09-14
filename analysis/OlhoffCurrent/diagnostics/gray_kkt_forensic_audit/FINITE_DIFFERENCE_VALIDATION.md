# Finite-difference validation

FINAL_STATE_SENSITIVITY_VALIDATED

| mesh | unique elements | delta | median relative error | max relative error | max error / raw gradient RMS |
| --- | --- | --- | --- | --- | --- |
| 400 | 12 | 0.001 | 4.50967e-06 | 3.16457 | 1.32067e-05 |
| 400 | 12 | 0.0003 | 2.18057e-05 | 1.77738 | 1.1017e-05 |
| 400 | 12 | 0.0001 | 7.87975e-05 | 1.33613 | 7.84295e-05 |
| 480 | 15 | 0.001 | 6.98858e-06 | 0.372397 | 1.44728e-05 |
| 480 | 15 | 0.0003 | 1.47983e-05 | 3.81268 | 1.53849e-05 |
| 480 | 15 | 0.0001 | 7.0426e-05 | 8.76952 | 7.36241e-05 |
| 800 | 15 | 0.001 | 1.21538e-05 | 0.333878 | 3.34012e-05 |
| 800 | 15 | 0.0003 | 4.20275e-05 | 0.203523 | 3.09812e-05 |
| 800 | 15 | 0.0001 | 5.00942e-05 | 2.02661 | 5.39196e-05 |

Validation scope is the raw first-order derivative at the accuracy relevant to the KKT findings, and the increment derivative of the implemented filtered subspace prediction. It is **not** uniform relative-accuracy validation of tiny void sensitivities. The deterministic sample rule was frozen before inspecting derivatives; sample IDs depend solely on saved rho classes and column-major quantiles. 400 has no broad-core class and has 12 unique samples; 480/800 each have 15. IDs, classes and deduplication are in FD_SAMPLE_PREREGISTERED.json. Three step sizes, centered or second-order one-sided as preregistered, produce 126 accepted rows / 252 perturbed FE solves. No rho was advanced by an optimization update.

All accepted rows have absolute derivative error below 7.85e-5 of the raw interior gradient RMS, over a thousand times below the .1 stationarity scale. Median relative errors are ~1e-5–8e-5. Weak void derivatives have large relative errors (up to 8.77); shrinking delta often worsens error, consistent with eigensolver/subtraction roundoff. Those weak derivative signs are not certified. The gray/broad KKT verdict does not rely on them. This is an explained numerical limitation, not an unexplained material analytic/FD disagreement.

The frozen subspace directional checks use the smallest eigenvalue of the full 2×2 offset matrix, not an untracked individual mode. Raw and filtered model derivative errors are tiny at 400/480; at 800 their maximum normalized errors are 1.83e-6 / 2.38e-5. Physical FE FD does not match the filtered vector: normalized RMS discrepancies .365/.348/.0471 across sampled rows are expected because the filter modifies sensitivities without changing FE rho. This validates the distinction, not a supposed density chain rule.

Full numerical rows are FD_RESULTS_*.csv; delta convergence is F19. Corrected floating-point recordings are fd_*.mat; INVALID_INTEGER_SERIALIZATION files are explicitly unusable scratch. See EXECUTION_NOTES.md for all attempts. No result-dependent sample replacement or best-delta filtering was performed.
