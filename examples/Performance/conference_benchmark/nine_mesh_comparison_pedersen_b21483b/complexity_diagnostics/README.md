# Complexity diagnostics

Derived from recorded timings and counts; no solver rerun. Stage time = Time 1 + Time 2. Other and total wall time are not changed.

- native_counts.png: each method's native counts, never summed across nested levels.
- cost_per_native_unit.png: measured component time / native count, with log-power fits below 2^17 full DOFs.
- transition_ratios.png: observed unit cost divided by that below-threshold prediction.
- stage_time_residuals.png: free and fixed 1.5 exponents fitted in the SAME log-time space.
- stage_time_validation.png: leave-one-mesh-out and below-to-above-threshold prediction MAPE.

The CSVs contain full-precision observations, fitted parameters, residuals, predictions, MAPE, RMSE in seconds and RMSE in log time. Residual sign is prediction / observation - 1. Fits require the campaign scaling gate and finite positive ok records. Measured counts and costs remain visible even when a fit is refused; use CSV status/eligibility fields.

The vertical line marks the FIRST OBSERVED mesh at or above 2^17 full DOFs, computed as 2*(nelx+1)*(nely+1). It does not estimate a continuous breakpoint. Only two current meshes lie above it. This is consistent with the prior sparse-assembly audit, not a new causal experiment. Do not interpret nine observations as proof of asymptotic complexity or repeated-run timing uncertainty. Unit-cost fits do not predict iteration counts.

The original four complexity plots retain their historical free-log/fixed-seconds objectives; these supplementary residual comparisons use log space for BOTH models.

Regenerate via confbench_refresh_complexity_plots(campaignDir), with conference_bench and analysis/Olhoff on the MATLAB path. New benchmark runs generate this set automatically.
