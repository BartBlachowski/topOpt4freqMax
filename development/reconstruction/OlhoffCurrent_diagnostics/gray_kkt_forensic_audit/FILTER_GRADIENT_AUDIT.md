# Filter gradient audit

| mesh | gray RMS filtered/raw | gray std filtered/raw | gray sign flips % | raw gray-fit KKT RMS | filtered gray-fit subproblem RMS |
| --- | --- | --- | --- | --- | --- |
| 400 | 0.906078 | 0.0626838 | 43.2628 | 0.354769 | 0.0222383 |
| 480 | 0.906556 | 0.146669 | 20.2804 | 0.334039 | 0.0489931 |
| 800 | 1.01287 | 0.996391 | 3.20424 | 1.16254 | 1.15835 |

Residuals share the raw interior objective-gradient RMS scale. The filter mostly suppresses **spatial variation**, not the absolute magnitude of the whole gradient. At 400/480 it replaces heterogeneous signed gray sensitivities by nearly constant positive values. It is inaccurate to describe this as a universal near-zero objective gradient: the filtered derivative can be balanced by a positive volume multiplier.

The exact first-mode 800 gradient is not similarly flattened. A relaxed two-mode fit lowers its filtered gray residual to .05045 on the same scale, while its optimistic raw lower bound remains .12676. Thus single-mode attenuation comparisons alone are inadequate there. This fit is not an exact KKT certificate because lambda2 is still separated.

The formula is recovered exactly, all tensor blocks are filtered, and finite differences of the frozen filtered subspace prediction validate its increment derivative. FE finite differences validate the raw derivative instead. This explained discrepancy is a **formulation/surrogate consistency issue**, not evidence of an incorrectly transcribed filter implementation. No claim of an unknown density chain rule or of a proved non-integrable vector field is made.

No causal filter A/B was run. Quantitative first-order mismatch supports investigating the filter/optimality interface; correlation with mesh alone would not.
