# CSV data dictionary

An empty field denotes an unavailable/not-recorded quantity; it must never be read as zero. Booleans are 0/1. Iteration counts, element counts, ranks, solver flags, status labels, checksums, ratios, fractions, grayness, normalized residuals, and classifications are dimensionless unless stated otherwise.

| CSV | Row meaning and units |
|---|---|
| `configuration_identity.csv` | One row per replay target. Mesh is elements; equality/PASS fields are dimensionless. |
| `olhoff_640_failure_diagnostics.csv` | One failed LP attempt. `omega1`–`omega3`: rad/s; `lamref`: (rad/s)^2; `move`: density fraction; LP residuals/row norms/rcond: normalized or solver-native dimensionless quantities. Empty residual/activity fields mean no primal point was returned. |
| `olhoff_640_history_window.csv` | One successful outer update plus the failed attempt marker. Frequencies: rad/s; density changes, move, volume and gap: dimensionless. |
| `yuksel_800_history.csv` | One optimizer iteration. Density changes/fractions/residuals/grayness/turnover: dimensionless; objective: moving-load compliance in the frozen model's native units; `mode_angle_deg`: degrees. Empty snapshot fields mean no snapshot at that iteration. |
| `yuksel_800_cap_diagnosis.csv` | One late-history window. Density/objective-relative metrics and fractions: dimensionless. |
| `proposed_160_history.csv` | One optimizer iteration per run. Frequencies: rad/s; density changes/fractions/residuals/grayness: dimensionless; objective: compliance in the frozen model's native units. |
| `proposed_160_determinism.csv` | One row per repeat. Frequencies and common evaluator values: rad/s; `finalDx`: density fraction. |
| `proposed_160_common_evaluators.csv` | One mode per row. All native/common columns: rad/s. Common evaluators are comparison models, not ground truth. |
| `proposed_160_mode_localization.csv` | One native mode per row. `omega`: rad/s; energy/displacement fractions and weighted density: dimensionless; participation: elements. |
| `publication_readiness_delta.csv` | One publication claim; categorical, no physical units. |
