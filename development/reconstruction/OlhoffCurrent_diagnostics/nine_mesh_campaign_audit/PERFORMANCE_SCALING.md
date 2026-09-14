# Performance scaling and timing quality

**PERFORMANCE_SCALING_COSTLY_BUT_INTERPRETABLE** for the actual legacy campaign. Three-rung fine-mesh performance is unmeasured.

Total solver time ranges from 119.76 s to 2397.57 s, with C800 dropping to 1987.06 s because outer work falls 223→170. The nine Olhoff runs consume 2.6681 solver-hours. This is feasible as a small workstation campaign but expensive for repeated studies, and it does not establish cost to reach comparable scientific maturity.

## Fits with prefactors

N means **number of elements**, not DOFs or elements per edge. For timings T is in seconds. Fit log(T)=log(C)+p log(N) by unweighted OLS. Counts use the same descriptive form. All rows below are legacy solves; none is excluded as capped.

| Quantity | Subset | C | p | R² log | p 95% lower | p 95% upper |
| --- | --- | --- | --- | --- | --- | --- |
| runtime_total_s | all9 | 0.0370186 | 0.982507 | 0.980021 | 0.857131 | 1.10788 |
| runtime_total_s | fine5 | 0.121755 | 0.876545 | 0.853244 | 0.208607 | 1.54448 |
| eigen_s | all9 | 2.50321e-05 | 1.40747 | 0.990244 | 1.28261 | 1.53233 |
| eigen_s | fine5 | 4.37359e-06 | 1.57101 | 0.944152 | 0.86897 | 2.27305 |
| gradient_s | all9 | 1.44667e-05 | 1.16005 | 0.991771 | 1.06561 | 1.25449 |
| gradient_s | fine5 | 1.57759e-05 | 1.15418 | 0.928226 | 0.564482 | 1.74388 |
| inner_s | all9 | 0.0434742 | 0.96073 | 0.978458 | 0.833325 | 1.08813 |
| inner_s | fine5 | 0.199955 | 0.823821 | 0.82958 | 0.137758 | 1.50988 |
| outer | all9 | 10.5297 | 0.265134 | 0.893156 | 0.183176 | 0.347092 |
| outer | fine5 | 57.8376 | 0.109051 | 0.126513 | -0.417441 | 0.635543 |
| inner_MMA | all9 | 302.568 | 0.23686 | 0.857159 | 0.150443 | 0.323277 |
| inner_MMA | fine5 | 761.579 | 0.153942 | 0.226351 | -0.368982 | 0.676866 |


All-nine total: **T=0.03701857 N^0.9825066 s**, R²log=0.98002; C 95% interval [0.0104451,0.1311975], p interval [0.85713,1.10788]. Fine-five: **T=0.1217553 N^0.8765449 s**, R²log=0.85324; C interval [8.86245e−5,167.2715], p interval [0.20861,1.54448]. Fine-four gives C=1.483635, p=0.650307, R²log=0.66401 with p interval [−0.7571,2.0577]. The huge fine-subset uncertainty and sensitivity rule out a reliable asymptotic exponent. Prefactors depend on the chosen N units and are correlated with exponent estimates.

[SCALING_FITS.csv](SCALING_FITS.csv) includes all 33 fits, C and p confidence intervals, log RMSE, all-nine/fine-five/fine-four subsets and per-iteration quantities. [SCALING_RESIDUALS.csv](SCALING_RESIDUALS.csv) gives every observed, fitted, absolute, relative and log residual. The intervals are conditional on independent homoscedastic log residuals; these are single ordered observations, not performance-repeat confidence intervals. [F05](figures/F05_total_runtime.png) and [F12](figures/F12_scaling_residuals.png) show fit and residuals.

## Work versus per-iteration cost

Total exponent 0.98251 decomposes algebraically into outer-count exponent 0.26513 plus cost-per-outer exponent 0.71738 on the same all-nine log fit. Fine-five is 0.10908+0.76747. Outer growth is moderate through 720 (91→223); C800's decline is a stopping-regime effect, not better asymptotic complexity. The fine-five outer fit R²=0.1265 is particularly uninformative. Cumulative inner work has exponent 0.23687 all-nine; cost per inner step has exponent 0.72385. Mean inner count per outer stays roughly 20–25. Most scaling comes from cost per step, not exploding inner iterations per outer.

Assembly+eigensolve per outer has exponent 1.14233 all-nine and 1.46197 fine-five. It accelerates near 720, but remains a minority of total cost. Nested MMA per outer grows from 1.284 to 10.530 s; assembly+eigensolve per outer from 0.0282 to 1.1142 s. At 800 the decomposition is MMA 1790.086 s (90.087%), assembly+eigensolve 189.417 s (9.533%), gradients/filtering 6.070 s (0.3055%), other 1.491 s (0.0750%). At 160, MMA is 97.597% and assembly+eigensolve 2.145%.

`tEig` includes FE K/M assembly plus eigSolve, not pure ARPACK time. Pure eigensolver cost cannot be isolated. `tGrad` includes generalized gradients and filtering. Other includes outer bookkeeping and setup/final analysis. These are non-overlapping; summed components reproduce caller-side solver wall time to numerical roundoff. The “outer excluding inner” column is not the eigensolver column. Timing excludes path/configuration work and common evaluator/export, so it is not total human wait time.

## Timing quality verdicts

**Strong:** recorded nesting/accounting consistency, positive costs, same reported host/MATLAB/thread policy/source and timing definitions, overwhelming nested-MMA share, and the distinction between inner counts and per-step cost.

**Indicative:** empirical per-outer scaling, the fine eigensolve cost increase, fitted C/p over this mesh range, and rough workstation practicality. The single ordered sequence confounds mesh with thermal/load effects.

**Unreliable:** bitwise timing reproducibility; machine-independent prefactors; asymptotic complexity from nine endpoints with different maturity; fine-only exponent precision; attributing the 800 time decrease to an optimization improvement; comparing historical controller wall-time ratios without host/toolchain normalization.

No competing-load, temperature, CPU model, RAM or peak memory telemetry is available. Warm-up is recorded but cold/warm effects at larger meshes are not isolated. No resumes are declared; no full external process log authenticates uninterrupted execution. None of these limitations makes the retained scaling data useless; they limit interpretation.
