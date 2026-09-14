# Assessment of the latest Pedersen stage-time complexity curves

Source: `examples/Performance/conference_benchmark/nine_mesh_comparison_pedersen_b21483b/benchmark_results.json`.
Nine meshes per method, 3,200–80,000 elements; all 27 records are marked ok. Stage time is the recorded Time 1 + Time 2. No optimization or timing measurement was repeated, and existing plots and evidence were not modified during this assessment.

## Conclusion

The existing power laws are useful summaries, but they are not established asymptotic complexity laws or demonstrably optimal predictive models. The most informative assessment separates iteration counts, cost per native iteration, and the observed implementation-related timing step. There is no single best fitted curve independently of the intended error measure and prediction range.

## The current curves optimize different errors

The free-exponent curve minimizes squared errors in log time. The fixed-exponent curve fixes p = 1.5 and minimizes squared errors in seconds. Their reported R² values are consequently computed in different spaces and cannot be compared to select a winner. The linear-axis and logarithmic-axis images of each fit contain the same fitted curve; they are two views, not independent validations.

| Method | Current free p (log fit) | Free p when minimizing error in seconds | Current free curve error at 800×100 |
|---|---:|---:|---:|
| Proposed | 1.495 | 2.346 | −31.5% |
| Yuksel | 1.845 | 1.734 | −6.4% |
| Du–Olhoff, Pedersen | 0.961 | 1.613 | −32.0% |

Errors are (prediction / observation − 1). Fixed p = 1.5 has R² = 0.990 in seconds for Du–Olhoff, but underpredicts its 160×20 time by 83.0%. For Yuksel that curve overpredicts the smallest-mesh time by 108.8%. A good large-time fit does not guarantee consistent relative accuracy over the mesh range. Stage time still includes algorithmic initialization inside a native stage, e.g. the reference eigenanalysis; it is not a pure kernel benchmark.

## Leave-one-mesh-out checks

Each point was withheld in turn and predicted using the other eight. The table reports mean absolute percentage error (MAPE), with equal weight per mesh. This measures interpolation and endpoint sensitivity within this small dataset, not independent validation on a new machine or a larger mesh. JSON also contains errors in seconds and log time.

| Model | Proposed | Yuksel | Du–Olhoff |
|---|---:|---:|---:|
| Current free power, log fit | 29.7% | 29.4% | 32.9% |
| Fixed 1.5, log fit | 27.6% | 43.9% | 49.0% |
| Free power, seconds fit | 42.3% | 43.9% | 27.5% |
| Current fixed 1.5, seconds fit | 42.6% | 83.8% | 24.5% |
| Quadratic in log mesh size, log-time fit | 77.2% | 21.1% | 15.6% |
| Power law plus timing-step indicator, log fit | 19.0% | 28.0% | 23.2% |

The quadratic model is log T = a + b log(N/3200) + c log²(N/3200); it is a descriptive curvature model, not a fixed-order complexity claim. The step model is log T = a + b log(N/3200) + d I(N ≥ 64800), with the split suggested by the pre-existing campaign audit, not optimized over these nine points. Only two observations lie above the split. Selecting a model using these checks is exploratory and does not provide an unbiased final estimate of its generalization error.

A forward check is more sobering: fit the first seven meshes and predict the last two. The current log-power curve has MAPE 50.6%, 15.1%, and 48.1% for Proposed, Yuksel, and Du–Olhoff respectively. Yuksel's quadratic log model improves leave-one-out error but has 80.6% forward error. A step size cannot be learned from the first seven points because they all precede it. Thus lower leave-one-out error alone is not grounds to adopt a flexible curve for extrapolation.

## Separate solver cost from optimization effort

For each native stage, T_j = n_j × average cost per iteration. For Proposed, n_1 = 1 denotes the reference solve, not an optimization iteration. For Du–Olhoff the outer-exclusive and nested MMA stages must remain separate; their iteration counts must never be added.

Du–Olhoff outer counts are 121, 111, 101, 93, 112, 130, 156, 204, 246. A single power curve hides that reversal. Its global fitted stage exponent is 0.961, whereas its last-five-mesh exponent is 1.613. Nested MMA accounts for 98.1% of stage time at 160×20 and 94.2% at 800×100; MMA iterations per outer stay approximately 18–21. Its per-outer cost rises smoothly enough to have a global log-power exponent of 0.755, while varying outer counts account for the rest of the global stage-time slope. This empirical sublinear per-outer exponent is not an asymptotic claim.

Yuksel takes 3,542 combined iterations at 640×80, but only 2,069 at 720×90. Its modest stage-time increase over that interval conceals a substantial increase in cost per iteration.

| Native computational unit | Cost exponent through 640×80 | Observed cost ratio 720×90 / 640×80 | 720×90 cost / prediction from first seven meshes |
|---|---:|---:|---:|
| Proposed SIMP iteration | 1.044 | 2.47× | 1.97× |
| Yuksel Stage 1 iteration | 0.943 | 1.75× | 1.48× |
| Yuksel Stage 2 iteration | 0.857 | 2.04× | 1.80× |
| Du–Olhoff outer-exclusive work / outer iteration | 0.987 | 1.78× | 1.49× |
| Du–Olhoff nested MMA iteration | 0.741 | 1.20× | 1.11× |

The element-count ratio in that interval is only 1.266×. The existing BENCHMARK_NOTES.md associates the step with the sparse-assembly DOF threshold at 2^17, crossed between these meshes. These recorded results are consistent with that explanation; this assessment did not repeat the kernel experiments and cannot isolate causality from the aggregate timings alone.

As a conditional diagnostic, fitting the two per-stage costs separately and supplying the held-out run's observed iteration counts reduces leave-one-out MAPE to 3.3% for Proposed and 6.8% for Yuksel with the known step indicator; Du–Olhoff reaches 10.0% with ordinary per-stage log-power fits. These are explanations conditional on known counts, NOT advance predictions of complete optimization runtime. They use more structure and parameters than a single power law. Future-runtime prediction also needs a model or uncertainty range for iteration counts.

## Recommended assessment

1. Keep measured stage-time and wall-time curves as cost-to-native-stopping summaries, with clear labels. They do not compare identical stopping criteria or identical solution quality.
2. Add separate iteration-count and cost-per-iteration panels, retaining each method's native stages; show the 640×80 to 720×90 transition explicitly.
3. Add relative-residual plots. If comparing fixed and free exponents, fit both in the SAME error space and report MAPE, error in seconds, and held-out checks. For cross-mesh relative accuracy, log-space fitting is a useful default; for estimating seconds on the largest meshes, absolute-time error is a different valid objective.
4. Treat p = 1.5 as a reference hypothesis, not a universal conclusion. Report the fitted mesh interval and sensitivity to that interval alongside every exponent.
5. Before extrapolation, measure additional meshes around and above the transition and repeat timed solves in controlled sessions. The saved records provide one run per method/mesh and do not quantify run-to-run timing uncertainty. Current alternative fits cannot supply reliable predictive uncertainty by themselves.

Reproduce numerical checks from the repository root with `python3 analysis/performance_stage_time_assessment/assess.py` (NumPy and SciPy). Full metrics and residuals are in `assessment.json`.

Statistical background: [NIST model validation](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm) emphasizes residual analysis; [NIST functional model adequacy](https://itl.nist.gov/div898/handbook/pmd/section4/pmd441.htm) explains why R² alone cannot establish a model's adequacy. All numerical findings above were computed from the local campaign records.
