# Causal attribution

MMA_FAILURE_MULTICAUSAL

The clearest identified defect is an absolute approximate-subproblem accuracy budget that is too coarse for the small scaled spectral gain. Tightening epsimin alone at the canonical cap gives STRONG CAUSAL EVIDENCE at both registered checkpoints; the independent G1-to-G2 accuracy pair corroborates it. The local .2-width asymptote cap is a second, interacting restriction: restoring 10 at tight accuracy gives a material improvement at call 100 and at the last common call 455, but the missing S4 call 500 limits this edge to MODERATE evidence. With the original accuracy, restoring the cap improves distance while worsening gain. Accuracy alone under the original cap also fails the joint material-effect bar. These conditional results support an interacting, multicausal explanation rather than a uniquely sufficient one-line correction. The production stop at 19 compounds the loss, but continued B0 through 5000 still fails. All approximate methods fail strict design/KKT fidelity within their fixed budgets, so the remaining convergence limitation is not fully isolated. No claim of asymptotic impossibility is made.

| Control → treatment | Factor | Evidence by frozen bars | Recovery change at target 100 / 500 | d2 reduction at target 100 / 500 | Actual call pairs |
|---|---|---|---|---|---|
| B0_CURRENT_REPEATED_MMA → S2_ASYINIT_001 | S2 init .01 | WEAK ASSOCIATION | +0.0034 / -0.2472 | -1.1% / -2.7% | [100, 100] / [500, 500] |
| B0_CURRENT_REPEATED_MMA → S3_CANONICAL_CLAMP | S3 cap 10 | WEAK ASSOCIATION | -0.0825 / -0.1636 | +23.8% / +20.9% | [100, 100] / [500, 500] |
| B0_CURRENT_REPEATED_MMA → S4_SUBSOLV_ACCURACY | S4 accuracy 1e-12 | WEAK ASSOCIATION | +0.4668 / +0.4529 | +1.8% / +10.6% | [100, 100] / [455, 455] |
| B0_CURRENT_REPEATED_MMA → S5_UNIT_BOX | S5 unit box | WEAK ASSOCIATION | -0.0229 / +0.1360 | +1.1% / +11.3% | [100, 100] / [500, 500] |
| B0_CURRENT_REPEATED_MMA → S34_CLAMP_ACCURACY | S34 cap + accuracy | JOINT INTERVENTION; not a single-factor cause | +0.7830 / +0.6074 | +58.4% / +78.3% | [100, 100] / [500, 500] |
| G0_UNSAFE → G1_GCMMA | safeguard | EVIDENCE AGAINST | +0.0000 / +0.0000 | +0.0% / +0.0% | [100, 100] / [500, 500] |
| G1_GCMMA → G2_GCMMA_ACCURATE | GCMMA accuracy | STRONG CAUSAL EVIDENCE | +0.8913 / +0.8002 | +40.8% / +70.5% | [100, 100] / [500, 500] |
| S3_CANONICAL_CLAMP → S34_CLAMP_ACCURACY | accuracy at canonical clamp | STRONG CAUSAL EVIDENCE | +0.8654 / +0.7709 | +45.4% / +72.6% | [100, 100] / [500, 500] |
| S4_SUBSOLV_ACCURACY → S34_CLAMP_ACCURACY | clamp at tight accuracy | MODERATE CAUSAL EVIDENCE | +0.3162 / +0.2196 | +57.7% / +74.7% | [100, 100] / [455, 455] |


The registered strong-effect rule requires >=.10 higher recovery AND >=20%
lower d2 at both equal-work checkpoints, with acceptable feasibility. If a
time-capped method lacks call 500, the terminal pair uses the last common call
for descriptive comparison; that pair cannot supply the missing registered
500-call confirmation for STRONG evidence. An
objective-only or distance-only improvement cannot meet it. A missing 500-call checkpoint (e.g. a time-capped run) cannot support STRONG
attribution; the actual matched call counts are shown. The joint S34
comparison is not assigned single-factor causality. Comparisons along its
2x2 edges hold the other factor fixed; their interpretation is conditional.

| Rank | Candidate cause | Evidence grade | Interpretation |
|---|---|---|---|
| 1 | Approximate-solve accuracy, epsimin=1e-7 | STRONG CAUSAL EVIDENCE, conditional | S3 to S34 and G1 to G2 meet both gain and distance bars at 100 and 500. B0 to S4 alone does not. The barrier products and first-call objective loss expose the numerical mechanism. |
| 2 | Local maximum asymptote distance .2 widths | MODERATE CAUSAL EVIDENCE, conditional | S4 to S34 materially improves both outcomes at 100 and common call 455; missing call 500 prevents STRONG attribution. Cap correction alone has mixed outcomes. |
| 3 | Relative-step stopping criterion | WEAK ASSOCIATION with eventual failure | It demonstrably truncates B0 at 19 far from the oracle, but every later stop hit also fails fidelity. There is EVIDENCE AGAINST a tolerance-only cure in the measured 5000 calls. |
| 4 | Increment coordinate scaling | WEAK ASSOCIATION | Unit-box coordinates greatly reduce observed solver work and change the finite-accuracy path, but do not meet the registered joint improvement/fidelity bars. Beta, objective and row scaling are NOT ISOLATED. |
| 5 | Remaining reciprocal-approximation curvature, regularization, Newton initialization and iteration limits | NOT ISOLATED | Even the combined cap/accuracy and accurate GCMMA methods retain substantial design error. No further factor was varied or tuned. |
| 6 | Reset of MMA history between inner calls | EVIDENCE AGAINST | Source and the bitwise S1 continuation show history already persists. Cross-outer warm starts are NOT ISOLATED. |
| 7 | Noncanonical asymptote initialization in the active path | EVIDENCE AGAINST | Production already uses .5. Historical .01 worsens terminal recovery and cannot be the missing canonical correction. |
| 8 | Absence of the conservative acceptance safeguard | EVIDENCE AGAINST on the tested paths | G0 and G1 are bitwise identical; no correction is requested. Approximation conservativeness does not ensure descent from a coarsely solved subproblem. Other fixed-NLP globalization policies are NOT ISOLATED. |
| 9 | Physical move-box interaction | NOT ISOLATED | Bound saturation makes accurate complementarity important, but move and density bounds were never changed. An asymptote cap is a different solver setting. |
| 10 | Filtered-sensitivity amplification in void | WEAK ASSOCIATION | Void amplification is 30.69x, yet gray elements carry 82.18% of B0-5000 squared design error and within-void amplification-distance correlation is negative. No filter intervention establishes causality. |

Mechanistic check: at call 19 the approximate bound complementarity products
cluster tightly around 1e-7. Their sum is 0.00576020084,
3.2142 times the entire certified scaled beta gain.
One such product normalized by sRow0*move is 0.0155409,
where the fidelity bar is 1e-6 (raw 6.435e-12). The artificial
bounds lie inside the physical box, so reusing their multipliers without checking
physical slacks understates the original-problem residual. The retained first approximate solve even returns a worse model objective
than its feasible starting point (1.0711e-5 including auxiliary terms), despite
conservativeness at the candidate. These arithmetic facts explain why small iterate steps are not evidence of solving the fixed NLP.
The accuracy intervention establishes causality where its registered bars pass;
the arithmetic alone does not establish that it is the sole cause.
