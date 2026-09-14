# Approximate-solve accuracy mechanism

The production primal-dual Newton solver reduces its barrier parameter from 1,
stopping at epsimin=1e-7. At production call 19, both artificial bound products
xi*(x-alfa) and eta*(beta-x) have medians approximately 1e-7. Their sum is
0.00576020084; the oracle's total scaled objective gain is
only 0.001792128848. The sum is an approximate-subproblem
central-path gap contribution, not the exact NLP's total duality gap. It exposes
an accuracy budget poorly matched to this small frozen objective gain.

The exact-problem normalized complementarity bar of 1e-6 corresponds here to
raw 6.435e-12, far below 1e-7. The S4 accuracy choice 1e-12 was
registered from this scale before solver outcomes; it was not tuned to topology.

| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0_CURRENT_REPEATED_MMA | 500 | 39.198% | 0.921062 | 1.00529 | 81.725% | 0.000% | 6.345e-02 | FAIL |
| S4_SUBSOLV_ACCURACY | 455 | 77.965% | 0.824808 | 1.00324 | 99.219% | 22.589% | 1.224e-02 | FAIL |
| S3_CANONICAL_CLAMP | 500 | 22.840% | 0.728821 | 1.02982 | 97.286% | 0.000% | 1.419e-02 | FAIL |
| S34_CLAMP_ACCURACY | 500 | 99.935% | 0.199492 | 1.10236 | 99.784% | 22.589% | 3.524e-04 | FAIL |
| G1_GCMMA | 500 | 19.886% | 0.733473 | 1.014 | 98.808% | 0.000% | 1.372e-02 | FAIL |
| G2_GCMMA_ACCURATE | 500 | 99.911% | 0.216586 | 1.07377 | 99.722% | 22.596% | 4.679e-04 | FAIL |


Comparing S4 with B0 isolates accuracy under the current cap. Comparing G2 with
G1 isolates accuracy under native GCMMA's cap and regularization. Comparing
S34 with S3 isolates accuracy at cap 10 with production regularization. The
factorial's other edge isolates the cap after accuracy is tightened. Their
predefined evidence grades appear in CAUSAL_ATTRIBUTION.md.

Even an accurately solved MMA approximation need not yet solve the original
nonlinear problem: approximation curvature and limited motion can still leave
a large oracle design distance. This separates approximate-subproblem accuracy
from exact-problem convergence and from a conservative-acceptance safeguard.
No density, beta objective scaling, gradient filter or physical box was changed.
The variant names refer to requested accuracy; near-singular-system warnings
mean the requested tolerance is not itself a certificate. Exact NLP KKT and
oracle metrics decide success.

S34 also has a visible finite-accuracy setback: gain recovery falls from
0.997148 at call 210 to 0.647486 at call 211, with exact KKT rising to 0.237262,
while primal feasibility is preserved. It subsequently recovers. G2's recorded
gain is monotone over its 500 calls, but still fails design/KKT fidelity. No
Newton-state trace isolates the precise cause of S34's single setback; it must
not be hidden by reporting only the favorable terminal objective.

The retained first-call model gives an additional direct check: x=0, bs=1,
y=z=0 is feasible in that MMA approximation with objective -1. The returned
point has total approximate objective worse by 1.0711e-5, while true objective
is worse by 9.6343e-6. Every model component is conservative at the returned
point (maximum true-minus-approximate=-5.7687e-7). Thus failure of approximate
objective descent is demonstrated before any global-NLP attribution. See
evaluations/initial_model_diagnostic.json; no additional solve was used.
