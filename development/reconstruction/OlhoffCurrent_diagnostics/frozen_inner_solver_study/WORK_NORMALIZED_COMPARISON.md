# Work-normalized comparison

| Method | Approx. solves | Algorithm constraint evaluations | Gradient requests | Solver wall s | KKT | Recovery | d2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_CURRENT_REPEATED_MMA | 500 | 500 | 500 | 597.415 | 6.345e-02 | 39.198% | 0.921062 |
| S2_ASYINIT_001 | 500 | 500 | 500 | 651.651 | 6.220e-02 | 14.481% | 0.945952 |
| S3_CANONICAL_CLAMP | 500 | 500 | 500 | 351.493 | 1.419e-02 | 22.840% | 0.728821 |
| S4_SUBSOLV_ACCURACY | 455 | 455 | 455 | 1803.726 | 1.224e-02 | 77.965% | 0.824808 |
| S5_UNIT_BOX | 500 | 500 | 500 | 28.399 | 1.378e-02 | 52.797% | 0.816843 |
| G0_UNSAFE | 500 | 1000 | 500 | 221.853 | 1.372e-02 | 19.886% | 0.733473 |
| G1_GCMMA | 500 | 1000 | 500 | 207.889 | 1.372e-02 | 19.886% | 0.733473 |
| G2_GCMMA_ACCURATE | 500 | 1000 | 500 | 646.658 | 4.679e-04 | 99.911% | 0.216586 |
| S34_CLAMP_ACCURACY | 500 | 500 | 500 | 738.385 | 3.524e-04 | 99.935% | 0.199492 |


All terminal approximation comparisons receive at most 500 solves, including
rejected GCMMA trials. Equal-call values at 100 and 500 determine causal labels;
no method wins from receiving an extra two orders of magnitude of work. The
retained B0-5000 history is contextual long-run evidence, not an equal-budget
competitor. SOCP uses a different cone interior-point algorithm; its internal
iterations are not mmasub calls. Assembly, solve, certification and total process
memory are separately reported in SOCP_ORACLE_COST.md.

The exact production evaluator computes gradients even when GCMMA requests
only trial values. Thus GCMMA's actually computed algorithm gradients equal its
constraint evaluations (accepted bases plus trials), while the gradient-request
column follows the algorithm's logical interface. Audit metrics require four
additional combined evaluations per new MMA iteration and three per accepted
GCMMA iteration; final CSV/JSON count these separately. Raw run logs originally
counted one metric event per iterate, corrected by source-based accounting in
fi_outputs.py. No solver sequence depends on this accounting correction.

Only direct SOCP reaches all preregistered oracle fidelity bars. It is therefore
the only measured method eligible for 'cheapest reaching fidelity'; a cheaper
inaccurate endpoint is not ranked as a success. Concurrent host load limits wall
time interpretation. Figures 15,16,21 show function-work and call-work curves.
