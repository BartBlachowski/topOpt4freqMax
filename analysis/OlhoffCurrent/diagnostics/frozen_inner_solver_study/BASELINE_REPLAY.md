# Exact baseline reproduction

The production innerLoop itself returned bitwise-identical drho and beta at
nInner=19 under its unchanged tolerance. The independently instrumented replay
matches at 19 and matches the earlier 500-call endpoint bit-for-bit. A distinct
persistence-boundary execution also matches call 50 bit-for-bit.

| Iteration | Source | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Relative step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 19 | fresh + authenticated reference | 0.277% | 0.994774 | 1.000406 | 94.419% | 0.000% | 5.032e-02 | 4.118e-02 |
| 50 | fresh + authenticated reference | 12.186% | 0.984487 | 1.000405 | 94.850% | 0.000% | 5.319e-02 | 1.883e-02 |
| 100 | fresh + authenticated reference | 20.441% | 0.969934 | 1.000294 | 96.331% | 0.000% | 4.376e-02 | 9.638e-03 |
| 500 | fresh + authenticated reference | 39.198% | 0.921062 | 1.005293 | 81.725% | 0.000% | 6.345e-02 | 3.724e-03 |
| 1000 | authenticated retained replay | 44.847% | 0.908687 | 1.013407 | 80.121% | 0.000% | 3.984e-02 | 2.922e-03 |
| 5000 | authenticated retained replay | 53.702% | 0.829748 | 1.016775 | 92.065% | 0.000% | 8.752e-02 | 2.077e-03 |


Fresh B0 work: 500 calls plus the separate 19-call production reproduction;
solver wall for the 500 calls: 597.415s.
The reference 5000-call replay cost 5984.418s in its original study. It was not
rerun in full: all 113 retained checkpoints were hash-authenticated and freshly
re-evaluated, avoiding a redundant roughly 100-minute control. Per-checkpoint
retained timings are unavailable. This limitation was preregistered.

At 5000, objective and design distances have decreased from production-19, but
only 53.702% of gain is recovered and d2 remains 0.829748.
No oracle-active bound is reached within the declared bound tolerance. Full
active agreement is 0.056%, accounted for by oracle-interior
entries, not recovery of the saturated optimum. No convergence claim follows.

All checkpoint quantities, duals and asymptotes available in the original data
are in MAT/JSON; MASTER_METRICS.csv supplies the common scalar definitions.
