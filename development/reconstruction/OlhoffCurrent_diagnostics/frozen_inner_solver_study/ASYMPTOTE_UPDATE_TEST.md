# Asymptote update policy

Both source copies use the 1.2/.7 sign-based history update and the .01 minimum
asymptote distance. The active copy caps maximum distance at .2 widths; official
Svanberg source caps it at 10. Its name and README did not disclose this remaining
local modification. S3 changes exactly those two symmetric cap expressions.

| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0_CURRENT_REPEATED_MMA | 500 | 39.198% | 0.921062 | 1.00529 | 81.725% | 0.000% | 6.345e-02 | FAIL |
| S3_CANONICAL_CLAMP | 500 | 22.840% | 0.728821 | 1.02982 | 97.286% | 0.000% | 1.419e-02 | FAIL |
| S4_SUBSOLV_ACCURACY | 455 | 77.965% | 0.824808 | 1.00324 | 99.219% | 22.589% | 1.224e-02 | FAIL |
| S34_CLAMP_ACCURACY | 500 | 99.935% | 0.199492 | 1.10236 | 99.784% | 22.589% | 3.524e-04 | FAIL |


The 2x2 comparison separates cap and approximate-solve accuracy. S34 is a joint
intervention; it cannot alone assign causality to either factor. All four edges
and the predefined material-effect bars appear in CAUSAL_ATTRIBUTION.md.
The outer move box is unchanged throughout. A maximum MMA asymptote distance
is different from the physical increment move limit.
