# Asymptote initialization

Production already uses the official .5 initialization for the first two calls.
The historical .01 value is verified in the inactive as-found copy. S2 changes
only that value; it does not import the other historical settings.

| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0_CURRENT_REPEATED_MMA | 500 | 39.198% | 0.921062 | 1.00529 | 81.725% | 0.000% | 6.345e-02 | FAIL |
| S2_ASYINIT_001 | 500 | 14.481% | 0.945952 | 1.00711 | 75.989% | 0.000% | 6.220e-02 | FAIL |


The historical value changes the trajectory but does not repair oracle fidelity.
It cannot be the nonstandard initialization of the active frozen path, which
already uses .5. Initial low=xmin and upp=xmax are overwritten on calls 1 and 2.
No parameter search was performed.
