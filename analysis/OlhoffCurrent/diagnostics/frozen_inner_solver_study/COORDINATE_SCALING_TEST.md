# Coordinate scaling

S5 uses t_e=(drho_e-lo_e)/(hi_e-lo_e), with t in [0,1]. Beta remains beta/lamref.
All physical values, derivatives, fixed box bounds and state are transformed
consistently; the stopping step is evaluated in physical increment coordinates.
No candidate density coordinates are formed.

| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0_CURRENT_REPEATED_MMA | 500 | 39.198% | 0.921062 | 1.00529 | 81.725% | 0.000% | 6.345e-02 | FAIL |
| S5_UNIT_BOX | 500 | 52.797% | 0.816843 | 1.00866 | 95.981% | 0.000% | 1.378e-02 | FAIL |


The final S5 solver time is 28.399s for 500
calls, compared with 597.415s in B0.
This is a large observed numerical-work effect, subject to shared-host timing
limits. It does not meet the registered joint fidelity or strong-effect bars.
The unchanged 1e-7 approximate-solve target still imposes its accuracy floor.

Mathematically the MMA reciprocal approximation transforms consistently under
this positive affine map: gradients acquire a box-width factor, asymptote
intervals lose it, and the 1/width regularizer compensates. The code's 1e-5
minimum-width guard is inactive on both representations. Finite Newton stopping,
residual norms, initialization safeguards and floating-point cancellation are
not generally invariant. Thus an affine reformulation can change numerical
work and the finite-accuracy path without changing problem (25). A separate
beta/objective or constraint-row scaling intervention was not registered and is
NOT ISOLATED. No extra scaled variant was added after seeing the results.
