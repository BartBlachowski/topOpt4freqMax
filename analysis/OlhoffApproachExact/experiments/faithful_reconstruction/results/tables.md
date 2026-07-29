## Ablation matrix — outcome per variant

| tag | variant | mesh | inner budget | steps | cont | FC | status | iters | omega1 final (p=3) | % paper | final N | min g12 | classification |
|---|---|---|---:|---|:-:|:-:|---|---:|---:|---:|---:|---:|---|
| V0_CC_160x20_i30 | V0 | 160x20 | 30 | paper-literal | n | n | MAX_ITERATIONS | 300 | 0.02 | 0.0% | 1 | 2.449e-03 | MECHANISM_COLLAPSE |
| V0_CC_240x30_i30 | V0 | 240x30 | 30 | paper-literal | n | n | MAX_ITERATIONS | 120 | 9.08 | 2.0% | 1 | 1.399e-02 | MECHANISM_COLLAPSE |
| V1_CC_160x20_i30 | V1 | 160x20 | 30 | paper-literal | Y | n | MAX_ITERATIONS | 300 | 0.02 | 0.0% | 1 | 8.766e-01 | MECHANISM_COLLAPSE |
| V2_CC_160x20_i30 | V2 | 160x20 | 30 | paper-literal | n | Y | INNER_FAILURE | 1 | 145.57 | 31.9% | 1 | 1.494e+00 | INNER_FAILURE |
| V3_CC_160x20_i30 | V3 | 160x20 | 30 | paper-literal | Y | Y | INNER_FAILURE | 1 | 145.57 | 31.9% | 1 | 1.494e+00 | INNER_FAILURE |
| V4_CC_160x20_i30 | V4 | 160x20 | 30 | regimeB | n | Y | INNER_FAILURE | 1 | 145.57 | 31.9% | 1 | 1.494e+00 | INNER_FAILURE |
| V5_CC_160x20_i30 | V5 | 160x20 | 30 | regimeB | Y | Y | INNER_FAILURE | 1 | 145.57 | 31.9% | 1 | 1.494e+00 | INNER_FAILURE |
| VR_CC_160x20_i30 | VR | 160x20 | 30 | regimeB | n | n | MAX_ITERATIONS | 300 | 328.55 | 72.0% | 1 | 4.781e-03 | OUTER_LIMIT_CYCLE |
| VR_CC_240x30_i30 | VR | 240x30 | 30 | regimeB | n | n | MAX_ITERATIONS | 150 | 371.54 | 81.4% | 1 | 4.654e-03 | OUTER_LIMIT_CYCLE |
| V0_CC_160x20_i2000 | V0 | 160x20 | 2000 | paper-literal | n | n | MAX_ITERATIONS | 15 | 0.04 | 0.0% | 1 | 2.634e-05 | MECHANISM_COLLAPSE |
| V1_CC_160x20_i2000 | V1 | 160x20 | 2000 | paper-literal | Y | n | MAX_ITERATIONS | 15 | 0.04 | 0.0% | 1 | 1.246e+00 | MECHANISM_COLLAPSE |
| V2_CC_160x20_i2000 | V2 | 160x20 | 2000 | paper-literal | n | Y | INNER_FAILURE | 13 | 0.02 | 0.0% | 1 | 2.634e-05 | INNER_FAILURE |
| V3_CC_160x20_i2000 | V3 | 160x20 | 2000 | paper-literal | Y | Y | MAX_ITERATIONS | 15 | 0.04 | 0.0% | 1 | 1.246e+00 | MECHANISM_COLLAPSE |
| V4_CC_160x20_i2000 | V4 | 160x20 | 2000 | regimeB | n | Y | MAX_ITERATIONS | 300 | 300.90 | 65.9% | 1 | 1.477e-03 | OUTER_LIMIT_CYCLE |
| V4_CC_240x30_i2000 | V4 | 240x30 | 2000 | regimeB | n | Y | MAX_ITERATIONS | 150 | 343.04 | 75.2% | 1 | 4.900e-04 | OUTER_LIMIT_CYCLE |
| V5_CC_160x20_i2000 | V5 | 160x20 | 2000 | regimeB | Y | Y | MAX_ITERATIONS | 300 | 312.28 | 68.4% | 1 | 2.009e-02 | MAX_ITERATIONS |
| V5_CC_240x30_i2000 | V5 | 240x30 | 2000 | regimeB | Y | Y | MAX_ITERATIONS | 150 | 248.76 | 54.5% | 1 | 1.346e-02 | MAX_ITERATIONS |
| V5a_CC_160x20_i2000 | V5a | 160x20 | 2000 | regimeB | Y | Y | MAX_ITERATIONS | 300 | 354.49 | 77.7% | 1 | 4.774e-03 | MECHANISM_COLLAPSE |
| V5b_CC_160x20_i2000 | V5b | 160x20 | 2000 | regimeB | Y | Y | MAX_ITERATIONS | 300 | 296.71 | 65.0% | 1 | 5.488e-03 | MAX_ITERATIONS |

## Inner-solve validity

| tag | inner budget | converged / total | rejected outer steps | median inner iters | singular warns |
|---|---:|---:|---:|---:|---:|
| V0_CC_160x20_i2000 | 2000 | 14/15 | 1 | 113 | 2764 |
| V0_CC_160x20_i30 | 30 | 0/300 | 300 | 30 | 8560 |
| V0_CC_240x30_i30 | 30 | 0/120 | 120 | 30 | 2331 |
| V1_CC_160x20_i2000 | 2000 | 15/15 | 0 | 65 | 1015 |
| V1_CC_160x20_i30 | 30 | 0/300 | 300 | 30 | 8293 |
| V2_CC_160x20_i2000 | 2000 | 12/13 | 1 | 129 | 2585 |
| V2_CC_160x20_i30 | 30 | 0/1 | 1 | 30 | 0 |
| V3_CC_160x20_i2000 | 2000 | 15/15 | 0 | 65 | 1015 |
| V3_CC_160x20_i30 | 30 | 0/1 | 1 | 30 | 0 |
| V4_CC_160x20_i2000 | 2000 | 300/300 | 0 | 152 | 0 |
| V4_CC_160x20_i30 | 30 | 0/1 | 1 | 30 | 0 |
| V4_CC_240x30_i2000 | 2000 | 150/150 | 0 | 164 | 7 |
| V5_CC_160x20_i2000 | 2000 | 300/300 | 0 | 204 | 0 |
| V5_CC_160x20_i30 | 30 | 0/1 | 1 | 30 | 0 |
| V5_CC_240x30_i2000 | 2000 | 150/150 | 0 | 188 | 0 |
| V5a_CC_160x20_i2000 | 2000 | 300/300 | 0 | 210 | 0 |
| V5b_CC_160x20_i2000 | 2000 | 300/300 | 0 | 209 | 0 |
| VR_CC_160x20_i30 | 30 | 0/300 | 300 | 30 | 0 |
| VR_CC_240x30_i30 | 30 | 0/150 | 150 | 30 | 0 |

## Multiplicity audit

| tag | first N=2 (pre) | first N=2 (post) | # N=2 iters | # N=2 @tol 1e-2 | min g12 | final N | final g12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0_CC_160x20_i2000 | 5 | 4 | 1 | 1 | 2.634e-05 | 1 | 1.545e+00 |
| V0_CC_160x20_i30 | -1 | -1 | 0 | 1 | 2.449e-03 | 1 | 2.139e+00 |
| V0_CC_240x30_i30 | -1 | -1 | 0 | 0 | 1.399e-02 | 1 | 8.652e-01 |
| V1_CC_160x20_i2000 | -1 | -1 | 0 | 0 | 1.246e+00 | 1 | 1.581e+00 |
| V1_CC_160x20_i30 | -1 | -1 | 0 | 0 | 8.766e-01 | 1 | 2.099e+00 |
| V2_CC_160x20_i2000 | 5 | 4 | 1 | 1 | 2.634e-05 | 1 | 1.821e+00 |
| V2_CC_160x20_i30 | -1 | -1 | 0 | 0 | nan | 1 | 1.494e+00 |
| V3_CC_160x20_i2000 | -1 | -1 | 0 | 0 | 1.246e+00 | 1 | 1.581e+00 |
| V3_CC_160x20_i30 | -1 | -1 | 0 | 0 | nan | 1 | 1.494e+00 |
| V4_CC_160x20_i2000 | -1 | -1 | 0 | 1 | 1.477e-03 | 1 | 3.863e-01 |
| V4_CC_160x20_i30 | -1 | -1 | 0 | 0 | nan | 1 | 1.494e+00 |
| V4_CC_240x30_i2000 | 24 | 23 | 1 | 1 | 4.900e-04 | 1 | 3.058e-01 |
| V5_CC_160x20_i2000 | -1 | -1 | 0 | 0 | 2.009e-02 | 1 | 2.591e-01 |
| V5_CC_160x20_i30 | -1 | -1 | 0 | 0 | nan | 1 | 1.494e+00 |
| V5_CC_240x30_i2000 | -1 | -1 | 0 | 0 | 1.346e-02 | 1 | 1.863e-01 |
| V5a_CC_160x20_i2000 | -1 | -1 | 0 | 1 | 4.774e-03 | 1 | 7.090e-02 |
| V5b_CC_160x20_i2000 | -1 | -1 | 0 | 3 | 5.488e-03 | 1 | 1.430e-01 |
| VR_CC_160x20_i30 | -1 | -1 | 0 | 4 | 4.781e-03 | 1 | 3.004e-01 |
| VR_CC_240x30_i30 | -1 | -1 | 0 | 2 | 4.654e-03 | 1 | 2.190e-01 |

## Terminal-behaviour classification

| tag | class | obj stationary | d1_rms final | d1_inf final | lag2/lag1 | lag3/lag1 | log10 decay slope | omega1 tail CV |
|---|---|:-:|---:|---:|---:|---:|---:|---:|
| V0_CC_160x20_i2000 | MECHANISM_COLLAPSE | n | 9.925e-01 | 9.990e-01 | 0.173 | 0.990 | 1.076e-02 | 2.739e+00 |
| V0_CC_160x20_i30 | MECHANISM_COLLAPSE | n | 9.933e-01 | 9.990e-01 | 0.000 | 1.000 | 9.128e-09 | 4.457e-01 |
| V0_CC_240x30_i30 | MECHANISM_COLLAPSE | n | 9.820e-01 | 9.987e-01 | 0.001 | 1.000 | 3.182e-07 | 9.874e-01 |
| V1_CC_160x20_i2000 | MECHANISM_COLLAPSE | n | 9.929e-01 | 9.990e-01 | 0.139 | 0.994 | 8.122e-03 | 2.130e+00 |
| V1_CC_160x20_i30 | MECHANISM_COLLAPSE | n | 9.935e-01 | 9.990e-01 | 0.009 | 1.000 | 1.423e-06 | 4.295e-01 |
| V2_CC_160x20_i2000 | INNER_FAILURE | n | nan | nan | 0.184 | 0.999 | 1.564e-02 | 2.416e+00 |
| V2_CC_160x20_i30 | INNER_FAILURE | n | nan | nan | nan | nan | nan | nan |
| V3_CC_160x20_i2000 | MECHANISM_COLLAPSE | n | 9.929e-01 | 9.990e-01 | 0.139 | 0.994 | 8.122e-03 | 2.130e+00 |
| V3_CC_160x20_i30 | INNER_FAILURE | n | nan | nan | nan | nan | nan | nan |
| V4_CC_160x20_i2000 | OUTER_LIMIT_CYCLE | Y | 8.670e-02 | 1.000e-01 | 0.189 | 1.003 | 2.304e-05 | 1.904e-03 |
| V4_CC_160x20_i30 | INNER_FAILURE | n | nan | nan | nan | nan | nan | nan |
| V4_CC_240x30_i2000 | OUTER_LIMIT_CYCLE | Y | 7.604e-02 | 1.000e-01 | 0.190 | 1.004 | 5.362e-05 | 2.551e-03 |
| V5_CC_160x20_i2000 | MAX_ITERATIONS | n | 5.808e-02 | 1.000e-01 | 1.044 | 1.076 | -1.525e-03 | 2.592e-01 |
| V5_CC_160x20_i30 | INNER_FAILURE | n | nan | nan | nan | nan | nan | nan |
| V5_CC_240x30_i2000 | MAX_ITERATIONS | n | 6.156e-02 | 1.000e-01 | 1.092 | 1.098 | -6.230e-04 | 2.225e-01 |
| V5a_CC_160x20_i2000 | MECHANISM_COLLAPSE | n | 7.259e-02 | 1.000e-01 | 1.060 | 1.121 | -1.447e-03 | 2.581e-01 |
| V5b_CC_160x20_i2000 | MAX_ITERATIONS | n | 6.835e-02 | 1.000e-01 | 1.120 | 1.096 | -8.604e-04 | 2.808e-01 |
| VR_CC_160x20_i30 | OUTER_LIMIT_CYCLE | Y | 5.685e-02 | 1.000e-01 | 0.142 | 1.001 | -5.938e-06 | 1.028e-03 |
| VR_CC_240x30_i30 | OUTER_LIMIT_CYCLE | Y | 4.244e-02 | 9.999e-02 | 0.084 | 1.001 | 1.251e-06 | 6.188e-03 |

## Topology descriptors (final design)

| tag | 8conn | extra members | spanning | centre-third rho | grey frac | y-symmetry | x-symmetry |
|---|---:|---:|:-:|---:|---:|---:|---:|
| V0_CC_160x20_i2000 | 3 | 2 | n | 0.608 | 0.001 | 1.000 | 1.000 |
| V0_CC_160x20_i30 | 2 | 1 | n | 0.449 | 0.010 | 1.000 | 0.979 |
| V0_CC_240x30_i30 | 2 | 1 | n | 0.002 | 0.002 | 1.000 | 0.805 |
| V1_CC_160x20_i2000 | 3 | 2 | n | 0.502 | 0.001 | 1.000 | 1.000 |
| V1_CC_160x20_i30 | 2 | 1 | n | 0.430 | 0.010 | 1.000 | 1.000 |
| V2_CC_160x20_i2000 | 2 | 1 | n | 0.386 | 0.001 | 1.000 | 1.000 |
| V2_CC_160x20_i30 | 1 | 0 | Y | 0.500 | 1.000 | nan | nan |
| V3_CC_160x20_i2000 | 3 | 2 | n | 0.502 | 0.001 | 1.000 | 1.000 |
| V3_CC_160x20_i30 | 1 | 0 | Y | 0.500 | 1.000 | nan | nan |
| V4_CC_160x20_i2000 | 5 | 0 | Y | 0.306 | 0.772 | 0.999 | 0.947 |
| V4_CC_160x20_i30 | 1 | 0 | Y | 0.500 | 1.000 | nan | nan |
| V4_CC_240x30_i2000 | 3 | 0 | Y | 0.244 | 0.604 | 0.971 | 0.954 |
| V5_CC_160x20_i2000 | 3 | 1 | n | 0.215 | 0.664 | 0.606 | 0.802 |
| V5_CC_160x20_i30 | 1 | 0 | Y | 0.500 | 1.000 | nan | nan |
| V5_CC_240x30_i2000 | 11 | 1 | n | 0.119 | 0.374 | 0.682 | 0.731 |
| V5a_CC_160x20_i2000 | 10 | 2 | n | 0.164 | 0.639 | 0.676 | 0.884 |
| V5b_CC_160x20_i2000 | 9 | 1 | n | 0.144 | 0.599 | 0.722 | 0.736 |
| VR_CC_160x20_i30 | 1 | 0 | Y | 0.324 | 0.570 | 1.000 | 0.977 |
| VR_CC_240x30_i30 | 2 | 0 | Y | 0.310 | 0.348 | 1.000 | 0.978 |

## Acceptance gates

| tag | G1 inner | G2 no-mech | G3 feasible | G4 spectral | G5 multiplicity | G6 trajectory | G7 mesh | G8 topology | passed |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---:|
| V0_CC_160x20_i2000 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 1/8 |
| V0_CC_160x20_i30 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | 2/8 |
| V0_CC_240x30_i30 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | 2/8 |
| V1_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V1_CC_160x20_i30 | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 1/8 |
| V2_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V2_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V3_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V3_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V4_CC_160x20_i2000 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | FAIL | 5/8 |
| V4_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V4_CC_240x30_i2000 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | 6/8 |
| V5_CC_160x20_i2000 | PASS | FAIL | PASS | PASS | FAIL | FAIL | PASS | FAIL | 4/8 |
| V5_CC_160x20_i30 | PASS | PASS | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 3/8 |
| V5_CC_240x30_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | 3/8 |
| V5a_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| V5b_CC_160x20_i2000 | PASS | FAIL | PASS | FAIL | FAIL | FAIL | n/a | FAIL | 2/8 |
| VR_CC_160x20_i30 | FAIL | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | 5/8 |
| VR_CC_240x30_i30 | FAIL | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | 5/8 |
