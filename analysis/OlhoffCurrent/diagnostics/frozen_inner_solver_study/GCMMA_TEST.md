# GCMMA and conservative acceptance

GCMMA_APPROACHES_BUT_DOES_NOT_REACH_ORACLE

The repository had no GCMMA routines. The official GPLv3 distribution was
obtained from [Svanberg's site](https://www.smoptit.se/), frozen with SHA-256,
and retained audit-only with its license. Numerical subsolv expressions match
the local copy; the upstream version adds a diagnostic print near iteration
limits. Toy reproduction and an analytic constrained quadratic validate the
interface. The printed nine-iteration toy sequence alone has KKT 1.447e-5;
continuing it reaches 5.420e-7 and passes the validation bar.

| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0_CURRENT_REPEATED_MMA | 500 | 39.198% | 0.921062 | 1.00529 | 81.725% | 0.000% | 6.345e-02 | FAIL |
| G0_UNSAFE | 500 | 19.886% | 0.733473 | 1.014 | 98.808% | 0.000% | 1.372e-02 | FAIL |
| G1_GCMMA | 500 | 19.886% | 0.733473 | 1.014 | 98.808% | 0.000% | 1.372e-02 | FAIL |
| G2_GCMMA_ACCURATE | 500 | 99.911% | 0.216586 | 1.07377 | 99.722% | 22.596% | 4.679e-04 | FAIL |


G0 and G1 isolate only the conservative acceptance/correction switch. Their
500-call trajectories agree exactly and G1 requires zero corrections. The
maximum true-minus-approximation value is -4.803e-9: all its accepted trials
are conservative even without the 1e-7 allowance. B0's corresponding maximum
is -2.861e-7. Therefore lack of a conservative safeguard is not an observed
failure mechanism on these paths.

G2 changes only approximate-solve epsimin to 1e-12; concheck remains at 1e-7.
It reaches 99.911% recovery, d2=0.216586,
dinf=1.073769, and KKT=4.679e-04. Its strict fidelity result is
FAIL. Correction and approximation iterations,
including every raa value and trial check, are retained in the JSON/MAT files.
The tight solve emits ill-conditioned-system warnings; exact-problem residuals,
not suppressed warnings or relative steps, determine the verdict.

The comparison of native GCMMA against B0 changes cap and regularization policy
as well as method; it is not an isolated globalization comparison. No global
convergence claim beyond this frozen test is made.
