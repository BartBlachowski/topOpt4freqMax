# Computational cost

Phase-2F measured sparse shift-invert solves of 0.07 s at 160x20 and 2.97 s at 720x90 for
12 modes, with scaling approximately `T proportional N_el^1.264`. At 160x20, requesting
3/6/12/24/48 modes cost 0.0465/0.0550/0.0663/0.0903/0.1557 s. Eigenvector energy reduction
added 9.3% at 160x20 and 2.9% at 720x90; 12 eigenvectors used 12.6 MB at 720x90.

Only 244 of 16,536 records required more than three modes; 69
required more than six, and five required the 12-to-24 escalation. A geometric schedule is
therefore practical. Reusing a factorization/subspace is an implementation optimization,
not a scientific rule.

The Phase-2F estimate is about 30 single-threaded hours for three methods x three evaluators
over the eight available meshes at 1,600 states. Extrapolating the measured scaling to the
missing 800x100 mesh gives roughly 45 hours for all nine meshes, or order 6 hours with
eight-way state/mesh parallelism. This remains practical post hoc. All evaluator work stays
outside native optimizer timing.
