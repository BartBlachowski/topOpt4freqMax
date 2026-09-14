# Newly supplied Pedersen (2000) paper

User supplied `docs/s001580050130.pdf` during this study: N.L. Pedersen,
“Maximization of eigenvalues using topology optimization”, Structural and
Multidisciplinary Optimization 20, 2-11 (2000).

Useful evidence: section 2 explains low-density localized eigenmodes and the
influence of mass/stiffness interpolation. Equation (6), printed p.6, gives the
density-weighted sensitivity filter with the inverse receiving-density factor;
it is directly relevant background for possible amplification in void regions.
Printed p.7 identifies MMA and cites separate multiple-eigenvalue treatment.

It does not specify MMA state persistence, asymptote constants, approximate
subproblem accuracy, a numerical inner stopping criterion, or conservative
acceptance. It therefore does not resolve the fixed-problem solver question.
Its interpolation and node-handling proposals alter the physics/eigensolver
and are outside the zero-update lock. No variant was added or changed because
of this paper. It is contextual literature, not a new causal experiment.
