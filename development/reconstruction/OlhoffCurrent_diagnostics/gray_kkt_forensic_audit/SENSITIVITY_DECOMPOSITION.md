# Elementwise spectral decomposition

All three native frozen-state eigensolves and the complete raw/filtered N=2 generalized tensors are retained. F=gK+gM with gM signed negative for a diagonal mode; off-diagonal contributions can have either sign. Diagonal blocks use their own lambda; off-diagonals use lambda1; fJJ is retained separately. No optimized increment was computed. The plotted active derivative is obtained from native deltaLambda at zero increment with the recorded diagonal offsets.

The lowest eigenvalue is simple at each exact saved state, including 800: its frequency gap is small but approximately 3.49e-5 and well above the measured eigen residual. The first derivative at zero is therefore the first diagonal block. This statement is local; it does not treat the 800 optimization path as single-mode. Its nearby nonsmooth optimum may involve both modes, and the full tensor robustness analysis in KKT_STATIONARITY.md is essential.

The maps F05–F07 show gLambda, stiffness and signed mass in a common mesh-comparable scale NE/lambda1. Color limits clip pooled 99.5% tails, disclosed on titles; numeric files retain all values. Raw unitful distributions are in stationarity.json. The derivative is of lambda=omega²; divide by 2omega for omega derivatives. That positive conversion does not change first-order stationarity if multipliers/scales are converted consistently.

Eigen residuals are ~2.3e-10,3.4e-10,9.3e-10 for the first mode; mass-orthogonality Frobenius errors are below 7e-14. These are measured residuals after assembly, not claims of exact arithmetic or a proof of FE discretization accuracy.
