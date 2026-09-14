# MULTIPLICITY_COMPARISON

**Identical.** Code: `+olh/+multi/detect.m`, `algo/deltaLambda.m`, the multiplicity/gradient block of
`olhoffSolve.m` (textually identical apart from the stiffness-law argument), `innerLoop.m`.
Configuration: `multiplicity.method = subspace`, `subspaceSize = 2`, `diagonalOffsets = true`,
`offDiagonal = true`, `tolerance = 0.05`, `eigen.maxCluster = 4` in all four 480×60 configs.

| item | both implementations | same-state result |
|---|---|---|
| detection | none: N fixed at 2 (class C) | N equal at 9 states |
| cluster | idx = {1, 2}; J = 3; Jcalc = 5 eigenpairs | equal |
| λ̃ | λ₁ (first of cluster, §3.5.1) | bitwise |
| diagonal blocks | f_jj at the mode's own λ_j | bitwise |
| off-diagonal | f₁₂ at λ̃, full (25d) determinant (erratum form) | bitwise |
| dOff | λ_idx − λ₁ added on the diagonal of A | bitwise |
| next mode | (25b) row with f_JJ at λ_J; ω_J multiple → logged only | multJ flags equal |
| inner gradients | ∂Δλ_j/∂Δρ = Σ_{s,k} v_js v_jk f_sk | rows bitwise |

Behavioural differences arise only through the design trajectory: S480 ends with gap12 22.5 %
(native), C480 13.0 %; the "ω_J is itself multiple" flag fired at 2 iterations in S480 (4, 6),
4 in C480 (12, 13, 15, 16) and 13 in M1 (4, 20, 22, 25, 38, 53–57, 59, 63, 64 — mostly at its
localized-mode spikes). None of these is an implementation difference.
