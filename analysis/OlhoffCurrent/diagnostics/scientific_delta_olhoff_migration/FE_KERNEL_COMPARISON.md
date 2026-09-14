# FE_KERNEL_COMPARISON

**FE assembly and eigensolver paths are identical** for a given material law.

| kernel | status | same-state evidence |
|---|---|---|
| `fem/model2D.m`, `fem/elemMats2D.m` (Q4 plane stress, consistent mass, mid-height pins, u_x restrained at both ends) | byte-identical | — |
| `fem/massScale.m`, `+olh/+material/massInterpolation.m` | byte-identical | M SHA-256 equal (T vs S1) at 9 states |
| `fem/assemble2D.m` | source routes g_K through `olh.material.stiffnessInterpolation`; its `simp` branch computes `rho.^p` exactly as the target line | K SHA-256 equal at 9 states |
| `fem/eigSolve.m` | source adds an `opts` pass-through whose defaults (tol 1e−12, maxit 5000, pFactor 4) are the target constants; fixed deterministic v0 unchanged | ω₁…ω₅ bitwise at 9 states; target in-run ω reproduced bitwise at iterations 1, 11, 21, 101 |
| `fem/classifyModes.m` | byte-identical (final analysis only) | — |
| `algo/genGrad.m` | source routes the stiffness derivative through the same function (`p*rho.^(p-1)` for SIMP) | raw f_sk bitwise at 9 states |

Material law is the only FE-level difference (FORMULATION_COMPARISON.md). Eigenvector signs agree
bitwise under identical K, M (deterministic v0), so no sign alignment was needed for the identity
test; it was applied for the formulation comparison.

Performance note (not a scientific difference): the M1 run (single thread, this host, another MATLAB
session in parallel) took 395.7 s for 64 outer iterations; S480 in the committed sweep took 1958 s
for 112 while nine runs shared the machine. Per-iteration cost comparisons across these runs are not
used for any verdict.
