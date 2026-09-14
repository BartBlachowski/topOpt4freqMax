# Final-harness impact

Nothing was edited in this audit. A later controlled implementation must:

1. add the C modal classifier, adaptive batches, eigenvector diagnostics, residual checks,
   selected ordinal/frequency, vote margins, IPR, requested-mode counts, and explicit
   `STRUCTURAL_MODE_NOT_FOUND` status;
2. route C-selected E1/E2/E3 values into `Q`, reference, persistence, and tables while
   leaving the hard topology gate separate and unchanged;
3. keep evaluator wall time outside optimizer timing and report it separately;
4. extend regression tests, negative controls, provenance hashes, and precision-artifact
   binding;
5. retain E1/E2/E3 and robust normalized minimum; do not simplify the family in this phase;
6. support `olhoff.variant = 'lp' | 'mma' | 'both'`.

Olhoff accounting remains route-specific:

- LP (principal): outer updates and LP calls, separately; one successful LP call per outer
  in the settled route. Do not call simplex/HiGHS internals optimizer iterations.
- Nested MMA (secondary): outer updates; total MMA inner iterations; mean, median, and p95
  inner iterations; cap hits; and converged-inner fraction. Never fold these into a
  fictitious universal iteration count.

The LP/MMA selection is unchanged: LP is the principal comparator and nested MMA is the
secondary paper-literal sensitivity variant.
