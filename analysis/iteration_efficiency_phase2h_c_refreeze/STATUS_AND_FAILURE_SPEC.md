# Candidate C status and failure specification

If any of E1/E2/E3 cannot select a unanimously valid mode because the eigensolver fails,
eigenpairs are invalid, diagnostics are nonfinite, or the technical search space is
exhausted, the common evaluator fails closed with `STRUCTURAL_MODE_NOT_FOUND`. Q is NaN for
that state. Such a state is neither a low frequency nor a topology failure.

The status propagates through the reference and measurement phases. It has precedence
after successful statuses and before reference-solver, solver, topology, quality, and
persistence failure classifications. Error detail and per-evaluator search evidence are
retained.

