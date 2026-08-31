# Phase 2H test report

No optimizer or production campaign was run.

- Stored-evidence offline regression: PASS (6/6 groups).
- Candidate C MATLAB suite: PASS (11/11).
- Existing Phase 2A harness regression after amendment: PASS (12/12).
- Qualification/preflight negative controls: PASS (7/7 stale/malformed cases; all three
  missing qualifications block production).

The MATLAB suite covers first-batch selection, early mode parking, k=252 ordinals 4/5,
k=594 diagnostic disagreement, adaptive requests through 24 for ordinals 13 and 18, a
late state, binary-D separation, injected eigensolver/invalid/nonfinite/resource failures,
status propagation, and LP/MMA route accounting.

The offline suite reproduces 16,536 selections, maximum ordinal 18, counts 244/69/6/5
above ordinals 3/6/10/12, zero unresolved stored states, the 0.48–0.56 hard-gate plateau,
the k=594 negative control, and the k=833 severe binary mechanism.

