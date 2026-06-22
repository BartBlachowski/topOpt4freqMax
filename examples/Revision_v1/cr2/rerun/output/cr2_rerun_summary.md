# CR2 Protocol Rerun Summary

- Outcome category: **DIAGNOSTIC ALGORITHM-FAILURE EVIDENCE**
- Accepted converged comparison eligible: false (settings intentionally unmatched)
- Scientific claim: `withheld_unmatched_stabilization_screen`
- Load: `F(x) = omega0^2 * M(x) * Phi0`
- Acceptance: design change <= 1e-3, feasibility <= 1e-4, MAC >= 0.8, not capped, all artifacts present.
- Further tuning performed: no

## Variant A (omitted)

- Accepted: 0
- Protocol run outcome: `inconclusive`
- Solver success: 1
- Iterations: 400 / 400
- Final design change: 5.244782e-03
- Final feasibility: 9.293465e-05
- Final tracked mode: 1, MAC 0.999310
- Final-50 objective plateau: 1
- Algorithm-failure signature: 0
- Failures: iteration cap reached; design change 5.245e-03 exceeds 1.000e-03

## Variant B (complete)

- Accepted: 0
- Protocol run outcome: `diagnostic algorithm-failure evidence`
- Solver success: 1
- Iterations: 400 / 400
- Final design change: 2.000000e-02
- Final feasibility: 0.000000e+00
- Final tracked mode: 2, MAC 0.990721
- Final-50 objective plateau: 1
- Algorithm-failure signature: 1
- Failures: iteration cap reached; design change 2.000e-02 exceeds 1.000e-03

This unmatched stabilization screen permits no converged A/B endpoint comparison.
