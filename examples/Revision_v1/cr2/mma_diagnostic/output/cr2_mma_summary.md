# CR2-MMA Diagnostic Summary

- **Outcome:** INCONCLUSIVE — neither variant converged within 400 iterations under MMA
- Optimizer: MMA (both variants, replacing OC)
- Load: F(x) = omega0^2 * M(x) * Phi0
- Acceptance: dc <= 1e-03, feas <= 1e-04, MAC >= 0.8, not capped, all artifacts present
- No further tuning performed

## Variant A (omitted)

- Accepted: 0
- Solver success: 1
- Iterations: 400 / 400
- Final design_change: 7.402e-02
- Final feasibility: 0.000e+00
- Final MAC: 0.8116 (mode 1 vs solid reference)
- Final omega_1: 2.771 rad/s (post-loop eigensolver)
- Timing: 34.6 s
- Algorithm failure signature: False (no move saturation, no period-2, no stable cycle)
- Failures: iteration cap reached; design_change 7.402e-02 > 1.0e-03

## Variant B (complete)

- Accepted: 0
- Solver success: 1
- Iterations: 400 / 400
- Final design_change: 1.811e-02
- Final feasibility: 0.000e+00
- Final MAC: 0.9921 (best match is mode 2 vs solid reference)
- Final omega_1: 132.6 rad/s (post-loop eigensolver)
- Timing: 27.6 s
- Algorithm failure signature: False (no move saturation, no period-2, no stable cycle)
- Failures: iteration cap reached; design_change 1.811e-02 > 1.0e-03

## Topology Difference

| Metric | Value |
|--------|-------|
| MAD | 0.543 |
| RMS | 0.704 |
| Max |Δ| | 0.998 |
| |Δ|>0.05 fraction | 0.632 |
| Pearson r | -0.111 |
