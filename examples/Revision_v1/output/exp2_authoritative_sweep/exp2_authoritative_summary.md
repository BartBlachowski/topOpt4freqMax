# Exp2 Authoritative Alpha Sweep Summary

Implementation language: MATLAB.
Scope: Exp2 authoritative clamped-beam alpha sweep only. No Python, Exp3, A4, S1, P1, CR2, or manuscript edits.

Authoritative load: F(x) = omega0^2 * M(x) * Phi0
Mesh/settings: 200x25, max_iters=2000, design_change_tolerance=0.001, feasibility_tolerance=1e-08, MAC threshold=0.8

| alpha | classification | iterations | design_change | feasibility | tracked_mode | tracked_MAC | A5 lowest-mode | grayness | omega1 rad/s |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1.00 | accepted | 1052/2000 | 0.00016619 | 0 | 1 | 0.992533 | pass | 0.0857778 | 141.789825 |
| 0.75 | accepted | 1/2000 | 0.000685727 | 0 | 1 | 1.000000 | pass | 1 | 145.557221 |
| 0.50 | mode invalid | 1241/2000 | 0.00054274 | 0 | 1 | 0.748396 | pass | 0.0445347 | 2.975271 |
| 0.25 | mode invalid | 2000/2000 | 0.15522 | 0 | 1 | 0.793750 | pass | 0.0840965 | 2.589367 |
| 0.00 | accepted | 1/2000 | 0.000977643 | 0 | 1 | 1.000000 | pass | 1 | 145.518044 |

All cases accepted: 0

Alpha=0.75 diagnosis:
- alpha=0.75 classification=accepted, iterations=1/2000, design_change=0.00068572676014, feasibility=0, tracked_mode_index=1, tracked_MAC=0.999999998725, A5_lowest_mode=1, grayness=0.999999890016. The diagnosis is standalone for this case and does not rely on monotonicity across alpha.

No monotonic trend is assumed or used for classification; each alpha is judged only against the declared convergence, feasibility, tracked-MAC, cap, artifact, and A5 lowest-mode checks.

This is experiment evidence only and makes no manuscript claim.
