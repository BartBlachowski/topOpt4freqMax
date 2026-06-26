# Exp3 Authoritative Mesh Convergence Summary

Implementation language: MATLAB.
Scope: Exp3 alpha=1.00 mesh convergence only. No Exp2 alpha sweep, CR2, A4, S1, P1, Python, or manuscript edits.

Classification: inconclusive/capped/mode invalid
Authoritative load: F(x) = omega0^2 * M(x) * Phi0
Filter physical radius: 0.08
Predeclared criteria: both meshes accepted; tracked mode match; tracked MAC >= 0.8; relative tracked omega change <= 0.05; topology correlation >= 0.8; topology MAD <= 0.15.

| mesh | classification | iterations | omega rad/s | tracked mode | MAC | design_change | feasibility | grayness | A5 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 200x25 | accepted | 1052/2000 | [141.789825, 374.309772, 614.650190] | 1 | 0.992533 | 0.00016619 | 0 | 0.0857778 | pass |
| 400x50 | mode invalid | 1750/2000 | [64.393652, 88.179095, 101.804705] | 1 | 0.786127 | 0.00090311 | 0 | 0.0607932 | pass |

Relative tracked omega change: 0.545851393273
Tracked mode match: 1
Topology correlation: -0.087969251187
Topology mean absolute difference: 0.518759676734
Topology RMS difference: 0.664722272939
Grayness difference fine-minus-coarse: -0.0249846441607
Constraint residual difference fine-minus-coarse: 0

This is experiment evidence only and makes no manuscript claim.
