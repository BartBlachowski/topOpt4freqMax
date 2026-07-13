# Phase 3 outer_move=0.05 Result

Generated: 2026-06-30T09:55:00Z

## Parameter Verification
- pass: `true`
- changed parameter: `outer_move` 0.2 -> 0.05
- other differences: 0

## Decision
**B. Smaller outer_move substantially reduces the 2-cycle.**

## Outer Convergence
| metric | Phase 2 | Phase 3 |
|---|---:|---:|
| final design change | 0.0921098 | 0.0206388 |
| mean design change, last 50 | 0.0915786 | 0.0206685 |
| omega parity gap | 1.63382 | 1.14922 |
| beta parity gap | 18.4989 | 6.03482 |
| omega oscillation amplitude | 3.92221 | 1.23164 |
| beta oscillation amplitude | 25.2246 | 6.35641 |
| design-change trend slope | 9.97612e-06 | 9.53789e-07 |

## Inner Behaviour
| metric | Phase 2 | Phase 3 |
|---|---:|---:|
| average inner iterations | 162.308 | 114.552 |
| cap-hit fraction | 0.01 | 0 |
| cap hits | 4 | 0 |
| average CPU time s | 1.12935 | 1.99277 |
| MMA state reuse fraction | 0.9975 | 0.9975 |

## Final Solution
| mode | omega rad/s | Hz |
|---:|---:|---:|
| 1 | 329.02991 | 52.366737 |
| 2 | 360.24315 | 57.334477 |
| 3 | 462.46452 | 73.603514 |
| 4 | 558.8636 | 88.945905 |
| 5 | 580.56167 | 92.39926 |
| 6 | 599.76152 | 95.45501 |
- S1 overall: ambiguous
- localized low-density modes: 0/10
- mode 1 ld_strain_frac: 0.0012539794
- support-connected components: 0

## Final Section
1. Yes. Decision B: Smaller outer_move substantially reduces the 2-cycle.
2. Yes; linearization error from too-large outer updates is now the dominant explanation.
3. The remaining fixed-point/linearized-increment outer update dynamics under the current alpha and MMA inner-subproblem formulation. Phase 3 omega parity gap is 1.14922 rad/s and mean design change is 0.0206685.
