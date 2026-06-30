# Phase 4 outer_move=0.02 Result

Generated: 2026-06-30T10:40:44Z

## Parameter Verification
- pass: `true`
- changed parameter: `outer_move` 0.05 -> 0.02
- other differences: 0

## Decision
**A. outer_move = 0.02 removes the oscillation and achieves convergence.**

## Outer Convergence
| metric | Phase 3 | Phase 4 |
|---|---:|---:|
| final design change | 0.0206388 | 0.000101792 |
| mean design change, last 50 | 0.0206685 | 0.0081628 |
| omega parity gap | 1.14922 | 2.97799 |
| beta parity gap | 6.03482 | 2.80241 |
| omega oscillation amplitude | 1.23164 | 71.2235 |
| beta oscillation amplitude | 6.35641 | 70.0391 |
| design-change trend slope | 9.53789e-07 | -0.000172528 |

## Inner Behaviour
| metric | Phase 3 | Phase 4 |
|---|---:|---:|
| average inner iterations | 114.552 | 52.625 |
| maximum inner iterations | 166 | 71 |
| cap-hit fraction | 0 | 0 |
| cap hits | 0 | 0 |
| average CPU time s | 1.99277 | 0.839847 |
| MMA state reuse fraction | 0.9975 | 0.958333 |

## Final Solution
| mode | omega rad/s | Hz |
|---:|---:|---:|
| 1 | 220.64835 | 35.117276 |
| 2 | 421.76701 | 67.126304 |
| 3 | 647.638 | 103.07479 |
| 4 | 709.04388 | 112.84784 |
| 5 | 950.04083 | 151.20369 |
| 6 | 1244.1344 | 198.01014 |
- S1 overall: ambiguous
- localized low-density modes: 0/10
- mode 1 ld_strain_frac: 0
- support-connected components: 0

## Final Section
1. Yes. Decision A: outer_move = 0.02 removes the oscillation and achieves convergence.
2. Yes; the residual oscillation is below the convergence threshold.
3. The remaining first-order outer-update fixed-point dynamics under the current alpha and trust-region schedule. Phase 4 omega parity gap is 2.97799 rad/s and mean design change is 0.0081628.
4. Yes; freeze the implementation for the final experimental campaign.

Recommendation: Freeze the implementation for the final experimental campaign.
