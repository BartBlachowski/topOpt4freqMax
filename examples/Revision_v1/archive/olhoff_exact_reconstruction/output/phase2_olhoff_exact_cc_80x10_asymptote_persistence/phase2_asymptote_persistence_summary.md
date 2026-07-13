# Phase 2 Asymptote Persistence Result

Generated: 2026-06-30T09:30:18Z

## Parameter Verification
- pass: `true`
- changed implementation feature: `persistent_mma_state` false -> true
- other differences: 0

## Decision
**C. Persistent asymptotes have negligible effect.**

## Outer Convergence
| metric | Phase 1 | Phase 2 |
|---|---:|---:|
| omega parity gap | 3.70072 | 1.81826 |
| beta parity gap | 15.6703 | 18.1142 |
| mean design change, last window | 0.0918889 | 0.0915099 |
| final design change | 0.0914339 | 0.0921098 |

## Inner Behaviour
| metric | Phase 1 | Phase 2 |
|---|---:|---:|
| average inner iterations | 155.315 | 162.308 |
| cap hits | 6 | 4 |
| average CPU time s | 1.19237 | 1.12935 |
| MMA state reuse fraction | 0 | 0.9975 |

## Final Solution
| mode | omega rad/s | Hz |
|---:|---:|---:|
| 1 | 270.74666 | 43.09067 |
| 2 | 387.81165 | 61.722142 |
| 3 | 575.96846 | 91.668228 |
| 4 | 621.68332 | 98.943974 |
| 5 | 700.37347 | 111.4679 |
| 6 | 831.09164 | 132.27234 |
- S1 overall: ambiguous
- localized low-density modes: 0/10
- mode 1 ld_strain_frac: 0.0011981387
- support-connected components: 0

## Final Section
1. No. Decision C: Persistent asymptotes have negligible effect.
2. No; asymptote persistence is not the dominant explanation of the previous oscillation.
3. The remaining fixed-point/linearized-increment outer update dynamics under the current outer move and alpha settings. Phase 2 omega parity gap is 1.81826 rad/s and mean design change is 0.0915099.
