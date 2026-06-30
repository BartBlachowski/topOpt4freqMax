# Phase 1 Inner-Max-Iter 300 Result

Generated: 2026-06-30T09:05:31Z

## Parameter Verification

- pass: `true`
- changed parameter: `inner_max_iter` 30 -> 300
- other differences: 0

## Decision

**C. Inner solver still reaches the cap.**

## Inner Solver

| metric | inner=30 | inner=300 |
|---|---:|---:|
| average inner iterations | 30 | 155.315 |
| min inner iterations | 30 | 81 |
| max inner iterations | 30 | 300 |
| fraction hitting cap | 1 | 0.015 |
| average inner CPU time s | n/a | 1.19237 |

## Outer Convergence

| metric | inner=30 | inner=300 |
|---|---:|---:|
| omega parity gap | 6.96163 | 3.70072 |
| beta parity gap | 7.5174 | 15.6703 |
| mean design change, last window | 0.0767391 | 0.0918889 |
| final design change | 0.0763754 | 0.0914339 |

## Final Solution

| mode | omega rad/s | Hz |
|---:|---:|---:|
| 1 | 263.65218 | 41.961548 |
| 2 | 380.9177 | 60.624934 |
| 3 | 554.07725 | 88.184132 |
| 4 | 623.56417 | 99.24332 |
| 5 | 716.38319 | 114.01593 |
| 6 | 848.4677 | 135.03783 |

- S1 overall: ambiguous
- localized low-density modes: 0/10
- mode 1 ld_strain_frac: 0.0016501133
- support-connected components: 0

## Final Section

1. No. Fraction hitting the inner cap is 0.015.
2. No. Omega parity gap is 3.70072 rad/s and design-change mean is 0.0918889.
3. The MMA inner subproblem still failing to converge within the enlarged cap.
