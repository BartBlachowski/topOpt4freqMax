# MMA_REFERENCE_COMPARISON — Part 11

Reference: the certified conic solution (`bs_ref = 1.001792128848`,
gain `bs_ref − 1 = 1.7921e−3`). MMA states: P19 = production `DRHO(:,386)`;
M500 = prior audit's 500-iterate state; M5000 = this task's frozen replay
(bitwise-reproducing P19 at 19 and M500 at 500). Source
`evaluations/mma_comparison.json`.

## The three states against the reference

| metric | P19 | M500 | M5000 |
|---|---|---|---|
| `bs` | 1.0000049637 | 1.0007024808 | 1.0009624400 |
| β | 26 873.7447 | 26 892.4895 | 26 899.4745 |
| objective gap `bs_ref − bs` | 1.787e−3 | 1.090e−3 | 8.30e−4 |
| **normalized gap G** (fraction of achievable gain lost) | **0.9972** | **0.6080** | **0.4630** |
| ‖drho − drho_ref‖₂ | 0.9980 | 0.9241 | 0.8325 |
| ‖drho − drho_ref‖₂ / ‖drho_ref‖₂ | 0.9948 | 0.9211 | 0.8297 |
| ‖drho − drho_ref‖∞ / move | 1.000 | 1.005 | 1.017 |
| cosine similarity | 0.669 | 0.598 | 0.707 |
| ‖drho‖₂ (reference: 1.0033) | 0.0079 | 0.1447 | 0.2739 |
| max\|drho\|/move | 0.062 | 0.876 | 0.979 |
| constraint violation (exact problem) | 0 | 0 | 0 |
| exact-MMA-dual KKT residual, normalized RMS | 0.360 | 0.586 | 0.597 |
| projected residual under the reference multipliers | 1.010 | 1.010 | 1.010 |
| sign agreement with reference, per element | 0.843 | 0.902 | 0.941 |
| variables within 1e−6·width of a bound | 0 | 0 | 0 |
| variables within 5 %·width of a bound (all on the reference's side) | 4 716 | 14 918 | 16 830 |
| gray elements (8 272) within 10 % of ±move | 0 | 14 | 112 |
| median \|drho/drho_ref\| over gray elements | 0.004 | 0.057 | 0.151 |

Active-set overlap (Jaccard) at the preregistered 1e−6 tolerance is 0 for
every MMA state because `mmasub` is an interior-point subsolver that never
puts a variable on a bound; the loose-tolerance rows above are the
informative ones: **every** element that any MMA state brings near a bound is
on the same side as the reference, but the MMA states bring only the
narrow-box void/solid elements near their bounds, and essentially none of the
gray elements to ±move.

## What the numbers say

1. **Production P19 loses 99.7 % of the achievable gain.** Its increment is
   0.8 % of the reference in norm, 6 % of the move limit at its largest
   element, and points only loosely in the reference direction (cosine 0.67).
   `production_truncation_materially_premature = true` (preregistered rule
   `G_P19 ≥ 0.5`).
2. **M500 and M5000 recover about half of the gain** (G = 0.61, 0.46) while
   reaching 88 % and 98 % of the move limit **on a few elements**: their norms
   are 14 % and 27 % of the reference's, and in the gray band — where the
   reference is bang-bang ±move on all but 12 of 8 272 elements — the median
   MMA increment is 6 % and 15 % of the reference's. "max|drho| ≈ move" is a
   property of isolated elements, not of the step.
3. **The KKT residual of the true problem does not improve** along the
   sequence (0.36 → 0.59 → 0.60 with the MMA duals; 1.01 flat under the
   reference multipliers), while the objective gap does (0.997 → 0.61 → 0.46).
   The iterates climb the objective without approaching stationarity.
4. **No MMA iterate is infeasible** for the exact problem: the MMA
   approximation errs on the conservative side here.
5. The MMA duals themselves drift: `λ_volume` 0.63 → 1.26 → 1.47 against the
   certified 0.729; `λ_cluster` 0.997 → 1.003 → 1.003 against 1.000.

## Difference maps and scatters

`FIG_04_reference_minus_P19`: the difference is the reference itself (P19 is
nearly zero). `FIG_05_reference_minus_M5000`: the difference is concentrated in
the gray band and along the member edges, where M5000 has not reached ±move.
`FIG_11`/`FIG_12`: P19 collapses onto the horizontal axis; M5000 spreads along
the diagonal with the correct sign for 94 % of elements but magnitudes far
short of the bounds.

## Verdict inputs (preregistration §8)

`G_P19 = 0.9972` (> 0.1: not Case B), `G_M5000 = 0.4630` (> 0.1: Case C's
objective condition fails; the move-bound fraction condition, 0.337 < 0.5,
would also fail), a certified global reference exists (not Case E), no
multiple local solutions (not Case D). ⇒

```
REPEATED_MMA_REALIZATION_FAILS_PROBLEM25_REFERENCE
```
