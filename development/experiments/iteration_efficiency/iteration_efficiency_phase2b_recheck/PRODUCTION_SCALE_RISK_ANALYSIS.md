# WP13 — Production-scale precision-risk coverage

## Method

The dominant precision mechanism identified in this recheck (main report, WP8B) is a
**branch flip in the E2/E3 mass law**, which is discontinuous at x = 0.1:

    g(x) = x^6   for x <= 0.1
    g(x) = x     otherwise

At x = 0.1 the two branches differ by a factor 1e5. Because IEEE double->single rounding
is monotone, an element is branch-ambiguous under single storage **if and only if** its
stored single value equals `single(0.1)` = 0.10000000149011611938, which is strictly
greater than double 0.1. That test is decidable from the single trajectory alone, so the
at-risk census is exact and complete even though the original double is unrecoverable.

A second, independent mechanism affects the exact-count projection: single rounding can
only *collapse* distinct doubles onto one float32, so a binary difference requires a
newly-created tie spanning the cutoff rank. At-risk states are exactly those whose single
values at ranks nSolid and nSolid+1 are equal.

Both censuses were run over every stored state of every available frozen Olhoff
trajectory. See `PRODUCTION_SCALE_RISK.csv`.

## Result

| mesh | states | elements | at-risk states (0.1 branch) | fraction | max at-risk elements in one state | cutoff-tie states | a_sig (elements) |
|---|---|---|---|---|---|---|---|
| 160x20 | 1601 | 3200 | 761 | 47.5% | 620 | 300 | 4 |
| 240x30 | 1601 | 7200 | 760 | 47.5% | 1496 | 575 | 9 |
| 320x40 | 1601 | 12800 | 861 | 53.8% | 2680 | 698 | 16 |
| 400x50 | 1601 | 20000 | 1011 | 63.1% | 4240 | 774 | 25 |
| 480x60 | 358 | 28800 | 139 | 38.8% | 5666 | 315 | 36 |
| 560x70 | 400 | 39200 | 160 | 40.0% | 7561 | 358 | 49 |
| 640x80 | 1067 | 51200 | 701 | 65.7% | 10032 | 818 | 64 |
| 720x90 | 1601 | 64800 | 1254 | 78.3% | 12477 | 1169 | 81 |
| 800x100 | — | — | — | — | — | — | 100 |

800x100 remains `RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT`; no evidence was invented for it.

## Coverage comparison against the qualification evidence

The qualification trajectory (96x12, horizon 3200) contains 1560 at-risk states of 3200
(48.8%), with at most 192 at-risk elements in a single state, and 160 cutoff-tie states.

| metric | qualification (96x12) | production range | covered? |
|---|---|---|---|
| fraction of states at-risk | 48.8% | 38.8% – 78.3% | PARTIAL — covers the lower half of the range, not 720x90's 78.3% |
| max at-risk elements per state | 192 | 620 – 12477 | **NOT COVERED** — production is 3x to 65x worse |
| at-risk elements as fraction of mesh | 16.7% | 19.4% – 19.3% | comparable |
| cutoff-tie states | 160 of 3200 (5.0%) | 300–1169 of 358–1601 (18.7%–88.0%) | **NOT COVERED** — production far worse |
| a_sig in elements | 1.44 | 4 – 100 | STRICTER_THAN_PRODUCTION (Patch 5) |

## Interpretation

The qualification evidence is **not conservative** with respect to the mechanism that
actually fails. Production states carry between 3x and 65x more branch-ambiguous elements
than the worst qualification state, and between 3.7x and 17.6x the rate of cutoff ties.
Since the measured E2/E3 error grows monotonically with the at-risk count
(`EVALUATOR_ERROR_STRATIFIED.csv`: max relative E2 error 0 for states with no at-risk
elements, 2.69e-3 for 1–16, 7.93e-3 for 17–64, 2.27e-2 at 192), the production-scale
error is expected to be **larger** than the 2.27e-2 measured here, not smaller.

The one direction in which the qualification evidence is conservative is topology: at
a_sig = 1.44 elements the 96x12 gate tolerates far less detached material than production
(4 to 100 elements), so the observed topology-decision invariance at 45 of 45 paired
states is a *stricter* test than production would apply. That strengthens the topology
PASS result and does not offset the spectral failure.

## Consequence for the historical trajectories

Because the double original is unrecoverable at intermediate states, it cannot be
determined retrospectively how many of these production at-risk elements actually sat
below 0.1. In the genuine paired qualification evidence, **255 of 255 at-risk elements had
a double value at or below 0.1**, i.e. a 100% flip rate, because the move-limit arithmetic
descends onto 0.0999999999999996447 and stalls there. If that rate carries to production,
every one of the at-risk states listed above is affected in E2/E3.

This is a coverage finding, not a proof about historical data, and is reported as such.
