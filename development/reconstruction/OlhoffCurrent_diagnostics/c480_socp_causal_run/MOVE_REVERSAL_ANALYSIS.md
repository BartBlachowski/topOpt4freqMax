# MOVE_REVERSAL_ANALYSIS — Part 6

Per-iteration values: `evaluations/traj_treatment.csv`, `evaluations/traj_control.csv`
(identical code). Bound tolerance 1e-6·box width. Classes are taken at ρₖ₋₁:
void ρ ≤ 0.1, gray shell = gray band minus mid band, gray core = 0.4 ≤ ρ ≤ 0.6,
solid ρ ≥ 0.9.

## Bound saturation (treatment)

| k | ± move bound | density bound | interior | gray elements at ±move |
|---|---|---|---|---|
| 1 | 100.00 % | 0 | 0 | 100.00 % |
| 2 | 99.94 % | 0 | 0.06 % | 99.94 % |
| 3 | 94.18 % | 0 | **5.82 %** (1 676 elements) | 94.18 % |
| 4–10 | 99.89–99.99 % | 0 | ≤ 0.11 % | 99.89–99.99 % |
| 11–12 | 100.00 % | 0 | 0 | 100.00 % |
| 13 | 61.36 % | 38.62 % | 0.01 % | 99.98 % |
| 14 | 61.65 % | 38.35 % | 0 | 100.00 % |

At outer 13 the first elements reached ρ_min or 1 (0.5 ± 12·0.04 = 0.02 / 0.98, so
the next full step is density-limited). **Every gray element took the full ±move at
effectively every accepted step.** This extends to a dynamic trajectory the frozen
oracle's finding that the exact solution of (25) is bang-bang. The control's MMA
increments had **0 %** of entries on any bound (within 1e-6·width) at all 386
iterations.

## Reversal and two-cycle metrics (treatment)

| k | cos(Δρₖ, Δρₖ₋₁) | ‖Δρₖ+Δρₖ₋₁‖/(‖Δρₖ‖+‖Δρₖ₋₁‖) | sign reversal (all) | gray core | gray shell | recurrence ‖ρₖ−ρₖ₋₂‖/‖ρₖ−ρₖ₋₁‖ |
|---|---|---|---|---|---|---|
| 2 | 0.842 | 0.960 | 7.9 % | 7.9 % | — | — |
| 4 | 0.592 | 0.892 | 20.4 % | 41.0 % | 16.2 % | 1.78 |
| 8 | 0.529 | 0.875 | 23.5 % | 31.5 % | 22.5 % | 1.75 |
| 11 | 0.444 | 0.850 | 27.8 % | 57.4 % | 18.0 % | 1.70 |
| 12 | 0.474 | 0.859 | 26.3 % | 41.3 % | 43.1 % | 1.72 |
| 13 | 0.338 | 0.819 | 26.0 % | 46.8 % | 38.2 % | 1.79 |
| 14 | 0.217 | 0.780 | **38.5 %** | 38.0 % | 44.0 % | 1.62 |

(Void and solid classes appear only from k = 12; their reversal fractions are
0–13 %.)

- **No two-cycle.** A two-cycle gives cos → −1, sum-norm ratio → 0 and recurrence → 0.
  Observed: cos stayed positive (0.22–0.84), the sum-norm ratio 0.78–0.96, and
  recurrence 1.6–1.9 (close to √3 ≈ 1.73, the value for two successive
  equal-length steps meeting at 60°).
- **Fraction of steps with cos < −0.5:** 0 of 13.
- **What was happening.** Coherence was declining monotonically and the reversal
  fraction was rising, concentrated in the gray core (mid-density) elements. Up to
  ~57 % of core elements flipped direction between consecutive full-move steps.
  That is decisive sharpening with increasing element-level dithering in the core,
  not yet move-bound oscillation. Whether it would have become a cycle is unknown.
- **Control for comparison.** Over its first 14 steps the control had median
  cos 0.994 and median sign reversal 3.2 %.

Figures: `figures/FIG_12_bound_saturation_vs_iteration`, `FIG_13_reversal_cosine_vs_iteration`.
