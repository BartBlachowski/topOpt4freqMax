# WP13 — Reassessment of evaluator neutrality
READ_ONLY_AUDIT — NOT_NEW_OPTIMIZATION_EVIDENCE

The frozen design uses three co-primary evaluators and a robust minimum,
`robust = min_e ( Q_e / Q_ref,e )`, to blunt any single method's home-field advantage.
This WP asks whether the discontinuity undermines that construction.

## 1. Is the robust minimum still method-neutral?

**No, not in the intended sense.** Neutrality was argued from *which physical model* each
evaluator represents. The discontinuity adds a second, unintended axis of variation: how
often a given trajectory happens to place densities within one float32 ULP of 0.1. That is
a property of a method's *update arithmetic* — its move limits, its initial density, its
clamping — not of its design quality.

## 2. Does a discontinuous E2/E3 penalize methods by proximity to 0.1?

Yes, and the exposure is measurable and method-specific in origin. WP8 shows the Olhoff
parking value arises from sequential subtraction of the move limit from rho0 = 0.5:
`0.5` minus `0.005` eighty times yields exactly `0.099999999999999644729`. A method with a
different move limit, a different rho0, or a projection/filter step that perturbs densities
off the exact accumulation lattice would land on 0.1 far less often, and would be scored by
a *continuous* portion of E2/E3.

This cannot presently be quantified across methods: **no Proposed or Yuksel density
trajectories are stored in the repository** (only Olhoff `rho_snapshots` exist under
`examples/Performance/final_campaign/raw/`). Establishing whether threshold parking is
Olhoff-specific or common would require new optimization runs, which this audit forbids.
Recorded as a limitation, not as a finding of Olhoff-specificity.

## 3. Does min(E1,E2,E3) let the least stable evaluator dominate?

**Yes, and it is measurable.** On the 96x12 reference-length trajectory (3200 states):

| binding evaluator in the robust minimum | x^6 branch (double) | linear branch (float32) |
|---|---|---|
| E1 | 2695 | 2712 |
| E2 | 20 | 166 |
| E3 | 485 | 322 |

**751 of 3200 states (23.5%) change which evaluator binds the robust minimum, purely from
branch side.** The minimum operator actively selects the evaluator that the branch flip has
pushed downward, so the robust statistic preferentially reports the *unstable* evaluator
precisely at the states where it is unstable. A minimum over a set containing a
discontinuous member inherits that member's discontinuity wherever the member binds.

## 4. Can near-identical density fields receive materially different robust scores?

Yes. Two fields differing by one double ULP (2.8e-17) in the at-risk elements produce
E2/E3 differences of 2.8e-3 to 4.0e-3 (WP6). Fields differing only by float32 storage of
the same state produce differences up to 2.65e-2 on a production mesh (WP11, 160x20).

## 5. Is evaluator ordering stable under one-ULP perturbations?

No. Since a one-ULP perturbation moves E2/E3 by up to 4.0e-3 while leaving E1 unchanged to
1e-13, and the three evaluators are within a few percent of one another in ratio terms, the
ordering `argmin_e (Q_e/Q_ref,e)` is not stable. Item 3 quantifies the consequence.

## Conclusion

The robust minimum was a sound device for its stated purpose, but it is not robust in the
numerical sense against a discontinuous member. The neutrality argument for E1/E2/E3 as
co-primary evaluators does not survive contact with the x = 0.1 discontinuity at the frozen
0.5–2% quality scale.
