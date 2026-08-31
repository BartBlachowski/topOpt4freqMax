# WP14 — Phase-2B historical classification
READ_ONLY_INDEPENDENT_DELTA_AUDIT

## Direction 1 — Phase 2B must not be retroactively labelled erroneous

**Confirmed.** Phase 2B tested the evaluator definition that was frozen at the time, using
the frozen reference, budget and persistence engines, on a genuine 96x12 reference-length
trajectory. Its result — `b_ref` 2200 (double) vs 2100 (single), `k_enter` 233/315/609 vs
232/309/524, `k_cert` 332/414/708 vs 331/408/623, `FROZEN_DECISION_EQUIVALENCE = FAIL` —
was **reproduced exactly** by this audit's independent Python re-implementation of those
engines from the stored quality arrays. The experiment was correct, the mechanism it
identified (branch crossing under a discontinuous mass law) was correct, and Phase 2C
confirmed it. Phase-2B artifacts are unmodified: no file in
`analysis/iteration_efficiency_phase2b_precision/` or
`analysis/iteration_efficiency_phase2b_recheck/` falls in the Phase-2D modification window.

Classification stands: **VALID NEGATIVE RESULT UNDER THE PRE-AMENDMENT FROZEN EVALUATOR.**

## Direction 2 — Phase 2B cannot automatically reject float32 storage under an amended evaluator

**Confirmed, and quantified.** Phase 2B failed through one identified mechanism: elements
parked at `0.099999999999999644729`, whose float32 image `0.10000000149011611938` lies on the
other side of the branch, so that `g` jumped by a factor of 1e5. This audit verified that
arithmetic directly:

| | double 0.09999999999999964 | its float32 image 0.10000000149011612 | relative change in g |
|---|---|---|---|
| Eq. (4) | g = 9.99999999999979e-07 | g = 1.00000001490116e-01 | **9.99990e+04** |
| Eq. (4a) | g = 9.99999999999979e-02 | g = 1.00000001490116e-01 | **1.49012e-08** |

The two values **still fall on opposite algebraic branches** under Eq. (4a) — the amendment
does not stop branch crossing. What it removes is the finite value discontinuity that branch
crossing caused. The failure mechanism is therefore absent, not merely attenuated, and the
Phase-2B negative result does not transfer.

Symmetrically, Phase 2B cannot *authorise* float32 storage under the amended evaluator
either. It tested a different instrument.

## Ruling

    PHASE2B_HISTORICAL_CLASSIFICATION = PASS

Phase 2D's classification is correct in both directions and is adopted unchanged.
