# Phase 2D — Narrow common-evaluator amendment: Du & Olhoff Eq. (4) → Eq. (4a)
OFFLINE_AMENDMENT_VALIDATION — NO_NEW_OPTIMIZATION

Date 2026-08-30 · branch `benchmark-methodology-r2` · HEAD `632e9b01811845709de33f93051fd853373ed5e1`
No optimizer was run or modified. No production campaign. No self-refreeze.

## What was changed

Exactly one functional change, in exactly two lines, in the **common post-hoc evaluator
only**: the E2 and E3 low-density mass branch now uses the source-defined continuous
Du & Olhoff (2007) Eq. (4a), `g(x) = 1e5·x⁶` for `x ≤ 0.1`, in place of the discontinuous
Eq. (4), `g(x) = x⁶`.

    E2:  g(low) = z(low).^6;    ->  g(low) = 1e5 * z(low).^6;
    E3:  g(low) = z3(low).^6;   ->  g(low) = 1e5 * z3(low).^6;

E1 is untouched. All three stiffness laws, both density floors, the mesh, supports,
eigensolver, deterministic start vector, exact-count projection and every topology and
volume diagnostic are byte-identical. **All six native optimizer sources and all
`protected_numerical_sources` hashes re-verify unchanged**, and the frozen evaluator itself
was deliberately left unmodified so that Phase-2A/2B/2C provenance still verifies.

## The measured effect

Every measurement below is offline recomputation from stored density evidence.

| mechanism | evidence | Eq. (4) | Eq. (4a) | reduction |
|---|---|---|---|---|
| **one double ULP** at the branch | 6 states, 96x12 paired | **4.0021e-03** | **2.1551e-13** | **1.86e+10** |
| **float32 storage** cross-branch | 236 genuine paired states | **2.6736e-02** | **5.5955e-08** | **4.78e+05** |
| **branch side**, production mesh | 160x20, all 1600 states | **2.6496e-02** | **2.6533e-10** | **9.99e+07** |
| **binding evaluator instability** | 160x20, all 1600 states | **150 / 1600 (9.38%)** | **0 / 1600 (0.00%)** | eliminated |

The amended E2/E3 float32 error, **5.5955e-08**, is indistinguishable from E1's
**5.5951e-08** on the same states. E2 and E3 now respond to storage precision exactly as the
continuous E1 always did — the defect is not merely reduced, its mechanism is gone.

Against the tightest frozen acceptance band (0.5% at q = 0.995):

| perturbation as a fraction of the 0.5% band | Eq. (4) | Eq. (4a) |
|---|---|---|
| one double ULP | **0.80×** | 4.31e-11× |
| float32 storage | **5.35×** | 1.12e-05× |

Under Eq. (4) a single double ULP consumed 80% of the band and float32 storage exceeded it
fivefold. Under Eq. (4a) both are eleven and five orders of magnitude below it.

Final states of all eight available meshes were re-evaluated: old relative E2 error up to
1.035e-03 falls to ~1e-11. The one exception, 640x80 (1.576e-09 → 1.626e-09), had no at-risk
elements at its final state, so it was never affected and correctly does not change.

## What the amendment does NOT do

**Eq. (4a) is C0, not C1.** One-sided derivatives at 0.1 are 6 and 1. The amendment removes
the finite jump in the function; it does not make the interpolation smooth. Eq. (4b)
(`c1 = 6e5, c2 = −5e6`) would give C1 and is equally source-defined and equally implemented
in this repository, but Phase 2C established a *finite jump in the function value*, not a
slope defect, and a post-hoc evaluator is never differentiated. Taking C1 would exceed what
the evidence requires. This is flagged for the delta auditor.

**Native identity is genuinely lost.** E2 and E3 no longer reproduce the Yuksel and Olhoff
native mass interpolations. This is stated in the amended normative documents rather than
concealed, and the native optimizers continue to run Eq. (4) unchanged.

**Eq. (4a) is not claimed to be uniquely physically correct.** It is claimed only to be a
source-defined continuous alternative that Du & Olhoff themselves offer and report as
producing "negligible differences in the final results" relative to Eq. (4).

## Neutrality, redefined honestly (WP17)

The old neutrality argument was structural symmetry: one evaluator per contestant, then a
robust minimum. That symmetry is now partially broken — E1 remains Proposed's own law while
E2/E3 depart from their natives in the low-density branch.

Neutrality now rests on four properties, none of which is native identity:

1. **Identical treatment.** The same evaluator family is applied to every density field by
   the same code path, regardless of which optimizer produced it.
2. **Producer-independence.** No evaluator input depends on method identity, run order, or
   provenance — only on the density field.
3. **Stability at the decision scale.** Nearly identical density fields now receive nearly
   identical scores: the previous failure mode, where two fields differing by one ULP could
   receive materially different robust scores and even swap which evaluator bound the
   minimum, is eliminated (150/1600 → 0/1600).
4. **Preserved multi-model perspective.** Three distinct stiffness laws and two distinct
   floor conventions remain.

Disclosed plainly: **E2 and E3 still share substantial structure.** They share the same
low-density mass law — Eq. (4a) now, as they shared Eq. (4) before — and differ only in
stiffness floor convention. The three-evaluator minimum remains closer to two-way in
evidential terms, exactly as the pre-amendment specification already conceded. This
amendment does not improve evaluator independence and does not claim three independent
physical models.

The trade is explicit: for a common *post-hoc measurement instrument*, stable and
method-independent evaluation of nearly identical density fields matters more than literal
reproduction of each contestant's internal numerical device. The device Eq. (4) exists to
suppress spurious modes *during optimization*; importing it into the measuring instrument
imported its discontinuity along with it.

## Limitations

**No Proposed or Yuksel density trajectories exist in the repository** (Phase 2C, confirmed
here). Cross-method empirical exposure cannot be quantified. This is **non-blocking for the
amendment**: the property being fixed is the *mathematical continuity of the evaluator*,
which is established analytically and verified numerically at the branch and is independent
of which trajectories exist. It is a **blocking validation obligation for the future
campaign**, where exposure must be measured per method. Running Proposed/Yuksel optimization
was forbidden in Phase 2D and was not done.

**Reference and persistence were not exercised end-to-end offline.** The frozen design
requires a separate `B_ref = 3200` reference trajectory
(`reference.trajectory_separate_from_measurement: true`), while every stored production
artifact is a 1600-horizon measurement run on which reference does not establish under
either the old or the amended evaluator. The 96x12 reference-length trajectory from Phase 2B
saved only its Q arrays, not its density fields. Regenerating either requires an optimizer
run, which this phase forbids. The amendment's effect on `b_ref`, `k_enter` and `k_cert` is
therefore established **by mechanism and by per-state evaluator stability, not by a direct
end-to-end re-run**. Flagged for the delta auditor as the item to scrutinise hardest.

---

## Required final summary

1. **Old common E2 mass law.** `1e-9 + (1-1e-9)·g(x)`, `g = x⁶` for `x ≤ 0.1`, else `x`.
2. **Amended E2 mass law.** `1e-9 + (1-1e-9)·g(x)`, `g = 1e5·x⁶` for `x ≤ 0.1`, else `x`.
3. **Old common E3 mass law.** `g(max(x,1e-3))`, `g = x⁶` for `x ≤ 0.1`, else `x`.
4. **Amended E3 mass law.** `g(max(x,1e-3))`, `g = 1e5·x⁶` for `x ≤ 0.1`, else `x`.
5. **Components changed.** Only the E2 and E3 low-density mass branch — two lines. E1, all stiffness laws, floors, mesh, supports, eigensolver, projection and topology unchanged.
6. **Native optimizer components changed?** None. All native and protected hashes re-verified identical.
7. **Is Eq. (4a) source-defined?** Yes — Du & Olhoff (2007) §2.2, `c0 = 1e5` "enforces the C0 continuity at the value ρe = 0.1".
8. **Amended E2 C0 continuous at 0.1?** Yes — residual 6.939e-17, one ULP.
9. **Amended E3 C0 continuous?** Yes — same.
10. **C1 continuous?** **No** — one-sided derivatives 6 and 1. Not claimed.
11. **Old max double-ULP E2 instability.** 4.0021e-03.
12. **New max double-ULP E2 instability.** 2.1551e-13.
13. **Old max double-ULP E3 instability.** 4.0021e-03.
14. **New max double-ULP E3 instability.** 3.7033e-13.
15. **Old decisive float32 E2/E3 error.** 2.6736e-02 (236 paired states, 70 with crossings).
16. **New error on the same evidence.** 5.5955e-08 (E2), 5.5948e-08 (E3) — matching E1's 5.5951e-08.
17. **Old production-scale 160x20 branch effect.** 2.6496e-02.
18. **New effect on the same states.** 2.6533e-10.
19. **Old branch-side binding-evaluator instability.** 150 of 1600 states (9.38%).
20. **New binding-evaluator instability.** 0 of 1600 states (0.00%).
21. **Hard-gate invariance.** Identical 1600/1600; `topology_metrics` consumes only the density field and no evaluator value.
22. **Olhoff states re-evaluated.** 1600 (160x20, full, both branch sides) + 236 genuine paired states + 8 mesh final states.
23. **Reference semantics exercised offline?** No — see Limitations.
24. **Persistence exercised offline?** No — same reason.
25. **Amended `b_ref` for the decisive Phase-2B case.** Not obtainable without an optimizer run; recorded as a limitation, not worked around.
26. **Amended `B_meas`.** Same — not obtainable. Note `B_meas` was already insensitive, since Olhoff `B0 = B_ref = 3200` saturates the formula.
27. **Amended `k_enter` double/single.** Not directly re-run; per-state evaluator divergence that drove the old difference falls from 2.67e-02 to 5.60e-08.
28. **Amended `k_cert` double/single.** Same.
29. **Amended status double/single.** Same.
30. **Max amended perturbation vs the 0.5% band.** double-ULP 4.31e-11×; float32 1.12e-05×.
31. **Minimum observed classification margin.** Not recomputable without a reference (see 23); under Eq. (4) the Phase-2C figures were 8.67e-05 / 2.59e-05 / 5.77e-06, all of which now exceed the amended perturbation by 3 to 8 orders of magnitude.
32. **Any amended quality classification flip from branch side?** None observable: branch-side robust perturbation is 2.65e-10 and the binding evaluator never changes (0/1600).
33. **Proposed/Yuksel trajectory data still unavailable?** Yes.
34. **Blocking for this amendment?** No — non-blocking for the amendment, blocking as a validation obligation for the future campaign.
35. **Normative documents changed.** `QUALITY_EFFORT_SPEC.md` and `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md`. Contract and freeze record deferred to refreeze (category B).
36. **Native-identity claims retired/qualified?** Yes, in both amended documents, with pre-amendment copies preserved.
37. **Neutrality now defined as.** Identical treatment, producer-independence, stability at the decision scale, preserved multi-model perspective — with the shared E2/E3 structure disclosed.
38. **Prior results superseded.** E2/E3 columns of `common_evaluators.csv`, the frozen prior absolute-quality margins, and scaling fits over k_enter/k_cert. Topology, timing, accounting, native endpoints unaffected.
39. **Phase 2B historically valid?** Yes — VALID NEGATIVE RESULT UNDER THE PRE-AMENDMENT FROZEN EVALUATOR. Unmodified.
40. **New precision qualification required after refreeze?** **Yes.** Eq. (4a) changes the representation sensitivity that caused Phase 2B to fail, so the old negative result can neither authorize nor reject single storage under the amended methodology.
41. **Topology rules changed?** No.
42. **Persistence rules changed?** No — P, q levels, k_enter/k_cert definitions and the scan are untouched.
43. **Timing/accounting/scaling rules changed?** No.
44. **Protected optimizer hashes preserved?** Yes — all six protected numerical sources, three profile sources and six audit records re-verify against the Phase-2A record.
45. **Ready for independent delta audit?** Yes.
46. **Production still blocked?** Yes.

---

# PHASE 2D AMENDMENT VALIDATED — READY FOR INDEPENDENT EVALUATOR-AMENDMENT DELTA AUDIT

PRODUCTION STATUS: BLOCKED
