# Phase 2C — E2/E3 interpolation discontinuity: provenance, stability and impact audit
READ_ONLY_AUDIT — NOT_NEW_OPTIMIZATION_EVIDENCE

Date 2026-08-30 · branch `benchmark-methodology-r2` · HEAD `632e9b01811845709de33f93051fd853373ed5e1`
No optimization was run. No evaluator, optimizer, methodology document, Phase-2A or
Phase-2B artifact was modified. Production remains blocked.

## Central answer

**E2 and E3 are exactly source-faithful, and they are not well-posed as common quality
evaluators at the frozen 0.5–2% quality scale.** Those two statements are both true and
are not in tension: the defect is a methodological mismatch between what the source law was
built for and what the frozen study uses it for.

The audit's four candidate explanations resolve to **B — a faithful method-specific
interpolation that is unsuitable as a common evaluator.** It is not an implementation
error (A), and the discontinuity is intentional in the source but *not* defensible here (C
is refuted on its second clause).

## 1. Provenance — the discontinuity is in the literature

Du & Olhoff (2007) §2.2 prints Eq. (4) in exactly the coefficient-free form the evaluator
implements, and then says so in as many words:

> "It is noted that (4) is discontinuous at the low value ρe = 0.1 of the material
> density. Numerically, this is not a serious problem, as the discontinuity only occurs at
> a single point."

Eq. (4a) adds `c0 = 1e5` to "enforce the C0 continuity at the value ρe = 0.1"; Eq. (4b)
adds `c1 = 6e5, c2 = −5e6` for C1. Yuksel (2025) Eq. (10) restates the same discontinuous
form, citing Du & Olhoff.

**No coefficient is missing.** The hypothesis that `c0 = 1e5` had been dropped in
transcription — the natural suspicion, since `1e5·0.1^6 = 0.1` exactly — was tested and
**refuted**: c0 belongs to Eq. (4a), which the source presents as a separate optional
improvement. This repository in fact implements all four variants (`'lin'`, `'4'`, `'4a'`,
`'4b'`) in `massScale.m` and `mass_interp.m` with correct continuity docstrings, and the
Olhoff runner's own default is `massInterp = '4'` (`defaultCfg.m:12`). The evaluator
mirrors the native choice deliberately.

The branch's purpose is stated by the source: suppression of "spurious, localized
eigenmodes" in low-density regions, following Pedersen (2000) and Tcherniak (2002). It is a
**numerical device internal to an optimization scheme, not a physical material model.**

## 2. Why the source's own justification does not transfer

The authors tolerate the discontinuity on two premises. Both fail for this study.

**P1, "the discontinuity only occurs at a single point."** True as measure theory, false as
numerics here. WP8 reconstructs the observed parking value exactly: starting from
`rho0 = 0.5` and subtracting the move limit `0.005` **sequentially** eighty times yields

    0.099999999999999644729

bit-for-bit — 26 double ULPs below 0.1 — and the frozen mixed sequence (0.005 ×60 then
0.0025 ×40) lands on the *identical* value. By contrast the single expression
`0.5 - 80*0.005` gives `0.099999999999999977796`. The parking value is a deterministic
artifact of move-limit accumulation, so the "single point" is an **attractor of the update
law**, reached with probability 1.

**P2, "negligible differences in the final results."** The authors compared converged 0–1
designs. This study evaluates *gray intermediate states* and extracts the *iteration index*
of first persistent attainment of a quality band. That estimand is sensitive at exactly the
scale of the effect.

A third mismatch the source never contemplated: E2/E3 are applied **post hoc to another
method's trajectory**. A device tuned to keep one optimizer away from spurious modes
carries no neutrality guarantee as a cross-method scorer.

## 3. Three distinct problems (WP12)

| | Problem | Severity | Evidence level |
|---|---|---|---|
| **A** | **Single storage** crosses the branch | **High** | Direct, genuine paired double/single states. Max relative E2/E3 error 2.674e-02 (independent recomputation, WP9); 2.650e-02 on the production 160x20 trajectory (WP11) |
| **B** | **Double-precision instability**: adjacent representable doubles select different branches | **High — and independent of A** | Direct. A perturbation of one double ULP (2.776e-17) changes E2/E3 by 2.8e-03 to 4.0e-03 across six tested states (WP6), against a 0.5% band at q = 0.995 |
| **C** | **Common-evaluator semantics**: a method-specific discontinuous device used as a neutral post-hoc scorer | **High** | Structural, plus measurement: 751 of 3200 states (23.5%) change which evaluator binds the robust minimum purely from branch side (WP13) |

Problem B is the finding that most exceeds Phase 2B's scope. **Even with perfect double
storage, E2/E3 are not stable functions of the design at the frozen quality scale.**

## 4. Exposure (WP7)

Stored Olhoff trajectories, branch-ambiguous elements (`single(value) == single(0.1)`):

| mesh | states | elements | ambiguous elements | states exposed | fraction |
|---|---|---|---|---|---|
| 160x20 | 1601 | 3200 | 13,434 | 761 | 47.5% |
| 240x30 | 1601 | 7200 | 54,701 | 760 | 47.5% |
| 320x40 | 1601 | 12800 | 40,176 | 861 | 53.8% |
| 400x50 | 1601 | 20000 | 79,287 | 1011 | 63.1% |
| 480x60 | 358 | 28800 | 29,940 | 139 | 38.8% |
| 560x70 | 400 | 39200 | 57,471 | 160 | 40.0% |
| 640x80 | 1067 | 51200 | 120,267 | 701 | 65.7% |
| 720x90 | 1601 | 64800 | 203,526 | 1254 | 78.3% |
| 800x100 | — | — | — | — | unavailable (`RUN_ERROR`) |

**Limitation:** no Proposed or Yuksel density trajectories are stored anywhere in the
repository, so cross-method exposure cannot be quantified without new optimization runs.
Whether threshold parking is Olhoff-specific or common is therefore **undetermined**. WP8
gives a mechanism — sequential move-limit accumulation from rho0 — that would apply to any
method whose updates walk an exact arithmetic lattice, but this is reasoning, not
measurement.

## 5. Estimand impact (WP10/WP11)

On the 96x12 reference-length trajectory (3200 states, matching the frozen `B_ref`), the
frozen machinery was re-propagated independently:

| quantity | x^6 branch (double) | linear branch (float32) | identical |
|---|---|---|---|
| reference status | PASS | PASS | yes |
| **b_ref** | **2200** | **2100** | **no** |
| B_meas | 3200 | 3200 | yes — insensitive, since Olhoff `B0 = B_ref = 3200` saturates the formula |
| **k_enter** q = .98/.99/.995 | **233 / 315 / 609** | **232 / 309 / 524** | **no** |
| **k_cert** q = .98/.99/.995 | **332 / 414 / 708** | **331 / 408 / 623** | **no** |
| acceptance differences | — | — | 3 / 3 / 29 states |

This reproduces Phase 2B exactly. `k_cert` at q = 0.995 moves **85 iterations, 13.6%**.

On the frozen **production** 160x20 trajectory the branch-side E2/E3 perturbation is
**2.650e-02**, i.e. the effect is not an artifact of the reduced qualification mesh. The
downstream estimands are *not evaluable on that artifact*, because the reference procedure
requires a separate `B_ref = 3200` reference trajectory
(`reference.trajectory_separate_from_measurement: true`) and the stored production files are
the 1600-horizon measurement runs; both branch sides return `REFERENCE_NOT_ESTABLISHED`.
This is a property of the artifact set, not a defect.

## 6. Independent reproduction (WP9)

Recomputed from stored paired density fields with an independent script, not by reusing any
Phase-2B result: 236 paired states, max relative error **E1 5.595e-08, E2 2.674e-02,
E3 2.674e-02**; hard gate (volume ∧ topology) identical at **236/236**; 70 states contain
branch crossings. Phase 2B's reported facts are confirmed: E1 approximately invariant,
E2/E3 ~2.67e-02 in the decisive case, topology gate invariant.

## 7. What is NOT implicated

The topology and volume hard gate is invariant across every paired state examined here
(236/236) and in Phase 2B (45/45). Iteration accounting, timing, scaling, method profiles
and mesh sequence are untouched by this finding. Persistence *semantics* are unaffected;
only the computed values change, because their evaluator input changes.

---

## Required final summary

1. **E1 mass.** `1e-6 + (1-1e-6)·z`.
2. **E2 mass.** `1e-9 + (1-1e-9)·g(z)`, `g = z^6` for `z ≤ 0.1`, else `z`.
3. **E3 mass.** `g(z3)`, `z3 = max(z,1e-3)`, `g = z3^6` for `z3 ≤ 0.1`, else `z3`.
4. **E1 continuous?** Yes, C-infinity.
5. **E2 continuous?** No — discontinuous at z = 0.1.
6. **E3 continuous?** No — discontinuous at z3 = 0.1 (and C0-not-C1 at the 1e-3 clamp).
7. **Exact jump at 0.1.** left limit and value 1e-6; right limit 0.1; absolute jump 9.9999e-02; multiplicative jump 1.0e+05.
8. **Origin of threshold 0.1.** Du & Olhoff (2007) §2.2, Eq. (4); inherited by Yuksel (2025) Eq. (10) as `xMassCut`.
9. **Origin of exponent 6.** Du & Olhoff (2007): "r is chosen to be about r = 6, i.e., much larger than the penalization power p for the stiffness, which is kept unchanged at a value about p = 3." Yuksel implements it as `dMass = 6.0`.
10. **Does the literature specify the discontinuity?** Yes, explicitly, and it names it: "(4) is discontinuous at the low value ρe = 0.1".
11. **Is a coefficient missing?** No. `c0 = 1e5` belongs to the separate Eq. (4a). Hypothesis tested and refuted.
12. **Purpose of the branch.** Suppression of spurious localized eigenmodes in low-density regions, after Pedersen (2000) and Tcherniak (2002).
13. **Optimization device or physical evaluation?** Optimization device. Not a physical material model.
14. **Proposed states exposed.** Undetermined — no stored Proposed density trajectories.
15. **Yuksel states exposed.** Undetermined — no stored Yuksel density trajectories.
16. **Olhoff states exposed.** 761 / 760 / 861 / 1011 / 139 / 160 / 701 / 1254 across the eight available meshes (38.8%–78.3%); 800x100 unavailable.
17. **Is threshold parking method-specific?** Undetermined by measurement. WP8 identifies a mechanism (sequential move-limit accumulation from rho0 = 0.5) that is not intrinsically Olhoff-specific.
18. **Max E2 change from one double ULP.** 4.002e-03 (state k = 204, 31 elements at the branch).
19. **Max E3 change from one double ULP.** 4.002e-03.
20. **Modal-order changes?** Not separately instrumented — only omega_1 was tracked. No mode reordering is implied by the data, but modal identity was not followed; recorded as a limitation.
21. **Phase-2B spectral effect reproduced?** Yes — max relative E2/E3 2.674e-02, E1 5.595e-08, from stored paired density fields.
22. **Phase-2B b_ref change reproduced?** Yes — 2200 vs 2100.
23. **Phase-2B k_enter changes reproduced?** Yes — 233/315/609 vs 232/309/524.
24. **Phase-2B k_cert changes reproduced?** Yes — 332/414/708 vs 331/408/623.
25. **Frozen quality classifications affected.** 3, 3 and 29 states at q = 0.98, 0.99, 0.995 on the 3200-state trajectory; 236/236 hard-gate decisions unaffected.
26. **Is E2 robust at the 0.5–2% scale?** No. One double ULP gives up to 4.0e-03; branch side gives up to 2.65e-02.
27. **Is E3 robust at that scale?** No — identical figures.
28. **Is robust min(E1,E2,E3) stable?** No. 751 of 3200 states change the binding evaluator by branch side alone.
29. **E2 source-fidelity.** SOURCE-FAITHFUL WITH DOCUMENTED NUMERICAL DEVICE.
30. **E3 source-fidelity.** SOURCE-FAITHFUL WITH DOCUMENTED NUMERICAL DEVICE.
31. **E2 common-evaluator validity.** NOT ROBUST AT FROZEN QUALITY SCALE.
32. **E3 common-evaluator validity.** NOT ROBUST AT FROZEN QUALITY SCALE.
33. **Is the problem single-storage-only?** No.
34. **Independent double-precision problem?** Yes — Problem B, confirmed at one double ULP.
35. **Does the freeze need reopening?** Yes, narrowly.
36. **Minimum scope of reopening.** Evaluator / quality-reference subsystem only: E2/E3 definitions, the co-primary robust-minimum acceptance rule, the evaluator specification in IMPLEMENTATION_REQUIREMENTS, recomputation of Q_ref/b_ref, and the frozen prior absolute-quality margins. Not the topology gate, iteration accounting, timing, scaling, persistence semantics, method profiles or mesh sequence.
37. **Minimum defensible correction.** Option B — adopt Du & Olhoff Eq. (4a) with `c0 = 1e5` in the **common evaluator only**, leaving every optimizer on its native variant.
38. **Offline re-evaluation possible after correction?** Yes. E1/E2/E3 are post-hoc functions of stored densities; the Olhoff historical float32 constraint and its recorded Mo9 scope limits still apply.
39. **Optimizer reruns required?** No.
40. **Does Phase 2B remain valid?** Yes, unmodified, as a negative result about the evaluator definition frozen at the time of testing.
41. **New precision qualification needed after correction?** Yes — the compared decisions all change. On the WP16 sensitivity estimate it would have a good prospect of passing, but that must be measured.
42. **Production still blocked?** Yes. No artifact created, no preflight edit, no token, no campaign.
43. **Recommended next work package.** Narrow methodology revision plus delta audit, scoped to the evaluator / quality-reference subsystem, sequenced as in `FREEZE_IMPACT.md` §WP19.

---

# E2/E3 SOURCE-FAITHFUL BUT NOT ROBUST AS COMMON EVALUATORS — NARROW METHODOLOGY REVISION REQUIRED

PRODUCTION STATUS: BLOCKED
