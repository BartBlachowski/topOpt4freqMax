BOTTOM LINE

**Olhoff@6b08708 and OlhoffCurrent run the same numerical machinery on two different problems under two different controllers.** Everything they share — FE model, eigensolver, generalized gradients, Sigmund filter, fixed N = 2 multiplicity with off-diagonals, problem (25), and the whole inner MMA loop — is **bitwise identical** at nine frozen designs. They differ on exactly two scientific axes:

1. **Low-density material law (a different relaxed formulation).** Source: Pedersen (2000) stiffness, ρ/100 below 0.1, with linear mass. Target: SIMP ρ³ with the eq.(4b) mass cut-off.
2. **Outer controller and stop.** Source: a per-element adaptive box (0.10 → 0.002, ×1.2 / ×0.7) and an unguarded ‖Δρ‖₂ < ε test. Target: a global move ladder, with stage exhaustion or β stall.

**What made the difference, from the one allowed run.** It was the source code at 480×60 with the *target's* material law (M1). It reproduced the committed Pedersen run bitwise for five iterations. Then, as soon as elements reached ρ ≤ 0.1:

- it developed 11 localized-mode spikes;
- 80 % of its boxes collapsed to the floor;
- it stopped "naturally" at iteration 64 inside a spurious mode, with ω₁ = 34.4 and M_nd 0.285.

The same final design evaluated under Pedersen has ω₁ = 163.6 and no localized modes.

**What that means.** The adaptive box gives the speed and the termination mechanism. The Pedersen law is what makes that controller stable and the stop credible. Neither is a bug fix, and the stop is a heuristic, not a better KKT point.

**Migration.** The source realization can enter OlhoffCurrent only as a second, explicitly named preset. The frozen preset stays as it is.

```
OLHOFF_SOURCE_COMMIT_VERIFIED
OLHOFFCURRENT_TARGET_IDENTITY_PASS
SOURCE_SWEEP_EVIDENCE_PASS
SOURCE_TARGET_DIFFERENT_SCIENTIFIC_FORMULATION
FIRST_DIVERGENCE_OUTER_BOX_CONTROLLER
SOURCE_SUCCESS_PRIMARILY_FORMULATION
PRESERVE_OLD_OLHOFFCURRENT_PRESET
SOURCE_METHOD_REQUIRES_DISTINCT_NAMED_PRESET
OLHOFFCURRENT_MIGRATION_READY_WITH_NAMED_FORMULATION_SPLIT
```

Supporting verdicts: `PREFIX_BITWISE_PASS` (M1 vs S480, iterations 1–5) · natural convergence = `HEURISTIC_STOP` (preregistered rule).

## Final purpose, in one paragraph

Olhoff@6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7 differs from OlhoffCurrent in two scientific respects only, and both are measured, not read from file names.

- **Material law (a new relaxed formulation, class B/D).** Pedersen's linearized low-density stiffness replaces SIMP below ρ = 0.1, and linear mass replaces the eq.(4b) cut-off.
- **Controller and stop (class C reconstruction).** A per-element adaptive move box with an unguarded ε-test replaces the global move ladder with its stage-exhaustion or β-stall logic.

**Identical code.** FE, eigensolver, gradients, filter, multiplicity, problem (25) and the MMA inner loop are bitwise identical.

**What explains the successful sweep.**

- The adaptive box creates the first divergence and the fast early sharpening. It supplies the mechanism by which the ε-test fires.
- The Pedersen law is what keeps that controller out of localized-mode spikes and false gray stops. Single-factor evidence at 480×60 attributes the lower final M_nd to it (share 1.17), and the same pattern holds at 240 and 800.
- The filter non-conservativity and MMA attenuation found earlier are still present in the source, so they cannot explain the difference.
- The source's stop is not closer to physical KKT stationarity.

**What can be migrated safely.**

- All 19 source files can be promoted as code, because they are inert under the old presets.
- The source realization may enter only as a new, formulation-named preset.
- The frozen `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` preset, the stage-exhaustion controller and `tOuter` must survive unchanged.
- The in-place preset replacement, the uncommitted §7 settings and every Phase 6 change to other methods must not be promoted under this audit.

## Answers

1. **Was source commit 6b087085… verified exactly?** Yes.
   - It is HEAD of `repro/natural-convergence`; parent `695f03b`, tree `809c671e`.
   - The 405 snapshot files are blob-verified against the commit.
2. **Are the successful sweep artifacts committed at that hash?** Yes.
   - Both SWEEP tables and CSVs, all 18 run directories (describe, res.mat, summary, three PNGs, log), NOTES §32 and the plan are in the commit tree.
   - Every table column recomputes from the committed `res.mat` to 4e−15.
3. **Was the source repository left untouched?** By this audit, yes: HEAD is unchanged, and only `git archive/ls-tree/show/status/diff` were run.
   - Its working tree was edited externally at 16:32:45: an uncommitted +55-line §7 in the plan.
   - That edit was recorded and excluded.
4. **What target branch/HEAD was audited?** `benchmark-methodology-r2` @ `013cc48451d33bed61c5c4eea174bbd898d548a2`, with `+impl` tree `edbfe47e…` (75 files).
5. **Was OlhoffCurrent production left untouched?** Yes.
   - The manifest verifies, and there is no tracked change.
   - The only new path is this untracked audit directory.
6. **Do source and target solve exactly the same mathematical problem?** No.
7. **What are the formulation differences?**
   - Stiffness: g_K = ρ/100 instead of ρ³ for ρ < 0.1, giving a 10⁴× stiffer void at ρ_min.
   - Mass: g_M = ρ instead of 6·10⁵ρ⁶ − 5·10⁶ρ⁷ for ρ ≤ 0.1, giving 10⁹× heavier void. M/K is bounded at 100 instead of varying non-monotonically from 6e−4 to 100.
   - Objective, constraints, filter, multiplicity, bounds, p, ρ₀, V and R are identical.
   - The difference is zero while every ρ > 0.1 and activates at iteration 6 on the source trajectory.
8. **Does source use Pedersen low-density stiffness treatment?** Yes: `material.stiffness.model = pedersen`, `linearBelow = 0.1`, in all 18 sweep runs.
9. **Does target use the same treatment?** No. It uses SIMP ρ³ everywhere, in both the C480 canary and canonical production.
10. **What mass interpolation does each use?** Source: eq.(2), linear, q = 1, with no cut-off. Target: eq.(4b), C¹ polynomial below 0.1.
11. **Are the sensitivity filters mathematically identical?** Yes.
    - The files are byte-identical, and the filtered f_sk are bitwise equal at 9 states.
    - Their *inputs* differ under Pedersen.
12. **Are FE assembly and eigensolver paths identical?** Yes, for a given law: K and M SHA-256 are equal and ω₁…ω₅ bitwise.
13. **Are multiplicity and off-diagonal treatments identical?** Yes: subspace N = 2, offsets, full (25d), and the (25b) next-mode row, all bitwise.
14. **Is problem (25) constructed identically?** Yes: rows and gradients are bitwise.
    - The only difference is the move part of the box: a per-element vector in the source, a scalar in the target.
15. **Is the inner MMA realization identical?** Yes: the same `innerLoop`, published `mmasub/subsolv`, tolerance and caps, and bitwise identical steps under a common box.
16. **Does source preserve/reset different solver state?**
    - Inner MMA state: identical (persists across sub-iterates, resets per outer).
    - Outer controller state differs: source keeps per-element boxes plus ρ_{k−1}, ρ_{k−2}; target keeps the ladder stage and the exhaustion windows (or the β history in production).
17. **What are the asymptote/subsolv differences?** None in code: asyinit 0.5, asyincr 1.2, asydecr 0.7, albefa 0.1, epsimin 1e−7.
    - Effective asymptote distances scale with each element's box, a box-mediated (D7) effect.
18. **What exact outer box/move mechanism does source use?**
    - d_e = 0.10 at outer iterations 1–2.
    - From outer 3: d_e ← clamp_[0.002, 0.10](f·d_e), with f = 1.2 / 0.7 / 1 when the last two outer steps of that element agree in sign, reverse, or one is zero.
    - Per element, with no β, no stages, and no guards.
19. **What exact outer mechanism does target use?**
    - C480: a global ladder [0.04, 0.02, 0.01]. It descends on stage exhaustion E = A ∨ B (W = 20, P = 20, Wnp = 10, stage-local) and terminates on E at the last rung.
    - Canonical production: [0.04, 0.02, 0.01, 0.005] with the β-stall detector (window 10, tol 5e−3).
20. **What stopping criterion does source use?**
    - ‖Δρ‖₂ < ε = 0.05√(NE/3200) (0.15 at 480), with no persistence and no guards.
    - It is an absolute, mesh-scaled design-change test. It is not a KKT test.
21. **What stopping criterion does target use?**
    - C480: terminal stage-exhaustion declaration (branch B: ‖Δρ‖₂ < ε with positive median coherence for 20 iterations) at move 0.01.
    - Production: ‖Δρ‖₂ < ε on a settled move.
22. **What is the earliest mathematically meaningful divergence?** D7, the outer box, at ρ₀: 0.10 vs 0.04 gives ‖Δρ₁‖₂ 13.39 vs 4.27 (cos 0.960).
    - D0–D6 are bitwise identical under M1.
    - The native material-law operator activates at iteration 6.
23. **At identical ρ, do eigenvalues agree?**
    - Implementations: exactly, at all 9 states.
    - Material laws: exactly while ρ > 0.1; otherwise λ₁ differs by 0.26–0.64 %. At the M1 endpoint it is 34.4 vs 163.6.
24. **At identical ρ, do raw gradients agree?**
    - Implementations: exactly.
    - Laws: raw f₁₁ differs by 25–45 % (L2), concentrated in ρ < 0.1; only 0.2–1.5 % on ρ ≥ 0.1.
25. **At identical ρ, do filtered gradients agree?**
    - Implementations: exactly.
    - Laws: filtered f₁₁ differs by 19–31 % (L2); 0.4–2.5 % on ρ ≥ 0.1.
26. **At identical ρ, does the first local problem-(25) step agree?**
    - Implementations with the same box: exactly.
    - Different boxes at ρ₀: no (above).
    - Laws with box 0.04: cos 0.61–0.99 on gray states, and −0.23 at the M1 endpoint.
27. **When does grayness begin to diverge?**
    - S480 vs C480 at iteration 2, caused by the box.
    - S480 vs M1 at iteration 30, caused by the material law, after M1's first spikes at 20 and 23.
28. **Which difference best explains source natural convergence?** The adaptive box (with the ε-test) supplies the mechanism: oscillating elements contract to the floor until ‖Δρ‖₂ < ε.
    - The Pedersen law makes it credible: without it the same mechanism fired inside a spike at 64.
    - The stop fired after the design had already stopped changing, so it preserved sharpness rather than creating it.
29. **Which difference best explains lower source M_nd?** The Pedersen low-density law, under the source controller.
    - Single factor at 480: 0.285 → 0.131, broad gray core 1.19 → 0.004.
    - The box alone does not lower the endpoint (0.285 vs 0.263) but greatly speeds early progress.
30. **Is the source success primarily formulation, solver, controller, or a combination?**
    - By the preregistered decomposition: primarily formulation (share 1.17).
    - Mechanistically, it is formulation-enabled controller behaviour.
    - The inner solver is identical and excluded.
31. **Which source changes are paper fidelity improvements?** None in the strict sense.
    - Linear mass eq.(2) is a printed option, but so is the target's eq.(4b).
    - Pedersen is the paper-*named* alternative (B), not the paper's choice.
32. **Which are reconstruction choices?** The adaptive per-element box, its constants, ε without guards, move 0.10, and R = 0.06 (shared).
33. **Which constitute a new scientific formulation?** The Pedersen stiffness law together with linear mass, relative to SIMP plus eq.(4b).
34. **Should the old OlhoffCurrent preset remain available?** Yes, unchanged, with its hashes and its bitwise equivalence test.
35. **Should the source formulation receive a new explicit preset name?** Yes.
    - Recommended: `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` → `duOlhoffAdaptivePedersen@6b08708`.
    - Selection must be explicit and versioned.
36. **Is Fable's 15-file promotion justified?**
    - As a code promotion, yes: exactly 15 modified + 4 new files, the unchanged list verified 20/20, and inert for the old presets.
    - As a change of the production preset's meaning, no.
37. **Which migration-plan phases are approved?** None unconditionally.
    - Phases 0, 1, 2, 3, 4, 5 are approved with modification.
    - Phase 3's in-place replacement mechanism is rejected.
38. **Which require modification?**
    - Phase 0: pin the commit, set the branch, run the anchors.
    - Phase 1: reclassify the refuted and SIMP adaptive presets; note 75 target files.
    - Phase 2: upstream-first (2.4).
    - Phase 3: add a named preset, no replacement, keep aliases, extend the caveat.
    - Phase 4: keep the frozen test and add a new one; fix the PROVENANCE inconsistency.
    - Phase 5: per-preset assertions, formulation-bearing display name, evaluator-model column.
39. **Should Phase 6 changes to other methods be deferred?** Yes.
    - 6.1 is a benchmark-policy change and 6.2 a scientific-method change; both need separate authorization.
    - 6.3 is reporting-only and may proceed independently.
    - The uncommitted §7 is likewise out of scope.
40. **Is migration ready?** Ready with a named formulation split, under the seven conditions in MIGRATION_GATE.md.
41. **What is the single next action?** Option C.
    - First promote the stage-exhaustion controller and `hist.tOuter` upstream as default-off options on top of `6b08708`.
    - Verify A1_frozen160, the C320 three-rung record and S160x20 bitwise there.
    - Then execute the modified migration as a byte copy plus the named-preset addition.

## Where to look

| question | document |
|---|---|
| identities, snapshot, sweep recomputation | SOURCE_IDENTITY.md, TARGET_IDENTITY.md, PROVENANCE.md |
| every difference | SCIENTIFIC_DELTA_TABLE.md / MASTER_DELTA.csv, SOURCE_TO_TARGET_FILE_MAP.md, CONFIG_COMPARISON.md |
| formulation, low-density modes | FORMULATION_COMPARISON.md, MASS_STIFFNESS_COMPARISON.md |
| inner solver, filter, multiplicity, FE | SOLVER_COMPARISON.md, FILTER_COMPARISON.md, MULTIPLICITY_COMPARISON.md, FE_KERNEL_COMPARISON.md |
| controller, stop | CONTROLLER_COMPARISON.md, STOPPING_COMPARISON.md |
| the 480×60 experiment, grayness | 480_MATCHED_EXPERIMENT.md |
| first divergence, same-state | FIRST_DIVERGENCE.md |
| causes, migration | CAUSAL_ATTRIBUTION.md, MIGRATION_CLASSIFICATION.md, MIGRATION_GATE.md |
| preregistration | AUDIT_PREREGISTRATION.md, PREREGISTRATION_AMENDMENT_1.md |
| figures | `figures/fig01…fig20` |
