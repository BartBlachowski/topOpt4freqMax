# REPORT — one 240×30 run to resolve the THRESHOLD_SPLITTING concern

**Exactly one** scientific optimization run was executed: `C240x30`, the
**unchanged** frozen four-rung `A OR B` controller.

> **Headline.** The run **converged at 1358**. The exact three-rung endpoint
> `S3` occurs at **284** (Branch B, offset 38, full persistence). The final
> `move = 0.005` rung then ran for **1074 more outer iterations and 38 675 more
> inner MMA iterations — 87.5 % of the entire budget — to deliver a `ω₁` gain of
> +0.00672 %, against a frozen bar of 0.10 %.** It is below **every** frozen
> materiality bar, at the endpoint *and* at its running best anywhere in the
> tail. Rung 4 is now measured directly on **four** meshes, spanning −0.005 % to
> +0.020 %, with zero material results. **`THRESHOLD_SPLITTING_CONCERN_RESOLVED`.**

---

## The four required verdicts

```
C240_THREE_RUNG_COUNTERFACTUAL_EXACT
THRESHOLD_SPLITTING_CONCERN_RESOLVED
THREE_RUNG_ARCHITECTURE_SUPPORTED
THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED
```

All eleven Phase-23 conditions hold.

---

## 1. The forty-six questions

**1. What branch/starting HEAD was used?**
`benchmark-methodology-r2`, HEAD `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`,
re-read at task start rather than inherited from a previous brief.

**2. Was the starting tree clean or dirty?**
**Dirty — 22 paths** (four prior studies' uncommitted deliverables and two
top-level gate files), becoming 23 once this study's directory was created.
**None under `+impl/`.**

**3. What was the `+impl` hash?**
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files),
at start **and** at end. `git status` reports 0 modifications under `+impl/`.

**4. Did provenance pass?**
Yes — **`C240_PROVENANCE_PASS`**, 13/13: currentness `CURRENT`, source integrity
`PASS`, published MMA resolving to `+impl/mma_published/mmasub.m`, sensitivity
filter resolving correctly, forbidden Olhoff paths absent, four-rung controller
unambiguous, tolerance law identical at all four meshes, four inherited
preregistration digests valid, 27/27 required artifacts hashed, and the
controller-, two-rung- and three-rung-study finalization gates all PASS.

**5. Was prior three-rung evidence sealed before execution?**
Yes, verified independently **before** the run: `FINAL_SHA256.txt` 43/43 digests
re-verified with `shasum -c`, `DATA_MANIFEST.json` covering the identical
43-file set, `EVIDENCE.json` 11/11 declared artifacts hash-valid, and the
recorded verdicts intact. No prior evidence was modified and no prior verdict
rewritten.

**6. Was exactly one scientific optimization run executed?**
**Yes — exactly one.** The driver refuses to overwrite an existing trajectory,
and no other solve of any kind was performed.

**7. Was it exactly 240×30 with the frozen four-rung controller?**
Yes. `NE = 7200`, `move.levels = [0.04 0.02 0.01 0.005]`,
`move.continuation.signal = stop.rule = 'stageExhaustion'`, `tol = 0.075`,
cap 1600. The resolved config hash
`33833323efa08facaa5849c24fe32d6c9c47b5924f88c00d34fb65f7140d54d6` was
**asserted in code before the solve** against the preregistered value.

**8. Was the controller unchanged?**
Yes, and recovered **by call rather than by copy**: `cv_config('C',240,30)`,
`cv_telemetry` and `cv_export` from `two_branch_controller_validation` were
invoked unmodified, so the controller is identical *by reference* to the one that
produced the 160/320/400 evidence. `cv_run.m` hard-refuses other meshes by design
and is a hashed artifact of a sealed study — it was **not edited**; the new
driver mirrors its call sequence and adds six *extra* asserts, removing none.

**9. Were A/B unchanged?**
Yes. `exhaustion.m` was read and hashed
(`17b37a384b1aa5d987d9c861e16071d1140af130f92406ccc11cec4518bcae0c`), never
written. `W = 20`, `P = 20`, `W_np = 10`, `tol = 0.05·√(NE/3200)`, both
predicates, the union, the persistence and the stage-local reset are unchanged —
and the offline replay reproduces the in-loop trace element-wise in all four
stages as proof.

**10. Were all scientific formulation fields unchanged?**
Yes. Field-for-field against each prior causal arm, exactly five fields differ:
`nelx`, `nely`, `stop.tolerance` (the deterministic `0.05·√(NE/3200)` law),
`runtime.name`, and `provenance.overrides` — the last **verified element-wise**
to differ in exactly the `nelx`, `nely` and run-label entries. `p = 3` fixed,
mass `eq4b`, `q = 1`, sensitivity filter on all, `R = 0.06·b`, projection off,
subspace size 2 with off-diagonals on, published MMA, same FE formulation,
eigensolver, objective and volume constraint. `singleFactorOk = 1`, `lockOk = 1`.

**11. What safety cap was preregistered?**
**1600** — inherited from the `CAP` constant in `cv_config.m`, the same cap every
prior causal arm used. Not chosen for this run and not altered afterwards. The
run converged at 1358, so it was never reached.

**12. What were S1/S2/S3/F indices?**
**S1 = 206, S2 = 245, S3 = 284, F = 1358.** Stage starts 1 / 207 / 246 / 285.

**13. Which branch fired at S1/S2/S3?**
**Branch B at all three** — and at the terminal event too. This groups 240×30
with 400×50 (B at every stage), unlike 160×20 (A, A, B) or 320×40 (A, B, B).

**14. What were each stage's start indices?**
1 (move 0.04), 207 (0.02), 246 (0.01), 285 (0.005).

**15. Did lower-stage declaration timing reproduce `stageStart + 38`?**
**Partly — and the exception is important.** Stages 2 and 3 declared at offset
**38**, the arithmetic minimum. **Stage 4 did not: it took 1073.** The earlier
generalisation was too broad; the correct narrow statement is that `move = 0.02`
and `move = 0.01` declare at the minimum, while `move = 0.005` does not.

**16. Was `E` already true at the earliest evaluable lower-stage window?**
For stages 2 and 3, **yes** — true from the first evaluable iteration (226, 265)
and unbroken through the persistence window (100 % of the window). For stage 4,
**no**: `E` was false when first evaluable (304), first became true at 327, and
held only **5.9 %** of the stage.

**17. Is the 240×30 three-rung counterfactual exact?**
**Yes.** Re-derived under the current effective configuration, not inherited: the
one site depending on the ladder tail (`olhoffSolve.m:485`) never runs, for two
independent reasons (`anyStopGuard = false` *and* `exhaustStop = true`). The two
active sites depend on ladder **length** only and are consumed exclusively with
`ex.declared`, false between declarations. All ten post-run validity checks pass
and the replay matches element-wise in every stage.
**`C240_THREE_RUNG_COUNTERFACTUAL_EXACT`.**

**18. What is `ω₁(S3)`?** **167.03846293238027**

**19. What is `ω₁(F)`?** **167.04969317770568**

**20. What is the exact relative rung-4 `ω₁` benefit?**
`100 · (167.04969317770568 − 167.03846293238027) / 167.03846293238027` =
**+0.0067231493443241 %**, using the frozen denominator convention (normalised by
the earlier state).

**21. Is it above or below 0.10 %?**
**Below — by a factor of 14.9.** Classification **`IMMATERIAL`**. No uncertainty
padding was applied; none was preregistered.

**22. What is `M_nd(S3)`?** **12.916524091561493**

**23. What is `M_nd(F)`?** **12.942486226218918**

**24. Is rung-4 `M_nd` benefit material?**
**No — and it is not a benefit.** `M_nd` moves **+0.20100 %**, i.e. the design
ends *less* discrete than at `S3`. Immaterial against the 2 % bar either way.

**25. Is `S3 → F` topology change material?**
**No.** Mean \|Δρ_e\| = **0.002120** against a 0.01 bar; Δgray = −0.000556,
Δmid = +0.000278, both far below 0.01. Max \|Δρ_e\| = 0.06693.

**26. Does rung 4 provide any material multiplicity/gap benefit?**
**No.** Subspace size 2 at both states and throughout; mode order unchanged;
`ω₂ > ω₁` everywhere; no NaN/Inf; zero non-converged inner solves. The gap moves
−0.001306, and gap magnitude alone is explicitly not a materiality criterion
because the objective is `ω₁`.

**27. Does rung 4 provide any material volume benefit?**
**No.** \|volume − 0.5\| goes 4.48e-07 → 7.99e-07, a *worsening* of 3.51e-07 —
28× below the 1e-5 bar, and both states are two orders of magnitude inside the
1e-4 acceptance gate.

**28. How many outer iterations occur after S3?**
**1 074** — 79.1 % of the whole run.

**29. How much inner MMA work occurs after S3?**
**38 675 inner MMA iterations — 87.5 % of the whole run.** Rungs 2 and 3 together
cost 1 476; rung 4 costs **26×** that.

**30. Does full four-rung C240 CONVERGE or CAP_HIT?**
**CONVERGED @1358**, Branch B, full 20-iteration persistence, window [1339, 1358],
with 242 iterations of headroom below the 1600 cap.

**31. If CAP_HIT, is S3 already a scientifically valid terminal state?**
Not applicable — the run converged. `S3` is independently a valid persistent-`E`
exhausted state: declared, Branch B, `nB = 20`, at `move = 0.01`, zero
bound-active elements, native stop predicate already holding.

**32. If CAP_HIT, did rung 4 produce any material gain before pathology?**
Not applicable, but the preregistered running-best tail analysis was performed
anyway: the **best** `ω₁` anywhere in the 1074-iteration tail is 167.059612 at
iteration 1175 (**+0.01266 %** vs `S3`, still 7.9× below the bar) and the best
`M_nd` is 12.88639 at 609 (**−0.23330 %**, 8.6× below the bar). **At no point in
the tail was rung 4 materially ahead of `S3`.**

**33. How does 240 rung-4 value compare with 160?**
160×20: +0.02025 % `ω₁`. 240×30: **+0.00672 %** — 3× smaller. Both immaterial.

**34. How does it compare with 320?**
320×40: **−0.00504 %** — rung 4 there moves the objective *backwards*. 240×30 is
positive but still 14.9× below the bar. Both immaterial.

**35. How does it compare with 400?**
400×50: +0.00815 %. 240×30's +0.00672 % is the closest match of the three — as the
preregistration predicted, since both meshes exit stage 1 on Branch B with
`‖Δρ‖₂/tol` near 0.6–0.7.

**36. Does the new evidence resolve the threshold-splitting concern?**
**Yes.** The concern was that rung 4's dismissal at 160×20 rested on subdividing
one above-bar block into two below-bar halves. At 240×30 there is nothing to
subdivide: the combined `S2 → F` residual is +0.00778 %, itself 13× below the bar,
so rung 4 is measured **in isolation** with no dependence on where the block
boundary falls — and it delivers nothing on any bar.
**`THRESHOLD_SPLITTING_CONCERN_RESOLVED`.**

**37. Does it confirm the concern?**
No. Confirmation would require rung 4 to be materially beneficial at 240×30. Its
largest effect on any frozen bar, at its most generous reading, is 7.9× below
that bar.

**38. Is any threshold retuned?**
**No.** `ω₁` 0.10 %, `M_nd` 2 %, topology 0.01, volume 1e-5, multiplicity
qualitative, cost-domination 2× — all inherited verbatim, with the same
denominator convention. Nothing loosened, tightened, replaced, padded or made
mesh-specific.

**39. Was any new branch added?**
**No.** No Branch C. `E = A OR B` unchanged.

**40. Was any architecture other than the frozen four-rung run and exact
three-rung counterfactual tested?**
**No.** One run of `[0.04, 0.02, 0.01, 0.005]`; one exact counterfactual,
`[0.04, 0.02, 0.01]`. No `[0.04, 0.01]`, no `[0.04, 0.02, 0.005]`, no adaptive
move, no mesh-dependent ladder, no fixed dwell.

**41. Is the three-rung architecture now supported?**
**Yes — all eleven Phase-23 conditions hold.** See §3.

**42. Is three-rung policy preregistration justified?**
**Yes — `THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED`.** The splitting concern is
resolved and no other scientific blocker remains.

**43. Was production changed?**
**No.** The preset still resolves to `move.levels = [0.04 0.02 0.01 0.005]` with
`move.continuation.signal = 'boundVariable'`, verified in the static audit.
**`PRODUCTION_CONTROLLER_NOT_CHANGED`.**

**44. Did fail-closed finalization pass?**
**Yes — G1–G5 all PASS.** `EVIDENCE.json` declares the raw evidence; every
required declared artifact is present and hash-valid; `FINAL_SHA256.txt` exists
and is self-verifying; no `.mat` named by any manifest is absent.

**45. Is all evidence hash-valid?**
Yes. 27/27 required artifacts at the Phase-0 gate; four inherited preregistration
digests verified live; the new trajectory declared by measured SHA-256; and
`FINAL_SHA256.txt` re-verified independently with `shasum -c`.

**46. Is the nine-mesh campaign still blocked in this task?**
**Yes — `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`**, by Phase 26, unconditionally
and even though the architecture is supported.

---

## 2. The two findings that are new science, not bookkeeping

### 2.1 `move = 0.005` is generically pathological, and 320×40 was not an anomaly

Stage 4 at 240×30 spent **92.4 %** of its 1074 iterations in *low-amplitude
cancellation* — `‖Δρ‖₂ < tol` together with `med₂₀ cosθ < 0` — which is precisely
the documented hole in the union: too small for Branch A, not coherent enough for
Branch B. `E` was true only 5.9 % of the time. The run escaped only when a
20-iteration coherent window finally appeared, at 1358.

320×40 entered the same regime and no such window appeared before its cap.

**So `CONVERGED` versus `CAP_HIT` at `move = 0.005` is not a difference of kind
but of luck** — whether a qualifying window happens to appear before the cap. The
earlier 320×40 result therefore looks less like a one-mesh anomaly and more like
the generic terminal behaviour of this rule at the finest move level. Two of four
meshes show it; the other two (160×20, 400×50) declare at the arithmetic minimum.

This is recorded as an observation. **No rule was changed on the strength of it**,
and no mesh law is fitted — the four meshes show no ordering in `NE`
(3 200 fast, 7 200 slow, 12 800 never, 20 000 fast).

### 2.2 The `stageStart + 38` generalisation was too broad

The prior audit found every firing lower stage declaring at the arithmetic
minimum. That holds here for stages 2 and 3 and **fails for stage 4** (offset
1073). The supportable statement is narrower than before:

> At `move = 0.02` and `move = 0.01` the exhaustion condition is already
> satisfied once enough post-transition history exists, and those stages cost
> exactly 39 outer iterations. At `move = 0.005` it is not.

Correcting a previous over-generalisation is part of the result, not a footnote.

## 3. Verdict derivation

| Phase-23 condition | result |
|---|---|
| 1. C240 run valid | ✅ CONVERGED, 0 non-converged inner, exact rebuild, all asserts passed |
| 2. S3 counterfactual exact | ✅ 10/10 checks + element-wise replay + static audit |
| 3. S3 a valid persistent-`E` exhausted state | ✅ declared, Branch B, `nB = 20`, `move = 0.01` |
| 4. rung 4 below all frozen bars at 240×30 | ✅ every bar, endpoint **and** running best |
| 5. no material multiplicity benefit | ✅ subspace 2→2, no mode change |
| 6. volume feasibility acceptable | ✅ 4.48e-07, two orders inside the gate |
| 7. no threshold changed | ✅ all inherited verbatim |
| 8. A/B unchanged | ✅ hashed, read-only, replay-verified |
| 9. 160/320/400 rung-4 evidence still valid | ✅ prior study re-verified, 43/43 + 11/11 |
| 10. threshold-splitting concern resolved | ✅ |
| 11. retention / finalization passes | ✅ G1–G5 |

→ **`THREE_RUNG_ARCHITECTURE_SUPPORTED`** →
**`THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED`**.

## 4. One honest caveat

The preregistration disclosed **before the run** that the direction of this result
was foreseeable: the prior *fixed-move* 240×30 arm already showed this mesh
leaving `move = 0.04` on Branch B with `‖Δρ‖₂/tol = 0.68` and zero bound-active
elements — grouping it with 400×50, where rung 4 was already known to be
immaterial. The preregistration therefore predicted `S1 = 206 ± 10` Branch B and
an immaterial rung 4, and both came true.

That does not weaken the evidence: the measurement is real, independent, and was
the only way to obtain a 240×30 rung-4 number, since no trajectory at this mesh
had ever descended below `move = 0.04`. But this run **confirmed an expectation**
rather than probing a genuinely open direction, and saying so is more useful than
presenting it as a blind test.

What was genuinely unpredicted, and is the more interesting result, is §2.1.

## 5. Figures

`F01` `ω₁` history · `F02` `M_nd` history · `F03` move/stage history ·
`F04` A/B/E by stage · `F05` declaration timing · `F06`–`F08` rung-by-rung `ω₁`,
`M_nd`, topology · `F09`–`F10` cumulative outer and inner work ·
`F11` S3 vs F topology · `F12` gap/multiplicity ·
`F13`–`F15` cross-mesh rung-4 `ω₁`, `M_nd`, cost ·
`F16` threshold-splitting concern and resolution.

---

## 6. Final summary

| | |
|---|---|
| branch | `benchmark-methodology-r2` |
| starting HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| final HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (no commit made) |
| dirty state start / end | 22 paths / 24 paths (this study's directory + its durable evidence directory) |
| `+impl` hash start / end | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` / **unchanged** |
| **scientific runs** | **1** |
| mesh | **240 × 30** |
| controller | frozen four-rung `A OR B`, `[0.04 0.02 0.01 0.005]`, unchanged |
| tests | 6/6 PASS at end (`test_finalization_gate` failed at start only, self-referentially) |
| preregistration hash | `f86d022e5259beb0936072d204e54e3761f14b2fd903c1794a6d4ccb2c5652cc`, frozen `2026-09-09T17:44:21Z`, before the run |
| evidence manifest | `EVIDENCE.json` declared + hash-valid · `DATA_MANIFEST.json` · `FINAL_SHA256.txt` self-verifying |
| finalization gate | **PASS** (G1–G5) |

```
C240_THREE_RUNG_COUNTERFACTUAL_EXACT

THRESHOLD_SPLITTING_CONCERN_RESOLVED

THREE_RUNG_ARCHITECTURE_SUPPORTED

THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED

PRODUCTION_CONTROLLER_NOT_CHANGED

NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```
