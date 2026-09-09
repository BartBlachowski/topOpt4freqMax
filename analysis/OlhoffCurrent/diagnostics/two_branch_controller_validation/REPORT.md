# REPORT — causal validation of the frozen two-branch stage-exhaustion controller

**Does replacing β-driven move continuation with the frozen `E = A OR B` rule
actually make the optimizer produce better trajectories?**

**On topology, decisively yes — and the benefit comes from one place. On
termination, not everywhere.** Holding `move = 0.04` until the frozen rule
declares exhaustion cut final `M_nd` by **44.7 %** at 320×40 and **52.6 %** at
400×50 while *raising* ω₁ on all three meshes. Every one of the nine move
transitions was attributable to a completed 20-iteration `A` or `B` window, and
none to β, whose stall predicate was already true at all nine. But at 320×40 the
terminal rung entered a **low-amplitude cancelling** regime that satisfies
neither branch, and the run reached the 1600 cap without ever admitting
convergence — the exact blindness recorded as known limitation 1 before this task
began.

| | |
|---|---|
| Branch | `benchmark-methodology-r2` |
| HEAD at task start | `b6014ba8bca41f85671d79ab4c8bdee7419880bb`, tree **clean** |
| HEAD at task end | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (a commit made by the user mid-task; see Q1) |
| `+impl/` tree | `c1455374…` (74 files) → `edbfe47e…` (75 files) |
| MATLAB / threads | 25.2.0.2998904 (R2025b) / `maxNumCompThreads(1)` |
| Preregistration | SHA-256 `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf`, frozen **2026-09-08T18:51:03Z**, before the controller existed |
| Scientific runs | **exactly three** — 160×20, 320×40, 400×50 |
| Solver copies | **none** |
| Production behaviour changed | **no** — bitwise verified |

Detail: [`CAUSAL_ANALYSIS.md`](CAUSAL_ANALYSIS.md), [`PROMOTION.md`](PROMOTION.md),
[`PROVENANCE.md`](PROVENANCE.md), [`IMPLEMENTATION.md`](IMPLEMENTATION.md),
[`SINGLE_FACTOR_AUDIT.md`](SINGLE_FACTOR_AUDIT.md), [`BASELINES.md`](BASELINES.md),
[`PERFORMANCE_READINESS.md`](PERFORMANCE_READINESS.md), `METRICS.json`, `figures/F1`–`F14`.

---

## Headline

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| production → candidate `M_nd` [%] | 13.4025 → **12.7041** | 23.3596 → **12.9233** | 32.3283 → **15.3311** |
| Δ`M_nd` | **−5.21 %** | **−44.68 %** | **−52.58 %** |
| production → candidate ω₁ | 169.4952 → **170.0113** | 165.9508 → **166.4267** | 162.8826 → **166.4563** |
| Δω₁ | **+0.304 %** | **+0.287 %** | **+2.194 %** |
| first descent, production → candidate | 79 → **103** | 130 → **275** | 138 → **389** |
| candidate status | `CONVERGED` @ 219 | **`CAP_HIT` @ 1600** | `CONVERGED` @ 505 |
| cost (outer / inner / wall) | ×2.41 / ×2.26 / ×3.39 | **×12.21 / ×29.28 / ×88.60** | ×3.63 / ×3.53 / ×6.54 |

**Where the gain comes from.** ≥ 98 % of the fine-mesh improvement is banked
*before the first descent*: −44.21 of −44.68 % at 320×40, −51.53 of −52.58 % at
400×50. The ladder below 0.04 costs iterations and buys almost nothing.

**The one clean failure.** 320×40's terminal rung ran 1248 iterations in which
`M_nd` moved 0.40 % of its own value, because amplitude ≈ 0.049 × `tol` blocks
Branch A while `med₂₀cosθ ≈ −0.885` blocks Branch B.

---

## The fifty required answers

**1. What branch/starting HEAD was used?** Branch `benchmark-methodology-r2`,
HEAD `b6014ba8bca41f85671d79ab4c8bdee7419880bb`. The final HEAD is
`1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`: the user committed the
work-in-progress under the message "A & B tests" at 2026-09-09 07:28:32 +0200,
while C400 was still running. That commit was not made by this task, was not
amended or reverted, and changed no content.

**2. Was the starting tree clean?** **Yes** — `git status --porcelain` was empty
at task start (`evidence/provenance_start.json`).

**3. What exact frozen A definition was recovered?**
```
A(k) = med_20 cosθ(k) < 0  AND  med_20 net_path(k) < 0.5  AND  ‖Δρ_k‖₂ >= tol(NE)
```
From `two_branch_maturity_240/PREREGISTRATION.md` §4 and `scripts/tb_branches.m`.

**4. What exact frozen B definition was recovered?**
```
B(k) = ‖Δρ_k‖₂ < tol(NE)  AND  med_20 cosθ(k) > 0
```
§5 of the same file — the inherited native design-change stop plus a coherence
guard plus persistence, and presented as nothing more.

**5. What exact persistence/window/indexing semantics were recovered?**
`W = 20` trailing median window, `'omitnan'`, defined only from `k ≥ W`;
persistence `P = 20` consecutive iterations; `W_np = 10` for `net_path`;
`ρ_k` is the design after outer iteration `k`, `ρ_0` the uniform initial design,
`Δρ_k = ρ_k − ρ_{k−1}`; `cosθ` and `net_path` formed from the visited designs
while the amplitude is `hist.dxNorm2 = ‖drho‖₂`, the increment the sub-problem
returned; `tol(NE) = 0.05·√(NE/3200)`, which is exactly `cfg.stop.tolerance`
under the production `meshScaled` rule, so **no new constant is introduced**. The
event is the first iteration at which either branch's sustained 20-window begins.

**6. Did those definitions exactly match the withheld 240×30 preregistration?**
**Yes**, and this was verified numerically rather than by reading. Re-running the
frozen `tb_branches` against the one surviving raw fixed-move trajectory
(`F400_400x50_trajectory.mat`) reproduced nine recorded 400×50 quantities
bit-exactly, including `kB = 369`, `med₂₀cosθ(369) = 0.9937451892796663`,
`M_nd(369) = 16.158892933214315` and β-stall first firing at 138.
Verdict `CONTROLLER_DEFINITION_RECOVERY_PASS`.

**7. Was the causal preregistration frozen before candidate results?** **Yes** —
SHA-256 `8e323f83…` at 2026-09-08T18:51:03Z, before the controller was
implemented and therefore before any candidate result of any kind existed.

**8. Did the single-factor gate pass?** **Yes** —
`CONTROLLER_SINGLE_FACTOR_PASS`. At all three meshes exactly three schema fields
differ: `move.continuation.signal`, `stop.rule` (the declared intervention) and
`runtime.name` (a label the config hash excludes). The full scientific lock was
verified field-by-field and re-asserted in code before every solve.

**9. Did all software tests pass?** **Yes** — 17/17 controller tests, 0 failures,
and the repository suite 5/5, 0 failures. Test 16 is the load-bearing one: the
online detector reproduces `tb_branches` **element by element** on the real
400×50 arm and on four synthetic regimes, declaring at exactly `event + P − 1`.

**10. Were exactly three candidate scientific runs executed?** **Yes.** `cv_run`
refuses any mesh outside the authorized set.

**11. Were they exactly 160×20, 320×40 and 400×50?** **Yes.** No 240×30
candidate, no 480×60 or finer, no fixed-move rerun, no production rerun.

**12. Was the controller unchanged across all three?** **Yes** — one source, one
`+impl` tree hash `edbfe47e…` stamped into all three run records, no
mesh-specific parameter, no edit between runs.

**13. Did β lose all authority over continuation?** **Yes.** β stall was true at
**all nine** transitions and drove none. Replayed on the candidate's own path,
production's ladder would have reached the last rung by iterations 101 / 152 /
160, while the candidate was still at `move = 0.04`.

**14. Did β lose all authority over terminal admission?** **Yes.** Under
`stop.rule = 'stageExhaustion'` the §3.5.1 design-change test and its guards are
replaced wholesale; β is not read. β is still computed as the bound variable of
Eq. (25a) and logged for the counterfactual.

**15. What triggered every candidate move transition?** A completed 20-iteration
window: **A** at 160×20/103 and /142 and at 320×40/275; **B** at 160×20/181,
320×40/314 and /353, and all three at 400×50 (389, 428, 467). Full table with the
predicate inputs: `CAUSAL_ANALYSIS.md` §2.

**16. Did any transition occur without A OR B?** **No.** On every mesh the set of
iterations at which `move` changed equals exactly {declaration iteration + 1},
each advancing exactly one rung.

**17. Did 160×20 escape its high-amplitude mature cycle?** **Yes.** Branch A's
window ran 83–102 — beginning at iteration 83, the same iteration the frozen rule
recorded on the fixed-move arm — with `med₂₀cosθ = −0.907`, `med₂₀ net/path =
0.133` and amplitude 12.6 × `tol`. The move descended at 103 and the run
traversed all four rungs without churning.

**18. Did 160×20 suffer any scientific regression?** **No.** `M_nd` −5.21 %, ω₁
+0.304 %, gray 0.1494 → 0.1444, volume error 8.7e-7. Every reported quantity is
equal or better than production.

**19. How much later than production did 320×40 remain at `move = 0.04`?**
**145 iterations** — production descended at 130, the candidate at 275.

**20. What did it gain during that interval?** `M_nd` 23.322 → 13.012
(**−44.21 %**), ω₁ 165.943 → 166.422 (+0.288 %), gray 0.2633 → 0.1523,
mid-density 0.0950 → 0.0297.

**21. How much later than production did 400×50 remain at `move = 0.04`?**
**251 iterations** — production descended at 138, the candidate at 389.

**22. What did it gain during that interval?** `M_nd` 32.318 → 15.665
(**−51.53 %**), ω₁ 162.877 → 166.418 (**+2.17 %**), gray 0.3472 → 0.1822,
mid-density 0.1881 → 0.0398. Figure F8 shows the physical meaning: production
freezes two large grey blobs at the beam ends; the candidate resolves them into
members.

**23. Final production vs candidate `M_nd`?** 13.4025 → 12.7041 · 23.3596 →
12.9233 · 32.3283 → 15.3311.

**24. Final production vs candidate ω₁?** 169.49522702 → 170.01131578 ·
165.95078925 → 166.42669696 · 162.88261563 → 166.45627579.

**25. Relative `M_nd` changes?** **−5.21 %**, **−44.68 %**, **−52.58 %**.

**26. Relative ω₁ changes?** **+0.304 %**, **+0.287 %**, **+2.194 %** — all
improvements, in a maximization problem.

**27. Were volume constraints satisfied?** **Yes.** Final volumes 0.49999913,
0.49999962, 0.49999943; errors ≤ 8.7e-7 against the 1e-4 bound.

**28. Was multiplicity behaviour acceptable?** **Yes.** The fixed two-mode
subspace held at size 2 on every iteration of every run; minimum relative gap
≈ 0.0041–0.0042 on all three, the same near-coalescence production sees; no
NaN/Inf; zero inner-solver non-convergences. Final gaps are larger than
production's at the fine meshes (0.224 vs 0.107; 0.211 vs 0.077), which is the
expected signature of converging to a different, more discrete optimum — and ω₁
improved, so it is a better one.

**29. Did all candidate runs reach `move = 0.005`?** **Yes**, all three, via all
four rungs in order.

**30. Did all reported CONVERGED states satisfy terminal A OR B persistence?**
**Yes.** 160×20: Branch B, window 200–219, `nB = 20`, `move = 0.005`, stage 4.
400×50: Branch B, window 486–505, `nB = 20`, `move = 0.005`, stage 4. Neither is
a relabelled cap, failure or β event.

**31. Did any candidate hit a cap?** **Yes — 320×40**, `CAP_HIT` at 1600. It is
reported as a cap. The cap was preregistered at 1600 and **was not raised** after
seeing the trajectory.

**32. Did any inner solver fail?** **No.** Zero non-convergences across all 2324
outer iterations; maximum inner iterations in any outer step 53 / 139 / 46
against a limit of 500.

**33. Outer-iteration cost multiplier?** ×2.41 · **×12.21** · ×3.63.

**34. Wall-time multiplier?** ×3.39 · **×88.60** · ×6.54.

**35. Inner-MMA-work multiplier?** ×2.26 · **×29.28** · ×3.53.

**36. Did all preregistered promotion gates pass?** **No — 14 of 15.** P13, the
cost bound (≤ 8× outer and ≤ 10× wall), **fails at 320×40** at ×12.21 and ×88.60.
P1–P12, P14 and P15 pass.

**37. Primary controller-validation verdict?**
**`TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED`.** Not `VALIDATED`, because P13
fails and 320×40 never reached genuine terminal exhaustion — and a controller
whose terminal-admission rule cannot fire on one of three tested meshes is not
ready to be the production stopping criterion, however good its topology. Not
`REJECTED`: nothing on the REJECTED list occurred. Not `INCONCLUSIVE`: the cap
bound *after* all four rungs had been traversed and `M_nd` had already improved
44.7 %, so the effect and the failure are both fully legible.

**38. Was production promoted?** **No** — `PRODUCTION_CONTROLLER_NOT_PROMOTED`.
Promotion requires `VALIDATED` with every gate passing.

**39. If promoted, is promoted production exactly the validated controller?**
Not applicable — no promotion.

**40. Did promotion equivalence pass?** Not applicable — Phase 23 applies only if
promotion occurs, so no equivalence verdict is issued.

**41. Final production HEAD/hash?** HEAD `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`;
`+impl/` tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`
(75 files); currentness `CURRENT`; source integrity 75/75, 0 mismatches.

**42. Is the production source tree clean/frozen?** The production **controller
and behaviour** are unchanged and bitwise verified: the preset still resolves to
`boundVariable`/`designChange`, and `test_preset_equivalence` reproduced the
frozen conference record bitwise after all edits (ρ bitwise, ω₁ bitwise, 91
outer, 2241 inner). The **source tree is not byte-identical to task start**: it
carries the candidate controller layer, default-inactive, manifested at
`edbfe47e…`. It was kept because it is the instrument that produced this
evidence; it is reversible in one `git checkout` since all 75 files are tracked
(`PROMOTION.md` §4). Working tree at hand-off is not clean — this study's own
final documents and the C400 outputs are uncommitted.

**43. Is all evidence durable and hash-valid?** **Yes.** The three raw
trajectories (433 MB) are declared in `EVIDENCE.json` with SHA-256, byte size and
variable dimensions, and `olhoffcurrent_evidence_gate` returns **PASS, 3/3
required present and matching**. `DATA_MANIFEST.json` hashes 52 artifacts. No
required artifact exists only in a gitignored or scratch location: the
`diagnostics/.gitignore` re-includes `*.png`, `*.csv` and `*.txt`, and the large
`.mat` files are untracked **by declaration**, which is the policy's explicit
distinction.

**44. Is any controller-specific scientific blocker unresolved?** **Yes, one.**
The frozen union has a hole at **low-amplitude cancellation** — `‖Δρ‖₂ < tol`
together with `med₂₀cosθ < 0` satisfies neither branch — and 320×40's terminal
rung sits in it for 1248 iterations. Not repaired here: Phase 19 freezes the
controller, and any fix would be an outcome-driven change made after seeing that
trajectory.

**45. Is the nine-mesh performance campaign now authorized?** **No** —
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`, for two independent and sufficient
reasons: the verdict is not `VALIDATED`, and an unresolved blocker remains. The
campaign would add five meshes finer than 400×50 — the direction in which
amplitude falls further against a move-independent `tol(NE)`, i.e. deeper into
the region where Branch A cannot fire.

**46. Does anything in this study justify projection?** **No.** Projection was
off throughout, is asserted off before every solve, and the solver now refuses
the exhaustion controller under projection outright. Nothing here bears on it.

**47. Does anything justify changing `R = 0.06·b`?** **No.** The filter radius
was identical in both arms at every mesh and was never varied. The `M_nd`
differences are produced by *when the move ladder descends*, which the prefix
check proves is the only thing that differed.

**48. Does anything justify changing `p`/mass/`q`?** **No.** `p = 3` fixed, mass
Eq. (4b), `q = 1`, no continuation, asserted in code before every solve and
verified field-by-field in the single-factor gate.

**49. Were any thresholds tuned after candidate results?** **No.** `A`, `B`,
`W = 20`, `P = 20`, `W_np = 10`, `tol(NE)`, the union, the reset semantics, the
move ladder and the 1600 cap are all exactly as frozen. One disclosed correction
was made **before** the first candidate run: `PREREGISTRATION.md` §8 test 5 gives
the earliest post-transition declaration as `s+39`/`s+47`, which is an arithmetic
slip in a test description; the binding §3.2 exactness requirement forces `s+38`
on both branches. The frozen file is unaltered, the implementation follows §3.2,
and nothing else is affected — the earliest theoretical declaration is far before
any observed event (102, 274, 388).

**50. Were any additional scientific runs executed?** **No.** Exactly three
candidate runs. The only other solves were software tests at 80×10 (≤ 45
iterations, explicitly not scientific evidence) and the 160×20
`test_preset_equivalence` regression, which is a production-reproduction test
against a pre-existing frozen record.

---

## Methodological distinction (Phase 25), stated plainly

* **Withheld mechanism validation** = **240×30**. The rule was frozen before that
  mesh was run and passed without retuning.
* **Causal controller validation** = the **160×20 / 320×40 / 400×50** runs here.
  The rule was *built* using information from these three meshes, so these runs
  are **not** new withheld validation of the maturity rule. They test a different
  proposition: whether intervening on the optimizer with the already-frozen rule
  causes the desired optimization behaviour.

The two are never conflated, and no claim of out-of-sample validation is made for
the present runs.

## Known limitations carried forward, and what happened to them

1. **Branch A is blind to sufficiently low-amplitude cancellation.** **Realized.**
   This is exactly why 320×40 capped. The limitation was recorded before the task
   and is now demonstrated to have operational consequences, not merely
   diagnostic ones.
2. **The trajectory pathway taxonomy does not generalize.** Confirmed again:
   320×40 descended on Branch **A** and then exhausted stages 2–3 on Branch **B**,
   and 160×20 mixed A, A, B. No mesh follows one pathway.
3. **The surviving claim is only that stage exhaustion is detectable by the
   frozen union.** Upheld for nine of nine transitions and two of three terminal
   admissions; **not** upheld for the third.
4. **Branch B is not novel physics.** Preserved. It is the inherited native stop
   plus a coherence guard plus persistence, and at 400×50 it fired at exactly the
   iteration the native stop first holds (369).
5. **160×20 remains the weakest mechanism case.** Its old acceptance rule was not
   retroactively tightened. In the causal test it behaved well: the later rungs
   recovered the improvement Branch A left on the table, `M_nd` falling
   13.036 → 12.704 across stages 2–4.

## Disclosures

* **Missing prior evidence.** Five raw `.mat` artefacts named by earlier frozen
  studies are absent from this machine (`PROVENANCE.md` §5) — the same retention
  loss `EVIDENCE_POLICY.md` was written about, recurring in studies completed
  after that policy. Consequence: the 160×20 and 320×40 production final density
  fields do not exist, so density-field distance and production topology images
  for those two meshes are marked **unavailable** and are not reconstructed.
* **MATLAB build.** The 160×20 and 320×40 production baselines were produced under
  25.2.0.3042426 (Update 1); the candidate runs used 25.2.0.2998904. The 400×50
  comparison is same-binary; the other two are same-configuration but not
  same-binary, and are never described as bitwise.
* **Licence interruption.** A transient MathWorks network-licence outage between
  C320 and C400 blocked MATLAB for a few minutes. C400 was launched
  automatically once the licence returned; no run was affected, and the analysis
  was ported to Python so it no longer depends on licence availability.
* **A user commit landed mid-task** (Q1). It changed no content.
