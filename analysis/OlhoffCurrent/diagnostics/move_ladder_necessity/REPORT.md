# REPORT — is the four-rung move ladder still scientifically justified?

**Zero scientific optimization runs were executed.** Every number below comes
from trajectories that already existed.

> **Short answer: the ladder cannot simply be removed, and it cannot be kept as
> it is.** At 160×20 the lower rungs are load-bearing — the single-stage endpoint
> is *worse than production* in ω₁, and only the rungs below 0.04 recover it. At
> 400×50 they buy a marginal 2.1 % `M_nd`. At 320×40 they buy nothing material,
> cost 97.7 % of the run's wall time, and end in `CAP_HIT`. And rungs 3 and 4 are
> immaterial on **every** mesh by **every** preregistered criterion.

| | |
|---|---|
| Branch | `benchmark-methodology-r2` |
| HEAD (start = end) | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| `+impl/` tree | `edbfe47eb…152cb` (75 files) — **unchanged by this task** |
| Preregistration | `a08b879b9d4893bc55dd9e5dfc2c26f897cb9d49355f0557ccc8644c7fc6ca3e`, frozen 2026-09-09T15:17:49Z |
| Scientific runs | **zero** |

Detail: [`LADDER_VALUE_ANALYSIS.md`](LADDER_VALUE_ANALYSIS.md) ·
[`CROSS_MESH_ANALYSIS.md`](CROSS_MESH_ANALYSIS.md) ·
[`SINGLE_STAGE_COUNTERFACTUAL.md`](SINGLE_STAGE_COUNTERFACTUAL.md) ·
[`MIDTASK_COMMIT_AUDIT.md`](MIDTASK_COMMIT_AUDIT.md) ·
[`RETENTION_AUDIT.md`](RETENTION_AUDIT.md) ·
[`EVIDENCE_INVENTORY.md`](EVIDENCE_INVENTORY.md) · `METRICS.json` · `figures/F1`–`F12`

---

## The forty-two required answers

**1. What branch/HEAD was audited?** `benchmark-methodology-r2` at
`1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`, unchanged from start to end of this
task. Working tree dirty at start (17 paths, the previous study's uncommitted
deliverables) and still dirty at end.

**2. Was the previous controller-validation study frozen safely?** **Yes, after
one correction.** `EVIDENCE.json` verifies **3/3** required raw trajectories
present and hash-matching; `DATA_MANIFEST.json` verifies **53/53**; the
preregistration is byte-identical to its frozen copy. `FINAL_SHA256.txt`
initially verified 70/71 — its own `DATA_MANIFEST.json` digest had gone stale
when that manifest was refreshed after a cosmetic figure regeneration. The digest
was corrected and **the correction is recorded inside the file**, not made
silently. It now verifies **71/71**.

**3. What caused the mid-task HEAD movement?** The repository owner committed
`1438aa3` ("A & B tests") at 2026-09-09 07:28:32, while C400 was running. It is
an ordinary bookkeeping commit of work already on disk, made outside the agent
session.

**4. Did commit `1438aa3` affect any scientific execution?** **No.** Seven
`+impl/` files were in the commit, so this was audited rather than assumed:
committed blob == on-disk file for all seven and for `SOURCE_MANIFEST.json`;
every `+impl` mtime is 2026-09-08 20:54–20:56, before C160 even started, and
nothing was written on 2026-09-09; all three runs stamped `implTree =
edbfe47e…152cb` at run time, which is the current tree hash. Decisively, the
commit landed during C400's outer iteration **~191**, which lies *inside* the
1–369 window that is bitwise identical to the independently produced `F400` arm —
so the bitwise identity **straddles the commit instant**.
`MIDTASK_COMMIT_NONINTERFERENCE_VERIFIED`.

**5. Is C400 still scientifically valid?** **Yes**, and so are C160 and C320,
both of which had already finished when the commit was made.

**6. Which prior raw trajectories are missing?** **Twelve hash-manifested
artifacts across five studies**:

```
dynamical_regime              evidence/dr_analysis.mat, runs/runA_400x50.mat, runs/runB_320x40.mat
fixedmove_400_dynamics        evidence/fm_analysis.mat, runs/runC_400x50.mat
move_stop                     runs/{baseline,fixedmove}_{160x20,320x40}.mat
topology_maturity_transition  evidence/phaseA_stats.mat
two_branch_maturity_240       evidence/tb_analysis.mat, runs/runD_240x30.mat
```

plus ~7 further trajectories referenced in prose or code that were never hashed
at all. **This corrects the previous report**, which stated five — that was only
the subset that study happened to probe. Nothing is corrupted: 327/327 surviving
digests match.

**7. Why were they lost despite `EVIDENCE_POLICY.md`?** Because the policy is
**opt-in and its gate cannot see a study that never declares anything**. The
correlation is perfect and is a function of *when*, not of care: every study
finalized **before** commit `5fbec9a` (which introduced the policy, the gate and
the declare helper) lacks an `EVIDENCE.json`, and five of them lost data; every
study finalized **after** it has one and has lost nothing. Two studies
(`admission_rule`, `move_transition`) show a clean `FINAL_SHA256.txt` only
because they never hashed their raw data — a green check that is green because
nothing was checked.

**8. Is future raw-evidence retention now fail-closed?** **Yes, for future
studies.** `olhoffcurrent_finalization_gate.m` refuses to call a study finalized
unless: G1 it declares its evidence (**"declares nothing" is no longer a passing
state** — the hole that lost the data), G2 every required declared artifact is
present and hash-valid, G3 a `FINAL_SHA256.txt` exists, G4 every digested line in
it resolves and matches (self-verifying, closing the staleness found in answer
2), and G5 no `.mat` the study *claims* to hold is absent. It counts only claims,
so a study that honestly documents an absent file is not penalised.
`test_finalization_gate.m` runs 12 checks, seven of them destructive fail-closed
cases, plus a ledger that fails if the eight legacy-deficient studies **grow** in
number. Two real bugs in the gate were caught by those tests before it was
accepted. The gate lives outside `+impl/` and is scientifically inert.

**9. Were ZERO new scientific optimization runs executed?** **Yes — zero.** No
mesh, no fixed-move arm, no production arm, no controller arm. The only MATLAB
executions were software tests, which build a sandbox study from `rand(32,12)`.

**10. What exact A OR B rule was used?** The frozen rule, unaltered:
`tol(NE) = 0.05·√(NE/3200)`, `W = 20`, `P = 20`, `W_np = 10`;
`A = med₂₀cosθ < 0 ∧ med₂₀ net/path < 0.5 ∧ ‖Δρ‖₂ ≥ tol`;
`B = ‖Δρ‖₂ < tol ∧ med₂₀cosθ > 0`; declaration at the first iteration where
either branch has held 20 consecutive times.

**11. Were first exhaustion events verified as 103/275/389?** **Yes, and
refined.** An independent re-implementation, run over raw `RHO` and
`hist.dxNorm2`, reproduces the controller's `A` and `B` **element-wise over every
iteration of every `move = 0.04` prefix**. The *declarations* are at **102 (A),
274 (A), 388 (B)**; the *first descents* are one iteration later at **103, 275,
389** — the frozen semantics `transition = declaration + 1`. The single-stage
endpoint is the declaration iteration.

**12. Are those states valid single-stage counterfactual endpoints?** **Yes.**
Both policies are identical functions of the trajectory up to the first
declaration and differ only in what they do afterwards, so no simulation is
needed. Checked, not merely argued: `move == 0.04` throughout every prefix,
`stage == 1` throughout, no descent recorded before the declaration, and — for
400×50 — iterations 1–369 are **bitwise identical** to the independently produced
`F400` fixed-move arm across ρ, all five ω, β, `‖Δρ‖₂`, inner counts, volume and
gap. No same-build fixed-move arm survives for 160×20 or 320×40, so bitwise
equivalence is **not** claimed there.

**13. S160 `M_nd`/ω₁?** `M_nd` = **13.036370 %**, ω₁ = **168.980391** (iteration
102, branch A).

**14. S320?** `M_nd` = **13.012132 %**, ω₁ = **166.421616** (iteration 274,
branch A).

**15. S400?** `M_nd` = **15.664940 %**, ω₁ = **166.417621** (iteration 388,
branch B).

**16. F160?** `M_nd` = **12.704121 %**, ω₁ = **170.011021** (`CONVERGED` @219).

**17. F320?** `M_nd` = **12.923311 %**, ω₁ = **166.418886** (**`CAP_HIT`** @1600).

**18. F400?** `M_nd` = **15.331078 %**, ω₁ = **166.456242** (`CONVERGED` @505).

**19. Lower-rung `M_nd` benefit per mesh?** 160×20 **−0.3322 (−2.549 %)**;
320×40 −0.0888 (−0.683 %); 400×50 **−0.3339 (−2.131 %)**. Material (bar −2.0 %)
at 160×20 and 400×50; the 400×50 margin is thin and is reported as marginal.

**20. Lower-rung ω₁ benefit per mesh?** 160×20 **+1.030631 (+0.6099 %)**;
320×40 −0.002730 (−0.0016 %); 400×50 +0.038621 (+0.0232 %). Material (bar
+0.10 %) **only at 160×20**.

**21. How much topology change occurs below `move = 0.04`?** Mean `|Δρ_e|`
between S and F: 0.00651 / 0.00340 / 0.00271. Gray fraction changes by −0.0019 /
−0.0002 / −0.0026, mid-density by +0.0000 / +0.0006 / −0.0006. **Below the 0.01
materiality bar on every mesh.**

**22. How much wall time is spent below `move = 0.04`?** **68.3 % / 97.7 % /
51.3 %** of each run — 290 s, **33 585 s**, 1 817 s.

**23. How much inner MMA work?** **46.5 % / 93.4 % / 28.8 %** — 2 358, **71 466**,
2 964 inner iterations. At 320×40 the terminal rung alone consumed 70 034 inner
iterations to move `M_nd` by 0.13 % of its own value.

**24. What percentage of production→candidate benefit is banked at the first
exhaustion event?** `M_nd`: **52.43 % / 99.15 % / 98.04 %**. ω₁: **−99.81 % /
100.58 % / 98.92 %**.

**25. Is the ≥ 98 % fine-mesh claim reproduced exactly?** **Yes** — 99.15 % at
320×40 and 98.04 % at 400×50, recomputed from the raw trajectories. It was
correctly scoped to the fine meshes: 160×20 banks only 52.4 % of the `M_nd` gain
and **−99.8 %** of the ω₁ gain.

**26. Does 160×20 also support ladder removal?** **No — it blocks it.** At S,
160×20's ω₁ is 168.9804 against production's 169.4952: **0.30 % worse than
production**, in a maximization problem. This is not a phase artifact of an
oscillating trajectory: the ω₁ *maximum* over the declaration window is still
below production, and across **all 102 iterations of the entire `move = 0.04`
stage the candidate's ω₁ never once reaches production's** (fraction above
production = 0.000). The lower rungs are what carry 160×20 from below production
to above it.

**27. Does any mesh show material multiplicity benefit from lower rungs?**
**No.** The fixed two-mode subspace holds at `N = 2` on every iteration of every
mesh in both states; `ω₂ > ω₁` throughout; no NaN/Inf. Recorded asymmetry: at
160×20 the lower rungs shrink the relative gap by a factor of 3.2 (0.0265 →
0.0083), moving the pair *towards* coalescence. Not a failure, but a reason for
care at that mesh rather than a benefit.

**28. Does any mesh show material volume/feasibility benefit?** **No.** Both
states satisfy the constraint to ≈ 1e-6; the S→F feasibility change is ≤ 6.3e-7
everywhere, against a 1e-5 bar.

**29. Does any mesh show material ω₁ benefit requiring lower rungs?** **Yes —
160×20, and only 160×20** (+0.61 % relative, crossing from below production to
above it).

**30. Does the lower ladder create or worsen failure risk?** **Yes.** 320×40's
`CAP_HIT` was *caused* by descending: amplitude falls with the move limit while
`tol(NE)` does not, so each rung pushes the mesh deeper into the region where
Branch A's amplitude clause cannot fire. At the terminal rung, amplitude ≈ 0.049
× tol blocked A while `med₂₀cosθ ≈ −0.885` blocked B, and `E` was false on all
1 248 iterations.

**31. Is the C320 `CAP_HIT` avoided by the single-stage counterfactual?**
**Yes, necessarily.** S terminates at iteration 274, so there is no terminal rung
to fail to exhaust. It avoids 1 326 outer iterations, 71 466 inner MMA
iterations and 33 585 s of wall time that bought −0.68 % `M_nd`.

**32. Is the exact four-rung ladder a Class-C reconstruction choice?** **Yes.**
`schema.m` classifies `move.policy`, `move.initial`, `move.levels` and every
continuation parameter as **class C** — "under-specified reconstruction choice".
`olh.move.limit` states it directly: the only bound Du & Olhoff (2007) place on
`Δρ` is the box (25f); the strings "move limit", "trust region" and "step size"
appear nowhere in the paper or in Olhoff & Du (2014); "every functional form —
the contraction rate, the ladder levels, the transition criteria and all numeric
values" is pure reconstruction. So changing the ladder would be **removing an
implementation heuristic, not changing a published scientific requirement**. That
is a licence to *consider* the change, not evidence that the change is right.

**33. Is `FOUR_RUNG_LADDER_NECESSARY` supported?** **Not as the brief defines
it.** H1 requires material benefit *"that justifies cost/risk"*. At 320×40 the
lower rungs deliver no material benefit while costing ×4.84 the outer iterations
of S, 97.7 % of the run's wall time, and a `CAP_HIT`. The cost/risk clause fails.
**Disclosed tension:** my own preregistered counting rule ("material on ≥ 2 of 3
⇒ NECESSARY") *would* have returned NECESSARY, since 160×20 and 400×50 are both
material. I am not applying it mechanically, because the brief's H1/H2 wording is
the authority for what the labels mean and H1's cost clause is explicitly
unsatisfied. Both readings agree on the practical consequence — do not remove the
ladder — so nothing is being shopped for; the difference is descriptive accuracy.

**34. Is `FOUR_RUNG_LADDER_PARTIALLY_USEFUL` supported?** **Yes.** The brief's H2
is *"lower levels matter on at least one mesh but not others, so simple removal
is not yet justified"* — which is the evidence sentence for sentence: material at
160×20 (both `M_nd` and ω₁) and 400×50 (`M_nd`, marginal), immaterial and harmful
at 320×40.

**35. Is `FOUR_RUNG_LADDER_NOT_JUSTIFIED` supported?** **No.** It requires
material benefit on **zero** meshes. Two meshes show material benefit, and at
160×20 removal would produce a design worse than production in ω₁.

**36. Is a single-stage policy now justified for preregistration?** **No.** Of
the seven Phase-19 conditions, three fail: condition 2 (S retains essentially all
material benefit over production — fails at 160×20, where S is *worse* than
production in ω₁), condition 3 (lower-rung gains below the materiality bars —
fails at 160×20 and 400×50), and condition 5 (160×20 reveals no blocking
lower-rung benefit — it reveals exactly that). Conditions 1, 4, 6 and 7 hold.

**37. Was production changed?** **No.** `PRODUCTION_CONTROLLER_NOT_CHANGED`. The
`+impl/` tree hash is byte-identical to the task start.

**38. Was A/B changed?** **No.** Not `A`, `B`, `W`, `P`, `W_np`, `tol`, the
union, the persistence, or the reset semantics.

**39. Was any Branch C created?** **No.**

**40. Were any scientific thresholds tuned?** **No.** The materiality bars were
frozen in `PREREGISTRATION.md` §6 before the decomposition was computed, and each
is anchored an order of magnitude tighter than a bar the project had already
committed to (20 % relative `M_nd` → 2 %; 1 % relative ω₁ → 0.10 %) — deliberately
generous to the ladder. §8 of that file discloses which downstream quantities
were already visible from the previous study, because this audit is not blind to
its own headline.

**41. Is all new audit evidence hash-valid?** **Yes.** `FINAL_SHA256.txt` covers
every deliverable, figure and script; `DATA_MANIFEST.json` records each with its
digest; `EVIDENCE.json` declares the five raw trajectories this audit depends on,
by hash, in the durable evidence root, and the finalization gate passes.

**42. Is the nine-mesh performance campaign authorized yet?** **No** —
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`. The controller remains only partially
validated, its terminal-admission defect at 320×40 is unresolved, and this audit
finds that the defect mechanism **strengthens** with refinement while the
lower-rung benefit **weakens** — precisely the wrong combination for a campaign
whose five new meshes are all finer than 400×50.

---

## What the evidence actually says

**Stage 1 does almost all the work, everywhere.** Rung 1 delivers −86.9 %, −86.9 %
and −84.3 % of `M_nd` at 160/320/400. Rungs 2–4 together deliver −2.5 %, −0.7 %
and −2.1 %.

**Within the lower ladder, only rung 2 ever matters.** It alone gives −1.90 %
`M_nd` and **+0.495 %** ω₁ at 160×20, and −1.43 % `M_nd` at 400×50. **Rungs 3 and
4 deliver ≤ 0.44 % `M_nd` and ≤ 0.094 % ω₁ on every mesh — below every
preregistered bar, everywhere** — while rung 4 at 320×40 consumed 91.5 % of that
run's inner work and ended in `CAP_HIT`.

That observation is recorded, not acted on: Phase 18 forbids proposing a new
ladder architecture here, and a three-mesh sample is not a basis for one.

**Why the meshes differ.** The lower rungs pay off exactly where the
`move = 0.04` stage terminates while still *bound-limited*. At S, `‖Δρ‖₂/tol` is
**12.59** at 160×20 (with `max|Δρ|` at 99.96 % of the move limit and 7 % of
elements bound-active) but **1.03** and **0.64** at 320×40 and 400×50, with zero
bound activity. The coarse mesh is oscillating at the resolution of its own move
limit; a smaller move is what lets it settle. This mechanism predicts the
coarse/fine split rather than merely describing it — and it predicts that
lower-rung benefit shrinks further as meshes refine.

---

## Verdicts

> ## `FOUR_RUNG_LADDER_PARTIALLY_USEFUL`
> ## `RETAIN_MOVE_LADDER_PENDING_REDESIGN`
> ## `PRODUCTION_CONTROLLER_NOT_CHANGED`
> ## `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

The ladder cannot be removed: 160×20 would regress below production in ω₁. It
cannot be kept unchanged: 320×40 cannot terminate under it. `RETAIN_MOVE_LADDER_PENDING_REDESIGN`
is the only verdict consistent with both facts.

## Supporting verdicts

```
MIDTASK_COMMIT_NONINTERFERENCE_VERIFIED
CONTROLLER_DEFINITION_RECOVERY_PASS        (recomputed element-wise, all three meshes)
RETENTION_ENFORCEMENT_ADDED                (fail-closed gate + 12 tests; legacy history unchanged)
ZERO_SCIENTIFIC_RUNS_EXECUTED
```
