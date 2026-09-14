# REPORT — does the three-rung ladder `[0.04, 0.02, 0.01]` suffice?

**Zero-scientific-run offline audit.** No optimization was executed. Every number
comes from three causal-controller trajectories that already existed.

> **Headline.** The exact `[0.04, 0.02, 0.01]` policy is a valid, exactly-derived,
> honestly-terminating architecture on all three primary meshes. It clears every
> preregistered acceptance gate at S3, it removes the 320×40 `CAP_HIT` entirely,
> and the omitted `move = 0.005` rung is **below every frozen materiality bar on
> every mesh** — including the 160×20 objective bar that blocked the two-rung
> policy, where the residual falls from **+0.1139 %** to **+0.0202 %** against a
> 0.10 % threshold.
>
> It is **not** fully supported, for one reason that was preregistered before any
> of these numbers were computed: the material two-rung residual is not
> *captured* by rung 3, it is *split*. Rung 3 is worth **+0.0937 %** and rung 4
> **+0.0202 %** — both individually below the 0.10 % bar whose sum, +0.1139 %, is
> above it. No retained rung below `move = 0.02` does material work by the
> project's own standard.

---

## The three required verdicts

```
THREE_RUNG_COUNTERFACTUAL_EXACT
THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED
MORE_THREE_RUNG_EVIDENCE_REQUIRED
```

All fifteen of the brief's Phase-21 conditions hold. The architecture verdict is
nevertheless capped at `PARTIALLY_SUPPORTED` by the `THRESHOLD_SPLITTING` guard
frozen in `PREREGISTRATION.md` §11. Both facts are reported; §5 below shows
exactly how the verdict is derived, so the reader can see what drives it.

---

## 1. The forty-eight questions

**1. What branch/HEAD was audited?**
`benchmark-methodology-r2`, HEAD `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`,
re-read at task start rather than inherited from a previous brief. Working tree
dirty on 22 paths, none under `+impl/`.

**2. Was `+impl` unchanged?**
Yes. Tree SHA-256 `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`
(75 files) at start and end — byte-identical to the values recorded by
`move_ladder_necessity` and `two_rung_architecture`. Currentness `CURRENT`,
source integrity `PASS` (0 mismatched, 0 missing, 0 extra).

**3. Were ZERO scientific optimization runs executed?**
Yes. No `olhoffSolve` call of any kind. MATLAB ran only the provenance gate, the
static config audit, the finalization gate and the test suite — all of which read
files, hash them, or resolve configurations. All numerical work is Python over
pre-existing `.mat` and `.csv`.

**4. What exact frozen A/B definitions were used?**
Those of `+impl/architecture/+olh/+move/exhaustion.m`, whose provenance line names
`two_branch_maturity_240/PREREGISTRATION.md` SHA-256 `62748225…` (verified live):

```
tol(NE) = 0.05*sqrt(NE/3200);   W = 20;  P = 20;  W_np = 10
A(k) = med20 cos < 0  AND  med20 net < 0.5  AND  amp >= tol
B(k) = amp < tol      AND  med20 cos > 0
E    = A OR B,  declared at the first k where either has held 20 consecutive iterations
reset: on descent stageStart <- the new stage's first iteration, counters zeroed,
       every step in every window stage-local (only the net-path anchor is pre-stage)
```

An independent Python re-implementation — **code-identical** to the two-rung
study's module, verified by diff — reproduced the controller's own
`exA`/`exB`/`exE`/`nA`/`nB` trace **element-wise in all 12 stages across the three
meshes**, and every recorded declaration exactly. Nothing was tuned.

**5. Was the new audit preregistered before complete S3 extraction?**
Yes. `PREREGISTRATION.md` SHA-256
`12c4bb960eeb6169521ea4b2e01b084a7bb3983d5387e94121e1729c2d71075c`, frozen
`2026-09-09T16:43:33Z`, before any S3 `M_nd`, `ω`, gap, gray, mid, volume, bound
fraction or density value was read. Its §16 discloses exactly what *was* already
visible — including that S3's `M_nd` and `ω₁` already sit in
`move_ladder_necessity/METRICS.json` as `rungs[3].Mnd_to` / `omega1_to`, which
were not opened before freezing, and that the per-rung cost figures were already
known (which is why §12 declines to invent a cost threshold). The materiality
thresholds, the acceptance gates and the relative-change denominator convention
are inherited **verbatim**; the objective bar remains **0.10 %**.

**6. Are S1 events reproduced exactly?**
Yes, by offline replay from the raw trajectory, not by reading the recorded log:
**102 (Branch A) / 274 (Branch A) / 388 (Branch B)**. Branch identities match the
prior audits on all three meshes.

**7. Are S2 events reproduced exactly?**
Yes: **141 (A) / 313 (B) / 427 (B)**, again with matching branch identities.

**8. What are the S3 event indices?**
**180 (160×20) / 352 (320×40) / 466 (400×50).**

**9. Which branch fires at each S3?**
**Branch B on all three meshes** — amplitude convergence with a positive
coherence median. `‖Δρ‖₂/tol` at S3 is 0.104 / 0.033 / 0.063.

**10. What is each stage-3 start iteration?**
**142 / 314 / 428** — one iteration after the stage-2 declaration in each case.

**11. What is S3 − stageStart for each mesh?**
**38 on all three meshes** — the earliest arithmetically possible offset (see Q35).

**12. Is S3 an exact counterfactual endpoint?**
Yes, on all three meshes. All ten preregistered validity checks pass, and the
static configuration audit shows that the one site whose value depends on the
*identity* of the final rung never executes. **`THREE_RUNG_COUNTERFACTUAL_EXACT`.**

**13. Does the proposed `[0.04,0.02,0.01]` trajectory first diverge from the
recorded four-rung trajectory only after S3?**
Yes — it diverges *at* `kE(3)`, and nowhere earlier. The two length-dependent
predicates (`stage < numel(levels)` in `olh.move.limit`, `stage >= numel(levels)`
in `olhoffSolve`) do take different values during stage 3, but **both are ANDed
with `ex.declared`, which is false at every iteration strictly between two
declarations**. So both ladders produce identical behaviour until the stage-3
declaration, where the four-rung run descends to 0.005 and the three-rung run
sets `convOuter = true`. The full derivation, with the complete dependence
inventory, is in `COUNTERFACTUAL_VALIDITY.md` §§1–2.

Critically — and this is why the brief was right to demand the check rather than
an inherited argument — `olhoffSolve.m:485` *does* depend on the final rung's
identity (`~any(moveLevels(stage+1:end) > epsRMS)`). It is inert here for **two
independent reasons**: `anyStopGuard = false` (neither stop guard is enabled) and
`exhaustStop = true`. Had either not held, the counterfactual would not have been
exact.

**14. What are P/S1/S2/S3/F `ω₁` values?**

| mesh | P | S1 | S2 | **S3** | F |
|---|---|---|---|---|---|
| 160×20 | 169.4952 | **168.9804** | 169.8175 | **169.9766** | 170.0110 |
| 320×40 | 165.9508 | 166.4216 | 166.4163 | **166.4273** | 166.4189 |
| 400×50 | 162.8826 | 166.4176 | 166.4355 | **166.4427** | 166.4562 |

**15. What are P/S1/S2/S3/F `M_nd` values?**

| mesh | P | S1 | S2 | **S3** | F |
|---|---|---|---|---|---|
| 160×20 | 13.4025 | 13.0364 | 12.7884 | **12.7561** | 12.7041 |
| 320×40 | 23.3596 | 13.0121 | 12.9797 | **12.9401** | 12.9233 (CAP_HIT) |
| 400×50 | 32.3283 | 15.6649 | 15.4413 | **15.3732** | 15.3311 |

**16. How much `ω₁` does rung 2 buy at 160×20?**
**+0.8372 absolute, +0.49541 % relative** — nearly 5× the 0.10 % bar, and the only
material lower rung anywhere in this audit. It is 81.2 % of the whole
lower-ladder `ω₁` gain, and it is what turns S1's 0.5148 deficit against
production into a 0.3223 surplus.

**17. How much `ω₁` does rung 3 buy at 160×20?**
**+0.1591 absolute, +0.09367 % relative** — 15.4 % of the lower-ladder gain, and
**below the 0.10 % materiality bar**. Rung 3 is not material by the project's own
standard.

**18. How much `ω₁` does rung 4 buy at 160×20?**
**+0.0344 absolute, +0.02025 % relative** — 3.3 % of the lower-ladder gain, five
times below the bar. Immaterial.

**19. Is `S3 → F` `ω₁` below the frozen 0.10 % materiality threshold?**
**Yes, on every mesh**, using the inherited denominator convention
`100·(ω₁(F) − ω₁(S3))/ω₁(S3)`:

| mesh | residual | bar | verdict |
|---|---|---|---|
| 160×20 | **+0.02025 %** | 0.10 % | below |
| 320×40 | **−0.00504 %** (rung 4 makes `ω₁` *worse*) | 0.10 % | below |
| 400×50 | **+0.00815 %** | 0.10 % | below |

**20. Does S3 resolve the exact failure that blocked the two-rung policy?**
**Numerically yes; structurally, only by subdivision.** The two-rung policy failed
because `S2 → F` at 160×20 was +0.11393 %, above the bar. `S3 → F` is +0.02025 %,
comfortably below it, so the stated gate is met. But the block did not shrink
because rung 3 captured a discrete effect — it shrank because it was cut in two,
and **neither piece is material on its own** (+0.09367 % and +0.02025 %). This is
the `THRESHOLD_SPLITTING` case that `PREREGISTRATION.md` §11 was written to catch
in advance, and it caps the verdict. See §4 below.

**21. Is S3 scientifically acceptable relative to production at 160×20?**
Yes, and better than both earlier endpoints. `M_nd` 12.7561 vs production's
13.4025 (**−4.82 %**); `ω₁` 169.9766 vs 169.4952 (**+0.284 %**, no regression).
Every inherited acceptance gate passes at S3: A1 (`M_nd ≤ 14.7427` ✅,
`ω₁ ≥ 167.8003` ✅), A4 ✅, A5 ✅, A6 ✅, A7 ✅, A8 ✅, and the no-regression gate
**A9** (`ω₁(S3) ≥ ω₁(P)`) ✅ — which **fails at S1** and passes at S2 and S3.

**22. What `M_nd` benefit remains in rung 4 at 160×20?**
−0.0520 absolute, **−0.4078 % relative**, against a 2 % bar. Immaterial.

**23. Does 320×40 satisfy `E` on `move = 0.01`?**
**Yes — Branch B at iteration 352**, window [333, 352], `med₂₀ cosθ = +0.594`,
`‖Δρ‖₂ = 0.00327` against `tol = 0.1`, full 20-iteration persistence.

**24. Would it terminate honestly at S3?**
Yes. Stage 3 is the last level under `[0.04, 0.02, 0.01]`, so `atLastLevel` is
true and `convOuter = ex.declared && atLastLevel` fires at 352 — the same frozen
concept that governs the descents, with no second terminal rule.

**25. Does that avoid the `move = 0.005` `CAP_HIT` pathology?**
**Yes, entirely.** The four-rung run reaches `CAP_HIT @1600` because its
`move = 0.005` stage never satisfies `E`: at its first evaluable iteration (372)
`E` is already false, and it stays that way for 1 248 iterations. The three-rung
policy never enters that stage. `F` at 320×40 remains labelled **CAP_HIT**
everywhere in this study and is never treated as a converged reference.

**26. How much `ω₁` changes at 320×40 after S3?**
**−0.0084 absolute, −0.00504 % relative.** Rung 4 there moves the maximized
objective in the **wrong direction**.

**27. How much `M_nd` changes after S3 [at 320×40]?**
−0.0168 absolute, −0.1297 % relative. Immaterial against the 2 % bar.

**28. How much inner MMA work is after S3 [at 320×40]?**
**70 034 inner MMA iterations — 91.5 % of that mesh's entire budget** — across
1 248 outer iterations, for the two immaterial changes above, ending in a cap.

**29. Does 400×50 satisfy `E` on `move = 0.01`?**
**Yes — Branch B at iteration 466**, window [447, 466], `‖Δρ‖₂/tol = 0.063`.

**30. How much `ω₁` benefit remains in rung 4 at 400×50?**
+0.0136 absolute, **+0.00815 % relative** — an order of magnitude below the bar.

**31. How much `M_nd` benefit remains [at 400×50]?**
−0.0422 absolute, **−0.2742 % relative**. Immaterial.

**32. Do any rung-4 topology changes exceed frozen materiality?**
**No, on any mesh.** Mean `|Δρ_e|` is 0.000424 / 0.001723 / 0.000208 against a
0.01 bar; gray and mid fractions move by ≤ 0.00078 against a 0.01 bar, and at
400×50 both are bit-identical between S3 and F.

**33. Does rung 4 provide any material multiplicity benefit?**
**No.** Subspace size is 2 at S3 and F on all three meshes; mode order never
changes; `ω₂ > ω₁` throughout; no NaN/Inf; zero non-converged inner solves. The
gap moves by at most 0.00077, in a direction that is not even consistent across
meshes, and gap magnitude alone is explicitly not a materiality criterion because
the objective is `ω₁`.

**34. Does rung 4 provide any material volume benefit?**
**No.** `|volume − 0.5|` changes by −2.36e-06 / −8.68e-07 / −3.19e-07 (all
improvements) against a 1e-5 bar, and every state is an order of magnitude or more
inside the 1e-4 gate.

**35. Is every lower-stage declaration at the earliest mathematically possible
persistence completion?**
**Yes — all 8 lower stages that fire, on all three meshes, declare at exactly
`stageStart + 38`**, the minimum possible given `W = 20` (first evaluable at
`stageStart + 19`) and `P = 20`. Stage 1 is the sole exception everywhere
(offsets 101 / 273 / 387). The ninth lower stage — 320×40's `move = 0.005` —
never declares at all.

**36. Is the `stageStart + 38` observation reproduced exactly?**
Yes, and extended: the first iteration at which the median is defined equals
`stageStart + 19` in all 12 stages, matching the theory exactly, and `E` is true
from that very iteration and unbroken to the declaration in all 8 firing lower
stages. Full table in `DECLARATION_TIMING_AUDIT.md` §2.

**37. Does this imply lower-stage `E` is already true when first evaluable?**
**Yes** — measured directly, in all 8 firing lower stages. The narrow supported
conclusion, and the only one drawn: *the lower-rung exhaustion detector is not
observing a newly developed dynamical transition within those stages; the
exhaustion condition is already satisfied once sufficient post-transition history
exists.* Operationally each terminating lower rung is exactly 39 outer iterations
long, a length set by the window/persistence arithmetic rather than by the design.

**38. Was that observation used only descriptively and not to change policy?**
**Yes.** `used_to_change_policy = false` in `METRICS.json`. It is explicitly *not*
claimed that 39 iterations are optimal, that the persistence window is
unnecessary, that lower stages should be a fixed dwell, that the controller should
descend immediately, or that history should be inherited across transitions — each
would need a separate test this task does not perform.
`DECLARATION_TIMING_AUDIT.md` §5 lists these non-claims explicitly.

**39. Is 240×30 unavailable for this comparison?**
**Yes — `UNAVAILABLE_FOR_THREE_RUNG_CAUSAL_COMPARISON`**, on three independent
grounds: zero files matching `*240x30*` exist under `analysis/OlhoffCurrent`;
`two_branch_maturity_240` has no `runs/` directory; and that arm was a
**fixed-move** arm that never ran a `0.02` stage, let alone a `0.01` stage. No S3
was inferred, no proxy substituted, nothing run.

**40. Was any A/B threshold changed?**
No. `exhaustion.m` was read, never written. `W = 20`, `P = 20`, `W_np = 10`,
`tol = 0.05·√(NE/3200)`, both predicates and the union are unchanged, and the
offline replay reproduces the in-loop trace element-wise as proof.

**41. Was persistence changed?**
No. `P = 20` throughout, with the stage-local reset semantics untouched.

**42. Was any new architecture tested?**
No. Exactly one architecture was audited: `[0.04, 0.02, 0.01]`. No
`[0.04, 0.01]`, no `[0.04, 0.02, 0.005]`, no adaptive move, no mesh-dependent
ladder, no fixed dwell, no inherited history, no Branch C.

**43. Was production changed?**
No. `move.levels` resolves to `[0.04 0.02 0.01 0.005]` and
`move.continuation.signal` to `boundVariable`, both verified in the static audit.
**`PRODUCTION_CONTROLLER_NOT_CHANGED`.**

**44. Is the three-rung architecture supported on all primary meshes?**
**Every mesh is three-rung-sufficient** — `E` fires on `move = 0.01`, S3 clears
every acceptance gate, and rung 4 is below every materiality bar. But the
architecture verdict is **`PARTIALLY_SUPPORTED`**, not `SUPPORTED`, because the
preregistered `THRESHOLD_SPLITTING` guard fires at 160×20: no retained rung below
`move = 0.02` is itself material.

**45. Is three-rung policy preregistration justified?**
**No — `MORE_THREE_RUNG_EVIDENCE_REQUIRED`.**
`THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED` requires the architecture verdict
`SUPPORTED` plus all fifteen Phase-21 conditions. The fifteen conditions hold; the
architecture verdict does not.

**46. Did finalization gate pass?**
**Yes — all five gates.** G1 declares raw evidence (`EVIDENCE.json`); G2 every
required declared artifact present and hash-valid; G3 `FINAL_SHA256.txt` present;
G4 it is self-verifying; G5 no `.mat` named by any manifest is absent.
`DATA_MANIFEST.json` and `FINAL_SHA256.txt` cover identical file sets.

**47. Is evidence hash-valid?**
Yes. 17/17 required artifacts present and hashed at the Phase-0 gate; all four
inherited preregistration digests verified live; `FINAL_SHA256.txt` independently
re-verified with `shasum -c`.

**48. Is the nine-mesh performance campaign still blocked?**
Yes. **`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`**, by Phase 24 unconditionally,
and independently because no policy is frozen to run it with.

---

## 2. What rung 4 costs, and what it buys

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| outer iterations | 39 | **1 248** | 39 |
| inner MMA iterations | 791 (15.6 % of total) | **70 034 (91.5 %)** | 1 453 (14.1 %) |
| `Δω₁` | +0.02025 % | **−0.00504 %** | +0.00815 % |
| `ΔM_nd` | −0.4078 % | −0.1297 % | −0.2742 % |
| mean \|Δρ_e\| | 0.000424 | 0.001723 | 0.000208 |
| terminal status | CONVERGED | **CAP_HIT** | CONVERGED |
| cost-dominated (≥2×, sub-material) | no | **yes (3.55×)** | no |
| **material on any bar** | **no** | **no** | **no** |

That half of the hypothesis holds cleanly and without qualification: **rung 4
earns nothing on any mesh, and on one mesh it costs 91.5 % of the budget, moves
the objective backwards, and never terminates.**

## 3. The declaration-timing result

All 8 firing lower stages declare at `stageStart + 38`, the arithmetic minimum,
with `E` already true at `stageStart + 19` and unbroken thereafter. Stage 1 is the
only stage on any mesh where the rule observes something develop (offsets
101/273/387, `E` false at the first evaluable iteration). Figures F11 and F12 show
this directly.

The supported reading is stated in Q37 and nowhere exceeded. It is context for
*why* every lower rung costs the same 39 iterations — not evidence for or against
the ladder, and not a licence to change anything.

## 4. Bound limitation — descriptive only (Phase 19)

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `‖Δρ‖₂/tol` at S1 | **12.59** | 1.03 | 0.64 |
| `‖Δρ‖₂/tol` at S2 | 2.19 | 0.108 | 0.144 |
| `‖Δρ‖₂/tol` at S3 | 0.104 | 0.033 | 0.063 |
| bound fraction at S1 | 0.0713 | 0.0 | 0.0 |
| bound fraction at S2 | 0.0069 | 0.0 | 0.0 |
| bound fraction at S3 | **0.0** | **0.0** | **0.0** |
| `max|Δρ|/move` at S1 → S2 → S3 | 1.000 → 0.997 → 0.155 | 0.672 → 0.039 → 0.023 | 0.143 → 0.060 → 0.027 |
| branch at S1 → S2 → S3 | A → A → B | A → B → B | B → B → B |

The pattern is consistent with the earlier observation and is stated as one.
160×20 leaves stage 1 still strongly step- and bound-limited, exits stage 2 on
Branch A with `max|Δρ|` still at 99.7 % of the move limit, and is the only mesh
where a lower rung buys anything material. By S3 the bound fraction is exactly
zero everywhere and every mesh has crossed to Branch B.

**Three points. No `NE` law, mesh scaling, move scaling or adaptive ladder is
fitted from this, and no architecture is proposed from it.**

## 5. Verdict derivation, step by step

Preregistered §14: a mesh is *three-rung-sufficient* when `E` fires on
`move = 0.01`, S3 clears every §10 gate, **and** rung 4 is below every §9
materiality bar.

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `E` on `move = 0.01` | ✅ B @180 | ✅ B @352 | ✅ B @466 |
| counterfactual exact + replay matches | ✅ | ✅ | ✅ |
| A1/A2/A3 `M_nd` bound at S3 | ✅ | ✅ | ✅ |
| A4 `ω₁ ≥ 0.99 × P` | ✅ | ✅ | ✅ |
| A5 volume ≤ 1e-4 | ✅ | ✅ | ✅ |
| A6 physics | ✅ | ✅ | ✅ |
| A7 cost multipliers | ✅ | ✅ | ✅ |
| A8 terminal persistence at last level | ✅ | ✅ | ✅ |
| A9 no `ω₁` regression (160×20) | ✅ | — | — |
| rung 4 immaterial on every bar | ✅ | ✅ | ✅ |
| **three-rung-sufficient** | **yes** | **yes** | **yes** |
| rung 3 itself material | **no** | no | no |

`REFUTED` requires `E` to fail somewhere, or rung 4 material on ≥ 2 meshes, or A4
to fail, or the 160×20 residual to remain ≥ 0.10 % → **none met**.

`SUPPORTED` requires all three sufficient **and** `THRESHOLD_SPLITTING` false
**and** A9 **and** an exact counterfactual. Three of four hold;
`THRESHOLD_SPLITTING` is **true** →  not `SUPPORTED`.

`PARTIALLY_SUPPORTED` requires every mesh to satisfy `E` on 0.01, an exact
counterfactual, and **exactly one** qualifying shortfall. Exactly one holds
(`THRESHOLD_SPLITTING` true while every mesh is otherwise sufficient) →
**`PARTIALLY_SUPPORTED`**.

Phase-21 conditions: **15 of 15 hold.** The guard is a separate, preregistered cap
on the architecture verdict, and it is the sole reason the verdict is not
`SUPPORTED`.

## 6. What the next task would need — and what this one deliberately does not do

The open question is not empirical. Every measurement here is unambiguous:
rung 4 is worthless on every mesh and actively harmful on one; rung 3 reduces the
160×20 residual below the bar; neither rung 3 nor rung 4 is individually material.

What is unresolved is a **judgement about the materiality criterion itself**: a
per-block relative bar can be met by subdividing a block, so it does not by itself
identify where a continuum of diminishing gains should be cut. Settling that
requires a criterion chosen before the answer is known — a cumulative or
absolute-`ω₁` formulation, say, or an explicit decision that +0.09 % at the
coarsest mesh is or is not worth a rung. That belongs in a future preregistration.

This task does **not** propose one, does not move the bar, does not test another
ladder, and does not repair the finding. `PREREGISTRATION.md` §17 forbids all of
it, and none of it was done.

## 7. Figures

| | |
|---|---|
| `F1_omega1_P_S1_S2_S3_F.png` | `ω₁`: P/S1/S2/S3/F, each mesh |
| `F2_Mnd_P_S1_S2_S3_F.png` | `M_nd`: P/S1/S2/S3/F, each mesh |
| `F3_rung_omega1.png` | rung-by-rung `ω₁` — the 160×20 panel shows both halves under the bar and their sum over it |
| `F4_rung_Mnd.png` | rung-by-rung `M_nd` — no lower rung material anywhere |
| `F5_rung_topology.png` | rung-by-rung mean \|Δρ_e\| against the 0.01 bar |
| `F6_rung_outer_work.png` | rung-by-rung outer iterations |
| `F7_rung_inner_work.png` | rung-by-rung inner MMA work |
| `F8_benefit_per_inner_work.png` | marginal `Δω₁` and `ΔM_nd` per 1000 inner MMA iterations |
| `F9_topology_S3_vs_F.png` | S3, F and their difference, each mesh |
| `F10_multiplicity_gap.png` | gap and subspace size with S1/S2/S3/F markers |
| `F11_declaration_offset.png` | declaration − stageStart, every stage, every mesh |
| `F12_first_evaluable_vs_declaration.png` | first evaluable window vs earliest possible vs actual declaration |
| `F13_C320_S3_vs_CAP_HIT.png` | 320×40: terminating at S3 versus the 1 248-iteration `move = 0.005` continuation |
| `F14_architecture_summary.png` | two-rung miss, three-rung endpoint, four-rung cost and failure |

---

## 8. Final summary

| | |
|---|---|
| branch | `benchmark-methodology-r2` |
| starting HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| final HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (no commit made) |
| dirty state, start | dirty, 22 paths |
| dirty state, end | dirty, 22 paths (same set; this study's files are all inside its own untracked directory) |
| `+impl/` hash, start | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| `+impl/` hash, end | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` — **unchanged** |
| **scientific runs** | **0** |
| tests | 6/6 PASS at end (`test_finalization_gate` failed at start only, self-referentially — see `PROVENANCE.md`) |
| gates | provenance `THREE_RUNG_EVIDENCE_GATE_PASS` (12/12) · frozen-rule replay element-wise in 12/12 stages · counterfactual validity 10/10 × 3 meshes · static config audit PASS · finalization gate **PASS** (G1–G5) |
| preregistration hash | `12c4bb960eeb6169521ea4b2e01b084a7bb3983d5387e94121e1729c2d71075c`, frozen `2026-09-09T16:43:33Z` |
| evidence manifest | `EVIDENCE.json` (required artifacts present + hash-valid) · `DATA_MANIFEST.json` · `FINAL_SHA256.txt`, self-verifying |
| finalization gate | **PASS** |

```
THREE_RUNG_COUNTERFACTUAL_EXACT

THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED

MORE_THREE_RUNG_EVIDENCE_REQUIRED

PRODUCTION_CONTROLLER_NOT_CHANGED

NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```
