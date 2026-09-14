# REPORT — withheld 240×30 two-branch mechanism test

**Does a classification frozen before the 240×30 run correctly identify the end
of useful fixed-move topology evolution, with no mesh-specific retuning?**

**Yes.** The frozen union fired at iteration **187** (Branch B) and left
**−0.71 %** useful topology evolution remaining — the best of all four meshes —
over a 1013-iteration confirmation tail in which M_nd never improved by more than
**0.03 %**. All six preregistered criteria pass.

| | |
|---|---|
| HEAD at task start | `7154d8201e9defb06d0d758da866c3769c07179a` (branch `benchmark-methodology-r2`) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` — **unchanged** |
| MATLAB / threads | R2025b Update 1 / `maxNumCompThreads(1)` |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256 `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd`, frozen **2026-09-08T12:04:53Z**, before the run and before any 240×30 numeric content was opened |
| Scientific runs | **exactly one** — 240×30, fixed move 0.04 |
| Solver copies | **none** |
| Production source changed | **none** |

Detail: [`TWO_BRANCH_ANALYSIS.md`](TWO_BRANCH_ANALYSIS.md),
[`CROSS_MESH_ANALYSIS.md`](CROSS_MESH_ANALYSIS.md),
[`PROVENANCE.md`](PROVENANCE.md), [`DATA_INVENTORY.md`](DATA_INVENTORY.md),
`METRICS.json`, `figures/F1`–`F12`.

---

## Headline

**The union works, at all four meshes, with one frozen rule:**

| | 160×20 | **240×30 (withheld)** | 320×40 | 400×50 |
|---|---|---|---|---|
| event | **83 (A)** | **187 (B)** | **255 (A)** | **369 (B)** |
| `remUseful` | 2.02 % | **−0.71 %** | −0.58 % | 1.11 % |
| `postRelImp` | 14.19 % | **0.03 %** | 0.16 % | 6.87 % |

**And it exposes one clean law.** β-stall — which is what production actually
descends on — precedes true exhaustion everywhere, and the gap grows strictly
with refinement:

| mesh | β-stall | exhaustion | **gap** |
|---|---|---|---|
| 160×20 | 79 | 83 | **+4** |
| **240×30** | **92** | **187** | **+95** |
| 320×40 | 130 | 255 | **+125** |
| 400×50 | 138 | 369 | **+231** |

The withheld mesh lands exactly where the sequence requires, from a rule frozen
before it was run.

**Two things went against me, and both are reported rather than buried.** My
preregistered prediction of *Branch A at ≈159* was **wrong** — Branch B fired, at
187. And **Branch A never fired at all**, despite 240×30 terminating in a
cancelling regime, because that cycle is *low-amplitude* (terminal `‖Δρ‖₂` =
0.83 × tol) and Branch A's own amplitude clause excludes it.

---

## The thirty-seven required answers

**1. What HEAD/config produced the run?** HEAD
`7154d8201e9defb06d0d758da866c3769c07179a`, `+impl/` tree `c1455374d5f8e256…`,
MATLAB R2025b Update 1, one thread, preset
`duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4`, config hash
`3a4653bd2c874c7b014d0fc330ccb5aa0378a5238031ac9e946b451270fd2bb6`, `NE = 7200`.

**2. Did all provenance/evidence gates pass?** **Yes.** currentness `CURRENT`;
source integrity 74/74; dispatch `ok=1`, 32 symbols, 0 blockers/warnings;
published MMA wins; sensitivity filter wins; no forbidden Olhoff path; **prior
160/320/400 evidence 157/157 hash-valid**; test suite **4/4, 0 failures**.
Not `WITHHELD240_PROVENANCE_FAIL`. Single-factor gate: only the four declared
overrides differ from production.

**3. Was exactly one scientific optimization run?** **Yes.** No 160×20, no
320×40, no 400×50, no second arm, no campaign.

**4. Was 240×30 genuinely withheld from threshold selection?** **Yes**, and the
full disclosure is in `PROVENANCE.md` §3. Before the freeze the only 240×30
information observed was *the existence of filenames* from a repo-wide `find`,
and the resolved production `stop.tolerance` = 0.075, which is a deterministic
function of `NE`. **No 240×30 numeric result was opened.** Every threshold,
window and persistence value came from 160/320/400, and the rule's training
behaviour — including its known 160×20 weakness — was tabulated in
`PREREGISTRATION.md` §11 before the run. This is a *withheld mechanism mesh*, not
a statistically independent sample; no such claim is made.

**5. What exact Branch-A definition was frozen?** Fires at the first `k` such
that for 20 consecutive iterations: `median₂₀ cosθ < 0` **and**
`median₂₀ net_path < 0.5` **and** `‖Δρ‖₂ ≥ tol(NE)`, with
`tol(NE) = 0.05·√(NE/3200)`, medians over `W = 20`, `net_path` over `W_np = 10`.

**6. What exact Branch-B definition was frozen?** Fires at the first `k` such
that for 20 consecutive iterations: `‖Δρ‖₂ < tol(NE)` **and**
`median₂₀ cosθ > 0`. Equivalently in mesh-normalized form,
`RMS(Δρ) < 8.838835e-04`, a **mesh-independent** constant — verified identical to
the `tol(NE)` form at every iteration on all training meshes. No NE exponent was
invented; the `√NE` scaling is inherited from the existing `meshScaled` rule.

**7. How is Branch B related to native stopping?** **Branch B is the inherited
native design-change stop criterion plus a coherence guard plus persistence.**
This was declared in `PREREGISTRATION.md` §6 before the run and is not presented
as a novel signal. Both additions demonstrably change the answer: the coherence
guard *blocks* a wrong Branch-B firing at 320×40 (median `cosθ` dips to −0.024
inside the window from native stop 216), and persistence delays 240×30 from the
native stop at **147** to **187**. At 400×50 Branch B coincides exactly with
native stopping (369).

**8. Which branch fired first at 240×30?** **Branch B.**

**9. At what iteration?** **187.**

**10. Did both branches fire?** **No.** Branch A never fired.

**11. Did neither fire?** No — Branch B fired.

**12. What was M_nd at first exhaustion?** **12.8388 %**. (M_nd's true minimum is
**12.7903 at k = 159**, 28 iterations earlier.)

**13. What was ω₁?** **167.0571** at the event; final 167.0134; run maximum
167.2883 at k = 107.

**14. What was cos(θ)?** Instantaneous **−0.604**; **median₂₀ = +0.217**, which
is the quantity the predicate uses and which satisfied Branch B's coherence
requirement for the full 20-iteration window.

**15. What was net/path?** Instantaneous **0.426**; median₂₀ **0.847**.

**16. What was normalized amplitude?** `‖Δρ‖₂ = 0.05067` against `tol = 0.075`,
i.e. **0.676 × tol**; `RMS(Δρ) = 5.971e-4` (< 8.8388e-4);
`max|Δρ|/move = 0.4232`.

**17. What fraction was bound-active?** **0.00000** — not one element of 7 200 at
the move bound, at the event or anywhere in the run.

**18. When would native stop have fired?** **147.**

**19. When would β stall have fired?** **92.** Since the fixed-move and
production arms are bitwise identical until production first descends, and
production descends on β-stall, **production's first descent at 240×30 is 92**
(inferred, not run — consistent with the archived scalar record of 104 outer
iterations and the 12-iteration descent-to-stop gap seen at the other meshes).

**20. How much useful topology evolution remained after exhaustion?**
**`remUseful` = −0.711 %** — the *best* of the four meshes. On the L1 density
basis and every other measure the design is finished.

**21. Did M_nd improve materially afterward?** **No.** `postRelImp = 0.029 %`.
M_nd went 12.8388 at the event to a post-event best of 12.8351 and then **worsened**
to 13.4507 at the cap.

**22. Did ω₁ improve materially afterward?** **No.** Best post-event gain
**+0.0008 %**; ω₁ drifted down from 167.0571 to 167.0134.

**23. Was the event a false positive?** **No.** `boundFrac = 0.00000`, the
predicate held 410 consecutive iterations after firing, and nothing useful
happened in the 1013-iteration tail. Phase 11 failure modes 1–4 and 6–8 all
checked negative (`TWO_BRANCH_ANALYSIS.md` §7).

**24. Was any mature state missed?** Not at 240×30 — the union fired. **But
Branch A missed one**: the cancellation signature holds for 492 iterations from
k = 709 and Branch A never fires, because the cycle is low-amplitude. The union
was carried by Branch B alone here.

**25. Does 240×30 resemble the 160, 320, or 400 regime?** **None of them — it is
a hybrid.** Its *endpoint* is cancelling like 160×20 and 320×40
(`cosθ = −0.917`, `net/path = 0.202`); its *classification* is Branch B like
400×50; and its *pathway* is new: **converge coherently, then re-excite into a
low-amplitude cycle.** No training mesh did that.

**26. Does Branch A generalize?** **Only partially — 2 of 4 meshes** (160×20,
320×40). It is blind to low-amplitude cancellation, and two of the three
cancelling meshes end *below* their own amplitude threshold (terminal
`‖Δρ‖₂/tol` = 0.83 at 240×30 and 0.98 at 320×40). 320×40 was caught only because
its cancellation began while amplitude was still marginally above `tol`.

**27. Does Branch B generalize?** **Only partially — 2 of 4 meshes** (240×30,
400×50). It cannot fire while amplitude stays above `tol`, which is 160×20's
permanent state (6.92 × tol).

**28. Does the A ∪ B union generalize?** **Yes — 4 of 4**, with
`remUseful ∈ [−0.71 %, +2.02 %]` everywhere. Each component covers exactly half
and is the only one that fires on its two meshes; they are genuinely
complementary, not one carrying the other.

**29. Does the union require mesh-specific tuning?** **No.** One rule, one set of
constants (`W = 20`, `P = 20`, `net/path < 0.5`, `cosθ` sign,
`RMS < 8.838835e-04`), all frozen before the withheld run, no per-mesh
parameters, no invented NE exponent.

**30. Did a third dynamical regime appear?** **No third *endpoint*** — 240×30
terminates cancelling, inside the existing taxonomy, so P6 passes. **But a third
*pathway* did appear** (converge-then-re-excite), which no training mesh showed.
Reported, not explained away.

**31. Can one two-branch controller now be preregistered without looking at its
future causal results?** **Yes.** The rule is fully specified, has fired
correctly at 4/4 meshes including a genuinely withheld one, uses only quantities
already computed per iteration, and its causal experiment is well defined
(apply as the move-transition trigger at 160×20/320×40/400×50, compare against
production) with success criteria statable in advance.

**32. If not, is another mechanism run justified or should the ladder be
reconsidered?** Not applicable — the hypothesis was confirmed. The ladder
architecture is not implicated by this result.

**33. Does anything justify projection?** **No.** The finding concerns *when* the
step is reduced. Projection is also under the scope lock.

**34. Does anything justify changing `R = 0.06·b`?** **No.** Nothing here bears
on filter radius; also under the scope lock.

**35. Did production source remain unchanged?** **Yes.** Nothing under `+impl/`
was written; tree hash re-verified unchanged. No preset, no schema field, no
solver copy. All scope-locked fields asserted before the solve.

**36. Is all new evidence durable and hash-valid?** **Yes.**
`runs/runD_240x30.mat` (52 MB, the full 7200 × 1200 trajectory plus telemetry and
resolved config) is hashed in `FINAL_SHA256.txt` and listed in
`DATA_MANIFEST.json`; the manifest was re-verified after cleanup. No scratch
artifact is cited. *Pre-existing debt, unchanged:* `move_transition/runs/arm{P,U}_*.mat`
remain gitignored and absent from that study's manifest — flagged, out of scope.

**37. Is the nine-mesh performance campaign still blocked?** **Yes —
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`**, per Phase 23, and independently
because no controller exists yet.

---

## Verdicts

### Scientific

> ### `TWO_BRANCH_MATURITY_HYPOTHESIS_CONFIRMED`

All six preregistered criteria pass at the withheld mesh, with no retuning:
P1 (Branch B at 187), P2 (`remUseful` −0.711 % ≤ 5 %), P3 (`postRelImp` 0.029 %
≤ 25 %), P4 (tail 1013 ≥ 400), P5 (`boundFrac` 0.00000), P6 (no third endpoint).
The union covers 4/4 meshes with `remUseful ∈ [−0.71 %, +2.02 %]`, and the
β-stall-to-exhaustion gap is strictly monotone in `NE` with the withheld mesh
landing where the sequence requires.

**Carried forward as explicit qualifications, not footnotes:**

1. **My branch prediction was wrong** (predicted A at ≈159; got B at 187).
2. **Branch A is blind to low-amplitude cancellation** — it never fired at
   240×30 despite a cancelling endpoint, and 320×40 was caught only by an
   ordering accident. The union's success here rested on Branch B alone.
3. **A new pathway exists** (converge → re-excite) that no training mesh showed.
4. **160×20 remains the weak case**: 14.19 % relative M_nd improvement still
   available at the event — disclosed before the test, inside the preregistered
   bound, and the state a controller must be judged against.

### Next step

> ### `TWO_BRANCH_CONTROLLER_PREREGISTRATION_JUSTIFIED`

Phase-16 conditions: 240×30 correctly classified without retuning ✔; event at the
end of useful evolution ✔ (best of four); no third endpoint ✔; union covers
160/240/320/400 ✔; no known false-positive maturity state — the 160×20 case is
the only candidate and sits inside the preregistered bound ✔ (qualified);
semantically expressible ✔; thresholds and persistence already frozen and
survived out-of-sample ✔.

This authorizes **one** controller preregistration and causal validation — and
nothing more. The four qualifications above are the specific things that
validation must be designed to expose, particularly 160×20.

### Production and performance status

> `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`
> `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

Both unchanged by this task, as Phase 23 requires.
