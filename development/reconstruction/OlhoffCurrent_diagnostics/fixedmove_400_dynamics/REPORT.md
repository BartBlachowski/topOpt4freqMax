# REPORT — 400×50 fixed-move dynamics

**Does the coherent → cancelling transition observed at 160×20 and 320×40
generalize to 400×50?**

**No.** 400×50 never enters a cancelling regime. It converges instead — motion
amplitude decays ~19× while direction stays coherent at `cosθ = +0.997`. The
dynamical observables that separate mature from premature states at the two
coarser meshes are **blind** at the finest one.

| | |
|---|---|
| HEAD at task start | `7154d8201e9defb06d0d758da866c3769c07179a` (branch `benchmark-methodology-r2`) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` — **unchanged** |
| MATLAB / threads | R2025b Update 1 / `maxNumCompThreads(1)` |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256 `f6e84d8a65a3d8180508693f307a07ec9ae8ada51cf852a4668c087b1aeafcbc`, frozen **2026-09-08T10:02:30Z, before the run** |
| Scientific runs | **exactly one** — RUN C, 400×50 fixed move 0.04 |
| Solver copies | **none** |
| Production source changed | **none** |

Detail: [`DYNAMICAL_ANALYSIS.md`](DYNAMICAL_ANALYSIS.md),
[`CROSS_MESH_ANALYSIS.md`](CROSS_MESH_ANALYSIS.md),
[`PROVENANCE.md`](PROVENANCE.md), [`DATA_INVENTORY.md`](DATA_INVENTORY.md),
`METRICS.json`, `figures/F1`–`F12`.

---

## Headline

RUN C: **1200 outer, `CAP_HIT`**, ω₁ = 166.157147836, M_nd = 15.2243 %, native
stop would have fired at **369**, 6104 s.

**No period-2 onset exists.** Median `cosθ` never goes negative — its minimum
over all 1200 iterations is **+0.914**. Median `q2` never drops below 1
(minimum 1.918). The `COHERENT` label holds for 1181 of 1200 iterations.

**And this is not a cap artifact.** `max|Δρ|` decays 2.10e-2 → 1.12e-3 (a factor
19), `max|Δρ|/move` reaches **0.031**, and `‖Δρ‖₂` ends a factor of 10 *below*
the native tolerance. The design converged. A limit cycle needs sustained
amplitude; running longer would decay further, not oscillate.

**The three fixed-move arms end in different states:**

| mesh | terminal `max|Δρ|/move` | terminal `cosθ` | terminal `net/path` |
|---|---|---|---|
| 160×20 | **0.999** (bound-pinned) | **−0.951** | **0.090** |
| 320×40 | **0.697** | **−0.965** | **0.132** |
| **400×50** | **0.031** (converged) | **+0.997** | **0.992** |

**Complementary blind spots.** Cancellation works at 160/320, blind at 400.
Amplitude works at 320/400, blind at 160 (pinned at the bound forever). Neither
generalizes; they overlap only at 320×40, which is why the mechanism looked
general when only those two meshes existed.

**What survives, and is now measured at 400×50 for the first time:** production
descends while the design is still coherent, and the penalty grows steeply with
refinement — **1.6 / 9.1 / 17.3 M_nd points** at 160/320/400. At 400×50
production abandons a **53.4 % relative M_nd reduction** and **+2.17 % ω₁**.

---

## The thirty-five required answers

**1. What HEAD/config produced the run?** HEAD
`7154d8201e9defb06d0d758da866c3769c07179a`, `+impl/` tree `c1455374d5f8e256…`,
MATLAB R2025b Update 1, one thread, preset
`duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4`, RUN C config hash
`9a78ba9c9a577fcd69f9314472def7ca3a4b282edf958e66f6ced4eb684189ad`.

**2. Did all provenance gates pass?** **Yes.** currentness `CURRENT`; source
integrity 74/74; dispatch `ok=1`, 32 symbols, 0 blockers/warnings; published MMA
wins; sensitivity filter wins; no forbidden Olhoff path; prior dynamical-regime
evidence 33/33 valid; 400×50 production and 320×40 extended fixed-move
trajectories both valid and manifested; **all retained evidence 129/129
hash-valid**; test suite **4/4, 0 failures**. Not `FIXEDMOVE400_PROVENANCE_FAIL`.
Single-factor gate: only the four declared overrides differ from production.

**3. Was exactly one scientific optimization run?** **Yes** — RUN C only. No
160×20, no 320×40, no repeat of 400×50 production, no other mesh, no campaign.

**4. Was the 400×50 production/fixed-move common prefix verified?**
**Yes — bitwise.** Over `k = 1 … 137`: max |Δρ| = **0**; ω₁, ω₂, M_nd, volume and
β differences all exactly **0**. First differing iteration = **138**, exactly as
preregistered (production descends *at* 138, so the update there uses a different
move). The counterfactual is valid.

**5. At what iteration would native stopping have occurred?** **369**
(`‖Δρ‖₂ < 0.125` with the settled-move guard). The replay was validated before
the run against `fixedmove_320x40` (predicts 216, archive records 216) and
`runA_400x50` (predicts 139, run converged at 139).

**6. Did 400×50 fixed move remain coherent beyond production descent?**
**Yes — for the entire run.** `COHERENT` from iteration 20 to the cap, 1181 of
1200 iterations.

**7. Did it eventually enter persistent cancellation?** **No.**

**8. At what iteration?** None. Median `cosθ` never negative (min **+0.914**);
median `q2` never below 1 (min 1.918); longest run satisfying both criteria = **0**
(20 required).

**9. What was M_nd at onset?** No onset. For reference, M_nd reaches its minimum
**15.0485 % at k = 547**, from 32.318 % at the production descent.

**10. What was ω₁ at onset?** No onset. ω₁ peaks at **166.4216 at k = 394** and
declines slowly thereafter to 166.158.

**11. What were raw and unsaturated `cosθ` at onset?** No onset. Throughout the
run `boundFrac = 0.00000`, so **`cosθ_unsat ≡ cosθ_raw` exactly** at every
iteration; both sit at ≈ +0.99.

**12. What was net/path at onset?** No onset. `net/path` stays at **0.98–0.99**
for the whole run, including a last-100 median of 0.992.

**13. What fraction of elements was move-bound?** **Zero — 0.00000 at every one
of 1200 iterations.** Not one element of 20 000 ever touched the move bound.

**14. How much useful topology evolution remained at onset?** Not defined (no
onset). Useful evolution ends around **k ≈ 400–550**: M_nd bottoms at 15.049 % at
547 and then drifts *up*; ω₁ peaks at 394 and then declines.

**15. What happened to M_nd after onset?** N/A. After its k=547 minimum M_nd
**worsens slowly**: 15.049 → 15.054 (600) → 15.174 (1000) → 15.224 (1200).

**16. What happened to ω₁ after onset?** N/A. After its k=394 peak ω₁ **declines
slowly**: 166.422 → 166.278 (600) → 166.158 (1200), i.e. −0.16 %.

**17. Was the cancelling regime persistent?** There was no cancelling regime at
400×50. At 160×20 and 320×40 it was persistent (519 and 947 iterations
respectively, still deepening at their caps).

**18. Was it spatially coherent?** N/A at 400×50 — reversal *falls* from 0.425
(k=138) to 0.135 (k=1199), the opposite of 160×20 (0.36 → 0.82) and 320×40
(0.44 → 0.90). What remains is void elements jittering at ρ_min; there is no
coherent oscillation to map.

**19. Was it caused by saturation?** **No — and it cannot have been**, in either
direction: `boundFrac = 0` throughout, so there is no saturated population either
to create a false signal or to mask a real one. Raw and unsaturated diagnostics
are identical.

**20. How much M_nd did production abandon by descending at 138?**
**17.28 percentage points** — production's final 32.3283 % against the fixed-move
best of **15.0485 %** (k = 547). A **53.4 % relative reduction**. Against the
fixed-move value at the cap (15.2243 %) it is 17.10 points.

**21. How much ω₁ did production abandon?** **+2.17 %** — production's final
162.882616 against the fixed-move peak **166.4216** (k = 394). At the cap the
gain is still +2.01 %. Production loses on both grayness and frequency.

**22. Are the previously unsourced 16.16 % and +2.14 % claims reproduced,
contradicted, or still unsupported?** **Reproduced — exactly, and their
provenance is now explained.**

| claim | measured | at |
|---|---|---|
| mature fixed-move M_nd ≈ **16.16 %** | **16.1589 %** | **k = 369** |
| ω₁ improves ≈ **+2.14 %** | **+2.1379 %** | **k = 369** |

and **k = 369 is precisely where the inherited native stop rule fires** on this
arm. Both were therefore genuine measurements from a 400×50 fixed-move run that
was terminated natively and subsequently lost. The earlier suspicion that 16.16
was "exactly half of 32.33" is **disproved**: half of 32.3283 is 16.1642 against
a measured 16.1589 — an independent quantity that merely lands nearby.

**23. Does the coherent → cancelling mechanism hold at all three meshes?**
**No.** It holds at 160×20 (onset 81) and 320×40 (onset 253). It **fails at
400×50**, which converges without ever cancelling.

**24. Does onset coincide approximately with the end of useful work at all three?**
Where an onset exists, yes and strikingly so — 2.08 % and −0.58 % of topology
evolution remaining at 160×20 and 320×40. At 400×50 there is no onset to
coincide with anything, although useful work does end (k ≈ 400–550).

**25. Does raw `cosθ` generalize?** **No.** At 400×50 it reads +0.995 at the
premature state (k=138, M_nd 32.3) and +0.998 at the mature state (k=1000,
M_nd 15.2) — no separation at all.

**26. Does unsaturated `cosθ` generalize?** **No** — it is identical to the raw
value at 400×50 (`boundFrac = 0`), so it inherits the same blindness. It is also
the weaker variant at 160×20, where it loses the sign at the onset instant
(−0.833 → −0.072).

**27. Does net/path generalize?** **No.** 0.977 premature vs 0.990 mature at
400×50 — the wrong way round and negligible either way.

**28. Is one observable sufficient?** **No.** No single tested observable
separates mature from premature at all three meshes. Cancellation is blind at
400×50; amplitude is blind at 160×20; their natural unification
(`‖ρ_k − ρ_{k−W}‖`) was already refuted as blind by the
`topology_maturity_transition` study.

**29. Is a small logically related observable set required?** **Apparently yes.**
The three meshes end in two physically different states — a bound-pinned limit
cycle (160×20) and ordinary convergence (400×50) — and a controller must
recognise both. A disjunction of the two existing signals would fire at
**81 / 216 / 369**, each near the end of useful work. **Recorded as an
observation only** — not proposed, preregistered, threshold-ed or implemented.
Since each branch was individually refuted at one mesh, whether the disjunction
is one principled mechanism or two patches is exactly what the next task must
settle.

**30. Can one controller now be preregistered without outcome-driven tuning?**
**No.** Phase-18 conditions 1, 4 and 5 fail: 400×50 does not reach a cancelling
regime; the qualitative mechanism does not agree across the three meshes; and no
observable or small logically-related set has yet been shown to capture maturity
across meshes without mesh-specific fitting. Choosing the disjunction now would
mean selecting a two-branch rule *because* it happens to cover the meshes we have
looked at — precisely the outcome-driven construction the protocol forbids.

**31. Does anything justify projection?** **No.** The finding concerns when the
step is reduced, not the material or filtering model; projection is also under
the scope lock.

**32. Does anything justify changing `R = 0.06·b`?** **No.** Nothing here bears
on filter radius, and it is under the scope lock.

**33. Did production source remain unchanged?** **Yes.** Nothing under `+impl/`
was written; the tree hash was re-verified unchanged after the run. No preset, no
schema field, no solver copy. All scope-locked fields were asserted in code
before the solve.

**34. Is all new evidence durable and manifested?** **Yes.**
`runs/runC_400x50.mat` (147 MB, the full 20000 × 1200 trajectory plus telemetry
and resolved config) is hashed in `FINAL_SHA256.txt` and listed in
`DATA_MANIFEST.json`. `FINAL_SHA256.txt` was re-verified after all cleanup. No
scratch artifact is cited as evidence. *Pre-existing debt, unchanged:*
`move_transition/runs/arm{P,U}_*.mat` remain gitignored and absent from that
study's manifest — flagged, out of scope here.

**35. Is the final performance campaign still blocked?** **Yes —
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`**, per Phase 23, and independently
because no controller exists.

---

## Verdicts

### Mechanism

> ### `CROSS_MESH_DYNAMICAL_MATURITY_MECHANISM_REFUTED`

400×50 does **not** independently exhibit the coherent → cancelling →
no-useful-work sequence, and at 400×50 the tested dynamical structure does not
separate mature from premature states (`cosθ` +0.995 vs +0.998; `net/path` 0.977
vs 0.990) — the brief's definition of `REFUTED`.

**Disclosed deviation from my own preregistration.** §9 of `PREREGISTRATION.md`
mapped a C1 failure to `INCONCLUSIVE`, on the premise that C1 would fail only
because the cap was too short. That premise is positively contradicted by the
data: motion amplitude decays 19× and `‖Δρ‖₂` ends a factor of 10 below the
native tolerance, so "evidence/caps are insufficient" would be a false statement.
I therefore report the verdict under the brief's Phase 21 definitions rather than
my own mapping, and record the deviation here so a reader can disagree. The
preregistered criteria outcomes are reported unaltered in
[`DYNAMICAL_ANALYSIS.md`](DYNAMICAL_ANALYSIS.md) §8.

**What is not refuted:** the mechanism at 160×20 and 320×40 stands entirely
unchanged, as does the finding that production descends prematurely at the fine
meshes — now quantified at 400×50 for the first time, and larger than at either
coarser mesh.

### Next step

> ### `MORE_DYNAMICAL_EVIDENCE_REQUIRED`

Not `..._ROUTE_ABANDON`: the underlying problem is real and growing with
refinement (17.3 M_nd points and 2.17 % ω₁ abandoned at 400×50), and a
two-branch construction covering both terminal regimes is visible in the
evidence. Not `..._PREREGISTRATION_JUSTIFIED`: three of the seven Phase-18
conditions fail, and preregistering the disjunction now would be outcome-driven.

### Production and performance status

> `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`
> `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

Both unchanged by this task, as Phase 23 requires.
