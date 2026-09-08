# REPORT — dynamical-regime evidence generation

**Is topology maturity better characterised by the transition from coherent net
progress to oscillatory/cancelling density motion than by motion magnitude or
objective stall?**

**Yes — clearly, and at every mesh where it could be measured.** But one
observable is not yet sufficient on its own, and the decisive mesh still lacks
its counterfactual.

| | |
|---|---|
| HEAD at task start | `7154d8201e9defb06d0d758da866c3769c07179a` (branch `benchmark-methodology-r2`) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` — **unchanged by this task** |
| MATLAB / threads | R2025b Update 1 / `maxNumCompThreads(1)` |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256 `2ac361e769fe87ce458a4668840174a53de01de508da6b802c257982cab974b3`, frozen **2026-09-08T07:50:04Z, before either run** |
| Scientific runs executed | **exactly two**, as authorized |
| Solver copies | **none** — both runs call `olhoffSolve` unmodified |
| Production source changed | **none** |

Supporting detail: [`DYNAMICAL_ANALYSIS.md`](DYNAMICAL_ANALYSIS.md),
[`DATA_INVENTORY.md`](DATA_INVENTORY.md), [`PROVENANCE.md`](PROVENANCE.md),
`METRICS.json`, `figures/F1`–`F12`.

---

## Headline

**The regime exists, it is sharp, and it marks the end of useful work.**
Under fixed `move = 0.04` both meshes end in a period-2 cancelling regime —
160×20 at iteration **81**, 320×40 at **253**. At onset, the topology evolution
still remaining is **2.08 %** and **−0.58 %** respectively: essentially none.
After onset, 320×40 spends **947 further iterations** moving at full amplitude
and achieving nothing, M_nd drifting the *wrong* way by 0.5 points while `cosθ`
deepens to −0.967.

**Production descends at the right moment only at 160×20, and by coincidence.**

| mesh | production first descent | period-2 onset | offset | `cosθ` at descent | `net/path` at descent |
|---|---|---|---|---|---|
| 160×20 | 79 | **81** | **+2** | **−0.818** | **0.346** |
| 320×40 | 130 | **253** | **+123** | **+0.933** | **0.962** |
| 400×50 | 138 (converged 139) | not reached | ≥ +1 | **+0.938** | **0.978** |

β stalls when the objective stops improving. At 160×20 the objective stops
improving *because* the design has begun to cancel, so the two coincide to within
two iterations. At the fine meshes the objective stalls while the design is still
descending coherently — so β-stall fires 123 iterations early at 320×40, and at
400×50 fires once at 138 whereupon production terminates at 139 with
**M_nd = 32.33 %**.

**400×50 production never traverses its ladder at all**: one descent (0.04→0.02),
then convergence. Zero of 20 000 elements were at the move bound at that descent,
so its coherence reading cannot be a saturation artifact.

---

## The twenty-six required answers

**1. What exact repository state produced the new runs?**
HEAD `7154d8201e9defb06d0d758da866c3769c07179a`, branch `benchmark-methodology-r2`,
`+impl/` tree `c1455374d5f8e256…` (74 files), MATLAB R2025b Update 1, one
computational thread, production preset
`duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4`. Config hashes:
RUN A `044d50a496cb64d4…`, RUN B `4d49b0abf33edf77…`.

**2. Were all provenance/evidence gates clean?**
**Yes.** currentness `CURRENT`; source integrity 74/74; dispatch `ok=1`, 32
symbols, 0 blockers, 0 warnings; **published MMA wins** (`mmasub` →
`+impl/mma_published/`); **sensitivity filter wins** (`applyFilter` →
`+impl/filter/`); no forbidden Olhoff path; test suite **4/4, 0 failures**;
retained evidence **96/96 hash-valid**. Not `DYNAMICAL_AUDIT_PROVENANCE_FAIL`.
Single-factor gate: RUN A differs from production in `runtime.name` **only**;
RUN B in exactly its four declared overrides. No unexpected drift at either.

**3. Was the 320×40 common prefix reproduced?**
**Yes — bitwise, against both references.** vs `fixedmove_320x40` over 1…216 and
vs `armU_320x40` over 1…213: max |Δρ| = **0**, ω₁/M_nd/volume differences **0**,
event structure identical. The two archives were also confirmed bitwise identical
to each other over 1…213. Not `FIXEDMOVE_PREFIX_REPRODUCTION_FAIL`.

**4. What happened in the 400×50 production run?**
139 outer iterations, status **`CONVERGED`** (cap 400, not a cap hit), 2918 inner
MMA iterations, ω₁ = **162.888779843**, M_nd = **32.3283 %**, volume 0.49999915,
640 s. Exactly **one** move descent, at iteration 138. β-stall first fires at
138 — the same iteration. The run never leaves the coherent regime (fraction of
iterations labelled PERIOD2: **0.000**; labelled COHERENT: 0.863).

**5. When did 400×50 first descend from move = 0.04?**
**Iteration 138** (0.04 → 0.02). The run then converged at 139, so `move = 0.01`
and `0.005` were never reached.

**6. What was its dynamical regime at that instant?**
**Coherent, by every measure.** `cosθ = +0.938`, `net/path = 0.978` (97.8 % of
the ten-iteration path length is net displacement), trailing-median `q2 = 2.000`,
classifier label **COHERENT**, `boundFrac = 0.00000`. Instantaneous `q2 = 3.934`
is inflated by the step-size change at the descent itself and is not the clean
reading.

**7. Is 160×20 truly period-2?**
**Yes.** Onset at **81** under fixed move. Established regime: `cosθ` −0.90 to
−0.97, `q2` 0.25–0.36, `net/path` 0.06–0.11, `revFrac` 0.59–0.82. Zero undefined
`q2`/`cosθ` samples in 600 iterations.

**8. Is the period-2 state persistent?**
**Yes.** It satisfies the preregistered 20-iteration persistence requirement and
then holds for the remaining **519** iterations (85.2 % of the run carries the
label). At 320×40 it holds for **947** iterations after onset (79.0 % of the run)
and is still deepening at the cap. No label was assigned from a single dip.

**9. Does 320×40 eventually enter period-2 under fixed move?**
**Yes — this is the new evidence.** `PERIOD2_REGIME_OBSERVED`.

**10. At what iteration?**
**253.** The previous fixed-move evidence ended at 213/216 — **40 iterations
short of onset** — which is precisely why the question had been unanswerable.

**11. What topology evolution remained at that onset?**
**Essentially none: −0.58 %** on the M_nd basis (9.42 % on L1 displacement).
M_nd was 13.0275 % at onset and 13.5242 % after 947 further iterations — it moved
*backwards*. By contrast, between production's descent (130) and onset (253) M_nd
fell **23.32 → 13.03**, i.e. the entire remaining evolution.

**12. Does `q2` distinguish mature from premature states?**
**Yes**, via its trailing median (1.18 at 160×20's descent vs 2.00 at both fine
meshes). Its *instantaneous* value at a descent iteration is contaminated by the
step-size change and must not be used raw.

**13. Does `cosθ`?**
**Yes, most sharply of the three — by a change of sign**: −0.818 (160×20) vs
+0.933 (320×40) vs +0.938 (400×50) at the matched event. The classifier's `cosθ`
threshold could be moved anywhere in ±0.2 without changing a single onset.

**14. Does net/path ratio?**
**Yes**, and it is the only one of the three that is genuinely independent (a
10-step statistic, not a 2-step one): 0.346 vs 0.962 vs 0.978. Bounded in [0,1],
so directly interpretable as the cancelled fraction of path length.

**15. Which observable is least sensitive to isolated bound motion?**
At the fine meshes **all three are exactly insensitive** — `boundFrac = 0.0000`
at 320×40 throughout its transition and regime, and at 400×50 at its descent, so
deleting the saturated set changes nothing whatsoever. At 160×20, where 2–7 % of
elements are saturated, **`cosθ` is the most robust in sign** (established-regime
median −0.93 → −0.55 with the set removed, still firmly negative), while
`net/path` is the most attenuated (0.094 → 0.318). At the onset *instant* at
160×20 the attenuation is severe (`cosθ` −0.833 → −0.072).

**16. Which observable is most mesh-consistent?**
**`cosθ`, used as a sign test.** The preregistered rule fires at 81 and 253 —
both at ≈0 % remaining topology evolution — with no mesh-specific constant and no
NE exponent anywhere. Magnitudes at onset are *not* mesh-consistent (`cosθ`
−0.833 vs −0.627; `net/path` 0.302 vs 0.454), which is why a magnitude threshold
would not transfer and a sign transition does.

**17. Is mature topology better associated with cancellation than small step size?**
**Yes, decisively.** At 160×20 in the mature regime `max|Δρ|` sits at the move
bound **0.0400 for all 600 iterations** while ~90 % of path length is cancelled.
The motion is maximal and the progress is nil. This is the direct refutation of
the magnitude framing that the preceding study found blind (its best
mesh-normalised candidates read 1.04–1.14× across opposite maturity states).

**18. Does β stall precede dynamical maturity at fine meshes?**
**Yes.** β-stall first fires at 130 (320×40) against onset at 253 — **123
iterations early**; and at 138 (400×50), where the regime is never reached at
all. At 160×20 it fires at 79 against onset at 81 — coincident to within two
iterations. β-stall is an objective-progress test that happens to align with
cancellation only at the coarse mesh.

**19. Is there evidence for a controller based on recurrence/cancellation?**
**Yes — good evidence.** A directional-coherence / cancellation concept
identifies, at both observable meshes, the point beyond which no further topology
evolution occurs, is robust to a wide sweep of classifier settings (onset ±15 %,
remaining M_nd invariant), needs no NE exponent, and at the fine meshes is exactly
bound-independent. **No controller was implemented, and no threshold, persistence
or formula was chosen for production** — per Phase O.

**20. Is a single candidate now preregistrable?**
**Not yet.** Three gaps: (a) **no 400×50 fixed-move counterfactual exists**, so
the decisive mesh's onset is unmeasured and the 81 → 253 → ? sequence is
unclosed; (b) the **160×20 onset-instant bound attenuation** (`cosθ` −0.833 →
−0.072) is unexplained, so it is not yet known whether a controller should read
the sign, the magnitude, or a bound-excluded variant; (c) `q2` is unusable raw at
a descent iteration, and how a live controller should handle the step-size change
across a descent has not been settled.

**21. If not, what evidence is still missing?**
A **400×50 fixed-move `0.04` run** long enough to observe onset (extrapolating
81 → 253, plausibly several hundred iterations; RUN B needed 1200 iterations /
4133 s at 320×40, so 400×50 would be materially more) — and a decision, made
before that run, on which variant of the observable a controller would read.

**22. Did any production scientific source change?**
**No.** Nothing under `+impl/` was written; the tree hash was re-verified
unchanged after both runs. No preset, no schema field, no solver copy. `p`, mass
law, `q`, filter, `R`, projection, multiplicity, MMA variant, FE formulation,
eigensolver, objective, volume constraint and `move.levels` were asserted in code
before each solve.

**23. Did any raw evidence remain unmanifested?**
**No, for this study.** Both trajectories (`runs/runA_400x50.mat` 17 MB,
`runs/runB_320x40.mat` 94 MB) are hashed in `FINAL_SHA256.txt` and listed in
`DATA_MANIFEST.json`. *Pre-existing debt, unchanged and flagged again:*
`move_transition/runs/arm{P,U}_*.mat` are gitignored **and** absent from that
study's manifest — repairing another study's manifest was outside this scope.

**24. Does anything justify projection?** **No.** The finding concerns *when* the
step is reduced, not the material or filtering model. Projection is also under
the scope lock.

**25. Does anything justify changing `R = 0.06·b`?** **No.** Nothing here bears
on filter radius, and it is under the scope lock.

**26. Is the nine-mesh performance campaign still blocked?**
**Yes — `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.** No controller was designed,
preregistered, validated or promoted in this task, and RUN A shows the fine-mesh
production result is a `move = 0.04` result that terminates one iteration after
its only descent, at M_nd = 32.33 %.

---

## Verdicts

Dynamical mechanism:

> ### `DYNAMICAL_MATURITY_SIGNAL_NARROWED`

The regime hypothesis is **confirmed, not refuted**: mature and premature states
are separated by the sign of the directional correlation and by a 2.8× difference
in cancelled path fraction, at matched events, with no mesh-specific tuning and
no NE exponent — where the preceding study's magnitude candidates were blind
(1.04–1.14×). Period-2 onset marks the end of topology evolution at both meshes
where it is observable (2.08 % and −0.58 % remaining). This is *clearly useful*.
It falls short of `IDENTIFIED` because no single observable is yet sufficient for
preregistration: the decisive 400×50 counterfactual does not exist, the 160×20
onset-instant reading is bound-carried, and `q2` is unusable raw across a
descent — see answer 20.

Next step:

> ### `MORE_DYNAMICAL_EVIDENCE_REQUIRED`

Specifically the 400×50 fixed-move counterfactual, plus a decision on which
variant of the observable a controller would read — both settled *before* any
controller preregistration.

Standing state, unchanged by this task:

> `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`
> `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

---

## Correction to the record

The preceding study established that a 400×50 figure quoted in an earlier
brief — "production first-descent M_nd ≈ 32.33 %" — had no evidentiary basis,
because no 400×50 run existed. RUN A now measures 400×50 production directly:
**M_nd = 32.3588 % at the first descent (138) and 32.3283 % at convergence
(139)**. The value 32.33 is therefore a real 400×50 production quantity. Its
companion claims remain unsupported: there is still no fixed-move 400×50 run, so
"mature fixed-move M_nd ≈ 16.16 %" (exactly half of 32.33) and "ω₁ improves
≈ 2.14 %" are not measured by anything, and were not used here.
