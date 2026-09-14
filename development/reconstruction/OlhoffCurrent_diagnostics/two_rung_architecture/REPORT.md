# REPORT — does the two-rung ladder `[0.04, 0.02]` suffice?

**Zero-scientific-run offline audit.** No optimization was executed. Every number
comes from three causal-controller trajectories that already existed.

> **Headline.** The exact `[0.04, 0.02]` policy is a valid, honestly-terminating,
> production-beating architecture on all three primary meshes. It repairs the
> 160×20 `ω₁` regression that sank the single-stage policy, it converts the
> 320×40 non-terminating four-rung run into a converged endpoint, and it saves
> 22–92 % of inner MMA work. It is **not** fully supported, for one reason and
> one only: at 160×20, rungs 3+4 still improve `ω₁` by **+0.114 %** against a
> preregistered materiality bar of **0.10 %**. The bar was frozen before the
> number was computed and the number is over it.

---

## The two required verdicts

```
TWO_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED
MORE_TWO_RUNG_EVIDENCE_REQUIRED
```

Both follow mechanically from the frozen mapping in `PREREGISTRATION.md` §12,
applied in `scripts/tr_metrics.py`. Eleven of the brief's twelve Phase-19
conditions hold; condition 8 ("rungs 3+4 contribute only immaterial incremental
benefit") fails at 160×20 and only at 160×20.

---

## 1. The forty questions

**1. What branch/HEAD was audited?**
`benchmark-methodology-r2`, HEAD `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`,
recorded at task start before anything else. Working tree dirty on 21 paths, none
of them under `+impl/`.

**2. Was `+impl` unchanged?**
Yes. Tree SHA-256 `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`
(75 files) at start and at end — byte-identical to the hash `move_ladder_necessity`
recorded. Currentness `CURRENT`, source integrity `PASS` (0 mismatched, 0 missing,
0 extra).

**3. Were zero scientific runs executed?**
Yes. No `olhoffSolve` call of any kind. MATLAB ran only `tr_provenance.m` and the
finalization gate, both of which read and hash files. All numerical work is Python
reading pre-existing `.mat` and `.csv`.

**4. What exact frozen `A`/`B` definitions were used?**
Those of `+impl/architecture/+olh/+move/exhaustion.m`, whose provenance line names
`two_branch_maturity_240/PREREGISTRATION.md` SHA-256 `62748225…` (verified live):

```
tol(NE) = 0.05*sqrt(NE/3200);   W = 20;  P = 20;  W_np = 10
A(k) = med20 cos < 0  AND  med20 net < 0.5  AND  amp >= tol
B(k) = amp < tol      AND  med20 cos > 0
E    = A OR B,  declared at the first k where either has held 20 consecutive iterations
reset: on descent stageStart <- new stage's first iteration, counters zeroed,
       every step in every window stage-local (only the net-path anchor is pre-stage)
```

An independent Python re-implementation reproduced the controller's own
`exA`/`exB`/`exE`/`nA`/`nB` trace **element-wise in every stage of every mesh**,
and every recorded declaration exactly. Nothing was tuned.

**5. Was the audit preregistered before final S2 extraction?**
Yes. `PREREGISTRATION.md` SHA-256 `b50455fbd3d97dcb72093fc13738f236d4d0f2e37a47caba0f515a16aaf7ca04`,
frozen `2026-09-09T15:57:27Z`, before any S2 `M_nd`, `ω`, gap, gray, mid, volume
or density value was read. Its §13 discloses exactly what *was* already visible —
including that S2's `M_nd` and `ω₁` already sit in
`move_ladder_necessity/METRICS.json` as `rungs[2].Mnd_to` / `omega1_to`, which
were not opened before freezing. The materiality thresholds and acceptance gates
are inherited verbatim from two earlier frozen preregistrations, not chosen here.

**6. What are the verified S1 event indices?**
`kE(1) =` **102** (160×20, branch A), **274** (320×40, branch A), **388**
(400×50, branch B). Descents applied at 103 / 275 / 389. Recovered from the raw
trajectory by independent replay, not hard-coded; they match the prior study's
recorded values exactly. The earlier briefs' "~102/103" is precisely this
declaration/descent pair — see `COUNTERFACTUAL_VALIDITY.md` §4 for the single
convention used throughout.

**7. What are the verified S2 event indices?**
`kE(2) =` **141** (160×20), **313** (320×40), **427** (400×50). Descents to 0.01
applied at 142 / 314 / 428 in the four-rung run.

**8. Which branch triggered each S2 event?**
160×20: **Branch A** (`med₂₀cosθ = −0.857`, `med₂₀net = 0.226`, `‖Δρ‖₂ = 0.1095`
against `tol = 0.05` — persistent cancellation at non-negligible amplitude).
320×40 and 400×50: **Branch B** (`med₂₀cosθ = +0.9999`, `‖Δρ‖₂/tol = 0.108` and
`0.145` — coherent motion at converged amplitude).

**9. Is each S2 state an exact valid counterfactual terminal state?**
Yes, on all three meshes. Under `stageExhaustion` the entire dependence on
`numel(cfg.move.levels)` is two predicates, `stage < numel` in `olh.move.limit`
and `stage >= numel` in `olhoffSolve`, each consumed only in conjunction with
`ex.declared`. Both ladders therefore agree at every iteration `k ≤ kE(2)` and
first diverge at `kE(2)` itself, where the four-rung run descends and the
two-rung run stops. The argument is set out in full, from the source lines, in
`COUNTERFACTUAL_VALIDITY.md` §1, and all five preregistered checks pass.

**10. Did every mesh remain at 0.02 until S2?**
Yes — check 3 of five. Also: exactly one descent occurred before `kE(2)`, at
`kE(1)+1` (check 2); no `move ≤ 0.01` iteration occurs at or before `kE(2)`
(check 4); and `hist.exStageStart = kE(1)+1` throughout stage 2, so the recorded
reset semantics are exactly the two-rung policy's (check 5).

**11. What are P/S1/S2/F `M_nd` values?**

| mesh | P | S1 | **S2** | F |
|---|---|---|---|---|
| 160×20 | 13.4025 | 13.0364 | **12.7884** | 12.7041 |
| 320×40 | 23.3596 | 13.0121 | **12.9797** | 12.9233 (CAP_HIT) |
| 400×50 | 32.3283 | 15.6649 | **15.4413** | 15.3311 |

**12. What are P/S1/S2/F `ω₁` values?**

| mesh | P | S1 | **S2** | F |
|---|---|---|---|---|
| 160×20 | 169.4952 | **168.9804** | **169.8175** | 170.0110 |
| 320×40 | 165.9508 | 166.4216 | **166.4163** | 166.4189 |
| 400×50 | 162.8826 | 166.4176 | **166.4355** | 166.4562 |

**13. Does S2 recover the unacceptable S1 `ω₁` regression at 160×20?**
**Yes, completely.** S1 sat at 168.9804, i.e. **0.5148 below** production —
the regression on the maximized objective that the previous audit called
unacceptable. S2 sits at 169.8175, **0.3223 above** production (+0.190 %). The
preregistered no-regression gate **A9** (`ω₁(S2) ≥ ω₁(P)`) **PASSES**, and it
fails at S1. Rung 2 turns a deficit into a surplus.

**14. How much `M_nd` does rung 2 buy at 160×20?**
−0.2479 absolute, **−1.902 % relative** — below the 2 % materiality bar, so not
material on `M_nd` alone. It is 74.6 % of the total `M_nd` gain the whole lower
ladder delivers there.

**15. How much `ω₁` does rung 2 buy at 160×20?**
+0.8372 absolute, **+0.4954 % relative** — nearly 5× the 0.10 % bar, and
**material**. It is 81.2 % of the total lower-ladder `ω₁` gain, and it includes
all of the part that erases the production deficit.

**16. How much do rungs 3+4 buy afterward?**
At 160×20: `M_nd` −0.0843 (−0.659 %, immaterial), `ω₁` +0.1935
(**+0.1139 %, material** — the single crossing in this audit), gray +0.000625,
mid +0.000625, mean |Δρ| 0.00143, volume and multiplicity unchanged. Cost: 78
outer, 1 427 inner.

**17. Does 320×40 satisfy `E` on `move = 0.02`?**
Yes — **Branch B at iteration 313**, window [294, 313], `med₂₀cosθ = +0.99996`,
`‖Δρ‖₂ = 0.01080` against `tol = 0.1`. Full 20-iteration persistence.

**18. Does terminating at S2 avoid the 320×40 CAP_HIT?**
**Yes.** The four-rung run reaches `CAP_HIT @1600` because its terminal
`move = 0.005` stage enters *low-amplitude cancellation* — at iteration 1600,
`‖Δρ‖₂ = 0.00419` (well below `tol = 0.1`) while `med₂₀cosθ = −0.887` — which is
precisely the documented hole in the union: neither Branch A (amplitude too
small) nor Branch B (motion not coherent) can fire. The two-rung policy never
enters that stage; it terminates honestly at 313 under the same frozen concept.
`F` at 320×40 remains labelled **CAP_HIT** everywhere in this study.

**19. How much scientific benefit is lost at 320×40 by omitting 0.01/0.005?**
`M_nd` 12.9797 → 12.9233, a **0.435 %** relative difference; `ω₁` 166.4163 →
166.4189, **+0.0016 %**; gray −0.000625; mid +0.000156; mean |Δρ| 0.00204;
subspace size, mode order, gap and volume all unchanged. **Below every
preregistered bar on every metric.**

**20. How much work is saved [at 320×40]?**
**1 287 outer iterations (80.4 %)** and **70 780 inner MMA iterations (92.5 %)**.
Rung 4 alone consumed 91.5 % of that mesh's entire inner budget.

**21. Does 400×50 satisfy `E` on `move = 0.02`?**
Yes — **Branch B at iteration 427**, window [408, 427], `med₂₀cosθ = +0.99964`,
`‖Δρ‖₂ = 0.01806` against `tol = 0.125`.

**22. How much scientific benefit is lost at 400×50 by omitting 0.01/0.005?**
`M_nd` 15.4413 → 15.3311 (**0.714 %**), `ω₁` +0.0207 (**+0.0124 %**), gray
−0.0004, mid −0.0001, mean |Δρ| 0.000598 — the smallest topology change of the
three meshes. **Below every bar.**

**23. How much work is saved [at 400×50]?**
78 outer iterations (**15.4 %**) and 2 264 inner MMA iterations (**22.0 %**).

**24. Are rungs 3+4 below all preregistered materiality bars on every mesh?**
**No — on two of three.** Below every bar at 320×40 and 400×50. At 160×20 they
clear the `ω₁` bar (+0.1139 % against 0.10 %) and are below every other bar.

**25. Do rungs 3+4 provide any material `ω₁` benefit?**
**Yes, at 160×20 only**: +0.1935 absolute, +0.1139 % relative, against a frozen
0.10 % bar. The margin over the bar is 0.0139 percentage points. It is narrow and
it is real; it is not explained away, and it is the sole reason this audit does
not return `SUPPORTED`. At 320×40 (+0.0016 %) and 400×50 (+0.0124 %) the gain is
one to two orders of magnitude below the bar.

**26. Do rungs 3+4 provide any material topology benefit?**
**No, on any mesh.** Mean |Δρ_e| is 0.00143 / 0.00204 / 0.000598 against a 0.01
bar; gray and mid fractions move by ≤ 0.000625 against a 0.01 bar. The S2 and F
designs are the same structure (`figures/F9`).

**27. Do rungs 3+4 provide any material multiplicity/gap benefit?**
**No.** Subspace size is 2 at S2 and at F on all three meshes; mode order never
changes; `ω₂ > ω₁` throughout; no NaN/Inf; zero non-converged inner solves. Gap
magnitude alone is explicitly not a materiality criterion (the objective is `ω₁`).
At 160×20 the gap narrows 0.01251 → 0.00835 as `ω₁` rises and `ω₂` falls inside
the two-mode cluster — the mechanism behind Q25's gain, not an independent
multiplicity effect.

**28. Do rungs 3+4 provide any material volume/feasibility benefit?**
**No.** |volume − 0.5| changes by −2.14e-06 / +2.87e-07 / +4.47e-07 against a
1e-5 bar, and every state is two to three orders of magnitude inside the 1e-4
gate.

**29. Is rung 2 genuinely load-bearing at 160×20?**
**Yes, decisively.** It is material on `ω₁` (+0.4954 %, ~5× the bar), it delivers
81.2 % of the whole lower ladder's `ω₁` gain, and it is what moves `ω₁` from
below production to above it. Without it the policy ships a regression on the
maximized eigenfrequency.

**30. Is rung 2 sufficient at 320×40?**
**Yes** — everything below `move = 0.02` is immaterial there, and everything
below it is also where the run fails to terminate.

**31. Is rung 2 sufficient at 400×50?**
**Yes** — rungs 3+4 are below every bar on every metric.

**32. Is the exact `[0.04, 0.02]` architecture supported on all three primary meshes?**
**Two of three fully; the third with one qualification.** All three: `E` fires on
`move = 0.02`, the counterfactual is exact, and every preregistered acceptance
gate passes at S2. 320×40 and 400×50 are additionally *two-rung-sufficient* —
rungs 3+4 are immaterial. 160×20 is not, because rungs 3+4 remain material there
on `ω₁`.

**33. Is any mesh-specific tuning required?**
**No, and none was performed.** One rule, one set of thresholds, one ladder, one
index convention, applied identically to all three meshes.

**34. Was A/B modified?**
No. `exhaustion.m` was read, never written. `W = 20`, `P = 20`, `W_np = 10`,
`tol = 0.05·√(NE/3200)`, the two predicates, the union and the reset semantics are
all unchanged, and the offline replay reproduces the in-loop trace element-wise.

**35. Was Branch C added?**
No.

**36. Was production modified?**
No. `move.levels` resolves to `[0.04 0.02 0.01 0.005]`, `move.continuation.signal`
to the β-stall signal, unchanged. **`PRODUCTION_CONTROLLER_NOT_CHANGED`.**

**37. Is evidence retention fail-closed?**
Yes. `EVIDENCE.json` declares the raw trajectories by immutable path and measured
SHA-256; `olhoffcurrent_evidence_declare` refuses to declare a required artifact
that is not on disk; `FINAL_SHA256.txt` is self-verifying; the finalization gate
enforces G1–G5.

**38. Did finalization gate pass?**
**Yes — all five gates.** `olhoffcurrent_finalization_gate`: G1 declares raw
evidence (`EVIDENCE.json`, 9 artifacts, 7 required) · G2 every required declared
artifact present and hash-valid · G3 `FINAL_SHA256.txt` present · G4 it is
self-verifying (37 paths, all resolve and hash as recorded) · G5 no `.mat` named
by any manifest is absent. `DATA_MANIFEST.json` and `FINAL_SHA256.txt` cover
identical file sets. **RESULT: PASS.**

**39. Is two-rung policy preregistration justified?**
**No — `MORE_TWO_RUNG_EVIDENCE_REQUIRED`.** `TWO_RUNG_POLICY_PREREGISTRATION_JUSTIFIED`
requires the architecture verdict `SUPPORTED` plus all twelve Phase-19 conditions.
Condition 8 fails at 160×20. The architecture is close, and the failure is
narrow, but the bar was frozen in advance and it was crossed.

**40. Is the nine-mesh campaign still blocked?**
Yes. **`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`**, by the brief's Phase 22
unconditionally, and independently because no policy is frozen to run it with.

---

## 2. What the audit found that the brief did not ask about

**Every lower stage declares at the earliest arithmetically possible iteration.**
The stage-2 declaration latency is **38 on all three meshes** — the minimum
possible, since the trailing 20-median is first defined at `stageStart + 19` and
the counter then needs 20 consecutive hits. The same holds for stage 3 on all
three meshes and stage 4 at 160×20 and 400×50. In other words, in every
post-first stage on which the rule fires, the predicate is *already true* at the
first iteration at which it can be evaluated. The detector is not observing a
transition inside those stages; it is confirming an inherited condition for the
minimum admissible time. Every terminating lower rung is therefore exactly 39
iterations long.

This does not weaken the counterfactual — S2 remains exactly the state the
two-rung policy returns — but it does mean stage 2 is operationally "39 more
iterations at half the move limit, then stop" rather than "run at 0.02 until the
dynamics say stop". **No rule was altered in response to this** (PREREGISTRATION
§14). It is recorded for the next task.

**The 320×40 `CAP_HIT` is exactly the documented hole, and it is confined to
`move = 0.005`.** At iteration 1600, `‖Δρ‖₂/tol = 0.042` with `med₂₀cosθ = −0.887`:
low-amplitude cancellation, which satisfies neither branch. The controller study
preregistered this risk before its runs and declined to repair it. The two-rung
architecture avoids it structurally rather than by patching the rule.

**Wall-clock time is not usable on these runs.** Seconds per inner MMA iteration
drift 3.8×–5.2× within each trajectory — a machine property, not an algorithmic
one. All cost claims here rest on outer iterations and inner MMA iterations.

## 3. Bound limitation — descriptive only (Phase 16)

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `‖Δρ‖₂/tol` at S1 | **12.59** | 1.03 | 0.64 |
| `‖Δρ‖₂/tol` at S2 | **2.19** | 0.108 | 0.144 |
| `max|Δρ|/move` at S1 | 0.9996 | 0.672 | 0.143 |
| `max|Δρ|/move` at S2 | 0.9971 | 0.039 | 0.060 |
| bound fraction at S1 | 0.0713 | 0.0 | 0.0 |
| branch at S1 → S2 | A → A | A → B | B → B |
| rung-2 `ω₁` gain | **+0.495 %** | −0.003 % | +0.011 % |

The pattern is consistent with the hypothesis and is stated as an observation,
not a law. 160×20 exits *both* stages on Branch A — persistent cancellation while
the step is still large — and is still moving at 2.2× the convergence tolerance
when the two-rung policy stops, with `max|Δρ|` still pinned at 99.7 % of the move
limit. That is the mesh where rung 2 buys something. The fine meshes exit on
Branch B with amplitude already an order of magnitude below tolerance, and there
the lower rungs buy nothing.

**This is three points. It is not promoted to a scaling law, and it altered no
rule in this task.** It offers a plausible reading of *why* the coarse mesh needs
rung 2 — the stage-1 exit is still step-limited, so halving the limit still
releases real motion — and nothing more.

## 4. Verdict derivation, step by step

Preregistered §12: a mesh is *two-rung-sufficient* when `E` fires on `move = 0.02`,
S2 clears every §10 gate, **and** rungs 3+4 are below every §9 materiality bar.

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `E` on `move = 0.02` | ✅ (A @141) | ✅ (B @313) | ✅ (B @427) |
| counterfactual exact + replay matches | ✅ | ✅ | ✅ |
| A1/A2/A3 `M_nd` bound at S2 | ✅ | ✅ | ✅ |
| A4 `ω₁ ≥ 0.99 × P` | ✅ | ✅ | ✅ |
| A5 volume ≤ 1e-4 | ✅ | ✅ | ✅ |
| A6 physics | ✅ | ✅ | ✅ |
| A7 cost multipliers | ✅ | ✅ | ✅ |
| A8 terminal persistence at last level | ✅ | ✅ | ✅ |
| A9 no `ω₁` regression (160×20) | ✅ | — | — |
| rungs 3+4 immaterial on every bar | **❌ (`ω₁`)** | ✅ | ✅ |
| **two-rung-sufficient** | **no** | **yes** | **yes** |

`SUPPORTED` requires all three sufficient → not met.
`REFUTED` requires `E` to fail somewhere, or rungs 3+4 material on ≥ 2 meshes, or
A4 to fail, or rung 2 to be immaterial at 160×20 while the regression persists →
none met.
`PARTIALLY_SUPPORTED` requires every mesh to satisfy `E` on 0.02, no validity
check to fail, and **exactly one** qualifying shortfall → met: a single mesh
shows a material rung-3+4 benefit.

Phase-19 conditions: **11 of 12 hold.** Condition 8 fails.
→ **`MORE_TWO_RUNG_EVIDENCE_REQUIRED`.**

## 5. What the next task would need

Not more architecture mining — the brief forbids it and this audit performed
none. What would resolve the one open question is a decision, taken in advance,
about whether **+0.114 % of `ω₁` at the coarsest mesh** justifies two rungs that
cannot terminate at 320×40. That is a judgement about the acceptable trade
between objective value and controller robustness, and it belongs in a
preregistration written before the answer is known — not in this report, and not
by relaxing a bar that has already been crossed.

## 6. Figures

| | |
|---|---|
| `F1_Mnd_P_S1_S2_F.png` | `M_nd`: P vs S1 vs S2 vs F, each mesh |
| `F2_omega1_P_S1_S2_F.png` | `ω₁`: P vs S1 vs S2 vs F, each mesh — the 160×20 panel is the safety story |
| `F3_move_history_events.png` | move history, four-rung vs two-rung counterfactual, with both `E` events |
| `F4_cumulative_inner_work.png` | cumulative inner MMA work with S1/S2/F markers |
| `F5_cumulative_wall_time.png` | cumulative wall time, marked **NOT RELIABLE** on its face |
| `F6_rung2_value.png` | rung-2 scientific value against the frozen bars |
| `F7_rung34_value.png` | rungs-3+4 incremental value — the single green bar is the 160×20 `ω₁` crossing |
| `F8_rung_cost.png` | rung-2 vs rungs-3+4 cost, outer and inner |
| `F9_topology_S2_vs_F.png` | S2, F and their difference, each mesh |
| `F10_multiplicity_gap.png` | gap and subspace size, with S1/S2/F markers |
| `F11_architecture_summary.png` | P/S1/S2/F across all three meshes |

---

## 7. Final summary

| | |
|---|---|
| branch | `benchmark-methodology-r2` |
| starting HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| final HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (no commit made) |
| dirty state, start | dirty, 21 paths (two prior studies' deliverables + this study's directory) |
| dirty state, end | dirty, 21 paths (same set; this study's files are all inside its own untracked directory) |
| `+impl/` hash, start | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| `+impl/` hash, end | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` — **unchanged** |
| **scientific runs** | **0** |
| tests / gates | provenance gate `TWO_RUNG_EVIDENCE_GATE_PASS` (10/10) · frozen-rule replay element-wise match in 12/12 stages · counterfactual validity 5/5 on 3/3 meshes · finalization gate **PASS** (G1–G5) |
| preregistration hash | `b50455fbd3d97dcb72093fc13738f236d4d0f2e37a47caba0f515a16aaf7ca04`, frozen `2026-09-09T15:57:27Z` |
| evidence manifest | `EVIDENCE.json` 9 artifacts (7 required, all present + hash-valid) · `DATA_MANIFEST.json` 37 artifacts · `FINAL_SHA256.txt` 37 paths, self-verifying |
| finalization gate | **PASS** |

```
TWO_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED

MORE_TWO_RUNG_EVIDENCE_REQUIRED

PRODUCTION_CONTROLLER_NOT_CHANGED

NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```
