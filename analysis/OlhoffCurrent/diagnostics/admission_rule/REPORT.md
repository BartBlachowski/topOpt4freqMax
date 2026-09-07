# REPORT — convergence admission rule

Can an admission rule be designed that the move ladder cannot trip?

| | |
|---|---|
| Implementation | `analysis/OlhoffCurrent` (sole production Olhoff) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 source files) |
| Repository HEAD at freeze | `2b89b070477c67aa7bbe5febc369f61fcad92da6` |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` |
| cfg hash 160×20 / 320×40 | `ca9a5c90d5c4…` (ε = 0.05) / `abae6b78ce19…` (ε = 0.10) |
| MATLAB | R2025b Update 1, **1 computational thread** |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), frozen before any evaluation |
| Provenance | [`PROVENANCE_LEDGER.md`](PROVENANCE_LEDGER.md) |

Path gate passed before every run: `CURRENT`, 74/74 source files, 32 owned
symbols inside `+impl/`, no competing Olhoff tree, published MMA and sensitivity
filter both winning.

---

## Headline

The candidate **does** enforce the invariant — a move descent can no longer
manufacture convergence — and it costs nothing: ω₁ improves slightly at both
meshes, volume stays feasible, the topology is unchanged in character.

**But fixing admission does not recover the fine-mesh grayness.** At 320×40 the
candidate reduces M_nd by 1.5 %, not the ≳25 % the previous diagnostic's
fixed-move arm achieved. The reason is measured here directly: with the ladder
still descending, the design **freezes** — running the unstopped trajectory 469
iterations past the production stop moves M_nd only 23.36 → 22.05, while holding
the move at 0.04 reached 13.28. **The move schedule, not the admission rule, is
what governs the grayness.**

That is a negative result for the grayness question and a positive one for
correctness, and the two must not be conflated.

---

## Method, and why it is exact

`convOuter` in `olhoffSolve` is computed **after** `olh.move.limit`, is never
written back into the move-controller state, and — with `projection.enabled =
false` and `stiffness.continuation.enabled = false` — is read only by the
guards and the `break`. The trajectory is therefore **independent of when
convergence is admitted**, so one *unstopped* run per mesh contains the exact
trajectory every candidate would follow, and candidates are evaluated offline
on it. The promoted production source was not modified.

Unstopping used two stopping-policy fields only (`stop.tolerance = 0`,
`stop.toleranceRule = 'explicit'`); `ar_run` asserts field-by-field over all 80
schema fields that nothing else differs from production.

**The independence claim was verified, not assumed** (§9 gate below).

---

## §9 Baseline anchor gate — PASS at both meshes

Replaying the *production* admission rule on the unstopped trajectory:

| | 160×20 | 320×40 |
|---|---|---|
| production admission fires at | **iteration 91** | **iteration 131** |
| archived conference record | 91 | 131 |
| design at that iteration vs archived final | **bitwise** | **bitwise** |
| per-iteration history over 1..N, 12 fields | **bitwise** | **bitwise** |
| post-loop ω₁ recomputed at that design | **169.495227** | **165.9507893** |
| archived ω₁ | 169.49522702153845 | 165.95078925220545 |

The offline reconstruction reproduces the archive exactly, including the
solver's terminal modal analysis. The evaluation method is sound.

*(`hist.omega` is recorded at the start of each iteration, before that
iteration's update, so it legitimately differs from the post-loop value; the
post-loop value was recomputed by repeating the solver's own terminal
`assemble2D` + `eigSolve`, and it matches the archive bitwise.)*

---

## The twenty questions

### 1. What was wrong with the old admission rule?

`‖Δρ‖₂ < ε` with the `settledMove` guard. Since `max|Δρ| ≤ move`, scaling the
move down scales the statistic down with it. Measured at 160×20: the descent at
iteration 90 cut `‖Δρ‖₂` by 2.75× across ε while M_nd moved +0.19 %, and the run
was declared converged at 91 — one iteration later, with
`max|Δρ|/move = 0.581`, i.e. **still pressing against its own bound**.

### 2. What new invariant does the candidate enforce?

> A change of the move limit cannot, by itself, cause convergence.

Enforced two ways. **Structurally**, by requiring the move level to have been
held for `D` iterations, so no admission can occur within `D` of any change.
**Substantively**, by requiring the dimensionless `max|Δρ|/move < τ_rel`, which
is invariant under a change of bound: a design still saturating its bound has
ratio ≈ 1 whatever the bound is, and halving the bound cannot reduce it.

### 3. How was the local-change threshold chosen?

`τ_abs = 0.01` is the convergence criterion of the Bendsøe & Sigmund 99-line
code (`change = max(max(abs(x-xold))); while change > 0.01`), which the book
itself footnotes as *"a rather 'sloppy' convergence criterion and could be
decreased if needed"*. The same criterion ships inside the production tree as
`top88_reference.m`. Literature, not tuning.

`τ_rel = 0.5` comes from the ladder's own halving factor: below it the realized
step already fits inside the **next** ladder level, so the next descent could not
further restrict it. Derived from the move schedule, not from output.

**Neither was chosen by looking at topology.** A crucial reason `τ_abs` alone is
not enough: `max|Δρ| ≤ move`, so once the ladder reaches 0.005 *any* absolute
threshold ≥ 0.005 is automatically satisfied — reproducing the very defect under
study. The dimensionless ratio is what carries the invariant.

### 4. How was the move-settling dwell chosen?

`D = 10` = `move.continuation.window`, the solver's own existing judgement of the
timescale on which a response to a move level can be assessed. Reused, not
invented.

### 5. How was objective stability defined?

Relative range of ω₁ over a window of `W = 10`, below `τ_obj = 5e-3` — the
solver's own `move.continuation.tolerance`, i.e. its existing definition of
"relative progress has stalled", applied to the objective instead of the bound
variable.

### 6–7. Did each mesh terminate naturally?

**Yes, both, well before the 600 cap.**

| Mesh | production | C1 admits | C2 admits |
|---|---|---|---|
| 160×20 | 91 | **100** | 117 |
| 320×40 | 131 | **140** | 151 |

### 8. Did either stop immediately after a move descent?

**No.** Gap to the preceding descent is **10** at both meshes for C1 (and 16 / 10
for C2) — never 1, as production does. By construction A guarantees ≥ `D`.

**But the two meshes differ in *why*, and this matters:**

* **160×20** — the production stop is rejected **on the ratio test itself**:
  `ratio = 0.5810 > τ_rel = 0.50`, so `Brel = 0` at iteration 91 independently of
  the dwell. The design really was bound-saturated there.
* **320×40** — the production stop **passes both B and C**
  (`ratio = 0.4094 < 0.50`, `objRange = 1.49e-3 < 5e-3`). Only the dwell rejects
  it. At this mesh the design was *not* bound-saturated at the production stop;
  the descent at 130 released it from the bound (ratio 0.647 → 0.388).

So the substantive part of the rule bites at the coarse mesh and not at the fine
one. Reported plainly rather than glossed.

### 9. max|Δρ| and max|Δρ|/move at stop

| Mesh | rule | iter | move | max\|Δρ\| | **ratio** | ‖Δρ‖₂ |
|---|---|---|---|---|---|---|
| 160×20 | production | 91 | 0.01 | 0.005810 | **0.5810** | 0.02358 |
| 160×20 | **C1** | 100 | 0.01 | 0.003783 | **0.3783** | 0.009340 |
| 160×20 | C2 | 117 | 0.005 | 0.001099 | 0.2198 | 0.003181 |
| 320×40 | production | 131 | 0.02 | 0.008189 | **0.4094** | 0.09008 |
| 320×40 | **C1** | 140 | 0.02 | 0.007861 | **0.3930** | 0.08332 |
| 320×40 | C2 | 151 | 0.01 | 0.002014 | 0.2014 | 0.02223 |

### 10. Was the objective still improving materially?

No, under the preregistered measure: ω₁ relative range over 10 iterations is
2.57e-4 (160×20) and 2.22e-4 (320×40) at the C1 admission points, both ~20×
inside `τ_obj`.

**A caveat that matters.** At 320×40 the objective had plateaued long before the
density field had. Over the 30 iterations preceding the production stop, ω₁
changes at a relative rate of **+3.0e-4 /iter** while M_nd falls at
**−6.9e-3 /iter** — 23× faster. **ω₁ stability is a poor proxy for design
maturity in this problem**, and component C therefore contributes almost nothing
at the fine mesh.

### 11–13. M_nd, mid-density, ω₁

| Mesh | metric | production | C1 | change |
|---|---|---|---|---|
| 160×20 | M_nd | 13.4025 % | 13.3325 % | **−0.52 %** |
| 160×20 | mid-density | 0.0250 | 0.0256 | +2.4 % |
| 160×20 | ω₁ (post-loop) | 169.495227 | 169.527086 | **+0.0188 %** |
| 320×40 | M_nd | 23.3596 % | 23.0104 % | **−1.49 %** |
| 320×40 | mid-density | 0.0956 | 0.0881 | **−7.84 %** |
| 320×40 | ω₁ (post-loop) | 165.950789 | 165.988027 | **+0.0224 %** |

ω₁ **improves** at both meshes; nothing is traded away. But the M_nd reduction
is ~1.5 %, nowhere near the 25 % the preregistration set for G13.

### 14. Was volume feasible?

Yes. `mean(ρ)` = 0.499999 (160×20) and 0.499999 (320×40); `|mean − 0.5| ≤ 1e-6`,
inside the 1e-3 gate.

### 15. Did solver health remain acceptable?

Yes. No solver failure, no non-finite design, volume feasible, the only log
entries being the pre-existing `omega_J (J=3) is itself multiple — (25b)
undefined` notices at iterations 8/9, identical to the archived baseline.

### 16. Did the candidate preserve reconstruction-like topology?

Yes — see `fig1_topology_comparison.png`. The candidate designs are the
production designs a few iterations further along the *same* trajectory; member
layout is unchanged.

### 17. Does the result support promotion?

**Partly, and not for the reason the study was commissioned.**

*For* promotion: the rule removes a real correctness defect, is derived entirely
from literature and existing solver constants, costs 9–10 extra outer iterations,
and slightly improves both ω₁ and the mid-density fraction at the fine mesh.

*Against* treating it as the grayness fix: it delivers ~1.5 % of M_nd where the
problem is ~43 %. Promoting it and declaring the grayness issue addressed would
be wrong.

Two further caveats a promotion review must weigh:

1. **B is evaluated instantaneously.** At 160×20 the ratio was 0.550 at
   iteration 99 and 0.378 at 100 — it fluctuates across `τ_rel`, so admission
   lands on a single favourable reading. A windowed form of B (require it to
   hold for the dwell, not merely at its end) would be more robust and is the
   obvious refinement.
2. **At 320×40 the admission is dwell-limited, not maturity-limited.** Descents
   fall at 130, 141, 152 — 11 apart — so a dwell of 10 barely fits between them
   and C1 admits at 140, one iteration before the next descent. The rule is
   being paced by the ladder, not by the design.

### 18. Does the result require 400×50 confirmation?

**No, and it is not authorized.** §17 of the brief permits it only if both
meshes pass. G13 fails at 320×40, so the confirmatory mesh was not run.

### 19. Does anything justify changing the filter radius now?

**No.** Nothing in this study varied the radius, and nothing here bears on it.
The residual-grayness confound recorded in the previous diagnostic — `R = 0.06·b`
is class C, unstated in both papers, and the strongest determinant of member
thickness — remains open and unaddressed.

### 20. Does anything justify projection now?

**No.** No evidence here bears on projection, which is a class-D modification
absent from every source. The finding points at the **move schedule**, which is
also class C and already in the realization — adding a new class-D mechanism
would be the wrong response to it.

---

## §14 Old stop vs new stop

| Field | 160×20 production | 160×20 C1 | 320×40 production | 320×40 C1 |
|---|---|---|---|---|
| iteration | 91 | 100 | 131 | 140 |
| **iterations since last descent** | **1** | **10** | **1** | **10** |
| move | 0.01 | 0.01 | 0.02 | 0.02 |
| ‖Δρ‖₂ | 0.02358 | 0.009340 | 0.09008 | 0.08332 |
| RMS(Δρ) | 4.169e-4 | 1.651e-4 | 7.962e-4 | 7.365e-4 |
| max\|Δρ\| | 0.005810 | 0.003783 | 0.008189 | 0.007861 |
| **max\|Δρ\|/move** | **0.5810** | **0.3783** | **0.4094** | **0.3930** |
| ω₁ relative range (W=10) | 1.10e-3 | 2.57e-4 | 1.49e-3 | 2.22e-4 |
| M_nd | 13.4025 % | 13.3325 % | 23.3596 % | 23.0104 % |

**Did the new rule admit because the design settled, or because another
schedule artifact slipped through?**

At **160×20**, because the design settled: the ratio genuinely fell from 0.581
to 0.378 and the rule refused the earlier point on that ground.

At **320×40**, neither cleanly. The rule did not admit on a descent artifact —
the dwell forbids that — but it admitted at the first iteration the dwell
allowed, with M_nd still falling. The stop is **dwell-paced**, and the dwell is
paced by the ladder's descent spacing. Honest answer: the artifact was removed,
but maturity was not demonstrated at this mesh.

---

## §13 Gate results — candidate C1

**160×20:** G1 ✓ (100 < 600) · G2 ✓ (+0.0188 %) · G3 ✓ · G4 ✓ · G5 ✓ (gap 10) ·
G6 ✓ (0.00378 < 0.01 and 0.378 < 0.50) · G7 ✓ — **all pass**.

**320×40:** G8 ✓ (140) · G9 ✓ (+0.0224 %) · G10 ✓ · G11 ✓ · G12 ✓ (gap 10) ·
**G13 ✗** · G14 ✓ (−7.84 %) · G15 ✓ (2.22e-4).

**G13 fails on both of its clauses.** M_nd is 1.49 % below baseline, against the
25 % preregistered threshold; and the alternative clause — "demonstrated mature
stopping rather than schedule-triggered stopping" — is not met, because the
admission is dwell-paced and M_nd is still falling there.

C2 fails G13 identically (M_nd −2.04 %).

G13 is a **non-safety** gate (safety = G3/G4/G10/G11, all passed), and the
invariant is enforced at both meshes.

---

## The decisive measurement: the ladder, not the admission rule

Running the unstopped trajectory far past every admission point, **with the
ladder active**:

| 320×40, iteration | move | M_nd | ω₁ |
|---|---|---|---|
| 131 (production stop) | 0.02 | 23.3596 | 165.947 |
| 140 (C1) | 0.02 | 23.0104 | 165.984 |
| 200 | 0.005 | 22.7883 | 166.010 |
| 400 | 0.005 | 22.4181 | 166.055 |
| **600** | 0.005 | **22.0508** | 166.091 |
| *fixed move 0.04, converged at 216* | *0.04* | ***13.2824*** | *166.531* |

469 extra iterations under the ladder buy 1.3 points of M_nd. Holding the move
at 0.04 buys 10.1. **The ladder's descent to small moves freezes the design in a
gray state**, and no admission rule can undo that, because by the time admission
is even considered the move is already small.

This is the single most important result of the study, and it redirects the next
question: the lever is the **move schedule** (a class-C reconstruction choice
with no numeric value anywhere in the Du–Olhoff lineage), not the stopping test.

---

## Figures

`fig1_topology_comparison` · `fig2_maxabs` · `fig3_ratio` (the
invariant-enforcing quantity, with both thresholds drawn) · `fig4_objective_change` ·
`fig5_move` · `fig6_predicate_components` (every component of the predicate per
iteration) · `fig7_Mnd` · `fig8_omega1`. Move descents and all three admission
points are marked on every per-iteration panel.

## Data

`METRICS.json`; `runs/unstopped_*_iterations.csv` — 29 columns including every
predicate component for both candidates, so any admission decision can be
re-derived. `runs/*.mat` is not tracked (large, reproducible from `code/`).

## Deviations from the preregistration

1. **No 400×50** — correctly withheld, since G13 failed (§17 of the brief).
2. **`ar_state_at`/`ar_predicate` report `hist.omega`, which is pre-update.** The
   post-loop ω₁ used for gates G2/G9 was recomputed by repeating the solver's own
   terminal `assemble2D`+`eigSolve` at each admission design. This was not
   anticipated in the preregistration; it makes the ω₁ comparison stricter, not
   looser, and it reproduces the archived baseline ω₁ bitwise.
3. No thresholds were changed after seeing results; both preregistered candidates
   are reported.

## What was not done

No production file modified. No preset changed. No filter, p, q, mass model,
multiplicity, MMA, move level, stall signal/window/tolerance, projection,
density filter, eigen solver, objective or FE model altered. No 800×100. No
nine-mesh campaign. No candidate promoted.

---

# ADMISSION_RULE_CANDIDATE_PARTIAL

The invariant is enforced at both meshes and no safety gate fails, but **G13
fails at 320×40**: the rule delivers a 1.5 % M_nd reduction where 25 % was
required, and at that mesh the admission is dwell-paced rather than
maturity-demonstrated.

The candidate is a sound **correctness** fix — a move descent can no longer
manufacture convergence — obtained with no objective regression (ω₁ improves at
both meshes) and thresholds taken entirely from the literature and the solver's
own existing constants. It is **not** a grayness remedy, and the study
establishes why: with the ladder descending, the design freezes long before any
admission rule is consulted.

# KEEP_PRODUCTION_PRESET_PENDING_PROMOTION_REVIEW

No production file was changed by this task.
