# REPORT — what design-activity signal tells us a move stage has matured

**Offline analysis only.** No optimiser was run, no production code or preset was
touched, no previous diagnostic directory was modified. Where the recorded data
cannot answer a question in the brief, this report says so instead of estimating.

| | |
|---|---|
| Implementation | `analysis/OlhoffCurrent` |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` — **74/74 files verified** |
| Branch / HEAD | `benchmark-methodology-r2` / `7154d82` (**not** the stated `a1f2c6c`; see [`PROVENANCE.md`](PROVENANCE.md)) |
| Prior artifacts | **0 mismatches** across all three studies; **4 files missing** (see below) |
| Inputs | 10 per-iteration CSVs + `move_transition/METRICS.json` |
| Companion files | [`DATA_INVENTORY.md`](DATA_INVENTORY.md) · [`CANDIDATE_STATISTICS.md`](CANDIDATE_STATISTICS.md) · [`ARCHITECTURE_RECOMMENDATION.md`](ARCHITECTURE_RECOMMENDATION.md) · [`METRICS.json`](METRICS.json) |

---

## Two findings before the questions

**The per-element density history is gone.** All three prior studies built a full
`NE x nOuter` density matrix and saved it to `.mat`; `.mat` is git-ignored inside
`diagnostics/`, and every one of those files has been deleted. `move_stop` had
hashed its four in `FINAL_SHA256.txt`, so their loss is provable rather than
merely apparent; `admission_rule` and `move_transition` never listed theirs at
all. **Brief sections 6 (saturated-element persistence) and 7 (spatial maps), and
required figures 5–7, cannot be produced from recorded data.** Per the brief's
instruction, this is reported rather than repaired by rerunning an optimiser.

**The central hypothesis does not survive contact with the data — but a narrower
one does.** Every statistic that weights the loud tail of the increment
distribution (`max`, `RMS`, the participation number, high-threshold active
fractions) is ordered *backwards* across the two meshes: each reads **quieter**
at 320x40, the mesh where 4.8x more topology evolution remained. Only a
**low**-threshold active-set size orders them correctly.

---

## The measurement everything rests on

`baseline` and `fixedmove` are **bitwise identical** (max abs difference
`0.000e+00` in `Mnd`, `maxAbs`, `l2`) for iterations 1–78 at 160x20 and 1–129 at
320x40 — up to the iteration before production's first move descent. The
fixed-move arm is therefore an **exact counterfactual continuation** of
production at `move = 0.04`, not a similar run. That makes the following a
measurement:

| | 160x20, it. 78 | 320x40, it. 129 |
|---|---:|---:|
| `M_nd` at the descent | 13.494 | 23.451 |
| `M_nd` if `move=0.04` is held to the end | 12.275 | 13.282 |
| **evolution still to come** | **1.218 pts (9.0%)** | **10.168 pts (43.4%)** |
| `max(u)` | 0.99974 | 0.64707 |
| `RMS(u)` | 0.23134 | 0.05945 |
| `N_eff` (effective active elements) | 171.3 (5.4%) | 108.1 (0.84%) |
| `frac(u > 0.25)` | 0.07188 | 0.00828 |
| `frac(u > 0.025)` | 0.22125 | 0.25578 |

A rule *"descend when `S < theta`"* must **permit** at 160x20/78 and **block** at
320x40/129, which requires `S(160x20) < S(320x40)`. Five of the seven statistics
violate that before any threshold is chosen.

---

## The twenty questions

### 1. What element-level trajectory data are actually available?

**None.** No per-element `rho`, `Delta rho`, or element identity survives, at any
mesh, in any of the three studies. What survives per iteration is `max|drho|`,
`||drho||_2` (and `rms`, which is `l2/sqrt(NE)` — an identity verified to
5.8e-15, carrying no extra information), `move`, `stage`, `M_nd`, `gray`, `mid`,
`omega1`, `volume`, `beta`.

The one exception carries most of this study: **the four `move_stop` runs record
exact active-set counts** `sum(|drho_e| > tau)` for
`tau = {1e-4, 8.83883e-4, 1e-3, 1e-2}`. These are the only measured points of the
increment distribution anywhere in the record. Full detail in
[`DATA_INVENTORY.md`](DATA_INVENTORY.md).

Two exact reconstructions are available and are used throughout:

- the **participation number** `N_eff = (||drho||_2 / max|drho|)^2`, the effective
  number of elements carrying the increment — exact, and **move-free** (the move
  cancels identically, curing the non-invariance the previous study found in
  `r_rho`, its point 8);
- the **exact one-sided bound** `frac(u >= t) <= RMS(u)^2 / t^2`. It *caps* the
  active fraction; it cannot lower-bound it beyond `1/NE`. Against the exact
  counts it runs a median 9.4x loose (612 comparisons), so it is used only for
  genuine upper limits, never as a stand-in measurement.

### 2. Is `max(u)` dominated by a tiny exceptional population?

**Yes, and it is worse than that — `max(u)` is nearly uninformative.**

At the 160x20 descent, `N_eff = 171` of 3200 elements: 5.4% of the design
effectively carries the entire increment. By iteration 400 it is 95 (3.0%). The
previous study's snapshot (66/3200 at the bound, 2.1%) is the same phenomenon at
iteration 600.

But the decisive number is the rank correlation between `max(u)` and the topology
evolution actually remaining: **0.066 at 160x20**. `max(u)` sits at ~0.9995 for
the entire 400-iteration fixed-move run while `M_nd` falls from 99.5 to 12.3
(figures [1](figures/fig1_utilisation_distribution_160x20.png),
[9](figures/fig9_Mnd_with_activity.png)). And at *both* 160x20 production
descents `max(u)` was saturated (~0.999 at it. 79, ~0.996 at it. 90) while only
9.0% of evolution remained — **a saturated `max` does not even imply
immaturity.** It is not merely too sensitive; at the coarse mesh it carries
almost no maturity information in either direction.

### 3. Are those elements persistent or rotating?

**Not answerable from recorded data.** Element identity did not survive, so
Jaccard overlap between iterations, persistence duration, and birth/death rates
cannot be computed. No estimate is offered. Figure
[7](figures/fig7_active_population_size.png) shows the aggregate *size* of the
active population over time, which is what the surviving data support; it is
labelled on its face as a substitute for the requested persistence map, not as
one.

### 4. Where are they spatially located?

**Not answerable from recorded data.** No per-element values means no spatial
map, no clustering measure, no boundary or void/solid-interface association.
Figures [5](figures/fig5_distribution_at_descent_160x20.png) and
[6](figures/fig6_distribution_at_descent_320x40.png) instead show what *is* known
about the utilisation distribution at the descent — the exact measured points and
the exact upper bound — and say on their face that they are not the requested
spatial maps.

### 5. At production descent, what fraction of the design is still materially active?

Depends entirely on what "materially" means, and that dependence is the finding:

| | 160x20, it. 78 | 320x40, it. 129 |
|---|---:|---:|
| moving > 25% of the move (`u>0.25`) | 7.19% (230 elts) | **0.83%** (106 elts) |
| moving > 2.5% of the move (`u>0.025`) | 22.1% (708 elts) | **25.6%** (3274 elts) |
| effective participation `N_eff/NE` | 5.4% | 0.84% |

At the fine mesh **a quarter of the design is still creeping** while almost
nothing moves fast. Production descended into that.

### 6. How does that differ between 160x20 and 320x40?

They differ in *opposite directions depending on the threshold*, which is why no
tail-weighted statistic can work. Going from coarse to fine, the fast-moving
population shrinks 8.7x (7.19% -> 0.83%) while the slow-drifting population grows
slightly (22.1% -> 25.6%) and grows 4.6x in absolute count (708 -> 3274). The
fine mesh is *quieter* by every loud measure and *more extensively active* by
every quiet one.

### 7. Do high percentiles behave more robustly than `max`?

**Untestable.** P75/P90/P95/P97.5/P99 are not recoverable per-iteration.
`mt_spatial.m` computed quantiles `[0.50 0.90 0.99 1.00]` every iteration but
`mt_export.m` never wrote them to CSV; only single stop-iteration snapshots
survive in `METRICS.json`. This family can be neither recommended nor dismissed
on evidence, and the report does not pretend otherwise.

Indirect evidence is unfavourable but not decisive: P99 is a tail statistic, and
every tail statistic that *is* measurable is ordered backwards.

### 8. Does an active-fraction statistic behave more robustly?

**A low-threshold one does; a high-threshold one does not.** `frac(u>0.25)` is
inverted (0.0719 vs 0.0083). `frac(u>0.025)` is correctly ordered (0.2213 vs
0.2558), as are `frac(u>0.0221)` and `frac(u>0.0025)`.

But the *fraction* normalisation is not the right one — see Q11. The
correctly-ordered statistic with real margin is the **count**: 708 vs 3274, a
4.6x separation, against the fraction's 1.16x.

### 9. Does RMS / global activity hide localized meaningful evolution?

**It fails, but not by that mechanism.** `RMS(u)` is inverted (0.2313 vs 0.0595),
and its rank correlation is decent within a mesh (0.62 / 0.78). It does not hide
a moving front; it measures the wrong thing. RMS measures **how fast** the design
is moving, and across meshes speed is *anti-correlated* with **how much
evolution remains**. The 320x40 design at iteration 129 was moving slowly and
steadily and had 87 more iterations of real topology change ahead of it.

### 10. Which statistic families are clearly unsuitable?

- **`max(u)`** — rank correlation 0.066 at 160x20; inverted; saturated at descents
  that were in fact nearly mature.
- **`RMS(u)`** — inverted; measures speed, not extent.
- **High-threshold active fraction/count (`u > 0.25`)** — inverted; does not
  collapse across meshes (best-fit exponent -0.19, residual 0.90).
- **`N_eff`, the participation number** — inverted; does not collapse
  (exponent -0.20, residual 0.78). This one deserves an explicit note because it
  is the most attractive candidate on paper: exactly computable and *provably
  move-invariant*, which is precisely the defect the previous study identified in
  `r_rho`. It still fails. Its only admissible threshold window is **2% wide**
  (91.98..93.74) — two noisy curves crossing, which is exactly the coincidence
  the brief warns against, not a margin.

### 11. Which statistic family deserves a real optimization experiment next?

**The size of the low-activity-threshold active set** — elements whose increment
exceeds a small absolute threshold (`|drho| ~ 1e-3`, i.e. `u ~ 0.025` at
`move=0.04`) — **under a mesh normalisation that is neither a raw count nor an
area fraction.**

Its credentials: correct ordering with a 4.6x margin; within-mesh rank
correlation with remaining evolution of **0.687 (160x20) and 0.987 (320x40)**; and
it is the only family whose curves genuinely collapse across meshes at matched
maturity (figure [8](figures/fig8_candidate_statistics_across_meshes.png)).

**The normalisation, and the section-11 answer.** Writing the statistic as
`activeCount / NE^alpha`, two independent methods agree:

| method | `alpha` for `|drho|>1e-3` |
|---|---|
| threshold-window overlap (`c_min` 0.99, `D` 10) | 0.513 .. 0.929 |
| mesh-collapse fit at matched `M_nd` maturity | **0.808** (band 0.71–0.90, residual 0.12) |

**Constant count (`alpha=0`) is excluded outright.** Constant fraction
(`alpha=1`) sits at the upper edge. Interface-length scaling (`alpha=0.5`) sits at
the lower edge and is excluded by the collapse fit. So the mature active set
scales as roughly `NE^0.8` — between an interface length and an area fraction
(figure [11](figures/fig11_normalisation_exponent.png)).

**This is one exponent fitted to two meshes.** The bands are fit sensitivity, not
statistical confidence; two points cannot validate a power law. That the `1e-4`
and `1e-3` thresholds disagree (0.96 vs 0.81) by more than either band is itself
a sign the single-exponent model is being strained.

### 12. Is there enough evidence to preregister ONE next transition candidate?

**Yes — as a candidate to be *calibrated*, not one asserted to work.** Four
families are eliminated on evidence, one survives every test the data support,
and the one open quantity (`alpha`) is *not resolvable offline*: it needs a third
mesh, which needs an optimisation run. Offline evidence is exhausted, so the next
step is an experiment.

### 13. State its semantic form (do not run it)

```
move.policy                                    = 'ladder'
move.levels                                    = [0.04 0.02 0.01 0.005]     (unchanged)
move.transition.type                           = 'designActivity'
move.transition.activity.statistic             = 'activeCount'
move.transition.activity.incrementThreshold    = 1e-3          # on |drho_e|, absolute
move.transition.activity.normalisation         = 'meshPower'
move.transition.activity.normalisationExponent = TO BE DETERMINED BY THE EXPERIMENT
move.transition.activity.threshold             = TO BE DETERMINED
move.transition.activity.persistence           = 10
move.transition.dwell                          = 10            (production's implicit guard, made explicit)
```

Three conditions that experiment must satisfy, all of them consequences of what
went wrong in the studies that preceded it:

1. **It must persist per-element density history and hash it in the manifest.**
   Three consecutive studies destroyed the data that would have answered
   questions 3, 4 and 7 here. This is the single highest-value change, and it
   costs one line and some disk.
2. **It must run at least three meshes.** Its primary purpose is to *measure*
   `alpha`, not to demonstrate an `M_nd` improvement. Two meshes cannot pin an
   exponent, and 160x20/320x40 alone will reproduce this study's ambiguity.
3. **It must express the rule as configuration, not a solver copy** — see
   [`ARCHITECTURE_RECOMMENDATION.md`](ARCHITECTURE_RECOMMENDATION.md).

Per brief sec. 14, offline replay establishes only that a rule *would have fired
at iteration k on the recorded trajectory*. Once a transition changes, the
trajectory changes; nothing here predicts the topology such a run would reach.

### 14. If no, what evidence is missing?

Answered as a matter of record even though Q12 is yes. Missing: per-element
history at every mesh (for the percentile family, and for questions 3–4 which
remain open); a third mesh (for `alpha`); and active counts at more than the four
thresholds `move_stop` happened to record for another purpose.

### 15. Does `omega1` stability remain a poor proxy for topology maturity?

**Yes — confirmed and quantified.** Holding `move = 0.04` from the production
descent to the end of the run:

| | `omega1` rel. change | `M_nd` rel. change | ratio |
|---|---:|---:|---:|
| 160x20, it. 78 -> 400 | 0.233% | 9.03% | **38.7x** |
| 320x40, it. 129 -> 216 | 0.375% | 43.36% | **115.8x** |

At the fine mesh `omega1` moves by under four parts in a thousand while the
topology completes 43% of its remaining discreteness evolution (figure
[10](figures/fig10_omega1_with_activity.png)).

One nuance worth recording, because it cuts against a naive reading: `omega1`'s
*rank* correlation with remaining evolution (0.617 / 0.923) is comparable to the
activity statistics'. The defect is one of **scale, not ordering** — `omega1`
tracks maturity in rank but compresses it into a range ~100x smaller, so any
threshold on `omega1` stability is hypersensitive and mesh-fragile. It is a poor
*proxy*, not a meaningless signal.

### 16. Does this change the conclusion that production move descent is premature?

**It confirms it at 320x40 and refutes the 160x20 evidence for it.** This is a
correction to a premise stated as established.

- **320x40: confirmed, and it is the real phenomenon.** 43.4% of the remaining
  `M_nd` evolution was still ahead when production descended at iteration 130.
- **160x20: the descent was roughly on time.** Only 9.0% remained. The premise's
  established fact 3 cites "~99.9% of the permitted move at 160x20" as evidence
  of prematurity. That inference does not hold: move utilisation was ~99.9%
  *because a handful of elements were at the bound*, while the design as a whole
  was ~99% matured (`M_nd` 13.494 vs a fixed-move limit of 12.275). **High move
  consumption at 160x20 was not evidence of immaturity**, and `max(u)`'s rank
  correlation of 0.066 there is the direct measurement of why.

The corrected statement: premature descent is a **fine-mesh** phenomenon, and the
coarse-mesh symptom that motivated the previous study was a false positive
produced by the very statistic under test.

### 17. Does anything now justify projection?

**No.** Nothing in this analysis bears on projection. It remains scope-locked and
unjustified by this evidence.

### 18. Does anything now justify density filtering?

**No.** Unchanged. The analysis concerns only the move ladder's transition rule.

### 19. Does anything justify changing `R = 0.06*b`?

**No.** The filter radius is untouched by this evidence and remains scope-locked.

One observation that must *not* be misread as support: the finding that the
mature active set scales as ~`NE^0.8` rather than `NE^1.0` is a statement about
the *design increment*, not about the filter's length scale. It is not a
filter-radius result and should not be cited as one.

### 20. What should the clean move-controller configuration API look like?

See [`ARCHITECTURE_RECOMMENDATION.md`](ARCHITECTURE_RECOMMENDATION.md).
In brief: separate **levels**, **transition rule** and **transition statistic**;
make the transition a discriminated union on `move.transition.type` with enforced
incompatible-field validation; promote the currently-invisible dwell guard to a
real field; keep every field's A/B/C/D provenance (all of it is **C — pure
reconstruction**; Du & Olhoff bound the increment only by the box (25f)).

The concrete defect it repairs: `move_transition`'s `CONFIG_DIFF.json` reports
that its two arms differed **only in `runtime.name`**, because the experimental
factor lived in a two-line solver copy rather than in configuration. Under the
proposed schema that arm difference is five substantive fields and the failure is
structurally impossible.

---

## Figures

| # | file | note |
|---|---|---|
| 1 | [`fig1_utilisation_distribution_160x20.png`](figures/fig1_utilisation_distribution_160x20.png) | percentiles **not recoverable**; shows max, RMS, `N_eff` |
| 2 | [`fig2_utilisation_distribution_320x40.png`](figures/fig2_utilisation_distribution_320x40.png) | as above |
| 3 | [`fig3_active_fractions_160x20.png`](figures/fig3_active_fractions_160x20.png) | **exact** recorded counts |
| 4 | [`fig4_active_fractions_320x40.png`](figures/fig4_active_fractions_320x40.png) | **exact** recorded counts |
| 5 | [`fig5_distribution_at_descent_160x20.png`](figures/fig5_distribution_at_descent_160x20.png) | **substitute** — spatial map impossible |
| 6 | [`fig6_distribution_at_descent_320x40.png`](figures/fig6_distribution_at_descent_320x40.png) | **substitute** — spatial map impossible |
| 7 | [`fig7_active_population_size.png`](figures/fig7_active_population_size.png) | **substitute** — persistence map impossible |
| 8 | [`fig8_candidate_statistics_across_meshes.png`](figures/fig8_candidate_statistics_across_meshes.png) | the decisive cross-mesh comparison |
| 9 | [`fig9_Mnd_with_activity.png`](figures/fig9_Mnd_with_activity.png) | `M_nd` with activity statistics |
| 10 | [`fig10_omega1_with_activity.png`](figures/fig10_omega1_with_activity.png) | `omega1` with activity statistics |
| 11 | [`fig11_normalisation_exponent.png`](figures/fig11_normalisation_exponent.png) | the `alpha` result, both methods |

Production move descents are marked on every iteration-axis figure. Figures 5, 6
and 7 stand where the brief asked for spatial and persistence maps; each states
on its face what it is and what it is not. No topology was thresholded for
presentation — no topology could be, since no density field survives.

---

## Verdict

    ROBUST_MOVE_ACTIVITY_STATISTIC_NARROWED

Four families are eliminated on evidence (`max`, `RMS`, participation number,
high-threshold active fraction). One survives every test the surviving data
support: the **low-threshold active-set size under a mesh normalisation strictly
between constant count and constant fraction**. It is narrowed, not identified,
because its normalisation exponent cannot be pinned by two meshes.

    NEXT_TRANSITION_EXPERIMENT_READY_FOR_PREREGISTRATION

Offline evidence is exhausted — the open quantity requires a third mesh, and a
third mesh requires an optimisation run. The candidate's semantic form is stated
in Q13 and has **not** been run.
