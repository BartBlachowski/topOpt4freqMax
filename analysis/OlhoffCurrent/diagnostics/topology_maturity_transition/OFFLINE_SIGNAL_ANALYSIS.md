# OFFLINE_SIGNAL_ANALYSIS — Phase A

Can a *topology-maturity* observable be built out of temporal density evolution,
such that one threshold means the same thing at every mesh?

**No.** Every candidate in the brief's mandated family either reads the **same
value** at two states of demonstrably opposite maturity, or orders those states
**backwards**. The reason is structural, not statistical, and it is stated in §5.

Nothing here is fitted, and no solver was run: this is arithmetic on retained,
hash-valid trajectories.

---

## 1. Data actually used

From `diagnostics/move_transition/runs/` (36/36 hash-valid):

| arm | meaning |
|---|---|
| **ARM P** | production — move ladder descends on the β bound-variable stall detector |
| **ARM U** | identical configuration, move **held at ladder stage 1 (0.04)** until its own rule fires |

ARM U is the fixed-move counterfactual: it shows what the design *would have gone
on to do* had production not descended. Verified here, not assumed — ARM P and
ARM U are **bitwise identical** column-for-column until:

| mesh | first differing column | production first descent `kP` |
|---|---|---|
| 160×20 | **79** | 79 |
| 320×40 | **130** | 130 |

which independently reproduces the `move_transition` study's transition sets
{79, 90, 101} and {130, 141, 152}.

`W = 10` was used throughout and **never tuned** (§A2): it is the window already
present in `olh.move.limit` (`move.continuation.window`, default 10) and in both
preceding preregistered studies.

**400×50 contributes nothing to this analysis because no 400×50 trajectory
exists** — see `PROVENANCE.md` §3.2.

---

## 2. Ground truth: how mature *was* the design when production descended?

Two independent labels, both computed from ARM U.

**(a) Topology basis** — the fraction of the whole run's binarisation that was
still outstanding at `kP`:

```
remaining = ( M_nd(kP) − M_nd(mature) ) / ( M_nd(1) − M_nd(mature) )
```

**(b) Displacement basis** — L1 endpoint distance to the mature state, and
accumulated L1 path length, both normalised by their run totals.

| | 160×20 | 320×40 |
|---|---|---|
| M_nd(1) | 99.459 % | 99.614 % |
| M_nd at `kP` | 13.390 % | 23.322 % |
| M_nd at mature reference | 11.458 % | 13.304 % |
| **(a) remaining, topology basis** | **2.20 %** | **11.61 %** |
| (b) remaining, L1 endpoint | 15.23 % | 16.23 % |
| (b) remaining, L1 path | 64.48 % | 13.21 % |
| M_nd still on the table | 1.93 pts | **10.02 pts** |
| ω₁ forgone by descending | +0.176 % | +0.370 % |

Label (a) is the one that matters — it is the quantity the controller exists to
protect — and it separates the two states by a factor of **5.3×** in the
direction the brief expects: **160×20 at its descent is mature; 320×40 is not.**

Label (b) is reported for completeness and is *itself* evidence of the problem:
on the endpoint basis the two states look identical (15.2 % vs 16.2 %), and on
the path basis they invert (64.5 % vs 13.2 %). A displacement-based notion of
maturity is already unstable across these two meshes **before any statistic is
computed** — for the reason in §5.

Mature reference: 320×40 uses ARM U's own first descent, iteration **214**.
160×20 uses iteration 600 — because **ARM U at 160×20 never descends at all**,
which is itself the finding of §5.

---

## 3. The candidate family (§A8)

All computed on ARM U, with `W = 10`, `Δρ(j) = ρ(j) − ρ(j−1)`:

| # | statistic | definition |
|---|---|---|
| 1 | `inst_max` | max_e \|Δρ_e(k)\| |
| 2 | `inst_rms` | ‖Δρ(k)‖₂ / √NE |
| 3 | `DW_l1`, `DW_l2` | windowed **endpoint** distance ‖ρ(k) − ρ(k−W)‖ , L1/NE and L2/√NE |
| 4 | `CW_l1` | windowed **accumulated** path Σ‖Δρ(j)‖₁ / NE |
| 5 | `RW`, `RW_end` | `CW_l1`, `DW_l1` divided by `W·move` — dimensionless, in [0,1] |
| 5 | `CW_max`, `CW_q90`, `med_DW`, `trim_DW` | max / 90th pct / median / 20 %-trimmed mean of the per-element window motion |
| 5 | `frac_half`, `fracNet25` | fraction of elements exceeding 0.5·W·move (path) / 0.25·W·move (net) |
| 5 | `DW_unsat` | mean net motion **excluding** elements saturated over the window — the targeted fix for outlier domination |
| — | `coher` | `DW_l1 / CW_l1` — directedness of the motion, in [0,1] |

`RW`/`RW_end` carry the normalisation the brief asks for: a mean over elements
(no fitted NE exponent, §A10) divided by the step bound (§A9).

---

## 4. Result — the family fails

### 4.1 At the two known maturity states

Ground truth: **2.20 % remaining (160×20 @ 79)** vs **11.61 % remaining (320×40 @ 130)**.

| statistic | 160×20 @ 79 | 320×40 @ 130 | ratio | verdict |
|---|---|---|---|---|
| `RW_end` | 0.026068 | 0.025057 | **1.04** | **blind** |
| `DW_unsat` | 0.010615 | 0.010023 | **1.06** | **blind** |
| `med_DW` | 0.0000282 | 0.0000247 | **1.14** | blind, and at the noise floor |
| `inst_max` | 0.039978 | 0.026883 | 1.49 | **saturated** at the bound at 160×20 |
| `CW_l1` / `RW` | 0.033018 / 0.082545 | 0.010060 / 0.025149 | 3.28 | **backwards** |
| `inst_rms` | 0.009783 | 0.002357 | 4.15 | **backwards** |
| `trim_DW` | 0.002442 | 0.004729 | 0.52 | right way, but see 4.2 |
| `coher` | 0.315798 | 0.996311 | 0.32 | right way, but fails §A9/§A10 — see 4.3 |

The three statistics that are simultaneously mesh-normalised and
outlier-robust — exactly the ones §A10 demands — are **blind**: they read the
same number at a 2.2 %-remaining state and an 11.6 %-remaining state. Every
statistic that *does* move reads **larger at the more mature mesh**.

`DW_unsat` is the decisive entry. It is the surgical remedy for §A10's
isolated-element criterion — the saturated population is simply deleted before
averaging — and it still returns 1.06.

### 4.2 Across the whole maturity range

Sharper test: at *matched* ground-truth remaining-evolution levels, a usable
statistic must take *comparable values at both meshes*, or a single threshold
cannot mean one thing. Ratio (160×20 value)/(320×40 value):

| statistic | 40 % | 30 % | 20 % | 15 % | 10 % | 5 % | swing |
|---|---|---|---|---|---|---|---|
| `RW_end` / `DW_l1` | 1.58 | 1.92 | 4.85 | 0.78 | **0.12** | 0.28 | **40×** |
| `DW_unsat` | 1.56 | 1.94 | 4.49 | 0.80 | **0.12** | 0.29 | **37×** |
| `trim_DW` | 1.90 | 2.42 | 7.30 | 0.35 | **0.09** | 0.19 | **81×** |
| `CW_l1` / `RW` | 1.79 | 2.19 | 6.33 | 3.20 | 1.87 | 1.22 | 5× |
| `CW_q90` | 1.75 | 2.08 | 6.27 | 1.85 | 0.59 | 0.52 | 12× |
| `inst_rms` | 2.63 | 2.16 | 4.42 | 4.06 | 2.52 | 1.88 | 2.4× |
| `coher` | 0.88 | 0.88 | 0.77 | 0.24 | **0.07** | 0.23 | 13× |
| `fracNet25` | 1.77 | 2.17 | 12.23 | 1.66 | **0.00** | 0.08 | ∞ |
| `frac_half` | 5.12 | 4.23 | 41.93 | 186 | 130 | 10 | 45× |
| `inst_max` | 1.01 | 1.00 | 1.14 | 1.47 | 1.57 | 1.05 | *pinned at 0.0400 at 160×20* |

Not one candidate holds a stable ratio. Most are **non-monotone and cross 1.0**,
so no threshold is even consistently conservative — the same value means "keep
going" at one mesh and "descend" at the other, and which is which flips with
maturity. This is with **two** meshes; the brief requires three.

### 4.3 The one statistic that orders correctly still fails

`coher = D_W/C_W` separates the states cleanly (0.32 vs 0.996) and is the only
candidate that does. It is nevertheless inadmissible under §A10, on two counts,
both verified:

* **It is driven by the outlier population.** At 160×20 the low value is
  *produced by* the 56–250 elements that stay pinned at the move bound: they
  inflate `C_W` without contributing to `D_W`. A small saturated minority is
  what makes it say "mature" — precisely the failure mode §A10 excludes, merely
  with the sign reversed relative to `max|Δρ|/move`.
* **It is satisfied by reducing the move (§A9).** At 320×40 it sits at
  0.98–0.99 while `move = 0.04`, then collapses to **0.24–0.44** once the ladder
  descends and the motion becomes small-amplitude noise. Descending would
  immediately re-authorise descending — the self-fulfilling coupling the brief
  forbids.

---

## 5. Why the family fails — the structural reason

The two meshes are not at different points on one trajectory. At `move = 0.04`
they are in **qualitatively different dynamical regimes**. The 2-step / 1-step
displacement ratio `d₂/d₁` is the diagnosis (figure `F1`); coherent directed
motion gives ≈ 2, a period-2 cycle gives < 1:

| iteration | 160×20 `d₂/d₁` | elements at bound | 320×40 `d₂/d₁` | elements at bound |
|---|---|---|---|---|
| 100 | **0.37** | 210 | **2.20** | 2 |
| 150 | **0.40** | 250 | **1.96** | 0 |
| 200 | **0.34** | 114 | **1.88** | 0 |
| 300 | **0.36** | 90 | 2.15 | 0 |
| 400 | **0.30** | 78 | 0.80 | 0 |
| 598 | **0.35** | 71 | 2.44 | 0 |

* **160×20 is a period-2 limit cycle.** It goes forward and comes back:
  `d₂ ≈ 0.35 d₁`. `max|Δρ|` is **0.0400 at every one of 600 iterations** and
  56–250 elements never leave the bound. It never settles, which is why ARM U
  never descends there and why the study's `max|Δρ|/move` candidate produced
  `CAP_HIT`.
* **320×40 is a coherent convergent descent.** Displacements add
  (`d₂ ≈ 2 d₁`), and after iteration ~100 **zero** elements are at the bound.

So at 160×20 the residual motion is **large but unproductive**; at 320×40 it is
**small but productive**. Any statistic measuring the *magnitude* of density
motion therefore reads high at the mature mesh and low at the immature one —
backwards, necessarily, not by accident. Any statistic measuring *directedness*
must divide by that same magnitude and so inherits the saturated minority that
generates it.

A single scalar summary of density motion cannot be calibrated simultaneously
across a limit cycle and a convergent descent. That is the finding.

---

## 6. Disposition against §A10

| §A10 requirement | best candidate's status |
|---|---|
| orders the known maturity states sensibly | **FAIL** — blind (1.04–1.14×) or inverted; only `coher` orders, and it fails below |
| not dominated by tiny outlier populations | FAIL for `inst_max`, `CW_max`, `coher`; `DW_unsat` passes but is blind |
| detects persistent low-amplitude coherent evolution | passes for `D_W`, `C_W` |
| transparent mesh normalisation | passes for `RW`, `RW_end`, `DW_unsat` (mean over elements, no fitted exponent) |
| not an objective-progress proxy | passes for all |
| not a grayness/binarity objective | passes for all (§A4 respected — no candidate uses M_nd, gray fraction or ρ(1−ρ); M_nd appears only as an evaluation *label*) |
| not automatically satisfied by reducing move | FAIL for `coher`; passes for `RW_end` |
| no fitted arbitrary NE exponent | passes for all except the raw counts |

No statistic satisfies the conjunction.

> ### `TOPOLOGY_MATURITY_SIGNAL_NOT_IDENTIFIED`

Per §A10 this is terminal: Phase B was not entered, no threshold was chosen, and
no rule was preregistered.

---

## 7. What this does *not* license

* **No topology-aware weighting was implemented.** §A4 permits it to be *discussed*
  as a future hypothesis once plain temporal measures demonstrably fail, and they
  now demonstrably have — but it was not implemented, not preregistered, and not
  tested here, exactly as §A4 requires.
* **Nothing here justifies projection**, or any change to `R = 0.06·b`, `p`, the
  mass interpolation, `q`, the filter, the MMA variant, or the ladder values.
  The failure is in the *observability* of stage maturity, and no formulation
  change addresses that.
* **The move ladder was not changed.** `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`
  stands, now with a mechanism for *why* the replacement signal was not found.

---

## 8. A hypothesis for the next task — measured, not adopted

The regime split in §5 suggests "has the design finished exploiting this move
level?" may not be a *magnitude* question at all. At 160×20 "finished" means
**entered the limit cycle**; at 320×40 it means **stopped descending coherently**.
A signal detecting limit-cycle *onset* — persistent `d₂/d₁ < 1` — is a different
object from a motion threshold, so it was measured before being written down:

| criterion: `d₂/d₁ < 1` held for N consecutive iterations | 160×20 | 320×40 |
|---|---|---|
| N = 5 | 75 | 298 |
| **N = 10** | **80** | **402** |
| N = 20 | 90 | 412 |
| production first descent | **79** | **130** |
| fixed-move (ARM U) first descent | never | 214 |
| `d₂/d₁` at the production descent | **0.546** | **2.005** |

**At 160×20 the limit cycle begins at iteration 80 and production descends at 79 —
a one-iteration difference.** That is a real explanation of a previously
unexplained fact: 160×20 is unharmed by the β-stall trigger because at that mesh
the trigger happens to fire at the exact moment coherent progress ends. It is
coincidence, not correctness — β stalls there *because* the design started
oscillating — but it explains why the coarse-mesh reconstruction anchor survives
a trigger that is demonstrably wrong at 320×40, where `d₂/d₁ = 2.005` at the
descent: production descends while the design is still moving perfectly
coherently.

**The hypothesis is nevertheless untestable on existing data.** At 320×40 the
fixed-move arm holds `move = 0.04` only until **214**, and it is still coherent
there (`d₂/d₁ ≈ 2`). Every 320×40 sample beyond 214 was taken at `move ≤ 0.02`,
so the `402` in the table is **not** a fixed-move measurement and must not be read
as one. Testing a limit-cycle-onset rule needs a 320×40 arm with the move locked
at 0.04 well past 214, and a 400×50 arm that does not exist at all.

Recorded as a hypothesis. **Not implemented, not preregistered, not adopted** —
per §A4 and the brief's one-candidate-per-task rule. It also inherits the §A9
question (a limit cycle at `move = 0.04` says nothing about `move = 0.02`) which
would have to be answered before any preregistration.
