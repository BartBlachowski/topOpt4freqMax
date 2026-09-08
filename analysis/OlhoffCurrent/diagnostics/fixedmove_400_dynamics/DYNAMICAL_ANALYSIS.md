# DYNAMICAL_ANALYSIS — 400×50 under fixed move 0.04

**The mechanism does not generalize.** 400×50 never enters the cancelling regime
that 160×20 and 320×40 both reach. It *converges* instead — motion amplitude
decays by a factor of ~19 while the direction stays coherent to `cosθ = +0.997`.

At 400×50 the tested dynamical observables are **blind** to the very distinction
they were introduced to detect.

All definitions are those frozen in `dynamical_regime/PREREGISTRATION.md` §§2–4,
used here by **direct reuse of that study's code**, not by re-typing.

---

## 1. The run

| | RUN C |
|---|---|
| mesh / policy | 400×50, **fixed move 0.04**, unstopped |
| cap | 1200 (preregistered) |
| **outcome** | **1200 outer, `CAP_HIT`** |
| ω₁ final | 166.157147836 |
| M_nd final | 15.2243 % |
| volume | 0.4999992 |
| wall (1 thread) | 6104 s |
| **native stop would have fired at** | **369** |

`CAP_HIT` is reported as `CAP_HIT`. The run continued 831 iterations past the
point the inherited native rule would have stopped it.

### Common prefix vs the retained production arm — bitwise

| quantity, `k = 1 … 137` | max difference |
|---|---|
| ρ (elementwise) | **0** |
| ω₁, ω₂, M_nd, volume, β | **0**, all |
| first differing iteration | **138**, exactly as preregistered |

Production holds `move = 0.04` through 137 and descends **at** 138; the two arms
must and do diverge there. The counterfactual is valid, so the Phase 11
attribution below is causal.

---

## 2. The primary result: no cancelling transition exists at 400×50

Preregistered classifier (trailing-20 medians; `PERIOD2` = median `q2 < 1` **and**
median `cosθ < 0`, sustained 20 iterations):

| | 400×50 fixed move |
|---|---|
| `PERIOD2` onset | **none** |
| `COHERENT` onset | 20 |
| label counts | COHERENT **1181**, OTHER 19, PERIOD2 **0** |
| **minimum of median `cosθ` over all 1200 iterations** | **+0.914** (at k = 92) |
| minimum of median `q2` | **1.918** (at k = 77) |
| iterations with median `cosθ < 0` | **0** |
| iterations with median `q2 < 1` | **0** |
| longest run satisfying both | **0** (20 required) |
| undefined `q2` / `cosθ` samples | 0 / 0 |

This is not a marginal miss. Median `cosθ` never comes within 0.9 of the
threshold, and `net/path` never falls below ~0.9.

### This is not a cap artifact — the design converged

| k | 138 | 369 | 600 | 1199 |
|---|---|---|---|---|
| `max|Δρ|` | 2.098e-2 | 1.033e-2 | 2.387e-3 | **1.115e-3** |
| `max|Δρ|/move` | 0.524 | 0.258 | 0.060 | **0.031** |
| `‖Δρ‖₂` | 2.729e-1 | 1.241e-1 | 1.586e-2 | **1.198e-2** |

Motion amplitude decays monotonically by ~19×; `‖Δρ‖₂` ends a factor of **10
below** the native tolerance of 0.125. A limit cycle requires sustained
amplitude. Running longer would decay further, not oscillate. The failure to
classify is a property of the trajectory, not of the cap.

### Terminal states of the three fixed-move arms — the finest mesh is the opposite

| mesh | `max|Δρ|/move` | `‖Δρ‖₂` | `cosθ` | `net/path` | `revFrac` |
|---|---|---|---|---|---|
| 160×20 | **0.9993** | 3.46e-1 | **−0.951** | **0.090** | 0.724 |
| 320×40 | **0.6966** | 9.75e-2 | **−0.965** | **0.132** | 0.878 |
| **400×50** | **0.0308** | **1.19e-2** | **+0.997** | **0.992** | 0.134 |

(medians over each arm's last 100 iterations)

160×20 and 320×40 end in a full- or substantial-amplitude period-2 limit cycle.
**400×50 ends converged.**

---

## 3. Useful evolution still ends — but nothing dynamical marks it

| k | 138 | 369 | 400 | 547 | 600 | 1000 | 1200 |
|---|---|---|---|---|---|---|---|
| M_nd % | 32.318 | 16.159 | 15.488 | **15.049** | 15.054 | 15.174 | 15.224 |
| ω₁ | 162.877 | 166.365 | 166.418 | — | 166.278 | 166.183 | 166.158 |

M_nd falls to its minimum **15.0485 % at k = 547**, then drifts *upward*; ω₁
peaks at **166.4216 at k = 394** and then declines. Useful topology evolution
therefore ends around **k ≈ 400–550** — and across that entire transition the
dynamical observables register nothing:

| state | M_nd | `cosθ` | `net/path` | `max|Δρ|/move` | `‖Δρ‖₂` |
|---|---|---|---|---|---|
| **premature** (k = 138) | 32.318 | **0.995** | **0.977** | 0.524 | 2.73e-1 |
| **mature** (k = 1000) | 15.174 | **0.998** | **0.990** | 0.057 | 1.29e-2 |

`cosθ` and `net/path` are **indistinguishable** between a 32 %-gray premature
state and a 15 %-gray mature one. What *does* separate them at 400×50 is
**amplitude** — the very quantity the `topology_maturity_transition` study
refuted at 160×20, where it is pinned at the bound forever.

**The two candidate families have exactly complementary blind spots:**

| observable | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| cancellation (`cosθ`, `net/path`) | **works** | **works** | **blind** |
| amplitude (`max|Δρ|/move`, `‖Δρ‖₂`) | **blind** (pinned at 1.00) | works | **works** |

Neither generalizes across all three meshes.

---

## 4. Saturation robustness (Phases 6, 15)

`boundFrac = 0.00000` throughout RUN C — not one element of 20 000 touches the
move bound at any iteration. Consequently `cosθ_unsat ≡ cosθ_raw` and
`net_path_unsat ≡ net_path_raw` exactly, at every iteration.

So the 400×50 result **cannot** be a saturation artifact in either direction:
there is no saturated population to create or to mask a signal. Revisiting all
three meshes:

| mesh | `boundFrac` at the decisive event | raw vs unsaturated |
|---|---|---|
| 160×20 (onset 81) | 0.0556 | `cosθ` −0.833 → **−0.072** (sign lost at the instant); established regime −0.93 → −0.55 (sign kept) |
| 320×40 (onset 253) | **0.0000** | identical |
| 400×50 (no onset) | **0.0000** | identical |

The cross-mesh statement that survives saturation removal is therefore only:
*at 320×40 the cancelling regime is real and bound-free.* At 160×20 the onset
instant does not survive; at 400×50 there is nothing to survive.

---

## 5. Spatial structure (Phase 16)

Reversal fraction and its composition at four stages of RUN C (`F11`):

| stage | k | `revFrac` | `boundFrac` |
|---|---|---|---|
| production descent | 138 | 0.425 | 0.0000 |
| native stop would fire | 369 | — | 0.0000 |
| M_nd plateau | 600 | — | 0.0000 |
| terminal | 1199 | 0.135 | 0.0000 |

Reversal *falls* from 0.43 to 0.13 as the run matures — the opposite of 160×20
(0.36 → 0.82) and 320×40 (0.44 → 0.90). What reversal remains at 400×50 is
confined to void elements jittering at ρ_min; it is not member-following, not
interface-localised in any growing sense, and not saturation-driven. There is no
spatially coherent oscillation to map.

---

## 6. Objective vs dynamical maturity (Phase 17)

Native-stop replay applied to each **fixed-move** arm (`tol = 0.05·√(NE/3200)`):

| mesh | tol | native stop fires | cancellation onset | M_nd at native | M_nd at onset | best M_nd |
|---|---|---|---|---|---|---|
| 160×20 | 0.050 | **never** | **81** | — | 13.289 | 11.356 |
| 320×40 | 0.100 | **216** | **253** | 13.282 | 13.028 | 13.003 |
| 400×50 | 0.125 | **369** | **never** | 16.159 | — | 15.049 |

This is the sharpest statement of the split. The **native (objective/amplitude)
rule** detects maturity at 320×40 and 400×50 but never fires at 160×20. The
**cancellation rule** detects it at 160×20 and 320×40 but never fires at 400×50.
Each covers exactly the two meshes the other partly misses, and they overlap only
at 320×40.

Refinement therefore does **not** progressively separate objective stationarity
from dynamical maturity in one direction. It changes the *kind* of terminal state:
coarse meshes end in a bound-pinned limit cycle where the objective never settles;
the finest mesh ends in ordinary convergence where the objective settles and
nothing oscillates.

---

## 7. What production abandoned at 400×50 (Phase 11)

Valid because the prefix is bitwise identical through 137.

| | production (RUN A) | fixed move (RUN C) |
|---|---|---|
| stops at | 139 (`CONVERGED`) | 1200 (`CAP_HIT`) |
| M_nd | **32.3283 %** | 16.159 % at k=369 · **15.049 % best** · 15.224 % at cap |
| ω₁ | **162.882616** | 166.365 at k=369 · **166.4216 best** · 166.158 at cap |

Between production's descent at 138 and the fixed-move plateau:

* **M_nd falls 32.318 → 15.049 %** — a **53.4 % relative reduction**, more than
  halving the gray content;
* **ω₁ rises 162.877 → 166.422** — **+2.17 %**;
* density-field distance ‖ρ₅₄₇ − ρ₁₃₈‖₁/NE = see `METRICS.json`.

Production is not trading grayness for frequency here either: it loses on both.
This is the first valid measurement of what 400×50 production abandons.

### The two previously unsourced claims are now reproduced exactly

Earlier briefs asserted, without traceable evidence, "mature fixed-move
M_nd ≈ 16.16 %" and "ω₁ improves ≈ 2.14 %". The preceding two studies could not
source them and reported them as unsupported. RUN C reproduces both:

| claim | measured | at |
|---|---|---|
| M_nd ≈ **16.16 %** | **16.1589 %** | **k = 369** |
| ω₁ ≈ **+2.14 %** | **+2.1379 %** | **k = 369** |

and **k = 369 is exactly the iteration at which the inherited native stop rule
fires** on this arm. Both numbers were therefore real measurements from a
400×50 fixed-move run that was terminated natively and subsequently lost. The
earlier suspicion that 16.16 was "exactly half of 32.33" is disproved: half of
32.3283 is 16.1642, whereas the measured value is 16.1589 — close, but an
independent quantity.

---

## 8. Preregistered mechanism criteria

| criterion | required | result |
|---|---|---|
| **C1** onset classified at `k* ≤ 1200` | yes | **FAIL — no onset** |
| **C2** `k* > 138` | yes | n/a |
| **C3** label at 138 is `COHERENT` | yes | **PASS** (median `cosθ` = 0.985) |
| **C4** `|remaining_Mnd|` ≤ 5 % | yes | n/a |
| **C5** tail ≥ 400 with no >5 % M_nd improvement | yes | n/a |
| **C6** `boundFrac` < 0.10 and unsaturated `cosθ` negative | yes | partial — `boundFrac` = 0 ✔, but `cosθ` never negative |

C1 fails, so the preregistered mechanism criteria are not met.
