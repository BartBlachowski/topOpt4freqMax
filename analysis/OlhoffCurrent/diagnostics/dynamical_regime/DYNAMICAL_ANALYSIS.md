# DYNAMICAL_ANALYSIS — is maturity a regime, not a magnitude?

**Yes.** The transition from coherent net progress to period-2 cancellation is a
sharp, mesh-consistent, largely bound-independent event, and it lands almost
exactly where topology evolution actually stops. Production's β-stall trigger
coincides with it at 160×20 by accident, and misses it by 123 iterations at
320×40 — and at 400×50 production stops before the regime is ever reached.

All quantities are defined in [`PREREGISTRATION.md`](PREREGISTRATION.md) §§2–5
and were frozen before either run.

---

## 1. The two runs

| | RUN A | RUN B |
|---|---|---|
| mesh | 400×50 | 320×40 |
| policy | **production**: ladder + β-stall + production stop | **fixed move 0.04**, unstopped |
| cap | 400 | 1200 |
| **outcome** | **139 outer, `CONVERGED`** | **1200 outer, `CAP_HIT`** |
| ω₁ | 162.888779843 | 166.315388525 |
| M_nd final | **32.3283 %** | **13.5242 %** |
| volume | 0.49999915 | 0.49999956 |
| inner MMA | 2918 | — |
| wall (1 thread) | 640 s | 4133 s |

`CAP_HIT` is reported as `CAP_HIT`. RUN B was not stopped early: the run
continued 984 iterations past the point (216) at which the inherited production
criterion would have admitted convergence, exactly as preregistered.

### Prefix reproduction (Phase D) — **bitwise, both references**

| reference | span | ρ max diff | ω₁ | M_nd | volume | events |
|---|---|---|---|---|---|---|
| `move_stop/runs/fixedmove_320x40.mat` | 1…216 | **0** | 0 | 0 | 0 | identical |
| `move_transition/runs/armU_320x40.mat` | 1…213 | **0** | 0 | 0 | 0 | identical |

Not `FIXEDMOVE_PREFIX_REPRODUCTION_FAIL`. The two archives were also verified
bitwise identical to *each other* over 1…213, so the reference itself is
self-consistent. RUN A has no historical raw production trajectory and no
bitwise claim is made for it.

---

## 2. The regime transition exists, and it is sharp

Preregistered classifier: trailing-20 medians, `PERIOD2` when median `q2 < 1`
**and** median `cosθ < 0`, sustained 20 consecutive iterations.

| trajectory | PERIOD2 onset | final label | frac PERIOD2 | undefined `q2`/`cosθ` |
|---|---|---|---|---|
| 160×20 fixed 0.04 | **81** | PERIOD2 | 0.852 | 0 / 0 |
| **320×40 fixed 0.04 (RUN B)** | **253** | PERIOD2 | 0.790 | 0 / 0 |
| 400×50 production (RUN A) | **never** | COHERENT | **0.000** | 0 / 0 |
| 320×40 production | never | COHERENT | 0.000 | 0 / 0 |
| 160×20 production | never (label `OTHER` at 79, `PERIOD2` at 90) | OTHER | 0.020 | 0 / 0 |

**320×40 does enter the same regime as 160×20** — it simply takes 253 iterations
instead of 81. The previous fixed-move evidence ended at 213/216, i.e. **40
iterations before onset**, which is exactly why the question had been
unanswerable.

RUN B's transition, from the run itself:

| k | M_nd | ω₁ | median `q2` | median `cosθ` | label |
|---|---|---|---|---|---|
| 130 | 23.3224 | 165.9433 | 2.005 | **+0.983** | COHERENT ← *production descends here* |
| 200 | 13.6146 | 166.6601 | 1.760 | +0.474 | COHERENT |
| **253** | **13.0275** | 166.4319 | **0.986** | **−0.516** | **PERIOD2 onset** |
| 400 | 13.1500 | 166.3816 | 0.468 | −0.890 | PERIOD2 |
| 800 | 13.4973 | 166.3272 | 0.302 | −0.955 | PERIOD2 |
| 1200 | 13.5242 | 166.3292 | 0.260 | −0.967 | PERIOD2 |

---

## 3. The onset marks the end of productive evolution (Phase I)

This is the claim that makes the regime *useful* rather than merely real.

| | 160×20 fixed | 320×40 fixed (RUN B) |
|---|---|---|
| PERIOD2 onset | 81 | 253 |
| M_nd at onset | 13.2886 % | 13.0275 % |
| M_nd at run end | 11.4580 % (k=600) | 13.5242 % (k=1200) |
| **remaining M_nd evolution at onset** | **2.08 %** | **−0.58 %** |
| remaining L1 displacement at onset | 15.09 % | 9.42 % |
| iterations spent after onset | 519 | 947 |

At 320×40, M_nd falls **23.32 → 13.03** between production's descent (130) and
onset (253) — that is the entire remaining topology evolution — and then moves
**+0.50 points in the wrong direction** over the following 947 iterations while
`cosθ` deepens to −0.967. After onset the optimizer is doing work and achieving
nothing: the motion is real (‖Δρ‖ is not small) but it cancels.

**Maturity is therefore better described as cancellation than as small motion.**
This is the direct refutation of the magnitude framing: at 160×20 in the mature
regime `max|Δρ|` sits at the move bound 0.0400 for all 600 iterations, yet 90 %
of the path length is being cancelled.

---

## 4. Where production descends, dynamically (Phases L, J)

At each mesh's **first** production move descent:

| mesh | descent k | move | `q2`† | `cosθ` | `net/path` | `boundFrac` | `revFrac` | label | M_nd |
|---|---|---|---|---|---|---|---|---|---|
| 160×20 | **79** | 0.04→0.02 | 1.172 | **−0.818** | **0.346** | 0.0519 | 0.704 | OTHER | 13.443 |
| 320×40 | **130** | 0.04→0.02 | 3.848 | **+0.933** | **0.962** | **0.00000** | 0.305 | COHERENT | 23.401 |
| **400×50** | **138** | 0.04→0.02 | 3.934 | **+0.938** | **0.978** | **0.00000** | 0.425 | COHERENT | 32.359 |

† `q2` evaluated *at* a descent iteration is contaminated by the step-size change
itself (the denominator `d1` shrinks with the move), which is why it exceeds 2.
The trailing-window median — 2.000 at both fine meshes — is the clean reading,
and it is what the classifier uses.

**The answer to the central Phase L question is unambiguous: production reduces
the move while the trajectory still exhibits near-perfect coherent directional
progress.** At 400×50, 97.8 % of the path length over the preceding ten
iterations is net displacement, `cosθ = +0.938`, and **not one element of 20 000
is at the move bound**.

### Distance between the β-stall descent and the dynamical event

| mesh | production first descent | period-2 onset (fixed move) | offset |
|---|---|---|---|
| 160×20 | 79 | **81** | **+2** |
| 320×40 | 130 | **253** | **+123** |
| 400×50 | 138 (run converged at 139) | not reached in 139 iterations | ≥ +1, unmeasured |

This is the mechanism behind every previous puzzle. β stalls when the objective
stops improving. At 160×20 the objective stops improving *because* the design has
begun to cancel — the two events coincide to within two iterations, so β-stall
descends at the right moment and the coarse-mesh anchor looks correct. At the
fine meshes the objective stalls **while the design is still coherently
descending**, so β-stall descends 123 iterations early (320×40) or descends once
and immediately terminates (400×50).

**β stall precedes dynamical maturity at the fine meshes, and coincides with it
at the coarse mesh.** That is why it is not a maturity signal.

### What it costs

| mesh | production final M_nd | fixed-move M_nd at/after onset | production ω₁ | fixed-move ω₁ |
|---|---|---|---|---|
| 160×20 | 13.0988 | 11.4580 | 169.6893 | 169.5747 |
| 320×40 | **22.0508** | **13.5242** | 166.0907 | **166.3292** |
| 400×50 | **32.3283** | — (no fixed-move run authorized) | 162.8826 | — |

At 320×40 the fixed-move trajectory reaches **both** a lower M_nd (13.52 vs
22.05) **and** a higher ω₁ (166.329 vs 166.091). Production is not trading
grayness for frequency there; it is losing on both.

---

## 5. Which observable, and how robust (Phases K6–K10)

### Separation at the matched event "first production descent"

| observable | 160×20 (mature) | 320×40 | 400×50 | separation |
|---|---|---|---|---|
| `cosθ` | **−0.818** | +0.933 | +0.938 | **sign change** |
| `net/path` | **0.346** | 0.962 | 0.978 | 2.8× , bounded [0,1] |
| trailing-median `q2` | 1.182 | 2.007 | 2.000 | ~1.7× |
| `revFrac` | 0.704 | 0.305 | 0.425 | 2.3× |
| `max|Δρ|/move` (prior study) | ≈1.0 | — | — | *blind — refuted* |

All are **dimensionless ratios or correlations. No NE exponent appears anywhere**,
and none was fitted.

### Sensitivity to isolated bound-touching elements

This is the criterion that killed every earlier candidate, so it was tested
directly by deleting the bound-saturated set and recomputing.

| | `boundFrac` | `cosθ` | `cosθ` bound-removed | `net/path` | bound-removed |
|---|---|---|---|---|---|
| **320×40 fixed, at onset (253)** | **0.0000** | −0.627 | **−0.627** | 0.454 | **0.454** |
| **320×40 fixed, median over onset+400** | **0.0000** | −0.915 | **−0.915** | 0.210 | **0.210** |
| 400×50 prod, at descent (138) | **0.0000** | +0.938 | **+0.938** | 0.978 | **0.978** |
| 320×40 prod, at descent (130) | **0.0000** | +0.933 | **+0.933** | 0.962 | **0.962** |
| 160×20 fixed, at onset (81) | 0.0556 | −0.833 | −0.072 | 0.302 | 0.759 |
| 160×20 fixed, median over onset+400 | 0.0334 | −0.929 | **−0.554** | 0.094 | 0.318 |

**At 320×40 and 400×50 the diagnostics are exactly bound-independent — not one
element is at the move bound, so removal changes nothing at all.** The fine-mesh
readings, which are the ones that matter for the fine-mesh pathology, cannot be
artifacts of a saturated minority.

**At 160×20 there is a genuine, reportable attenuation.** Removing the ~3–7 %
saturated elements moves the established-regime median `cosθ` from −0.93 to
**−0.55** — still firmly negative, still the correct side of zero, but visibly
carried in part by the bound population. At the onset *instant* the attenuation
is severe (−0.833 → −0.072). So: **the sign of `cosθ` survives outlier removal
at every mesh; its magnitude at the coarse mesh does not.** This is a materially
better robustness profile than the previous `D_W/C_W` coherence ratio, which was
*produced by* the outliers rather than merely amplified by them — but it is not
perfect, and it is the main reason this study does not claim more than it does.

`net/path` is the least sensitive of the three to isolated bound motion at the
fine meshes (identical with and without removal) and the most sensitive at
160×20 (0.094 → 0.318). `cosθ` retains its sign everywhere.

### Classifier robustness (reporting only — no threshold was adopted)

The preregistered rule was held fixed. This sweep characterises it; nothing was
re-tuned, and the preregistered setting remains the one used everywhere above.

| `q2` thr | `cosθ` thr | P | 160×20 onset (rem. M_nd) | 320×40 onset (rem. M_nd) |
|---|---|---|---|---|
| **1.0** | **0.0** | **20** | **81 (2.08 %)** | **253 (−0.58 %)** |
| 0.8 | 0.0 | 20 | 85 (1.90 %) | 280 (−0.59 %) |
| 1.2 | 0.0 | 20 | 79 (2.20 %) | 237 (−0.51 %) |
| 1.0 | ∓0.2 | 20 | 81 (2.08 %) | 253 (−0.58 %) |
| 1.0 | 0.0 | 10 | 76 (2.44 %) | 248 (−0.55 %) |
| 1.0 | 0.0 | 40 | 91 (1.84 %) | 263 (−0.59 %) |
| 1.2 | +0.2 | 10 | 74 (2.62 %) | 232 (−0.47 %) |
| 0.8 | −0.2 | 40 | 95 (1.83 %) | 290 (−0.59 %) |

Onset moves by ±15 % at worst; **remaining M_nd at onset is essentially invariant**
(1.8–2.6 % at 160×20, −0.6…−0.5 % at 320×40). The `cosθ` threshold makes *no
difference at all* across ±0.2, because the transition through zero is abrupt.
The qualitative conclusion — onset ≈ production descent at 160×20, onset ≫
production descent at 320×40 — holds under every setting tried.

### Declared non-independence

As stated in the preregistration, when consecutive step norms are equal
`q2² = 2(1 + cosθ)` exactly. `q2` and `cosθ` agreeing is **not** corroboration;
they are two readings of the same 2-step geometry. `net/path` at W = 10 is a
genuinely independent 10-step statistic, and it agrees — that *is* corroboration.

---

## 6. Where the oscillation lives (Phase N)

Sign of `Δρ_k · Δρ_{k−1}` per element, and the composition of the reversing set:

**160×20, mature regime** — reversal is **global and spatially coherent**, not an
interface effect and not a bound effect:

| k | `revFrac` | among void | among gray | among solid | gray frac | `boundFrac` | rev among bound | neighbour support |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.794 | 0.975 | 0.359 | 0.761 | 0.146 | 0.068 | 0.991 | 0.994 |
| 200 | 0.691 | 0.895 | 0.438 | 0.572 | 0.141 | 0.033 | 1.000 | 0.988 |
| 400 | 0.612 | 0.843 | 0.412 | 0.446 | 0.135 | 0.028 | 1.000 | 0.979 |
| 599 | 0.746 | 0.843 | 0.683 | 0.672 | 0.138 | 0.022 | 0.986 | 0.995 |

36–82 % of the domain reverses each iteration; reversers are almost entirely
surrounded by other reversers (neighbour support 0.95–0.99). *Every*
bound-touching element reverses, but they are only 2–7 % of the domain — the bulk
of the oscillation is in the void and solid phases.

**400×50 at its production descent** — the opposite picture:

| k | `revFrac` | among void | **among gray** | among solid | gray frac | `boundFrac` |
|---|---|---|---|---|---|---|
| 120 | 0.183 | 0.113 | **0.0096** | 0.462 | 0.365 | 0.0000 |
| 137 | 0.334 | 0.717 | **0.0078** | 0.285 | 0.348 | 0.0000 |
| 138 | 0.425 | 0.775 | **0.0328** | 0.485 | 0.348 | 0.0000 |

**34.8 % of the 400×50 domain is gray, and under 3 % of it reverses.** The
design-active phase is moving monotonically when production cuts the step. What
little reversal exists is confined to void elements jittering at ρ_min.

This locates the mechanism: **maturity is the gray phase ceasing to make net
progress and beginning to cancel.** Recorded as a mechanism observation only — no
gray-weighted formula is proposed, chosen or implemented here.

---

## 7. What is still missing

* **No 400×50 fixed-move counterfactual exists.** RUN A converged at 139 in the
  coherent regime, so 400×50's period-2 onset is unobserved. The onset sequence
  81 → 253 → ? grows steeply with refinement, and 400×50 production stops at 139,
  but the third point is *not measured* and is not asserted.
* **RUN B is `CAP_HIT` at 1200**, not converged. The regime was classified long
  before the cap (onset 253) so the classification is unaffected, but the
  fixed-move endpoint at 320×40 is a cap, not a fixed point.
* **The 160×20 onset-instant attenuation** under bound removal (−0.833 → −0.072)
  is unexplained and would need to be understood before any controller could rely
  on the magnitude rather than the sign.
* The regime at 400×50 was characterised **only** at production events, because
  only production was authorized there.
