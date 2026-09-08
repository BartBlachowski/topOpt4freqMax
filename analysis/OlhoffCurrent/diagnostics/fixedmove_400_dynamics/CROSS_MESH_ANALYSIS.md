# CROSS_MESH_ANALYSIS — 160×20, 320×40, 400×50

The required Phase 13 table, and what it shows: **the three meshes do not share
one terminal dynamical regime**, so no single tested observable marks maturity
across them.

160×20 and 320×40 quantities are taken directly from the hash-valid
`dynamical_regime/evidence/dr_analysis.mat` — they are literally the numbers that
study reported, not recomputations.

---

## 1. The required table

| | **160×20** | **320×40** | **400×50** |
|---|---|---|---|
| NE | 3 200 | 12 800 | 20 000 |
| **production first descent** | 79 | 130 | **138** |
| production `cosθ` at descent | **−0.818** | +0.933 | **+0.938** |
| production `net/path` at descent | **0.346** | 0.962 | **0.978** |
| production label at descent | OTHER | COHERENT | **COHERENT** |
| production M_nd at descent | 13.443 | 23.401 | **32.359** |
| production `boundFrac` at descent | 0.0519 | 0.0000 | **0.0000** |
| **fixed-move cancellation onset** | **81** | **253** | **NONE (≤1200)** |
| offset (onset − descent) | **+2** | **+123** | **n/a** |
| M_nd at onset | 13.289 | 13.028 | — |
| **useful M_nd evolution remaining at onset** | **2.08 %** | **−0.58 %** | — |
| `boundFrac` at onset | 0.0556 | **0.0000** | — |
| raw `cosθ` at onset | −0.833 | −0.627 | — |
| **unsaturated `cosθ` at onset** | **−0.072** | **−0.627** | — |
| `net/path` at onset | 0.302 | 0.454 | — |
| post-onset tail (iterations) | 519 | 947 | — |
| post-onset M_nd behaviour | −1.83 pts (improves slightly) | **+0.50 pts (worsens)** | — |
| **native stop on the fixed-move arm** | **never** | **216** | **369** |
| fixed-move final M_nd | 11.458 | 13.524 | **15.224** (best 15.049 @ 547) |
| fixed-move final ω₁ | 169.575 | 166.329 | **166.158** (best 166.422 @ 394) |
| production final M_nd | 13.099 | 22.051 | **32.328** |
| production final ω₁ | 169.689 | 166.091 | **162.883** |
| **terminal `max|Δρ|/move`** | **0.999** | **0.697** | **0.031** |
| **terminal `cosθ`** | **−0.951** | **−0.965** | **+0.997** |
| **terminal `net/path`** | **0.090** | **0.132** | **0.992** |

---

## 2. What the table says

### 2.1 The mechanism holds at two meshes and fails at the third

160×20 and 320×40 behave identically once aligned at their onsets (figure `F8`):
`cosθ` collapses from ≈ +1 to ≈ −0.95 within ~30 iterations, `net/path` falls to
0.09–0.13, and topology evolution stops (2.08 % and −0.58 % remaining).

400×50 does none of this. Its `cosθ` sits flat at ≈ +1.0 across the whole window
and its `net/path` at ≈ 0.99. Instead of oscillating it **converges**:
`max|Δρ|/move` falls to 0.031 and `‖Δρ‖₂` to a tenth of the native tolerance.

### 2.2 Complementary blind spots

| observable family | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| **cancellation** (`cosθ`, `net/path`, `q2`) | works | works | **blind** |
| **amplitude** (`max|Δρ|/move`, `‖Δρ‖₂`, native rule) | **blind** — pinned at 0.999 | works (native at 216) | works (native at 369) |

Each family covers exactly two meshes, and they overlap only at 320×40. This is
why no single observable has yet been found: the coarse and fine meshes terminate
in *physically different* states.

* **160×20** ends in a **bound-pinned limit cycle**: the design keeps taking
  full-size steps forever and they cancel. Amplitude cannot see this; direction
  can.
* **400×50** ends in **ordinary convergence**: the design stops moving, coherently.
  Direction cannot see this; amplitude can.
* **320×40** passes through both: amplitude decays to 0.70 *and* the direction
  reverses. It is the only mesh where the two families agree, which is why the
  mechanism looked general when only 160 and 320 were available.

### 2.3 What is common to all three — and it is not dynamical

The one statement that does hold at every mesh is about **production**, not about
the dynamics:

| mesh | production descends at | trajectory state there | M_nd production leaves on the table |
|---|---|---|---|
| 160×20 | 79 | already cancelling (`cosθ` −0.818) | 13.10 → 11.46 = **1.64 pts** |
| 320×40 | 130 | coherent (`cosθ` +0.933) | 22.05 → 13.00 = **9.05 pts** |
| 400×50 | 138 | coherent (`cosθ` +0.938) | 32.33 → **15.05** = **17.28 pts** |

**The penalty grows steeply with refinement** — 1.6, 9.1, 17.3 M_nd points — and
at the two fine meshes production descends while the design is coherently
descending. That finding is unaffected by the refutation and is now measured at
400×50 for the first time: production abandons a **53.4 % relative reduction in
M_nd** and **+2.17 % in ω₁**.

### 2.4 Objective stationarity vs dynamical maturity

Refinement does not monotonically separate the two, as had been hypothesised.
It changes which of them is even *defined*:

| mesh | native (objective/amplitude) rule | cancellation rule |
|---|---|---|
| 160×20 | **never fires** — objective never settles under a limit cycle | fires at 81, 2.08 % remaining |
| 320×40 | fires at 216 (M_nd 13.28) | fires at 253 (M_nd 13.03) |
| 400×50 | fires at **369** (M_nd 16.16) | **never fires** |

At 320×40 and 400×50 the native rule fires close to the end of useful work
(best M_nd 13.00 and 15.05 respectively). Its real failure is at **160×20**,
where the limit cycle keeps `‖Δρ‖₂` at 0.346 — seven times the tolerance —
forever.

---

## 3. Consequence for a future controller

A controller that must work at 160×20, 320×40 and 400×50 has to recognise **two
different terminal states**: a persistent limit cycle *and* ordinary convergence.
No single observable tested across these three studies does both — cancellation
is blind to one, amplitude to the other, and their natural unification
(net displacement over W, `‖ρ_k − ρ_{k−W}‖`) was already refuted as blind by the
`topology_maturity_transition` study.

The obvious remaining construction is a **disjunction** of the two existing
signals — "descend when the design has stopped making net progress, whether
because it has stopped moving or because its motion cancels" — which would fire
at 81 / 216 / 369 across the three meshes, each near the end of useful work.
**This is recorded as an observation, not proposed, preregistered, threshold-ed
or implemented**, per the task's Phase 14/18 boundary. Its two branches were each
refuted individually at one mesh, so whether their disjunction is a principled
single mechanism or two patches stitched together is exactly the question the
next task would have to settle *before* any controller is designed.
