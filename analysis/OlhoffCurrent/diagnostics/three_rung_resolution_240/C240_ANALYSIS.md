# C240×30 ANALYSIS — the one new causal trajectory

All values from `evidence/analysis.json`. Index convention per
`COUNTERFACTUAL_VALIDITY.md` §4. Relative changes normalised by the **earlier**
state (inherited convention).

---

## 1. Run summary

| | |
|---|---|
| Mesh | 240 × 30, `NE = 7200` |
| Controller | frozen four-rung `A OR B`, `[0.04 0.02 0.01 0.005]`, unchanged |
| `stop.tolerance` | 0.075 |
| Cap | 1600 (inherited) |
| **Status** | **`CONVERGED` @ 1358** — the cap was **not** reached |
| Terminal branch / persistence | **B**, 20 consecutive, window [1339, 1358] |
| Wall | 14 160 s (≈ 3.93 h), single thread |
| Inner MMA total | 44 181 |
| Non-converged inner solves | **0** |
| Trajectory rebuild | exact; clamp displacement max `5.55e-17` |
| Final `ρ` SHA-256 | `ed2dd8cd37cbe6388b7476b1b86e08664bcfe9782aaa28905e7df4e58338b5cf` |

## 2. The four states

| | outer | move | branch | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | 206 | 0.04 | B | 12.8843 | 167.045037 | 197.3370 | 0.181340 | 0.14861 | 0.03000 | 0.499999839 |
| **S2** | 245 | 0.02 | B | 12.9260 | 167.036705 | 197.2459 | 0.180854 | 0.14972 | 0.03000 | 0.499999895 |
| **S3** | **284** | **0.01** | **B** | **12.9165** | **167.038463** | 197.2343 | 0.180772 | 0.15000 | 0.03083 | 0.499999552 |
| **F** | 1358 | 0.005 | B | 12.9425 | 167.049693 | 197.0294 | 0.179466 | 0.14944 | 0.03111 | 0.499999201 |

**Every event fires on Branch B** — amplitude convergence with a positive
coherence median. This places 240×30 with 400×50 (B at every stage) rather than
with 160×20 (A, A, B) or 320×40 (A, B, B).

## 3. The S3 event record — the three-rung terminal state

| | value |
|---|---|
| stage-3 start | 246 |
| **S3 declaration** | **284** |
| offset from stage start | **38** (the arithmetic minimum) |
| triggering branch | **B** |
| sustained window | [265, 284] |
| `nA` / `nB` | 0 / 20 |
| `med₂₀ cosθ` | +0.78613 |
| `med₂₀ net/path` | +0.89203 |
| `‖Δρ‖₂` | 0.0035783 |
| `‖Δρ‖₂ / tol` | 0.04771 |
| RMS `Δρ` | 4.2171e-05 |
| `max|Δρ|` | 3.3831e-04 |
| `max|Δρ| / move` | 0.03383 |
| bound fraction | **0** |
| native stop predicate holds | **yes** |
| β-stall had fired by then | yes (first at 92) |
| subspace size | 2 |
| cumulative outer / inner | 284 / 5 506 |
| `ρ` SHA-256 | `c9361fc6…` (full value in `METRICS.json`) |

`S3` is a **valid persistent-`E` exhausted state**: declared, Branch B, full
20-iteration persistence, at `move = 0.01`, with zero bound-active elements.

## 4. Stage 4 — what the final rung actually did

| | value |
|---|---|
| stage-4 start | 285 |
| first mathematically evaluable iteration | 304 (`start + 19`) |
| earliest arithmetically possible declaration | 323 (`start + 38`) |
| first iteration at which `E` is true | **327** |
| **actual declaration** | **1358** |
| **offset** | **1073** — *not* the minimum |
| fraction of stage 4 with `E` true | **5.9 %** |
| fraction with `‖Δρ‖₂ < tol` | **100 %** |
| fraction with `med₂₀ cosθ < 0` | **92.4 %** |
| fraction in **low-amplitude cancellation** (`amp < tol` **and** `med cos < 0`) | **92.4 %** |

**Stage 4 at 240×30 is the same regime that caps 320×40.** The documented hole in
the union — low amplitude together with negative coherence satisfies neither
Branch A (amplitude too small) nor Branch B (motion not coherent) — holds for
92.4 % of the stage. The run spent 1073 iterations in it before a 20-iteration
run of coherent low-amplitude steps finally appeared and admitted termination.

This mesh escaped the cap; 320×40 did not. The difference between `CONVERGED` and
`CAP_HIT` at `move = 0.005` is therefore **whether a 20-iteration `E`-true window
happens to appear before the cap**, not a difference of kind. That is new
evidence about the rule's terminal behaviour, and it is reported as an
observation — **no rule is changed on the strength of it.**

## 5. Declaration timing (Phase 12)

| stage | move | start | first evaluable | first `E` true | earliest possible | declaration | offset | at minimum? | `E` true at first evaluable? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 1 | 20 | 147 | 39 | **206** | 205 | no | **no** |
| 2 | 0.02 | 207 | 226 | **226** | 245 | **245** | **38** | **yes** | **yes** |
| 3 | 0.01 | 246 | 265 | **265** | 284 | **284** | **38** | **yes** | **yes** |
| 4 | 0.005 | 285 | 304 | 327 | 323 | **1358** | **1073** | **no** | **no** |

**The prior "lower stages declare at `stageStart + 38`" observation reproduces
for stages 2 and 3 — and breaks at stage 4.** Stages 2 and 3 have `E` true from
the very first evaluable iteration and hold it unbroken (100 % of the window).
Stage 4 does not: `E` is false when first evaluable, first becomes true 4
iterations after the earliest possible declaration, and then holds only 5.9 % of
the time.

So the earlier generalisation was **too broad**. The correct narrow statement,
supported by all four meshes:

> At `move = 0.02` and `move = 0.01` the exhaustion condition is already
> satisfied as soon as enough post-transition history exists, and those stages
> cost exactly 39 outer iterations. At `move = 0.005` it is not: that stage
> enters low-amplitude cancellation, which the frozen union cannot admit, and
> terminates late (240×30) or not at all (320×40).

## 6. Cost

| rung | move | range | outer | % outer | inner MMA | % inner | status |
|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 1–206 | 206 | 15.2 % | 4 030 | 9.1 % | DESCENDED |
| 2 | 0.02 | 207–245 | 39 | 2.9 % | 663 | 1.5 % | DESCENDED |
| 3 | 0.01 | 246–284 | 39 | 2.9 % | 813 | 1.8 % | DESCENDED |
| **4** | **0.005** | **285–1358** | **1074** | **79.1 %** | **38 675** | **87.5 %** | CONVERGED |

Terminating at `S3` saves **1074 outer iterations (79.1 %)** and **38 675 inner
MMA iterations (87.5 %)**. Rung 4's cost multiplier against the state it starts
from is **3.78×** — cost-dominated under the inherited criterion.

Wall time is reported but **not** relied upon: seconds per inner MMA iteration
drift **5.63×** within this run (0.061 → 0.346), reproducing the contention seen
in all prior arms.

## 7. Preregistered prediction — outcome

| prediction (frozen before the run) | outcome |
|---|---|
| `S1` declares at **206 ± 10**, Branch **B** | **206, Branch B — exact hit** |
| `S1` window begins where the fixed-move arm recorded (187) | **187 — exact** |
| `S2 ≈ 245`, `S3 ≈ 284` at `stageStart + 38` | **245 and 284 — both exact** |
| rung-4 `ω₁` benefit **below** 0.10 % | **+0.00672 % — confirmed** |
| stage-4 terminal status: **no prediction made** | `CONVERGED` after 1073 iterations |

The S1 agreement with a *separately produced* fixed-move arm — different `+impl`
tree, different MATLAB build — is a strong independent check that the driver is
correctly wired and that the frozen rule behaves as recorded.
