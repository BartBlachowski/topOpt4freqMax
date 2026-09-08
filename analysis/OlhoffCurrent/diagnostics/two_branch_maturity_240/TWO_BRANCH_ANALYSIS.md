# TWO_BRANCH_ANALYSIS — the withheld 240×30 test

**The frozen union fired correctly at the withheld mesh, and better than at any
training mesh** — but it fired via **Branch B**, while 240×30 terminates deep in
the **Branch A** region. The rule works; the branch labels do not describe the
endpoint.

Every predicate, threshold, window and persistence value was fixed in
`PREREGISTRATION.md` (SHA-256 `62748225253f85f6…`, frozen 2026-09-08T12:04:53Z)
before any 240×30 numeric content was opened. Nothing was retuned.

---

## 1. The run

| | RUN D |
|---|---|
| mesh / policy | 240×30, **fixed move 0.04**, unstopped |
| cap | 1200 (preregistered) |
| **outcome** | **1200 outer, `CAP_HIT`** |
| ω₁ final | 167.013409434 |
| M_nd final | 13.4507 % |
| volume | 0.4999993 |
| wall (1 thread) | 2387 s |
| **native stop would have fired at** | **147** |
| **β-stall first fires at** | **92** |

`CAP_HIT` is reported as `CAP_HIT`.

## 2. The frozen rule's verdict at 240×30

| | value |
|---|---|
| Branch A fires | **never** |
| Branch B fires | **187** |
| **first exhaustion event** | **187, Branch B** |
| M_nd at event | 12.8388 % |
| ω₁ at event | 167.0571 |
| `cosθ` at event (instantaneous / median₂₀) | −0.604 / **+0.217** |
| `net/path` at event (inst. / median₂₀) | 0.426 / 0.847 |
| `‖Δρ‖₂` at event / tol | 0.05067 / **0.075** |
| RMS(Δρ) at event | 5.971e-4 (< 8.8388e-4) |
| `max|Δρ|/move` at event | 0.4232 |
| **bound-active fraction** | **0.00000** |
| confirmation tail | **1013 iterations** |

### Preregistered criteria — all six pass

| criterion | bound | measured | |
|---|---|---|---|
| **P1** one branch, no retuning | — | Branch B at 187 | **PASS** |
| **P2** `|remUseful|` | ≤ 5.0 % | **−0.711 %** | **PASS** |
| **P3** `postRelImp` | ≤ 25 % | **0.029 %** | **PASS** |
| **P4** confirmation tail | ≥ 400 | **1013** | **PASS** |
| **P5** not a saturation artifact | — | `boundFrac = 0.00000` | **PASS** |
| **P6** no third endpoint | — | terminal cancelling | **PASS** |

**P2 and P3 are the best of all four meshes.** M_nd reaches its minimum
**12.7903 at k = 159**; the event is at 187 with M_nd = 12.8388, and over the
following 1013 iterations M_nd never improves by more than **0.029 %** relative
and ω₁ never by more than **0.0008 %**. The event lands 28 iterations after the
true optimum and nothing useful happens afterwards.

## 3. My preregistered prediction was wrong — disclosed

`PREREGISTRATION.md` §12 predicted **Branch A at ≈ 159**, and a terminal
`max|Δρ|/move` in [0.70, 1.00]. Measured: **Branch B at 187**, terminal
`max|Δρ|/move` = **0.6807**. The iteration was close (187 vs 159, both inside the
preregistered `[100, 400]` window); the **branch identity was wrong** and the
terminal amplitude fell just below the predicted band.

The preregistration stated in advance that a wrong branch prediction is still a
pass provided the criteria hold. They do. The miss is recorded because it is
exactly the kind of thing post-hoc narrative would otherwise bury.

## 4. Why Branch A never fired — a real, newly exposed blind spot

240×30 **does** end in a cancelling regime: terminal median `cosθ = −0.917`,
terminal `net/path = 0.202`, `max|Δρ|/move = 0.68`. Branch A is supposed to
detect exactly that. It never fires. The reason is its own amplitude clause:

| condition | iterations satisfied | first |
|---|---|---|
| `med cosθ < 0` **and** `med net/path < 0.5` (the cancellation signature) | **492** | k = 709 |
| `‖Δρ‖₂ ≥ tol` (amplitude non-negligible) | 170 | last at k = 896 |
| **both together** (Branch A's raw predicate) | **6** | never 20 consecutive |

**The cancelling cycle at 240×30 is low-amplitude**: terminal `‖Δρ‖₂ = 0.0621`
against `tol = 0.075`, i.e. **0.83 × tol**. Branch A's "amplitude
non-negligible" guard — added to stop cancellation being declared on numerical
noise — excludes it.

This is not unique to 240×30. Terminal `‖Δρ‖₂ / tol` across the four arms:

| mesh | 160×20 | **240×30** | 320×40 | 400×50 |
|---|---|---|---|---|
| terminal `‖Δρ‖₂ / tol` | **6.92** | **0.83** | **0.98** | 0.095 |
| terminal `cosθ` | −0.951 | −0.917 | −0.965 | +0.997 |

**Two of the three cancelling meshes end below their own amplitude threshold.**
320×40 was caught only because its cancellation *began* while amplitude was still
marginally above `tol` (Branch A fired at 255; the amplitude slipped under `tol`
afterwards). Shift that ordering slightly and Branch A would have missed it too.

## 5. Why the union still worked — and how robust that was

240×30 reaches its cancelling endpoint by a route none of the training meshes
took: **it converges coherently first, then re-excites into a low-amplitude
cycle.**

| k | 147 | 187 | 250 | 500 | 700 | 900 | 1199 |
|---|---|---|---|---|---|---|---|
| `‖Δρ‖₂` | 0.0720 | 0.0507 | 0.0302 | 0.0213 | 0.0415 | 0.0419 | 0.0580 |
| median `cosθ` | +0.960 | +0.217 | +0.995 | +0.998 | +0.949 | **−0.605** | **−0.923** |
| `max|Δρ|/move` | 0.172 | 0.423 | 0.077 | 0.058 | 0.462 | 0.432 | 0.643 |

Branch B caught the coherent-converged phase at 187, which is where useful work
actually ended. The later re-excitation produced **nothing** (M_nd 12.84 → 13.45,
i.e. *worse*).

**The firing was robust, not marginal.** After the event Branch B's predicate held
for **410 consecutive iterations** (20 required) and was true on 51 % of the 1013
post-event iterations. Between the native stop (147) and the firing (187) the
predicate was blocked on 15 iterations — **all 15 by amplitude re-excursions,
none by the coherence guard**. So the delay 147 → 187 is the persistence
requirement doing its job against a still-noisy amplitude, not a near-miss on
coherence.

## 6. Relation to the inherited native stop (Phase 4, restated with the result)

As declared before the run, **Branch B is the inherited native design-change stop
criterion plus a coherence guard plus persistence** — not a novel signal.

| mesh | native stop | Branch B | difference |
|---|---|---|---|
| 160×20 | never | never (amplitude) | — |
| **240×30** | **147** | **187** | **+40** (persistence against amplitude re-excursions) |
| 320×40 | 216 | never (**coherence guard blocks it**) | — |
| 400×50 | 369 | **369** | **0** |

The guard and the persistence requirement each demonstrably change the answer on
at least one mesh: the guard blocks a wrong Branch-B firing at 320×40, and
persistence delays 240×30 from 147 to 187. Neither is decorative — but neither
makes Branch B independent of the inherited criterion, and this report does not
claim otherwise.

## 7. Phase 11 failure modes, checked one by one

| # | failure mode | at 240×30 |
|---|---|---|
| 1 | neither branch fires but design clearly mature | **no** — Branch B fired at 187 |
| 2 | a branch fires while substantial useful evolution remains | **no** — `remUseful` −0.71 %, `postRelImp` 0.03 % |
| 3 | event triggered by a small pathological subset | **no** — `boundFrac = 0.00000` at the event and throughout |
| 4 | branch identity requires mesh-specific retuning | **no** — nothing was retuned |
| 5 | a qualitatively third *endpoint* | **no** — terminal state is cancelling, in the existing taxonomy. **But a third *pathway* did appear** (converge-then-re-excite), which no training mesh showed |
| 6 | classification depends on `M_nd` to choose a threshold | **no** — no branch uses `M_nd`, gray or interface weighting |
| 7 | Branch B falsely accepts a high-amplitude cycle | **no** — at the event `‖Δρ‖₂ = 0.051 < 0.075` and median `cosθ > 0`; the cycle developed 500+ iterations later and produced no useful work |
| 8 | Branch A falsely accepts coherent productive descent | **no** — Branch A never fired at all |

Failure mode 5 is the one that needs care: the *endpoint* taxonomy survives, the
*pathway* taxonomy does not. That is reported, not explained away.

## 8. Component vs union performance (Phase 14) — reported separately

| | 160×20 | 240×30 | 320×40 | 400×50 | coverage |
|---|---|---|---|---|---|
| **Branch A alone** | 83 | **never** | 255 | never | **2 / 4** |
| **Branch B alone** | never | **187** | never | 369 | **2 / 4** |
| **Union A ∪ B** | 83 | **187** | 255 | 369 | **4 / 4** |

Neither component covers more than half the meshes. The union covers all four,
and each component is the *only* one that fires on its two meshes — they are
genuinely complementary, not one carrying the other.

`remUseful` at the union event: **2.02 %, −0.71 %, −0.58 %, 1.11 %** — all inside
the preregistered 5 % bound, at every mesh, with one frozen rule.

## 9. The weakest point, stated plainly

At **160×20** the event fires at 83 while M_nd still falls from 13.23 to 11.36 —
a **14.19 % relative improvement** left on the table. This was recorded in
`PREREGISTRATION.md` §11 *before* the withheld test, so it is not a post-hoc
discovery, and it sits inside the preregistered `postRelImp ≤ 25 %` bound. It
remains the union's least convincing case and is the state a future controller
would most need to be evaluated against.

The other three meshes are 0.03 %, 0.16 % and 6.87 %.
