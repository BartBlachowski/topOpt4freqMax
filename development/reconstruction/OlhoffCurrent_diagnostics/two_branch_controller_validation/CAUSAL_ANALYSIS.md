# CAUSAL ANALYSIS — Phases 13–18

Machine-readable: `evidence/analysis.json`. Figures `F1`–`F14`.

---

## 1. Headline

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| production status / outer | `NATIVE_CONVERGED` / 91 | `NATIVE_CONVERGED` / 131 | `CONVERGED` / 139 |
| **candidate status / outer** | **`CONVERGED` / 219** | **`CAP_HIT` / 1600** | **`CONVERGED` / 505** |
| production first descent | 79 | 130 | 138 |
| **candidate first descent** | **103** (+24) | **275** (+145) | **389** (+251) |
| `M_nd` production → candidate | 13.4025 → **12.7041** | 23.3596 → **12.9233** | 32.3283 → **15.3311** |
| **Δ`M_nd`** | **−0.698 (−5.21 %)** | **−10.436 (−44.68 %)** | **−16.997 (−52.58 %)** |
| ω₁ production → candidate | 169.4952 → **170.0113** | 165.9508 → **166.4267** | 162.8826 → **166.4563** |
| **Δω₁** | **+0.516 (+0.304 %)** | **+0.476 (+0.287 %)** | **+3.574 (+2.194 %)** |
| outer multiplier | ×2.41 | **×12.21** | ×3.63 |
| inner-MMA multiplier | ×2.26 | **×29.28** | ×3.53 |
| wall multiplier | ×3.39 | **×88.60** | ×6.54 |
| terminal exhaustion genuine | **yes** (B, window 200–219) | **no** — never exhausted | **yes** (B, window 486–505) |

The intervention works, and on the fine meshes it works dramatically. It also
fails, once, in a way the frozen rule was already known to be capable of failing.

## 2. Phase 13 — transition audit

Every move transition, on every mesh. `decl` is the iteration at which the
frozen persistence completed; the transition is applied at `decl + 1`.

| mesh | # | iter | move | branch | decl (window) | A | B | nA | nB | β stalled? | prod. shadow stage | `med₂₀cosθ` | `med₂₀ net/path` | amp ‖Δρ‖₂ | amp/tol | cosθ | net/path | bound frac | ω₁ | `M_nd` | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160×20 | 1 | 103 | .04→.02 | **A** | 102 (83–102) | 1 | 0 | **20** | 0 | yes | 4 | −0.9066 | 0.1330 | 0.6296 | 12.59 | −0.9197 | 0.1203 | 0.0712 | 168.9804 | 13.0364 | 0.1462 | 0.0269 | 0.499998 |
| 160×20 | 2 | 142 | .02→.01 | **A** | 141 (122–141) | 1 | 0 | **20** | 0 | yes | 4 | −0.8567 | 0.2261 | 0.1095 | 2.19 | −0.8733 | 0.2270 | 0.0069 | 169.8175 | 12.7884 | 0.1437 | 0.0262 | 0.499999 |
| 160×20 | 3 | 181 | .01→.005 | **B** | 180 (161–180) | 0 | 1 | 0 | **20** | yes | 4 | +0.5681 | 0.8703 | 0.0052 | 0.10 | +0.8186 | 0.7928 | 0.0000 | 169.9766 | 12.7561 | 0.1450 | 0.0269 | 0.500000 |
| 320×40 | 1 | 275 | .04→.02 | **A** | 274 (255–274) | 1 | 0 | **20** | 0 | yes | 4 | −0.6539 | 0.4211 | 0.1031 | 1.03 | −0.6753 | 0.3952 | 0.0000 | 166.4216 | 13.0121 | 0.1523 | 0.0297 | 0.500000 |
| 320×40 | 2 | 314 | .02→.01 | **B** | 313 (294–313) | 0 | 1 | 0 | **20** | yes | 4 | +1.0000 | 0.9997 | 0.0108 | 0.11 | +0.9999 | 0.9997 | 0.0000 | 166.4163 | 12.9797 | 0.1528 | 0.0302 | 0.500000 |
| 320×40 | 3 | 353 | .01→.005 | **B** | 352 (333–352) | 0 | 1 | 0 | **20** | yes | 4 | +0.5942 | 0.7527 | 0.0033 | 0.03 | +0.2381 | 0.8221 | 0.0000 | 166.4273 | 12.9401 | 0.1530 | 0.0302 | 0.500000 |
| 400×50 | 1 | 389 | .04→.02 | **B** | 388 (369–388) | 0 | 1 | 0 | **20** | yes | 4 | +0.9981 | 0.9935 | 0.0800 | 0.64 | +0.9986 | 0.9937 | 0.0000 | 166.4176 | 15.6649 | 0.1822 | 0.0398 | 0.499999 |
| 400×50 | 2 | 428 | .02→.01 | **B** | 427 (408–427) | 0 | 1 | 0 | **20** | yes | 4 | +0.9996 | 0.9988 | 0.0181 | 0.14 | +0.9998 | 0.9986 | 0.0000 | 166.4355 | 15.4413 | 0.1800 | 0.0393 | 0.500000 |
| 400×50 | 3 | 467 | .01→.005 | **B** | 466 (447–466) | 0 | 1 | 0 | **20** | yes | 4 | +0.1907 | 0.5667 | 0.0079 | 0.06 | −0.7052 | 0.5845 | 0.0000 | 166.4427 | 15.3732 | 0.1796 | 0.0392 | 0.500000 |

All quantities are the candidate's own values at the **declaration** iteration —
the moment the frozen persistence completed — not at the window's first
iteration. `med₂₀cosθ` and `med₂₀ net/path` are the quantities the predicate
actually tests; `cosθ` and `net/path` are the single-iteration values, shown so
that the difference between an instantaneous reading and the persistent median is
visible (row 400×50 #3 is the clear case: `cosθ = −0.71` at that one iteration
while the 20-median is still `+0.19`, so Branch B holds — which is exactly what
the median window is for).

**No transition occurred without the frozen rule.** On every mesh the set of
iterations at which `move` changed equals exactly the set of declaration
iterations + 1; each advanced exactly one rung; each carried a completed
20-iteration persistence window on the branch named.

**No transition occurred because β stalled.** β had *already* stalled at every
one of these iterations — the "β stalled?" column is `yes` nine times out of nine
— and the ladder did not move until the frozen rule said so. The clearest form of
this: on all three meshes production's own rule, replayed on the candidate's
path, would have driven the ladder to its **last** rung by iterations 101, 152
and 160 respectively, while the candidate was still at `move = 0.04`.

## 3. Phase 14 — termination audit

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| status | `CONVERGED` | **`CAP_HIT`** | `CONVERGED` |
| iteration | 219 | 1600 | 505 |
| move / stage | 0.005 / 4 | 0.005 / 4 | 0.005 / 4 |
| A / B / E at the end | 0 / 1 / 1 | 0 / 0 / 0 | 0 / 1 / 1 |
| persistence nA / nB | 0 / **20** | **0 / 0** | 0 / **20** |
| terminal branch (window) | **B** (200–219) | — | **B** (486–505) |
| ω₁ | 170.0113 | 166.4267 | 166.4563 |
| `M_nd` [%] | 12.7041 | 12.9233 | 15.3311 |
| gray / mid | 0.1444 / 0.0269 | 0.1522 / 0.0303 | 0.1796 / 0.0392 |
| volume | 0.49999913 | 0.49999962 | 0.49999943 |
| max\|Δρ\| | 4.86e-04 | 1.02e-04 | 3.55e-04 |
| ‖Δρ‖₂ | 2.88e-03 | 4.19e-03 | 1.11e-03 |
| β stalled at the end | yes | yes | yes |
| inherited native stop would hold | yes | yes | yes |
| inner-solver status | all converged | all converged | all converged |

Both `CONVERGED` states are genuine: `move == 0.005`, stage 4, and a completed
20-iteration Branch-B window. Neither is a relabelled cap, failure or β event.
The `CAP_HIT` is reported as a cap and nothing else.

## 4. Phase 16 — the fine-mesh causal test

**320×40**

1. *Did the candidate stay at `move = 0.04` beyond production's β descent?* **Yes.**
2. *By how much?* Production descended at **130**; the candidate at **275** — **145 further iterations**.
3. *What changed in that interval?* gray fraction 0.2633 → 0.1523 (−0.111), mid-density 0.0950 → 0.0297 (−0.065): the grey end regions resolved into members.
4. *`M_nd`?* 23.322 → 13.012, **−44.21 %**.
5. *ω₁?* 165.943 → 166.422, **+0.288 %**.
6. *Which branch permitted descent?* **A** — persistent cancellation at non-negligible amplitude.
7. *Did later stages exhaust?* Stages 2 and 3 did (Branch B, at 314 and 353). **Stage 4 did not.**
8. *Genuine terminal exhaustion?* **No.** `CAP_HIT` at 1600.
9. *`M_nd` materially better?* **Yes** — 12.923 vs 23.360, −44.68 %, far inside the −20 % bar.
10. *Cost?* ×12.21 outer, ×29.28 inner MMA, ×88.60 wall — **outside the preregistered bound**, and almost all of it wasted: see §6.

**400×50**

1. **Yes.**
2. Production descended at **138**; the candidate at **389** — **251 further iterations**.
3. gray 0.3472 → 0.1822 (−0.165), mid 0.1881 → 0.0398 (−0.148). Figure F8 shows what this means physically: production freezes two large grey blobs at the ends of the beam; the candidate resolves them into a clean truss.
4. `M_nd` 32.318 → 15.665, **−51.53 %**.
5. ω₁ 162.877 → 166.418, **+2.17 %** — the design is not merely crisper, it is stiffer.
6. **B** — amplitude convergence with coherent motion, at exactly the iteration the inherited native stop first holds (369).
7. **Yes** — stages 2, 3 and 4 all exhausted, all Branch B, at 428, 467 and 505.
8. **Yes** — `CONVERGED` at 505 on a completed Branch-B window 486–505.
9. **Yes** — 15.331 vs 32.328, **−52.58 %**.
10. ×3.63 outer, ×3.53 inner, ×6.54 wall — **inside** the preregistered bound.

## 5. Phase 17 — the 160×20 coarse-mesh safety test

1. *Did Branch A recognise the mature high-amplitude cycle?* **Yes** — A's window ran 83–102, completing at 102 with `med₂₀cosθ = −0.907`, `med₂₀ net/path = 0.133` and amplitude `12.6 × tol`: persistent reversal, 87 % of the path length cancelled, and an amplitude an order of magnitude above the convergence scale. That window begins at **83**, the same iteration the frozen rule recorded on the 160×20 fixed-move arm.
2. *Did the move descend without indefinite churning?* **Yes** — first descent at 103, well inside the preregistered ≤ 400.
3. *Did it traverse the later levels?* **Yes** — 103, 142, 181, i.e. all four rungs, each after a completed window.
4. *ω₁ acceptable?* **170.011 vs 169.495, +0.304 %** — better than production.
5. *`M_nd` acceptable?* **12.704 vs 13.402, −5.21 %** — better than production.
6. *Any regression?* **None found.** Every reported quantity is equal or better; gray falls 0.1494 → 0.1444, volume error 1.3e-6.
7. *Did any branch fire pathologically early?* **No.** The earliest declaration the rule can make is `stageStart + 38`; the observed declarations were at 102, 141, 180, 219 — the first is 63 iterations later than the earliest possible, the rest are at the 38–39-iteration minimum, meaning each new rung entered its mature regime immediately.
8. *Terminal admission honest?* **Yes** — Branch B, window 200–219, at `move = 0.005`.

The coarse mesh is safe. The known weakness of the 160×20 mechanism case — that
Branch A fires while relative `M_nd` improvement remains available — did not
become a controller failure, because the *later rungs* recovered it: `M_nd` fell
13.036 → 12.704 across stages 2–4.

## 6. Where the benefit actually comes from

This is the most important structural finding, and it is uniform across all three
meshes.

| mesh | total Δ`M_nd` | of which, before the first descent | after the first descent |
|---|---|---|---|
| 160×20 | −5.21 % | **−2.64 %** | −2.57 % |
| 320×40 | −44.68 % | **−44.21 %** | −0.47 % |
| 400×50 | −52.58 % | **−51.53 %** | −1.05 % |

**The causal benefit is delayed first continuation, not the ladder below it.** At
the fine meshes ≥ 98 % of the improvement is banked before `move` ever leaves
0.04. Stages 2–4 cost iterations and buy almost nothing.

At 320×40 this is stark: the terminal stage ran **1248 iterations** during which
`M_nd` moved by 0.40 % of its own value and ω₁ by 0.017 %. Those 1248 iterations
are ≈ 78 % of the run and the overwhelming majority of its ×88.6 wall-time cost,
and they produced nothing.

## 7. Why 320×40 never terminated — the diagnosed failure

At the terminal rung the frozen union has a hole, and 320×40 falls straight into
it. Over all 1248 terminal-stage iterations:

```
amplitude  ‖Δρ‖₂  median 0.00487  =  0.049 x tol(NE)   ->  Branch A blocked: A requires ‖Δρ‖₂ >= tol
med₂₀cos θ        median  -0.885, < 0 on 100 % of them ->  Branch B blocked: B requires med₂₀cos θ > 0
                                                           A never true, B never true, nA = nB = 0
```

This is exactly **known limitation 1**, carried forward from the withheld 240×30
study: *"Branch A is blind to sufficiently low-amplitude cancellation."* At
240×30 that blindness was harmless because Branch B had already declared
exhaustion earlier. Here there is no earlier declaration to fall back on: the
regime *is* low-amplitude cancellation, from iteration 353 to the cap.

The contrast with the other two meshes isolates the cause precisely — it is
coherence, not amplitude:

| mesh | terminal amp / tol | terminal `med₂₀cos θ` | terminal branch |
|---|---|---|---|
| 160×20 | 0.0575 | **+0.921** | B fires |
| 400×50 | 0.0129 | **+0.511** | B fires |
| 320×40 | 0.0487 | **−0.885** | **neither** |

All three sit far below `tol` in amplitude. 320×40 differs only in that its
terminal motion *reverses direction*, which the coherence guard — the very clause
that makes Branch B safe at 160×20 — reads as "not converged".

**No repair is attempted here.** Under Phase 19 the controller is frozen; a
Branch C, a relaxed coherence guard, or an amplitude-scaled tolerance would each
be an outcome-driven change made after seeing this trajectory. The failure is
recorded as the finding it is.

## 8. Phase 18 — multiplicity and physics safety

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| ω₁, ω₂ finite | yes | yes | yes |
| ω₂ > ω₁ throughout | yes | yes | yes |
| subspace size (min/max) | 2 / 2 | 2 / 2 | 2 / 2 |
| minimum relative gap over the run | 0.00415 | 0.00414 | 0.00422 |
| final relative gap | 0.00833 | 0.22387 | 0.21079 |
| any NaN/Inf in ω or `M_nd` | none | none | none |
| inner MMA non-convergences | **0** | **0** | **0** |
| max inner iterations in any outer step | 53 | 139 | 46 |
| final volume | 0.49999913 | 0.49999962 | 0.49999943 |
| max \|volume − 0.5\| over the run | 4.3e-05 | 1.0e-04 | 1.6e-04 |

No multiplicity failure. The fixed two-mode subspace holds at size 2 on every
iteration of every run; the minimum relative gap is ≈ 0.004 on all three meshes,
i.e. the pair approaches coalescence to the same degree it always has, and the
off-diagonal treatment handles it as before.

The **final gaps differ substantially** from production at the fine meshes
(0.224 vs 0.107 at 320×40; 0.211 vs 0.077 at 400×50). That is expected and is not
a defect: the candidate converges to a different, more discrete design whose
second mode is further away. ω₁ — the objective — improves on all three meshes,
so the changed gap reflects a better optimum, not a lost one.

Volume feasibility holds: the final designs are within 6e-7 of the 0.5 constraint
on all three meshes. The transient maxima above (up to 1.6e-4 at 400×50) occur
mid-run and are the same order production exhibits.

## 9. Phase 4 — β's counterfactual, on the record

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| first β stall | 79 | 130 | 138 |
| iterations on which β stall held | 141 / 219 | 1471 / 1600 | 368 / 505 |
| production ladder, replayed on the candidate path, reaches rung 4 at | **101** | **152** | **160** |
| production stop test, replayed, would admit convergence at | **143** | **216** | **369** |
| candidate descents | 103, 142, 181 | 275, 314, 353 | 389, 428, 467 |
| candidate termination | 219 | never (cap) | 505 |

β would have ended every run far earlier and far worse. Its predicate was true
at each of the nine candidate transitions and drove none of them.

The replay is a **predicate replay on the realized path**, not a counterfactual
trajectory — production under its own rule would have left this path at its first
descent. It shows what production's rule *says* about a mature design, not what
production would have produced.

## 10. §6 prefix equivalence — the single-factor proof by construction

At stage 1 the candidate holds `move = 0.04` and changes nothing else, so its
prefix must equal a fixed-move arm. Against `F400_400x50_trajectory.mat` (same
mesh, same MATLAB build, 369 outer):

```
iterations compared      369
rho                      BITWISE IDENTICAL   (20000 x 369)
omega (all 5 modes)      BITWISE IDENTICAL
beta                     BITWISE IDENTICAL
||drho||_2               BITWISE IDENTICAL
inner iteration counts   BITWISE IDENTICAL
volume, relative gap     BITWISE IDENTICAL
candidate move           0.04 on every one of the 369
                                                   -> PASS
```

Two hundred and fifty-one of those iterations are ones production had already
abandoned. The controller changed *when* the ladder moves, and demonstrably
nothing else.
