# CROSS-MESH ANALYSIS — does one story hold across the meshes?

**No. And that is the finding.**

---

## 1. Three meshes, three different answers

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| branch that ended stage 1 | A | A | B |
| `M_nd` gain banked at S | 52.4 % | 99.2 % | 98.0 % |
| ω₁ gain banked at S | **−99.8 %** | 100.6 % | 98.9 % |
| lower rungs material? | **YES** (`M_nd` **and** ω₁) | **NO** | **YES** (`M_nd`, marginal) |
| four-rung outcome | `CONVERGED` @219 | **`CAP_HIT` @1600** | `CONVERGED` @505 |
| lower-rung cost | ×1.15 of S | **×4.84 of S** | ×0.30 of S |

The lower ladder is **load-bearing at 160×20, marginal at 400×50, and actively
harmful at 320×40**. Any single sentence about "the ladder" is wrong on at least
one mesh.

## 2. Why 160×20 is different

The coarse mesh is the only one whose `move = 0.04` stage ends in a
**high-amplitude cancelling** regime:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `‖Δρ‖₂ / tol` at S | **12.59** | 1.03 | 0.64 |
| `max|Δρ|/move` at S | **0.9996** | 0.6724 | 0.1426 |
| bound-active fraction at S | **0.0713** | 0 | 0 |
| `cosθ` at S | −0.9197 | −0.6753 | +0.9986 |
| ω₁ peak-to-peak in the window | **0.7708** | 0.0352 | 0.0582 |

At 160×20 the design is still slamming into the move bound (`max|Δρ|` is
99.96 % of the move limit, 7 % of elements bound-active) and swinging ω₁ by 0.77
between iterations. Branch A correctly calls that regime exhausted **for
`move = 0.04`** — the stage genuinely cannot do better at that step size. But the
design is not converged; it is *oscillating at the resolution of its own move
limit*. Shrinking the move is exactly what lets it settle, which is why rung 2
alone buys +0.495 % ω₁ there.

At 320×40 and 400×50 the amplitude at S is already at or below `tol`, the bound
is nowhere active, and there is correspondingly little for a smaller step to
recover.

**This is the mechanism, and it predicts the direction of the coarse/fine split
rather than merely describing it:** the lower rungs pay off exactly where the
`move = 0.04` stage terminates while still bound-limited.

## 3. Why 320×40 is different again

320×40's terminal rung is the one place the frozen union has no answer. Over all
1 248 terminal-stage iterations:

```
amplitude  ~0.049 x tol   ->  Branch A blocked (A requires >= tol)
med20 cos  ~ -0.885       ->  Branch B blocked (B requires > 0)
                              E false on every iteration; nA = nB = 0
```

Low-amplitude cancellation — the hole recorded as known limitation 1 before the
controller was ever built. Cost: 70 034 inner MMA iterations and 32 975 s (91.5 %
of the run's inner work, 95.9 % of its wall time) to move `M_nd` by 0.13 % of its
own value.

Note the ordering: 320×40 reached that trap **by descending**. Amplitude falls
with the move limit while `tol(NE)` does not, so each rung pushes a mesh further
into the region where Branch A cannot fire. The ladder created the conditions for
its own terminal failure.

## 4. The one thing that is uniform

Stage 1 does the overwhelming majority of the work on every mesh:

| mesh | rung-1 Δ`M_nd` | rungs 2–4 combined |
|---|---|---|
| 160×20 | −86.42 (−86.9 %) | −0.33 (−2.5 %) |
| 320×40 | −86.60 (−86.9 %) | −0.09 (−0.7 %) |
| 400×50 | −84.02 (−84.3 %) | −0.33 (−2.1 %) |

and rungs 3 and 4 are immaterial everywhere: ≤ 0.44 % `M_nd`, ≤ 0.094 % ω₁, on
every mesh.

So the evidence does not support "the four-rung ladder is unnecessary". It
supports something narrower and more specific: **rung 2 sometimes matters; rungs
3 and 4 never did, on any mesh, by any preregistered criterion.** Acting on that
observation is outside this task's scope (Phase 18 forbids proposing a new ladder
architecture) and is recorded as an observation, not a recommendation.

## 5. Consistency with the supporting mesh

240×30 (fixed move 0.04, no ladder, cap 1200) exhausted on **Branch B at 187**
with `‖Δρ‖₂ = 0.0507` against `tol = 0.075`, i.e. `0.68 × tol` and **not**
bound-limited — the 320×40/400×50 pattern, not the 160×20 one. Continuing 1 013
further iterations at `move = 0.04` made `M_nd` **worse** by 4.77 %.

It is consistent with the picture in §2, and it cannot test the ladder because no
four-rung 240×30 run exists.

## 6. What generalizes to the nine-mesh campaign

The campaign would add 480×60 … 800×100 — all finer than 400×50. On the evidence:

* the coarse-mesh mechanism (bound-limited stage-1 exit) **weakens** with
  refinement: `‖Δρ‖₂/tol` at S falls 12.59 → 1.03 → 0.64 across 160/320/400, and
  bound activity is already zero at 320×40. Lower-rung benefit should therefore
  shrink further at finer meshes;
* the 320×40 failure mechanism **strengthens** with refinement, for the same
  reason: amplitude keeps falling against a move-independent `tol(NE)`, pushing
  more meshes into the region where Branch A cannot fire.

That combination — less benefit, more failure risk, at exactly the meshes the
campaign adds — is why the campaign stays blocked regardless of this audit's
architecture verdict.
