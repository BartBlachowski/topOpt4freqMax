# PERFORMANCE-CAMPAIGN READINESS

The frozen nine meshes:

```
160x20  240x30  320x40  400x50  480x60  560x70  640x80  720x90  800x100
```

**This task ran nothing.** Zero scientific optimization runs of any mesh.

## Authorization conditions

| condition | status |
|---|---|
| a validated continuation controller exists | **NO** — `TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED`, unchanged by this audit |
| the move-ladder architecture is settled | **NO** — `FOUR_RUNG_LADDER_PARTIALLY_USEFUL` / `RETAIN_MOVE_LADDER_PENDING_REDESIGN` |
| production frozen and unchanged | yes — `+impl/` tree byte-identical, `edbfe47eb…152cb` |
| tests pass | yes — repository suite 6/6, 0 failures (including the new finalization gate) |
| evidence complete and hash-valid | yes for this audit and for the controller study; **no** for eight legacy studies (`RETENTION_AUDIT.md`) |
| no scientific blocker remains | **NO** — two |

## The two blockers

**1. The controller cannot terminate at 320×40.** Unresolved from the previous
study: the terminal rung enters low-amplitude cancellation, where amplitude
≈ 0.049 × `tol` blocks Branch A and `med₂₀cosθ ≈ −0.885` blocks Branch B, so
`E` is false for all 1 248 terminal-stage iterations and the run caps.

**2. The ladder architecture is unsettled, and this audit makes that sharper
rather than softer.** The lower rungs are load-bearing at 160×20 (without them
the design is *worse than production* in ω₁), immaterial at 320×40, and marginal
at 400×50. Rungs 3 and 4 are immaterial on every mesh tested.

## Why this audit makes the campaign a worse bet than before

The campaign adds 480×60 … 800×100 — **all finer than 400×50**. Two trends run in
opposite and unhelpful directions:

| | 160×20 | 320×40 | 400×50 | direction |
|---|---|---|---|---|
| `‖Δρ‖₂ / tol` at the stage-1 exit | 12.59 | 1.03 | 0.64 | **falling** |
| bound-active fraction at the exit | 0.0713 | 0 | 0 | **gone by 320×40** |
| lower-rung benefit | material (×2) | none | marginal | **weakening** |

* The mechanism that makes lower rungs valuable — exiting stage 1 while still
  bound-limited — has already vanished by 320×40 and should not return at finer
  meshes. So the ladder's benefit is expected to keep shrinking.
* The mechanism that made 320×40 fail — amplitude falling against a
  move-independent `tol(NE)` until Branch A cannot fire — **strengthens** with
  refinement. So the failure risk is expected to grow.

Running five finer meshes under the current controller therefore risks repeating
the 320×40 outcome — which cost **34 374 s for one mesh, ×88.6 production wall
time**, with 91.5 % of its inner work spent on a rung that moved `M_nd` by 0.13 %
of its own value.

## Decision

> ## `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

Unchanged, and now with a second, independent reason. The campaign also cannot
proceed on the *production* controller: nothing in this study or its predecessors
validates β-stall continuation, which on all three meshes descends at the β stall
and then stops 1–12 iterations later without ever reaching the bottom of its own
declared ladder.
