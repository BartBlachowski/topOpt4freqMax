# PERFORMANCE-CAMPAIGN READINESS — Phase 24

The final scaling campaign uses the fixed nine meshes:

```
160x20  240x30  320x40  400x50  480x60  560x70  640x80  720x90  800x100
```

**This task did not run that campaign, and none of its runs were part of it.**
Exactly three candidate runs were executed: 160×20, 320×40, 400×50.

## Authorization conditions and their status

| condition | status |
|---|---|
| `TWO_BRANCH_CONTROLLER_VALIDATED` | **NO** — `TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED` |
| promotion equivalence `PASS` (if promotion occurred) | not applicable — no promotion was performed |
| production source/config frozen | production **behaviour** frozen and bitwise-verified; source tree carries a default-inactive controller layer, manifested at `edbfe47e…` |
| tests pass | **YES** — repository suite 5/5 (0 failures), controller suite 17/17 (0 failures) |
| evidence complete | **YES** — three trajectories declared and hash-valid; two prior-study fields marked unavailable, none fabricated |
| no scientific blocker remains from this controller study | **NO** — a blocker remains |

## The blocker

At 320×40 the controller **cannot admit termination**. Its terminal rung entered
a low-amplitude cancelling regime at iteration 353 and stayed there to the 1600
cap: amplitude ≈ 0.049 × `tol` blocks Branch A, `med₂₀cosθ ≈ −0.885` blocks
Branch B, so `E` was false on all 1248 iterations and both persistence counters
stayed at zero.

This matters more for the campaign than for any single run. The campaign's
purpose is *scaling*, and it would add five meshes finer than 400×50 — exactly
the direction in which amplitudes fall further relative to the move-independent
`tol(NE)`, i.e. deeper into the region where Branch A cannot fire. On the
evidence here, running the campaign under this controller would risk five more
capped runs at roughly the cost 320×40 showed: **34 374 s for one mesh, ×88.6
production wall time**, of which ≈ 78 % of the iterations produced an `M_nd`
change of 0.40 % of its own value.

## Decision

> ## `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

Two independent reasons, either sufficient:

1. the primary verdict is not `TWO_BRANCH_CONTROLLER_VALIDATED`;
2. an unresolved scientific blocker remains — terminal admission fails at
   320×40 through the frozen union's low-amplitude-cancellation hole.

The campaign also cannot proceed on the *production* controller on the strength
of this study: nothing here validates β-stall continuation, and §2 of
`BASELINES.md` records that production never reaches the bottom of its own move
ladder on any of the three meshes.
