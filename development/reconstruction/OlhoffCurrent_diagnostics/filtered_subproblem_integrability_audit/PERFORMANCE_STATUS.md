# PERFORMANCE_STATUS — Part 16

## Verdict

```
PERFORMANCE_CAMPAIGN_STILL_BLOCKED
```

The nine-mesh campaign was not run and remains blocked. This audit did not
remove the blocker the previous two studies established; it identified what the
blocker actually is.

## What changed, and what did not

`gray_kkt_forensic_audit` issued
`PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE` on the grounds that the
saved designs are not KKT-stationary for the physical relaxed problem. That
verdict stands, and this audit sharpens it in a way that makes resumption
*harder* to justify, not easier:

* the filtered update field is **not the gradient of any scalar function**
  locally, so there is no optimization problem — physical or regularized — of
  which the endpoints are stationary points;
* the final inner subproblem is **not solved to its own KKT** either, and
  solving it further drives the increment from 6 % to 88 % of the move limit, so
  the algorithm's behaviour depends on where the inner loop is truncated.

A performance campaign measures the cost of reaching a converged design. Both
findings bear directly on what "converged" means for this algorithm, and neither
is resolved.

## Why resuming as a heuristic reconstruction is not offered

`PERFORMANCE_CAMPAIGN_CAN_RESUME_AS_HEURISTIC_RECONSTRUCTION` would be
defensible if the algorithm's endpoints were reproducible, well-defined fixed
points that could honestly be labelled "the output of a documented heuristic".
Two things block that label today:

1. **The endpoint depends on a truncation parameter, not only on the
   formulation.** `tolInner = 0.05` is a relative-step test, and the
   certification shows the same subproblem, solved further, moves the design
   14× more per outer iteration. A timing benchmark of a scheme whose endpoint
   is set by an inner stopping tolerance would be measuring that tolerance as
   much as the method.
2. **The campaign's purpose is mesh convergence.** The three-rung canary study
   already showed M_nd rising monotonically with refinement (12.9 → 15.4 → 26.3
   → 34.4 % at 320/400/480/800) under a correctly functioning controller. Nine
   meshes would document that trend nine times. Explaining it requires the
   diagnosis, not the campaign.

## What would change the status

Not this audit's findings alone. The decision is the owner's, and the honest
options are:

* **resolve the inner-solve question first** (`NEXT_ACTION.md`) — determine what
  the algorithm does when its surrogate is solved accurately, and whether the
  endpoint is then well defined;
* **or** accept the scheme explicitly as a heuristic whose endpoint is
  defined by its stopping rules, state that limitation in the paper, and run the
  campaign as a timing study of *that* documented procedure — a legitimate
  choice, but one that must be made deliberately and written down, not arrived
  at by default.

This audit does not choose between them and recommends neither. It records that
the second option is now a *statable* position, which it was not before: the
scheme can be described precisely as a fixed-point iteration on an explicitly
characterized non-conservative field, rather than as an optimization method
whose convergence theory is assumed.

## Not recommended

Projection and p-continuation both remain outside what this evidence supports —
see `NEXT_ACTION.md` §4. Nothing here is a reason to start either.
