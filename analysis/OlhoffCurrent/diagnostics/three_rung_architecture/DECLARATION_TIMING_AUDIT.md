# DECLARATION TIMING AUDIT — when does the frozen rule fire, and why

Phase 14 of the brief. **Descriptive only.** No rule, window, persistence,
threshold or move value is altered anywhere in this task.

---

## 1. The arithmetic of the earliest possible declaration

From `+impl/architecture/+olh/+move/exhaustion.m`, with stage start `s`:

```
amp(j)      defined for j >= s
cos(j)      defined for j >= s+1        (needs d_{j-1} in-stage)
net(j)      defined for j >= s+9        (needs 10 in-stage steps; anchor rho_{s-1})
med20 x (j) defined only once the whole trailing 20-window lies inside the stage,
            i.e.  j >= s + W - 1  =  s + 19
```

so the predicate `E` is **first mathematically evaluable at `s + 19`**, and the
persistence counter then needs `P = 20` consecutive hits, giving

```
earliest arithmetically possible declaration  =  s + 19 + 19  =  s + 38
```

This is not a new derivation: `two_branch_controller_validation/scripts/cv_tests.m`
test 4 asserts exactly this boundary (19 → no declaration, 20 → declares, at
`stageStart + 38`). Here it is verified against the trajectories.

**Verification that theory matches the data:** for all 12 stages across the three
meshes, the first iteration at which `med₂₀ cosθ` is non-`NaN` in the offline
replay equals `stageStart + 19` exactly. `first_evaluable_matches_theory = true`.

## 2. The measured table

| mesh | stage | move | stageStart | first evaluable (`s+19`) | earliest possible (`s+38`) | actual declaration | offset | at earliest? | `E` true at first evaluable? | `E` unbroken to declaration? |
|---|---|---|---|---|---|---|---|---|---|---|
| 160×20 | 1 | 0.04 | 1 | 20 | 39 | **102** | **101** | no | **no** | — |
| 160×20 | 2 | 0.02 | 103 | 122 | 141 | **141** | **38** | **yes** | **yes** | yes |
| 160×20 | 3 | 0.01 | 142 | 161 | 180 | **180** | **38** | **yes** | **yes** | yes |
| 160×20 | 4 | 0.005 | 181 | 200 | 219 | **219** | **38** | **yes** | **yes** | yes |
| 320×40 | 1 | 0.04 | 1 | 20 | 39 | **274** | **273** | no | **no** | — |
| 320×40 | 2 | 0.02 | 275 | 294 | 313 | **313** | **38** | **yes** | **yes** | yes |
| 320×40 | 3 | 0.01 | 314 | 333 | 352 | **352** | **38** | **yes** | **yes** | yes |
| 320×40 | 4 | 0.005 | 353 | 372 | 391 | **never** | — | — | **no** | — |
| 400×50 | 1 | 0.04 | 1 | 20 | 39 | **388** | **387** | no | **no** | — |
| 400×50 | 2 | 0.02 | 389 | 408 | 427 | **427** | **38** | **yes** | **yes** | yes |
| 400×50 | 3 | 0.01 | 428 | 447 | 466 | **466** | **38** | **yes** | **yes** | yes |
| 400×50 | 4 | 0.005 | 467 | 486 | 505 | **505** | **38** | **yes** | **yes** | yes |

## 3. What is reproduced

* **The `stageStart + 38` observation from the two-rung audit is reproduced
  exactly.** All 9 lower stages (stages 2–4 on three meshes) were examined; 8 of
  them fire, and **all 8 fire at offset exactly 38** — the minimum arithmetically
  possible.
* **In all 8, `E` is already true at the first iteration at which it can be
  evaluated** (`stageStart + 19`), and remains **unbroken** through the whole
  persistence window to the declaration.
* **Stage 1 is the sole exception on every mesh**: offsets 101 / 273 / 387, and
  `E` is *false* at the first evaluable iteration in all three cases. Stage 1 is
  where the rule actually observes something develop.
* **The one lower stage that does not fire is 320×40's stage 4** (`move = 0.005`),
  where `E` is false at the first evaluable iteration (372) and never becomes
  persistently true through iteration 1600. That is the `CAP_HIT`.

## 4. The narrow supported conclusion

> **The lower-rung exhaustion detector is not observing a newly developed
> dynamical transition within those stages; the exhaustion condition is already
> satisfied once sufficient post-transition history exists.**

That is the whole of what this evidence supports. Operationally, every lower rung
that terminates is exactly **39 outer iterations** long, and its length is set by
the window and persistence arithmetic rather than by anything the design does.

The mechanism is visible in the data: the move limit halves at the descent, so
`‖Δρ‖₂` drops mechanically, and at the fine meshes it is already below `tol`
before the stage begins. Branch B (`amp < tol` and `med cos > 0`) is then true
from the moment the medians become defined. The detector's stage-local reset —
which exists precisely so that a mechanically halved amplitude cannot be read as
convergence across a transition — delays the declaration by the full window and
persistence, but does not prevent it.

## 5. What is NOT claimed

The following would each require a separate test that this task does not perform,
and none of them is asserted anywhere in this study:

* ✗ that 39 iterations are optimal;
* ✗ that the persistence window is unnecessary;
* ✗ that lower stages should be replaced by a fixed dwell;
* ✗ that the controller should descend immediately on a declaration at the
  previous level;
* ✗ that history should be inherited across transitions;
* ✗ that `W`, `P` or `W_np` should change.

`used_to_change_policy = false` is recorded in `METRICS.json`. Branch A, Branch B,
the union, the windows, the persistence and the reset semantics are **unchanged**,
and the offline replay reproduces the in-loop trace element-wise in all 12 stages
as proof that they are.

## 6. Bearing on the architecture question

This observation is **context, not evidence for or against the three-rung
ladder**. It explains why every lower rung costs the same 39 iterations, and it
explains why 320×40's `move = 0.005` stage is the outlier — but the architecture
verdict in `REPORT.md` rests on the materiality of what each rung *achieves*, not
on how long each takes to declare.

One connection is worth stating, carefully: because each lower stage runs for a
fixed 39 iterations regardless of mesh or state, the amount of design change a
lower rung delivers is whatever 39 iterations at that move limit happen to
produce. That is consistent with the `THRESHOLD_SPLITTING` finding in
`RUNG_VALUE_DECOMPOSITION.md` §4 — a continuum of small, similarly-sized
contributions rather than discrete captured effects — but it does not prove it,
and no rule is changed on the strength of either.
