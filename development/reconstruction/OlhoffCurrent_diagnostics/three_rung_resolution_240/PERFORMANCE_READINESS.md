# PERFORMANCE READINESS — cost of the final rung, and campaign status

---

## 1. Wall time is again unreliable and decides nothing

Seconds per inner MMA iteration within this single run:

| block | s/inner |
|---|---|
| first 50 iterations | 0.0614 |
| rung 1 (0.04) | 0.0965 |
| rung 2 (0.02) | 0.2131 |
| rung 3 (0.01) | 0.3254 |
| rung 4 (0.005) | 0.3456 |
| **drift** | **5.63×** |

The mesh, assembly, eigensolve and MMA sub-problem do not change size within a
run, so a 5.6× drift in cost per inner iteration is a property of the machine,
not the algorithm. This reproduces the 3.8–5.7× drift measured in every prior
causal arm. Wall times are reported for completeness and **explicitly
down-weighted**; **no verdict depends on a wall-clock number**, exactly as
`PREREGISTRATION.md` §11 requires.

**Primary cost evidence: outer iterations and inner MMA iterations.**

## 2. Cost of rung 4 at 240×30

| | value |
|---|---|
| total outer | 1 358 |
| outer to `S3` | **284** |
| **outer after `S3`** | **1 074 (79.1 %)** |
| total inner MMA | 44 181 |
| inner MMA to `S3` | **5 506** |
| **inner MMA after `S3`** | **38 675 (87.5 %)** |
| rung-4 cost multiplier vs `S3` | **3.78×** |
| cost-dominated (inherited criterion) | **yes** |
| wall after `S3` | 13 365 s of 14 160 s — *reported, not relied upon* |

Terminating at `S3` would have finished this run in **284 outer iterations
instead of 1 358**, using **one eighth** of the inner MMA work, for a design that
differs from `F` by less than a quarter of the topology bar and less than a
fifteenth of the objective bar.

## 3. Per-rung breakdown

| rung | move | range | outer | % outer | inner MMA | % inner |
|---|---|---|---|---|---|---|
| 1 | 0.04 | 1–206 | 206 | 15.2 % | 4 030 | 9.1 % |
| 2 | 0.02 | 207–245 | 39 | 2.9 % | 663 | 1.5 % |
| 3 | 0.01 | 246–284 | 39 | 2.9 % | 813 | 1.8 % |
| **4** | **0.005** | **285–1358** | **1 074** | **79.1 %** | **38 675** | **87.5 %** |

Rungs 2 and 3 together cost 78 outer iterations and 1 476 inner MMA iterations —
3.3 % of the inner budget. Rung 4 alone costs 26× that.

## 4. Cross-mesh cost of rung 4

| mesh | outer after S3 | % of run | inner after S3 | % of run | terminal status |
|---|---|---|---|---|---|
| 160×20 | 39 | 17.8 % | 791 | 15.6 % | CONVERGED |
| **240×30** | **1 074** | **79.1 %** | **38 675** | **87.5 %** | CONVERGED |
| 320×40 | 1 248 | 78.0 % | 70 034 | 91.5 % | **CAP_HIT** |
| 400×50 | 39 | 7.7 % | 1 453 | 14.1 % | CONVERGED |

On two of four meshes rung 4 consumes roughly seven eighths of the entire budget.
On one of those it never terminates at all.

## 5. No 240×30 production baseline

There is none, so no production cost multiplier is reported for this mesh. It is
recorded as `UNAVAILABLE` rather than estimated from a neighbouring mesh.

## 6. What is not claimed

* **No performance claim is promoted.** This is one trajectory at one mesh,
  sliced at four points. It is not a benchmark and not a scaling law.
* **No timing claim is made at all**, given §1.
* Nothing here concerns 800×100 or the nine-mesh grid.

## 7. Status

Production remains **`PRODUCTION_CONTROLLER_NOT_CHANGED`** — the preset still
resolves to `move.levels = [0.04 0.02 0.01 0.005]` with
`move.continuation.signal = 'boundVariable'`, verified in the static audit.

The nine-mesh performance campaign remains
**`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`** in this task, by the brief's Phase 26,
**unconditionally and even though the three-rung architecture is supported**.
Authorizing it is the business of a later task that first freezes and promotes
the policy.
