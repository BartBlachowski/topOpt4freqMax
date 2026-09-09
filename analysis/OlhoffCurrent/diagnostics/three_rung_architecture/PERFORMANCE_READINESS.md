# PERFORMANCE READINESS — cost of terminating at S3, and why the campaign stays blocked

---

## 1. Wall time remains unreliable, and decides nothing

Seconds per **inner MMA iteration** should be near-constant at a fixed mesh. It is
not, on any of the three trajectories — the two-rung audit's finding is
reproduced here with the rung boundaries refined:

| mesh | first 50 | rung 1 | rung 2 | rung 3 | rung 4 | drift |
|---|---|---|---|---|---|---|
| 160×20 | 0.0381 s | 0.0497 s | 0.0878 s | 0.1333 s | 0.1563 s | **4.10×** |
| 320×40 | 0.0910 s | 0.1557 s | 0.3253 s | 0.5188 s | 0.4708 s | **5.70×** |
| 400×50 | 0.1343 s | 0.2348 s | 0.4902 s | 0.6749 s | 0.6374 s | **5.03×** |

The mesh, the FE assembly, the eigensolve and the MMA sub-problem do not change
size within a run, so a 4–6× drift in cost per inner iteration is a property of
the machine during these runs, not of the algorithm. Wall times are reported for
completeness and **explicitly down-weighted**, exactly as `PREREGISTRATION.md` §13
requires. No timing is fabricated and **no verdict depends on a wall-clock
number**.

**The durable cost metrics used throughout are outer iterations and cumulative
inner MMA iterations.**

## 2. Cost saved by terminating at S3

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| four-rung total outer | 219 | **1 600 (CAP_HIT)** | 505 |
| three-rung outer (S3) | 180 | 352 | 466 |
| **outer saved** | 39 (**17.8 %**) | **1 248 (78.0 %)** | 39 (**7.7 %**) |
| four-rung total inner MMA | 5 074 | **76 532** | 10 301 |
| three-rung inner MMA (S3) | 4 283 | 6 498 | 8 848 |
| **inner MMA saved** | 791 (**15.6 %**) | **70 034 (91.5 %)** | 1 453 (**14.1 %**) |
| wall saved (unreliable) | 124 s (29.1 %) | 32 975 s (95.9 %) | 926 s (26.2 %) |

The saving is modest at the two meshes where rung 4 terminates (14–18 % of work)
and overwhelming at the one where it does not.

**No new numeric savings threshold was invented.** `PREREGISTRATION.md` §12
declines to define one, because the per-rung cost figures were already visible to
this audit from the prior task, and any fresh bar would have been chosen with
knowledge of the answer. The only binary cost bars are the inherited ones:

| inherited criterion | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| rung 4 cost multiplier vs S3 (bar: ≥ 2× while sub-material) | 0.22× | **3.55×** | 0.08× |
| **cost-dominated?** | no | **yes** | no |
| **failure risk (`CAP_HIT` / non-terminating stage)?** | no | **yes** | no |

## 3. Where the 320×40 work actually goes

| rung | move | outer | inner MMA | share of total inner | `Δω₁` over the rung | `ΔM_nd` over the rung |
|---|---|---|---|---|---|---|
| 1 | 0.04 | 274 | 5 066 | 6.6 % | +143.75 % | −86.94 % |
| 2 | 0.02 | 39 | 686 | 0.9 % | −0.0032 % | −0.249 % |
| 3 | 0.01 | 39 | 746 | 1.0 % | +0.0066 % | −0.305 % |
| 4 | 0.005 | **1 248** | **70 034** | **91.5 %** | **−0.0050 %** | −0.130 % |

Rung 4 at 320×40 spends 91.5 % of the entire inner budget, moves `ω₁` the wrong
way, and never terminates. Terminating at S3 removes it in full and converts
`CAP_HIT @1600` into `CONVERGED @352` under the same frozen rule.

## 4. Cost against production

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| production outer / inner | 91 / 2 241 | 131 / 2 614 | 139 / 2 918 |
| S2 outer × / inner × | 1.55× / 1.63× | 2.39× / 2.20× | 3.07× / 2.75× |
| **S3 outer × / inner ×** | **1.98× / 1.91×** | **2.69× / 2.49×** | **3.35× / 3.03×** |
| F outer × / inner × | 2.41× / 2.26× | **12.21× / 29.28×** | 3.63× / 3.53× |
| preregistered gate A7 (outer ≤ 8×) at S3 | **PASS** | **PASS** | **PASS** |

At 320×40 the four-rung controller costs **29× production's inner work** and still
does not converge. The three-rung endpoint costs 2.5× and does converge. That
remains the strongest operational argument in this line of work — and it was
already true of the two-rung endpoint at 2.2×.

## 5. What is *not* claimed

* **No performance claim is promoted.** These are the costs of three existing
  causal-controller trajectories, sliced at four points. They are not a benchmark,
  not a scaling law, and not evidence about any mesh not listed.
* **No timing claim is made at all**, given §1.
* Nothing here concerns 240×30, 800×100, or the nine-mesh grid.
* The bound-limitation statistics in `REPORT.md` §4 are descriptive; no `NE` law,
  mesh scaling or move scaling is fitted from three meshes.

## 6. Campaign status

The nine-mesh performance campaign remains
**`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`** in this task, unconditionally by the
brief's Phase 24 — and independently because the architecture verdict is
`PARTIALLY_SUPPORTED`, not `SUPPORTED`, so no policy is frozen to run a campaign
with.

Production remains **`PRODUCTION_CONTROLLER_NOT_CHANGED`**: `move.levels` still
resolves to `[0.04 0.02 0.01 0.005]` and `move.continuation.signal` to
`boundVariable`.
