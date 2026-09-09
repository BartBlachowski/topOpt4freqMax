# PERFORMANCE READINESS — cost of terminating at S2, and why the campaign stays blocked

---

## 1. Wall time is not trustworthy here, and is not used to decide anything

Seconds per **inner MMA iteration** should be near-constant at a fixed mesh. It
is not, on any of the three trajectories:

| mesh | first 50 iters | rung 1 | rung 2 | rungs 3+4 | drift |
|---|---|---|---|---|---|
| 160×20 | 0.0381 s | 0.0497 s | 0.0878 s | 0.1461 s | **3.83×** |
| 320×40 | 0.0910 s | 0.1557 s | 0.3253 s | 0.4713 s | **5.18×** |
| 400×50 | 0.1343 s | 0.2348 s | 0.4902 s | 0.6508 s | **4.85×** |

The mesh, the FE assembly, the eigensolve and the MMA sub-problem do not change
size within a run, so a 4–5× drift in cost per inner iteration is a property of
the machine during these runs, not of the algorithm. Elapsed times are therefore
reported for completeness and **explicitly down-weighted**, exactly as
`PREREGISTRATION.md` §11 requires. No timing is fabricated, and no verdict
depends on a wall-clock number.

**The durable cost metrics used throughout are outer iterations and cumulative
inner MMA iterations.**

## 2. Cost saved by terminating at S2

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| four-rung total outer | 219 | **1 600 (CAP_HIT)** | 505 |
| two-rung outer (S2) | 141 | 313 | 427 |
| **outer saved** | 78 (**35.6 %**) | **1 287 (80.4 %)** | 78 (**15.4 %**) |
| four-rung total inner MMA | 5 074 | **76 532** | 10 301 |
| two-rung inner MMA (S2) | 3 647 | 5 752 | 8 037 |
| **inner MMA saved** | 1 427 (**28.1 %**) | **70 780 (92.5 %)** | 2 264 (**22.0 %**) |
| wall saved (unreliable) | 208 s (49.0 %) | 33 362 s (97.1 %) | 1 473 s (41.6 %) |

The saving is material on every mesh by the durable metrics: at worst 15.4 % of
outer iterations and 22.0 % of inner work, at best 80.4 % and 92.5 %.

## 3. Where the 320×40 work actually goes

| rung | move | outer | inner MMA | share of total inner | `ΔM_nd` over the rung | `Δω₁` over the rung |
|---|---|---|---|---|---|---|
| 1 | 0.04 | 274 | 5 066 | 6.6 % | −86.94 % rel. | +143.75 % rel. |
| 2 | 0.02 | 39 | 686 | 0.9 % | −0.249 % | −0.0032 % |
| 3 | 0.01 | 39 | 746 | 1.0 % | — | — |
| 4 | 0.005 | **1 248** | **70 034** | **91.5 %** | — | — |
| 3+4 combined | | 1 287 | 70 780 | 92.5 % | **−0.435 %** | **+0.0016 %** |

Rung 4 at 320×40 spends 91.5 % of the entire inner budget and never terminates.
That is the `CAP_HIT`. Terminating at S2 removes it entirely.

## 4. Cost against production

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| production outer / inner | 91 / 2 241 | 131 / 2 614 | 139 / 2 918 |
| **S2** outer mult. / inner mult. | 1.55× / 1.63× | 2.39× / 2.20× | 3.07× / 2.75× |
| F outer mult. / inner mult. | 2.41× / 2.26× | **12.21× / 29.28×** | 3.63× / 3.53× |
| preregistered gate A7 (outer ≤ 8×) at S2 | **PASS** | **PASS** | **PASS** |

At 320×40 the four-rung controller costs **29× production's inner work** and
still does not converge. The two-rung endpoint costs 2.2× and does converge.
This is the single strongest operational argument in the audit.

## 5. What is *not* claimed

* **No performance claim is promoted.** These are the costs of three existing
  causal-controller trajectories, sliced at three points. They are not a
  benchmark, not a scaling law, and not evidence about any mesh not listed.
* **No timing claim is made at all**, given §1.
* Nothing here concerns 240×30, 800×100, or the nine-mesh grid.

## 6. Campaign status

The nine-mesh performance campaign remains **`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`**
in this task, unconditionally and by the brief's own Phase 22 — and independently
because the architecture verdict is `PARTIALLY_SUPPORTED`, not `SUPPORTED`, so no
policy is frozen to run a campaign with.

Production remains **`PRODUCTION_CONTROLLER_NOT_CHANGED`**.
