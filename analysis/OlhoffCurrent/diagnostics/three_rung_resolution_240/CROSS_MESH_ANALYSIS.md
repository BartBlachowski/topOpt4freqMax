# CROSS-MESH ANALYSIS — rung 4 across four meshes

Phase 21. Descriptive only. **No scaling law is fitted from four points.**

---

## 1. Events and terminal status

| mesh | `NE` | S1 | S2 | S3 | F | status | stage-4 offset |
|---|---|---|---|---|---|---|---|
| 160×20 | 3 200 | 102 (A) | 141 (A) | 180 (B) | 219 | CONVERGED | 38 |
| **240×30** | **7 200** | **206 (B)** | **245 (B)** | **284 (B)** | **1358** | **CONVERGED** | **1073** |
| 320×40 | 12 800 | 274 (A) | 313 (B) | 352 (B) | 1600 | **CAP_HIT** | never |
| 400×50 | 20 000 | 388 (B) | 427 (B) | 466 (B) | 505 | CONVERGED | 38 |

Every S2 and S3 across all four meshes declares at exactly `stageStart + 38`.
Stage 4 is the outlier: minimum latency at 160×20 and 400×50, **1073** at
240×30, and **never** at 320×40.

## 2. Rung-4 effect

| mesh | `Δω₁` rel. | `ΔM_nd` rel. | mean \|Δρ_e\| | Δgap₁₂ | Δsubspace | material? |
|---|---|---|---|---|---|---|
| 160×20 | +0.02025 % | −0.4078 % | 0.000424 | −0.000767 | 0 | **no** |
| **240×30** | **+0.00672 %** | **+0.2010 %** | **0.002120** | −0.001306 | 0 | **no** |
| 320×40 | **−0.00504 %** | −0.1297 % | 0.001723 | +0.000634 | 0 | **no** |
| 400×50 | +0.00815 % | −0.2742 % | 0.000208 | +0.000146 | 0 | **no** |
| **bar** | **0.10 %** | **2 %** | **0.01** | not a bar | any change | |

## 3. Rung-4 cost

| mesh | outer after S3 | % of run | inner MMA after S3 | % of run |
|---|---|---|---|---|
| 160×20 | 39 | 17.8 % | 791 | 15.6 % |
| **240×30** | **1 074** | **79.1 %** | **38 675** | **87.5 %** |
| 320×40 | 1 248 | 78.0 % | 70 034 | 91.5 % |
| 400×50 | 39 | 7.7 % | 1 453 | 14.1 % |

## 4. What the pattern looks like — stated descriptively, not fitted

Two of the four meshes (160×20, 400×50) have a rung 4 that declares at the
arithmetic minimum and costs 14–18 % of the run. The other two (240×30, 320×40)
have a rung 4 that enters **low-amplitude cancellation** — `‖Δρ‖₂ < tol` together
with `med₂₀ cosθ < 0`, the documented hole that satisfies neither branch — and
costs 87–92 % of the run.

At 240×30 that regime held for **92.4 %** of stage 4 and `E` was true only 5.9 %
of the time; the run escaped after 1073 iterations when a 20-iteration coherent
window finally appeared. At 320×40 no such window appeared before the cap.

**The new evidence therefore reframes the 320×40 `CAP_HIT`**: it is not a
one-mesh anomaly but the same terminal regime seen here, distinguished only by
whether a qualifying window happens to appear before the cap. That is a
statement about the frozen rule's behaviour at `move = 0.005`, recorded as an
observation.

**No mesh law is fitted.** Four points, two of which show one behaviour and two
the other, with no ordering in `NE` (`NE` = 3 200 fast, 7 200 slow, 12 800 never,
20 000 fast). Any attempt to extract a scaling from that would be unsupported,
and none is made.

## 5. The one thing four meshes do agree on

Rung 4's contribution to the maximized objective spans **−0.005 % to +0.020 %**
across a 6× range of element count. Every value is at least 5× below the frozen
0.10 % bar, and one is negative. On `M_nd`, topology, multiplicity, gap and
volume the same holds at every mesh.

That agreement — not any trend — is what supports the architecture verdict.
