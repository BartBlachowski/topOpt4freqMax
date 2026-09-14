# RUNG VALUE DECOMPOSITION — what rung 2 buys, and what rungs 3+4 buy after it

Thresholds are those of `PREREGISTRATION.md` §9, inherited verbatim from
`move_ladder_necessity/PREREGISTRATION.md` §6 (SHA-256 `a08b879b…`). They were
fixed before any of the numbers below were computed, and are the same bars the
previous audit applied to rungs 2+3+4 as a block.

```
M_nd         material if the block improves M_nd by >= 2 % relative
omega1       material if the block improves omega1 by >= 0.10 % relative
topology     material if |d gray| >= 0.01 or |d mid| >= 0.01 or mean|d rho_e| >= 0.01
volume       material if |volume - 0.5| worsens by >= 1e-5
multiplicity material if subspace size leaves 2, mode order changes, omega2 <= omega1,
             or a NaN/Inf appears.  GAP MAGNITUDE ALONE IS NOT MATERIAL.
cost         a block is cost-dominated if it costs >= 2x the outer iterations used to
             reach the state it starts from while being sub-material on every metric
failure risk any rung sequence producing CAP_HIT or a non-terminating stage
```

---

## 1. RUNG 2 (move = 0.02):  S1 → S2

| quantity | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ΔM_nd` | −0.2479 (**−1.902 %**) | −0.0324 (−0.249 %) | −0.2236 (−1.428 %) |
| `Δω₁` | +0.8372 (**+0.4954 %**) ✅ | −0.0054 (−0.0032 %) | +0.0179 (+0.0108 %) |
| `Δω₂` | −1.5215 | +0.1296 | +0.3949 |
| Δ gap₁₂ | −0.014020 | +0.000818 | +0.002243 |
| Δ gray | −0.002500 | +0.000469 | −0.002200 |
| Δ mid | −0.000625 | +0.000469 | −0.000500 |
| mean \|Δρ_e\| | 0.005700 | 0.001367 | 0.002119 |
| max \|Δρ_e\| | 0.3263 | 0.0310 | 0.0530 |
| Δ\|volume−0.5\| | +1.52e-06 | −3.63e-07 | −1.53e-07 |
| Δ subspace size | 0 | 0 | 0 |
| cost: outer / inner | 39 / 931 | 39 / 686 | 39 / 700 |
| **material?** | **YES — `ω₁`** | **no** | **no** |

**Rung 2 is load-bearing at 160×20 and only at 160×20.** It clears the `ω₁` bar
there by a factor of ~5 (0.4954 % against 0.10 %), and it is *below* every bar at
both fine meshes. Its `M_nd` contribution at 160×20 (−1.902 %) sits just under
the 2 % bar and is not, on its own, material.

The scientific significance of the 160×20 `ω₁` gain is larger than its size
suggests: it is the difference between shipping a regression on the maximized
objective and not shipping one (§3 below).

## 2. RUNGS 3+4 (move = 0.01, 0.005):  S2 → F

| quantity | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ΔM_nd` | −0.0843 (−0.659 %) | −0.0564 (−0.435 %) | −0.1102 (−0.714 %) |
| `Δω₁` | +0.1935 (**+0.1139 %**) ⚠️ | +0.0026 (+0.0016 %) | +0.0207 (+0.0124 %) |
| `Δω₂` | −0.5114 | +0.1236 | +0.0838 |
| Δ gap₁₂ | −0.004160 | +0.000723 | +0.000353 |
| Δ gray | +0.000625 | −0.000625 | −0.000400 |
| Δ mid | +0.000625 | +0.000156 | −0.000100 |
| mean \|Δρ_e\| | 0.001432 | 0.002036 | 0.000598 |
| max \|Δρ_e\| | 0.1124 | 0.0497 | 0.0145 |
| Δ\|volume−0.5\| | −2.14e-06 | +2.87e-07 | +4.47e-07 |
| Δ subspace size | 0 | 0 | 0 |
| cost: outer / inner | 78 / 1 427 | **1 287 / 70 780** | 78 / 2 264 |
| cost multiplier vs the state they start from | 0.55× | **4.11×** | 0.18× |
| cost-dominated | no | **yes** | no |
| failure risk | no | **yes — CAP_HIT @1600** | no |
| **material?** | **YES — `ω₁`, by a narrow margin** | **no** | **no** |

### The one crossing, stated plainly

At 160×20 rungs 3+4 improve `ω₁` by **+0.1139 % relative**, against a
preregistered bar of **0.10 %**. That is a **material benefit** under the frozen
threshold. It is a narrow crossing — the margin over the bar is 14 % of the bar,
0.0139 percentage points, +0.0235 in absolute `ω₁` beyond what the bar demands —
but the bar was frozen before the number was computed, and the number is over it.

**It is not explained away.** It is the reason this audit does not return
`SUPPORTED`. Two facts are recorded alongside it, neither of which cancels it:

* the bar is expressed in *relative* terms and was anchored at one tenth of the
  controller study's 1 % `ω₁` allowance; at 160×20 it corresponds to ≈ 0.17
  absolute `ω₁`, and the observed gain is 0.1935;
* rungs 3+4 buy this at 160×20 while, on the same mesh, buying nothing on
  `M_nd`, topology, volume or multiplicity, and while producing a `CAP_HIT` at
  320×40.

Whether 0.11 % of `ω₁` is worth a rung that cannot terminate at another mesh is a
judgement for the *next* task, made against a bar chosen for that purpose. This
task reports that the frozen bar was crossed.

## 3. The 160×20 objective story end to end

| state | `ω₁` | vs production 169.4952 |
|---|---|---|
| P | 169.4952 | — |
| S1 | 168.9804 | **−0.5148 (−0.304 %) — a regression** |
| **S2** | **169.8175** | **+0.3223 (+0.190 %) — no regression** |
| F | 170.0110 | +0.5158 (+0.304 %) |

Rung 2 moves `ω₁` from **below** production to **above** it: `+0.8372`, of which
`+0.5148` is spent erasing the deficit and `+0.3223` is genuine surplus. Rungs
3+4 then add a further `+0.1935`.

So of the total `+1.0306` that the lower ladder buys at 160×20:

* **rung 2 delivers 81.2 %**, and delivers *all* of the part that matters for
  the no-regression question;
* **rungs 3+4 deliver 18.8 %**, entirely surplus above production.

The same decomposition for `M_nd` (total −0.3322 from S1 to F): rung 2 delivers
**74.6 %**, rungs 3+4 **25.4 %**; neither block is material on `M_nd` alone.

## 4. Cost decomposition

Share of each mesh's total inner MMA work, by rung:

| rung | move | 160×20 | 320×40 | 400×50 |
|---|---|---|---|---|
| 1 | 0.04 | 2 716 (53.5 %) | 5 066 (6.6 %) | 7 337 (71.2 %) |
| 2 | 0.02 | 931 (18.4 %) | 686 (0.9 %) | 700 (6.8 %) |
| 3 | 0.01 | 636 (12.5 %) | 746 (1.0 %) | 811 (7.9 %) |
| 4 | 0.005 | 791 (15.6 %) | **70 034 (91.5 %)** | 1 453 (14.1 %) |

Outer iterations by rung: 102/39/39/39 (160×20), 274/39/39/**1248** (320×40),
388/39/39/39 (400×50).

Every rung that terminates costs exactly 39 outer iterations, for the structural
reason given in `TWO_RUNG_ANALYSIS.md` §3. The single exception is 320×40's rung
4, which never terminates and consumes 91.5 % of that mesh's entire inner budget
to change `M_nd` by 0.44 % and `ω₁` by 0.0016 %.

## 5. Answer to the decomposition question

> Do rungs 3+4 provide *scientifically material* additional value?

* **320×40: no** — below every bar, cost-dominated (4.11×), and carrying the
  `CAP_HIT` failure.
* **400×50: no** — below every bar on every metric.
* **160×20: yes, on `ω₁` only, by a narrow margin over a frozen bar.**

One mesh of three. Under the previous audit's decision rule that same count made
the four-rung ladder "partially useful"; under this study's frozen mapping it
makes the two-rung architecture **partially supported**, not supported.
