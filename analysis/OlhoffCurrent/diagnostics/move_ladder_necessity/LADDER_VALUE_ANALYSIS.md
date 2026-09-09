# LADDER VALUE ANALYSIS — what the lower rungs actually buy

Phases 9–13, 15. All quantities from `evidence/ladder_analysis.json`.

---

## 1. The three-way table (Phase 14)

`P` production · `S` single-stage endpoint · `F` four-rung candidate.
Field values are in-loop telemetry for S and F; production scalars are the frozen
`METRICS.json` values from a final re-analysis. That convention difference is
≤ 0.015 in ω₁ and exactly 0 in `M_nd`, and changes no conclusion (§7).

### 160×20

| | P | S | F |
|---|---|---|---|
| status | `NATIVE_CONVERGED` | terminate on A @102 | `CONVERGED` |
| outer | 91 | **102** | 219 |
| inner MMA | 2 241 | **2 716** | 5 074 |
| wall [s] | 125.5 | **135.0** | 425.2 |
| ω₁ | 169.495227 | **168.980391** | 170.011021 |
| `M_nd` [%] | 13.402499 | **13.036370** | 12.704121 |
| gray | 0.149375 | 0.146250 | 0.144375 |
| mid | 0.025000 | 0.026875 | 0.026875 |
| volume | 0.499999 | 0.499999 | 0.499999 |
| relative gap | 0.014766 | 0.026528 | 0.008347 |

### 320×40

| | P | S | F |
|---|---|---|---|
| status | `NATIVE_CONVERGED` | terminate on A @274 | **`CAP_HIT`** |
| outer | 131 | **274** | 1 600 |
| inner MMA | 2 614 | **5 066** | 76 532 |
| wall [s] | 388.0 | **788.8** | 34 373.6 |
| ω₁ | 165.950789 | **166.421616** | 166.418886 |
| `M_nd` [%] | 23.359568 | **13.012132** | 12.923311 |
| gray | 0.263750 | 0.152344 | 0.152188 |
| mid | 0.095625 | 0.029687 | 0.030312 |
| volume | 0.499999 | 0.500000 | 0.500000 |
| relative gap | 0.106742 | 0.222336 | 0.223878 |

### 400×50

| | P | S | F |
|---|---|---|---|
| status | `CONVERGED` | terminate on B @388 | `CONVERGED` |
| outer | 139 | **388** | 505 |
| inner MMA | 2 918 | **7 337** | 10 301 |
| wall [s] | 541.0 | **1 722.4** | 3 539.0 |
| ω₁ | 162.882616 | **166.417621** | 166.456242 |
| `M_nd` [%] | 32.328326 | **15.664940** | 15.331078 |
| gray | 0.347600 | 0.182200 | 0.179600 |
| mid | 0.188200 | 0.039800 | 0.039200 |
| volume | 0.499999 | 0.500000 | 0.499999 |
| relative gap | 0.076873 | 0.208193 | 0.210789 |

## 2. What the lower rungs bought, against the preregistered bars

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| Δ`M_nd` (F − S) | −0.3322 | −0.0888 | −0.3339 |
| **relative** | **−2.549 %** | −0.683 % | **−2.131 %** |
| material? (bar −2.0 %) | **YES** | no | **YES** (marginal) |
| Δω₁ | **+1.030631** | −0.002730 | +0.038621 |
| **relative** | **+0.6099 %** | −0.0016 % | +0.0232 % |
| material? (bar +0.10 %) | **YES** | no | no |
| Δgray | −0.00187 | −0.00016 | −0.00260 |
| Δmid | +0.00000 | +0.00063 | −0.00060 |
| mean \|Δρ_e\| S→F | 0.00651 | 0.00340 | 0.00271 |
| material? (bar 0.01) | no | no | no |
| volume feasibility | −6.3e-07 | −7.6e-08 | +2.9e-07 |
| material? (bar 1e-5) | no | no | no |
| multiplicity | N=2→2 | N=2→2 | N=2→2 |
| material? | no | no | no |
| **ANY material benefit** | **YES** | **NO** | **YES** |
| cost: extra outer | +117 (×1.15 of S) | **+1 326 (×4.84)** | +117 (×0.30) |
| cost: extra inner MMA | +2 358 | **+71 466** | +2 964 |
| cost: extra wall [s] | +290 | **+33 585** | +1 817 |
| share of run below 0.04 | 68.3 % wall | **97.7 % wall** | 51.3 % wall |
| cost-dominated? | no | **YES** | no |
| failure risk | no | **`CAP_HIT`** | no |

**Two of three primary meshes show material lower-rung benefit.**

## 3. Rung by rung (Phase 10)

Each row is the state change from entering that rung to leaving it.

### 160×20

| rung | move | iters | inner | wall s | Δ`M_nd` | Δ`M_nd` % | Δω₁ % | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 102 | 2 716 | 135 | −86.4229 | −86.893 | +147.052 | descended |
| 2 | 0.02 | 39 | 931 | 82 | −0.2479 | −1.902 | **+0.4954** | descended |
| 3 | 0.01 | 39 | 636 | 85 | −0.0323 | −0.253 | +0.0937 | descended |
| 4 | 0.005 | 39 | 791 | 124 | −0.0520 | −0.408 | +0.0202 | `CONVERGED` |

### 320×40

| rung | move | iters | inner | wall s | Δ`M_nd` | Δ`M_nd` % | Δω₁ % | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 274 | 5 066 | 789 | −86.6019 | −86.937 | +143.753 | descended |
| 2 | 0.02 | 39 | 686 | 223 | −0.0324 | −0.249 | −0.0032 | descended |
| 3 | 0.01 | 39 | 746 | 387 | −0.0396 | −0.305 | +0.0066 | descended |
| 4 | 0.005 | **1 248** | **70 034** | **32 975** | −0.0168 | −0.130 | −0.0050 | **`CAP_HIT`** |

### 400×50

| rung | move | iters | inner | wall s | Δ`M_nd` | Δ`M_nd` % | Δω₁ % | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 388 | 7 337 | 1 722 | −84.0211 | −84.286 | +143.866 | descended |
| 2 | 0.02 | 39 | 700 | 343 | −0.2236 | −1.428 | +0.0108 | descended |
| 3 | 0.01 | 39 | 811 | 547 | −0.0681 | −0.441 | +0.0043 | descended |
| 4 | 0.005 | 39 | 1 453 | 926 | −0.0422 | −0.274 | +0.0082 | `CONVERGED` |

Every completed lower rung takes exactly **39 iterations** — the frozen rule's
minimum (`stageStart + 38`, descent one later). Each new rung enters its mature
regime immediately. The single exception is 320×40's terminal rung, which never
exhausts at all.

**Rung 2 is where the lower-ladder value is.** It delivers −1.90 % `M_nd` and
+0.495 % ω₁ at 160×20, and −1.43 % `M_nd` at 400×50. Rungs 3 and 4 deliver
≤ 0.44 % `M_nd` and ≤ 0.094 % ω₁ anywhere — below every materiality bar on every
mesh.

## 4. Fraction banked at S (Phase 13)

| mesh | % of P→F `M_nd` gain banked at S | % of P→F ω₁ gain banked at S | % outer after | % inner after | % wall after |
|---|---|---|---|---|---|
| 160×20 | **52.43** | **−99.81** | 53.4 | 46.5 | 68.3 |
| 320×40 | **99.15** | **100.58** | 82.9 | 93.4 | 97.7 |
| 400×50 | **98.04** | **98.92** | 23.2 | 28.8 | 51.3 |

**The ≥ 98 % fine-mesh claim is reproduced exactly**: 99.15 % and 98.04 % for
`M_nd`. It was correctly scoped to the fine meshes — at 160×20 only 52.4 % of the
`M_nd` gain is banked at S, and the ω₁ figure is **−99.8 %**, meaning that at the
single-stage endpoint the coarse mesh has *negative* ω₁ gain: it is **worse than
production**, and only the lower rungs recover it.

## 5. The 160×20 finding, and whether it survives scrutiny (Phase 12)

At S, 160×20 has ω₁ = 168.9804 against production's 169.4952 — **0.30 % below**.
S's `M_nd` is only −2.73 % better than production, against −5.21 % for F.

The obvious objection is that 160×20's mature regime is a high-amplitude
cancelling cycle, so S might simply have landed on an unlucky phase. It did not:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| ω₁ peak-to-peak in the declaration window | **0.7708** | 0.0352 | 0.0582 |
| ω₁ at S − production | **−0.5148** | +0.4708 | +3.5350 |
| window **median** − production | **−0.2726** | +0.4752 | +3.5183 |
| window **maximum** − production | **−0.0207** | +0.4945 | +3.5350 |
| fraction of the whole `move=0.04` prefix with ω₁ > production | **0.000** | 0.526 | 0.644 |

The oscillation is real and large at 160×20 — but **the best iteration in the
window is still below production, and across all 102 iterations of the entire
`move = 0.04` stage the candidate's ω₁ never once reaches production's.** So the
deficit is a property of the stage, not of where the window happened to close.
The lower rungs are what carry 160×20 from below production to above it
(+0.53 in-loop, +0.52 by final analysis).

This is a **blocking** result for ladder removal at the coarse mesh, and it is
reported on its own rather than averaged with the fine meshes, exactly as the
brief requires.

## 6. Multiplicity, physics and feasibility (Phase 15)

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| subspace size at S / at F | 2 / 2 | 2 / 2 | 2 / 2 |
| min/max N over S→F | 2 / 2 | 2 / 2 | 2 / 2 |
| ω₂ > ω₁ throughout S→F | yes | yes | yes |
| any non-finite ω | none | none | none |
| relative gap S → F | 0.026528 → **0.008347** | 0.222336 → 0.223878 | 0.208193 → 0.210789 |
| min gap over S→F | 0.008347 | 0.222336 | 0.208193 |
| volume S → F | 0.4999985 → 0.4999991 | 0.5000000 → 0.5000000 | 0.4999997 → 0.4999994 |
| max \|vol−0.5\| over S→F | 2.3e-05 | 2.7e-06 | 1.7e-06 |
| inner non-convergences over S→F | 0 | 0 | 0 |

No multiplicity benefit and no multiplicity hazard: the fixed two-mode subspace
holds at size 2 on every iteration, on every mesh, in both states. No volume
benefit either — both states satisfy the constraint to ≈ 1e-6.

One asymmetry worth recording: at **160×20 the lower rungs shrink the relative
gap by a factor of 3.2** (0.0265 → 0.0083), driving the pair closer to
coalescence, while at the fine meshes the gap is essentially unchanged. This is
not a failure — `N` stays 2 and `ω₂ > ω₁` throughout — but it means the coarse
mesh's lower-rung ω₁ gain is obtained by moving *towards* the multiplicity
regime, which is precisely where this formulation's off-diagonal treatment
matters most. It is a reason to be careful about 160×20, not a reason to
discount its benefit.

## 7. Convention check

Production scalars come from a post-loop re-analysis; S and F are in-loop. The
difference:

| mesh | ω₁ in-loop | ω₁ final | Δ | `M_nd` Δ |
|---|---|---|---|---|
| 160×20 | 169.480593 | 169.495227 | −0.0146 | 0.000000 |
| 320×40 | 165.946839 | 165.950789 | −0.0040 | 0.000000 |
| 400×50 | 162.882616 | 162.882616 | 0.0000 | 0.000000 |

Under the in-loop convention S−P at 160×20 is −0.500 instead of −0.515, and F−P
is +0.530 instead of +0.516. **Every sign, every materiality decision and the
verdict are unchanged.**

## 8. The 240×30 supporting datapoint

240×30 has no four-rung run and never will, so it cannot speak to the ladder.
What it does show — from the surviving scalars — is what continuing **at
`move = 0.04`** past the exhaustion event buys:

| | at event 187 (branch B) | at cap 1200 | change |
|---|---|---|---|
| `M_nd` [%] | 12.8388 | 13.4507 | **+4.77 % — worse** |
| ω₁ | 167.0571 | 167.0248 | −0.019 % |
| gray | 0.14917 | 0.15694 | +0.0078 |
| mid | 0.02889 | 0.03778 | +0.0089 |

1 013 further iterations at `move = 0.04` made the design measurably **worse**.
That supports terminating stage 1 at the exhaustion event — which both policies
do — and says nothing about whether to descend afterwards.
