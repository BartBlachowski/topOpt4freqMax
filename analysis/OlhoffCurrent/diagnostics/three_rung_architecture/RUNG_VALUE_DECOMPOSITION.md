# RUNG VALUE DECOMPOSITION — rung 3 isolated from rung 4 for the first time

Thresholds are those of `PREREGISTRATION.md` §9, inherited **verbatim** from
`move_ladder_necessity/PREREGISTRATION.md` §6 by way of
`two_rung_architecture/PREREGISTRATION.md` §9. Nothing is re-anchored here.

```
omega1       material if the block improves omega1 by >= 0.10 % relative
M_nd         material if the block improves M_nd  by >= 2 % relative
topology     material if |d gray| >= 0.01 or |d mid| >= 0.01 or mean|d rho_e| >= 0.01
volume       material if |volume - 0.5| worsens by >= 1e-5
multiplicity material if subspace size leaves 2, mode order changes, omega2 <= omega1,
             or a NaN/Inf appears.  GAP MAGNITUDE ALONE IS NOT MATERIAL.
cost         a block is cost-dominated if it costs >= 2x the outer iterations used to
             reach the state it starts from while being sub-material on every metric
failure risk any rung sequence producing CAP_HIT or a non-terminating stage

relative change = 100*(b - a)/a, normalised by the EARLIER state
```

---

## 1. The full four-rung decomposition

### 160×20

| rung | move | block | `Δω₁` | rel. | `ΔM_nd` | rel. | mean \|Δρ_e\| | outer | inner | **material?** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | start → S1 | +100.5818 | +147.05 % | −86.4229 | −86.89 % | 0.43698 | 102 | 2 716 | yes (everything) |
| 2 | 0.02 | S1 → S2 | +0.8372 | **+0.49541 %** | −0.2479 | −1.9018 % | 0.00570 | 39 | 931 | **yes — `ω₁`** |
| 3 | 0.01 | S2 → S3 | +0.1591 | **+0.09367 %** | −0.0323 | −0.2526 % | 0.00118 | 39 | 636 | **no** |
| 4 | 0.005 | S3 → F | +0.0344 | **+0.02025 %** | −0.0520 | −0.4078 % | 0.00042 | 39 | 791 | **no** |

### 320×40

| rung | move | block | `Δω₁` rel. | `ΔM_nd` rel. | mean \|Δρ_e\| | outer | inner | **material?** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | start → S1 | +143.75 % | −86.94 % | 0.43652 | 274 | 5 066 | yes (everything) |
| 2 | 0.02 | S1 → S2 | −0.00322 % | −0.2492 % | 0.00137 | 39 | 686 | no |
| 3 | 0.01 | S2 → S3 | +0.00662 % | −0.3052 % | 0.00034 | 39 | 746 | no |
| 4 | 0.005 | S3 → F | **−0.00504 %** | −0.1297 % | 0.00172 | **1 248** | **70 034** | **no — and cost-dominated, and CAP_HIT** |

### 400×50

| rung | move | block | `Δω₁` rel. | `ΔM_nd` rel. | mean \|Δρ_e\| | outer | inner | **material?** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | start → S1 | +143.87 % | −84.29 % | 0.42762 | 388 | 7 337 | yes (everything) |
| 2 | 0.02 | S1 → S2 | +0.01076 % | −1.4276 % | 0.00212 | 39 | 700 | no |
| 3 | 0.01 | S2 → S3 | +0.00429 % | −0.4408 % | 0.00040 | 39 | 811 | no |
| 4 | 0.005 | S3 → F | +0.00815 % | −0.2742 % | 0.00021 | 39 | 1 453 | no |

**Rung 4 is below every materiality bar on every mesh.** So is rung 3.
The only material lower rung anywhere is **rung 2 at 160×20**.

## 2. Rung 4 in detail — what the final `move = 0.005` stage buys

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `Δω₁` relative | +0.02025 % | **−0.00504 %** | +0.00815 % |
| bar | 0.10 % | 0.10 % | 0.10 % |
| `ΔM_nd` relative | −0.4078 % (bar 2 %) | −0.1297 % | −0.2742 % |
| Δ gray / Δ mid | −0.00063 / 0.00000 | −0.00078 / +0.00016 | 0.00000 / 0.00000 |
| mean \|Δρ_e\| (bar 0.01) | 0.000424 | 0.001723 | 0.000208 |
| Δ\|volume−0.5\| (bar 1e-5) | −2.36e-06 (improves) | −8.68e-07 (improves) | −3.19e-07 (improves) |
| Δ subspace size | 0 | 0 | 0 |
| cost multiplier vs S3 | 0.22× | **3.55×** | 0.08× |
| cost-dominated | no | **yes** | no |
| failure risk | no | **yes — CAP_HIT @1600** | no |

At 320×40 rung 4 makes `ω₁` **worse** by 0.0084 while consuming 1 248 outer and
70 034 inner MMA iterations and never terminating. There is no reading of the
evidence on which that rung earns its place at that mesh.

## 3. The 160×20 `ω₁` story, rung by rung

| state | `ω₁` | vs production 169.4952 |
|---|---|---|
| P | 169.4952 | — |
| S1 | 168.9804 | **−0.5148 (−0.304 %) — a regression** |
| S2 | 169.8175 | +0.3223 (+0.190 %) |
| **S3** | **169.9766** | **+0.4814 (+0.284 %)** |
| F | 170.0110 | +0.5158 (+0.304 %) |

Of the total `+1.0306` that the lower ladder buys from S1 to F:

| rung | `Δω₁` | share of S1→F |
|---|---|---|
| rung 2 (0.02) | +0.8372 | **81.2 %** |
| rung 3 (0.01) | +0.1591 | **15.4 %** |
| rung 4 (0.005) | +0.0344 | **3.3 %** |

Rung 2 alone erases the 0.5148 deficit and adds 0.3223 of surplus. Rung 3 adds
another 0.1591; rung 4 adds 0.0344.

The corresponding `M_nd` split (total −0.3322 from S1 to F): rung 2 **74.6 %**,
rung 3 **9.7 %**, rung 4 **15.7 %** — and none of the three is material on `M_nd`.

## 4. THRESHOLD SPLITTING — the finding that decides this audit

The two-rung policy failed because the **combined** residual it left on the
table at 160×20 was material:

```
S2 -> F   omega1  +0.11393 %   >=  0.10 % bar   ->  MATERIAL, two-rung blocked
```

Splitting that block at S3 gives:

```
S2 -> S3  (rung 3, retained)   +0.09367 %   <  0.10 %   ->  NOT material
S3 -> F   (rung 4, omitted)    +0.02025 %   <  0.10 %   ->  NOT material
```

The two halves sum to essentially the whole (0.09367 + 0.02025 = 0.11392 against
a combined 0.11393; the denominators differ negligibly at this scale).

**So the three-rung policy passes the preregistered residual gate — the thing
that blocked the two-rung policy — but it does so because the material block has
been divided into two individually immaterial pieces, not because rung 3 captures
a discrete effect that rung 4 lacks.**

This is exactly the situation `PREREGISTRATION.md` §11 was written to catch,
before any of these numbers were computed. It sets `THRESHOLD_SPLITTING = true`
and caps the architecture verdict at `PARTIALLY_SUPPORTED`.

### Why this matters, stated plainly and without embellishment

* Every quantity in this audit is real and every gate is passed as stated. The
  three-rung policy genuinely terminates on all three meshes, genuinely beats
  production on `ω₁` and `M_nd` everywhere, and genuinely leaves only +0.02 % of
  `ω₁` behind at 160×20.
* But **no retained rung below `move = 0.02` does material work by the project's
  own standard.** Rung 3 is kept solely because keeping it pushes the residual
  under the bar. The bar is a *per-block* relative threshold, so a block that
  exceeds it can in general be brought under it by subdividing the block — which
  is what has happened here. The gate is therefore not scale-free with respect to
  how finely the ladder is cut.
* The honest reading is that the 160×20 lower-ladder `ω₁` benefit beyond rung 2
  is a **continuum**, not a step. The materiality bar identifies no natural
  stopping point in it, so "where to stop" is not a question this evidence can
  answer by itself.

### What this audit does *not* conclude

It does not conclude that the three-rung ladder is wrong, that rung 3 should be
dropped, that the bar should be moved, or that a different ladder should be
tried. All of those would require a preregistration written before the answer is
known. This task tests exactly `[0.04, 0.02, 0.01]` and reports what it finds.

## 5. Cost decomposition

Share of each mesh's total inner MMA work, by rung:

| rung | move | 160×20 | 320×40 | 400×50 |
|---|---|---|---|---|
| 1 | 0.04 | 2 716 (53.5 %) | 5 066 (6.6 %) | 7 337 (71.2 %) |
| 2 | 0.02 | 931 (18.3 %) | 686 (0.9 %) | 700 (6.8 %) |
| 3 | 0.01 | 636 (12.5 %) | 746 (1.0 %) | 811 (7.9 %) |
| 4 | 0.005 | **791 (15.6 %)** | **70 034 (91.5 %)** | **1 453 (14.1 %)** |

Outer iterations by rung: 102/39/39/39 (160×20), 274/39/39/**1248** (320×40),
388/39/39/39 (400×50).

Every rung that terminates costs exactly **39** outer iterations — the structural
consequence documented in `DECLARATION_TIMING_AUDIT.md`. The single exception is
320×40's rung 4, which never terminates.

## 6. Marginal benefit per unit work

`Δω₁` per 1000 inner MMA iterations:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| rung 2 | +0.899 | −0.0078 | +0.0256 |
| rung 3 | +0.250 | +0.0148 | +0.0088 |
| rung 4 | +0.0435 | **−0.00012** | +0.0093 |

Rung 2 at 160×20 returns an order of magnitude more objective per unit of work
than anything below it, and rung 4 at 320×40 returns a negative amount.

## 7. Answer to the decomposition question

> Does rung 3 capture the residual scientific value needed at 160×20, while
> rung 4 contributes no preregistered material benefit?

**Rung 4 contributes no material benefit on any mesh — that half of the
hypothesis holds cleanly.** Rung 3 reduces the 160×20 residual below the bar, but
is **not itself material** by the same bar. So rung 3 does not "capture a value";
it truncates a continuum at a point that happens to fall on the right side of a
fixed threshold.
