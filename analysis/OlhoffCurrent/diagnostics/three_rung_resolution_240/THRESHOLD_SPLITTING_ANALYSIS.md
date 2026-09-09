# THRESHOLD-SPLITTING ANALYSIS — the concern, and whether 240×30 settles it

---

## 1. What the concern was

`three_rung_architecture` returned `PARTIALLY_SUPPORTED` rather than `SUPPORTED`
because of a guard preregistered *before* its numbers were computed. At 160×20:

| block | `Δω₁` relative | vs the 0.10 % bar |
|---|---|---|
| rungs 3+4 combined (`S2 → F`) | **+0.11393 %** | **above** |
| rung 3 alone (`S2 → S3`) | +0.09367 % | below |
| rung 4 alone (`S3 → F`) | +0.02025 % | below |

One above-threshold effect, subdivided into two individually sub-threshold
halves. The bar is a **per-block relative** threshold, so any block that exceeds
it can in principle be brought under it by cutting the block more finely. The
prior audit refused to conclude "rung 3 captures a material residual and rung 4
does not" from a subdivision of a single mesh's total — and demanded independent
evidence about **rung 4 specifically**.

## 2. Why 240×30 is the right experiment

Because no 240×30 lower-rung trajectory had ever existed. The only prior 240×30
arm was **fixed-move** (`move = 0.04` held, `CAP_HIT @1200`) and never descended,
so `S2`, `S3`, `F` and every rung-3/rung-4 quantity at this mesh were unmeasured.

Critically, this run measures rung 4 **in isolation** on a mesh where the
combined lower residual is nowhere near the bar, so no subdivision question
arises here at all:

| block at 240×30 | `Δω₁` relative | vs bar |
|---|---|---|
| rungs 3+4 combined (`S2 → F`) | **+0.00778 %** | 12.9× below |
| rung 3 alone | +0.00105 % | 95× below |
| rung 4 alone | **+0.00672 %** | **14.9× below** |

There is nothing to split. Rung 4's contribution is measured directly against
the bar, with no dependence on where the block boundary is drawn.

## 3. The preregistered resolution rule, applied

`PREREGISTRATION.md` §7, frozen before the run:

> **RESOLVED in favour of three-rung support** iff the new 240×30 evidence shows
> `S3 → F` is below **ALL** frozen materiality bars; **or** rung 4 becomes
> pathological / non-terminating while `S3` is already a valid exhausted state
> and no material pre-pathology benefit is obtained.
> **RESOLVED AGAINST** iff rung 4 produces a benefit above **any** frozen bar.
> **UNRESOLVED** iff no defensible `S3`/`F` comparison is possible.

Evaluation:

| requirement | measured | verdict |
|---|---|---|
| `S3 → F` `ω₁` below 0.10 % | +0.00672 % | ✅ |
| `S3 → F` `M_nd` below 2 % | +0.20100 % (and in the *worse* direction) | ✅ |
| `S3 → F` topology below 0.01 | mean \|Δρ\| 0.00212; gray/mid ≤ 0.00056 | ✅ |
| `S3 → F` volume below 1e-5 | +3.51e-07 | ✅ |
| `S3 → F` multiplicity immaterial | subspace 2→2, no mode change | ✅ |
| **running best** over the whole tail also below every bar | `ω₁` +0.01266 %, `M_nd` −0.23330 % | ✅ |
| a defensible `S3`/`F` comparison exists | counterfactual exact, `S3` a valid persistent-`E` state | ✅ |

**`THRESHOLD_SPLITTING_CONCERN_RESOLVED`.**

## 4. Rung 4 now measured directly on four independent meshes

| mesh | `NE` | rung-4 `Δω₁` rel. | rung-4 `ΔM_nd` rel. | mean \|Δρ\| | material? | four-rung terminal status |
|---|---|---|---|---|---|---|
| 160×20 | 3 200 | +0.02025 % | −0.4078 % | 0.000424 | no | CONVERGED @219 |
| **240×30** | **7 200** | **+0.00672 %** | **+0.2010 %** | **0.002120** | **no** | **CONVERGED @1358** |
| 320×40 | 12 800 | **−0.00504 %** | −0.1297 % | 0.001723 | no | **CAP_HIT @1600** |
| 400×50 | 20 000 | +0.00815 % | −0.2742 % | 0.000208 | no | CONVERGED @505 |

Bar: 0.10 %. Range of measured values: **−0.005 % to +0.020 %** — the largest is
5× below the bar, and one is negative. **Four meshes, four independent
measurements, zero material.**

This is a direct test of rung 4, not an inference from subdivision.

## 5. What this does — and does not — settle

**Settled.** The question the guard posed: *is rung 4 dismissible, or was it only
made to look dismissible by cutting a material block in two?* Answer: rung 4 is
dismissible on its own merits. It was measured alone at a mesh where the block
boundary is irrelevant, and it delivered nothing.

**Not settled by this run, and not claimed.** Whether **rung 3** does material
work at 160×20 is a different question, and this run does not address it. At
160×20 rung 3 remains worth +0.09367 % — below the bar, yet the largest lower-rung
`ω₁` contribution recorded anywhere except rung 2's +0.49541 % at that same mesh.
The three-rung architecture retains rung 3; whether *it* earns its place is a
question about the `[0.04, 0.02]` versus `[0.04, 0.02, 0.01]` boundary, which
`two_rung_architecture` already examined and which this task was not asked to
revisit.

What has changed is that the **specific objection** raised against the three-rung
verdict — that rung 4's dismissal rested on subdivision — no longer stands.

## 6. A caveat recorded honestly

The preregistration disclosed, before the run, that the direction of this result
was foreseeable: the fixed-move 240×30 arm already showed this mesh exiting
`move = 0.04` on Branch B with `‖Δρ‖₂/tol = 0.68` and zero bound-active elements,
grouping it with 400×50, where rung 4 was already known to be immaterial.

That does not weaken the evidence — the measurement is real, independent, and was
the only way to obtain a 240×30 rung-4 number — but it does mean this run
**confirmed an expectation** rather than probing a genuinely open direction. The
preregistration says so in §10, and the outcome matched the prediction it made.
