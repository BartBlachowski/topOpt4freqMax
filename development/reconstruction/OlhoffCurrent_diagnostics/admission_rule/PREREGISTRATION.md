# PREREGISTRATION — convergence admission rule

Frozen **before any candidate evaluation**. Rule, thresholds, meshes, caps and
gates are fixed here and are not revised after results.

| | |
|---|---|
| Repository HEAD at freeze | `2b89b070477c67aa7bbe5febc369f61fcad92da6` |
| Implementation | `analysis/OlhoffCurrent` |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 source files) |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` |
| Effective cfg hash 160×20 | `ca9a5c90d5c412759b43ca5c44f8d6b3d3e13ca70bfde6d8f6e9cead0da7cde6` (ε = 0.05) |
| Effective cfg hash 320×40 | `abae6b78ce199c76fdc137a989bb6d1408ba88192ad30aa7e9484a12c2d0dd07` (ε = 0.10) |
| MATLAB | R2025b Update 1 (25.2.0.3042426), **1 computational thread** |

---

## 1. The invariant to enforce

> **Changing the move limit cannot, by itself, cause convergence.**

## 2. What is wrong with the current rule

`convOuter = ‖Δρ‖₂ < ε`, guarded by `settledMove`. Because `max|Δρ| ≤ move`,
scaling the move down scales `‖Δρ‖₂` down with it. The prior diagnostic measured
a 2.75–2.90× collapse of `‖Δρ‖₂` at a single descent against a ±0.2 % change in
M_nd, with termination exactly one iteration later at both meshes. `settledMove`
delays admission by one iteration; it does not remove the mechanism.

## 3. Candidate rule — `settledLocalObjective`

Admission requires **all three** components simultaneously:

```
admit(k) =  A(k)  AND  B(k)  AND  C(k)

A  move-settled dwell
     move(k) has been unchanged for the last D iterations
       #{ j in [k-D+1, k] : move(j) == move(k) } == D

B  local design change, absolute AND dimensionless
     max_e |Δρ_e(k)|            <  τ_abs
     max_e |Δρ_e(k)| / move(k)  <  τ_rel

C  objective stability over a window
     ( max ω₁(j) - min ω₁(j) ) / mean ω₁(j)  <  τ_obj ,  j in [k-W+1, k]
```

`B` is the part that enforces the invariant: `max|Δρ|/move` is **invariant under
a change of move limit**, so a descent cannot reduce it. `A` additionally
forbids testing at all until the level has been held. `C` requires the physics,
not only the design, to have stopped moving.

## 4. Thresholds — fixed here, derived, not tuned

| Symbol | Value | Origin (see `PROVENANCE_LEDGER.md` §3) |
|---|---|---|
| `τ_abs` | **0.01** | Bendsøe & Sigmund 99-line code; `top88_reference.m` in this tree |
| `τ_rel` | **0.5** | the ladder's own halving factor — below it, the realized step already fits inside the next ladder level |
| `D` | **10** | `move.continuation.window` |
| `W` | **10** | same |
| `τ_obj` | **5e-3** | `move.continuation.tolerance` |

**Preregistered candidate set — exactly two, no sweep:**

* **C1 (primary)** — `τ_abs = 0.01, τ_rel = 0.50, D = 10, W = 10, τ_obj = 5e-3`
* **C2 (sensitivity)** — identical except **`τ_rel = 0.25`**, testing the one
  threshold derived from the ladder rather than from literature.

No other value will be evaluated. No threshold will change after results.

## 5. Meshes and cap

**160×20 and 320×40 only.** 400×50 only if both pass and the result is not
ambiguous (§17 of the brief). Never 800×100, never the nine-mesh campaign.

**Safety cap: `runtime.maxOuter = 600`**, both meshes. Reaching it is
**`CAP_HIT`** and is reported as such; it is never relabelled convergence. A
candidate that cannot terminate at 160×20 fails the production-admission gate —
**no fallback is defined, deliberately**, so that failure is visible rather than
absorbed.

## 6. Evaluation method, and why it is exact

The candidate changes only *when the loop breaks*. In the production
configuration `convOuter` is read at exactly four places in `olhoffSolve`: the
`settledMove` guard, the stop guards, the projection-continuation trigger
(`projection.enabled = false` → dead), the p-continuation stop-block
(`stiffness.continuation.enabled = false` → dead), and the `break`. It is
computed **after** `olh.move.limit` and is never written back into `mvState`.

**Therefore the trajectory is independent of the admission rule**, and one
*unstopped* run per mesh contains the exact trajectory that every candidate
would follow. Candidates are evaluated offline on it. This is exact, not an
approximation, and it avoids modifying the promoted production source.

Unstopping is done with the two stopping-policy fields only —
`stop.tolerance = 0`, `stop.toleranceRule = 'explicit'` — which §0 of the brief
places under explicit experimental control. Nothing else is touched.

**This independence is verified, not assumed:** the unstopped run must reproduce
the archived baseline **bitwise** over the baseline's own iterations
(design, ω₁, per-iteration history). If it does not, the method is invalid and
the study stops.

## 7. Baseline arm

The unchanged production preset at both meshes, compared against the archived
nine-mesh conference record for final design, ω₁, ω₂, gap12, M_nd, gray and
mid-density fractions, volume, outer and inner counts, move transitions and
stopping fields. **Bitwise preferred.** Failure ⇒ `BASELINE_REPRODUCTION_FAIL`,
and the study stops.

## 8. Hard gates (fixed; none added after results)

**160×20:** G1 natural termination before cap · G2 ω₁ regression < 3 % vs
baseline · G3 volume feasible · G4 no solver failure · G5 no admission within
`D` iterations of a descent · G6 final local change genuinely small under the
new criterion (both parts of B) · G7 topology qualitatively consistent with the
reconstruction anchor.

**320×40:** G8 natural termination · G9 ω₁ regression < 3 % · G10 volume
feasible · G11 no solver failure · G12 no admission within `D` of a descent ·
G13 M_nd materially below baseline **or** demonstrated mature stopping ·
G14 mid-density fraction not materially worse · G15 objective evolution small at
termination under the preregistered rule.

"Materially below" for G13 is fixed here as **≥ 25 % relative reduction in
M_nd**. "Not materially worse" for G14 is fixed as **≤ 10 % relative increase**
in mid-density fraction. Volume feasible = `|mean(ρ) − 0.5| ≤ 1e-3`.

## 9. Verdicts

| Verdict | Condition |
|---|---|
| `ADMISSION_RULE_CANDIDATE_PASS` | all applicable gates hold at both meshes for at least one preregistered candidate |
| `ADMISSION_RULE_CANDIDATE_PARTIAL` | invariant demonstrably enforced, but a non-safety gate fails at one mesh |
| `ADMISSION_RULE_CANDIDATE_FAIL` | the rule still admits right after a descent, or fails a safety gate (G3/G4/G10/G11), or regresses ω₁ ≥ 3 % |
| `ADMISSION_RULE_INCONCLUSIVE` | evidence insufficient or internally contradictory |

**Tie-breaking if both C1 and C2 pass:** prefer **C1**, because its `τ_rel`
sits at the ladder's own halving factor and is the weaker (less restrictive)
assumption; C2 is reported as a sensitivity check, not a competitor.

**Regression** = ω₁ lower than baseline by ≥ 3 %, or volume infeasible, or
solver failure, or mid-density fraction ≥ 10 % worse.
**Improvement** = M_nd reduced ≥ 25 % at 320×40 with ω₁ not regressed.

Production verdict is fixed in advance regardless of outcome:
**`KEEP_PRODUCTION_PRESET_PENDING_PROMOTION_REVIEW`.** No production file will
be modified, and no conference campaign will be rerun.

## 10. Prohibited

Changing filter type or radius, p, q, mass interpolation, multiplicity,
off-diagonal treatment, MMA, move levels, move-stall signal/window/tolerance,
projection, density filtering, eigen solver, objective, FE model; 800×100; the
nine-mesh campaign; modifying the production preset; promoting the candidate.
