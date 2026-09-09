# PROMOTION — Phases 20–23

## 1. The preregistered promotion gates

Bounds fixed in `PREREGISTRATION.md` §11 **before the first candidate run**.
Evaluated in `evidence/analysis.json`.

| gate | requirement | result |
|---|---|---|
| **P1** | software gate (17/17) and single-factor gate pass | **PASS** |
| **P2** | no mesh terminates falsely | **PASS** |
| **P3** | no unacknowledged failure | **PASS** |
| **P4** | 160×20 first descent ≤ 400 | **PASS** — 103 |
| **P5** | 160×20 `M_nd` ≤ 14.743 and ω₁ ≥ 167.800 | **PASS** — 12.704, 170.011 |
| **P6** | 320×40 first descent ≥ 180 and `M_nd` ≤ 18.688 | **PASS** — 275, 12.923 |
| **P7** | 400×50 first descent ≥ 188 and `M_nd` ≤ 25.863 | **PASS** — 389, 15.331 |
| **P8** | ω₁ ≥ 0.99 × production on every mesh | **PASS** — +0.30 %, +0.29 %, +2.19 % |
| **P9** | \|volume − 0.5\| ≤ 1e-4 on every mesh | **PASS** — 8.7e-7, 3.8e-7, 5.7e-7 |
| **P10** | multiplicity/physics acceptable | **PASS** |
| **P11** | every transition attributable to frozen A or B | **PASS** — 9/9 |
| **P12** | terminal convergence only at `move_min` after frozen exhaustion | **PASS** |
| **P13** | outer multiplier ≤ 8× **and** wall multiplier ≤ 10× on every mesh | **FAIL** — 320×40: ×12.21 outer, ×88.60 wall |
| **P14** | no mesh-specific tuning | **PASS** |
| **P15** | evidence complete and hash-valid | **PASS** |

Fourteen of fifteen gates pass. **P13 fails at 320×40.**

## 2. Verdict

> ## `TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED`

By the frozen mapping (`PREREGISTRATION.md` §12): P1–P3 and P9–P12 pass, meshes
meet their improvement gates, and P13 fails — which is precisely the
`PARTIALLY_VALIDATED` branch.

It is **not** `REJECTED`: nothing on the REJECTED list occurred. No false
termination, no unacknowledged failure, no physics blocker, no transition
unattributable to the frozen rule, no regression beyond P5 or P8. The 320×40
`CAP_HIT` is an acknowledged, reported cap.

It is **not** `INCONCLUSIVE`: the cap at 320×40 bound *after* all four rungs had
been traversed and `M_nd` had already improved 44.7 %, then held flat for 1248
iterations. The controller's effect is not obscured — it is fully legible, and
so is its failure.

It is **not** `VALIDATED`, and would not be even though 320×40 and 400×50 look
spectacular. Two things stop it, and both are real:

1. **P13, the preregistered cost bound, fails at 320×40** — ×12.2 outer and ×88.6
   wall against bounds of 8× and 10×.
2. **320×40 never reached genuine terminal exhaustion.** A controller whose
   terminal-admission rule cannot fire on one of three tested meshes is not ready
   to be the production stopping criterion, whatever its topology gains.

## 3. Promotion decision

> ## `PRODUCTION_CONTROLLER_NOT_PROMOTED`

Promotion is authorized **only** on `TWO_BRANCH_CONTROLLER_VALIDATED` with every
gate passing. That condition is not met, so no promotion is performed and **no
promotion-equivalence verdict is issued** — Phase 23 applies only if promotion
occurs.

## 4. What "production remains unchanged" means concretely here

The production **controller and its behaviour are unchanged**, and this is
proven, not asserted:

* the production preset still resolves to `move.continuation.signal =
  'boundVariable'` and `stop.rule = 'designChange'` — the two candidate switches
  default to production;
* `test_preset_equivalence` re-solved 160×20 through the production entry point
  after all edits and reproduced the frozen conference record **bitwise** — ρ
  bitwise, ω₁ bitwise, 91 outer, 2241 inner, `CONVERGED`;
* β retains its authority over continuation and termination in production.

The production **source tree is not byte-identical to task start**: it carries
the candidate controller layer, default-inactive.

```
+impl/ at task start   c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c   74 files
+impl/ now             edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb   75 files
```

The layer is kept, rather than reverted, for one reason: it is the instrument
that produced this study's evidence, and removing it would make the three
scientific runs unreproducible while leaving their trajectories on disk. It is
proven inert for production by the bitwise test above.

If a byte-identical production tree is preferred, the change is fully reversible
in one step, because all 75 files are tracked:

```
git checkout b6014ba -- analysis/OlhoffCurrent/+impl analysis/OlhoffCurrent/SOURCE_MANIFEST.json
git rm --cached analysis/OlhoffCurrent/+impl/architecture/+olh/+move/exhaustion.m   # if it was added
```

That is a decision for the maintainer, not one this task should make unilaterally.

## 5. What a future task would have to fix — recorded, not attempted

Stated so the finding is usable, and explicitly **not** implemented here (Phase
19 freezes the controller; every item below would be an outcome-driven change
made after seeing the 320×40 trajectory):

* The union `A OR B` has a hole at **low-amplitude cancellation**: `‖Δρ‖₂ < tol`
  together with `med₂₀cosθ < 0` satisfies neither branch. That is the 320×40
  terminal regime, for 1248 iterations.
* The hole is reached *because the ladder descends*. Amplitude falls with the
  move limit while `tol(NE)` does not, so the lower rungs push every mesh toward
  the region where Branch A's amplitude clause can no longer fire.
* §6 of `CAUSAL_ANALYSIS.md` shows the lower rungs buy almost nothing: ≥ 98 % of
  the fine-mesh `M_nd` gain is banked before the first descent. The cheapest
  honest fix may therefore not be a new branch at all, but a question about
  whether the ladder below 0.04 earns its cost.

None of this is a proposal. It is the evidence a proposal would have to start from.
