# PROMOTION — decision and its basis

**Verdict: `PRODUCTION_CONTROLLER_NOT_PROMOTED`. Production is unchanged.**

---

## 1. The rule that decides this

Preregistration §12:

> **Promotion** only on `TWO_BRANCH_CONTROLLER_VALIDATED` with all gates passing,
> and then only as the exact tested controller […]
>
> If validation is PARTIAL, REJECTED or INCONCLUSIVE: **DO NOT PROMOTE.**
> Production remains unchanged.

and §12's definition of the primary verdict:

> **`TWO_BRANCH_CONTROLLER_VALIDATED`** — every gate P1–P15 passes.

So promotion turns on a single question: did every one of the fifteen preregistered
gates pass? It did not.

## 2. Why promotion is not authorized

**P13 fails at 320×40.** The preregistered bound is an outer-iteration multiplier
≤ 8× and a wall-time multiplier ≤ 10× *on every mesh*. Realized at 320×40:

| | production | candidate | multiplier | bound | result |
|---|---|---|---|---|---|
| outer iterations | 131 | 1600 | **×12.21** | ≤ 8 | **FAIL** |
| inner MMA iterations | 2 614 | 76 532 | ×29.28 | — | — |
| wall time | 388.0 s | 34 373.8 s | **×88.60** | ≤ 10 | **FAIL** |

This is not a marginal miss and it is not an artifact of the cap being too small: the
run reached `move = 0.005` at iteration 353 and then could never terminate, because
the frozen union is blind to the low-amplitude cancelling regime it entered
(`REPORT.md` §8.2). The cap did what a cap is for — it stopped an otherwise
non-terminating run — and `CAP_HIT` is reported as the honest outcome it is.

Because `VALIDATED` requires *all* of P1–P15, one failing gate is dispositive. The
primary verdict is therefore at best `TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED`, and
§12 forbids promotion on anything short of `VALIDATED`.

## 3. What was deliberately *not* done

The temptation here is specific and worth naming, because resisting it is the point of
a preregistration. 320×40's failure has an obvious cause and an obvious fix: its
terminal stage is a stationary limit cycle, converged to 0.4 % in M_nd and 0.017 % in
ω₁ over 1 248 iterations, with the move limit inactive. Almost any additional clause —
relaxing B's coherence guard, adding a stationarity test, adding a Branch C — would
have admitted it, turned `CAP_HIT` into `CONVERGED`, brought the outer multiplier to
roughly ×2.98, and produced a clean sweep of all fifteen gates.

None of that was done. Preregistration §13 and Phase 19 forbid changing `A`, `B`, `W`,
`P`, the tolerance, the persistence rule, the reset semantics, the move ladder or the
stopping semantics after the first candidate result, and forbid adding a Branch C or
any mesh-specific parameter. The controller that ran at 400×50 is byte-identical to
the one that ran at 160×20, and the `+impl` tree hash
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` recorded inside
every run record matches the tree present at the end of the study. An undesirable
scientific outcome was not reclassified as a software bug.

## 4. Consequences

Because no promotion occurred:

* **Phase 22** is not entered. No controller logic was moved into canonical
  production; no threshold was altered; β retains its production authority over move
  transition and terminal admission in the production configuration.
* **Phase 23** does not apply. `CONTROLLER_PROMOTION_EQUIVALENCE_PASS` /
  `_FAIL` is **not issued**, because there is nothing to prove equivalent — the
  question only arises if promotion was performed.
* Production remains exactly the solver that existed at task start, and this is
  proved rather than asserted: with both switches at their defaults,
  `test_preset_equivalence` reproduces the frozen conference record **bitwise** at
  160×20 — ρ bitwise, ω₁ = 169.49522702153845 bitwise, volume bitwise, 91 outer,
  2 241 inner, `CONVERGED`.

## 5. What the study nevertheless establishes

Not promoting is not the same as learning nothing. The intervention confirmed its
central causal claim at both fine meshes: holding `move = 0.04` past the β stall is
what produces the grayness improvement, and the improvement is large (M_nd −44.68 %
at 320×40) while ω₁ *improves* rather than degrades. The mechanism that blocks
promotion is narrow, identified, measured, and orthogonal to that claim — it concerns
how a stage is declared *finished at the bottom of the ladder*, not whether delaying
continuation helps. That is a well-posed question for a subsequent task, which this
one is explicitly not permitted to answer.
