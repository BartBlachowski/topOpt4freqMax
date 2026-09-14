# IMPLEMENTATION — the candidate two-branch stage-exhaustion controller

What was added to the canonical implementation, why it is the *only* intervention,
and how production behaviour stays bitwise reproducible while it is unproven.

---

## 1. The intervention, in one sentence

β-driven move-ladder descent and the §3.5.1 design-change stop are replaced by the
frozen exhaustion rule `E = A OR B`, evaluated online; β keeps its place in the
Du–Olhoff optimization problem and loses all authority over continuation and
termination.

## 2. Files touched

Task-start `+impl/` tree SHA-256 `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files).
Post-implementation `+impl/` tree SHA-256 `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files).

| file | change |
|---|---|
| `+olh/+move/exhaustion.m` | **NEW.** The frozen detector, evaluated forwards in time. |
| `+olh/+move/limit.m` | `ladder` policy gains the `stageExhaustion` signal branch; controller state gains `ex`, `stageStarts`, `descents`. |
| `+olh/+config/schema.m` | `move.continuation.signal` enum gains `stageExhaustion`; **new** `stop.rule` (`designChange` \| `stageExhaustion`), default `designChange`. |
| `+olh/+config/validate.m` | refuses the half-selected controller and any non-`ladder` policy under it. |
| `+olh/+config/toLegacy.m`, `fromLegacy.m` | round-trip the two new values. |
| `olhoffSolve.m` | advances the detector once per outer iteration after the design update; replaces the admission rule when `stop.rule == 'stageExhaustion'`; records the controller trace; refuses the controller under projection. |

Nothing else in `+impl/` changed. Outside `+impl/`, two tests were maintained:
`tests/test_source_integrity.m` (`nFiles == 74` → `75`, since production gained
one source file) and `tests/test_evidence_retention.m` (`R10b` now compares the
`+impl` tree hash captured at test start instead of a pinned literal digest —
strictly stronger, and it asks the question the test is for).

## 3. Two switches, both defaulting to production

```
move.continuation.signal :  'boundVariable' (default, production) | 'designRms' | 'stageExhaustion'
stop.rule                :  'designChange'  (default, production) | 'stageExhaustion'
```

A configuration that names neither is byte-for-byte the solver that existed at
task start. Proven, not asserted: `test_preset_equivalence` re-solved 160×20
through the production entry point and matched the frozen conference record
**bitwise** — ρ bitwise, ω₁ bitwise, 91 outer, 2241 inner, `CONVERGED`.

`validate.m` refuses `stop.rule='stageExhaustion'` without
`move.continuation.signal='stageExhaustion'`: the terminal half alone would leave
a declaration below the last rung with nothing to consume it, and the run could
never stop. One rule, or neither.

## 4. Where the detector sits in the iteration

```
outer iteration k
  ├─ olh.move.limit(cfg, k, hist, mvState)        <- sees the detector through k-1
  │     stageExhaustion: if declared and stage < 4, descend one rung,
  │                      set stageStart := k, clear counters and the declaration
  ├─ FE · eigenproblem · sensitivities · filter   (UNCHANGED)
  ├─ nested MMA sub-problem                        (UNCHANGED)
  ├─ design update  rho <- clamp(rho + drho)       (UNCHANGED)
  ├─ olh.move.exhaustion(...)                     <- advanced ONCE, here
  └─ admission:  stop.rule == 'stageExhaustion'
        convOuter = declared AND stage == 4
```

The detector is read at the top of the next iteration and at the bottom of this
one. That is exactly the causal structure the β-stall detector already had
(`hist.beta` holds `1..k-1` when `limit` is called), so the intervention changes
*which* signal is consulted, not *when*.

Consequences, both preregistered:

* a declaration at `t` below the last rung ⇒ the first iteration executed at the
  new move level is `t+1`;
* a declaration at `t` **at** the last rung ⇒ the run ends at `t`.

## 5. What the detector computes

Character-for-character the frozen rule (`two_branch_maturity_240`
`PREREGISTRATION.md` §§2–7, executable form `scripts/tb_branches.m`):

```
amp(k)  = ||drho_k||_2                                   the increment the sub-problem returned
                                                          (= hist.dxNorm2, the inherited native measure)
d_k     = rho_k - rho_{k-1}                              the step actually taken
cos(k)  = <d_k, d_{k-1}> / (||d_k|| ||d_{k-1}||)
net(k)  = n2(rho_k - rho_{k-10}) / sum_{j=k-9..k} n2(d_j),      n2 = ||.||/sqrt(NE)
med20   = trailing 20-iteration median, 'omitnan'
tol     = cfg.stop.tolerance = 0.05*sqrt(NE/3200)         NO new constant

A(k) = med20 cos < 0  AND  med20 net < 0.5  AND  amp >= tol
B(k) = amp < tol      AND  med20 cos > 0
E    = A OR B,  declared after P = 20 consecutive iterations
```

Note that `amp` is the *increment* norm while `cos`/`net` are formed from the
*visited designs* — exactly the asymmetry the frozen rule has (`per.l2 =
hist.dxNorm2` from `dr_telemetry`, `cosT`/`net_ratio` from the `RHO` matrix).
Clamping makes the two differ only where the box is active; measured clamp
displacement on every candidate run is ≤ 5.6e-17.

## 6. Stage locality

`stageStart = s` is the first iteration executed at the current move level. Every
predicate input is formed only from `j ≥ s`; the trailing medians are defined only
once the 20-window lies wholly inside the stage (`j ≥ s+19`), which is exactly the
`k ≥ W` condition `tb_branches` applies with `s = 1`. Earliest possible
declaration is therefore `s+38`, on either branch.

Only the *anchor* `ρ_{s−1}` is pre-stage, used by `net_path` over the stage's
first ten steps — the same role `ρ_0` plays in the first stage. Every *step* in
every window is stage-local.

The reset is conservative: it can only delay a descent, never advance one. It
exists because under a move change `‖Δρ‖` is bounded by a different constant, so a
straddling window reports the schedule rather than the design — the defect
`stop.guards.settledMove` was written to suppress, applied to the whole window
rather than to one iteration.

## 7. β's remaining role

`β` is still the bound variable of Eq. (25a), still solved for, still recorded.
It is read by **nothing** in the candidate's continuation or admission path. Its
predicate is replayed at every iteration (`betaStallRel`, `betaStallFires`,
`prodStageShadow`, `prodMoveShadow`, `prodStopRaw`, `prodSettled`,
`prodStopAdmit`) and written to the per-iteration telemetry so the counterfactual
is on the record.

That replay is a **predicate replay on the realized path**, not a counterfactual
trajectory: production under its own rule would have left this path at its first
descent. It is reported as such and never as "what production would have
produced".

## 8. Deviation from the frozen preregistration, disclosed

`PREREGISTRATION.md` §8 test 5 states the earliest post-transition declarations as
`s+39` (B) and `s+47` (A). Those numbers are an arithmetic slip in a *test
description*. The binding requirement is §3.2 — "for the first stage the
stage-local definitions coincide exactly with `tb_branches`" — and exactness
forces `s+38` on both branches, because `tb_branches` defines its medians from
`k = W = 20` and then requires 20 consecutive, closing the first possible window
at 39.

The implementation follows §3.2, the binding clause. The frozen file is left
byte-identical; this note records the correction. It was made **before the first
candidate run**, it is inert for the science (the earliest theoretical
declaration is far before any observed event: 102, 274, 388 at stage 1), and no
threshold, window, persistence length or tolerance is affected.
