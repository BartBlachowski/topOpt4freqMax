# REPORT — causal validation of the frozen two-branch stage-exhaustion controller

Study directory `analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation`.
Preregistration frozen **before** implementation and therefore before any candidate
result existed, SHA-256 `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf`.

This study ran across two sessions. Session 1 (2026-09-08, HEAD `b6014ba`) froze the
preregistration, implemented the controller, ran the software and single-factor
gates, and executed C160 and C320. Session 2 (2026-09-09, HEAD `1438aa3`) recovered
lost raw evidence, re-verified every gate under a changed MATLAB build, and executed
the third authorized run. The controller source is **byte-identical** across both
sessions (`+impl` tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`),
which is what makes them one experiment rather than two.

---

## 1. Provenance and currentness (Phase 0)

**Q1. Branch / starting HEAD.** Branch `benchmark-methodology-r2`. HEAD at the start
of *this* session `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` ("A & B tests"), which
is session 1's committed work. Session 1's own starting HEAD was `b6014ba8bca4…`.
The brief's instruction to record current HEAD rather than assume one mattered: the
session-start snapshot showed `9ad79be`, three commits behind the real HEAD.

**Q2. Was the starting tree clean?** Yes — `git status --porcelain` empty at both
session starts. Machine-readable: `evidence/provenance_start.json`,
`evidence/provenance_resume_20260909.json`.

Gate results at resumption: currentness `CURRENT`; source integrity **PASS**
(0 mismatches, 0 missing, 0 extra, 75 files); dispatch ok with 0 blockers and 0
warnings; **published MMA wins**; **sensitivity filter wins**; forbidden Olhoff paths
**absent**; the `tol(NE) = 0.05·sqrt(NE/3200)` law identical to `cfg.stop.tolerance`
at all four meshes. Overall `gatePass = 1`.

**The MATLAB build changed between the runs and the resumption.** The preregistration
and both session-1 runs record `25.2.0.2998904` (R2025b base); the only R2025b
installation present now reports `25.2.0.3042426 (R2025b) Update 1`, and the base
build is no longer available. Because §6 of the preregistration conditions the C400
prefix check on *the same build*, this was tested rather than assumed, three
independent ways:

| test | mesh | scope | result |
|---|---|---|---|
| C160 candidate rerun vs committed session-1 run | 160×20 | 219 iterations × 55 columns | **54/55 bitwise**; only `tOuter` (wall clock) differs; final ρ SHA-256 identical |
| P400 production recompute vs committed telemetry | 400×50 | 139 iterations × 35 columns | **35/35 bitwise**; final ρ SHA-256 equals the frozen `baselines.json` value |
| `test_preset_equivalence` vs the frozen conference record | 160×20 | ρ, ω₁, volume, counts | **bitwise**; ω₁ 169.49522702153845, 91 outer, 2241 inner |

**The build change is numerically inert for this solver**, at 3 200 and at 20 000
elements. The three candidate runs are therefore cross-comparable and §6 remains a
meaningful bitwise test. Wall-clock times remain build- and load-sensitive and are
reported as such (the C160 rerun took 477.7 s against 425.3 s originally).

### 1.1 Raw evidence lost, recovered, and recomputed

`analysis/OlhoffCurrent/evidence/` was **empty** at resumption — the untracked
"fresh clone" state `EVIDENCE_POLICY.md` anticipates. The declared
`move_activity_400` evidence gate correctly reported `ok=0, required=2, missing=2`.

| artifact | status |
|---|---|
| `C160x20_trajectory.mat` | restored by the rerun; final ρ hash identical |
| `C320x40_trajectory.mat` | lost; regeneration run in session 2 (see §14, P15) |
| `P400_400x50_trajectory.mat` | **recomputed**; content bitwise, container hash new |
| `F400_400x50_trajectory.mat` | **recomputed**; enables §6 as a same-binary check |

Conversely, files the preregistration recorded as absent **are present**: the
production baseline trajectories at 160×20 and 320×40, and all four fixed-move
mechanism arms. They were verified rather than assumed — re-deriving `M_nd`, `gray`,
`mid` and `volume` from each baseline's final density column reproduces the frozen
`evidence/baselines.json` scalars to ≤1.8e-14 at the recorded `nOuter`. So the
production final densities that §9 marked UNAVAILABLE are in fact available, and the
topology comparison the preregistration expected to be impossible at 160×20 and
320×40 can be made. Details and the additive-only recovery procedure:
`PROVENANCE.md` §§A3–A4, `BASELINES.md` addendum, `evidence/baseline_recovery.json`.

**A convention inconsistency, disclosed.** `evidence/baselines.json` inherited two
different final-ω₁ conventions: 160×20 and 320×40 carry `res.omega(1)` (the final
re-solve, the same quantity `cv_run` records for every candidate), while 400×50
carries `hist.omega(end)` (162.882615630062). The like-for-like production value at
400×50, measured in this session, is `res.omega(1) = 162.8887798` — a relative
difference of 3.8e-5, three orders of magnitude below the P8 gate of 1 %. Both are
reported; P8 evaluates identically against either. See `PROVENANCE.md` §A5.

---

## 2. The recovered frozen rule (Phase 0, Q3–Q6)

**Q3. Branch A.**

```
A(k)  =  med20 cosθ(k) < 0   AND   med20 net_path(k) < 0.5   AND   amp(k) >= tol(NE)
```

**Q4. Branch B.**

```
B(k)  =  amp(k) < tol(NE)    AND   med20 cosθ(k) > 0
```

Branch B is the **inherited native design-change stop criterion plus a coherence
guard plus persistence**. It is not novel physics and is not presented as such.

**Q5. Persistence, windows, indexing, normalization.**

```
d_k       = rho_k - rho_{k-1}                       rho_0 = the uniform initial design
cosθ(k)   = <d_k, d_{k-1}> / (||d_k||·||d_{k-1}||)  NaN if either norm is 0
net_path  = ||rho_k - rho_{k-10}|| / sum_{j=k-9..k} ||d_j||        W_np = 10
amp(k)    = ||d_k||_2                                the inherited native measure
med20     = trailing median over [k-19, k], 'omitnan'; NaN median => predicate false
W = 20    P = 20    tol(NE) = 0.05*sqrt(NE/3200) = cfg.stop.tolerance
E(k)      = A(k) OR B(k)
```

A and B are mutually exclusive at any single `k` (`amp ≥ tol` versus `amp < tol`), so
at most one persistence counter runs and no tie is possible. Online, `E` is declared
at the first `t` where `nA(t) ≥ 20` or `nB(t) ≥ 20`; that is the same event the
retrospective detector reports as beginning at `t − 19`. Every quantity is
**stage-local**: at a move transition `stageStart := t+1` and the windows and counters
reset, with `net_path` anchored on the design at the moment the stage began. The reset
is conservative — it can only delay a descent, never advance one.

**Q6. Did the definitions match the withheld 240×30 preregistration?** Yes. They were
recovered from `two_branch_maturity_240/PREREGISTRATION.md` (SHA-256
`62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd`) §§2–7 and its
executable form `scripts/tb_branches.m`, and reproduced character-for-character in
§2 of this study's preregistration. No threshold, window, persistence length,
normalization or tolerance was introduced or altered.

Recovery was verified numerically, not merely read. Session 1 re-executed
`tb_branches` against the surviving F400 arm and reproduced the frozen 400×50 numbers
bit-exactly. Session 2, after F400 had to be recomputed, re-derived the same
quantities from the frozen *definitions* in an independent Python implementation
(`scripts/cv_frozen_rule_check.py`), so agreement is now between two implementations
rather than a re-run of one. Verdict: **`CONTROLLER_DEFINITION_RECOVERY_PASS`**.

**Disclosed limitation, carried forward.** F400 stops at its own native stop at
iteration 369, so the online declaration implied for that arm (388) lies beyond the
available data. The *first* iteration of Branch B's window is verifiable there; its
20-iteration persistence is not. This was recorded before the runs and is why the
400×50 declaration iteration was preregistered as a prediction rather than a
certainty.
---

## 3. Preregistration, single-factor and software gates (Q7–Q9)

**Q7. Was the causal preregistration frozen before candidate results?** Yes.
`evidence/PREREGISTRATION.sha256` records the freeze at 2026-09-08T18:51:03Z against
HEAD `b6014ba`, explicitly *before the controller was implemented* and therefore
before any candidate result of any kind existed. `PREREGISTRATION.md` and
`evidence/PREREGISTRATION.frozen` are byte-identical and both hash to
`8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf`, matching the
recorded value. Nothing in it was altered afterwards.

**Q8. Did the single-factor gate pass?** Yes — **`CONTROLLER_SINGLE_FACTOR_PASS`**,
re-verified in session 2 under the new build with byte-identical output
(`git diff` on `evidence/single_factor.json` is empty). At each of the three meshes
every field of the configuration schema was compared by `isequaln` between the
production entry point *at the candidate's cap and recorder setting* and the
candidate. Exactly three fields differ at every mesh:

| field | production | candidate | role |
|---|---|---|---|
| `move.continuation.signal` | `boundVariable` | `stageExhaustion` | **the intervention** |
| `stop.rule` | `designChange` | `stageExhaustion` | **the intervention** |
| `runtime.name` | label | label | free text, excluded from the config hash |

`unexpected = {}` at all three meshes and the scientific lock reads OK. Comparing
against production *at cap 1600 with diagnostics on* is what prevents the safety cap
and the recorder from masquerading as the intervention. Candidate config hashes:
160×20 `31d2ef38…`, 320×40 `2359a111…`, 400×50 `0afb0d4d…` — each equal to the
`cfgHash` recorded inside the corresponding run record, so the configuration audited
is the configuration that ran.

The single-factor argument is additionally proved *dynamically*, not only by
comparing fields. Under the candidate, stage 1 holds `move = 0.04` and changes
nothing else, so its prefix must coincide with an independent fixed-move 0.04 arm:

| mesh | comparison | result |
|---|---|---|
| 160×20 | candidate ρ trajectory, iterations 1…102, vs `fixedmove_160x20.mat` | **bitwise identical**; first difference at exactly iteration **103**, the descent |
| 400×50 | candidate ρ trajectory vs recomputed `F400`, §6 check | see §6 below |

The 160×20 result is a check the preregistration expected to be unavailable, and it
is the sharpest single-factor evidence in the study: the candidate is provably the
production solver until the iteration on which the controller acts.

**Q9. Did all software tests pass?**

*Controller tests (preregistration §8).* Session 1: **17 of 17 pass, nFail = 0**,
recorded in `evidence/software_tests.json` before any scientific run — which is what
Phase 7 requires. Re-run in session 2 (`evidence/software_tests_rerun_20260909.json`):
**16 of 17**, the single failure being item 16's F400 sub-check reporting
`F400:MISSING` because the trajectory had been lost. Two things must be said plainly
about that item: its four synthetic sub-cases (A, B, neither, both-in-sequence) all
pass, matching the frozen `event + 19` rule and branch identity; and its F400
sub-case was **already vacuous when the file existed** — session 1 recorded it as
`F400:noEvent(decl=0)`, because the online declaration for that arm falls at 388,
past F400's 369 iterations. So the re-run failure is a statement about data
availability, not about controller logic, and the substantive content of test 16
passes in both sessions. Both records are retained; neither overwrote the other.

Coverage of the mandated items: 1–3 branch truth table (plus `3b`, the documented
hole); 4 persistence boundary 19 vs 20; 5 window reset after descent; 6 exactly one
rung per accepted transition; 7 `move` never below 0.005; 8 β stall alone cannot
descend; 9 β stall alone cannot admit convergence; 10 `move_min` + `E` false cannot
converge; 11 `move_min` + persistent `E` converges at the declaring iteration; 12
`CAP_HIT` remains `CAP_HIT`; 13 failure remains failure; 14 production config still
selects the production controller; 15 telemetry alters no numerical result; 16
online == frozen offline detector.

*Repository suite.* `analysis/OlhoffCurrent/tests`: **5 of 5 pass**
(`evidence/suite_tests_20260909.json`) — `test_currentness`,
`test_evidence_retention`, `test_path_isolation`, `test_source_integrity`,
`test_preset_equivalence`. One harness caveat is recorded rather than hidden:
`test_path_isolation` deliberately mutates the MATLAB path, and when the whole suite
runs in a single process it can leave `tests/` off the path, which made the two
following tests report `Unrecognized function`. Re-run in a fresh process both pass.
That is a harness ordering artifact, not a product failure.

`test_preset_equivalence` deserves its own line because it discharges the strongest
form of item 14: with both switches at their production defaults the solver
reproduces the frozen conference record **bitwise** — ρ bitwise, ω₁ bitwise, volume
bitwise, 91 outer, 2241 inner, `CONVERGED`. Production behaviour is not merely
"still selectable"; it is unchanged to the last bit.

---

## 4. Implementation and β's loss of authority (Q13, Q14)

The intervention is two configuration switches, both defaulting to production:

```
move.continuation.signal :  'boundVariable' (default) | 'designRms' | 'stageExhaustion'
stop.rule                :  'designChange'  (default) | 'stageExhaustion'
```

`validate.m` refuses the half-selected controller: `stop.rule='stageExhaustion'`
without the matching signal would leave a declaration below the last rung with
nothing to consume it and the run could never stop. One rule, or neither.

**Q13. Did β lose all authority over continuation?** Yes. Under the candidate the
ladder descends only on a declared `E`. β is still computed and is still the bound
variable of Du–Olhoff Eq. (25a) — untouched inside the optimization problem — and the
controller never reads it.

**Q14. Did β lose all authority over terminal admission?** Yes. Terminal admission is

```
convOuter = mvState.ex.declared && (stage >= numel(moveLevels))
```

with no β term and with production's §3.5.1 design-change test and its
settled-move/restoration guards replaced wholesale rather than combined with. Those
production predicates remain computed and are logged at every iteration as the causal
counterfactual (`prodStageShadow`, `prodMoveShadow`, `prodStopRaw`, `prodSettled`,
`prodStopAdmit`, `betaStallRel`, `betaStallFires`). Tests 8 and 9 exercise exactly
these two properties and pass.

A minor documentation/code discrepancy found while auditing this path, recorded for
completeness and affecting nothing here: `local_status`'s docstring says the cap
outranks convergence, but the code tests `CONVERGED` before `CAP_HIT`, so a run that
converged exactly at `maxOuter` would report `CONVERGED`. No run in this study is
near that edge — C160 converged at 219 of 1600, and C320 reached 1600 with no
convergence log and is correctly classified `CAP_HIT`.
---

## 5. The three authorized runs (Q10–Q12)

**Q10/Q11. Were exactly three candidate scientific runs executed, and were they
160×20, 320×40 and 400×50?** Yes, and no others. `cv_run` refuses any other mesh in
code rather than trusting discipline: `AUTHORIZED = [160 20; 320 40; 400 50]`, and an
unauthorized mesh raises `cv_run:UnauthorizedMesh`. No 240×30 candidate was run; none
of 480×60, 560×70, 640×80, 720×90, 800×100 was run; the nine-mesh campaign was not
started.

| run | status | outer | inner | wall s | descents (branch) | terminal |
|---|---|---|---|---|---|---|
| **C160x20** | `CONVERGED` | 219 | 5 074 | 425.3 | 103 (A), 142 (A), 181 (B) | B at 219, window 200–219 |
| **C320x40** | **`CAP_HIT`** | 1 600 | 76 532 | 34 373.8 | 275 (A), 314 (B), 353 (B) | **none** — cap |
| **C400x50** | `CONVERGED` | 505 | 10 301 | 4 689.3 | 389 (B), 428 (B), 467 (B) | B at 505, window 486–505 |

**Q12. Was the controller unchanged across all three?** Yes, and this is machine-proved
rather than asserted. Every run record carries

```
implTree = edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
```

and that is the `+impl` tree hash measured independently at the start of *this*
session, after all three runs. One controller, byte-identical source, three meshes,
two sessions and two MATLAB builds. Each run's `cfgHash` also equals the value the
single-factor audit recorded for that mesh before any of them ran
(160×20 `31d2ef38…`, 320×40 `2359a111…`, 400×50 `0afb0d4d…`).

**Q31. Did any candidate hit a cap?** Yes — **320×40**, at the preregistered
`runtime.maxOuter = 1600`. It is reported as `CAP_HIT`, not reclassified, and the cap
was **not** raised after seeing the trajectory. §8.2 explains the mechanism.

**Q32. Did any inner solver fail?** No. `innerNonConv = 0` on all three runs
(0 of 219, 0 of 1 600, 0 of 505); no `SOLVER_FAILURE`; `innerMax` 53, 139 and 46
respectively, all below the inner cap.

---

## 6. The 160×20 coarse-mesh safety test (Phase 17, Q17–Q18)

The preregistration calls 160×20 "the critical coarse-mesh safety case" and forbids
large fine-mesh gains from hiding a coarse-mesh failure. It does not fail.

| | production | candidate |
|---|---|---|
| status | `CONVERGED` (native) at 91 | **`CONVERGED`** at 219 |
| move transitions | 79 (.04→.02), 90 (.02→.01) | **103 (A), 142 (A), 181 (B)** |
| final move / stage | 0.01 / 3 — ladder never finished | **0.005 / 4 — ladder finished** |
| ω₁ | 169.495227 | **170.011316** (+0.304 %) |
| ω₂ / gap₁₂ | 171.9600 / 0.014766 | 171.4283 / 0.008334 |
| M_nd % | 13.4025 | **12.7041** (−5.21 %) |
| gray / mid | 0.149375 / 0.025000 | 0.144375 / 0.026875 |
| volume | 0.499999009 | 0.499999134 (\|v−0.5\| = 8.7e-7) |
| outer / inner / wall | 91 / 2241 / 125.5 s | 219 / 5074 / 425.3 s (×2.41 / ×2.26 / ×3.39) |

Answering Phase 17 point by point:

1. **Did Branch A recognize the mature high-amplitude cycle?** Yes. A declared at
   iteration 102 on the window 83–102, with `med₂₀cos = −0.907`, `med₂₀net = 0.133`
   and `amp = 0.6296` against `tol = 0.05` — a textbook high-amplitude cancellation
   signature, comfortably above A's amplitude clause. The preregistered prediction
   was "A declares at 102, first descent 103 (±5)". The realized values are **102 and
   103 — exact, not merely within tolerance.**
2. **Did move descend without indefinite churning?** Yes. Three descents at 103, 142,
   181 — 39 iterations per stage after the first, the minimum possible under a
   20-iteration window plus a 20-iteration persistence requirement on stage-local
   data. There is no churn: each stage exhausted at the earliest iteration the frozen
   rule permits.
3. **Did the candidate traverse later move levels?** Yes — all four. It is the only
   configuration in this study, candidate or production, that reaches `move = 0.005`
   and converges there. Production never leaves 0.01.
4. **Did final ω₁ remain scientifically acceptable?** Better than acceptable: ω₁
   *improved* by +0.304 % in a maximization problem. Bound P5 required ≥ 167.8003;
   realized 170.0113.
5. **Did final M_nd remain acceptable?** It improved by 5.21 %. Bound P5 allowed up to
   1.10 × production (≤ 14.7427); realized 12.7041.
6. **Did the candidate introduce a regression relative to production?** No — not in
   M_nd, ω₁, gray fraction, volume feasibility or termination honesty. The only cost
   is computational: ×2.41 outer, ×2.26 inner, ×3.39 wall, all inside P13's ×8 / ×10.
7. **Did any branch fire pathologically early?** No. The earliest arithmetically
   possible declaration in stage 1 is iteration 20 (a 20-iteration window from
   `s = 1`, with `net_path` defined from 10); A fired at 102. In later stages the
   earliest possible is `s + 38`; each fired at exactly `s + 38`, which is the
   earliest legal value but not an early *artifact* — it is the rule's own floor
   given a stage-local reset, and it is reached because the design genuinely was
   cycling on arrival.
8. **Did terminal admission remain honest?** Yes. Termination at 219 required
   `stage == 4` **and** a declared `E`: Branch B held for 20 consecutive iterations
   (200–219) with `amp = 0.00234 < tol = 0.05` and `med₂₀cos = +0.892 > 0`. The
   terminal stage ran 39 iterations with A never true and B true, `amp/tol` median
   0.0575 — genuine amplitude convergence, coherent, not a manufactured signal.

**Q17. Did 160×20 escape its high-amplitude mature cycle?** Yes, and by the intended
mechanism: Branch A detected the cycle and the ladder descended, after which the
amplitude fell sharply at every stage (stage medians `amp` 0.597 → 0.116 → 0.0063 →
0.0029, i.e. ×5, ×18, ×2) and the dynamics turned coherent (stage medians `med₂₀cos`
−0.850 → +0.595 → +0.921 for stages 2, 3, 4). The coarse mesh is where Branch A carries the whole controller, and it works.

**Q18. Did 160×20 suffer any scientific regression?** No. Both preregistered
one-sided bounds pass with margin, in the favourable direction.

One honest qualification, inherited and not re-litigated here: the preregistration
records 160×20 as "the weakest mechanism case", because its exhaustion event leaves
more relative M_nd improvement available than the finer meshes do. That remains
true — and this task was explicitly forbidden from retroactively tightening the old
acceptance rule, so it does not.
---

## 7. What the later move stages actually do — a structural finding

This was not anticipated by the preregistration and is the most consequential thing
the intervention revealed. It is a *diagnosis*, not a proposed change; nothing was
altered in response to it.

Because every predicate is stage-local, the trailing 20-median first becomes
evaluable at `s + 19` (`'omitnan'` means a window holding 19 defined entries already
yields a defined median), so the earliest arithmetically possible declaration in any
stage is `s + 38`. Measured across both completed meshes:

| mesh | stage | `s` | med₂₀ first defined | `E` first true | declared | span |
|---|---|---|---|---|---|---|
| 160×20 | 1 | 1 | 20 = s+19 | **83** = s+82 | 102 | 20 |
| 160×20 | 2 | 103 | 122 = s+19 | **122** = s+19 | 141 = **s+38** | 20 |
| 160×20 | 3 | 142 | 161 = s+19 | **161** = s+19 | 180 = **s+38** | 20 |
| 160×20 | 4 | 181 | 200 = s+19 | **200** = s+19 | 219 = **s+38** | 20 |
| 320×40 | 1 | 1 | 20 = s+19 | 216, but lapses | 274 (window 255–274) | 20 |
| 320×40 | 2 | 275 | 294 = s+19 | **294** = s+19 | 313 = **s+38** | 20 |
| 320×40 | 3 | 314 | 333 = s+19 | **333** = s+19 | 352 = **s+38** | 20 |
| 320×40 | 4 | 353 | 372 = s+19 | **never** | never | — |

In **every** stage after the first, on **both** meshes, `E` becomes true at the very
first iteration on which it is capable of being true, and the declaration follows
exactly 20 iterations later at the arithmetic floor `s + 38`. Stage 1 is different and
genuinely informative. At 160×20 `E` first holds at 83 — 82 iterations into the stage —
and then holds cleanly for exactly the required 20. At 320×40 it holds for **7**
iterations (216–222), **lapses for 32**, and only then holds the required 20
(255–274).

That lapse is worth pausing on, because it is the only place in the study where the
`P = 20` persistence requirement demonstrably earns its keep: without it the ladder
would have descended at roughly iteration 223, **53 iterations earlier**, on a signal
that did not hold. The persistence rule is not decorative at the one mesh and in the
one stage where the detector is doing real work.

**The controller's scientific content is therefore confined to stage 1.** Stages 2–4
detect nothing — by the time the window is evaluable the design already satisfies the
rule — and each contributes a fixed 39 iterations of window-and-persistence overhead
independent of the mesh and of the design. At 160×20 that is 117 of 219 outer
iterations, **53 % of the run**, spent waiting out the rule's own constants rather
than measuring the optimizer.

This is not a hidden defect in the frozen rule so much as a consequence of what the
rule was characterized for. It was developed and withheld-validated as a *maturity
detector at a fixed move of 0.04*, where the design is still evolving. Stages 2–4 ask
it a different question — whether an already-converged design at a small move has
"exhausted" — and there it answers immediately and uninformatively, or, at 320×40's
terminal stage, never (§8).

**One internal inconsistency in the frozen preregistration, disclosed.** §8 item 5
states the earliest possible post-transition declarations are `s+39` (B) and `s+47`
(A). The measured floor is `s+38` for both, and the implementation's own test asserts
"no decl before s+38". The `s+39`/`s+47` figures follow from requiring the 20-window
to be *entirely* filled with defined entries; the authoritative definition in §2.1
specifies `'omitnan'` medians, which become defined one iteration earlier and make
both branches evaluable from `s+19`. The implementation follows §2.1, the definition
section, over §8's arithmetic remark. No threshold, window length, persistence length,
tolerance or union was affected — the discrepancy is purely about when the median
first exists, and §2.1 fixes that unambiguously.

---

## 8. The 320×40 fine-mesh causal test (Phase 16, Q19–Q20)

This is the primary causal case, and it splits cleanly into a confirmed causal effect
and a failure to terminate.

### 8.1 The causal effect is confirmed, and is large

**Q19. How much later than production did 320×40 remain at move = 0.04?**
Production descends at **130**; the candidate descends at **275** — **145 iterations
later**, far beyond P6's required ≥ 180.

**Q20. What did it gain during that interval?** Nearly everything. The comparison is
unusually clean because until production's first descent the two runs are the same
trajectory:

| at iteration | candidate M_nd | candidate ω₁ | candidate gray |
|---|---|---|---|
| 130 — production's descent | 23.3224 | 165.9433 | 0.2633 |
| 131 — production's convergence (final: M_nd 23.3596, ω₁ 165.9508) | 23.1962 | 165.9671 | 0.2619 |
| 275 — candidate's own first descent | **12.9957** | **166.4072** | **0.1523** |

So at the moment production stops, the candidate is in materially the same state
production ends in (M_nd 23.32 vs 23.36). Over the next 145 iterations at unchanged
`move = 0.04`, M_nd falls **−10.33 points, −44.28 %**, while ω₁ *rises* by +0.46.
Grayness nearly halves. **The entire improvement is attributable to delaying the
premature β-driven continuation** — there is no other difference between the runs, as
§3 establishes field-by-field and, at 160×20, bitwise.

Final outcome: M_nd **23.3596 → 12.9233 (−44.68 %)**, ω₁ **165.950789 → 166.426697
(+0.287 %)**, volume 0.499999617. Both P6 clauses pass with wide margin (M_nd
12.9233 against the ≤ 18.6877 bar).

M_nd first comes within 0.1 of its final value at iteration **257** — before the first
descent. Everything of scientific value at this mesh is achieved in stage 1.

### 8.2 The run never terminated — the preregistered hole, realized

The candidate reached `move = 0.005` at iteration 353 and then ran to the cap:
**`CAP_HIT` at 1600**. The mechanism is exactly the one §10 preregistered as an
accepted risk:

```
terminal stage (353…1600, 1248 iterations at move = 0.005)
    amp median      0.00487        tol = 0.1        so  amp/tol ≈ 0.049
    med20 cos       −0.885         (min −0.935, max −0.339; never positive)
    Branch A        blocked by its amplitude clause   (amp >= tol fails)
    Branch B        blocked by its coherence guard    (med20 cos > 0 fails)
    E               never true; nA and nB never exceed 0
```

This is **low-amplitude cancellation**: the design oscillates with period-2
cancellation at an amplitude twenty times below the convergence scale. It satisfies
neither branch, so the union is blind to it and no terminal declaration is possible.

Three further measurements make the character of this state precise:

* **It is a stationary limit cycle, not slow progress.** Over 1248 iterations M_nd
  spans 0.052 (0.4 % of its value) with a drift between halves of −0.00055; ω₁ spans
  0.028 with drift −0.0025; gray spans 0.00078. Nothing is happening.
* **It is not a move-bound artifact.** `max|Δρ|/move` has median **0.0201** in the
  terminal stage — the move limit is 50× slacker than the steps being taken and is
  simply not active. The oscillation is intrinsic to the optimizer at a converged
  design, so descending the ladder further could not have helped either.
* **It is where all the cost is.** The terminal stage consumes **91.5 % of all inner
  MMA iterations** (70 034 of 76 532; median `nInner` rises from 17–19 in earlier
  stages to 62) and **95.9 % of wall time**, producing no change in the design.

For the record, and explicitly as counterfactual arithmetic rather than a proposed
remedy: had a terminal declaration been possible at this stage's arithmetic floor
(`s+38` = 391), the run would have ended near iteration 391 for an outer multiplier
of ≈ 2.98× — inside P13's ×8. **P13's failure at this mesh is caused solely by the
union's blind spot, not by the delayed continuation that is the intervention's
purpose.** That distinction matters for interpreting the verdict, and it is the
reason the verdict is not a flat rejection. Nothing was changed in response to it;
repairing the rule is explicitly out of scope for this task.
---

## 9. The 400×50 fine-mesh causal test (Phase 16, Q21–Q22)

This is the cleanest case in the study: the causal effect is the largest, the run
terminated genuinely, and the preregistered prediction was exact.

**The prediction was met to the iteration.** §10 preregistered: *"Branch B declares at
388, first descent 389 (exact — same MATLAB build, and the §6 prefix check applies)."*
Realized: declaration at **388** on the window **369–388**, first descent at **389**.
Not within a tolerance — the stated value.

That precision is itself a substantive result. F400 stops at its own native stop at
iteration 369, so the preregistration could verify only that Branch B *first* holds
there, and explicitly disclosed that its 20-iteration persistence "cannot be
re-verified from surviving data". C400 supplies exactly that missing evidence: B held
continuously across 369–388, so the declaration fell where the frozen rule said it
would.

**Q21. How much later than production did 400×50 remain at move = 0.04?**
Production descends at **138**; the candidate at **389** — **251 iterations later**,
against P7's requirement of ≥ 188.

**Q22. What did it gain during that interval?**

| at iteration | M_nd | ω₁ | note |
|---|---|---|---|
| 138 — production's descent | 32.3178 | 162.8766 | production ends one iteration later at M_nd 32.3283, ω₁ 162.8826 |
| 388 — candidate's declaration | **15.6649** | **166.4176** | −51.53 % M_nd, **+2.174 % ω₁** |

Over 251 extra iterations at unchanged `move = 0.04`, M_nd falls by more than half
*and* ω₁ rises by 2.17 % — in a maximization problem, the objective improves while the
design becomes discrete. Final: M_nd **32.3283 → 15.3311 (−52.58 %)**, ω₁
**162.882616 → 166.456276 (+2.194 %)**, volume 0.499999432.

Answering Phase 16 point by point:

1. **Remained at 0.04 beyond production's β descent?** Yes.
2. **By how many iterations?** 251 (138 → 389).
3. **What topology change occurred?** Grayness roughly halved (gray fraction
   0.3476 → 0.1796; mid-density 0.1882 → 0.0392). Figure `F8_topology.png` shows what
   the scalars mean: production ends with large diffuse grey regions at both ends of
   the beam; the candidate resolves those into discrete truss members.
4. **M_nd change?** −16.9972 absolute, **−52.58 %** relative.
5. **ω₁ change?** **+3.5737 absolute, +2.194 %** — an improvement, not a regression.
6. **Which branch triggered descent?** **B**, at all three transitions, and again at
   terminal admission. 400×50 is Branch B's mesh exactly as the mechanism studies
   found: coherent amplitude convergence, never the cancellation signature.
7. **Did later move stages also exhaust successfully?** Yes — 428 and 467, both B.
8. **Genuine terminal exhaustion?** Yes. `CONVERGED` at 505 with `stage == 4`,
   `move == 0.005` and Branch B held for the full 20 iterations 486–505; the terminal
   stage ran 39 iterations with `amp/tol` median 0.0129 and `med₂₀cos` median +0.511.
9. **Final M_nd materially better?** Yes: 15.3311 against 32.3283, a 52.58 % reduction
   against a 20 % bar.
10. **Cost?** ×3.63 outer, ×3.53 inner MMA, ×8.67 wall — all inside P13's ×8 / ×10.
    The wall figure is *conservative*: C400 shared the machine with the C320
    verification rerun, so contention can only have inflated it.

### 9.1 The §6 prefix-equivalence check — the hard falsifiable test

The preregistration set one check that could have falsified the whole single-factor
claim outright:

> **C400 outer iterations 1…369 must reproduce `F400` bitwise** — `RHO` columns,
> `hist.omega`, `hist.beta`, `hist.dxNorm2`, `hist.nInner` all exactly equal.

**It passes, on all five quantities:**

| compared over iterations 1…369 | result |
|---|---|
| `RHO` (20 000 × 369 densities) | **bitwise identical** |
| `hist.omega` | **bitwise identical** |
| `hist.beta` | **bitwise identical** |
| `hist.dxNorm2` | **bitwise identical** |
| `hist.nInner` | **bitwise identical** |
| `hist.move` ≡ 0.04 throughout | yes |

This check was in jeopardy: F400 had been lost, and the preregistration conditions it
on the same MATLAB build, which had changed. Recomputing F400 under the current build
restored it as a *same-binary* test — arguably a stronger one than originally
available. The candidate is therefore provably the production solver, to the last bit,
for 369 iterations, and differs only from the iteration on which the controller acts.
---

## 10. What triggered every transition, and what β would have done (Q15, Q16)

**Q15/Q16. What triggered every candidate move transition? Did any occur without
`A OR B`?** Every transition on every mesh was triggered by a frozen declaration with
its persistence counter at exactly 20, and **none** occurred without one. The full
audit is in `CAUSAL_ANALYSIS.md` §2; the branch attribution is:

| mesh | T1 | T2 | T3 | terminal |
|---|---|---|---|---|
| 160×20 | 103 — **A** (window 83–102) | 142 — **A** (122–141) | 181 — **B** (161–180) | 219 — **B** (200–219) |
| 320×40 | 275 — **A** (255–274) | 314 — **B** (294–313) | 353 — **B** (333–352) | none — cap |

Both branches carry real weight: A fires 3 times, B fires 4 times (including one
terminal admission). Neither branch alone would have sufficed — which is the same
conclusion the withheld 240×30 mechanism test reached, now reproduced under
intervention.

**The β counterfactual is stark.** β's stall detector and production's stage ladder
were replayed at every iteration and logged but never read:

| mesh | β first stalls | production ladder would reach stage 4 by | production's native stop would admit at | candidate first descent |
|---|---|---|---|---|
| 160×20 | **79** | 101 (descents 79, 90, 101) | 143 | **103** |
| 320×40 | **130** | 152 (descents 130, 141, 152) | 216 | **275** |

At every candidate transition the β-stall flag was already raised (`betaStall = 1`)
and production's shadow stage was already **4** — production would long since have
run the ladder to the bottom. That the candidate descended at 103 and 275 instead
demonstrates the authority transfer directly: the controller ignored a signal that
was continuously asserting "descend", and waited for the design instead. This is the
causal content of `BETA_TRANSITION_SIGNAL_STRUCTURALLY_UNSUITABLE`, now shown by
intervention rather than by correlation.

---

## 11. Feasibility and physics safety (Phase 18, Q27, Q28)

**Q27. Were volume constraints satisfied?** Yes, on every mesh, with three orders of
magnitude of margin against the preregistered P9 bound of 1e-4:

| mesh | final volume | \|volume − 0.5\| |
|---|---|---|
| 160×20 | 0.49999913419341135 | 8.66e-7 |
| 320×40 | 0.49999961671422705 | 3.83e-7 |

**Q28. Was multiplicity behaviour acceptable?** Yes. Checked per iteration over the
whole trajectory, not only at the endpoint:

| check | 160×20 | 320×40 |
|---|---|---|
| ω₁, ω₂ finite everywhere | yes | yes |
| ω₂ > ω₁ at every iteration | yes | yes |
| subspace size (`multN`) | 2 throughout | 2 throughout |
| minimum gap₁₂ over the run | 0.004151 | 0.004141 |
| final gap₁₂ | 0.008334 | 0.223872 |
| any NaN/Inf in `hist` | none | none |
| inner-solver non-convergences | **0** of 219 | **0** of 1600 |

The fixed two-mode subspace with diagonal offsets and off-diagonal terms was never
violated, and no eigensolver failure occurred. Two early log entries per run note
`omega_J (J=3) is itself multiple -- (25b) undefined` at iterations 8/12 (160×20) and
9 (320×40); this is the implementation's existing, declared caveat handling in the
first few iterations from a uniform design, not a controller effect.

Worth noting rather than passing over: at 320×40 the candidate's final gap₁₂ is
**0.2239** against production's 0.1067 — the candidate ends *further* from
multiplicity than production, with ω₂ at 203.68 versus 183.77. The controller changes
the trajectory by design, so equality with production was never required; the
direction here is away from the pathological case, not toward it.
---

## 12. Final comparison and cost (Phase 15, Q23–Q26, Q33–Q35)

**Q23/Q25. Production versus candidate M_nd, and relative change.**

| mesh | production | candidate | absolute | relative |
|---|---|---|---|---|
| 160×20 | 13.4025 | **12.7041** | −0.6984 | **−5.21 %** |
| 320×40 | 23.3596 | **12.9233** | −10.4363 | **−44.68 %** |
| 400×50 | 32.3283 | **15.3311** | −16.9972 | **−52.58 %** |

**Q24/Q26. Production versus candidate ω₁, and relative change.**

| mesh | production | candidate | absolute | relative |
|---|---|---|---|---|
| 160×20 | 169.495227 | **170.011316** | +0.516089 | **+0.304 %** |
| 320×40 | 165.950789 | **166.426697** | +0.475908 | **+0.287 %** |
| 400×50 | 162.882616 | **166.456276** | +3.573660 | **+2.194 %** |

ω₁ improves on **every** mesh. This deserves emphasis because it is the opposite of
the usual trade: the controller does not buy discreteness by giving up the objective.
It improves both, and the improvement in both grows with mesh refinement — which is
consistent with the mechanism, since a finer mesh gives production's premature descent
more topology evolution left to discard.

Note also the scaling: M_nd improvement 5.21 % → 44.68 % → 52.58 % as the mesh
refines. Production's grayness problem is a fine-mesh problem, and the intervention
addresses it where it exists.

**Q33–Q35. Cost multipliers.**

| mesh | outer | inner MMA | wall |
|---|---|---|---|
| 160×20 | 91 → 219, **×2.41** | 2 241 → 5 074, **×2.26** | 125.5 → 425.3 s, **×3.39** |
| 320×40 | 131 → 1 600, **×12.21** | 2 614 → 76 532, **×29.28** | 388.0 → 34 373.8 s, **×88.60** |
| 400×50 | 139 → 505, **×3.63** | 2 918 → 10 301, **×3.53** | 541.0 → 4 689.3 s, **×8.67** |

Two of three meshes sit comfortably inside the preregistered ×8 outer / ×10 wall
envelope. 320×40 does not, and it is the reason P13 fails and promotion is refused.
Its cost is not the price of the intervention: §8.2 shows 91.5 % of its inner work and
95.9 % of its wall time were spent in a stationary limit cycle *after* the science was
finished, because the frozen union could not declare the terminal stage exhausted.

**No improvement in this report is stated without its cost**, as Phase 15 requires. The
honest one-line summary is: at 400×50 the controller buys a 52.58 % grayness reduction
*and* a 2.19 % ω₁ improvement for ×3.6 the iterations; at 160×20 it buys 5.21 % and
0.30 % for ×2.4; at 320×40 the same class of gain is available but the run cannot stop
and costs ×12.2 before a cap intervenes.

**Measurement caveat, disclosed.** C400 and the C320 verification rerun shared the
machine, both single-threaded on ten cores, at the user's direction to shorten total
elapsed time. C400's ×8.67 wall multiplier is therefore an **upper bound** — contention
can only inflate it — and the outer and inner multipliers, which count work rather
than time, are unaffected. C160's and C320's wall figures are the uncontended
session-1 measurements. This cannot change any gate: ×8.67 passes P13 and would pass
by more if measured alone.

---

## 13. Termination audit (Phase 14, Q29, Q30)

**Q29. Did all candidate runs reach move = 0.005?** Yes, all three — the first
configuration in this project's history to traverse the full declared ladder. Production
reaches 0.01 at 160×20 and only 0.02 at both fine meshes.

**Q30. Did all reported `CONVERGED` states satisfy terminal `A OR B` persistence?**
Yes, and the only run that did *not* satisfy it did not report `CONVERGED`.

| mesh | status | iter | move | stage | branch | nB | amp/tol median | med₂₀cos median | ‖Δρ‖₂ | genuine |
|---|---|---|---|---|---|---|---|---|---|---|
| 160×20 | `CONVERGED` | 219 | 0.005 | 4 | **B** (200–219) | **20** | 0.0575 | +0.921 | 0.00234 | **yes** |
| 320×40 | **`CAP_HIT`** | 1600 | 0.005 | 4 | — | 0 | 0.0487 | **−0.885** | 0.00419 | no |
| 400×50 | `CONVERGED` | 505 | 0.005 | 4 | **B** (486–505) | **20** | 0.0129 | +0.511 | 0.00135 | **yes** |

Both convergences satisfy the preregistered conjunction exactly: `move == 0.005`
**and** `stage == 4` **and** a frozen branch held its full 20-iteration persistence.
Neither was admitted by β stall, by the native design-change test, or by the cap. The
320×40 row is the honest negative: it reached the terminal rung and stayed there,
`E` never true, and it is classified `CAP_HIT`.

Nothing was relabelled. `CAP_HIT` remained `CAP_HIT` (software test 12), failure would
have remained failure (test 13), and no cap hit, inner failure, globalization failure
or numerical failure is described anywhere in this study as convergence.
---

## 14. Promotion gates and verdict (Phase 20–24, Q36–Q45)

**Q36. Did all preregistered promotion gates pass?** No — **thirteen of fifteen pass.
P13 fails on cost at 320×40, and P15 is failing at the time of writing because one
declared raw trajectory is still being regenerated.**

| gate | requirement | result | evidence |
|---|---|---|---|
| **P1** | software gate (§8) and single-factor gate (§5) both pass | **PASS** | 17/17 before the runs and 17/17 after evidence restoration; `CONTROLLER_SINGLE_FACTOR_PASS` |
| **P2** | no mesh terminates falsely | **PASS** | both `CONVERGED` runs at `move = 0.005`, stage 4, full 20-iteration persistence |
| **P3** | no unacknowledged failure | **PASS** | no `SOLVER_FAILURE`; `innerNonConv = 0` on all three; `CAP_HIT` reported as such |
| **P4** | 160×20 first descent ≤ 400 | **PASS** | 103 |
| **P5** | 160×20 M_nd ≤ 14.7427 and ω₁ ≥ 167.8003 | **PASS** | 12.7041 and 170.0113 |
| **P6** | 320×40 descent ≥ 180 and M_nd ≤ 18.6877 | **PASS** | 275 and 12.9233 |
| **P7** | 400×50 descent ≥ 188 and M_nd ≤ 25.8627 | **PASS** | 389 and 15.3311 |
| **P8** | ω₁ ≥ 0.99 × baseline on every mesh | **PASS** | +0.304 %, +0.287 %, +2.194 % — all improvements; also passes against the like-for-like 400×50 value (`PROVENANCE.md` §A5) |
| **P9** | \|volume − 0.5\| ≤ 1e-4 on every mesh | **PASS** | 8.66e-7, 3.83e-7, 5.68e-7 |
| **P10** | multiplicity / physics acceptable | **PASS** | ω finite, ω₂ > ω₁ everywhere, subspace 2 throughout, no NaN/Inf, no eigensolver failure |
| **P11** | every transition attributable to frozen `A` or `B`, none to β | **PASS** | 9 transitions + 2 terminal admissions, each with its counter at exactly 20 |
| **P12** | terminal convergence only at `move_min` after frozen exhaustion | **PASS** | see §12 |
| **P13** | outer ≤ ×8 and wall ≤ ×10 on **every** mesh | **FAIL** | 320×40: **×12.21** outer, **×88.60** wall |
| **P14** | no mesh-specific tuning; one controller | **PASS** | identical `implTree` in all three run records |
| **P15** | evidence complete, declared, hash-valid | **FAIL — one raw artifact in flight** | see below |

**On P15.** This gate is reported as **FAIL**, not as a near-pass, because the
manifest says `nRawMissing = 1` and a gate is not satisfied by an artifact that is
expected to exist later. The `move_activity_400` evidence gate, which reported
`ok=0, required=2, missing=2` at resumption, now reports **PASS** after both artifacts
were recomputed and re-declared. `DATA_MANIFEST.json` and `FINAL_SHA256.txt` enumerate
every artifact with sizes and hashes. One declared raw trajectory —
`C320x40_trajectory.mat` — was lost with the rest of the untracked evidence root and
its regeneration was still running when this report was written; the manifest marks it
`present: false` rather than claiming otherwise. Its loss costs one figure panel
(F8's 320×40 candidate topology) and nothing else: C320's full 55-column
per-iteration telemetry is tracked and intact, its final density SHA-256
`0348b288d…` is recorded, and the build has been proved numerically inert, so the run
is reproducible on demand — which is precisely what the regeneration is doing. When it
lands, re-running `scripts/cv_manifest.py` and `scripts/cv_analyze.py` flips P15 to
PASS with no other change. **That would not alter the verdict**: P15 is not part of the
`PARTIALLY_VALIDATED` conjunction, and `VALIDATED` is already excluded by P13.

**Q37. What is the primary controller-validation verdict?**

> ### `TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED`

This is the verdict §12 defines for exactly this pattern: *"P1–P3 and P9–P12 pass, at
least one mesh meets its improvement gate, but at least one of P4–P8 or P13 fails."*
P1–P3 pass, P9–P12 pass, **both** fine meshes meet their improvement gates by wide
margins, and P13 fails at one mesh.

It is not `VALIDATED`, because `VALIDATED` requires all fifteen gates and P13 fails.
It is not `REJECTED`: there was no false termination, no unacknowledged failure, no
physics blocker, no transition unattributable to the frozen rule, and no scientific
regression — every mesh improved in both M_nd and ω₁. It is not `INCONCLUSIVE`: the
implementation and provenance are sound, the caps did not bind early enough to obscure
anything, and two of three runs terminated genuinely.

**What is validated, and what is not.** The controller's *causal claim* is confirmed at
all three meshes: replacing β-driven continuation with the frozen `A OR B` rule delays
the descent, and the delay is what produces the improvement — established not by
correlation but by intervention, with a bitwise-identical prefix proving nothing else
changed. What is **not** validated is the rule's ability to declare the *terminal*
stage exhausted: at 320×40 it cannot, and the run does not stop.

**Q38–Q40. Was production promoted?** **No** — `PRODUCTION_CONTROLLER_NOT_PROMOTED`.
§12 authorizes promotion only on `VALIDATED`. Consequently Phase 23 does not apply and
`CONTROLLER_PROMOTION_EQUIVALENCE_PASS` / `_FAIL` is **not issued**: no promotion was
performed, so there is nothing to prove equivalent. Details and the reasoning in
`PROMOTION.md`.

**Q41/Q42. Final production HEAD and tree state.** HEAD
`1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` on `benchmark-methodology-r2`. The
production `+impl` tree is unchanged and frozen at
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files), source
integrity **PASS**, currentness `CURRENT`. Production behaviour is unchanged and
proved so: with both switches at their defaults `test_preset_equivalence` reproduces
the frozen conference record bitwise. The working tree carries this study's new
documentation, evidence and figures, and the re-declared `move_activity_400/EVIDENCE.json`.

**Q43. Is all evidence durable and hash-valid?** Yes, with the one in-flight item
above disclosed rather than glossed. Raw trajectories live in the declared evidence
root and are gate-checked; all tracked artifacts are hashed in `FINAL_SHA256.txt`.

**Q44. Is any controller-specific scientific blocker unresolved?** **Yes — one.** The
frozen union is blind to the low-amplitude cancelling regime (`amp < tol` with
`med₂₀cos < 0`), which satisfies neither branch. It was preregistered as a known hole
and an accepted risk; at 320×40 it was realized and prevented termination. It is
identified, measured and reported, and — as this task requires — **not repaired here**.

**Q45. Is the nine-mesh performance campaign now authorized?** **No.**

> ### `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

Authorization requires `VALIDATED`; the verdict is `PARTIALLY_VALIDATED`, and the
blocker in Q44 remains open. The reasoning is set out in `PERFORMANCE_READINESS.md`;
in short, a controller whose termination depends on which dynamical regime a mesh
happens to enter would produce scaling curves that characterize the blind spot rather
than the method — and six of the nine campaign meshes are finer than anything tested
here.

---

## 15. Methodological distinction (Phase 25)

Stated explicitly because it is easy to overclaim and this study will not.

The 240×30 experiment was the **withheld mechanism validation**: 240×30 was held out
of rule construction, and the frozen union predicted its exhaustion event without
retuning.

The 160×20 / 320×40 / 400×50 runs reported here are **causal controller validation**.
The rule was developed using information from these three meshes, so they are **not**
new withheld validation of the maturity rule, and are nowhere described as such. They
test a different proposition — whether intervening on the optimizer with the
already-frozen rule causes the desired optimization behaviour.

The 400×50 result comes closest to a genuine prediction and is worth stating precisely
so it is not oversold: the declaration iteration 388 was preregistered *before the run*
and was met exactly, and the 20-iteration persistence that produced it was
**not** verifiable from the pre-existing fixed-move arm, which stops at 369. That is a
real out-of-sample confirmation of one preregistered quantity. It is not a withheld
validation of the rule itself.
---

## 16. Scope discipline (Q46–Q50)

**Q46. Does anything in this study justify projection?** **No.** Projection was off in
every run and nothing observed argues for it. The grayness improvements at both
completed meshes (M_nd −5.21 % and −44.68 %) were obtained with projection off, by
changing *when* continuation happens and nothing else. The one pathology found — the
low-amplitude cancelling limit cycle at 320×40's terminal stage — is a
continuation/termination problem, not a density-interpolation problem: the design
there is stationary to 0.4 % and the move limit is inactive, so a projection would
have nothing to bite on. `olhoffSolve` additionally refuses the controller under
projection.

**Q47. Does anything justify changing R = 0.06·b?** **No.** The filter radius was
never varied and no result depends on it. The improvements are attributable to
delayed continuation with a bitwise-identical stage-1 prefix, which pins the filter
as unchanged.

**Q48. Does anything justify changing p, the mass model, or q?** **No.** `p = 3` with
no continuation, mass Eq. (4b), `q = 1` were asserted in code immediately before each
solve and verified field-by-field against production at all three meshes. Nothing
observed bears on them; ω₁ moved in the *favourable* direction on both completed
meshes, so there is no physics deficit to attribute to the interpolation.

**Q49. Were any thresholds tuned after candidate results?** **No.** `A`, `B`, `W = 20`,
`P = 20`, `W_np = 10`, `tol(NE) = 0.05·sqrt(NE/3200)`, the union, the persistence
rule, the reset semantics, the move ladder `[0.04 0.02 0.01 0.005]` and the safety cap
1600 are exactly as frozen. The `+impl` tree hash
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` is identical at
resumption to the value recorded inside every run record, which is machine proof that
the controller source did not change between or after the runs. No Branch C was added;
no mesh-specific parameter and no NE exponent exists.

The one place where a temptation existed and was refused deserves naming: 320×40's
`CAP_HIT` is caused by a known hole in the union, its cost consequence is severe
(×12.21 outer), and the fix is obvious — the terminal stage is plainly converged and
almost any additional clause would have admitted it. The preregistration forbids
repairing the rule in this task (§13, Phase 19), so it was not repaired, and the run
is reported as a cap hit. The counterfactual arithmetic in §8.2 is labelled as such
and changed nothing.

**Q50. Were any additional scientific runs executed?** No additional *candidate
scientific* runs: exactly the three authorized meshes, 160×20, 320×40 and 400×50. No
240×30 candidate, no 480×60 / 560×70 / 640×80 / 720×90 / 800×100, and no nine-mesh
campaign.

Runs that were executed and are **not** candidate scientific runs, all disclosed:

| run | why | status |
|---|---|---|
| C160 rerun (160×20) | restore the lost trajectory; test the MATLAB build change | verification; bitwise identical to the run of record, which is retained unchanged |
| C320 rerun (320×40) | restore the lost trajectory | verification only; see §14, P15 |
| `ma4_run('P',400,50)` | restore a declared required artifact lost from `evidence/` | recomputation; bitwise identical to committed telemetry |
| `ma4_run('F',400,50)` | restore a declared required artifact; re-enable the §6 prefix check as a same-binary test | recomputation |
| `test_preset_equivalence` | repository test suite | 160×20 production solve, bitwise vs the frozen conference record |

None of these is a fixed-move mechanism arm rerun in the sense the brief prohibits:
`P400` and `F400` are restorations of *declared, hash-registered* artifacts that had
been lost from an untracked directory, reproduced bitwise, not new mechanism
experiments — and no mechanism conclusion was re-derived from them. No production
baseline value was replaced by a rerun: the frozen `evidence/baselines.json` scalars
remain the baseline of record, and the recomputation was used only to verify them and
to recover density fields.
