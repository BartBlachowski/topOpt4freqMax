# PREREGISTRATION — one 240×30 causal run to resolve the THRESHOLD_SPLITTING concern

**Exactly one** scientific optimization run is authorized and will be executed:
`C240x30`, the **unchanged** frozen four-rung `A OR B` stage-exhaustion
controller. Nothing else is run.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (re-read at task start, not inherited) |
| Tree at task start | dirty, **22** paths (four prior studies' uncommitted deliverables + two top-level gate files); **23** after this study's directory was created. None under `+impl/`. |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| MATLAB | `25.2.0.2998904 (R2025b)` |
| Threads | `maxNumCompThreads(1)`, asserted in the run driver |
| Prior status | `THREE_RUNG_COUNTERFACTUAL_EXACT` · `THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED` · `MORE_THREE_RUNG_EVIDENCE_REQUIRED` · `PRODUCTION_CONTROLLER_NOT_CHANGED` · `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` |
| Phase-0 gate | **`C240_PROVENANCE_PASS`** (13/13; the one test failure is this study's own unfinalized directory, allowed at tag `start` only) |

---

## 1. The question

> On the previously unavailable 240×30 **causal** mesh, does the final
> `move = 0.005` rung provide any scientifically material benefit beyond the
> exact three-rung endpoint `S3`?

This is the single missing experiment demanded by the `THRESHOLD_SPLITTING`
guard of `three_rung_architecture/PREREGISTRATION.md` §11 (SHA-256
`12c4bb960eeb6169521ea4b2e01b084a7bb3983d5387e94121e1729c2d71075c`).

## 2. The controller — recovered by CALL, not by copy

The run uses `cv_config('C', 240, 30)` from
`two_branch_controller_validation/scripts/cv_config.m` — **the same function
that produced the 160×20, 320×40 and 400×50 causal evidence**. Telemetry uses
`cv_telemetry.m` and `cv_export.m` from the same study, also by call. None of
those files is modified.

`cv_run.m` refuses any mesh outside its three authorized ones by design. It is a
hashed artifact of a sealed study and **will not be edited**. The new driver
`scripts/r240_run.m` reproduces its call sequence line for line and authorizes
**only** 240×30.

Frozen source digests, verified at task start:

```
+impl/architecture/+olh/+move/exhaustion.m  17b37a384b1aa5d987d9c861e16071d1140af130f92406ccc11cec4518bcae0c
+impl/architecture/+olh/+move/limit.m       61fa923d430121ead764a229b4f48dfd77617540b4ab6db883ccd9d5904b1c1e
+impl/architecture/olhoffSolve.m            1e5a114cbf91717e01e5592e86203f401ed5cdb3e5f5ce9a727b8163cdf2fba3
```

The rule, restated for the record only (authoritative form is `exhaustion.m`):

```
tol(NE) = 0.05*sqrt(NE/3200)          W = 20, P = 20, W_np = 10
amp(k)  = ||drho_k||_2                cos, net as in exhaustion.m
A(k) = med20 cos < 0  AND  med20 net < 0.5  AND  amp >= tol
B(k) = amp < tol      AND  med20 cos > 0
E    = A OR B, DECLARED at the first k where either has held 20 consecutive
RESET on descent: stageStart <- new stage's first iteration, counters zeroed,
                  windows stage-local (only the net-path anchor is pre-stage)
TERMINAL: convOuter = ex.declared && (stage >= numel(move.levels))
```

**Nothing is changed**: not `A`, not `B`, not `E`, not `W`, not `P`, not `W_np`,
not `tol`, not the persistence, not the windows, not the reset semantics, not
the move levels, not the terminal admission. No Branch C.

**Move ladder: `[0.04, 0.02, 0.01, 0.005]`** — the full four rungs, so that both
`S3` and `F` come from the same single trajectory.

## 3. Mesh, configuration and the single-factor lock

`domain.mesh = 240 × 30`, `NE = 7200`, `stop.tolerance = 0.075`
(`= 0.05·√(7200/3200)`, the inherited `meshScaled` law — a deterministic
function of `NE`, not a free choice).

Resolved config hash: `33833323efa08facaa5849c24fe32d6c9c47b5924f88c00d34fb65f7140d54d6`.

**Single-factor audit already run and recorded** (`evidence/single_factor.json`).
Field-for-field against each prior causal arm, exactly five fields differ:

| field | status |
|---|---|
| `domain.mesh.nelx`, `domain.mesh.nely` | **the authorized experimental factor** |
| `stop.tolerance` | deterministic function of `NE` under the inherited law |
| `runtime.name` | a label; enters no computation |
| `provenance.overrides` | the recorded override list; **verified element-wise** to differ in exactly the `nelx`, `nely` and `runtime.name` entries and nothing else |

`singleFactorOk = 1`. The scientific lock is asserted in code before the solve:
`p = 3` fixed, no `p` continuation, mass `eq4b`, `q = 1`, sensitivity filter
applied to all, `R = 0.06·b`, projection **off**, subspace multiplicity size 2
with diagonal offsets and off-diagonal terms **on**, published MMA,
`move.policy = 'ladder'`, `move.levels = [0.04 0.02 0.01 0.005]`,
`stop.rule = 'stageExhaustion'`, `move.continuation.signal = 'stageExhaustion'`.
Any mismatch aborts before the solve.

## 4. Safety cap — 1600, inherited, fixed now

`CAP = 1600`, taken from the `CAP` constant inside `cv_config.m`. It is **the
same cap every prior causal arm used** (160×20, 320×40, 400×50), so it is
inherited rather than chosen for this run, and all four arms remain directly
comparable in extent.

It is sufficient for the purpose Phase 8 requires. From the prior **fixed-move**
240×30 arm (§10) the stage-1 event is expected near iteration 206; each lower
stage that terminates costs exactly 39 outer iterations, so `move = 0.005` is
expected to begin near iteration 285, leaving ≈ 1315 iterations to observe
either a genuine terminal declaration or the same persistent low-amplitude
cancellation pathology seen at 320×40.

**The cap will not be extended after seeing the trajectory.** If it is reached,
the result is reported as **`CAP_HIT`** and is never relabelled convergence.

## 5. Index convention — inherited verbatim, one convention throughout

* **`kE(s)`** — the event index of stage `s` — is the outer iteration at which
  the detector **declares**: `nA` or `nB` first reaches 20 within that stage,
  equivalently the first iteration with `hist.exDecl = 1` after the stage began.
  The sustained window is `[kE − 19, kE]`.
* **The descent** is applied at `kE(s) + 1`.
* **The terminal state of a policy whose last level is stage `s`** is the row at
  `outer = kE(s)` — the design after that iteration's update, which is exactly
  what `olhoffSolve` returns.
* Hence `S1 = row(kE(1))`, `S2 = row(kE(2))`, `S3 = row(kE(3))`, `F = last row`.

Cost at a state: `cumInner` at that row; outer = the row index.

**Off-by-one, stated explicitly.** The earlier `two_branch_maturity_240` study
reported its events as the iteration at which the sustained window *begins*
(= declaration − 19). This study uses the **declaration** index everywhere. Where
that study's numbers are quoted, they are converted and the conversion is shown.

## 6. Materiality bars — INHERITED VERBATIM, frozen, not mesh-specific

Taken **unchanged** from `move_ladder_necessity/PREREGISTRATION.md` §6 (SHA-256
`a08b879b…`) by way of `two_rung_architecture` §9 (`b50455fb…`) and
`three_rung_architecture` §9 (`12c4bb96…`). **No bar is loosened, tightened,
replaced, padded, or made mesh-specific.**

| quantity | material if | 
|---|---|
| `ω₁` | the block improves `ω₁` by **≥ 0.10 % relative** |
| `M_nd` | the block improves `M_nd` by **≥ 2 % relative** |
| topology | gray or mid fraction changes by **≥ 0.01 absolute**, or mean \|Δρ_e\| **≥ 0.01** |
| volume feasibility | \|volume − 0.5\| worsens by **≥ 1e-5** |
| multiplicity / physics | subspace size leaves 2, mode order changes, ω₂ ≤ ω₁, or a NaN/Inf appears. **Gap magnitude alone is not material**, because the objective is ω₁ |
| cost domination | a rung block costs **≥ 2×** the outer iterations used to reach the state it starts from, while sub-material on every metric above |
| failure risk | any rung sequence producing `CAP_HIT` or a non-terminating stage |

**Denominator convention, fixed now.** For a block `a → b` the relative change is
`100·(b − a)/a`, normalised by the **earlier** state. The decisive quantity is

```
rung4_omega1_pct = 100 * ( omega1(F) - omega1(S3) ) / omega1(S3)
```

compared against **0.10 %**. A value just above 0.10 % is **MATERIAL**; a value
just below is **IMMATERIAL**. **No uncertainty padding will be added after the
number is seen** — no such rule is preregistered, and none will be invented.

Production-relative acceptance at `S3` is additionally checked using the
inherited controller gates (`two_branch_controller_validation` §11): `ω₁ ≥ 0.99 ×`
production, `|volume − 0.5| ≤ 1e-4`, subspace size 2 with `ω₂ > ω₁` throughout,
all `ω` finite, no non-converged inner solve. **No 240×30 production baseline
exists**, so the production-relative `ω₁` and `M_nd` gates are reported as
`UNAVAILABLE` rather than imputed (§11).

## 7. THRESHOLD_SPLITTING resolution rule — frozen before the run

Transcribed from the brief's Phase 6 and binding:

* **RESOLVED IN FAVOUR of three-rung support** iff the new 240×30 evidence shows
  `S3 → F` is **below ALL** frozen materiality bars of §6; **or** rung 4 becomes
  pathological / non-terminating while `S3` is already a valid, scientifically
  acceptable exhausted state **and** no material pre-pathology benefit is
  obtained.
* **RESOLVED AGAINST three-rung support** iff rung 4 produces a benefit **above
  any** frozen scientific bar of §6.
* **UNRESOLVED** iff `C240` cannot produce a defensible `S3`/`F` comparison —
  the counterfactual is not exact, `E` never fires on `move = 0.01`, the run
  fails mechanically, or required evidence is unusable.

**This mapping will not be reinterpreted after the result is seen.**

### The "pre-pathology benefit" test, defined now (brief Phase 19)

If rung 4 `CAP_HIT`s, `CAP_HIT` alone does **not** count as evidence for the
three-rung architecture. All of the following are required first:

1. `S3` is a valid persistent-`E` exhausted state (declared, 20-iteration
   persistence, at `move = 0.01`);
2. the **running best** over the whole rung-4 tail is examined, not just the
   final row: `max_j ω₁(j)` and `min_j M_nd(j)` for `j > kE(3)`. If the best
   value reached anywhere in the tail beats `S3` by a §6-material amount, rung 4
   **did** produce material gain and the concern is resolved AGAINST;
3. the tail is shown to churn/cancel without benefit;
4. the failure mechanism is compared with 320×40's (low-amplitude cancellation:
   `amp < tol` together with `med cos < 0`, satisfying neither branch);
5. no §6-material quantity is still trending upward at the cap — assessed by
   comparing the last 100 tail iterations against the preceding 100.

## 8. Counterfactual validity (brief Phase 11) — re-audited, not inherited

The pre-run **static** audit is already recorded (`evidence/single_factor.json`),
under the resolved 240×30 configuration rather than by inheritance:

| site | expression | active? |
|---|---|---|
| `olhoffSolve.m:485` | `~any(moveLevels(stage+1:end) > epsRMS)` — the **only** site depending on the ladder **tail / final-rung identity** | **never runs**: `stop.guards.ladderExhausted = false` and `stop.guards.maxDesignChange = false` ⇒ `anyStopGuard = false`, **and** `exhaustStop = true` |
| `olhoffSolve.m:312` | `mvNow = moveLevels(1)` | **never runs**: `material.stiffness.continuation.enabled = false` ⇒ `pOwnCounter = false` |
| projection guard block | | **never runs**: `projection.enabled = false` |
| `olhoffSolve.m:509` | `atLastLevel = stage >= numel(levels)` | active; **length only**, consumed with `ex.declared` |
| `limit.m:109` | `stage < numel(levels)` | active; **length only**, consumed with `ex.declared` |
| `limit.m:122` | `mv = levels(stage)` | active; value lookup, identical for stages 1–3 |

`move.levels = [0.04 0.02 0.01]` additionally resolves to a **legal**
configuration at 240×30, differing from the four-rung resolution in `move.levels`
alone (plus provenance metadata).

**After the run** this is re-verified against the realized trajectory: the ten
per-mesh checks of `three_rung_architecture` §7, plus an element-wise replay of
the frozen rule from the raw `RHO` / `hist.dxNorm2` against the recorded
`exA`/`exB`/`exE`/`exNA`/`exNB` trace. Verdict
`C240_THREE_RUNG_COUNTERFACTUAL_EXACT` or `..._NOT_EXACT`.

## 9. Telemetry and evidence retention

Full per-iteration telemetry via `cv_telemetry` (the same 55-column table as the
prior causal arms) and the **full raw density trajectory** `RHO` (7200 × nOuter)
saved to the durable evidence root
`analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat`,
declared in `EVIDENCE.json` by measured SHA-256, and covered by a self-verifying
`FINAL_SHA256.txt`. The fail-closed finalization gate must return **PASS** on
G1–G5; no final PASS is permitted if any required artifact is absent,
unmanifested, stale-digest or hash-invalid.

## 10. Disclosure — what was already visible when this file was frozen

**Published rung-4 results at the other three meshes** (`three_rung_architecture`):
`S3 → F` relative `ω₁` = **+0.02025 % (160×20)**, **−0.00504 % (320×40)**,
**+0.00815 % (400×50)**; all below the 0.10 % bar. `M_nd` residuals ≤ 0.41 %;
mean \|Δρ\| ≤ 0.0017; multiplicity, gap and volume immaterial everywhere. 320×40
ended `CAP_HIT @1600` with rung 4 consuming 91.5 % of inner work. The
`THRESHOLD_SPLITTING` split at 160×20 (rung 3 +0.09367 %, rung 4 +0.02025 %,
combined +0.11393 %).

**The prior 240×30 FIXED-MOVE arm** (`two_branch_maturity_240`, `runD`) — opened
deliberately *before* freezing so that this disclosure is accurate and so that a
sharp prediction can be made. That arm held `move = 0.04` for 1200 iterations and
ended `CAP_HIT`; its raw trajectory is lost but its `METRICS.json` survives:

* frozen-rule event at `move = 0.04`: **Branch B**, window begins **187** ⇒ under
  this study's convention a **declaration at 206**;
* at that event: `M_nd` 12.8388, `ω₁` 167.0571, `‖Δρ‖₂` 0.05067 against
  `tol = 0.075` (**ratio 0.676**), `max|Δρ|/move` 0.4232, bound fraction **0**;
* native stop first held at 147; β-stall first fired at 92.

**What this implies, stated before the run.** Under `move = 0.04` the controller
is identical to a fixed-move arm until its first descent, so `S1` is
*predictable*. Moreover 240×30 exits stage 1 on **Branch B** with `‖Δρ‖₂/tol =
0.68` and zero bound-active elements — placing it with the **fine** meshes
(400×50: Branch B, ratio 0.64) rather than with 160×20 (Branch A, ratio 12.59) or
320×40 (Branch A, ratio 1.03). Since rung 4 was immaterial at 400×50, **the
direction of this audit's answer is foreseeable, and I say so plainly rather than
claim a blind test.**

**What is NOT foreseeable and is genuinely new.** A fixed-move arm never
descends, so no 240×30 evidence of any kind exists below `move = 0.04`. `S2`,
`S3`, `F`, the rung-3 and rung-4 increments, whether stage 4 terminates or
`CAP_HIT`s, and every quantity that decides this audit are unmeasured. That is
precisely the gap this run fills.

## 11. Missing-data and failure rules

* **No 240×30 production baseline exists.** Production-relative `ω₁` and `M_nd`
  gates are reported `UNAVAILABLE` with the reason. They are **never** imputed
  from another mesh, interpolated, or replaced by a proxy. Absolute and
  `S3`-relative quantities are unaffected.
* Wall time is reported descriptively only; prior studies measured 4–6× drift in
  seconds-per-inner-iteration within a run. **Primary cost evidence is outer
  iterations and inner MMA work.** Timing is never fabricated.
* If the run fails through a clearly **mechanical software** fault (exception,
  I/O failure, assertion in the scope lock), the failed artifact is preserved,
  the rerun requirement is disclosed in `REPORT.md` before rerunning, and the
  fault is named. **An undesirable scientific outcome is never classified as a
  software bug.**
* If the counterfactual is `NOT_EXACT`, the architectural comparison is not
  treated as exact and the verdict is `THRESHOLD_SPLITTING_CONCERN_UNRESOLVED`.

## 12. Preregistered prediction — stated to make the test sharp

Falsifiable, and recorded so post-hoc rationalisation is visible:

* **`S1` declares at iteration 206 ± 10, Branch B** (from §10; a mismatch is not
  by itself a failure — the fixed-move arm ran under a different `+impl` tree and
  MATLAB build — but any mismatch must be explained, not absorbed);
* `S2` ≈ 245, `S3` ≈ 284, each at `stageStart + 38` if the lower-stage timing
  pattern holds;
* rung 4 `ω₁` benefit **below** 0.10 %, i.e. immaterial, as at all three prior
  meshes;
* stage 4 terminal status genuinely uncertain: either `CONVERGED` (as at 160×20
  and 400×50) or `CAP_HIT` (as at 320×40). **No prediction is made.**

A confirmed hypothesis with a wrong prediction is still a pass; the prediction
exists to expose rationalisation, not to be defended.

## 13. Verdict mapping — fixed now

**Counterfactual:** `C240_THREE_RUNG_COUNTERFACTUAL_EXACT` iff all ten post-run
validity checks pass and the frozen-rule replay matches the recorded trace
element-wise in every stage; otherwise `..._NOT_EXACT`.

**Threshold splitting:** exactly one of
`THRESHOLD_SPLITTING_CONCERN_RESOLVED` / `..._CONFIRMED` / `..._UNRESOLVED`,
by the rule in §7.

**Architecture:** `THREE_RUNG_ARCHITECTURE_SUPPORTED` requires all eleven of the
brief's Phase-23 conditions, including: the run is valid; `S3` is exact; `S3` is
a valid persistent-`E` exhausted state; rung 4 below all §6 bars at 240×30 **or**
pathological without prior material benefit; no material multiplicity benefit;
volume feasibility acceptable; no threshold changed; `A`/`B` unchanged; the
160/320/400 rung-4 evidence still valid; the splitting concern resolved; and
retention/finalization passing. Otherwise `PARTIALLY_SUPPORTED`, `REFUTED` or
`INCONCLUSIVE` as warranted.

**If any §6 bar is exceeded by `S3 → F`, the verdict is
`THREE_RUNG_ARCHITECTURE_NOT_SUPPORTED_BY_240` and it will be reported as such.**
No threshold will be changed, no extra condition added, no claim that 240×30 is
an anomaly, no mesh law fitted, no rescue attempted.

**Next step:** `THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED` only if the
splitting concern is genuinely resolved and no other scientific blocker remains;
otherwise `MORE_THREE_RUNG_EVIDENCE_REQUIRED` or
`MOVE_LADDER_REDESIGN_STILL_REQUIRED`.

## 14. Cross-mesh reporting (brief Phase 21)

After the C240 evidence is frozen, rung-4 effects are tabulated across
160 / 240 / 320 / 400 for relative `ω₁`, `M_nd`, topology distance, multiplicity,
outer cost, inner cost and terminal status. **No scaling law is fitted from four
points.** Any trend is reported descriptively only.

## 15. What this task will not do

No second scientific optimization run of any kind. No rerun of 160×20, 320×40 or
400×50. No modification of `A`, `B`, `E`, the persistence, the windows, the
thresholds, the move levels or the terminal admission. No Branch C. No
alternative ladder. No architecture mining. No promotion or implementation of any
production policy. No projection; no change to `R`, `p`, mass interpolation, `q`,
multiplicity, MMA, FE formulation, eigensolver, objective or volume constraint.
No regeneration of lost evidence.

Production remains `PRODUCTION_CONTROLLER_NOT_CHANGED`; the nine-mesh campaign
remains `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` in this task **even if the
three-rung architecture is fully supported**.
