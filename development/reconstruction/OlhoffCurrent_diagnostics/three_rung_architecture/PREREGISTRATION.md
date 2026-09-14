# PREREGISTRATION — does the exact three-rung ladder `[0.04, 0.02, 0.01]` suffice?

A **zero-scientific-run** offline audit. No optimization of any kind is executed.
Every number comes from causal-controller trajectories that already exist on disk.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (re-read at task start, not inherited) |
| Tree at task start | dirty, 22 paths (three prior studies' uncommitted deliverables + this study's directory) |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| MATLAB available now | `25.2.0.2998904 (R2025b)` — **not** the `25.2.0.3042426 (R2025b) Update 1` build that produced the C-arm trajectories. Used here only for hashing, config resolution and gates; **no solve is executed**, so the build difference cannot affect any scientific number. |
| Analysis language | Python 3 (offline replay); MATLAB for provenance, static config audit and gates |
| Prior status | `TWO_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED` · `MORE_TWO_RUNG_EVIDENCE_REQUIRED` · `PRODUCTION_CONTROLLER_NOT_CHANGED` · `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` |

---

## 1. The question

> Does the exact three-rung policy `move levels = [0.04, 0.02, 0.01]`, governed
> end to end by the **already-frozen** exhaustion rule `E = A OR B`, capture the
> scientifically material value that blocked the two-rung policy at 160×20,
> while eliminating the cost and the non-termination risk of the final
> `move = 0.005` rung?

Exactly one architecture is under test. Nothing is tuned, invented, or varied.

## 2. The counterfactual policy, stated exactly

```
THREE-RUNG POLICY (S3-policy)
    cfg.move.levels              = [0.04, 0.02, 0.01]     <- the ONLY change
    cfg.move.continuation.signal = 'stageExhaustion'          (unchanged)
    cfg.stop.rule                = 'stageExhaustion'          (unchanged)

    STAGE 1     move = 0.04, held until the frozen detector DECLARES E = A OR B
    DESCENT 1   0.04 -> 0.02, detector reset to the new stage
    STAGE 2     move = 0.02, held until the frozen detector DECLARES E
    DESCENT 2   0.02 -> 0.01, detector reset to the new stage
    STAGE 3     move = 0.01, held until the frozen detector DECLARES E
    TERMINATE   at that declaration (stage 3 is now the last level)

    move = 0.005 is NEVER entered.
```

compared against four reference states:

```
P   production baseline           beta-stall ladder, the frozen baselines
S1  single-stage endpoint         first frozen declaration at move = 0.04
S2  two-rung endpoint             first frozen declaration at move = 0.02
F   four-rung candidate endpoint  the controller validated in
                                  two_branch_controller_validation
```

## 3. Why no simulation is needed — the prefix argument, re-derived for THREE rungs

**The two-rung proof is not assumed to extend.** It is re-derived, and a
*stronger* fact is established by static audit of the resolved configuration
(`scripts/tr3_configaudit.m`, `evidence/config_audit.json`).

Every site in `+impl/` that reads `move.levels` on any code path:

| # | site | expression | active? | depends on ladder **length**? | depends on **final-rung identity**? |
|---|---|---|---|---|---|
| 1 | `olhoffSolve.m:104` | `moveLevels = g('move.levels')` | yes | no | no |
| 2 | `olhoffSolve.m:312` | `mvNow = moveLevels(1)` | **no** — inside `if pOwnCounter`, and `material.stiffness.continuation.enabled = false` | no | no |
| 3 | `olhoffSolve.m:485` | `~any(moveLevels(stage+1:end) > epsRMS)` | **no** — inside `if anyStopGuard && ~exhaustStop`, and **`anyStopGuard = false`** (`stop.guards.ladderExhausted = false`, `stop.guards.maxDesignChange = false`) *and* `exhaustStop = true`; the block is skipped for two independent reasons | yes | **yes** |
| 4 | `olhoffSolve.m:509` | `atLastLevel = stage >= numel(moveLevels)` | yes | **yes** | no |
| 5 | `limit.m:109` | `state.stage < numel(cfg.move.levels)` | yes | **yes** | no |
| 6 | `limit.m:122` | `mv = cfg.move.levels(state.stage)` | yes | no | no |
| 7 | `limit.m:148,152` | β-stall descent branch | **no** — `move.continuation.signal = 'stageExhaustion'` returns before it | yes | no |

Site 3 is the **only** place the identity of the final rung could ever enter the
computation, and it is doubly inert in this configuration. Sites 4 and 5 depend
on the ladder's *length* only, and both are consumed exclusively in conjunction
with `ex.declared`, which is false at every iteration strictly between two
declarations. Site 6 is a lookup by stage index, identical for stages 1–3 in
both ladders. `state.mv` initialises to `cfg.move.initial`, a config key
independent of `move.levels`, and is overwritten by site 6 on the first call.

Therefore, with `levels = [0.04, 0.02, 0.01]` versus
`levels = [0.04, 0.02, 0.01, 0.005]`, the two runs are **bitwise identical for
every outer iteration up to and including the stage-3 declaration `kE(3)`**, and
first differ at `kE(3)` itself: the four-rung run descends to `0.005`, the
three-rung run sets `convOuter = true` and stops.

**S3 is therefore the recorded state of the existing four-rung candidate
trajectory at its stage-3 declaration, exactly.** The argument is *verified* in
§7; if any of the ten checks fails, the verdict is
`THREE_RUNG_COUNTERFACTUAL_NOT_EXACT` and the scientific interpretation stops.

The three-rung ladder is additionally confirmed to be a **legal** configuration:
`schema.m` allows `move.levels` of length `[1 Inf]`; `validate.m` requires it to
be non-increasing and requires the paired `move.policy = 'ladder'` with matching
`stop.rule` / `move.continuation.signal` — all satisfied. Resolving
`move.levels = [0.04 0.02 0.01]` against the four-rung resolution differs in
exactly two fields: `move.levels` itself, and the recorded override list in
`provenance` (metadata, not science).

## 4. Sources — the exact trajectories used

| mesh | role | trajectory | telemetry |
|---|---|---|---|
| 160×20 | primary | `evidence/two_branch_controller_validation/C160x20_trajectory.mat` | `runs/C160x20_iterations.csv` |
| 320×40 | primary | `.../C320x40_trajectory.mat` | `runs/C320x40_iterations.csv` |
| 400×50 | primary | `.../C400x50_trajectory.mat` | `runs/C400x50_iterations.csv` |
| 240×30 | **unavailable** | none exists | none |

Production baselines: `two_branch_controller_validation/evidence/baselines.json`.

**240×30, declared now.** `analysis/OlhoffCurrent` contains **zero** files
matching `*240x30*`. The only 240×30 study has no `runs/` directory, and its arm
was a **fixed-move** arm (`move.policy = 'fixed'`, `move.initial = 0.04`) that
never executed a `move = 0.02` stage, let alone a `move = 0.01` stage. It is
marked `UNAVAILABLE_FOR_THREE_RUNG_CAUSAL_COMPARISON`. No S3 will be inferred or
invented for it, and it will not be run.

**Production density fields, declared now.** `baselines.json` records
`rho_available = false` for 160×20 and 320×40 (raw `.mat` lost). Density-field
distance to P is computable only at 400×50; elsewhere it is reported as a gap.

## 5. The frozen rule — recovered, not re-derived

Authoritative executable definition:
`analysis/OlhoffCurrent/+impl/architecture/+olh/+move/exhaustion.m`, whose
provenance line names `two_branch_maturity_240/PREREGISTRATION.md` SHA-256
`62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` (verified live
at task start). Restated for the record only:

```
tol(NE) = 0.05*sqrt(NE/3200)          W = 20, P = 20, W_np = 10
d_k     = rho_k - rho_{k-1}           amp(k) = ||drho_k||_2   ( = hist.dxNorm2 )
cos(k)  = <d_k, d_{k-1}> / (||d_k||*||d_{k-1}||)
net(k)  = n2(rho_k - rho_{k-10}) / sum_{j=k-9..k} n2(d_j),   n2 = ||.||/sqrt(NE)
med20 x = trailing 20-median, omitnan, defined only once the whole window lies
          inside the current stage

A(k) = med20 cos < 0  AND  med20 net < 0.5  AND  amp >= tol
B(k) = amp < tol      AND  med20 cos > 0
E(k) = A(k) OR B(k)

DECLARATION = first k at which either branch has held for P = 20 consecutive
              iterations.  Window begins at k - 19.

RESET SEMANTICS: on descent, ex.stageStart <- the first iteration at the new
level, cntA = cntB = 0, declared = false.  Every step entering a predicate is
stage-local; only the net-path ANCHOR rho_{s-1} is pre-stage.
```

**Nothing in this rule is modified, extended, relaxed, or supplemented in this
task.** No Branch C. No terminal exception. No change to `W`, `P`, `W_np`, `tol`,
the union, the persistence, the windows, the reset semantics, or the move values.

**Verification requirement.** An independent Python re-implementation
(`scripts/tr3_frozen.py`, code-identical to `two_rung_architecture/scripts/tr_frozen.py`)
must reproduce, **element-wise over every stage of every mesh**, the
`exA`/`exB`/`exE`/`exNA`/`exNB` trace the solver actually acted on, and must
reproduce each recorded declaration exactly, **including S1 and S2**. Any
disagreement ⇒ `THREE_RUNG_ARCHITECTURE_INCONCLUSIVE`.

## 6. Index convention — fixed now, used everywhere, identical to the two-rung audit

* **`kE(s)` — the event index of stage `s`** is the outer iteration at which the
  frozen detector *declares*: the iteration at which `nA` or `nB` first reaches
  20 within that stage, equivalently the first iteration with `hist.exDecl = 1`
  after the stage began. The sustained window is `[kE − 19, kE]`.
* **The terminal state of a policy whose last level is stage `s`** is the
  recorded row at `outer = kE(s)` — the design `ρ_{kE}` *after* that iteration's
  update. This is exactly what `olhoffSolve` returns: `convOuter` is set inside
  the same iteration in which the declaration is made.
* **The descent index** is `kE(s) + 1`.
* Hence `S1 = row(kE(1))`, `S2 = row(kE(2))`, `S3 = row(kE(3))`, `F = last row`.

Cost at a state: `cumInner` at that row; cumulative wall = `sum(tOuter)` through
that row; outer = the row index. **No other convention is used anywhere.**

## 7. Counterfactual-validity checks (Phase 3) — all ten required, per mesh

1. same initialization (`design.initial` = 0.5, the uniform start);
2. `move = 0.04` and `stage = 1` for every `k ≤ kE(1)`;
3. exactly **two** descents in `[1, kE(3)]`, at `kE(1)+1` and `kE(2)+1`;
4. the first descent is to `move = 0.02`;
5. `move = 0.02` and `stage = 2` for every `k ∈ [kE(1)+1, kE(2)]`;
6. the second descent is to `move = 0.01`;
7. `move = 0.01` and `stage = 3` for every `k ∈ [kE(2)+1, kE(3)]`;
8. no `move ≤ 0.005` iteration occurs at or before `kE(3)`;
9. `hist.exStageStart` equals `kE(1)+1` throughout stage 2 and `kE(2)+1`
   throughout stage 3 — the recorded reset semantics are exactly the three-rung
   policy's;
10. the architectural divergence occurs only at `kE(3)`: the state there is at
    `stage = 3`, `move = 0.01`, and the next recorded iteration (if any) is at
    `move = 0.005`.

Plus the static configuration audit of §3, which must show sites 2, 3 and 7 inert.

Verdict: `THREE_RUNG_COUNTERFACTUAL_EXACT` iff all ten hold on all three meshes
and the static audit shows no active dependence on final-rung identity;
otherwise `THREE_RUNG_COUNTERFACTUAL_NOT_EXACT`.

## 8. Extraction (Phase 5) — the S3 record

At `S1`, `S2`, `S3` and `F`, per mesh, record: outer, stage, stage-start,
offset from stage start, move, triggering branch, `A`/`B`/`E`, `nA`/`nB`,
`exStageStart`, `ω₁`, `ω₂`, `gap12`, volume, `volErr`, `M_nd`, gray fraction,
mid-density fraction, `max|Δρ|`, `max|Δρ|/move`, `‖Δρ‖₂`, RMS `Δρ`, `‖Δρ‖₂/tol`,
`cosθ`, net/path, bound fraction, β-stall state, native-stop state, `nInner`,
`cumInner`, cumulative wall, subspace size, degeneracy count, and the SHA-256 of
the density vector.

Density-field differences are reported as mean `|Δρ_e|`, `‖Δρ‖₂/√NE`, and
`max|Δρ_e|`.

## 9. Materiality thresholds — INHERITED VERBATIM, fixed before extraction

Taken **unchanged** from `move_ladder_necessity/PREREGISTRATION.md` §6 (SHA-256
`a08b879b9d4893bc55dd9e5dfc2c26f897cb9d49355f0557ccc8644c7fc6ca3e`) by way of
`two_rung_architecture/PREREGISTRATION.md` §9 (SHA-256
`b50455fbd3d97dcb72093fc13738f236d4d0f2e37a47caba0f515a16aaf7ca04`). **No
threshold is re-derived, re-anchored, relaxed or replaced here.**

| quantity | material if | anchor |
|---|---|---|
| `ω₁` | the rung block improves `ω₁` by **≥ 0.10 % relative** | one tenth of the controller study's 1 % acceptable-degradation bound |
| `M_nd` | the rung block improves `M_nd` by **≥ 2 % relative** | one tenth of the controller study's 20 % "materially better" bar |
| topology | gray or mid fraction changes by **≥ 0.01 absolute**, or mean \|Δρ_e\| **≥ 0.01** | one tenth of the smallest change the project has called meaningful |
| volume feasibility | \|volume − 0.5\| worsens by **≥ 1e-5** | both arms achieve ≈ 1e-6–1e-7 |
| multiplicity / physics | subspace size leaves 2, mode order changes, ω₂ ≤ ω₁, or a NaN/Inf appears | qualitative; **gap magnitude alone is not material**, because the objective is ω₁ |
| cost domination | a rung block costs **≥ 2×** the outer iterations used to reach the state it starts from, while delivering sub-material benefit on every metric above | the ladder must earn its cost |
| failure risk | any rung sequence producing `CAP_HIT` or a non-terminating stage | a controller that cannot stop is a defect regardless of quality |

**Relative-change convention, fixed now and not varied.** For a block `a → b`,
the relative change is `100·(b − a)/a`, i.e. **normalised by the earlier state**
— identical to `move_ladder_necessity` and `two_rung_architecture`. In
particular the decisive quantity is

```
residual_rung4_omega1_pct  =  100 * ( omega1(F) - omega1(S3) ) / omega1(S3)
```

and it is compared against **0.10 %**. Because each block uses its own
denominator, the per-rung percentages do **not** sum exactly to the S1→F
percentage; that is a property of the inherited convention, is stated here in
advance, and is not used to move any number across a bar.

## 10. Production-relative acceptance gates for S3 — INHERITED VERBATIM

`S3` is a candidate *terminal state*, so it must clear the same gates the
four-rung controller was required to clear. Taken unchanged from
`two_branch_controller_validation/PREREGISTRATION.md` §11 (SHA-256
`8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf`), applied at
S3 exactly as `two_rung_architecture` applied them at S2:

| gate | requirement at S3 |
|---|---|
| **A1** (= P5) | 160×20: `M_nd ≤ 1.10 ×` production 13.4025 (≤ 14.7427) **and** `ω₁ ≥ 0.99 ×` production 169.4952 (≥ 167.8003) |
| **A2** (= P6) | 320×40: `M_nd ≤ 0.80 ×` production 23.3596 (≤ 18.6877) |
| **A3** (= P7) | 400×50: `M_nd ≤ 0.80 ×` production 32.3283 (≤ 25.8627) |
| **A4** (= P8) | every mesh: `ω₁ ≥ 0.99 ×` its production `ω₁` |
| **A5** (= P9) | every mesh: `|volume − 0.5| ≤ 1e-4` |
| **A6** (= P10) | every mesh, over the three-rung prefix `[1, kE(3)]`: subspace size 2 throughout, `ω₂ > ω₁`, all `ω` finite, no NaN/Inf, no non-converged inner solve. (`degen`, which counts *expected* near-degeneracy hits in the multiplicity-aware subspace, is **not** part of this gate and is reported descriptively only.) |
| **A7** (= P13) | every mesh: outer multiplier ≤ 8× and wall multiplier ≤ 10× the production baseline |
| **A8** (= P12) | termination occurs only at the last level after frozen persistence |
| **A9** | 160×20 no-regression: `ω₁(S3) ≥ ω₁(P) = 169.49522702153845` |

## 11. Threshold-splitting guard — fixed now, can only downgrade

The two-rung policy failed because the **combined** `S2 → F` `ω₁` gain at 160×20
(+0.1139 %) exceeded the 0.10 % bar. If the three-rung policy passes merely
because that combined gain has been *split* into two individually sub-material
halves — rung 3 below 0.10 % **and** rung 4 below 0.10 % — then no retained rung
is doing material work and the architecture's success would be an artifact of
where the bar falls relative to an arbitrary split point.

That case is recorded as `THRESHOLD_SPLITTING = true` and **caps the verdict at
`PARTIALLY_SUPPORTED`**. Clean support requires rung 3 (`S2 → S3`) to be itself
material at 160×20 by at least one §9 criterion — the same demand
`two_rung_architecture` §12 placed on rung 2. This guard can only lower a
verdict, never raise one.

## 12. Cost criteria — no new numeric bar is invented

**Disclosed: the per-rung cost figures are already known to this audit.** They
are published in `move_ladder_necessity/METRICS.json` and were printed during the
two-rung task: rung 4 consumes 791/5074 (160×20), 70034/76532 (320×40) and
1453/10301 (400×50) inner MMA iterations. Choosing a fresh numeric savings
threshold now would be threshold-shopping against numbers already seen.

Therefore **no new cost threshold is defined**. Cost is handled as follows:

* savings from omitting rung 4 (outer, inner MMA, and wall) are **reported as
  measured**, per mesh;
* the only binary cost bars are the **inherited** ones in §9: cost domination
  (≥ 2× outer of the state the block starts from, while sub-material on every
  science metric) and failure risk (`CAP_HIT` / non-terminating stage);
* the brief's Phase-21 condition 11 ("omitting rung 4 saves material outer/inner
  work") is evaluated as: **rung 4 is cost-dominated or carries failure risk on
  at least one mesh**, which is an inherited criterion, not a new one.

Cost is reported but is **not** the determinant of the architecture verdict; the
science gates of §§9–11 are.

## 13. Missing-data rules

* A metric that cannot be computed because its raw evidence is absent is
  reported as `UNAVAILABLE` with the reason. It is **never** imputed,
  interpolated, or replaced by a proxy.
* Wall time is reported but the durable cost metrics are outer iterations and
  cumulative inner MMA iterations. The two-rung audit measured 3.8×–5.2× drift
  in seconds-per-inner-MMA-iteration within each run; that measurement is
  repeated here, and if the drift persists wall time is explicitly marked
  unreliable and used for nothing decisive. Timing is never fabricated.
* Missing production density fields (160×20, 320×40) block only the
  S3-versus-P *topology distance*; every other P comparison uses recorded scalars.
* If any §7 check fails on any mesh, the verdict is
  `THREE_RUNG_COUNTERFACTUAL_NOT_EXACT` and the architecture verdict is
  `THREE_RUNG_ARCHITECTURE_INCONCLUSIVE`, regardless of the science.

## 14. Verdict mapping — fixed now

Let a mesh be **three-rung-sufficient** when all of:

* the frozen `E` is satisfied on `move = 0.01` (so the policy terminates
  honestly at that level);
* `S3` clears every applicable gate in §10;
* rung 4 (`S3 → F`) is **below every** materiality threshold in §9.

Primary architecture verdict:

* **`THREE_RUNG_ARCHITECTURE_SUPPORTED`** — all three primary meshes are
  three-rung-sufficient, **and** `THRESHOLD_SPLITTING` is false (rung 3 is itself
  material at 160×20), **and** A9 holds, **and** the counterfactual is exact.
* **`THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED`** — every mesh satisfies `E`
  on `move = 0.01`, the counterfactual is exact, and exactly one of: a single
  mesh shows a material rung-4 benefit; **or** `THRESHOLD_SPLITTING` is true
  while every mesh is otherwise three-rung-sufficient; **or** A9 fails while A4
  holds; **or** a single §10 gate other than A4/A9 fails.
* **`THREE_RUNG_ARCHITECTURE_REFUTED`** — any mesh fails to satisfy `E` on
  `move = 0.01`; **or** rung 4 is material on ≥ 2 of 3 meshes; **or** A4 fails on
  any mesh; **or** the 160×20 residual `S3 → F` `ω₁` gain is still ≥ 0.10 %
  (the three-rung ladder would then fail for the same reason the two-rung ladder
  failed).
* **`THREE_RUNG_ARCHITECTURE_INCONCLUSIVE`** — a §7 check fails, the frozen-rule
  replay disagrees with the recorded trace, or missing evidence prevents
  evaluating a primary mesh.

Next-step verdict `THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED` requires the
architecture verdict `SUPPORTED` **and** all fifteen of the brief's Phase-21
conditions. Otherwise `MORE_THREE_RUNG_EVIDENCE_REQUIRED` (partial support or an
evidence gap) or `MOVE_LADDER_REDESIGN_STILL_REQUIRED` (refuted, or rung 4 shown
necessary on ≥ 2 meshes).

**A narrow miss is a miss.** No bar in §9 or §10 may be relaxed, reinterpreted,
re-anchored or waived after the numbers are on screen, and the 160×20 two-rung
result (+0.1139 % against 0.10 %) is not revisited, softened or explained away.

## 15. Declaration-timing audit (Phase 14) — what is recorded, and what is not concluded

For every stage of every mesh, record: `stageStart`; the first iteration at which
the frozen predicate is **mathematically evaluable** with complete required
history (`stageStart + W − 1 = stageStart + 19`, since the trailing 20-median is
defined only once the whole window lies inside the stage); the earliest
arithmetically possible **declaration** (`stageStart + 38`); the actual
declaration; the offset; and whether `E` was already true at the first evaluable
iteration.

The **only** conclusion this supports, and the only one that will be drawn:

> the lower-rung exhaustion detector is not observing a newly developed
> dynamical transition within those stages; the exhaustion condition is already
> satisfied once sufficient post-transition history exists.

The following will **not** be claimed, because they would require separate tests
this task does not perform: that 39 iterations are optimal; that the persistence
window is unnecessary; that lower stages should be a fixed dwell; that the
controller should descend immediately; that history should be inherited across
transitions. **No rule is altered in response to this observation.**

## 16. Disclosure — what was already visible when this file was frozen

This audit is not blind to its own headline and will not pretend to be.

**Already published in prior studies, readable by anyone before this task:**

* the F-state finals (`M_nd` 12.7041 / 12.9233 / 15.3311; `ω₁` 170.0110 /
  166.4189 / 166.4562) and all three P baselines;
* the S1 and S2 states in full, in `two_rung_architecture/TWO_RUNG_ANALYSIS.md`;
* the combined `S2 → F` deltas, including the decisive 160×20 `ω₁` +0.1139 %;
* the rung boundaries and per-rung outer / inner / wall costs in
  `move_ladder_necessity/METRICS.json` — **including** that file's
  `rungs[3].Mnd_to` and `rungs[3].omega1_to`, which **are** S3's `M_nd` and `ω₁`
  at iterations 180 / 352 / 466. Those fields were **not opened before freezing**,
  but they exist in the repository and the direction of the answer is
  foreseeable. Claiming otherwise would be false;
* that rung 4 consumes 791 / 70 034 / 1 453 inner MMA iterations (§12);
* that 320×40's stage 4 never declares and ends `CAP_HIT @1600`.

**Established during Phases 0/1/3/4 of *this* task, before freezing** —
necessarily, because §§3, 6, 7 and 15 cannot be written without them:

* the static configuration audit result: `anyStopGuard = false`,
  `pOwnCounter = false`, `useProj = false`, so the only final-rung-identity
  dependence (`olhoffSolve.m:485`) is doubly inert;
* that the three-rung ladder resolves to a legal configuration differing from the
  four-rung one in `move.levels` and the override record alone;
* the stage structure of all three C-arm trajectories, and the exact reproduction
  of S1 (102 A / 274 A / 388 B) and S2 (141 A / 313 B / 427 B);
* the **S3 declaration indices 180 / 352 / 466, all Branch B**, with stage-3
  starts 142 / 314 / 428 and offset 38 on every mesh;
* the full declaration-timing table of §15: every lower stage that fires declares
  at `stageStart + 38`, with `E` already true at `stageStart + 19`; stage 1 is
  the sole exception on every mesh; 320×40's stage 4 has `E` false at its first
  evaluable iteration and never declares.

**Not yet computed, and governed by this file:** every S3 scientific value
(`ω₁`, `ω₂`, gap, `M_nd`, gray, mid, volume, densities, bound fraction, native-stop
state); the rung-3-versus-rung-4 split; the residual `S3 → F` percentages; the
per-rung decomposition; density-field distances; the multiplicity comparison; and
— decisively — the **thresholds and gates of §§9–11 and the verdict mapping of
§14**, which are fixed here and inherited from prior frozen commitments rather
than chosen once the table is on screen.

## 17. What this task will not do

No scientific optimization run of any mesh, arm or ladder. No modification of
`A`, `B`, `E`, `W`, `P`, `W_np`, `tol`, the persistence, the windows, the reset
semantics or the move values. No Branch C. No terminal exception, no second
terminal criterion, no adaptive move, no mesh-dependent ladder, no fixed
post-stage dwell, no inherited history, no `[0.04, 0.01]`, no
`[0.04, 0.02, 0.005]`, no re-test of `[0.04, 0.02]`, no alternative ladder of any
kind. No implementation or promotion of the three-rung policy. No projection; no
change to `R`, `p`, mass interpolation, `q`, multiplicity, MMA, FE formulation,
eigensolver, objective or volume constraint. No regeneration of lost evidence.
No fitting of an `NE` law, mesh scaling or move scaling from three meshes.

If the three-rung architecture fails, the failure is **reported, not repaired**.

Production remains `PRODUCTION_CONTROLLER_NOT_CHANGED`; the campaign remains
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` regardless of this audit's outcome.
