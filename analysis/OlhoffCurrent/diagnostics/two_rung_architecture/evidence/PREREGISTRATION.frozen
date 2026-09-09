# PREREGISTRATION — does the exact two-rung ladder `[0.04, 0.02]` suffice?

A **zero-scientific-run** offline audit. No optimization of any kind is executed.
Every number comes from causal-controller trajectories that already exist on disk.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| Tree at task start | dirty, 21 paths (two prior studies' uncommitted deliverables + this study's directory) |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| MATLAB available now | `25.2.0.2998904 (R2025b)` — **not** the `25.2.0.3042426 (R2025b) Update 1` build that produced the C-arm trajectories. Used here only for hashing and gate arithmetic; **no solve is executed**, so the build difference cannot affect any scientific number. |
| Analysis language | Python 3 (offline replay), MATLAB only for provenance/finalization gates |
| Prior status | `FOUR_RUNG_LADDER_PARTIALLY_USEFUL` · `RETAIN_MOVE_LADDER_PENDING_REDESIGN` · `PRODUCTION_CONTROLLER_NOT_CHANGED` · `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` |

---

## 1. The question

> Does the exact two-rung policy `move levels = [0.04, 0.02]`, governed end to end
> by the **already-frozen** exhaustion rule `E = A OR B`, retain all
> scientifically material benefit of the four-rung controller while avoiding the
> cost and the non-termination pathology of rungs 3 (`0.01`) and 4 (`0.005`)?

Exactly one architecture is under test. Nothing is tuned, invented, or varied.

## 2. The counterfactual policy, stated exactly

```
TWO-RUNG POLICY (S2-policy)
    cfg.move.levels          = [0.04, 0.02]        <- the ONLY change
    cfg.move.continuation.signal = 'stageExhaustion'   (unchanged)
    cfg.stop.rule                = 'stageExhaustion'   (unchanged)

    STAGE 1   move = 0.04, held until the frozen detector DECLARES E = A OR B
    DESCENT   exactly one, 0.04 -> 0.02, detector reset to the new stage
    STAGE 2   move = 0.02, held until the frozen detector DECLARES E = A OR B
    TERMINATE at that declaration (stage 2 is now the last level)
```

compared against three reference states:

```
P   production baseline           beta-stall ladder, the frozen baselines
S1  single-stage endpoint         first frozen declaration at move = 0.04
F   four-rung candidate endpoint  the controller already validated in
                                  two_branch_controller_validation
```

## 3. Why no simulation is needed — the prefix argument (to be *checked*, not assumed)

Under `stop.rule = 'stageExhaustion'` and
`move.continuation.signal = 'stageExhaustion'`, the ONLY places the solver reads
the length of `cfg.move.levels` on the active code path are

* `olh.move.limit` line 109 — descend iff `ex.declared && stage < numel(levels)`;
* `olhoffSolve` line 509 — `atLastLevel = stage >= numel(levels)`, and
  `convOuter = ex.declared && atLastLevel`.

The stop guards (`restorationReady`, ladder guard, max-change guard) are gated on
`~exhaustStop` and are therefore inert. Projection is off. `p` continuation is
off. Every other quantity — the move value at a stage, the detector, MMA, the
sensitivities, the objective, the constraint — is a function of the stage index
alone, not of how many levels exist below it.

Therefore, with `levels = [0.04, 0.02]` versus `levels = [0.04, 0.02, 0.01, 0.005]`,
the two runs are **bitwise identical** for every outer iteration up to and
including the stage-2 declaration `kE2`, because both predicates
(`stage < numel`, `stage >= numel`) evaluate identically for `stage ∈ {1, 2}`
while `stage < 4` and `stage < 2` first disagree only at `stage = 2`, i.e. only
at `kE2` itself — where the four-rung run descends and the two-rung run stops.

**S2 is therefore the recorded state of the existing four-rung candidate
trajectory at its stage-2 declaration, exactly.** This argument is *verified* in
Phase 3 against the recorded traces; if any of the five checks in §7 fails, the
verdict is `TWO_RUNG_ARCHITECTURE_INCONCLUSIVE`.

## 4. Sources — the exact trajectories used

| mesh | role | trajectory | telemetry |
|---|---|---|---|
| 160×20 | primary | `evidence/two_branch_controller_validation/C160x20_trajectory.mat` | `runs/C160x20_iterations.csv` |
| 320×40 | primary | `.../C320x40_trajectory.mat` | `runs/C320x40_iterations.csv` |
| 400×50 | primary | `.../C400x50_trajectory.mat` | `runs/C400x50_iterations.csv` |
| 240×30 | **unavailable** | none exists | none |

Production baselines: `two_branch_controller_validation/evidence/baselines.json`
(frozen, hash-recorded there).

**240×30, declared now.** `analysis/OlhoffCurrent` contains **zero** files
matching `*240x30*`. The only 240×30 study, `two_branch_maturity_240`, has no
`runs/` directory (its raw `.mat` artifacts are among the losses that motivated
the finalization gate), and its arm was in any case a **fixed-move** arm
(`move.policy = 'fixed'`, `move.initial = 0.04`) that never executed a
`move = 0.02` stage. 240×30 is therefore marked
`UNAVAILABLE_FOR_TWO_RUNG_CAUSAL_COMPARISON` on two independent grounds. It will
**not** be run, and no S2 will be inferred for it.

**Production density fields, declared now.** `baselines.json` records
`rho_available = false` for 160×20 and 320×40 (raw `.mat` lost). Density-field
distance to P is therefore computable only at 400×50, and is reported as a gap
elsewhere.

## 5. The frozen rule — recovered, not re-derived

The authoritative executable definition is
`analysis/OlhoffCurrent/+impl/architecture/+olh/+move/exhaustion.m`, whose
provenance line names
`two_branch_maturity_240/PREREGISTRATION.md` SHA-256
`62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd`
(verified live at task start). Restated for the record only:

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
stage-local; only the net-path ANCHOR rho_{s-1} is pre-stage.  Earliest possible
declaration in any stage is stageStart + 38.
```

**Nothing in this rule is modified, extended, relaxed, or supplemented in this
task.** No Branch C. No terminal exception. No mesh-dependent condition.

**Verification requirement.** An independent Python re-implementation
(`scripts/tr_frozen.py`, itself a re-use of `move_ladder_necessity/scripts/ml_frozen.py`
with stage-local windows) must reproduce, **element-wise over every stage**, the
`exA` / `exB` / `exE` / `exNA` / `exNB` trace the solver actually acted on, and
must reproduce each recorded declaration iteration exactly. Any disagreement ⇒
`TWO_RUNG_ARCHITECTURE_INCONCLUSIVE`.

## 6. Index convention — fixed now, used everywhere

One convention, stated once:

* **`kE(s)` — the event index of stage `s`** is the outer iteration at which the
  frozen detector *declares*, i.e. the iteration at which `nA` or `nB` first
  reaches 20 within that stage. Equivalently the first iteration at which
  `hist.exDecl = 1` after the stage began. The sustained window is
  `[kE − 19, kE]`.
* **The terminal state of a policy whose last level is stage `s`** is the
  recorded row at `outer = kE(s)` — the design `ρ_{kE}` *after* that iteration's
  update. This is exactly what `olhoffSolve` returns: `convOuter` is set inside
  the same iteration in which the declaration is made.
* **The descent index** is `kE(s) + 1`: the first iteration executed at the next
  level. In the four-rung trajectories this is the iteration at which
  `hist.move` first takes the new value.

Hence `S1 = row(kE(1))`, `S2 = row(kE(2))`, `F = last row`. Prior briefs quoted
stage-1 events as "~102/103" etc.; under this convention the event is the first
number (the declaration) and the second is the descent. **No other convention is
used anywhere in this study.**

Cost at a state: `cumInner` at that row; cumulative wall = `sum(tOuter)` through
that row; outer = the row index.

## 7. Counterfactual-validity checks (Phase 3) — all five required, per mesh

1. `hist.move = 0.04` for every `k ≤ kE(1)`, and `stage = 1` there;
2. exactly one descent event in `[1, kE(2)]`, at `kE(1)+1`, to `move = 0.02`;
3. `hist.move = 0.02` for every `k` in `[kE(1)+1, kE(2)]`, and `stage = 2` there;
4. no `move = 0.01` (or lower) iteration occurs at or before `kE(2)`;
5. `hist.exStageStart = kE(1)+1` throughout stage 2 — i.e. the recorded reset
   semantics are exactly those the two-rung policy would produce.

## 8. Extraction (Phase 5) — the S2 record

At `S1`, `S2` and `F`, per mesh, record: outer, move, stage, triggering branch,
`A`/`B`/`E` state, `nA`/`nB`, `exStageStart`, `ω₁`, `ω₂`, `gap12`, volume,
`volErr`, `M_nd`, gray fraction, mid-density fraction, `max|Δρ|`, `max|Δρ|/move`,
`‖Δρ‖₂`, RMS `Δρ`, `cosθ`, net/path, bound fraction, β-stall state (`betaStallRel`,
`betaStallFires`), native-stop state (`prodStopRaw`, `prodSettled`, `prodStopAdmit`),
`‖Δρ‖₂/tol`, `nInner`, `cumInner`, cumulative wall, subspace size `multN`,
degeneracy flag, and the SHA-256 of the density vector.

Density-field differences between two states are reported as mean `|Δρ_e|`,
`‖Δρ‖₂/√NE`, and `max|Δρ_e|`.

## 9. Materiality thresholds — inherited verbatim, fixed before extraction

Taken **unchanged** from `move_ladder_necessity/PREREGISTRATION.md` §6
(SHA-256 `a08b879b9d4893bc55dd9e5dfc2c26f897cb9d49355f0557ccc8644c7fc6ca3e`), so
that rungs 3+4 are judged by exactly the bar rungs 2+3+4 were judged by. They are
not re-derived and not re-anchored here.

| quantity | material if | anchor |
|---|---|---|
| `M_nd` | the rung block improves `M_nd` by **≥ 2 % relative** | one tenth of the controller study's 20 % "materially better" bar |
| `ω₁` | the rung block improves `ω₁` by **≥ 0.10 % relative** | one tenth of the controller study's 1 % acceptable-degradation bound (≈ 0.17 absolute here) |
| topology | gray or mid fraction changes by **≥ 0.01 absolute**, or mean \|Δρ_e\| **≥ 0.01** | one tenth of the smallest change the project has called meaningful |
| volume feasibility | \|volume − 0.5\| worsens by **≥ 1e-5** | both arms achieve ≈ 1e-6–1e-7 |
| multiplicity / physics | subspace size leaves 2, mode order changes, ω₂ ≤ ω₁, or a NaN/Inf appears | qualitative; **gap magnitude alone is not material**, because the objective is ω₁ |
| cost domination | a rung block costs **≥ 2×** the outer iterations used to reach the state it starts from, while delivering sub-material benefit on every metric above | the ladder must earn its cost |
| failure risk | any rung sequence producing `CAP_HIT` or a non-terminating stage | a controller that cannot stop is a defect regardless of quality |

## 10. Production-relative acceptance gates for S2 — inherited verbatim

`S2` is a candidate *terminal state*, so it must clear the same gates the
four-rung controller was required to clear.  Taken unchanged from
`two_branch_controller_validation/PREREGISTRATION.md` §11 (SHA-256
`8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf`):

| gate | requirement at S2 |
|---|---|
| **A1** (= P5) | 160×20: `M_nd ≤ 1.10 ×` production 13.4025 (≤ 14.7427) **and** `ω₁ ≥ 0.99 ×` production 169.4952 (≥ 167.8003) |
| **A2** (= P6) | 320×40: `M_nd ≤ 0.80 ×` production 23.3596 (≤ 18.6877) |
| **A3** (= P7) | 400×50: `M_nd ≤ 0.80 ×` production 32.3283 (≤ 25.8627) |
| **A4** (= P8) | every mesh: `ω₁ ≥ 0.99 ×` its production `ω₁` |
| **A5** (= P9) | every mesh: `|volume − 0.5| ≤ 1e-4` |
| **A6** (= P10) | every mesh: subspace size 2, `ω₂ > ω₁`, all `ω` finite, no NaN/Inf |
| **A7** (= P13) | every mesh: outer multiplier ≤ 8× and wall multiplier ≤ 10× the production baseline |
| **A8** (= P12) | termination occurs only at the last level after frozen persistence |

**One further gate, specific to this audit's central safety question, fixed now.**
The prior audit's stated objection to the single-stage policy was not that it
breached A4 — S1 does not — but that it **ships a regression on the maximized
objective**: `ω₁(S1) < ω₁(P)` at 160×20. Phase 19 condition 3 requires that
regression be repaired. So:

| gate | requirement |
|---|---|
| **A9 — no-regression** | 160×20: `ω₁(S2) ≥ ω₁(P) = 169.49522702153845` |

A4 and A9 are reported separately and neither is allowed to stand in for the
other.

## 11. Missing-data rules

* A metric that cannot be computed because its raw evidence is absent is
  reported as `UNAVAILABLE` with the reason. It is **never** imputed,
  interpolated, or replaced by a proxy.
* Wall time is reported, but the durable cost metrics are outer iterations and
  cumulative inner MMA iterations. If any mesh's `tOuter` shows evidence of
  machine contention, wall time for that mesh is explicitly down-weighted and
  said to be so. Timing is never fabricated.
* Missing production density fields (160×20, 320×40) block only the
  S2-versus-P *topology distance*; every other P comparison uses the recorded
  scalars.
* If any of the five §7 checks fails on any mesh, the verdict is
  `TWO_RUNG_ARCHITECTURE_INCONCLUSIVE` regardless of the science.

## 12. Verdict mapping — fixed now

Let a mesh be **two-rung-sufficient** when all of:

* the frozen `E` is satisfied on `move = 0.02` (so the policy terminates
  honestly at that level);
* `S2` clears every applicable gate in §10;
* rungs 3+4 (`S2 → F`) are **below every** materiality threshold in §9.

Primary architecture verdict:

* **`TWO_RUNG_ARCHITECTURE_SUPPORTED`** — all three primary meshes are
  two-rung-sufficient, **and** at 160×20 rung 2 (`S1 → S2`) is itself material
  by at least one §9 criterion (so the retained rung is load-bearing, not
  ballast), **and** A9 holds.
* **`TWO_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED`** — every mesh satisfies `E` on
  `move = 0.02` and no §7 check fails, but exactly one of: a single mesh shows a
  material rung-3+4 benefit; **or** A9 fails while A4 holds; **or** a single
  §10 gate other than A4/A9 fails.
* **`TWO_RUNG_ARCHITECTURE_REFUTED`** — any mesh fails to satisfy `E` on
  `move = 0.02`; **or** rungs 3+4 are material on ≥ 2 of 3 meshes; **or** A4
  fails on any mesh; **or** rung 2 is immaterial at 160×20 *and* the S1
  regression persists at S2 (the two-rung ladder would then be buying nothing).
* **`TWO_RUNG_ARCHITECTURE_INCONCLUSIVE`** — a §7 counterfactual check fails, the
  frozen-rule replay disagrees with the recorded trace, or missing evidence
  prevents evaluating a primary mesh.

Next-step verdict `TWO_RUNG_POLICY_PREREGISTRATION_JUSTIFIED` requires the
architecture verdict `SUPPORTED` **and** all twelve of the brief's Phase-19
conditions. Otherwise `MORE_TWO_RUNG_EVIDENCE_REQUIRED` (evidence gap or
partial support) or `RETAIN_FOUR_RUNG_ARCHITECTURE_PENDING_REDESIGN`
(refuted, or rungs 3+4 shown necessary).

## 13. Disclosure — what was already visible when this file was frozen

This audit is not blind to its own headline, and it will not pretend to be.
Before this file was hashed the following were already established, and are
listed so that nothing below can be presented as a surprise:

**Already published in prior studies (readable by anyone, before this task):**

* the F-state finals (`M_nd` 12.7041 / 12.9233 / 15.3311; `ω₁` 170.0110 /
  166.4189 / 166.4562) and all three P baselines;
* the S1-state values (160×20: `ω₁` 168.9804, `M_nd` 13.0364; 320×40: 166.4216 /
  13.0121; 400×50: 166.4176 / 15.6649);
* the rung boundaries and per-rung outer/inner/wall costs, in
  `move_ladder_necessity/METRICS.json` — **including** that file's
  `rungs[2].Mnd_to` and `rungs[2].omega1_to`, which *are* S2's `M_nd` and `ω₁`.
  Those two fields were **not opened before freezing**, but they exist in the
  repository and the direction of the answer is foreseeable. Claiming otherwise
  would be false;
* that 320×40 ended `CAP_HIT @1600` and that rung 4 there consumed ≈ 91.5 % of
  inner work for ≈ 0.13 % `M_nd`;
* that rung 2 was previously found to carry essentially all lower-ladder value.

**Established during Phase 0/1 of *this* task, before freezing** — necessarily,
because §3 and §7 cannot be written without them:

* the stage structure of all three C-arm trajectories (stage starts 1/103/142/181,
  1/275/314/353, 1/389/428/467);
* the stage-2 declaration indices **141 / 313 / 427** and their triggering
  branches (**A** at 160×20; **B** at 320×40 and 400×50), together with the
  detector telemetry (`medcos`, `mednet`, `amp`) at those iterations;
* the fact — visible in that telemetry and stated here rather than later — that
  every post-first stage declares at `stageStart + 38`, the **earliest
  arithmetically possible** iteration, on every mesh. The scientific consequence
  of that observation is analysed in the report and is **not** used to alter any
  rule.

**Not yet computed, and governed by this file:** the full S2 state extraction;
the `P/S1/S2/F` comparison on every metric; the rung-2 versus rungs-3+4
decomposition; density-field distances; multiplicity comparison; cost fractions;
the bound-limitation description; and — decisively — the **thresholds and gates
in §§9–10 and the verdict mapping in §12**, which are fixed here and inherited
from prior frozen commitments rather than chosen once the table is on screen.

## 14. What this task will not do

No scientific optimization run of any mesh, arm, or ladder. No modification of
`A`, `B`, `E`, `W`, `P`, `W_np`, `tol`, the persistence, the reset semantics or
the move values. No Branch C. No terminal exception, no second terminal
criterion, no adaptive move, no mesh-dependent ladder, no `0.04 → 0.01`, no
change to the initial move, no alternative ladder of any kind. No implementation
or promotion of the two-rung policy. No projection; no change to `R`, `p`, mass
interpolation, `q`, multiplicity, MMA, FE formulation, eigensolver, objective or
volume constraint. No regeneration of lost evidence.

Production remains `PRODUCTION_CONTROLLER_NOT_CHANGED`; the campaign remains
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` regardless of this audit's outcome.
