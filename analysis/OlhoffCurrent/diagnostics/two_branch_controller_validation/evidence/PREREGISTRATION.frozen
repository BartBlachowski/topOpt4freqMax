# PREREGISTRATION — causal validation of the frozen two-branch move-stage exhaustion controller

Frozen **before the controller is implemented** and therefore before any
candidate optimization result of any kind exists. Nothing below may change after
the first candidate scientific run begins.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `b6014ba8bca41f85671d79ab4c8bdee7419880bb` |
| Tree at task start | **clean** (`git status --porcelain` empty) |
| `+impl/` tree SHA-256 at task start | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files) |
| Currentness | `CURRENT` |
| MATLAB | **25.2.0.2998904 (R2025b)** — base, *not* Update 1 |
| Threads | `runtime.singleThread = true` → `maxNumCompThreads(1)` |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |
| Prior frozen rule | `diagnostics/two_branch_maturity_240/PREREGISTRATION.md`, SHA-256 `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` |
| Prior frozen rule code | `diagnostics/two_branch_maturity_240/scripts/tb_branches.m` |

This is **not** a mechanism-discovery task. No new observable is proposed, no
threshold is tuned, no branch is added. Exactly one already-frozen rule is
turned into an intervention and tested for its causal effect.

---

## 1. The question

> **If β-driven move descent and the native design-change stop are replaced by
> the frozen `A OR B` stage-exhaustion rule, does the optimizer actually produce
> better and scientifically acceptable trajectories?**

This is an intervention test, not a replay. The controller is judged by what the
optimizer does under it.

**Terminology lock (Phase 25).** 240×30 was the *withheld mechanism validation*.
The 160×20 / 320×40 / 400×50 runs authorized here are *causal controller
validation*. The rule was constructed using information from those three meshes,
so these runs are **not** new withheld validation of the maturity rule and will
never be described as such.

---

## 2. The recovered frozen rule — authoritative, not re-derived

Recovered from `two_branch_maturity_240/PREREGISTRATION.md` §§2–7 and its
executable form `two_branch_maturity_240/scripts/tb_branches.m`. Reproduced here
character-for-character so that the controller can be checked against it.

### 2.1 Shared quantities (inherited from `dynamical_regime` §§2–4)

`ρ_k` is the design after outer iteration `k`; `ρ_0` is the uniform initial
design; `Δρ_k = ρ_k − ρ_{k−1}`.

```
cosθ(k)     = <Δρ_k, Δρ_{k−1}> / (‖Δρ_k‖·‖Δρ_{k−1}‖)      NaN if either norm is 0
net_path(k) = ‖ρ_k − ρ_{k−10}‖ / Σ_{j=k−9}^{k} ‖Δρ_j‖      W_np = 10
amp(k)      = ‖Δρ_k‖_2                                     = hist.dxNorm2(k), plain L2
```

Median window `W = 20`, persistence `P = 20`, both inherited. Medians are
trailing over `[k−W+1, k]` and are taken **`'omitnan'`** — the median of the
defined entries in the window, `NaN` only when the window contains none. Any
predicate whose median is `NaN` is **false**.

### 2.2 The convergence scale

```
tol(NE) = 0.05 * sqrt(NE / 3200)
```

= 0.05 (160×20), 0.075 (240×30), 0.1 (320×40), 0.125 (400×50). This is
**exactly** `cfg.stop.tolerance` under the production `stop.toleranceRule =
'meshScaled'` (`olh.config.epsilonForMesh`), verified identical at all four
meshes in `evidence/provenance_start.json`. The controller **reads
`cfg.stop.tolerance`**; it introduces no new numerical constant.

### 2.3 BRANCH A — cancellation / recurrence

```
A(k)  =  med_20 cosθ(k) < 0        AND
         med_20 net_path(k) < 0.5  AND
         amp(k) >= tol(NE)
```

### 2.4 BRANCH B — amplitude convergence

```
B(k)  =  amp(k) < tol(NE)          AND
         med_20 cosθ(k) > 0
```

Branch B is the **inherited native design-change stop criterion plus a coherence
guard plus persistence**. It is not novel physics and is not presented as such.

### 2.5 The union and the event

```
E(k) = A(k) OR B(k)
```

A and B are mutually exclusive at any single `k` (`amp ≥ tol` vs `amp < tol`).
The **exhaustion event** is the first `k` at which either branch's `P = 20`
sustained window *begins*.

### 2.6 Recovery verification performed before freezing

`tb_branches` was re-executed against the one surviving raw fixed-move
trajectory, `evidence/move_activity_400/F400_400x50_trajectory.mat` (400×50,
fixed move 0.04, 369 outer, hash-declared and gate-verified). It reproduces the
recorded 400×50 numbers **bit-exactly**:

| quantity | recomputed now | recorded in `two_branch_maturity_240/METRICS.json` |
|---|---|---|
| `tol` | 0.125 | 0.125 |
| Branch A ever true | never | `kA = null` |
| first `B(k)` true | **369** | `kB = 369` |
| native stop first holds | 369 | `nativeStop = 369` |
| `med_20 cosθ(369)` | 0.9937451892796663 | 0.9937451892796663 |
| `med_20 net_path(369)` | 0.9728297014945854 | 0.9728297014945854 |
| `amp(369)` | 0.1241004791601378 | 0.1241004791601378 |
| `M_nd(369)` | 16.158892933214315 | 16.158892933214315 |
| `ω₁(369)` | 166.3649498603138 | 166.3649498603138 |
| β-stall first fires | 138 | `betaStallFirst = 138` |

Verdict: **`CONTROLLER_DEFINITION_RECOVERY_PASS`**. The definitions are recovered
unambiguously and, on the one mesh whose raw evidence survives, verified
numerically rather than merely read.

**Disclosed limitation.** The raw trajectories of the other three fixed-move arms
(`two_branch_maturity_240/runs/runD_240x30.mat`,
`dynamical_regime/runs/runB_320x40.mat`,
`fixedmove_400_dynamics/runs/runC_400x50.mat`, and the derived
`fm_analysis.mat` / `tb_analysis.mat`) are **absent from this machine** — the
same `.mat` retention loss `EVIDENCE_POLICY.md` was written about, recurring.
Their events are therefore recovered from the tracked `METRICS.json` and
`PREREGISTRATION.md` of the frozen studies, not recomputed. `F400` also stops at
369, so the *persistence* of Branch B beyond 369 cannot be re-verified from
surviving data either; only the window's first iteration can. This is recorded,
not worked around.

---

## 3. Online realization of the frozen rule — the only new semantics

The frozen rule is retrospective: the event is the *beginning* of a sustained
window, which is knowable only once the window closes. A controller must act on
information it already has. The following makes that precise. **Nothing here
changes A, B, `W`, `P`, `tol`, or the union; it fixes only *when* a controller
may act on them.**

### 3.1 Declaration

At outer iteration `t`, after the design update:

```
nA(t) = number of consecutive iterations ending at t on which A held
nB(t) = number of consecutive iterations ending at t on which B held
```

`E` is **declared at the first `t` with nA(t) ≥ 20 or nB(t) ≥ 20**, with branch
identity the counter that reached 20. Since A and B are mutually exclusive per
iteration, at most one counter is non-zero at a time and no tie is possible.

Declaration at `t` ⟺ the frozen window began at `k_begin = t − 19`. Scanning `t`
upward finds the smallest `k_begin` over both branches, so **within a stage the
online declaration reproduces `tb_branches` exactly.**

### 3.2 Stage locality and reset (frozen now)

Let `s` be the first outer iteration executed at the current move level
(`s = 1` for the first stage). Every quantity entering the predicate is computed
**only from iterations `j ≥ s`**:

| quantity | first stage-local iteration |
|---|---|
| `amp(j)` | `j ≥ s` |
| `cosθ(j)` | `j ≥ s+1` (needs `Δρ_{j−1}`) |
| `net_path(j)` | `j ≥ s+9` (needs `Δρ_{s..j}` and `ρ_{s−1}`) |
| `med_20 cosθ`, `med_20 net_path` | trailing 20-window restricted to stage-local entries, `'omitnan'` |
| `nA`, `nB` | reset to 0 at `j = s` |

**Why full reset and not carry-over.** Under a move change `‖Δρ‖` is bounded by
a different constant, so a window straddling the change reports the *schedule*
rather than the design — precisely the defect `stop.guards.settledMove` exists to
suppress, extended consistently to the whole window. Carrying the window across a
transition would let a mechanically halved amplitude satisfy Branch B's `amp <
tol` clause and trigger a spurious descent. The reset is conservative: it can
only delay a descent, never advance one.

**Consistency with the frozen rule.** For the first stage, `s = 1`, and the
stage-local definitions coincide *exactly* with `tb_branches` (`cosθ` defined
from 2, `net_path` from 10, `'omitnan'` medians). The candidate's stage-1
behaviour is therefore the frozen rule applied unchanged to a fixed-move
trajectory.

### 3.3 Acting on the declaration

The move controller is consulted at the **top** of iteration `k` with state
through `k−1`; the convergence test runs at the **bottom** of iteration `k` with
state through `k`. Hence, from one declaration at `t`:

* **stage descent** — if the current stage is not the last, `move` advances one
  rung and the **first iteration executed at the new level is `t+1`**. That is
  the reported *transition iteration*. `stageStart := t+1`; the reset of §3.2
  applies. Exactly one rung per declaration.
* **terminal admission** — if the current stage *is* the last (`move = 0.005`),
  there is no lower rung, so the same declaration admits convergence, and the
  **run ends at iteration `t`**.

This is the Phase-3 requirement: one scientific concept governs both the
transition and the terminal admission.

`move` can never fall below `0.005`: the ladder index is `min(stage+1,
numel(levels))` and the last rung converts the declaration into termination
rather than a descent.

### 3.4 β loses all authority

Under the candidate, `β` is still computed and is still the bound variable of Du
& Olhoff Eq. (25a) — it is untouched inside the optimization problem. It has
**no** authority to descend the move ladder and **no** authority to admit
convergence. Production's β-stall predicate is replayed and logged at every
iteration as the causal counterfactual, and never read by the controller.

---

## 4. Controller semantics, stated as the implementation contract

```
levels = [0.04 0.02 0.01 0.005]                     UNCHANGED
stage  = 1 at iteration 1,  stageStart = 1

at each outer iteration k:
    (top)     if stage < 4 and E was declared at k-1:
                  stage      := stage + 1
                  stageStart := k
              move := levels(stage)
    ... FE, eigenproblem, sensitivities, filter, MMA, design update (UNCHANGED)
    (bottom)  update amp/cosθ/net_path/medians/counters from stage-local data
              if stage == 4 and E declared at k:  CONVERGED at k
```

Terminal status is decided by the existing precedence
`SOLVER_FAILURE > CAP_HIT > CONVERGED > STOPPED_OTHER`, unchanged. `CAP_HIT`
stays `CAP_HIT`; an inner-solver failure stays a failure. A candidate may report
`CONVERGED` **only** if `move == 0.005` and the frozen persistence was satisfied
there.

---

## 5. Implementation isolation and the single-factor requirement

The controller is added to the canonical implementation behind **two explicit
configuration switches**, so production behaviour stays selectable and bitwise
reproducible:

```
move.continuation.signal :  'boundVariable' (production)  |  'stageExhaustion' (candidate)
stop.rule                :  'designChange'  (production)  |  'stageExhaustion' (candidate)
```

`stop.rule` is a new field whose **default is `'designChange'`**, i.e. exactly
today's behaviour. With both switches at their production values the solver must
be **bitwise identical** to the current production solver; this is a required
software test (§8, test 14).

Everything else is locked and must be field-for-field identical between
production baseline and candidate:

`p = 3` · no p continuation · mass Eq. (4b) · `q = 1` · sensitivity filter
applied to all `f_sk` · `R = 0.06·b` physical · projection **off** · subspace
multiplicity size 2 with diagonal offsets and off-diagonals · published MMA on
the increment · FE formulation · eigensolver · target mode · objective · volume
fraction 0.5 · initial design · `move.levels = [0.04 0.02 0.01 0.005]` ·
`stop.tolerance` (mesh-scaled) · `stop.norm` · single thread.

**Declared, necessary differences** — the intervention plus telemetry, nothing
else:

| field | production baseline | candidate | why |
|---|---|---|---|
| `move.continuation.signal` | `boundVariable` | `stageExhaustion` | **the intervention** |
| `stop.rule` | `designChange` | `stageExhaustion` | **the intervention** |
| `runtime.maxOuter` | 400 | **1600** | safety cap, §7 |
| `runtime.diagnostics` | true | true | identical |
| `runtime.name` | per arm | per arm | label only |

Any other differing scientific field ⇒ **`CONTROLLER_SINGLE_FACTOR_FAIL`**, stop
before interpreting results.

---

## 6. Prefix-equivalence check (a hard, falsifiable single-factor proof)

Under the candidate the first stage holds `move = 0.04` and changes nothing else,
so the candidate's stage-1 prefix must be **bitwise identical** to a fixed-move
0.04 arm on the same mesh, same MATLAB.

`F400_400x50_trajectory.mat` is such an arm, produced under **this same MATLAB
build** (`25.2.0.2998904`), 369 outer iterations. Preregistered requirement:

> **C400 outer iterations 1…369 must reproduce `F400` bitwise** — `RHO`
> columns, `hist.omega`, `hist.beta`, `hist.dxNorm2`, `hist.nInner` all exactly
> equal.

Any deviation is an implementation defect, not a scientific result. No equivalent
same-version fixed-move arm survives for 160×20 or 320×40, so the check is stated
for 400×50 only.

---

## 7. Safety caps — frozen before any candidate run

**`runtime.maxOuter = 1600`, common to all three meshes.**

Justification, from prior evidence only. The stage-1 exhaustion declarations
implied by the frozen rule are 102 (160×20), 274 (320×40) and 388 (400×50), so
first descents fall at 103 / 275 / 389. Stage 1 is the longest stage by
construction: it carries the largest move and all of the gross topology
evolution, and each later stage halves the move, which monotonically eases Branch
B's `amp < tol` clause. 1600 leaves ≥ 1200 iterations for the remaining three
stages on the worst mesh — over four times the longest stage ever observed.

The cap **will not be raised after seeing a trajectory.** `CAP_HIT` is a valid
scientific outcome and will be reported as one.

---

## 8. Software tests required before any scientific run

Deterministic, sub-160×20 meshes permitted; **these are software tests, not
scientific evidence.**

1. `A` true, `B` false ⇒ `E` true.
2. `A` false, `B` true ⇒ `E` true.
3. `A` false, `B` false ⇒ `E` false.
4. Persistence boundary: 19 consecutive ⇒ no declaration; 20 ⇒ declaration.
5. Window reset after descent: no stage-local quantity uses pre-transition data;
   earliest possible post-transition declarations are `s+39` (B) and `s+47` (A).
6. Exactly one rung per accepted transition.
7. `move` never drops below 0.005.
8. β stall alone cannot descend the move.
9. β stall alone cannot admit terminal convergence.
10. `move_min` with `E` false cannot converge.
11. `move_min` with persistent `E` true does converge, at the declaring iteration.
12. `CAP_HIT` remains `CAP_HIT`.
13. Solver failure remains a failure.
14. **Candidate OFF reproduces production bitwise** on a real solve.
15. Telemetry alters no numerical result (diagnostics on/off bitwise equal).
16. **Online/offline equivalence**: on fixed-move trajectories the online
    detector's declaration equals `tb_branches`' `event + 19` and branch
    identity, for the surviving `F400` arm and for constructed synthetic
    trajectories exercising A, B, neither, and both-in-sequence.

The existing `analysis/OlhoffCurrent/tests` suite must also pass. Any failure ⇒
**`CONTROLLER_SOFTWARE_GATE_FAIL`**, stop.

---

## 9. Frozen production baselines (Phase 8) — durable evidence, no reruns

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| source | `move_stop/runs/baseline_160x20_iterations.csv` + `METRICS.json` | `move_stop/runs/baseline_320x40_iterations.csv` + `METRICS.json` | `move_activity_400` `METRICS.json` + `evidence/.../P400_400x50_trajectory.mat` |
| status | `CONVERGED` | `CONVERGED` | `CONVERGED` |
| outer | 91 | 131 | 139 |
| inner (MMA) total | 2241 | 2614 | 2918 |
| wall s | 125.48 | 387.96 | 541.0 |
| transitions | 79 (.04→.02), 90 (.02→.01) | 130 (.04→.02) | 138 (.04→.02) |
| final move / stage | 0.01 / 3 | 0.02 / 2 | 0.02 / 2 |
| ω₁ | 169.49522702153845 | 165.95078925220545 | 162.882615630062 |
| ω₂ | 171.9599959181592 | 183.76971226824654 | 175.403921349379 |
| gap₁₂ | 0.01476553890041266 | 0.10674152016769037 | 0.0768731866865097 |
| volume | 0.49999900877797954 | 0.49999913861781514 | 0.499999149662958 |
| `M_nd` % | 13.402499279526742 | 23.359567952074254 | 32.3283256233213 |
| gray | 0.149375 | 0.26375 | 0.3476 |
| mid | 0.025 | 0.095625 | 0.1882 |
| final ρ available | **no** (`.mat` lost) | **no** (`.mat` lost) | **yes** |
| MATLAB | 25.2.0.3042426 (Update 1) | 25.2.0.3042426 (Update 1) | **25.2.0.2998904** (same as now) |

**No production rerun is authorized.** Fields unavailable (160×20 and 320×40
final density vectors and their hashes) are marked unavailable and never
fabricated. The 160×20 and 320×40 baselines were produced under MATLAB Update 1
while the candidate runs use the base build; this is disclosed and means those
two comparisons are *same-configuration* but not *same-binary*. The 400×50
comparison is same-binary.

**Note on what production actually does.** On none of the three meshes does
production reach `move = 0.005`. It descends at the β stall and then converges
1–12 iterations later, because halving the move mechanically drops `‖Δρ‖₂` below
`tol`. That is the behaviour under test.

---

## 10. Preregistered causal predictions

Stated now, before implementation, to expose post-hoc rationalization.

**Shared.**
* Every move transition is attributable to a declared `A` or `B`; none to β.
* All four levels remain reachable; `move` never goes below 0.005.
* No pathological non-termination other than an honest `CAP_HIT`.
* Volume feasibility maintained: `|volume − 0.5| ≤ 1e-4` at the final design.
* Terminal `CONVERGED` only at `move = 0.005` after frozen persistence.

**160×20** — the coarse-mesh safety case.
* Branch **A** declares at **102**; first descent at **103** (±5 iterations,
  allowing for the MATLAB-build difference against the Update-1 fixed-move arm).
* The candidate escapes the full-amplitude mature cycle by descending, rather
  than churning at `move = 0.04` indefinitely.
* Final `M_nd` **not materially worse** than production's 13.4025 (bound §11).
* ω₁ not materially worse than production's 169.4952 (bound §11).

**320×40.**
* Candidate holds `move = 0.04` **materially beyond** production's descent at
  130; Branch **A** declares at **274**, first descent **275** (±5).
* Final `M_nd` **materially better** than production's 23.3596 (bound §11). The
  fixed-move evidence puts `M_nd` at the event at 13.02, ≈44 % below production.
* ω₁ not degraded beyond §11.

**400×50.**
* Candidate holds `move = 0.04` materially beyond production's descent at 138;
  Branch **B** declares at **388**, first descent **389** (exact — same MATLAB
  build, and the §6 prefix check applies).
* Final `M_nd` materially better than production's 32.3283. The fixed-move
  evidence puts `M_nd` at the event at 16.16, ≈50 % below production.
* ω₁ not degraded beyond §11.

No exact final values are preregistered; only directions and the §11 bounds.

**Anticipated and accepted risk, recorded before the runs.** The union has a
known hole: `amp < tol` together with `med cosθ < 0` — *low-amplitude
cancellation* — satisfies neither branch. At the lower rungs `amp` falls
mechanically, so a mesh that ends in a low-amplitude cycling regime may exhaust
no further stage and reach the cap. If that happens it is a real limitation of
the frozen rule and will be reported as one. **It will not be repaired in this
task.**

---

## 11. Promotion gates — numerical bounds, frozen before the first run

All bounds derive from prior evidence (§9 baselines, the fixed-move arms) and
scientific reasoning, never from candidate outcomes.

| gate | requirement |
|---|---|
| **P1** | Software gate (§8, 16 tests) and single-factor gate (§5) both pass. |
| **P2** | No mesh terminates falsely: every `CONVERGED` has `move == 0.005` and satisfied frozen terminal persistence. |
| **P3** | No run ends in an unacknowledged failure; `SOLVER_FAILURE` / `CAP_HIT` reported as such. |
| **P4** | 160×20 is not trapped at `move = 0.04`: its first descent occurs at outer ≤ **400**. |
| **P5** | 160×20 regression bound: final `M_nd` ≤ **1.10 ×** production's 13.4025 (≤ 14.7427) **and** ω₁ ≥ **0.99 ×** production's 169.4952 (≥ 167.8003). |
| **P6** | 320×40: first descent ≥ production's 130 **+ 50** = **180**, and final `M_nd` ≤ **0.80 ×** production's 23.3596 (≤ 18.6877), i.e. ≥ 20 % relative improvement. |
| **P7** | 400×50: first descent ≥ production's 138 **+ 50** = **188**, and final `M_nd` ≤ **0.80 ×** production's 32.3283 (≤ 25.8627). |
| **P8** | ω₁ on **every** mesh ≥ 0.99 × the production baseline's ω₁. |
| **P9** | `|volume − 0.5| ≤ 1e-4` on every mesh. |
| **P10** | Multiplicity/physics acceptable: no eigensolver failure, all ω finite, ω₂ > ω₁, subspace size 2 throughout, no NaN/Inf in `hist`. |
| **P11** | Every move transition attributable to a declared frozen `A` or `B`; none to β stall; none without a declaration. |
| **P12** | Terminal convergence occurs only at `move_min` after frozen exhaustion. |
| **P13** | Cost: outer-iteration multiplier ≤ **8×** and wall-time multiplier ≤ **10×** the production baseline, on every mesh. Otherwise the cost is explicitly judged. |
| **P14** | No mesh-specific tuning: one controller, byte-identical source, across all three runs. |
| **P15** | Evidence complete, declared, hash-valid; the 16 software tests and the provenance gate recorded. |

*Rationale for the two quantitative bounds that carry the verdict.* The 20 %
`M_nd` improvement bar at the fine meshes is far below what the fixed-move
evidence says is available at the stage-1 event alone (≈44 % at 320×40, ≈50 % at
400×50) and far above any plausible trajectory noise, so it tests the causal
claim without being either a formality or an unattainable bar. The 1 % one-sided
ω₁ allowance is generous against the fixed-move evidence, in which ω₁ at the
event is *better* than production's at both fine meshes (+0.29 %, +2.14 %) and
only 0.17 % below at 160×20; this is an eigenfrequency **maximization** problem,
so a real loss of ω₁ would be a genuine scientific regression.

---

## 12. Verdicts

Exactly one primary verdict:

* **`TWO_BRANCH_CONTROLLER_VALIDATED`** — every gate P1–P15 passes.
* **`TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED`** — P1–P3 and P9–P12 pass, at
  least one mesh meets its improvement gate, but at least one of P4–P8 or P13
  fails.
* **`TWO_BRANCH_CONTROLLER_REJECTED`** — a false termination (P2), an
  unacknowledged failure (P3), a physics blocker (P10), a transition not
  attributable to the frozen rule (P11), or a scientific regression exceeding
  P5/P8 on any mesh.
* **`TWO_BRANCH_CONTROLLER_INCONCLUSIVE`** — an implementation or provenance
  failure prevents interpretation, or caps bind so early that the controller's
  effect cannot be read.

`VALIDATED` is **not** awarded because 320×40 and 400×50 look better. A coarse-mesh
failure is not offset by fine-mesh gains.

**Promotion** only on `TWO_BRANCH_CONTROLLER_VALIDATED` with all gates passing,
and then only as the exact tested controller, followed by
`CONTROLLER_PROMOTION_EQUIVALENCE_PASS` / `_FAIL`.

**`NINE_MESH_PERFORMANCE_CAMPAIGN_AUTHORIZED`** requires `VALIDATED`, promotion
equivalence `PASS` if promotion occurred, a frozen and clean production tree,
passing tests, complete evidence and no residual controller blocker. Otherwise
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.

---

## 13. What this task will not do

Exactly three candidate scientific runs: **160×20, 320×40, 400×50**. No 240×30
candidate. No 480×60 / 560×70 / 640×80 / 720×90 / 800×100. No nine-mesh campaign.
No rerun of any fixed-move mechanism arm. No production rerun. No change to `A`,
`B`, `W`, `P`, `tol`, persistence, the move ladder, or the reset semantics after
the first candidate result. No Branch C. No mesh-specific parameter. No NE
exponent. No projection. No change to `R`, `p`, the mass model, `q`, the filter
type, MMA, the FE formulation, the eigensolver, the objective or the volume
constraint. No tuning against `M_nd` or ω₁. If the controller does not validate,
the task stops with the failure evidence intact.
