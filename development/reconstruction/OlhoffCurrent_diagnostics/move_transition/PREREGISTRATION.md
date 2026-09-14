# PREREGISTRATION — move-stage transition signal

Frozen **before** any candidate optimization was run. Every threshold below is
**inherited**, not chosen here; none may be altered after results are observed.

| | |
|---|---|
| Implementation | `analysis/OlhoffCurrent` (sole production Olhoff) |
| Repository branch / HEAD at freeze | `benchmark-methodology-r2` / `a1f2c6c4dfae8d48ee76ba30cd735bf7a776d3c1` |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 source files) |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` (upstream `duOlhoffFrozenM4`) |
| Promoted upstream commit | `695f03bdac20c423a4e1d389cf9db9187597bcc3` |
| MATLAB | R2025b Update 1, **1 computational thread** (`runtime.singleThread = true`) |
| Path gate | `CURRENT` · dispatch ok · 32 owned symbols inside `+impl/` · published MMA wins · sensitivity filter wins · no competing Olhoff tree |

---

## 1. Question

Does replacing the **bound-variable-stall** trigger for move-ladder descent with
a **design-utilization** trigger allow the design to mature at each move level
before descending, and thereby remove a substantial part of the
refinement-dependent grayness?

The utilization statistic is

    r_rho(k) = max_e |Δρ_e(k)| / move(k)

evaluated on each **completed** outer iteration `k`, where `move(k)` is the move
limit actually in force at `k` (`hist.move(k)`) and `max|Δρ|` is the realized
design increment (`hist.dxOuter(k)`).

## 2. The two arms — and only two

### ARM P — PRODUCTION

The production preset at the study mesh. Move descent driven by the existing
`move.continuation.signal = 'boundVariable'` stall detector
(window 10, tolerance 5e-3). Production admission (`stop.norm = l2`,
`stop.guards.settledMove = true`, mesh-scaled ε).

### ARM U — UTILIZATION-GATED TRANSITION

Identical in every respect except the **move-stage transition admission**:

    move.transition.metric      = 'maxUtilization'
    move.transition.threshold   = 0.5
    move.transition.persistence = 10

Descent to the next **existing** ladder level is permitted only when
`r_rho < 0.5` on **10 consecutive completed iterations at the current,
unchanged move level**.

Terminal admission for ARM U uses the previously preregistered experimental
rule **`settledLocalObjective`** (admission_rule study, candidate C1), so that
the already-demonstrated production false-admission defect cannot terminate the
candidate artificially:

| component | value | origin |
|---|---|---|
| A — move-settled dwell `D` | 10 | `move.continuation.window` |
| B1 — `max|Δρ| < τ_abs` | 0.01 | Bendsøe & Sigmund 99-line / `top88_reference.m` |
| B2 — `max|Δρ|/move < τ_rel` | 0.50 | the ladder's own halving factor |
| C — rel. range of ω₁ over `W` | `W`=10, `τ_obj`=5e-3 | `move.continuation.window` / `.tolerance` |

**No third scientific arm is authorized.** The fixed-move arm of
`diagnostics/move_stop/` is historical evidence, shown for comparison and used
for integrity verification only; it is not rerun.

## 3. Literal semantics of "10 consecutive"

Implemented literally, as a counter evaluated at the start of outer iteration
`k` over completed iterations `j ≤ k-1`:

* the counter increments only when `r_rho(j) < 0.5` **and** `j` lies strictly
  after the most recent stage change;
* any `r_rho(j) ≥ 0.5` resets the counter to 0;
* a move transition resets the counter to 0;
* descent occurs when and only when the counter reaches 10.

**Explicitly forbidden substitutes:** mean over 10, median over 10, 9-of-10,
endpoint comparison, objective stall, cumulative average, percentile, RMS, any
robust spatial statistic, any exception clause.

## 4. Frozen constants — provenance, not fitting

| Constant | Value | Inherited from |
|---|---|---|
| utilization threshold | **0.5** | `τ_rel` of the preceding admission-rule preregistration: below it the realized step already fits inside the *next* ladder level |
| persistence | **10** | `move.continuation.window` — the solver's own existing timescale |
| ladder levels | `[0.04 0.02 0.01 0.005]` | unchanged production value |
| safety cap `maxOuter` | **600** | the cap of the immediately preceding admission-rule study; not a new number |

`0.5` and `10` are **not to be fitted in this task** under any outcome.

## 5. Why the ratio, not an absolute threshold

In `innerLoop` the sub-problem box is
`lo = max(ρ_min−ρ, −move)`, `hi = min(1−ρ, +move)`, so `max|Δρ| ≤ move`
identically and `r_rho ∈ [0,1]`. An **absolute** threshold on `max|Δρ|` is
therefore manufacturable: halving the move halves the bound on the statistic
with no change in the design's behaviour, and once the ladder reaches 0.005 any
absolute threshold ≥ 0.005 is satisfied automatically.

`r_rho` is dimensionless in the move: when the design saturates its bound,
`max|Δρ| = move` and `r_rho = 1` **whatever the move is**, so a descent cannot
reduce it. `r_rho` measures whether the move bound is *active*, i.e. whether the
design is still using the step it is allowed. This property is asserted
numerically in the study (§6 of the brief) and is **not** a claim that `r_rho`
is the correct final transition statistic — that is the hypothesis under test.

## 6. Meshes

**160×20** and **320×40** only. 400×50 is confirmatory and authorized only under
§10 below. **800×100 is forbidden.**

## 7. Method — and why one run per arm suffices

With `projection.enabled = false` and `material.stiffness.continuation.enabled =
false`, `convOuter` in `olhoffSolve` is computed **after** `olh.move.limit`, is
never written back into the move-controller state or into `hist`, and is read
only by the `settledMove` guard, the (dead) stop guards, the (dead) projection
trigger, the (dead) p-continuation block, and the `break`. **The trajectory is
therefore independent of when convergence is admitted.** This was verified
bitwise in the preceding admission-rule study (§9 anchor gate, both meshes).

Consequently each arm is run **unstopped** (`stop.tolerance = 0`,
`stop.toleranceRule = 'explicit'` — two stopping-policy fields only) to the
safety cap, and the terminal admission rule is evaluated **offline** on the
recorded trajectory. The move transition, which *does* alter the trajectory, is
live in the solver.

### Candidate implementation — default-off, production source untouched

No file under `+impl/` is modified; the source manifest must still hash to
`c1455374d5f8e256…` after the study. The candidate lives in
`diagnostics/move_transition/code/` as:

* `mt_moveLimit.m` — the transition controller. With
  `metric = 'boundVariableStall'` (the **default**) it delegates verbatim to the
  production `olh.move.limit` and is therefore bit-identical to production.
* `mt_olhoffSolveT.m` — a copy of `+impl/architecture/olhoffSolve.m` whose
  **only** difference is that the one line calling `olh.move.limit` calls
  `mt_moveLimit` instead (plus the function name and the extra argument). The
  diff is published in `CONFIG_DIFF.json` and must show exactly that.

The copy is validated by the §8 baseline gate below, not merely asserted.

## 8. BASELINE REGRESSION GATE (blocking)

ARM P is run through the candidate solver copy with the candidate **off** and
must reproduce the archived production trajectory. Required, in order:

* **B-1** per-iteration history over all recorded iterations — `omega1`,
  `omega2`, `gap12`, `volume`, `move`, `stage`, `beta`, `l2`, `maxAbs`,
  `nInner`, `innerConv`, `multN` — **bitwise** equal to
  `diagnostics/admission_rule/runs/unstopped_<mesh>_iterations.csv`;
* **B-2** the reconstructed design at the archived production stop iteration
  (91 at 160×20, 131 at 320×40) **bitwise** equal to the archived
  `diagnostics/move_stop/runs/baseline_<mesh>.mat` final design;
* **B-3** post-loop ω₁ at that design equal to the archived
  169.49522702153845 / 165.95078925220545 **bitwise**;
* **B-4** production move transitions reproduced exactly: 160×20 at
  {79, 90, 101}; 320×40 at {130, 141, 152}.

Failure of any ⇒ **`BASELINE_REPRODUCTION_FAIL`**, stop before interpreting ARM U.

## 9. HARD PRIMARY GATES — frozen

Volume feasible ≡ `|mean(ρ) − 0.5| ≤ 1e-3` (inherited).

**160×20 ARM U**
G1 no solver failure ·
G2 volume feasible at termination ·
G3 ω₁ regression < 3 % vs production ·
G4 topology remains reconstruction-like ·
G5 no move descent occurs unless `r_rho < 0.5` for 10 consecutive iterations ·
G6 no convergence through the old one-iteration-after-descent artifact ·
G7 natural termination **or** an honestly reported `CAP_HIT`.

**320×40 ARM U**
G8 no solver failure ·
G9 volume feasible ·
G10 ω₁ regression < 3 % vs production ·
G11 every move descent satisfies the frozen utilization criterion ·
G12 no old immediate-post-descent convergence artifact ·
G13 **M_nd materially improves vs production** ·
G14 mid-density fraction does not materially worsen ·
G15 topology remains connected/spanning and reconstruction-like.

### "Materially" — fixed here, before any ARM U run

* **G13**: **≥ 25 % relative reduction** in final M_nd versus the production
  final value at the same mesh.
* **G14**: **≤ 10 % relative increase** in mid-density fraction.

**Rationale, and why the number is not chosen now.** 25 % is taken verbatim from
the immediately preceding admission-rule preregistration, where it was fixed and
then *failed* (the candidate delivered 1.49 %). Keeping it identical means this
study cannot be accused of moving the bar. It is also defensible on the
established effect size: production final M_nd at 320×40 is 23.36 and the
historical fixed-move mature value is 13.28, an excess of 10.08 points ≡ 43 % of
production. A 25 % relative reduction is 5.84 points ≡ 58 % of that excess —
i.e. the candidate must recover a **majority** of the established move-schedule
excess to be called a material improvement.

## 10. Safety cap and CAP_HIT

`runtime.maxOuter = 600` for both arms and both meshes. The cap is generous
enough to characterize persistent move-bound utilization: the archived
fixed-move 160×20 arm ran 400 iterations at `move = 0.04`, and 600 provides 50 %
more.

**If ARM U reaches the cap it is `CAP_HIT` and stays `CAP_HIT`.** A stable
objective or topology at the cap is **not** to be reinterpreted as convergence.
The trajectory is nevertheless analysed scientifically.

## 11. Preregistered failure mode — no rescue

If a small number of elements keep `r_rho ≥ 0.5` indefinitely while the bulk
topology matures, so that the 10-consecutive test never fires, the finding is
recorded as

    MAX_UTILIZATION_TRANSITION_TOO_SENSITIVE

and **the candidate is not rescued**: 0.5 is not raised, 10 is not shortened,
`max` is not replaced by a percentile or an RMS or any robust statistic, and no
exception is added. A robust spatial statistic is a **future** task.

## 12. 400×50 authorization

Authorized **only** if ARM U passes at **both** 160×20 and 320×40, the mechanism
is clear, no threshold was changed, and the candidate terminates naturally or
otherwise satisfies the success logic above. If authorized it is run with the
identical frozen candidate, confirmatory only. **800×100 remains forbidden.**

## 13. Verdicts

Exactly one **mechanism** verdict from:
`PREMATURE_MOVE_TRANSITION_CONFIRMED` ·
`PREMATURE_MOVE_TRANSITION_PARTIAL` ·
`PREMATURE_MOVE_TRANSITION_REFUTED` ·
`MAX_UTILIZATION_TRANSITION_TOO_SENSITIVE` ·
`UTILIZATION_GATED_TRANSITION_FAIL` ·
`INCONCLUSIVE`

and exactly one **production** verdict from:
`MOVE_TRANSITION_CANDIDATE_READY_FOR_CONFIRMATION` ·
`KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`

**No production promotion occurs in this task.** The production preset is not
modified; the candidate is default-off.

## 14. Scope lock acknowledged

Unchanged and unexamined: SIMP formulation, p = 3, mass model eq4b, q = 1,
sensitivity filter, R = 0.06·b, multiplicity treatment, fixed 2-mode subspace,
off-diagonal terms, published MMA, move levels, projection (OFF), density
filtering (OFF), FE formulation, eigensolver, objective, volume constraint.

Not done: projection, β = 16, filter-radius change, p-continuation,
mass continuation, move-value change, threshold tuning, second transition rule,
800×100, nine-mesh campaign, production-preset modification.
