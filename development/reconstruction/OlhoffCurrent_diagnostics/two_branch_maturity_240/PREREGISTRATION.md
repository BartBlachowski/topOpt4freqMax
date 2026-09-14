# PREREGISTRATION — two-branch stage-exhaustion hypothesis, withheld 240×30 test

Frozen **before the 240×30 optimization run**, and built **exclusively** from the
existing 160×20 / 320×40 / 400×50 fixed-move evidence. No 240×30 numeric content
of any kind was opened before this file was hashed.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files) |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1 |
| Threads | `maxNumCompThreads(1)`, asserted in the run driver |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |

**No controller is implemented here.** The constants below are diagnostic
classifier parameters for a mechanism test; they are not production thresholds
and nothing is promoted.

---

## 1. The hypothesis under test

> A fixed-move stage has exhausted useful topology evolution when **either**
> **Branch A** (persistent cancellation while amplitude is non-negligible) **or**
> **Branch B** (amplitude convergence to the inherited design-change scale while
> motion stays coherent) fires.

The test is whether this rule, **frozen now**, correctly locates the end of
useful fixed-move topology evolution at 240×30 **with no mesh-specific
retuning**.

## 2. Indexing, window, and shared quantities

Inherited unchanged from `dynamical_regime/PREREGISTRATION.md` §§2–4 and used by
**direct reuse of that study's code** (`dr_telemetry.m`, `dr_dyn.m`), not by
re-typing.

* `RHO(:,k)` is the design after outer iteration `k`; `ρ_0` is the uniform
  initial design; `Δρ_k = ρ_k − ρ_{k−1}`.
* `cosθ(k) = ⟨Δρ_k, Δρ_{k−1}⟩ / (‖Δρ_k‖·‖Δρ_{k−1}‖)`, `NaN` when either norm is 0.
* `net_path(k) = ‖ρ_k − ρ_{k−10}‖ / Σ_{j=k−9}^{k} ‖Δρ_j‖`, window **W_np = 10**
  (inherited, not tuned).
* `‖Δρ_k‖₂` is `hist.dxNorm2`, the plain L2 norm — **the same quantity the
  inherited native stop criterion tests**.
* **Median window `W = 20`** and **persistence `P = 20`**, both inherited from the
  `dynamical_regime` classifier. Medians are trailing: `med(x)(k) = median` over
  `[k−W+1, k]`.

`q2` is **not** used in either branch. Under these definitions
`q2² = 2(1 + cosθ)` where consecutive step norms are equal, so it carries no
information independent of `cosθ`.

## 3. The convergence scale (shared by both branches)

```
tol(NE) = 0.05 * sqrt(NE / 3200)          the inherited meshScaled stop tolerance
```

giving 0.05 (160×20), **0.075 (240×30)**, 0.1 (320×40), 0.125 (400×50).

**Equivalent mesh-normalized form**, verified identical at every iteration on all
three training meshes:

```
||drho||_2 < tol(NE)   <=>   RMS(drho) = ||drho||_2 / sqrt(NE) < 8.838835e-04
```

The constant `8.838835e-04 = 0.05/sqrt(3200)` is **mesh-independent**. No NE
exponent is invented here; the `sqrt(NE)` scaling is inherited from the existing
`stop.toleranceRule = 'meshScaled'`.

## 4. BRANCH A — cancellation / recurrence (frozen)

Fires at the first `k` such that **for 20 consecutive iterations** `j = k … k+19`:

```
med_20 cosθ(j)      <  0.0        persistent directional reversal
AND med_20 net_path(j) < 0.5      the majority of path length is cancelled
AND ||drho_j||_2   >=  tol(NE)    amplitude is NOT negligible
```

*Rationale, from training evidence only.* `cosθ < 0` is the reversal signature.
`net_path < 0.5` is the principled cancellation cut — more than half of the path
travelled is cancelled — not a value fitted to any outcome. The amplitude clause
implements the brief's requirement that cancellation count only "while step
amplitude remains non-negligible", and makes Branch A and Branch B **mutually
exclusive by construction**.

Branch A uses no `M_nd`, no gray/interface weighting, no final topology, and no
240×30 information.

## 5. BRANCH B — amplitude convergence (frozen)

Fires at the first `k` such that **for 20 consecutive iterations** `j = k … k+19`:

```
||drho_j||_2  <  tol(NE)          amplitude has decayed to the inherited scale
AND med_20 cosθ(j) > 0.0          motion is still coherent, not oscillatory
```

**Guard against the known 160×20 pathology.** A full-amplitude bound-pinned cycle
must never be read as convergence. It cannot be: at 160×20 `‖Δρ‖₂ ≈ 0.346`
against `tol = 0.05`, so the amplitude clause alone blocks Branch B there, and
the coherence clause blocks it independently (`med cosθ ≈ −0.95`). Both guards
are in force.

## 6. Relation to the inherited native stop (Phase 4) — stated plainly

**Branch B is the inherited native design-change stop criterion, plus a coherence
guard, plus persistence.** It is *not* a novel signal and is not presented as
one. Precisely:

* native predicate (this configuration: `stop.norm='l2'`, `settledMove=true`,
  other guards off; under a fixed move the settled-move guard is trivially true
  for `k ≥ 2`): `‖Δρ_k‖₂ < tol(NE)`;
* Branch B = that predicate **AND** `med_20 cosθ > 0` **AND** sustained 20
  iterations.

**The coherence guard is load-bearing, verified on training data.** At 320×40 the
native predicate first holds at 216, but `med_20 cosθ` dips to **−0.024** inside
the following persistence window, so Branch B is **blocked** and the union
instead reports Branch A at 255. At 400×50 the guard **allows** Branch B, which
fires at **369 — exactly the native-stop iteration**. At 160×20 the native
predicate never holds at all.

So on the training set Branch B coincides with native stopping wherever it fires,
and the added value of the guard is that it *prevents* native stopping from being
read as maturity at 320×40.

## 7. The union, and event selection

```
EXHAUSTION EVENT = the first iteration at which either branch's
                   20-iteration sustained window BEGINS
```

If both qualify, the earlier begins the event and the branch identity is recorded;
they cannot both hold at the same iteration because the amplitude conditions are
complementary (`≥ tol` vs `< tol`).

Possible outcomes recorded: `A`, `B`, `both` (at different iterations), `neither`.

## 8. Safety cap — 1200 outer iterations

240×30 has `NE = 7200`, between 3200 and 12800. Training events: 83 (160×20),
255 (320×40), 369 (400×50, Branch B). Log-interpolating the Branch-A events on
`NE`: `83 · (255/83)^(ln(7200/3200)/ln(12800/3200)) = 83 · 1.92 ≈ 159`.

**Cap = 1200** — the same absolute cap as both prior fixed-move arms, so all
arms are directly comparable in extent, and ≈ 7.5× the extrapolated event and
3.3× even the latest training event (369). Expected cost ≈ 2400 s
(320×40 took 4133 s for 1200 iterations; `NE` ratio 7200/12800 = 0.56).

**The cap will not be extended after seeing the trajectory.** If neither branch
is classifiable by 1200: `CAP_HIT_BEFORE_TWO_BRANCH_CLASSIFICATION`.

## 9. Confirmation tail

**≥ 400 outer iterations after the event** for full confirmation (160×20 had 517,
320×40 945, 400×50 831). A shorter tail is reported as *persistence not fully
confirmed*, with the achieved length stated.

## 10. Useful-evolution metric — inherited, not replaced

```
remUseful = ( Mnd(k_event) − Mnd(kEnd) ) / ( Mnd(1) − Mnd(kEnd) )
```

exactly the definition used in `dynamical_regime` and `fixedmove_400_dynamics`.
Reported alongside, as a secondary and explicitly non-decisive figure:

```
postRelImp = 100 * ( Mnd(k_event) − min_{j>=k_event} Mnd(j) ) / Mnd(k_event)
```

`ω₁`, gray fraction, mid-density fraction, volume and density-field distance at
the event and at the end are also reported.

## 11. Training-set behaviour of the frozen rule — recorded now, for audit

| mesh | Branch A | Branch B | union event | M_nd at event | `remUseful` | `postRelImp` |
|---|---|---|---|---|---|---|
| 160×20 | **83** | never (amplitude) | **83** | 13.2333 | **2.017 %** | 14.19 % |
| 320×40 | **255** | never (**blocked by coherence guard**) | **255** | 13.0241 | **−0.581 %** | 0.16 % |
| 400×50 | never | **369** (= native stop) | **369** | 16.1589 | **1.107 %** | 6.87 % |

The rule is *not* perfect even in training: at 160×20 it fires while a 14.19 %
relative `M_nd` improvement still lies ahead. That weakness is recorded here,
before the withheld test, and is not to be explained away afterwards.

## 12. Preregistered prediction for 240×30

Stated now to make the test sharp and falsifiable:

* **branch identity: A** (240×30 lies between two Branch-A meshes);
* **event iteration ≈ 159**, and in any case within `[100, 400]`;
* fixed-move endpoint resembling 160×20/320×40 — a cancelling regime, with
  terminal `med cosθ < 0` and terminal `max|Δρ|/move` between 0.70 and 1.00.

A confirmed hypothesis with a *wrong* branch prediction is still a pass; the
prediction exists to expose post-hoc rationalisation.

## 13. Pass / fail criteria (frozen)

**PASS (all required):**

* **P1** exactly one branch fires first, and it is classified by the frozen rule
  with no retuning;
* **P2** `|remUseful|` at the event **≤ 5.0 %** (training max 2.02 %);
* **P3** `postRelImp` at the event **≤ 25 %** (training max 14.19 %; the margin
  makes this a genuine out-of-sample test rather than an automatic pass);
* **P4** confirmation tail ≥ 400 iterations, over which `M_nd` does not improve
  by more than `postRelImp`'s bound and the trajectory does not leave the branch
  regime;
* **P5** the event is not produced by a small pathological subset: for Branch A,
  the sustained window's median `cosθ` must remain `< 0` after removing
  bound-saturated elements (`cosθ_unsat`), or `boundFrac < 0.10` at the event;
  for Branch B, `boundFrac < 0.10` at the event;
* **P6** no third dynamical endpoint appears — the terminal state must be either
  cancelling (`terminal med cosθ < 0`) or converged
  (`terminal ‖Δρ‖₂ < tol` with `med cosθ > 0`).

**Verdict mapping:**

* `TWO_BRANCH_MATURITY_HYPOTHESIS_CONFIRMED` — P1–P6 all hold.
* `..._PARTIAL` — P1 and P2 hold, but one of P3–P5 fails while P6 holds.
* `..._REFUTED` — P2 fails (a branch fires with substantial useful evolution
  remaining), **or** P6 fails (a third regime), **or** neither branch fires while
  the design is clearly mature (`M_nd` plateaued and `remUseful` small at the cap),
  **or** a branch is shown to be a false positive under P5.
* `..._INCONCLUSIVE` — `CAP_HIT_BEFORE_TWO_BRANCH_CLASSIFICATION` with the design
  demonstrably still evolving, or a provenance failure invalidates interpretation.

**Next-step mapping:** `TWO_BRANCH_CONTROLLER_PREREGISTRATION_JUSTIFIED` requires
CONFIRMED **and** all seven Phase-16 conditions. `RECONSIDER_MOVE_LADDER_ARCHITECTURE`
if REFUTED. Otherwise `MORE_MATURITY_MECHANISM_EVIDENCE_REQUIRED`.

**If the hypothesis fails, Branch C will not be invented** (Phase 17).

## 14. Run configuration

```
cfg = olh.config.resolve('duOlhoffFrozenM4', ...
        'domain.mesh.nelx', 240, 'domain.mesh.nely', 30, ...
        'runtime.maxOuter', 1200, 'runtime.singleThread', true, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'move.policy', 'fixed', 'move.initial', 0.04, ...      % experimental factor
        'stop.tolerance', 0, 'stop.toleranceRule', 'explicit')  % minimal stop override
```

The same minimal stop-override architecture validated in `dynamical_regime`
RUN B and `fixedmove_400_dynamics` RUN C. It prevents native termination from
truncating the observation and does nothing else; the native predicate is
replayed offline and the iteration it *would* have fired is reported. MMA, the
move limit, sensitivities, the objective, the constraints and the admission
arithmetic are untouched. **No solver copy**; `olhoffSolve` is called unmodified.

Scope lock asserted in code before the solve: `p = 3` · `eq4b` · `q = 1` ·
sensitivity filter · `R = 0.06·b` · projection off · published MMA · subspace
multiplicity size 2, off-diagonals on · `move.levels = [0.04 0.02 0.01 0.005]`.
The resolved config is diffed field-for-field against
`olhoffcurrent_config(240, 30, 'MaxOuter', 1200, 'Diagnostics', true)`; anything
differing outside the declared override list aborts.

## 15. Out-of-sample discipline (Phase 15)

Everything used to choose the frozen rules is listed in §§3–7 and §11 and comes
**only** from 160×20, 320×40 and 400×50. Before this file was hashed, the only
240×30 information observed was the *existence of filenames* in a repository-wide
`find` for `*240x30*`, and the resolved production `stop.tolerance` (0.075),
which is a deterministic function of the mesh. **No 240×30 numeric result was
opened.** 240×30 is a withheld mechanism mesh, not a statistically independent
sample, and no such claim is made.

Any 240×30 production trajectory found later (Phase 8) is inventoried and used
only for reference; **it does not authorise a second scientific run**, and none
will be made.

## 16. What this task will not do

No controller; no causal validation; no production change; no Branch C; no
retuning of any threshold, window or persistence after the run; no gray or
interface weighting; no invented NE exponent; no run of 160×20, 320×40, 400×50,
800×100 or any other mesh; no nine-mesh campaign; no projection; no change to
filter type, `R`, `p`, mass interpolation, `q`, multiplicity, MMA, FE
formulation, eigensolver, objective or volume constraint. Production remains
`KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`; the campaign remains
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.
