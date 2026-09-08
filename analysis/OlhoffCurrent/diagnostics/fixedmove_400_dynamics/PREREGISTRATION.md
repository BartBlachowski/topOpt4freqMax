# PREREGISTRATION — 400×50 fixed-move dynamics

Frozen **before the single authorized optimization run**. The run, its cap, the
stopping override, every dynamical definition, the regime classifier, the
useful-evolution measure, the confirmation-tail rule and all verdict criteria
are fixed here and are not revisited after results are visible.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files) |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1 |
| Threads | `maxNumCompThreads(1)`, asserted in the run driver |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |

**No controller is implemented, and no production threshold is chosen.** The
classifier constants in §5 are diagnostic-labelling parameters carried over
unchanged from the preceding study; they are explicitly not controller
parameters and are not tuned here.

---

## 1. The single authorized run — RUN C

```
cfg = olh.config.resolve('duOlhoffFrozenM4', ...
        'domain.mesh.nelx', 400, 'domain.mesh.nely', 50, ...
        'runtime.maxOuter', 1200, 'runtime.singleThread', true, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'move.policy', 'fixed', 'move.initial', 0.04, ...      % experimental factor
        'stop.tolerance', 0, 'stop.toleranceRule', 'explicit')  % minimal stop override
```

This is the canonical production scientific formulation with exactly two
conceptual changes, both required by the mission:

* **`move.policy = 'fixed'`, `move.initial = 0.04`** — the mechanism-measurement
  arm. Identical to the override pair used by `move_stop`'s fixed-move arm and
  by `dynamical_regime` RUN B.
* **`stop.tolerance = 0`, `stop.toleranceRule = 'explicit'`** — the **minimal**
  stop override. It prevents native termination from truncating the mechanism
  measurement and does nothing else. MMA, the move limit, sensitivities, the
  objective, the constraints and the admission arithmetic are all untouched.

**Justification for the override.** At 320×40 the native rule admitted at
iteration **216** while the dynamical onset was at **253** — native stopping
demonstrably truncates the measurement. At 400×50 the native rule is expected to
admit even earlier relative to onset, since production already converged at 139.

**No information is lost.** The `admission_rule` study verified bitwise that,
with projection and p-continuation off, the outer convergence test is computed
after `olh.move.limit`, is never written back into `hist` or the move-controller
state, and only drives the `break`. The trajectory is therefore independent of
when convergence is admitted.

### The inherited native-stop predicate, replayed offline

For this configuration (`stop.norm = 'l2'`, `guards.settledMove = true`,
`ladderExhausted = false`, `maxDesignChange = false`) the native predicate is

```
nativeStop(k)  =  ( ||drho_k||_2 < tol )  AND  ( k >= 2  AND  move(k) == move(k-1) )
tol = production stop.tolerance at 400x50 = 0.125   (meshScaled: 0.05*sqrt(NE/3200))
```

Under a fixed move the settled-move guard is always satisfied for `k >= 2`.

**This replay was validated before freezing**, against two known answers:

| archive | replay predicts | archive records | match |
|---|---|---|---|
| `move_stop/runs/fixedmove_320x40.mat` (tol 0.1) | **216** | `NATIVE_CONVERGED`, 216 outer | ✔ |
| `dynamical_regime/runs/runA_400x50.mat` (tol 0.125) | **139** | `CONVERGED`, 139 outer | ✔ |

The iteration at which native stopping *would* have occurred is reported.

### Scope lock, asserted in code before the solve

`p = 3` · `eq4b` · `q = 1` · sensitivity filter · `R = 0.06·b` ·
projection off · published MMA · subspace multiplicity size 2, off-diagonals on ·
`move.levels = [0.04 0.02 0.01 0.005]`. Any mismatch aborts. The resolved config
is additionally diffed field-for-field over the whole schema against
`olhoffcurrent_config(400, 50, 'MaxOuter', 1200, 'Diagnostics', true)`; anything
differing outside the declared override list aborts.

**No solver copy.** The run calls `+impl/architecture/olhoffSolve.m` unmodified
through the canonical configuration route.

---

## 2. Safety cap — 1200 outer iterations

Fixed before the run, from the 320×40 evidence and **not** assuming linear NE
scaling.

Observed onsets: 160×20 (NE 3200) → **81**; 320×40 (NE 12800) → **253**. That is
a factor 3.12 in onset for a factor 4 in NE, i.e. an empirical exponent
α = ln 3.12 / ln 4 = **0.821**. Extrapolating to 400×50 (NE 20000, a factor
1.5625 over 320×40) under a range of assumptions:

| assumption | predicted onset |
|---|---|
| observed power law, α = 0.821 | **364** |
| linear in `nelx` (α ≈ 0.5) | 316 |
| linear in NE (α = 1) | **395** |
| pessimistic α = 1.5 | 494 |
| very pessimistic α = 2 | **618** |

**Cap = 1200** is ≥ 1.94× even the α = 2 extrapolation, 3.0× the linear-NE
extrapolation, and 3.3× the observed-power-law extrapolation. It is also the
*same absolute cap* used for the 320×40 fixed-move arm, so the two arms are
directly comparable in extent. Expected cost ≈ 5.4 s/iteration (from RUN B's
3.44 s/iter at 320×40 scaled by NE), i.e. ≈ 1.8 h single-threaded.

**The cap will not be extended after seeing the trajectory.** If no onset is
classified by 1200 the result is reported as
`CAP_HIT_BEFORE_DYNAMICAL_CLASSIFICATION`.

---

## 3. Raw evidence retained (Phase 4)

The full elementwise density trajectory `RHO` (20000 × nOuter), the resolved
`cfg`, and per-iteration telemetry: `rho(k)`, `Δρ(k)`, `omega1`, `omega2`,
`gap12`, `volume`, `M_nd`, gray fraction, mid-density fraction, `move`, `beta`,
the β-stall metric, the **inherited native-stop predicate**, inner MMA
iterations and status, `max|Δρ|`, RMS `Δρ`, L2 `Δρ`, and bound-active
count/fraction.

Density is reconstructed from `res.diag.drho{k}` exactly as `olhoffSolve` forms
it, validated against `hist.vol` to `< 1e-12`, with the final column asserted
identical to `res.rho`. A reconstruction failure aborts the run.

Every artifact is hashed into `FINAL_SHA256.txt` and listed in
`DATA_MANIFEST.json` with byte size and role. **No required raw trajectory may
exist only in an ignored/unmanifested location**, and `FINAL_SHA256.txt` is
re-verified after all cleanup. No scratch artifact is citable as evidence.

---

## 4. Dynamical definitions — inherited unchanged (Phase 5)

The definitions frozen in `dynamical_regime/PREREGISTRATION.md` §§2–4 are used
**by direct reuse of that study's code**, not by copying formulas:
`dr_telemetry.m`, `dr_dyn.m`, `dr_classify.m`, `dr_spatial.m` are called by
reference from `analysis/OlhoffCurrent/diagnostics/dynamical_regime/scripts/`
and their SHA-256 values are recorded as input dependencies. No definition is
redefined after seeing 400×50.

Restated for the reader (`NE = 20000`, `ρ_0` = uniform initial design, a real
state; `Δρ_k = ρ_k − ρ_{k−1}` for `k ≥ 1`):

| quantity | definition |
|---|---|
| primary norm | `‖v‖ = ‖v‖₂/√NE`; secondary `‖v‖₁/NE` |
| `d1(k)` | `‖ρ_k − ρ_{k−1}‖`, k ≥ 2 |
| `d2(k)` | `‖ρ_k − ρ_{k−2}‖`, k ≥ 3 |
| `q2(k)` | `d2(k)/d1(k)` where `d1 > 0` |
| `cosθ(k)` | `⟨Δρ_k, Δρ_{k−1}⟩ / (‖Δρ_k‖·‖Δρ_{k−1}‖)`, both norms > 0 |
| `net_path(k)` | `‖ρ_k − ρ_{k−W}‖ / Σ_{j=k−W+1}^{k} ‖Δρ_j‖`, **W = 10** |
| `cancellation(k)` | `1 − net_path(k)` |
| `boundFrac(k)` | fraction with `|Δρ_k,e| > 0.9·move(k)` |
| `revFrac(k)` | fraction with `Δρ_k,e · Δρ_{k−1},e < 0` |

`W = 10` is inherited and **not tuned**. Undefined values are recorded as `NaN`
and counted; a zero-norm step is never silently mapped to any number.

**Declared non-independence.** Where consecutive step norms are equal,
`q2² = 2(1 + cosθ)` exactly. `q2` and `cosθ` are **not** independent
confirmation. `net_path` at W = 10 is a genuinely distinct 10-step statistic.

### Unsaturated mask (Phase 6) — fixed before the run

Carried over unchanged from the preceding study's implementation:

* for `cosθ_unsat(k)`: exclude elements with
  `|Δρ_k,e| > 0.9·move(k)` **or** `|Δρ_{k−1},e| > 0.9·move(k−1)`;
* for `net_path_unsat(k)`: exclude elements with
  `|Δρ_j,e| > 0.9·max_j move(j)` for any `j` in the W-window.

Both raw and unsaturated versions are reported wherever defined. **Neither is
selected on the basis of which gives the more favourable answer**; both appear
in every headline table.

---

## 5. Regime classification and persistence (Phase 7)

Inherited unchanged. Trailing-window medians over **P = 20** outer iterations:

```
L(k) = PERIOD2    if median q2 < 1.0  AND median cosθ < 0.00
     = COHERENT   if median q2 > 1.5  AND median cosθ > 0.25
     = OTHER      otherwise
```

**Onset** of a label is the first `k` for which that label holds for **20
consecutive** iterations. A single negative `cosθ`, or a single low `net_path`,
can never establish a regime. Classification boundaries are **not** tuned using
`M_nd` or any other outcome.

**Reporting refinement (labelling only, no change to the classifier):** an
`OTHER` stretch lying after the last `COHERENT`-labelled iteration and before
`PERIOD2` onset is reported descriptively as **TRANSITIONAL**.

---

## 6. Useful-evolution measure (Phase 9) — inherited unchanged

```
remaining_Mnd = ( Mnd(k*) − Mnd(kEnd) ) / ( Mnd(1) − Mnd(kEnd) )
remaining_L1  = ‖ρ_kEnd − ρ_k*‖₁ / ‖ρ_kEnd − ρ_1‖₁
```

with `k*` the onset and `kEnd` the run's terminal iteration. `ω₁`, gray
fraction, mid-density fraction and volume at `k*` and `kEnd` are reported
alongside. A slight post-onset improvement or worsening is quantified, not
required to be zero: the hypothesis predicts onset *near the end of useful
evolution*, not that every scalar freezes.

---

## 7. Confirmation tail (Phase 12) — fixed before the run

The run proceeds in one batch to the cap; adaptive early termination is not used
because it would require modifying the solver, which is forbidden.

* **Full confirmation** requires **≥ 400 outer iterations after onset**
  (160×20 had 519, 320×40 had 947).
* If the tail is < 400 iterations the regime is reported as
  **persistence not fully confirmed**, with the achieved tail length stated.
* Over the tail, report whether `M_nd` improves, stagnates or worsens, whether
  `ω₁` improves materially, and whether the trajectory remains cancelling.

---

## 8. Production counterfactual and common prefix (Phase 10)

RUN C is compared against the retained `dynamical_regime/runs/runA_400x50.mat`
(400×50 production, 139 outer, first descent at 138).

Production holds `move = 0.04` through iteration **137** and the descent takes
effect **at** iteration 138. The design update at 138 therefore uses a different
move in the two arms, so:

* **bitwise identity is required over `k = 1 … 137`** for ρ, ω₁, ω₂, `M_nd`,
  volume, `beta` and event structure;
* **iteration 138 is expected to be the first differing iteration**, and this is
  verified rather than assumed.

Both arms were produced under the same MATLAB build, so bitwise identity is the
standard applied. Failure of that standard means the counterfactual is invalid
and no causal claim is made from it; the run is reported without Phase 11
attribution.

---

## 9. Preregistered verdict rules (Phases 21, 22)

### Mechanism criteria

* **C1** a `PERIOD2` onset `k*` is classified at `k* ≤ 1200`;
* **C2** `k* > 138`, i.e. onset falls strictly after the production descent;
* **C3** the classifier label at iteration 138 is `COHERENT`;
* **C4** `|remaining_Mnd|` at onset **≤ 5.0 %** (observed: 2.08 % at 160×20,
  −0.58 % at 320×40);
* **C5** confirmation tail ≥ 400 iterations, over which `M_nd` does not improve
  by more than **5 % relative** to its onset value;
* **C6** onset is not a saturation artifact: `boundFrac` at onset **< 0.10**
  **and** the median `cosθ` over `[k*, k*+20]` remains **negative** after
  removing saturated elements.

**Verdicts**

* `CROSS_MESH_DYNAMICAL_MATURITY_MECHANISM_CONFIRMED` — **C1–C6 all hold**, so
  400×50 independently exhibits coherent → cancelling → no-useful-work.
* `..._PARTIAL` — C1, C2 and C3 hold, but one or more of C4, C5, C6 fails.
* `..._REFUTED` — C1 holds but onset precedes the production descent (C2 fails),
  **or** `|remaining_Mnd|` at onset exceeds 20 %, i.e. cancellation begins while
  substantial useful evolution remains.
* `..._INCONCLUSIVE` — C1 fails (`CAP_HIT_BEFORE_DYNAMICAL_CLASSIFICATION`), or a
  provenance/prefix failure invalidates interpretation.

Numerical `cosθ` and `net_path` values at onset are **not** required to match
across meshes: the mechanism, not a universal threshold, is under test.

### Next-step rule

`DYNAMICAL_CONTROLLER_PREREGISTRATION_JUSTIFIED` requires **all seven** Phase-18
conditions. Otherwise `MORE_DYNAMICAL_EVIDENCE_REQUIRED`, or
`DYNAMICAL_CONTROLLER_ROUTE_ABANDON` if the mechanism is refuted at 400×50.

---

## 10. Interpretation rules

* All diagnostics are computed **post hoc**; nothing is fed back into the solver.
* `q2` evaluated *at* a move-change iteration is contaminated by the step-size
  change (its denominator shrinks with the move) and is not read raw there; the
  trailing-window median is the reading used.
* `q2`/`cosθ` agreement is not corroboration (§4). `net_path` agreement is.
* Spatial information (Phase 16) is descriptive only and is **not** converted
  into a controller. The gray/intermediate region is used as a diagnostic
  overlay only — no gray weighting enters any statistic.
* Previously unsourced claims (`M_nd = 16.16 %`, `ω₁ + 2.14 %`) are **not**
  assumed. They are tested against this run and reported as reproduced,
  contradicted, or still unsupported, whichever the measurement shows.
* A negative, partial or inconclusive outcome is reported as such.

## 11. What this task will not do

No controller; no production change; no threshold, persistence or `W` tuning; no
gray or active-set weighting as a controller; no run of 160×20, 320×40, 800×100
or any further mesh; no second fixed-move arm; no nine-mesh campaign; no
projection; no change to filter type, `R`, `p`, mass interpolation, `q`,
multiplicity, MMA variant, FE formulation, eigensolver, objective or volume
constraint. Production remains `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`; the
campaign remains `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` regardless of outcome.
