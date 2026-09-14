# PREREGISTRATION — dynamical-regime evidence generation

Frozen **before either scientific run**. Everything below — the two runs, their
caps, every dynamical definition, every indexing convention, the regime
classifier and its persistence requirement — is fixed here and is not revisited
after results are visible.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files) |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1 |
| Threads | `maxNumCompThreads(1)` asserted inside the run driver |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |

**This task implements no controller.** No transition rule is defined, no
threshold is chosen, no persistence value is proposed for production. The
persistence constant `P` fixed in §5 is a *diagnostic classifier* parameter for
labelling regimes in this report only, and is explicitly not a controller
parameter.

---

## 1. The two authorized runs — and nothing else

### RUN A — 400×50 production

```
cfg = olhoffcurrent_config(400, 50, 'MaxOuter', 400, 'Diagnostics', true)
```

Nothing else. The canonical production entry point, at the canonical production
preset, with the production move ladder, the production β-stall transition
signal and the **production admission/stopping rule active**. `Diagnostics` is
the additive per-iteration recorder (it feeds nothing back into the solver);
`MaxOuter = 400` is the production default (`stop`/`runtime` schema default), and
is deliberately **not a new number**.

*Purpose:* obtain the missing 400×50 production trajectory and determine the
dynamical state at each of its move descents.

*If it reaches 400 outer iterations it is `CAP_HIT`, reported as `CAP_HIT`.*

### RUN B — 320×40 extended fixed move

```
cfg = olh.config.resolve('duOlhoffFrozenM4', ...
        'domain.mesh.nelx', 320, 'domain.mesh.nely', 40, ...
        'runtime.maxOuter', 1200, 'runtime.singleThread', true, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'move.policy', 'fixed', 'move.initial', 0.04, ...      % experimental factor
        'stop.tolerance', 0, 'stop.toleranceRule', 'explicit')  % stopping policy
```

**Cap: 1200 outer iterations.** Generous by construction — 5.6× the 216 the
previous fixed-move arm reached, and 2× the 600 already explored by ARM U.

**Overrides, and why exactly these:**

* `move.policy = 'fixed'`, `move.initial = 0.04` — the experimental factor. The
  move never descends, so the trajectory stays in the `move = 0.04` regime for
  the whole run. This is the same override pair the retained
  `move_stop` fixed-move arm used.
* `stop.tolerance = 0`, `stop.toleranceRule = 'explicit'` — **run unstopped**,
  the same two stopping-policy fields the `move_transition` study used. Phase C
  forbids terminating merely because objective improvement is small; the
  inherited production stop rule *is* an objective/L2-stall admission, and the
  `admission_rule` study established that it fires prematurely. Suppressing it
  is therefore required to reach a regime classification at all.

**No information is lost by running unstopped.** The `admission_rule` study
verified *bitwise* that, with projection and p-continuation off, the outer
convergence test is computed after `olh.move.limit`, is never written back into
`hist` or into the move-controller state, and only drives the `break` — so the
trajectory is independent of when convergence is admitted. Accordingly the
inherited production stop criterion is **evaluated offline at every iteration**
and the exact iteration at which it *would* have admitted is reported. Should
that bitwise-independence premise fail the Phase D prefix checks, the offline
admission replay is void and is reported as such.

### Scope lock, asserted in code before each run

`material.stiffness.p = 3` · `material.mass.model = 'eq4b'` · `material.mass.q = 1`
· `filter.type = 'sensitivity'` · `filter.radiusPhysical = 0.06` ·
`projection.enabled = false` · `optimizer.inner.variant = 'published'` ·
`multiplicity.method = 'subspace'` · `multiplicity.subspaceSize = 2` ·
`multiplicity.offDiagonal = true` · `move.levels = [0.04 0.02 0.01 0.005]`.

The driver asserts every one of these and aborts on any mismatch. It also
diffs the resolved config field-for-field against
`olhoffcurrent_config(nelx, nely, 'MaxOuter', cap, 'Diagnostics', true)` and
aborts if anything differs outside the declared override list. **No solver copy
is made**: both runs go through the canonical `olhoffSolve` unmodified, because
neither run needs a non-schema control path.

---

## 2. Indexing conventions

* `RHO(:,k)` is the design **after** outer iteration `k`, for `k = 1 … nO`.
* `rho_0` is the uniform initial design `design.initial` (a real state, not padding).
* `Δρ_k = ρ_k − ρ_{k−1}`, defined for `k ≥ 1` (using `ρ_0` at `k = 1`).
* Per-iteration density is reconstructed from `res.diag.drho{k}` exactly as
  `olhoffSolve` forms it, validated against `hist.vol` to `< 1e-12`, and the
  final reconstructed column asserted identical to `res.rho`. A reconstruction
  failure aborts the run.
* All iteration numbers in the report are 1-based outer-iteration indices.

## 3. Norms

Primary norm, used for every headline number:

```
||v|| = ||v||_2 / sqrt(NE)          (RMS per element)
```

Secondary robustness norm, computed and reported alongside:

```
||v||_1 / NE                        (mean absolute per element)
```

The *same* norm is used consistently within any single derived quantity. Ratios
(`q2`, `net_ratio`) are invariant to the `1/sqrt(NE)` and `1/NE` scalings, which
is the point: **no NE exponent is fitted anywhere.**

## 4. Dynamical quantities

| quantity | definition | defined for |
|---|---|---|
| `d1(k)` | ‖ρ_k − ρ_{k−1}‖ | k ≥ 2 |
| `d2(k)` | ‖ρ_k − ρ_{k−2}‖ | k ≥ 3 |
| **`q2(k)`** | `d2(k) / d1(k)` | k ≥ 3, `d1(k) > 0` |
| **`cosθ(k)`** | ⟨Δρ_k, Δρ_{k−1}⟩ / (‖Δρ_k‖·‖Δρ_{k−1}‖) | k ≥ 2, both norms > 0 |
| `cos2(k)` | ⟨Δρ_k, Δρ_{k−2}⟩ / (‖Δρ_k‖·‖Δρ_{k−2}‖) | k ≥ 3, both norms > 0 |
| `path_W(k)` | Σ_{j=k−W+1}^{k} ‖Δρ_j‖ | k ≥ W |
| `net_W(k)` | ‖ρ_k − ρ_{k−W}‖ | k ≥ W |
| **`net_ratio(k)`** | `net_W(k) / path_W(k)` ∈ [0,1] | k ≥ W, `path_W > 0` |
| `cancellation(k)` | `1 − net_ratio(k)` | as above |
| `r2(k)`, `r4(k)` | ‖ρ_k − ρ_{k−2}‖, ‖ρ_k − ρ_{k−4}‖ | k ≥ 3, k ≥ 5 |
| `boundFrac(k)` | fraction of elements with \|Δρ_k,e\| > 0.9·move(k) | k ≥ 1 |
| `revFrac(k)` | fraction of elements with Δρ_k,e · Δρ_{k−1},e < 0 | k ≥ 2 |

**W = 10**, inherited from `move.continuation.window`. It is **not tuned**, and
no alternative W is used to improve separation.

**Undefined values are recorded as `NaN` and counted.** A zero-norm step is never
silently mapped to 0 or to any other number; the count of undefined `cosθ` and
`q2` samples is reported per run.

### Interpretation, fixed in advance

`q2 ≈ 2` coherent continuation · `q2 ≈ √2` reorientation · `q2 ≪ 1` cancellation.
`cosθ → +1` coherent · `→ 0` reorientation · `→ −1` reversal.
`net_ratio → 1` all motion is net progress · `→ 0` motion cancels.

**Declared dependency, so the report does not double-count:** when consecutive
step norms are equal, `q2² = 2(1 + cosθ)` exactly. `q2` and `cosθ` are therefore
**not independent evidence**; they differ only through the ratio of consecutive
step magnitudes. `net_ratio` at W = 10 is a genuinely different statistic
(10-step, not 2-step). This is stated now so that agreement between `q2` and
`cosθ` is not later reported as corroboration.

## 5. Regime classifier and persistence

Trailing-window medians over `P = 20` outer iterations (`P` fixed here, before
any run; 20 is the most conservative of the {5, 10, 20} persistence values
already used in the preceding study):

```
L(k) = PERIOD2    if median_{[k-P+1,k]} q2 < 1    AND median_{[k-P+1,k]} cosθ < 0
     = COHERENT   if median_{[k-P+1,k]} q2 > 1.5  AND median_{[k-P+1,k]} cosθ > 0.25
     = OTHER      otherwise
```

**Onset of a regime** is the first `k` such that `L(j)` holds that same label for
**all** `j ∈ [k, k + P − 1]` — i.e. sustained for `P = 20` consecutive
iterations. A single dip of `q2` is never called a limit cycle.

A run is classified `PERIOD2_REGIME_OBSERVED`, `COHERENT_DESCENT_PERSISTS`,
`OTHER_DYNAMICAL_REGIME`, or `CAP_HIT_BEFORE_CLASSIFICATION` (the cap was reached
with no sustained label established).

## 6. Mature-state comparison rules

For any onset iteration `k*`, remaining evolution is reported on two bases,
both relative to the run's own terminal state `kEnd`:

```
remaining_Mnd  = ( Mnd(k*)  − Mnd(kEnd)  ) / ( Mnd(1)  − Mnd(kEnd)  )
remaining_L1   = ‖ρ_{kEnd} − ρ_{k*}‖_1 / ‖ρ_{kEnd} − ρ_1‖_1
```

`ω₁`, gray fraction, mid-density fraction and volume at `k*` and at `kEnd` are
reported alongside. Cross-mesh comparison is made **only at matched events**
(§Phase J): start of the `move = 0.04` stage, first β-stall, first move descent,
period-2 onset if observed, fixed-move mature endpoint, final production stop.

No 400×50 quantity from any earlier brief is used. Only events observed in RUN A
are reported for 400×50.

## 7. Prefix reproduction requirement (Phase D)

RUN B must agree with the retained fixed-move evidence over the common prefix:

* vs `move_stop/runs/fixedmove_320x40.mat` (216 iterations, production stop rule)
  — **bitwise preferred** over `k = 1 … 216`;
* vs `move_transition/runs/armU_320x40.mat` (600 iterations, unstopped, its own
  rule descends at 214) — **bitwise preferred** over `k = 1 … 213`, the span on
  which ARM U held `move = 0.04`.

Both archives were produced under **the same MATLAB build** (R2025b Update 1) as
this task, so bitwise identity is expected and is the standard applied. Should it
nevertheless fail, the fallback standard — identical event structure, and ρ, ω,
`M_nd`, volume agreement at archived precision with no unexplained divergence —
is applied and the deviation documented. Failure of both standards yields
`FIXEDMOVE_PREFIX_REPRODUCTION_FAIL` and interpretation stops.

**RUN A has no historical raw production trajectory to reproduce.** No bitwise
claim of any kind will be made for it.

## 8. Raw evidence retention

Both runs save the **full** `RHO` matrix, the resolved `cfg`, and the full
telemetry to `runs/`. Every such file is hashed into `FINAL_SHA256.txt` and
listed in `DATA_MANIFEST.json` with its byte size and role. The preceding study
found that `move_transition`'s own `armP/armU` `.mat` files were gitignored *and*
absent from its manifest; that failure is not repeated here. **No required raw
trajectory may exist only in an unmanifested ignored path.**

## 9. Interpretation rules

* Diagnostics are computed **post hoc** from recorded trajectories. Nothing is
  fed back into any solver.
* No threshold in §5 is proposed for, or evaluated as, a production controller.
* `q2`/`cosθ` agreement is not corroboration (§4).
* Robustness to isolated bound-touching elements is assessed by recomputing every
  headline diagnostic with the bound-saturated element set removed, and by
  reporting `boundFrac` alongside. A diagnostic whose value is produced by a
  small saturated minority is reported as such.
* Mesh-consistency is assessed at matched events across 160×20 (retained),
  320×40 (retained + RUN B) and 400×50 (RUN A) **without any NE exponent**.
* A negative or inconclusive outcome is reported as such. No second candidate
  mechanism is invented inside this task.

## 10. What this task will not do

No controller; no change to production; no tuning of `q2`, `cosθ` or `net_ratio`
thresholds; no change to `W`; no 160×20 optimization run; no 800×100; no
nine-mesh campaign; no projection; no change to filter type, `R`, `p`, or the
mass law. Production remains `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW` and the
campaign remains `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.
