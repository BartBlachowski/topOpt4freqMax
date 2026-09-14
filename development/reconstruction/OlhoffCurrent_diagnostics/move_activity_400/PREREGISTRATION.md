# PREREGISTRATION — 400x50 third-mesh activity measurement

**Frozen before the first 400x50 optimisation run.** Its SHA-256 is recorded in
`PROVENANCE.md` and `FINAL_SHA256.txt`. Nothing below was chosen after seeing a
400x50 trajectory.

## 1. Scientific question

Not "can we make the topology less gray". The question is:

> At the first production move descent, how much topology evolution remains,
> what is the spatial distribution of element activity, and how does the active
> population scale relative to the existing 160x20 and 320x40 measurements?

Specifically: does `N_active ~ NE^alpha` with an approximately stable `alpha`
survive a third mesh, or was the two-mesh power law a coincidence? The prior
offline study obtained `alpha ~= 0.96` at one diagnostic threshold and
`alpha ~= 0.81` at another, from two meshes and therefore zero degrees of
freedom. A third mesh is the first opportunity for the model to fail.

**This is not a controller experiment.** No transition rule is implemented,
tested, tuned or promoted. Production move policy is untouched.

## 2. Mesh, configuration, provenance

Mesh **400x50**, `NE = 20000`. Resolved through the production entry point
`olhoffcurrent_config(400, 50, ...)`. Scientific baseline, verified before
freezing this document:

| field | value |
|---|---|
| `material.stiffness.p` | 3 (fixed) |
| `material.mass.model` / `.q` | `eq4b` / 1 |
| `filter.type` / `filter.radiusPhysical` | `sensitivity` / 0.06 |
| `projection.enabled` | `false` |
| `multiplicity.method` / `.subspaceSize` / `.offDiagonal` | `subspace` / 2 / `true` |
| `optimizer.inner.variant` | `published` |
| `move.policy` / `move.levels` | `ladder` / `[0.04 0.02 0.01 0.005]` |
| `stop.tolerance` / `.toleranceRule` | 0.125 / `meshScaled` |
| `design.initial` / `design.minimum` | 0.5 / 0.001 |
| config hash (400x50, production) | `044d50a496cb64d43ed4bf75a6a7976a6726cca6feaf65e0b911a57e4593969c` |
| `+impl` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74/74) |
| production preset | `duOlhoffFixedPenaltySensitivityFiltered` (upstream `duOlhoffFrozenM4`) |

`stop.tolerance = 0.05*sqrt(NE/3200) = 0.125`, so
`epsRMS = tolerance/sqrt(NE) = 8.83883476483184e-4` — **identical at 160x20,
320x40 and 400x50**. The diagnostic thresholds are therefore directly
comparable across all three meshes without rescaling.

Single-threaded (`maxNumCompThreads(1)`, `runtime.singleThread = true`) and
`runtime.diagnostics = true`, matching every prior study.

## 3. Arms — exactly two

**ARM P400 — production.** No scientific overrides. Production move ladder,
production admission and stopping, `runtime.maxOuter = 400` (the production
default; production is expected to stop far earlier). Only `runtime.name`
differs from what `olhoffcurrent_config` produces, and that is asserted
field-by-field over the whole schema before the run.

**ARM F400 — fixed-move counterfactual.** Identical except
`move.policy = 'fixed'`, `move.initial = 0.04` — the exact override pair used by
`move_stop`'s `fixedmove` arm at 160x20 and 320x40. Safety cap
`runtime.maxOuter = 600`.

The cap is deliberately more generous than `move_stop`'s 400 because 400x50 is a
new mesh and 160x20 hit that cap. **Enlarging a cap cannot change a trajectory,
only when it stops**, so this does not compromise comparability. Declared now so
it cannot be adjusted after seeing the run.

F400's purpose is **only** to measure how much evolution production truncates
when it first descends. It is not a candidate production method.

**No third arm.**

## 4. Counterfactual validity gate (blocking)

Before F400 may be read as production's continuation, verify P400 and F400 are
**bitwise identical** through the iteration immediately preceding P400's first
move descent, in:

- the full density field `rho` (every element, every iteration);
- `omega1`, `omega2`, `vol`, `dxOuter`, `dxNorm2`, `move`, `beta`, `nInner`.

Comparison is `isequaln` on doubles — bitwise, not tolerance-based. This
reproduces the property that made the 160x20/320x40 evidence strong (both were
`0.000e+00`).

If it fails: report **`COUNTERFACTUAL_PREFIX_MISMATCH`** and stop the causal
interpretation. F400 may still be described, but not as production's
continuation.

## 5. Raw trajectory retention (required)

Per `analysis/OlhoffCurrent/EVIDENCE_POLICY.md`. For **both** arms save to
`analysis/OlhoffCurrent/evidence/move_activity_400/<ARM>_400x50_trajectory.mat`
(MAT v7.3):

- `RHO` — `NE x nOuter` **double**, the physical density at every outer iteration;
- `DRHO` — `NE x nOuter` **double**, the increment the solver applied;
- `move`, and the complete `hist` struct and resolved `cfg`.

Declared `required` in `EVIDENCE.json` with byte size, SHA-256, dimensions and
precision. The study cannot be frozen unless
`olhoffcurrent_evidence_gate` passes.

Three reconstruction checks, all asserted at run time:

1. `mean(RHO(:,k)) == hist.vol(k)` to `< 1e-12` for every `k`;
2. `RHO(:,end)` **bitwise** equals `res.rho`;
3. `max|(RHO(:,k)-RHO(:,k-1)) - DRHO(:,k)| == 0` — i.e. the update's `min/max`
   clamp is inert and the recorded increment *is* the applied increment. Prior
   studies asserted this in a comment; here it is measured. If non-zero, `DRHO`
   is authoritative and the discrepancy is reported.

Derived summaries are **not** a substitute for the raw trajectory.

## 6. Activity definitions (identical to the prior offline study)

Utilisation, with the move-indexing convention established from
`olhoffSolve.m` (`move(k)` bounds the increment stored at index `k`, so it
governs `rho(k-1) -> rho(k)`):

```
u_e(k) = |rho_e(k) - rho_e(k-1)| / move(k)
```

Per iteration compute over all `NE` elements: `min`, `P50`, `P75`, `P90`, `P95`,
`P97.5`, `P99`, `max`, `mean`, `RMS`; and active counts and fractions at the
**absolute** `|Delta rho|` thresholds

```
tau = { 1e-4 ,  8.83883476483184e-4 (= epsRMS) ,  1e-3 ,  1e-2 }
```

**Units and normalisation, stated explicitly.** `tau` is on the *raw* increment
`|Delta rho_e|`, dimensionless density change — **not** on `u`. This is the
convention `move_stop` used (`ms_run.m:77`, `tau = [epsRMS, 1e-4, 1e-3, 1e-2]`)
and it is preserved so 160/320/400 counts are directly comparable. At
`move = 0.04` the `u`-equivalents are `u > 0.0025`, `0.0221`, `0.025`, `0.25`.
Percentiles of `u` are additionally reported, but the **scaling analysis uses
the raw-threshold counts**, because those are what exist at 160x20 and 320x40.

Also recorded: participation number `N_eff = (||drho||_2 / max|drho|)^2`
(exact, move-free).

## 7. "Remaining evolution" — formula frozen from the prior study

Let `i` be the last iteration at `move = 0.04` (the iteration immediately before
P400's first descent), and let F400's final recorded iteration be the mature/cap
endpoint. Using F400 values (which equal production's at `i` by the §4 gate):

```
remaining_absolute = M_nd(i) - M_nd(end)
remaining_relative = ( M_nd(i) - M_nd(end) ) / M_nd(i)
completion  c(k)   = ( M_nd(1) - M_nd(k) ) / ( M_nd(1) - M_nd(end) )
```

`M_nd = 100*mean(4*rho.*(1-rho))` on the physical density. These are exactly the
formulae used for 160x20 and 320x40 (`mao_analyze.py`), so the three meshes are
directly comparable. No substitute metric will be introduced.

For reference, the values they produced: 160x20 → 1.218 pts / **9.0%**;
320x40 → 10.168 pts / **43.4%**.

## 8. Scaling analysis

Primary comparison is at **matched M_nd maturity** `c`, not at each mesh's own
descent iteration — the meshes descend at different maturities (`c = 0.986` at
160x20, `0.882` at 320x40), so comparing there would confound maturity with
refinement. Matched-maturity is also what the prior collapse analysis used.

For each threshold `tau` and each `c in {0.90, 0.95, 0.99}` fit

```
N_active = C * NE^alpha
```

by least squares on `log N` vs `log NE` over the three meshes, and report `C`,
`alpha` and residuals. Also report, per §9 of the brief, the descent-point
values as a secondary descriptive comparison.

**Pairwise exponents are mandatory** and reported for every threshold:

```
alpha_ij = log( N_j / N_i ) / log( NE_j / NE_i )
```

giving `alpha_160_320`, `alpha_320_400`, `alpha_160_400` and the global 3-point
fit. A global regression that looks good must not be allowed to hide
incompatible pairwise scaling.

**Reference models:** constant count `alpha = 0`; interface-like
`alpha = 0.5`; area fraction `alpha = 1`.

## 9. Interpretation rules — decided now

Let `spread = max(alpha_pairwise) - min(alpha_pairwise)` within a threshold, and
`spread_tau` the spread of the global `alpha` across the four thresholds.

| verdict | condition |
|---|---|
| `SINGLE_POWER_LAW_ACTIVITY_SCALING_SUPPORTED` | `spread <= 0.10` for every threshold **and** `spread_tau <= 0.10` |
| `APPROXIMATE_POWER_LAW_FAMILY_SUPPORTED_BUT_ALPHA_UNCERTAIN` | `spread <= 0.25` and `spread_tau <= 0.25`, but not the above |
| `SINGLE_POWER_LAW_NOT_SUPPORTED` | `spread > 0.25` for a threshold, **or** `spread_tau > 0.25` |
| `ACTIVE_COUNT_FAMILY_NOT_SUPPORTED` | the low-threshold active count fails to decrease with maturity at 400x50 (Spearman vs remaining evolution `<= 0`), or the cross-mesh ordering property fails |
| `INSUFFICIENT_COMPARABLE_DATA` | §4 gate fails, a run fails, or F400 does not reach a state interpretable as mature |

Then separately:

- **`ACTIVE_COUNT_CONTROLLER_READY_FOR_DESIGN`** only if the scaling verdict is
  one of the first two **and** `alpha` is determined well enough that a
  normalisation could be written down (`spread_tau <= 0.25`).
- **`MOVE_ACTIVITY_MODEL_REQUIRES_REVISION`** otherwise.

Production remains **`KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`** in every case.

## 10. What this study will NOT do

No controller is implemented, run, or threshold-tuned — even if the scaling
result is clean. No change to projection, filter type, filter radius, `p`, mass
interpolation, admission or move policy. No 800x100, no nine-mesh campaign. No
rerun of 160x20 or 320x40 to recover deleted evidence.

One deliberate exception, declared here in advance: a **short 160x20 production
reproduction** may be run *for build-equivalence validation only*. This machine
has MATLAB `25.2.0.2998904`, while every prior study recorded
`25.2.0.3042426 (Update 1)`. Combining a 400x50 result from one build with
160/320 numbers from another is unsound unless the builds are shown to produce
identical arithmetic. This reproduction is a provenance control, not evidence
recovery: its output is compared against the committed
`move_stop/runs/baseline_160x20_iterations.csv` and then used only to decide
whether the three meshes may be combined. If it does not reproduce bitwise, the
cross-mesh scaling analysis is reported as
`INSUFFICIENT_COMPARABLE_DATA`.
