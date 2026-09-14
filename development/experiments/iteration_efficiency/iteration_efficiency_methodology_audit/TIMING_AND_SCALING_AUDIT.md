# Timing and scaling audit

Audit-only. `TIMING_SPEC.md` and `SCALING_AND_FIGURE_SPEC.md` were not modified.

---

# PART A — Timing

**Overall verdict: fair, and the strongest part of the package.** I went looking for
asymmetries and found one, which turns out to be immaterial in magnitude.

## A1. Does the design avoid observer/I/O contamination?

Yes, by construction. The trajectory-discovery run stores every density field and is
explicitly declared unpublishable as algorithm time. Timing comes from separate
deterministic **fixed-horizon replays** at the already-frozen `k_enter` and `k_cert`, with
common evaluators, images, density-history disk writes and post-certification continuation
all disabled, and with trajectory fingerprints verified at both endpoints.

This is the correct architecture and it is what most published method comparisons get
wrong. It is also already supported by passing evidence: `extension_invariance_validation.json`
shows bitwise prefix identity for all three methods at 160x20 with
`max_abs_diff_xphys_at_stop = 0` and `scalar_prefix_identical: true`, which is the property
the replay depends on.

## A2. Checklist

| item | present? | note |
|---|:--:|---|
| warm-up policy | yes | one discarded warm-up per method, **outside** the production mesh set — correct, avoids priming a measured cell |
| thread pinning | yes | single-thread, serial, requested **and active** thread count recorded; `maxNumCompThreads(1)` is already the convention in the frozen runners |
| solver startup | yes | MATLAB version/update, Optimization Toolbox, `linprog`/HiGHS version and exact options archived |
| initialization | yes | `T_init` defined, **explicitly including Proposed's frozen-solid reference eigensolve** — correct, it is real work Proposed must do |
| terminal eigensolve | yes | excluded from iteration counts, timed as `T_native_finalize`; symmetric, since Olhoff also gets a `T_native_finalize` |
| setup time | yes | folded into `T_init`, decomposed in the supplement |
| per-stage timing | yes | Yuksel stage times mandatory and separate; combined mean permitted only as an explicitly labelled time/count summary |
| post-processing | yes | disclosed, not charged |
| repeated-run policy | yes | median of three serial replays, range and MAD, frozen balanced randomised order |
| failed replay handling | yes | investigated and visibly excluded with reason — **not** discarded as a timing outlier. This is the right rule and is rarer than it should be |
| deterministic replay | yes | same executable and profile; only diagnostic writes and stop horizon differ; source/config hashes and prefix identity required |

`T_result_to_enter`, `T_result_to_cert` and the mean per-iteration times are reproducibly
defined:
`T_result_to_e = T_init + T_loop_to_e + T_native_finalize`, with `T_gate_offline` and
`T_observer_after_cert` disclosed but not charged.

## A3. Is any component unfairly excluded for one method?

One asymmetry exists: Olhoff's `T_init` and `T_native_finalize` are unattributed in the
frozen runner, and the spec records them as `NA` rather than zero. Recording `NA` is the
correct policy; the question is whether the missing quantity matters.

Magnitude check at 800x100, from `table1_performance.csv`:

| method | init (s) | loop (s) | post (s) | total wall (s) | unattributed (s) |
|---|---:|---:|---:|---:|---:|
| Olhoff | NaN | 2950.50 | NaN | 2953.97 | **3.47** |
| Proposed | 1.67 | 268.66 | 2.69 | 273.02 | 0.01 |
| Yuksel | 0.20 | 978.23 | 3.72 | 982.15 | 0.00 |

Olhoff's entire unattributed overhead is 3.47 s against a 2950 s loop — the same absolute
size as Proposed's *attributed* init+post, and negligible against either loop. **The `NA` is
a reporting gap, not a fairness gap.** A hostile reviewer would flag it; the honest answer
is that it does not matter here. It should still be instrumented in Phase 2 because it is
cheap.

Common-evaluator time is excluded equally for all three methods, which is correct — it is
experiment measurement, not algorithm work — and the exclusion is not concealing a large
number (see PART C).

## A4. Residual points

- The `single-run descriptive` fallback forbids mixing policies across **methods** but not
  across **meshes**. A run that used three replays at coarse meshes and single-run timers at
  fine meshes would satisfy the letter of the rule while making the timing scaling curve
  inhomogeneous. Extend the no-mixing rule to meshes (Finding Mi3).
- The prohibition on hardware-independent timing claims is explicit and correct. Keep it.

---

# PART B — Scaling

## B1. The mesh design

`N_e = nelx·nely` for the nine frozen meshes:

| mesh | 160x20 | 240x30 | 320x40 | 400x50 | 480x60 | 560x70 | 640x80 | 720x90 | 800x100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N_e` | 3200 | 7200 | 12800 | 20000 | 28800 | 39200 | 51200 | 64800 | 80000 |
| `log10 N_e` | 3.505 | 3.857 | 4.107 | 4.301 | 4.459 | 4.593 | 4.709 | 4.812 | 4.903 |

Consecutive spacing in `log10 N_e`: 0.352, 0.250, 0.194, 0.158, 0.134, 0.116, 0.102, 0.092.

**Total span: 1.398 decades.** The meshes are linearly spaced in `nelx`, so `N_e` is
quadratically spaced and the points bunch at the top: **five of nine lie in the last half
decade**, and the coarsest mesh sits alone with by far the greatest leverage.

**Are nine resolutions sufficient?** For a descriptive fit, yes. For an exponent anyone
should quote, marginal. Leverage check by jackknife on the frozen four-mesh Olhoff series
[186, 234, 234, 246]:

| dropped point | fitted `p` |
|---|---:|
| — (all four) | +0.145 |
| drop `N_e` = 3200 | **+0.047** |
| drop `N_e` = 7200 | +0.156 |
| drop `N_e` = 12800 | +0.148 |
| drop `N_e` = 20000 | +0.173 |

Dropping the coarsest mesh changes `p` by a factor of 3; dropping any other changes it by
at most 20 %. The single coarsest point effectively determines the exponent.

## B2. Censoring — handled well

Only certified baseline-R cells are fit-eligible. `PASS_WITH_LATER_FAILURE` remains eligible,
which is correct — certification genuinely preceded the failure, and excluding it would
discard a valid minimum-work observation. `QUALITY_NOT_REACHED`, `INVALID_TOPOLOGY`, solver
failure, `REFERENCE_INVALID`, persistent nonconvergence, insufficient budget, missing data
and trajectory/timing mismatch are censored, plotted with a distinct hollow/crossed/lower-
bound symbol and a status key, never connected as ordinary data, never regressed. Internal
missing meshes stay visibly missing. Explicit prohibitions on imputation, method pooling,
ranking-driven weighting, eligible-outlier removal and post-hoc subrange selection are all
present.

This is good practice and needs no correction.

## B3. The gap the spec leaves open — unequal support

The spec requires reporting `Ne_min`/`Ne_max` and forbids extrapolation. It does **not**
forbid placing exponents fitted on different mesh subsets side by side.

The completed campaign already does exactly this
(`examples/Performance/final_campaign/table1_complexity_fit.csv`):

| method | `C` | `p` | `R²` | `NPoints` | excluded meshes |
|---|---:|---:|---:|---:|---|
| Olhoff | 3.916e-03 | 1.1931 | 0.9960 | **6** | 480x60, 560x70, 640x80 (solver failures) |
| Yuksel | 3.614e-06 | 1.7055 | 0.9814 | **7** | 640x80, 800x100 (cap hits) |
| Proposed | 2.023e-05 | 1.4177 | 0.9640 | **9** | none |

Olhoff's fit has an internal gap covering 28 800–51 200; Yuksel's excludes the two largest
meshes entirely. Reading `1.71 > 1.42 > 1.19` as a scaling comparison is not valid — the
three numbers describe different `N_e` ranges.

Given the T1 result (see `TOPOLOGY_GATE_AUDIT.md`), the new study's support asymmetry will
be **worse**: Olhoff would be censored at 480x60, 560x70, 640x80 and 800x100, leaving at
most five valid meshes against Proposed's nine.

**Minimum correction:** require a companion **common-support fit** restricted to the meshes
valid for all three methods, reported beside each per-method full-range fit, and forbid
cross-method exponent comparison outside common support.

## B4. Fit mechanics

| item | assessment |
|---|---|
| `R²_log` | correctly computed on log values and correctly labelled. Note it is a weak diagnostic here: adding 99 to every point changed `p` by 32 % while `R²_log` moved from 0.839 to 0.843 |
| `n_valid`, three-point minimum, three distinct `N_e` | present and adequate; the two-point descriptive-connector rule with `C,p,R2 = NA` is exactly right |
| fitting range | bounded to observed valid data; no extrapolation |
| log-log OLS | correct estimator for a power law with multiplicative error |
| weighting | none — the correct default, but it means the high-leverage coarsest mesh dominates (§B1) |
| heteroscedasticity | not addressed, and cannot be with one deterministic run per cell |
| confidence bands | forbidden unless a predeclared method is supplied. Honest, but leaves a bare `p` with no interval |
| `C` | correctly stated to retain the plotted quantity's units. Should additionally be flagged as the value at `N_e = 1`, which is physically meaningless here, so `C` must never be interpreted physically |

**Minimum correction:** report a **leave-one-out range of `p`** with every fit. It is
deterministic, assumes no sampling model, does not conflict with the confidence-band
prohibition, and communicates exactly the leverage problem in §B1.

## B5. The `k_cert` additive constant (Audit Question 11)

`k_cert(N_e) = k_enter(N_e) + 99` at `P = 100`. Fitting `C·N_e^p` to this fits a power law
to **a power law plus a constant**, which is not a power law. The fitted exponent is a
property of the sum and varies with the counts' size relative to 99.

Quantified on the frozen Olhoff series:

| series | `C` | `p` | `R²_log` |
|---|---:|---:|---:|
| `k_enter` | 59.91 | **+0.1451** | 0.839 |
| `k_enter + 99` | 131.5 | **+0.0991** | 0.843 |

**32 % reduction in the exponent from a bookkeeping convention**, invisible in `R²_log`.
The distortion is largest where counts are smallest — for Proposed, whose native counts are
107–330.

**Recommendation (not applied):** `k_enter` scaling **primary**; `k_cert` scaling
**secondary/descriptive** with a convention caveat in the caption; and add an identity
check fitting `k_cert − (P−1)`, which must reproduce `p_enter` exactly and makes the
convention effect visible in one line.

The protocol already requires independent fits, warns about flattening and mandates the
`P = 50/200` sensitivity. What it does not do is subordinate `k_cert` scaling — it lists
both as co-equal mandatory layers. Given a quantified 32 % distortion, co-equal status is
too generous.

## B6. A hazard specific to Proposed

Frozen Proposed counts by mesh: 107, 236, 207, 182, 219, 256, 309, 297, 330. **Non-monotone**,
with only ~3× dynamic range over a 25× range in `N_e`. A power law fitted to that will have
a small, poorly identified exponent and a modest `R²_log`. Reporting it as *a small
exponent* rather than *a weakly identified one* would overstate the result. A preregistered
"weakly identified" label — triggered by low `R²_log` or a wide leave-one-out `p` range — is
the minimum correction (Finding Mo4).

## B7. Empirical scaling vs asymptotic complexity

`SCALING_AND_FIGURE_SPEC.md` §6 prohibits "asymptotic complexity", "order-optimal" and
equivalents, requires captions to say "empirical scaling over the tested mesh range", and
retires the old fixed-`p = 1.5` reference fit.

**This is exactly right and I endorse it without reservation.** With 1.398 decades, one
deterministic run per cell, method-dependent censoring and a coarsest point that moves `p`
by 3×, only empirical scaling is supportable. An asymptotic complexity claim would
additionally require an operation-count model, which this study does not have and does not
attempt.

---

# PART C — Offline gate cost (feasibility, not fairness)

The acceptance engine must evaluate `Q(k)` at **every** state: `Q_ref` is a max over windows
of a window-min, so it needs the whole trajectory, and no bisection is possible.

Estimated from the frozen Olhoff eigensolve timings. Median `tEig` (5 modes) by mesh:

| `N_e` | 3200 | 7200 | 12800 | 20000 | 28800 | 39200 | 51200 | 64800 | 80000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tEig` (s) | 0.0275 | 0.0628 | 0.1150 | 0.2021 | 0.3260 | 0.4212 | 0.7102 | 1.0082 | 1.3676 |

Free power fit: `t ≈ 1.316e-6 · N_e^1.214` s. Scaling to 3 modes (×0.7) and applying the full
budgets (Proposed 900, Yuksel 4000, Olhoff 3200 states per mesh):

- **E1-raw only, all nine meshes, all three methods: ≈ 6.4 h** single-threaded.
- **Complete E1/E2/E3 × raw/binary set: ≈ 38 h** single-threaded.
- Trajectory storage: **≈ 20 GB** uncompressed double precision.

Both are tractable. This **supports** the protocol's claim that the sensitivity plan needs
no extra optimization — the rescans really are cheap relative to the runs. Neither figure
appears in the Phase-A estimate and both should (Finding Mi4).

Precedent that per-state evaluation works in practice:
`analysis/olhoff_stabilization_audit/evaluate_stabilized_e1.m` already evaluates E1 at all
1601 states for four meshes and two profiles, and asserts agreement with
`study_evaluate_design.m` at checkpoints to `1e-8`.
