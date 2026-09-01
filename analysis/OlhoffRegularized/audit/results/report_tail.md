
## 12. WP7 — regression tests

`run_olhoff_regularized_tests.m` grew from one group to eleven. Every case is a
toy-mesh **software-mechanics** test (dispatch, parsing, accounting, controller
forcing, classification); none is cited as evidence about convergence quality,
topology or mesh behaviour.

| # | test | guards |
|---|---|---|
| 1 | four routes execute; volume held; caps respected; trust ≤ ceiling; ceiling monotone; **`offDiag` live for `olhoff/mma` and only there**; LP route refuses `offDiag` | route dispatch, accounting, F-1-adjacent wiring |
| 2 | forced stage exhaustion contracts the persistent ceiling and logs it; post-update trust ≤ ceiling | controller mechanics |
| 3 | **a significant single-step improvement blocks contraction**: with `progress_tolerance` wide open the cumulative test always passes, so contraction can only come from the spike guard; the guard is calibrated from the run's own observed maximum single-step progress, asserted to block at 0 and to permit at 10× | cumulative-progress logic cannot be fooled by one significant improvement |
| 4 | **`CAP_HIT` can never satisfy the native-convergence classification**: status, stop reason, and `max(convergenceCount) < persistence` | new |
| 5 | **minimum trust without stationarity ⇒ `GLOBALIZATION_STALLED`** (forced by `min_inner = max_inner`, `tol_inner = 0`, `move_min = move`) | new |
| 6 | **convergence requires the declared persistence**: `persistence = 3` converges at exactly outer 3 with counter 3; the identical run capped at 5 with `persistence = 8` is `CAP_HIT` | new |
| 7 | the convergence tolerance is clamped to `objective_tol/certificate_radius`; the caller's looser request is recorded but **cannot reach either gate**; behaviourally, a loose request produces **zero** contractions at a non-stationary point (CV-4) | new |
| 8 | the certificate is **radius-scale invariant** (1e-6 relative across a 10× radius change) and is evaluated at the fixed reference radius over a run in which the trust radius moves (CV-2) | new |
| 9 | the certificate uses the **exact** multiplicity: with `tol_mult = 3.0` the step model clusters, `certificateN == 1` (CV-1) | new |
| 10 | cantilever concentrated-mass path executes with a positive tip mass | retained |
| 11 | **every runner honours an externally selected route**: no unconditional assignment to `optimizer`/`formulation`, and all three still advertise the override (F-1) | new |

Result: `OLHOFF_REGULARIZED_TESTS_PASS groups=11`.

---

## 13. Scientific identity

Three things must not be conflated, and this report does not conflate them.

| | what it is | what may be claimed |
|---|---|---|
| **Historical reproduction** — `Matlab/reproduction2007/` | the frozen, fixed-work Du–Olhoff (2007) clean-room reproduction, 1600 outer iterations, no stopping test | paper reproduction. Untouched by this audit; `git status` on it is empty and the four primitives the regularized code consumes are byte-identical to their audited state |
| **Regularized Olhoff formulation** — `formulation='olhoff'` | the genuine multiple-eigenvalue Olhoff local model (Eq. (16)/(22) LP, or the full Eq. (25d) sub-eigenvalue problem through nested MMA) plus a trust-region acceptance test, a persistent move-limit continuation and a stopping certificate. The local subproblem is Olhoff's; **the globalization and the stopping rule are disclosed numerical extensions and are not in the paper** | a globalized Olhoff formulation. **Not** a paper-literal reproduction, and its iteration counts are not comparable with the 1600-iteration historical run |
| **KS-inspired alternative** — `formulation='ks'` | the spectral objective replaced by a smooth Kreisselmeier–Steinhauser lower aggregate of the lowest modes | an Olhoff-*inspired* regularization. **Never** an Olhoff reproduction. This audit additionally shows it optimises a different function: its terminal design is `CONVERGED` for its own aggregate while carrying a 2.93e-04 relative feasible ascent in the *physical* λ₁ |

The regularized implementation reuses the frozen FE, filter, generalized-gradient,
LP and MMA primitives, as permitted, through `repro2007_paths()` with an
identity assertion. The audit's own WP2 code deliberately does **not** reuse the
production model builder or its stopping logic.

---

## 14. Remaining uncertainties

1. **The certificate is a lower bound when the cluster is genuinely degenerate
   with N ≥ 2.** In production it uses the Eq. (22) equality-restricted LP, whose
   feasible set is contained in the true one. Measured at every terminal design
   in this audit the restriction is not binding (agreement to twelve digits with
   the unrestricted cutting-plane), and the value is combined by `max()` with the
   route's own model slope — which on the MMA route *is* the unrestricted model.
   But this is empirical, not proved for all designs. A production cutting-plane
   certificate would close it at a cost of 3–5 extra LP solves per outer
   iteration.
2. **The threshold is configuration-relative, not absolute.** "No feasible step
   within `move` improves λ₁ by more than `objective_tol`" is a well-posed
   statement, but its strength depends on the user's `move` and `objective_tol`.
   Raising `move` weakens it. This is now explicit rather than hidden, which is
   the improvement; it is not an absolute stationarity guarantee.
3. **Exact stationarity is not attained on this problem, by any of these
   routes.** §7.2 shows the criticality measure falling to ~8e-5 and then
   *rising* again, wandering in a 1e-4 – 3e-3 band for 1400 further iterations.
   The terminal region is a shallow non-smooth bimodal ridge. Every `CONVERGED`
   in this report is a tolerance statement about a near-stationary design, and
   should be cited as such.
4. **The nested MMA under-solves its own subproblem by ~2.8 %** (§7.3). The
   correction defends the *stopping test* against that bias but does not remove
   the bias from the *steps*. Whether tightening `tol_inner` changes the terminal
   design was not tested.
5. **`move_max = move` pins the trust radius at its ceiling for the entire ascent
   phase**, so the trust-region machinery only ever contracts. A genuinely
   adaptive radius (`move_max > move`) was not exercised.
6. **One filter radius policy.** Constant physical radius 0.075 across meshes.
   Mesh robustness of the stopping rule at other radii, and with other filter
   modes (`diag`, `all`, `none`) or the discontinuous mass law `4`, is untested —
   the code warns about the last of these.
7. **`n = 1` only.** Optimising a higher mode (`cfg.n > 1`) was not exercised.
8. **The 3000-iteration caps** used for the corrected LP runs are diagnostic
   headroom, not a proposed production cap. A production cap should be frozen
   only against the observed natural convergence iterations recorded here.

---

## 15. Recommendation

**Ready with qualification.**

Usable now, with the corrections in place:

* `formulation='olhoff', optimizer='lp'` — natural convergence at 160×20,
  240×30 and 320×40, with an independently reproduced certificate and physical
  fixed-step confirmation. This is the route to use for production work.
* `formulation='olhoff', optimizer='mma'` — see §11 for the measured result and
  its qualification.

Conditions of use:

1. **Cite `CONVERGED` as what it is.** The correct wording is *"no feasible step
   within the configured move limit improves λ₁ by more than `objective_tol`,
   sustained over `persistence` accepted updates"* — not "a stationary point".
   §7.2 shows the criticality measure does not reach zero on this problem.
2. **Do not raise `stationarity_tol`.** It is derived, and a looser request is
   clamped with a warning. Raising it by raising `objective_tol` re-opens
   exactly the defect this audit found.
3. **Do not compare iteration counts with `Matlab/reproduction2007`.** That is a
   fixed-work reproduction with no stopping test; these are converged runs of a
   different algorithm.
4. **The KS routes are not Olhoff.** `ks/lp` now correctly `CAP_HIT`s at 160×20
   because its terminal design carries a 2.93e-04 relative feasible ascent in the
   physical λ₁ while being converged for its own aggregate. Use KS as a control,
   never as a reproduction, and never quote a KS `CONVERGED` as an
   eigenfrequency-stationarity result.
5. **Re-run `run_olhoff_regularized_tests` before trusting any change** to the
   controller or the stopping test; groups 4–9 and 11 exist specifically to
   prevent the regressions this audit found.

Further work, in priority order:

1. Replace the Eq. (22)-restricted production certificate with the cutting-plane
   certificate for exactly degenerate clusters of size ≥ 2 (uncertainty 1).
2. Investigate the ~2.8 % under-solve of the nested MMA subproblem
   (uncertainty 4) — tighten `tol_inner`, or warm-start `low`/`upp` across trials.
3. Decide a production `max_outer_iterations` from the natural convergence
   iterations recorded in §9, with headroom, rather than from a fixed-work
   convention.
4. Exercise `move_max > move` so the trust region can genuinely expand
   (uncertainty 5).

---

## Appendix A — what "same basin?" means numerically

Terminal filtered-density fields compared pairwise on the 160×20 mesh
(`audit/scripts/audit_compare.m`; PNGs alongside each result directory):

| A | B | mean abs Δρ | max abs Δρ | Pearson r |
|---|---|---|---|---|
| `ss160x20_olhoff_lp` | `ss160x20_ks_lp` | 0.0131 | 0.4350 | 0.99594 |
| `ss160x20_olhoff_lp` | `R_ss160x20_olhoff_lp` | 0.0149 | 0.3544 | 0.99579 |
| `ss160x20_ks_lp` | `R_ss160x20_ks_lp` | 0.0071 | 0.1556 | 0.99901 |
| `ss160x20_olhoff_lp` | `DIAG_nocontract_olhoff_lp` | 0.0421 | 0.8988 | 0.95772 |
| `R_ss160x20_olhoff_lp` | `DIAG_nocontract_olhoff_lp` | 0.0303 | 0.8505 | 0.96873 |

Every 160×20 terminal design is the same topology to r ≥ 0.955. The routes and
the criteria do not select different optima — they stop at different points
along the same shallow bimodal ridge. That is why "the topology looks right" is
not evidence about the stopping test, and why this audit does not use it as
evidence.

## Appendix B — reproducing the audit

```matlab
addpath analysis/OlhoffRegularized/Matlab
run_olhoff_regularized_tests                       % 11 groups, toy meshes

addpath analysis/OlhoffRegularized/audit/scripts
audit_run('R_ss160x20_olhoff_lp','simply',160,20,1.5,'olhoff','lp',3000)
audit_stationarity('R_ss160x20_olhoff_lp')         % WP2, independent
audit_wp5_table({'R_ss160x20_olhoff_lp'})
audit_main_table({'R_ss160x20_olhoff_lp'})
```

`audit_run` writes `audit/results/<tag>/{run.mat,trajectory.csv}` and one
`AUDITRESULT` line; `audit_stationarity` adds `stationarity.mat` and prints the
certificate and the physical fixed-step table. `audit/scripts/summarize.py`
collects every `AUDITRESULT` line in `audit/logs/` into one table.
