
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
| 12 | **a negative predicted improvement is never read as stationarity** (CV-5): asserted structurally over every recorded iteration of every route — a finite stationarity slope may not coexist with `predicted < 0` — and the counter exists | new |
| 13 | **`cluster_lambda` dispatches, validates, and actually changes the diagonal gradients** when the cluster is not degenerate; an invalid value is rejected | new |

Result: `OLHOFF_REGULARIZED_TESTS_PASS groups=13`.

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

1. **The principal open item: why the nested-MMA route cannot close the mode
   gap.** The three routes that reach ω₁ ≈ 161.6 — `olhoff/lp`, `ks/lp`,
   `ks/mma` — are exactly the three that never form the off-diagonal `f_sk`
   (s ≠ k): the LP routes force `f_sk'drho = 0` by Eq. (22), the KS routes never
   build them. The one route that stalls at 156.05 is the only one that feeds the
   full Eq. (25d) coupling into its subproblem, and `ks/mma` proves the nested
   MMA optimizer itself is not at fault. Three mitigations were tested (§9.3.1):
   tightening `tol_mult` confirms the mechanism but recovers little objective;
   tightening `tol_inner` is not viable (0.972 → 0.991 at 4× cost, saturating
   the inner cap); per-mode λ̃ on the diagonal gradients is refuted outright.
   Diagnosing the off-diagonal coupling itself was out of scope, because removing
   it *is* removing the primary route.
2. **The certificate is a lower bound for an exactly degenerate cluster of size
   ≥ 2.** In production it uses the Eq. (22) equality-restricted LP, whose
   feasible set is contained in the true one. Measured at every terminal design
   here the restriction is not binding (twelve-digit agreement with the
   unrestricted cutting plane), and the value is combined by `max()` with the
   route's own slope — which on the MMA route *is* the unrestricted model. This
   is empirical, not proved for all designs.
3. **The threshold is configuration-relative.** "No feasible step within `move`
   improves λ₁ by more than `objective_tol`" is well posed, but its strength
   depends on the user's `move` and `objective_tol`. Raising `move` weakens it.
   This is now explicit rather than hidden; it is not an absolute guarantee.
4. **Exact stationarity is not attained on this problem by any route.** §7.2
   shows the criticality measure falling to ~8e-05 and then *rising*, wandering
   in a 1e-04 – 3e-03 band for 1400 further iterations. Every `CONVERGED` here is
   a tolerance statement about a near-stationary design, and should be cited as
   such.
5. **The nested MMA under-solves its own subproblem by ~2.8 %** and this is only
   partly reducible (§9.3.1b). The corrections defend the *stopping test* against
   that bias but do not remove it from the *steps*.
6. **`move_max = move` pins the trust radius at its ceiling** through the entire
   ascent phase, so the trust-region machinery only ever contracts. A genuinely
   adaptive radius (`move_max > move`) was not exercised.
7. **Runs stopped once their outcome was determined**, and are labelled as such
   rather than reported as completed: `olhoff/mma` at 320×40 (outer 590),
   fixed–pinned `olhoff/mma` (outer 607), the as-shipped `olhoff/mma` mesh sweep
   at 240×30 and 320×40 (outer 114/125), the headroom diagnostic (outer 380), and
   the three mitigation runs. In each case the quantity that decides the
   classification — mode-gap plateau, certificate, or subproblem accuracy — had
   already stabilised.
8. **One filter-radius policy** (constant physical 0.075). Other radii, the
   sensitivity-filter modes `diag`/`all`/`none`, and the discontinuous mass law
   `4` are untested; the code warns about the last.
9. **`n = 1` only.** Optimising a higher mode (`cfg.n > 1`) was not exercised.
10. **The audit's raw evidence is not versioned.** The repository `.gitignore`
    excludes `*.log`, `*.csv`, `*.txt` and `*.mat`, so every trajectory, log and
    WP2 output under `audit/` is untracked: 3.4 MB of logs, 6.2 MB of trajectory
    CSVs, 0.8 MB of WP2 text output and 15 MB of `run.mat`. Only the scripts and
    this report are committed. The numbers quoted here are therefore reproducible
    (Appendix B) but not archived, and a `git clean` would destroy them. Deciding
    whether to add a scoped un-ignore for the ~10 MB of text evidence is a
    repository-policy question left to the maintainer.
11. **The 3000-iteration caps** on the corrected LP runs are diagnostic headroom,
    not a proposed production cap. §11 shows the natural convergence iteration is
    strongly problem-dependent (574 / 1567 / 2961), so a production cap must be
    set from the problem.

## 15. Recommendation

**Ready with qualification — for the LP route. Further work required for the MMA
route.**

### Use now

`formulation='olhoff', optimizer='lp'` with the corrected defaults. Natural
convergence at 160×20 / 240×30 / 320×40 and on two further problems, every
terminal design independently certified below `objective_tol` and confirmed by
physical fixed-step eigensolves. This is the route for production work.

### Do not use for a convergence claim

`formulation='olhoff', optimizer='mma'`. It does not reach a certifiable design
at any mesh tested, and the cause is in the formulation's off-diagonal coupling,
not in the stopping rule (§14.1). It costs ~172× the LP route's wall time for a
worse objective and a certificate two orders of magnitude worse. It remains
useful as a research object and as the control that exposed CV-5.

The KS routes are diagnostics only. `ks/mma` attains the highest ω₁ of any run
(161.963) and is **not** stationary for the physical λ₁ (residual ascent 33×
`objective_tol`) — a high eigenfrequency is not evidence.

### Conditions of use

1. **Cite `CONVERGED` as what it is**: *"no feasible step within the configured
   move limit improves λ₁ by more than `objective_tol`, sustained over
   `persistence` accepted updates"* — not "a stationary point". §7.2 shows the
   criticality measure does not reach zero on this problem.
2. **Do not raise `stationarity_tol`.** It is derived and a looser request is
   clamped with a warning. Raising it via `objective_tol` re-opens CV-3 exactly.
3. **Set `max_outer_iterations` from the problem.** Measured natural convergence
   is 574 (simply supported), 1567 (fixed–pinned) and 2961 (cantilever). The
   fixed–pinned runner's existing 1600 is closer to the edge than is comfortable.
4. **Do not compare iteration counts with `Matlab/reproduction2007`.** That is a
   fixed-work reproduction with no stopping test.
5. **Never quote a KS `CONVERGED` as an eigenfrequency-stationarity result.**
6. **Re-run `run_olhoff_regularized_tests` before trusting any change** to the
   controller, the certificate or the step model. Groups 4–9 and 11–13 exist
   specifically to prevent the regressions this audit found.

### Further work, in priority order

1. **Diagnose the off-diagonal Eq. (25d) coupling** (§14.1). This is the one
   thing standing between the primary route and a usable result, and the KS
   control has narrowed it to a single term. Suggested next experiment: at a
   recorded near-degenerate design, compare the step the coupled model produces
   with the step the Eq. (22)-restricted model produces, and check which one the
   *physical* trial eigensolve prefers — a fixed-design comparison costing
   minutes rather than another thousand-iteration run.
2. **Replace the Eq. (22)-restricted production certificate** with the
   cutting-plane certificate for exactly degenerate clusters of size ≥ 2
   (§14.2), at a cost of 3–5 extra LP solves per outer iteration.
3. **Reconsider `tol_mult = 0.05` for the step model.** It creates a
   self-sustaining pseudo-degeneracy: the design parks just beneath whatever
   value it is given (§9.3, §9.3.1a). The same threshold-plateau signature that
   CV-3 exhibits in the stopping rule.
4. **Exercise `move_max > move`** so the trust region can genuinely expand
   (§14.6).

## Appendix A — "same basin?" is a per-route question

Superseded by §10.1, which carries the full table and the correction. The short
form: every LP-family terminal design at 160×20 is the same topology to
r ≥ 0.956, including the diagnostic with the move-ceiling controller disabled;
the MMA-family designs sit at r ≈ 0.80–0.86 from them.

That distinction is why "the topology looks right" is not evidence about the
stopping test, and why this audit never uses it as evidence. `ks/mma` makes the
point sharpest: highest ω₁ of any run, a different topology from the certified
design, and 40× the residual ascent.

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
