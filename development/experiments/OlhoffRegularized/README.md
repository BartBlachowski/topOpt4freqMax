# OlhoffRegularized

`OlhoffRegularized` is a separate, globalized implementation built from the
verified FE, filtering, generalized-gradient, LP, and MMA primitives in
`Matlab/reproduction2007`. It does not alter or replace the fixed-step historical
implementation in `analysis/OlhoffReproduced2007`.

The shared entry point is:

```matlab
[rho,omega,info] = topopt_olhoff_regularized( ...
    nelx,nely,volfrac,penal,rmin,move,bcType,runCfg);
```

Three thin runners reproduce the same problem surfaces:

```matlab
run_regularized_simply_supported
run_regularized_fixed_pinned
run_regularized_cantilever
```

Set `optimizer` and `formulation` before invoking a runner to override its
default without editing it:

```matlab
optimizer = "mma";
formulation = "ks";
run_regularized_fixed_pinned
```

## Routes and scientific labels

| formulation | optimizer | route |
|---|---|---|
| `olhoff` | `lp` | Eq. (16)/(22) equality-constrained LP, with added trust-region globalization |
| `olhoff` | `mma` | full Olhoff sub-eigenvalue inner problem, with mandatory inner convergence and added globalization |
| `ks` | `lp` | KS-regularized sequential LP; Olhoff-inspired, not paper-literal |
| `ks` | `mma` | KS-regularized nested MMA; Olhoff-inspired, not paper-literal |

The Olhoff routes preserve the local subproblem formulations but the adaptive
accept/reject controller is an explicitly disclosed numerical extension. The KS
routes change the spectral objective to a smooth lower aggregate and must never
be reported as an exact Olhoff reproduction.

The regularized defaults use the C1 mass interpolation `4b` and a density filter
with an exact chain-rule sensitivity and filtered-volume gradient. The historical
discontinuous mass law `4` and sensitivity-filter modes `diag`/`all` remain
available by explicit request, but they can make trial-step globalization stall.

## Authoritative iteration limits

Both limits live in `runCfg`; there is no positional outer-iteration argument:

```matlab
runCfg.max_outer_iterations = 1000; % major topology linearizations (runner-specific)
runCfg.max_inner_iterations = 500;  % nested MMA iterations per trial
runCfg.max_trial_steps = 8;         % trust-region attempts per outer iteration
```

LP requires one inner solver call per trial. `max_inner_iterations` applies to
the MMA routes. Trial eigensolves and rejected steps are counted separately in
`info.iterations`.

## Globalization and stopping

Every proposed update is checked by a trial eigensolve. The trust radius shrinks
when actual improvement is inconsistent with the local prediction, and grows
only after an accurate accepted boundary step. Failed or cap-hit MMA inner solves
are rejected rather than applied.

By default `move_max` equals the input `move`, so globalization cannot silently
become more aggressive than the runner configuration. A larger ceiling must be
requested explicitly.

Convergence requires persistent agreement of density change, objective change,
and a first-order **stationarity certificate**. Reaching the minimum trust
radius is not itself convergence. The terminal statuses are `CONVERGED`,
`CAP_HIT`, and `GLOBALIZATION_STALLED`.

### The stationarity certificate

Density change, RMS density change and relative objective change are all
step-size proxies: the move-limit controller can drive every one of them to zero
without the design being anywhere near stationary. The certificate is the only
part of the stopping test that cannot be satisfied that way, so it is specified
independently of the controller:

* it is evaluated at a **fixed reference radius** `certificate_radius`
  (default `move_max`), never at the current trust radius or move ceiling;
* its eigenvalue cluster is **self-consistent with its own prediction**, not
  the step model's deliberately loose `tol_mult`. It is seeded at the exact
  multiplicity (`certificate_mult_tol`, default `1e-6`) — because the
  directional derivative of the ordered `lambda_n` is the smallest eigenvalue of
  the sub-eigenvalue matrix over the *exactly* degenerate cluster, and folding a
  strictly separated mode into that cluster understates criticality — and then
  **grown while the model's own predicted gain would carry `lambda_n` past the
  first excluded eigenvalue**, because such a model is not a model of the
  *ordered* `lambda_n` over a step of that size. The fixed point introduces no
  new constant and collapses to the exact multiplicity whenever the gaps are
  large compared with the attainable gain. `info.history.certificateN`,
  `certificateNextGap` and `certificateGrown` record it;
* it is defined on the physical objective `lambda_n` for every formulation, so
  the KS routes are certified against the eigenfrequency and not against their
  own smooth aggregate.

`certificateSlope * certificate_radius` is the local model's best **relative**
gain in `lambda_n` for one step of the reference radius. The only threshold
consistent with the objective tolerance the same test declares is therefore

```matlab
stationarity_tol = objective_tol / certificate_radius     % 1e-5/5e-3 = 2e-3
```

which is the default. `stationarity_tol` may only be set **tighter**; a looser
request is clamped (with a warning) for both the convergence declaration and the
move-ceiling controller. Those two must use the same number: a controller
allowed to contract at a looser tolerance collapses the ceiling onto `move_min`
without the stopping test ever becoming satisfiable, which ends in a guaranteed
`GLOBALIZATION_STALLED`.

For an exactly degenerate cluster of size >= 2 the certificate uses the
Eq. (22) equality-restricted LP, whose feasible set is contained in the true
one, so its value is then a lower bound on the true criticality. It is combined
with the route's own model slope by `max()`, and the unrestricted check lives in
`audit/` (see `AUDIT_REPORT.md`).

Read `AUDIT_REPORT.md` before citing any `CONVERGED` status from this
implementation.

Accurate late-stage steps are handled by a persistent move-limit continuation,
separate from rejection-driven trust contraction.  Normal trust adaptation may
shrink and regrow only below `moveCeiling`; the ceiling itself never increases.
After a dwell period, a ceiling contraction requires all of the following over
an accepted-update window:

- cumulative relative objective improvement no greater than
  `progress_window*progress_tolerance`;
- no single relative improvement above `progress_spike_tolerance`;
- every scaled predicted-improvement slope below `stationarity_tol`.

The relevant controls are:

```matlab
runCfg.progress_tolerance = 1e-4;
runCfg.progress_window = 10;
runCfg.progress_spike_tolerance = 3e-4;
runCfg.progress_dwell = 20;
runCfg.progress_shrink_factor = 0.5;
```

After contraction, the progress window is cleared and the dwell counter is
restarted. `progress_persistence` remains a compatibility alias for
`progress_window`, but new configurations should use the explicit name.
Move continuation does not imply convergence: the unchanged stopping certificate
still requires persistent density-infinity, density-RMS, objective-change, and
scaled-stationarity passes. The fixed--pinned runner deliberately uses 1600 outer
iterations as diagnostic headroom; a smaller production cap should be frozen only
after observing a validated natural convergence iteration.

## Audit

`AUDIT_REPORT.md` records an independent audit of this implementation: an
end-to-end trace of the `olhoff`/`mma` route, a mathematical audit of the
stopping test, an independent fixed-design stationarity certificate with
physical fixed-step verification, and the scientific runs at 160x20, 240x30 and
320x40. Audit-only scripts and results are isolated under `audit/`. Five
corrections came out of it (runner route override; certificate radius, cluster
and threshold; the self-consistency of that cluster; contraction/convergence
tolerance split); the before/after evidence is in the report.

The most important telemetry is available under:

```matlab
info.history.accepted
info.history.trustUsed
info.history.moveCeilingUsed
info.history.moveCeilingNext
info.history.acceptanceRatio
info.history.predictedImprovement
info.history.actualImprovement
info.history.predictedSlope
info.history.certificateSlope
info.history.stationarityMeasure
info.history.certificateN
info.history.certificateRelativeGain
info.history.certificateNextGap
info.history.certificateGrown
info.history.densityChangeInf
info.history.densityChangeRms
info.history.innerConverged
info.history.windowCumulativeProgress
info.history.windowMaxStepProgress
info.history.progressStationarityPass
info.history.moveCeilingContracted
info.iterations
```
