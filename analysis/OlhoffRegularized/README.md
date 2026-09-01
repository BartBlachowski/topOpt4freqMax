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
runCfg.max_outer_iterations = 1000; % major topology linearizations
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
and a scaled predicted-improvement stationarity test. Reaching the minimum trust
radius is not itself convergence. The terminal statuses are `CONVERGED`,
`CAP_HIT`, and `GLOBALIZATION_STALLED`.

Accurate but persistently low-progress boundary steps trigger a separate trust
contraction (`progress_tolerance`, `progress_persistence`, and
`progress_shrink_factor`). This is what removes the constant-move terminal cycle;
it is distinct from rejection-driven trust contraction.

The most important telemetry is available under:

```matlab
info.history.accepted
info.history.trustUsed
info.history.acceptanceRatio
info.history.predictedImprovement
info.history.actualImprovement
info.history.predictedSlope
info.history.densityChangeInf
info.history.densityChangeRms
info.history.innerConverged
info.iterations
```
