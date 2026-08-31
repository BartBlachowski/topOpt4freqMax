# WP2 — Olhoff precision path, traced from source

Protected runner: `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m`
(SHA-256 `95240cf60f82b40f8e5e892b9eea9b20a8fd3744b5eca6fdfc8dde2698d82aec`, unchanged).
Not to be confused with the modified fork
`analysis/performance_campaign_targeted_replays/olhoffOptStabilizedDiagnostic.m`, which
was not used and from which no metric in this report derives.

## Line-level trace

| # | Question | Answer | Citation |
|---|---|---|---|
| 1 | precision of active `rho` | double | `:10` `rho=cfg.rho0*ones(NE,1)` |
| 2 | precision after accepted update | double | `:65` `rho=min(1,max(cfg.rhomin,rho+drho))` |
| 3 | precision of `res.rho` | double | `:88` packs `rho` unchanged |
| 4 | precision of `res.rho_snapshots` | single | `:21` `NaN(NE,cfg.maxOuter+1,'single')` |
| 5 | source location of the cast | `:21` and `:65` | `snapshots(:,1)=single(rho)`, `snapshots(:,outer+1)=single(rho)` |
| 6 | does the cast affect storage only | yes | the cast writes into `snapshots`; `rho` is never reassigned from it |
| 7 | does stored single feed back | no | `snapshots` is read only at `:80` (truncation) and `:90` (packing) |

## Dataflow

    rho (double, :10)
      |
      +-- LP context ctx.rho (:56) --> innerLoopLP --> drho ------+
      |                                                          |
      +-- assemble2D / genGrad / applyFilter (:25,:41,:53)        |
      |                                                          v
      +<--------------- rho = min(1,max(rhomin, rho+drho)) (:65) -+
      |
      +--> single(rho) --> snapshots(:,outer+1) (:65)   [OBSERVATIONAL SINK]
      |                        |
      |                        +--> snapshots(:,1:nDone+1) (:80)
      |                        +--> res.rho_snapshots (:90) --> offline analysis
      |
      +--> res.rho (:88, double)  --> final accepted state

Every consumer of `rho` inside the loop (`assemble2D`, `eigSolve`, `genGrad`,
`applyFilter`, `innerLoopLP` via `ctx.rho`) receives the double variable. `snapshots` is
write-only within the loop. There is no read of `snapshots` into any quantity that
influences gradients, LP construction, stopping, mode tracking or a future design update.

## Classification

**OBSERVATIONAL ONLY.** The optimizer evolves entirely in native double precision; the
single cast is a storage/logging sink. Q1 of the qualification criteria is satisfied.

## Consequence exploited by this qualification

Because the cast is observational and the algorithm is deterministic, a run capped at
`cfg.maxOuter = k` returns the genuine double state after k accepted updates in
`res.rho`, and simultaneously the exact single image of that same state in
`res.rho_snapshots(:,end)`. The two are the same state by construction
(`isequal(res.rho_snapshots(:,end), single(res.rho))` holds by lines 65 and 80), so no
cross-run correspondence is needed to form a pair. Cross-run comparison is used only to
prove prefix determinism.

## Storage-shape facts that govern every accessor

`rho_snapshots` is two-dimensional, `NE x (nDone+1)`. Column 1 is the INITIAL density
(`:21`), so the state after k accepted updates is column **k+1**. On the SOLVER_FAILURE
path the loop breaks at `:63` BEFORE the update at `:65`, so no partial state is stored
and `nDone` counts only completed updates; `res.rho` therefore still equals the final
stored column. This is verified per run rather than assumed (WP3/Patch 8).
