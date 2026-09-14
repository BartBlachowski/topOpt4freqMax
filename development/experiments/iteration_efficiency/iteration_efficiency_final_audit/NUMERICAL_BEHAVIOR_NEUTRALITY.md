# Numerical behaviour neutrality of the native optimizer modifications

Question: can enabling instrumentation change what the optimizer computes?
Answer: **No.** Proved by matched execution, not by reading the report.

## 1. Semantic diff — every changed executable line classified

### `Matlab/reproduction2007/algo/innerLoopLP.m` (+9 −3)

| change | class | can affect numerical state? |
|---|---|---|
| `[x,~,flag] = linprog(...)` → `[x,~,flag,output] = linprog(...)` | output capture | No. `linprog` returns the same `x`/`flag`; requesting a 4th output does not change the solve. |
| `lpIterations` extracted from `output.iterations` when present | telemetry | No. Read-only. |
| `st` gains field `lpIterations` | telemetry | No. Callers index `st` by name. |

No change to the LP objective `f`, the inequality set `A,b`, the equality set `Aeq,beq`, the bounds `lb,ub`, or `optimoptions`.

### `Matlab/reproduction2007/algo/olhoffOpt.m` (+34 −4)

| change | class | can affect numerical state? |
|---|---|---|
| `hist` gains `lpFlag`, `lpBackendIterations`, `innerDxHist`, `innerRelHist` | telemetry | No. Write-only history fields. |
| `captureTrajectory` / `captureInnerHistories` flags and `rhoSnapshots` preallocation | storage | No. `rho` itself is never read back from `rhoSnapshots`. |
| `extendBeyondNativeStop` control | **loop control** | Changes *when the loop stops*, never *what an update computes*. With the flag false the break is byte-for-byte the original behaviour; with it true the optimizer simply keeps taking the same updates past its own stop test. |
| convergence log string reworded | cosmetic | No. Grep confirms no consumer parses that string. |
| `res` gains `rho_snapshots`, `status`, `native_stop_iteration`, `extended_beyond_native_stop`, `trajectory_dtype` | telemetry | No. |

### `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` (+21 −3)

| change | class | can affect numerical state? |
|---|---|---|
| `authoritativeTrajectory` selects `double` snapshots; historical `single` preserved as default | storage | **No — and this is the key point.** The original line was `rho = min(1,max(cfg.rhomin,rho+drho)); snapshots(:,outer+1) = single(rho);`. The optimizer's own `rho` was *always* double; only the stored copy was downcast. The float32 loss was purely a storage artefact, so promoting storage to double cannot perturb the trajectory. |
| `captureTrajectory=false` path (`snapshots=zeros(NE,0)`) | storage | No. Used by the timing replays to remove I/O and memory pressure. |
| `hist.lpBackendIterations(outer)=st.lpIterations` | telemetry | No. |

## 2. Matched behaviour-neutrality execution

96×12, 40 outer updates, S1 policy, `maxNumCompThreads(1)`. Three runs compared: **PRE** = `git show HEAD:` versions of `olhoffOptStabilized.m` and `innerLoopLP.m` shadowing the instrumented files on the path; **ON** = instrumented with `captureTrajectory=true, authoritativeTrajectory=true`; **OFF** = instrumented with `captureTrajectory=false`.

| quantity | ON vs OFF | ON vs PRE |
|---|---|---|
| final `rho` (bit-identical, `isequal`) | **yes** | **yes** |
| final `rho` max abs difference | **0** | **0** |
| `omega` | yes | yes |
| `lambda` | — | yes |
| `hist.dxOuter` | — | yes |
| `hist.beta` | — | yes |
| `hist.lpFlag` | — | yes |
| `hist.nInner` | — | yes |
| `hist.N` | — | yes |
| `hist.policyStage` | — | yes |
| `hist.moveLimit` | — | yes |
| `nOuter` / `status` | 40 / CAP_HIT | 40 / CAP_HIT |

**Verdict: bit-identical.** Enabling observation/capture does not change the optimizer.

## 3. Specific neutrality requirements

- **Observer is write-only.** `tools/Matlab/topopt_history_record.m:90` calls `observer.callback(rec, k, observer);` with no output assigned. `observer_capture.m` contains no `assignin`, `evalin`, or `global`, and returns nothing. MATLAB copy-on-write means it cannot mutate the caller's `rec`. `topopt_history_record.m` is itself unmodified from HEAD (verified: empty diff), as are `topopt_freq.m` and `top99neo_inertial_freq.m`.
- **Capture cannot modify optimizer variables.** `rhoSnapshots`/`snapshots` are written but never read back into the update.
- **Route selector cannot execute both routes.** `localOlhoff` is a `switch` on `variant` with a single call per branch and an `otherwise` that errors. Under `variant='mma'` a live run recorded `hist.lpFlag` all-NaN — **zero LP calls**. Under `variant='lp'` `olhoffOpt` is never reached.
- **Diagnostic accounting cannot alter stopping.** `olhoffOptStabilized` has no convergence break at all (only `SOLVER_FAILURE`); `olhoffOpt`'s break is gated solely by `dxOuter < cfg.tolOuter` and `extendBeyondNativeStop`, neither of which reads any telemetry field.
- **Lossless trajectory capture does not alter the state update.** Proved above; additionally `run_trajectory.m` asserts at runtime that `isequal(r.rho, r.rho_snapshots(:,end))`, which held in every run performed.

## 4. Storage precision, measured

| path | `class(rho_snapshots)` | max abs representation error vs authoritative `rho` |
|---|---|---|
| instrumented, `authoritativeTrajectory=true` | `double` | **0** |
| historical default (`single`) | `single` | 2.62e-08 |

The historical single path is preserved but is not reachable from the production harness: `run_trajectory.localOlhoff` sets `authoritativeTrajectory=true` for LP, `olhoffOpt` has no single path at all, and `run_trajectory` then asserts `isa(tr.x_post,'double')`, `validate_results` rejects any row whose `trajectory_dtype ~= 'double'`, and `RESULT_SCHEMA.json` pins `trajectory_dtype` to `const: "double"`. **No authoritative float32 path is reachable by production.**

## 5. LP backend iteration telemetry is genuine

A concern worth ruling out: if `output.iterations` were uniformly 1 it would be indistinguishable from the `nInner=1` relabelling the contract forbids. Measured over 120 outer updates at 160×20:

- unique values `[1 2 3 4 5 6 8 10 11 12 13]`
- `all(v==1)` → **false**
- `sum(v) = 322` against `120` LP calls → the two columns are **not** the same number

HiGHS dual-simplex iteration counts are genuine and variable. `olhoff_lp_backend_iterations` may be reported as backend work. (Early outers do return 1; a short trajectory alone would have looked degenerate.)

**Every native modification is behaviour-neutral. This is not a blocker.**
