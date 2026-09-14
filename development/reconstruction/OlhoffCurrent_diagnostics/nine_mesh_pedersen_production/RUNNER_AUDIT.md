# RUNNER_AUDIT — `examples/Performance/performance_comparison.m` (Part 3)

The audit read the runner at HEAD `b21483b` (SHA-256 `5ed7ddc3…`) and the helpers it calls for the Olhoff column. The harness was not redesigned, and no harness or scientific file was changed, apart from the two run-selection literals described in §3.

## 1. How the runner works

| question | answer (file:line at HEAD) |
|---|---|
| **Entry form** | A MATLAB **script** that starts with `clear; clc; close all;` (`performance_comparison.m:24`). It has no function signature, no argument parsing and no environment-variable interface. The header says so directly (`:3-9`): *"Everything that decides WHAT is measured is in the USER CONFIGURATION block … in literals you can edit"*, and *"Reproducibility comes from RECORDING the configuration that ran (benchmark_manifest.json)"*. |
| **Method selection** | `cfg.methods = struct('proposed', …, 'yuksel', …, 'olhoff', …)` (`:92`), filtered into `methodKeys` (`:204-205`). `confbench_preflight` accepts any non-empty subset (check "at least one method enabled"). The gate's harness test already exported an Olhoff-only record set (`postmerge_campaign_gate/scripts/pmg_harness.m`). This is an **existing, documented Olhoff-only selector (Part 3B)**. |
| **Mesh selection** | The literal `cfg.resolutions` matrix (`:49-59`). At HEAD it is exactly the nine campaign meshes in ascending order. `CAMPAIGN_MESHES` (`:168`) derives `cfg.performanceCampaign = 1` only when the whole nine-row matrix runs with no truncation (`:184-186`). |
| **Preset selection** | `confbench_olhoff_preset()` returns the literal `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`. Preflight refuses unless that name is also the production preset recorded in `PROVENANCE.json` and passes the per-preset field assertions (`confbench_olhoff_assertions`). |
| **Olhoff configuration resolution** | Before any solve, the runner calls `confbench_method_config('olhoff', nelx, nely, outputDir)` (`:236-248`). That function calls `olhoffcurrent_config(nelx, nely, 'Preset', name)` under the `olhoffcurrent_paths` guard, and `outputDir` is **not used** for Olhoff. The **solve** re-resolves the configuration inside `confbench_run_case` → `runOlhoff`, which calls `olhoffcurrent_run(nelx, nely, 'Preset', mcfg.olhoff_preset)` → `olhoffcurrent_config(nelx, nely, 'Preset', preset.name)` → `olhoffSolve(cfg)`. Part 4 therefore verifies both routes, and the in-run PRECHECK hashes the `cfg` object actually handed to `olhoffSolve`. |
| **Output routing** | `cfg.outputDir = ''` becomes `examples/Performance/conference_benchmark/<cfg.runLabel>` (`:199-202`). The committed label is `campaign_9mesh_r2`, a directory holding **tracked** historical results. A rerun under that label overwrites them, and preflight only prints a note about it (`confbench_preflight` §10). |
| **Timing** | `total_wall_time_s` is a caller-side `tic/toc` around `olhoffSolve` only (`olhoffcurrent_run.m:124-126`). Outer, inner, eigen and gradient times come from `res.hist` and are aggregated in `olhoffcurrent_run.m` (`accounting`): Σ, mean and median of tOuter, eigen time per outer, and so on. The warm-up, evaluator, export, plots and topology images all run outside that timer. |
| **Per-iteration history** | **Not retained.** `olhoffcurrent_run` reduces `res.hist` to aggregates plus final values (`stopping`). The record keeps `x`, `omega(1:3)`, `effective_config`, `solver_log`, `accounting` and `stopping`. The gate disclosed this in `postmerge_campaign_gate/TELEMETRY_READINESS.md` ("Records keep tOuter aggregates … not the per-iteration vector"). `runtime.verbose = false` is part of the frozen hash, so stdout carries no per-iteration table either. |
| **Failure handling** | `confbench_run_case` catches solver exceptions into `status = RUN_ERROR`, and the loop continues with the next mesh. A nonzero count of unconverged nested MMA solves is forced to `SOLVER_FAILURE`. CAP_HIT is decided in `olhoffcurrent_run` by `nOuter >= maxOuter` without the convergence log line. `confbench_scaling_fit` and `confbench_export` run **before** `benchmark_records.mat` is saved (`:407-470`), and neither is inside try/catch. An exception at the end therefore loses every in-memory record. |
| **Repeated runs** | There is no resume and no per-mesh persistence. A rerun re-executes every enabled (method, mesh) pair and writes on top of the same label's directory. |
| **Process model** | One MATLAB process handles the warm-up (48×6, 5 outer, discarded) and then the full matrix, in `cfg.resolutions` order (`:347-382`). |

## 2. Decisions

1. **Selection mechanism (Part 3B).** The existing `cfg.methods` literal selects Olhoff only. Because the runner begins with `clear`, the literal can only be set by editing the `USER CONFIGURATION` block. No new selector was added (Part 3C was not needed), and no harness logic was changed.
2. **Output isolation (Part 5).** `cfg.runLabel = 'nine_mesh_pedersen_b21483b'` sends output to the fresh canonical root `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/`. The launcher refuses to start if that root exists.
3. **Process model (Part 9).** The runner is designed to run the nine-row campaign in **one** process, and `performance_campaign = 1` exists only in that form. That behaviour is kept and the order is the runner's ascending order (Part 7). Nine single-row invocations would each derive `performance_campaign = 0` and repeat the warm-up. That would change execution semantics for convenience, which the task prohibits.
4. **Retaining history without code changes (Part 9).** Three **conditional breakpoints** are armed in the committed `olhoffcurrent_run.m`, in the same process, before `run(...)` (`scripts/nmp_arm_hooks.m`):
   - line 124, `tCall = tic;` → `nmp_hook_pre`;
   - line 128, after `callWall = toc(tCall)` → `nmp_hook_tap`;
   - line 249, `if out.is_warmup` → `nmp_hook_post`.

   Each condition function writes files and returns `false`. The hooks supply the per-mesh PRECHECK/POSTCHECK inside the single process (Parts 8 and 12), durable per-mesh run directories (Part 5), and the solver's full `res`, apart from the FE model, for each production solve (Part 9).
5. **Fail-closed behaviour.**
   - A failed campaign precheck calls `exit(3)` before the solver timer starts, so the mesh is not solved.
   - A postcheck that finds changed or unverifiable identity calls `exit(4)`.
   - The launcher refuses on any lock, HEAD, runner-edit, tracked-dirty or fresh-output violation.

## 3. The run-selection edit (the only tracked change during the campaign)

`evidence/campaign_runner_edit.patch` (SHA-256 `57e20f0c…`) changes two lines:

```diff
-cfg.methods = struct('proposed', true, 'yuksel', true, 'olhoff', true);
+cfg.methods = struct('proposed', false, 'yuksel', false, 'olhoff', true);
-cfg.runLabel  = 'campaign_9mesh_r2';
+cfg.runLabel  = 'nine_mesh_pedersen_b21483b';
```

The edited file hashes to `05d043c3…`, pinned in `CAMPAIGN_LOCK.json`. Neither literal reaches the Olhoff configuration. Part 4 (`evidence/RUNNER_CONFIG_IDENTITY.json`) shows all nine hashes unchanged through the runner route with the campaign `outputDir`. After the campaign process exits, the file is restored to its committed bytes (`5ed7ddc3…`) and the restore is verified.

## 4. Mechanics evidence gathered before the campaign

| probe | where | result |
|---|---|---|
| Conditional breakpoint in `-batch` | scratchpad `tapdemo` | Fires with the function's local variables. Survives a script's `clear; clc; close all`, `addpath`, and `path()` restore. |
| Breakpoint cost to the callee | scratchpad, 3e7-iteration scalar loop in a separate file | Timings were 3.889/3.857/3.859/3.862 s without breakpoints, 3.869/3.861/3.861/3.860 s with them, and 3.859–3.865 s after `dbclear`. No measurable effect. |
| `exit(3)` inside a condition | scratchpad | Process exit code 3; the guarded line never ran. |
| Condition that throws | scratchpad | MATLAB warns and **stops in the debugger**, which hangs `-batch`. Every hook body is therefore inside try/catch, and hook arguments are only variables guaranteed to be assigned at those lines. |
| Smoke through the real runner (non-scientific) | `examples/Performance/conference_benchmark/smoke_nine_mesh_pedersen_b21483b/`, `logs/smoke_*` | Olhoff only, 160×20, `maxOuterOverride = 3`. PREFLIGHT PASS. All three hooks fired for the warm-up and for 160×20. The precheck correctly flagged the cap-3 configuration hash as not frozen, and nothing else. The tap matched the runner accounting bit for bit (ρ hash, ω₁₋₃, outer, inner, Σ tOuter, Σ tInner, Σ tEig, callWall). The breakpoints were still armed when the runner returned. Exit code 0. |
| Synthetic end path (no solve) | `evidence/SYNTHETIC_ENDPATH.json` | Nine Olhoff-only records with `performance_campaign = 1` passed through the scaling fit, manifest, export, `-v7.3` save, complexity plots and nine topology images. PASS. |
