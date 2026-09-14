# B-3 — cell-local failure containment

## Root cause

`+iefinal/run.m::localProduction` ran the mesh × method loop with no `try`/`catch`, and `iefinal.write_results` executed only after the loop finished. A genuine method failure — `run_trajectory`'s `assert(~strcmp(r.status,'SOLVER_FAILURE'), 'iefinal:OptimizerFailure', ...)`, documented as actually occurring (contract: `olhoff_800x100: RUN_ERROR`, `GENERIC_LP_ITERATION_LIMIT_ONLY`) — propagated out and terminated the campaign before any row was written, destroying every completed cell and emitting no `RUN_ERROR` evidence.

## Correction

A **production cell** is one `method × mesh × route/variant` execution identity. The cell body moved into `localCell`; the loop wraps it:

```
try   -> localCell
catch -> classify_cell_failure
         cell-local  ? RUN_ERROR rows + failure record + continue
         integrity   ? rethrow (campaign-fatal)
finally-> validate_results, accumulate, checkpoint rows to disk
```

`allRows` and the failure list are checkpointed to `analysis/rows_checkpoint.mat` after **every** cell, so no later failure can destroy completed work. Failed cells also write `reference/<id>/run_error_result.mat` and, at campaign end, `analysis/RUN_ERRORS.csv` and `analysis/run_errors.json` with the full exception report.

### Containment boundary — explicit allowlist

`+iefinal/classify_cell_failure.m` decides. It is an **allowlist**: anything not named is integrity, so an unforeseen error can never be silently reclassified as a scientific result.

**Cell-local scientific execution failure → `RUN_ERROR`, campaign continues**

| identifier | meaning |
|---|---|
| `iefinal:OptimizerFailure` | LP returned `SOLVER_FAILURE` |
| `iefinal:MissingTrajectory` | method produced no eligible states |
| `iefinal:NonfiniteTrajectory` | method produced a nonfinite state |
| `iefinal:MissingReferenceTrajectory` | native run too short for the reference horizon |
| `MATLAB:eigs:*`, `MATLAB:svds:*`, `MATLAB:decomposition:*`, `repro2007:*` | failure inside the native numerics |
| `MATLAB:nomem`, `MATLAB:pmaxsize`, `MATLAB:array:SizeLimitExceeded`, `MATLAB:singularMatrix`, `MATLAB:illConditionedMatrix`, `MATLAB:posdef` | resource / conditioning exhaustion |

**Integrity failure → campaign-fatal (rethrown)**

`iefinal:PreflightFailed`, `iefinal:ResultSchema`, `iefinal:OutputCollision`, `iefinal:ManifestSource`, `iefinal:FingerprintMismatch`, `iefinal:TrajectoryPrecision`, `iefinal:TrajectoryIdentity`, `iefinal:TrajectoryShape`, `iefinal:StateIndex`, `iefinal:TimingEvidenceMissing`, `iefinal:TimingWorkMismatch`, all `ie2a:*`, and ordinary programming errors (`MATLAB:badsubscript`, `MATLAB:undefinedFunction`, `MATLAB:nonExistentField`, …).

`FingerprintMismatch` is deliberately **fatal**, not `RUN_ERROR`: the study design records that a reference/measurement prefix mismatch "is an implementation failure, never a new reference opportunity."

### The RUN_ERROR row

`+iefinal/build_error_rows.m` emits the full 9-row q/P block. `+iefinal/empty_result_row.m` is now the single shared template, so a RUN_ERROR row and a successful row are structurally identical.

- **Preserved**: method, variant, mesh, `nelx`/`nely`/`element_count`, `q`, `P`, `B0`, `B_ref`, `evaluator_id`, `contract_hash`, `source_hashes`.
- **Recorded**: `status = 'RUN_ERROR'`, `censoring_reason = 'RUN_ERROR'`, `error_identifier`, `error_message` (two fields added to `RESULT_SCHEMA.json`).
- **N/A**: `k_enter`, `k_cert`, `b_ref`, `B_meas`, `E1`, `E2`, `E3`, `Q`, `topology_pass`, `volume_pass`, `hard_gate_pass`, and every accounting and timing field — all `NaN`, never 0.
- `trajectory_dtype` carries the campaign's declared authoritative storage policy (a schema `const`), not a measurement of the failed cell.

`validate_results` now **enforces** this: a RUN_ERROR row without an `error_identifier`, or with any non-`NaN` scientific quantity, is rejected with `iefinal:ResultSchema`. Both rejections are tested.

No special handling exists for 800×100 anywhere; the mechanism is mesh- and method-agnostic.

## Evidence — end-to-end campaign simulation

`verify_b3_containment.m` reproduces `localProduction`'s exact control flow over 3 production-floor meshes × 2 methods, injecting a genuine method failure at an **early** mesh (Olhoff-LP at 160×20, `iefinal:OptimizerFailure`) and at the **final** mesh (Proposed at 320×40, `MATLAB:eigs:ARPACKroutineError`).

| check | result |
|---|---|
| cells executed / total | **6 / 6** — campaign completed |
| failed cells | 2 |
| RUN_ERROR rows / PASS rows | 18 / 36 (9 per cell) |
| every RUN_ERROR row has an identifier | ✅ |
| every scientific quantity N/A | ✅ |
| method / mesh / route / contract identity preserved | ✅ |
| early-mesh failure then later success | ✅ |
| final-mesh failure contained | ✅ |
| results table rows | 54, including RUN_ERROR rows |
| S_common after failures | `{7200}`, `n = 1` → infeasible, failed meshes excluded ✅ |
| topology cells rendered / skipped | 12 / 4 — failed cells drawn as unavailable, none fabricated ✅ |
| integrity failures still fatal | ✅ |
| per-cell checkpoint written | ✅ |

Method-level coverage: Proposed and Olhoff-LP failures are exercised end-to-end above; Yuksel and Olhoff-MMA use the identical mechanism and their identifiers are covered by the classifier test. Output isolation is unchanged (`new_run_directory` still asserts `~isfolder` before `mkdir`) and is covered by `testOutputIsolation`.

### Scope of the evidence

`localProduction` itself cannot be executed without unlocking production, which this task must not do. The verification therefore drives the same public components through the same control flow, and the loop structure in `run.m` is additionally asserted statically (try / `classify_cell_failure` / `rethrow` on integrity / `build_error_rows` / checkpoint / continue). This is the boundary of the evidence and is stated rather than glossed.

## Verdict: **CLOSED**
