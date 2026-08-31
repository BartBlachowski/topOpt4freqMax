# Preflight and fail-closed negative controls

All controls below were **executed** in MATLAB R2025b against the actual harness, not inferred from documentation.

## 1. Preflight controls (`iefinal.preflight`)

| # | control | injected | outcome | verdict |
|---:|---|---|---|---|
| 1 | baseline | unmodified smoke config | `pass = true` | correct |
| 2 | **stale Candidate-C hash** | `common_evaluator.sha256 = 000…0` | `iefinal:PreflightFailed` | **rejected** |
| 3 | **wrong contract** | `scientific_contract.sha256 = aaa…a` | `iefinal:PreflightFailed` | **rejected** |
| 4 | stale topology-gate hash | `topology.sha256 = 111…1` | `iefinal:PreflightFailed` | **rejected** |
| 5 | **float32 authoritative policy** | `trajectory.authoritative_dtype = 'single'` | `iefinal:PreflightFailed` | **rejected** |
| 6 | stale four-mesh list | `production_meshes = 4 meshes` | `iefinal:PreflightFailed` | **rejected** |
| 7 | manifest self-authorization | `production_authorized = true` | `iefinal:PreflightFailed` (`no_self_authorization`) | **rejected** |
| 8 | production lock | `runMode = 'production'` | `iefinal:PreflightFailed: production_authorization` | **locked** |

Preflight also hash-verifies all four method sources and all ten pinned components on every run; each was confirmed matching (`SOURCE_IDENTITY_AUDIT.md`).

## 2. Evaluator and reference fail-closed controls

| # | control | outcome | verdict |
|---:|---|---|---|
| 9 | **unresolved structural mode** — all-invalid evaluator | `reference_phase` → `STRUCTURAL_MODE_NOT_FOUND` | **fails closed** |
| 10 | injected eigensolver failure (`InjectEigensolverFailure`) | `evaluate_common` → `STRUCTURAL_MODE_NOT_FOUND` | **fails closed** |
| 11 | injected invalid eigenpairs (`InjectInvalidEigenpairs`) | `evaluate_common` → `STRUCTURAL_MODE_NOT_FOUND` | **fails closed** |
| 12 | **missing reference** — trajectory shorter than `L_ref` | `REFERENCE_NOT_ESTABLISHED`, `b_ref = NaN` | **fails closed, no cap fallback** |
| 13 | valid reference | `PASS`, `b_ref = 600` (the frozen minimum) | correct |

Control 12 confirms there is no terminal-cap fallback: when no block endpoint qualifies, the harness reports `REFERENCE_NOT_ESTABLISHED` rather than substituting a horizon value.

## 3. Result-schema controls (`iefinal.validate_results`)

| # | control | outcome | verdict |
|---:|---|---|---|
| 14 | **invalid result schema** — `trajectory_dtype = 'single'` | `iefinal:ResultSchema` | **rejected** |
| 15 | stale `evaluator_id` | `iefinal:ResultSchema` | **rejected** |
| 16 | LP row carrying MMA accounting | `iefinal:ResultSchema` | **rejected** |
| 17 | MMA row carrying non-zero LP calls | `iefinal:ResultSchema` | **rejected** |

`write_results` re-validates before writing, so no invalid row can reach `results.json` / `results.csv`.

## 4. Output-isolation control

| # | control | outcome | verdict |
|---:|---|---|---|
| 18 | observer output outside the allowed roots | `ie2a:OutputIsolation` | **rejected** |
| 19 | run directory collision | `iefinal:OutputCollision` asserted before `mkdir` | **refuses to overwrite** |

## 5. Runtime trajectory-integrity controls (code-path verified)

| control | mechanism |
|---|---|
| non-double authoritative trajectory | `run_trajectory`: `assert(isa(tr.x_initial,'double') && isa(tr.x_post,'double'))` |
| stored terminal state ≠ optimizer state | `run_trajectory`: `assert(isequal(r.rho, r.rho_snapshots(:,end)))` — held in every run performed |
| wrong state indexing | `assert(isequal(tr.state_index, (0:N).'))` |
| reference/measurement prefix divergence | `run.m`: `ie2a.trajectory_fingerprint` equality assert at every shared count |
| MMA route provenance | `assert(strcmpi(cfg.innerSolver,'mma') && cfg.offDiag)` |
| stale MMA filter radius | `assert(abs(cfg.rminEl - 0.06/(b/nely)) < 1e-12)` on the **post-call** config |

**All 19 executed negative controls pass.** Fail-closed behaviour is genuine and not merely documented.

## 6. Gap

Two of these integrity controls (`iefinal:OptimizerFailure`, `iefinal:FingerprintMismatch`) fail closed by *throwing*, which aborts the entire nine-mesh campaign rather than recording the frozen `RUN_ERROR` / `FINGERPRINT_MISMATCH` status the contract defines. See finding **F-03**. Failing closed is right; failing the whole campaign is not.
