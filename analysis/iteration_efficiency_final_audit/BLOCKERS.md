# Blockers

Three findings block the nine-mesh production campaign. Each is narrowly scoped. None requires a new methodology phase, and none touches Candidate-C definitions, reference/persistence definitions, topology definitions, q levels, P, B_ref, B_meas, the production meshes, optimizer mathematics, or the LP/MMA scientific interpretation.

---

## B-1 (CRITICAL) — the Yuksel timing replay does not reproduce the measured computation

**Exact source location**

- `analysis/iteration_efficiency_final/+iefinal/run_timing_firewall.m`, `localOnce`, Yuksel branch: `prm.max_iters = c.horizon` with `prm.stage1_max_iters = 2000`
- `tools/Matlab/run_topopt_from_json.m:535`: `stage1MaxIter = min(stage1MaxIter, maxiter);`

**Observed evidence** (measured, 96×12, single thread)

| path | `max_iters` | stage1 effective | stage1 iterations | stage2 iterations |
|---|---:|---:|---:|---:|
| reference / measurement | 3200 | 2000 | **151** | 3200 |
| timing replay at horizon 150 | 150 | **150** | **150** | 150 |

`timing_replay_reproduces_stage1 = false`.

**Scientific consequence**

The timing horizon for Yuksel is a Stage-2 index, but it silently clamps the Stage-1 budget too. A truncated Stage 1 hands a different design to Stage 2, so the run being timed is not the run whose `k_enter`/`k_cert` are reported, and Stage-1 work is under-counted. This corrupts `native_total_time`, `native_total_time_to_enter`, `native_total_time_to_cert` and `mean_native_iteration_time` — the entire computational-performance half of the study — and does so silently, since nothing compares the replay's stage counts against the trajectory's.

It is not hypothetical: the frozen campaign records Yuksel Stage 1 hitting its cap at **640×80, 720×90 and 800×100**, exactly the meshes where timing matters most.

**Minimum correction**

1. Make the Stage-1 budget independent of the Stage-2 horizon on the timing path. Narrowest change: drop or condition the `min(stage1MaxIter, maxiter)` clamp at `run_topopt_from_json.m:535`. It is behaviour-neutral for the reference/measurement path, where `maxiter = 3200 ≥ stage1_max = 2000` makes `min()` a no-op.
2. Add a hard assertion in `run_timing_firewall.localOnce` that the replay's Stage-1 and Stage-2 counts equal the recorded trajectory's, failing closed on mismatch.

**Minimum re-verification**

- One matched Yuksel reference-vs-timing pair at a mesh where Stage 1 > `k_enter`, showing identical Stage-1 and Stage-2 counts.
- A Proposed + Yuksel reference-run neutrality check before and after the clamp change (bit-identical final design), since `run_topopt_from_json.m` drives both principal non-Olhoff methods.

---

## B-2 (MAJOR) — scaling fits assert "common support" without enforcing it

**Exact source location**

- `analysis/iteration_efficiency_final/+iefinal/fit_scaling_table.m`: `rec = struct(..., 'support', "common", ...)` — hardcoded, with the fit taken over `ix = string(T.method) == methods(i)`, i.e. each method's own meshes
- `analysis/iteration_efficiency_final/+iefinal/synthetic_scaling_validation.m`: `report = struct(..., 'common_support', true, ...)`

**Observed evidence** (measured)

With Proposed present at nine meshes and Olhoff-lp at seven:

| method | `support` | `n_valid` |
|---|---|---:|
| Proposed | `common` | 9 |
| Olhoff-lp | `common` | 7 |

No intersection is ever computed. The smoke test cannot expose this because its synthetic input gives every method every mesh.

**Scientific consequence**

`scaling_fits.csv` is the authoritative source for the paper's scaling table. It would report exponents `p` fitted over different mesh ranges under a label asserting shared support, inviting precisely the comparison the frozen contract forbids:

```json
"common_support_companion_required": true,
"cross_method_comparison_outside_common_support": false
```

Unequal support is the expected case: the contract records Olhoff 800×100 as `RUN_ERROR`, and a native LP iteration-limit failure is documented at 400×50.

**Minimum correction**

Compute the intersection of fit-eligible meshes across the compared methods and emit two labelled fit families — `support="available"` (per method, as now) and `support="common"` (restricted to the intersection). Do not touch `ie2a.fit_power_law`.

**Minimum re-verification**

Rerun `fit_scaling_table` on an unequal-support fixture (9 vs 7 meshes) and confirm two labelled families with correct `n_valid` per family.

---

## B-3 (MAJOR) — one cell failure aborts the whole campaign instead of recording a genuine status

**Exact source location**

- `analysis/iteration_efficiency_final/+iefinal/run.m`, `localProduction`: the mesh × method loop has no `try`/`catch`, and `iefinal.write_results` runs only after it completes
- `analysis/iteration_efficiency_final/+iefinal/run_trajectory.m`: `assert(~strcmp(r.status,'SOLVER_FAILURE'), 'iefinal:OptimizerFailure', ...)`
- `analysis/iteration_efficiency_final/+iefinal/run.m`: the `ie2a.trajectory_fingerprint` equality assert (`iefinal:FingerprintMismatch`)

**Observed evidence**

`olhoffOptStabilized` sets `status='SOLVER_FAILURE'` and breaks when `lpFlag ~= 1`; `run_trajectory` then throws. The frozen contract records `"olhoff_800x100": "RUN_ERROR / E1 N/A / UNVERIFIABLE_AT_PRESENT"` and `"known_olhoff_backend_subclass": "GENERIC_LP_ITERATION_LIMIT_ONLY: dual-simplex-highs returned exit flag 0"`. Project records also note a native LP iteration-limit failure at 400×50, iteration 627.

**Scientific consequence**

A failure at, say, 400×50 raises a MATLAB exception that terminates the run before any rows are written, destroying every completed cell — potentially days of compute. It also violates the requirement that a failing 800×100 cell **record the genuine production failure/status**: the contract defines `RUN_ERROR`, `SOLVER_TERMINATION`, `FINGERPRINT_MISMATCH` and `CAP_CENSORED` as reportable execution classes, and the harness emits none of them. Failing closed is correct; failing the entire campaign is not.

**Minimum correction**

1. Wrap each (method, mesh) cell in `try`/`catch`; on failure emit that cell's nine censored rows carrying the frozen status (`RUN_ERROR` / `SOLVER_TERMINATION` / `FINGERPRINT_MISMATCH`) with the MATLAB error identifier in `censoring_reason`, then continue to the next cell.
2. Checkpoint `allRows` to disk after each cell so a hard crash cannot destroy completed work.

**Minimum re-verification**

Force a solver failure on one cell of a two-mesh smoke-scale run; confirm the campaign completes, the failed cell carries the correct censored status with `k_enter = NaN`, the remaining cells are unaffected, and `validate_results` accepts the resulting row set.

---

## Not blockers

Findings F-04 through F-10 are MODERATE or MINOR and do not present a concrete path to scientific corruption. They are recorded in `FINDINGS.csv` and should be addressed, but they do not gate the campaign. In particular the native optimizer instrumentation is **proved bit-neutral** and is not a blocker.
