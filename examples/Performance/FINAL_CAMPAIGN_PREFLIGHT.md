# Final manual-run preflight for `performance_comparison.m`

## Verdict first

The task was **not** complete when this preflight started. Two helper files
(`final_campaign_config.m`, `final_campaign_preflight.m`) existed but nothing
called them, `performance_comparison.m` was untouched, and the script as it
stood would have run the **wrong four meshes with the wrong Olhoff profile**
and overwritten the superseded R3 artifacts in place. Those defects are fixed
and verified below.

## Repository state

| Item | Value |
|---|---|
| Branch | `benchmark-methodology-r2` |
| HEAD | `cb6353feb941f12b2aaa927e622649e1ccc926f7` |
| Working tree | dirty before this preflight and still dirty; all pre-existing modifications and untracked artifacts preserved |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1, MACA64 — the same release as the stabilization audit |
| Threads | pinned to 1 by the script, restored on exit |

No file under `Matlab/reproduction2007/`, `analysis/olhoff_stabilization_audit/`
or `analysis/three_method_parametric_study/` was modified. No optimizer or
numerical source was touched. No production mesh was solved.

## 1. Resolution set — FIXED

Before: the nine-resolution block was **commented out** and a four-mesh
development subset (160×20 … 400×50) was live.

After: `resolutions` is read from `mesh_sequence` in
`final_campaign_profile.json` rather than restated in the script, so the active
set cannot drift from the set the campaign was authorized for. Verified active:

```
160×20  240×30  320×40  400×50  480×60  560×70  640×80  720×90  800×100
```

`TOPOPT_BENCHMARK_MESHES` is now **refused** in campaign mode. `setenv`
survives for a whole MATLAB session, so a leftover smoke-test variable would
otherwise have produced a nine-mesh-looking artifact set from two toy meshes.

## 2. Frozen profile binding — FIXED

Before: every method was configured from `performance_benchmark_profile` and
the Olhoff column dispatched
`run_topopt_from_json → OlhoffDu2007Repro → olhoffOpt` — i.e. exactly the
**legacy S0 profile at r_min = 2** that Table E of the audit warns against.
Neither `final_campaign_config` nor the S1 runner was reachable from the script.

After, traced to the optimizer boundary:

| Method | Profile ID | Runner |
|---|---|---|
| Olhoff | `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1` | `analysis/olhoff_stabilization_audit/run_stabilization_case.m` |
| Yuksel | `yuksel_practical_move01_tol001` | `tools/Matlab/run_topopt_from_json.m` |
| Proposed | `proposed_practical_move02_tol001` | `tools/Matlab/run_topopt_from_json.m` |

Effective Olhoff S1 parameters, all read from the manifests and confirmed at
`repro2007_config('fig3a_best')`:

```
rminEl              1.3 element widths
move (initial)      0.005
trigger             N == 2 AND gap12 <= 0.01, 100 consecutive native evaluations
action              one reduction, move -> 0.0025
maxOuter            1600   (FIXED WORK HORIZON, not convergence)
tolMult 0.05 | rhomin 0.001 | innerSolver lp | filterMode diag | offDiag false | threads 1
```

The legacy path is excluded three ways: the campaign never calls
`run_topopt_from_json` for Olhoff; the preflight asserts the manifest names the
S1 runner; and `final_campaign_run_case` re-checks `rminEl`, `move`, `threads`,
`maxOuter`, the move sequence `[0.005 0.0025]`, `persistence = 100` and
`gap_threshold = 0.01` against the *loaded run result* before reporting it.
A run whose settings drifted raises rather than being tabulated.

The gap threshold is parsed out of the frozen condition string rather than
restated, so a manifest change stops matching instead of silently running the
old number.

## 3. Manifest consistency — PASS, fails closed

`final_campaign_preflight` runs before the first solve, with no optimization.
Existing hashing infrastructure (`sha256_hex`, the audit `provenance.json`) is
reused rather than duplicated. All 14 checks pass:

```
campaign_manifest_hash  selected_profile_hash  olhoff_wrapper_hash
common_evaluator_hash   profile_ids            olhoff_runner
output_isolated         output_not_legacy      generated_configurations
olhoff_effective_boundary  required_functions
olhoff_not_dispatched_legacy  olhoff_work_semantics_labelled  status_precedence
```

Two defects found and fixed in the pre-existing helper: it read
`provenance.deliverable_sha256.('final_campaign_profile.json')`, which
`jsondecode` never produces (it mangles `.` to `_`), so the gate **errored
instead of passing**; and `final_campaign_config` set the Olhoff
`gap_threshold` to the condition *string* rather than the number `0.01`.

Negative controls confirm the gate fails closed:

| Injected fault | Result |
|---|---|
| four-mesh subset | `final_campaign_preflight:ResolutionMismatch` |
| output dir = `examples/Performance` | `final_campaign_preflight:Failed` |
| `maxNumCompThreads(4)` | `final_campaign_preflight:ThreadMismatch` |

## 4. Mode selection — FIXED

`benchmarkMode` now defaults to `'final_campaign'`. `'r3'` is retained and
relabelled SUPERSEDED; `yuksel_table1` and the other diagnostics are unchanged.
The Yuksel-Table-1 view was previously guarded only by `~strcmp(mode,'r3')` and
would have printed during the campaign against a `data` struct that no longer
exists — now guarded on the diagnostic modes only. A `TOPOPT_BENCHMARK_MODE`
left over from an earlier session prints a loud stderr banner instead of a
one-line note.

## 5. Output safety — isolated

Everything is written under a directory that **did not exist** before this
preflight:

```
examples/Performance/final_campaign/
    table1_performance.csv            per-case timing, status, censoring, profile_id
    common_evaluators.csv             native vs common raw vs common binary E1/E2/E3
    benchmark_results.json            per-run records + frozen per-case configurations
    table1_paper_style.tex            paper-layout table
    table1_complexity_fit.csv/.png    free-exponent fit (+ _linear.png)
    table1_complexity_fit_fixedexp.*  fixed-exponent fit
    campaign_gate.json                preflight verdict and admission record
    raw/olhoff/s1_<nelx>x<nely>.mat   per-mesh Olhoff trajectory evidence
    warmup/                           discarded warm-up solves
```

Nothing outside that directory is created or overwritten. The stale in-place R3
artifacts, `diagnostic_yuksel_table1/`, `equivalence/`,
`analysis/three_method_parametric_study/` and
`analysis/olhoff_stabilization_audit/` are all untouched. No previous evidence
was deleted.

Disk: the Olhoff `.mat` evidence holds the full per-iteration snapshot array and
grows with mesh size — roughly 20 MB at 160×20 up to ~0.5 GB at 800×100, about
2 GB total. 1.3 TB free, so this is a note rather than a constraint.

## 6. Timing protocol — implemented (was absent)

Before: no thread pin anywhere in the script, and no warm-up.

| Element | Status |
|---|---|
| single-thread | `maxNumCompThreads(1)` from `threads_per_run` in the manifest; restored on exit via `onCleanup`; requested/active values printed and archived |
| warm-up | one discarded solve per method at **48×6**, outside the campaign mesh set, 5 outer iterations. Even `nely` is required — the Olhoff reproduction pins its supports at mid height and rejects odd `nely`. Warm-up results can never become observations |
| Olhoff total time | `res.wallclock`, measured inside the solver; excludes the evidence `.mat` save |
| Olhoff loop time | sum of per-iteration eigensolve + gradient + subproblem telemetry |
| Olhoff init/post | **not separable** inside the frozen audit runner; reported as null, never zero, with `unattributed_time_s` carrying the remainder |
| dispatched total time | caller-side `tic/toc` around `run_topopt_from_json`, unchanged |
| eigensolve telemetry | present for all three methods |
| Yuksel stage timing | stage iteration counts and shares were already recorded; stage **loop times** were computed by the solver and dropped — now archived, with `stage_time_sum_residual_s` (measured exactly 0) and `loop_within_total_s` |
| samples | `nSamples = 1` |

Olhoff peak RSS is sampled at 0.25 s (`fixedSpacing`) against 0.1 s
(`fixedRate`) in the dispatcher — lighter, and `fixedSpacing` avoids callback
pile-up on a long run. Olhoff's peak is dominated by a snapshot array allocated
once at the start, so the slower rate still catches it.

**The audit's Table C timings were taken with several single-thread MATLAB
processes running side by side. This campaign measures serially in one process.
The two must not be compared directly.** That warning is embedded in
`benchmark_results.json`, not only in this report.

## 7. Failure semantics — implemented (was absent)

A precedence-ordered status is now assigned to every case:

| Status | Meaning | Enters scaling fit |
|---|---|---|
| `VALID_STABILIZED_STATE_AT_FIXED_WORK` | Olhoff reached k = 1600 solver-healthy — **not convergence** | yes |
| `NATIVE_CONVERGED` | Yuksel/Proposed met their own native test | yes |
| `CAP_HIT` | iteration cap reached (any Yuksel stage counts) — **not convergence** | no |
| `SOLVER_FAILURE` | failed subproblem/LP or nonfinite state | no |
| `UNRECOGNIZED_STOP` | stop reason outside the frozen vocabulary | no |
| `RUN_ERROR` | the call raised; the campaign continues to the next case | no |

Censoring is a **separate array** (`tTotal_fit`) from the display arrays, so
failed and capped rows stay fully visible in the console tables, in
`table1_performance.csv` (`status`, `status_note`, `in_scaling_fit` columns), in
`common_evaluators.csv` and in `benchmark_results.json`, while entering no fit.
The fits now read `tTotal_fit`; previously they read every finite number.

A row-admission table is printed before the results, naming every censored row.
Olhoff's `convergence_tolerance` is null by construction — that profile has no
native convergence test, and a number there would invite the reading the audit
forbids.

## 8. Common evaluators — implemented (were absent)

`performance_comparison.m` previously exported **no** E1/E2/E3 quantities at
all. It now calls the unchanged `study_evaluate_design` — whose SHA-256 the
preflight checks against the audit provenance — once per case, **outside every
timing boundary**, and exports the three families to their own file:

* **native** — what each solver optimizes, under its own material model
  (`omega_1` in `table1_performance.csv`)
* **common raw** — E1/E2/E3 on the final raw density field
* **common binary** — E1/E2/E3 on the exact-count volume-preserving binary
  projection

plus volume, volume residual, grayness, gray fraction, and raw/binary
support-to-support connectivity. Column naming follows the study's existing
convention. **No evaluator definition was changed.**

## 9. MATLAB path independence — verified

Every path is derived from `mfilename('fullpath')`: `tools/Matlab`,
`analysis/olhoff_stabilization_audit`, `analysis/three_method_parametric_study`
and `Matlab/reproduction2007/runner`. Verified by running the full pipeline
end-to-end from `cd('/tmp')` with only `examples/Performance` on the path.

No environment variable and no setup command is required.

## 10. What was actually executed here

* `final_campaign_preflight` on all nine resolutions (no solves) — PASS
* three negative controls — all fail closed
* `checkcode` on all four files — no syntax or semantic warnings
* one full end-to-end pipeline run at **32×4 and 48×6** with a 24×4 warm-up,
  run from a foreign working directory, artifacts written to a throwaway
  `final_campaign_smoke/` and then deleted

No production case was run. Nothing at 160×20 or above was solved.

The smoke run exercised the real control flow: preflight, warm-up, S1 Olhoff at
its full 1600-iteration horizon (`VALID_STABILIZED_STATE_AT_FIXED_WORK`),
Yuksel and Proposed native stops (`NATIVE_CONVERGED`), evaluators, censoring,
all six artifacts, and the JSON schema.

## 11. Files changed by this preflight

| File | Change |
|---|---|
| `examples/Performance/performance_comparison.m` | modified — campaign mode, nine resolutions, frozen profile binding, thread pin, warm-up, status/censoring, evaluator export, output isolation, per-case JSON records |
| `examples/Performance/final_campaign_config.m` | modified — fixed the string/numeric `gap_threshold` defect; Olhoff config is now self-describing instead of a relabelled Yuksel clone; added `rmin_element`, `max_iters_expected`, `policy_id` |
| `examples/Performance/final_campaign_preflight.m` | modified — fixed the `jsondecode` field-name defect that made the gate error; added Olhoff-boundary, legacy-dispatch, work-semantics and status-precedence checks |
| `examples/Performance/final_campaign_run_case.m` | **new** — per-method runner adapter, optimizer-boundary configuration re-check, status classification, uniform telemetry |
| `examples/Performance/FINAL_CAMPAIGN_PREFLIGHT.md` | **new** — this report |

Nothing else was modified. The frozen numerical sources, the audit directory and
the study directory are byte-identical to their pre-preflight state.

## Manual run instructions

1. Open MATLAB **R2025b** (`/Applications/MATLAB_R2025b.app`).
2. Open `examples/Performance/performance_comparison.m`.
3. Press **Run**.

If the Editor offers *Change Folder* or *Add to Path*, either is fine — the
script resolves every path from its own location.

Two things worth knowing before you start:

* The script pins MATLAB to one computation thread and restores your setting
  when it finishes.
* If `TOPOPT_BENCHMARK_MODE` or `TOPOPT_BENCHMARK_MESHES` is set in the MATLAB
  session, the mode change is announced in red and the mesh override is
  refused outright. In a fresh session neither is set. To be certain:
  `setenv('TOPOPT_BENCHMARK_MODE',''); setenv('TOPOPT_BENCHMARK_MESHES','')`.

**MANUAL PERFORMANCE CAMPAIGN: READY TO RUN**

`performance_comparison.m` may now be run manually from MATLAB.
