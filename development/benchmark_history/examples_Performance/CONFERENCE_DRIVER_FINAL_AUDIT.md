# Conference performance driver — final software-integrity / run-readiness audit

## 1. Starting state (read-only, recorded before any edit)

- Branch `benchmark-methodology-r2`, HEAD `eb7d74d8b92f6af1e09c1803cfbe13cad9afedc0`
- Working tree at audit start: **1 modified, 130 deleted, 5 untracked**
  - modified: `analysis/OlhoffM4Reconstruction/README.md`
  - untracked: `analysis/OlhoffArchive.zip`, `analysis/OlhoffM4Reconstruction/+frozen/run_pinned_pinned.m`,
    `examples/Performance/conference_benchmark.zip`, `examples/conference_benchmark.zip`,
    `examples/conference_benchmark/`
  - deleted (130): the superseded Olhoff trees (`OlhoffApproachExact` 53, `OlhoffRegularized` 35,
    `OlhoffApproach` 13, `OlhoffReproduced2007` 5 — archived to `OlhoffArchive.zip`) and all 24 files
    under `examples/Performance/conference_benchmark/` (relocated untracked to
    `examples/conference_benchmark/`).
- Active driver `examples/Performance/performance_comparison.m` at audit start:
  sha256 `f8b2e0f0c9a3...` (recorded post-edit below; the file had been edited by the user between
  the previous audit and this one — `runLabel` set to `campaign_9mesh_r2`).
- `legacy_r3/performance_comparison_r3.m` exists and is **inert** — nothing references it.

## 2. Active call graph from `performance_comparison.m`

```
performance_comparison.m
├─ olhoffm4_scrub_forbidden_paths          (path guard, pre-preflight)
├─ confbench_frozen_budget                 (reads profile_freeze_manifest.json)
├─ confbench_method_config ─┬─ olhoffm4_config          (Olhoff, frozen import)
│                           └─ study_base_config        (Proposed / Yuksel)
├─ confbench_preflight ─┬─ olhoffm4_verify_import, olhoffm4_paths,
│                       │  olhoffm4_assert_dispatch, olhoffm4_forbidden_paths
│                       └─ confbench_frozen_budget, confbench_timing_schema, confbench_caveats
├─ confbench_run_case ─┬─ runOlhoff   → olhoffm4_run → +frozen/algo/olhoffOpt.m
│                      ├─ runProposed → run_topopt_from_json → ourApproach/topopt_freq.m
│                      ├─ runYuksel   → run_topopt_from_json → YukselApproach/top99neo_inertial_freq.m
│                      └─ confbench_accounting
├─ study_evaluate_design                   (common E1/E2/E3, OUTSIDE all timers)
├─ confbench_scaling_fit
├─ confbench_manifest
├─ confbench_export                        (CSV / JSON / LaTeX / notes)
├─ confbench_complexity_plots
├─ confbench_topology_images
└─ confbench_caveats, confbench_display_name
   confbench_selftest                      (not called by the driver; mechanics tests)
```

## 3. Read-only findings (before fixes)

| # | Severity | Finding |
|---|---|---|
| F1 | **HIGH** | **Stage-cap censoring undetectable.** `telemetry.stopping` (built at `run_topopt_from_json.m:912`) exposes only a single `stop_reason`; it has no `stage1_stop_reason` / `stage2_stop_reason`. The two `isfield` branches in `fillDispatched` are therefore **dead code**, and `capHit` is decided solely by the overall reason — which for Yuksel is Stage 2's. A Stage-1 cap hit followed by a Stage-2 tolerance stop is classified `NATIVE_CONVERGED`. **Confirmed in existing evidence:** `campaign_9mesh` Yuksel 720×90 has `stage1_iterations = 1000` (exactly the frozen cap) with `status = NATIVE_CONVERGED`, `ok = true`, and it **is** one of the 7 points in the fitted Yuksel exponent p = 1.7404. |
| F2 | MEDIUM | **Duplicate active config assignment.** `cfg.maxOuterOverride` is assigned twice (lines 61 and 93). Values agree (`[]`), so behaviour is currently correct, but it violates the one-authoritative-assignment requirement. |
| F3 | MEDIUM | **Stale / orphaned comments.** The guard comment block still says *"Set true on 2026-09-04 for the four-resolution partial campaign"* while nine meshes are active, and the long `cfg.yukselMaxIters` justification is now orphaned from its assignment (which moved up to line 63). |
| F4 | MEDIUM | **`maxNumCompThreads` is never restored.** The driver calls `maxNumCompThreads(1)` at line 149 and never restores the user's prior setting — not on normal completion, not on preflight failure (which `error`s), not on exception. The MATLAB session is left single-threaded. |
| F5 | MEDIUM | **Olhoff `ok` ignores inner-solve failures.** `olhoffm4_run` records `n_inner_not_converged` but gates `SOLVER_FAILURE` only on a nonfinite design / nonpositive ω₁. A run with failed inner MMA solves could report `NATIVE_CONVERGED`. Latent only — every existing row has 0 failures. |
| F6 | LOW | Output-overwrite guard checks only three legacy paths, not a same-label campaign directory. Currently moot (`campaign_9mesh_r2` does not exist). |
| F7 | LOW | Preflight note (correct, non-blocking): the external `/Users/piotrek/Programming/Matlab/Olhoff` tree no longer matches the imported state (`olhoffOpt.m`, `innerLoop.m`) because of later restoration experiments. Dispatch is unaffected — the in-repo `+frozen` copy is hash-pinned and verified. |

Verified **clean** in the read-only pass: dispatch and path isolation (22/22 preflight checks pass,
including per-mesh Olhoff config at all nine meshes); memory fully excluded; timing identity plus an
independent caller-vs-solver cross-check; scaling fit filters on `ok`; method-native export schema
with semantic field names; `addpath(genpath` appears only inside a comment; no `TODO`/`FIXME`/`eval(`/`cd(`/`pwd`.

---

## 4. Bugs fixed

| # | Sev | Fix | Where |
|---|---|---|---|
| F1 | **HIGH** | **Numeric per-stage cap detection.** The status decision was extracted into a new file `confbench_classify.m` — so the precedence lives in exactly one testable place — and now compares the **actual** counts against the **actual** runtime caps, read from `telemetry.yuksel.stage1_max_iters` and `mcfg.optimization.max_iters` (never restated as constants). Textual stop reasons remain advisory. Wired for Yuksel (`stage1`, `stage2`) and Proposed (`total`). | `confbench_classify.m` (new), `confbench_run_case.m` |
| F2 | MED | Removed the duplicate `cfg.maxOuterOverride` assignment; one authoritative assignment per field. | `performance_comparison.m` |
| F3 | MED | Rewrote the orphaned/stale guard comments (the "2026-09-04 four-resolution" text) and reunited the `yukselMaxIters` justification with its assignment. | `performance_comparison.m` |
| F4 | MED | **Thread restore.** Added `entryThreads` capture, an `onCleanup` backstop, and **two deterministic restores** — before the preflight bail-out and at the end of the script. *An earlier attempt using `onCleanup` alone was measured to be ineffective:* in script scope the object is not destroyed at script end (verified: pin still in force after return, lifts only on `clear`). The comment records this. | `performance_comparison.m` |
| F5 | MED | A failed nested MMA solve can no longer read as convergence: `runOlhoff` now downgrades to `SOLVER_FAILURE` when `n_inner_not_converged > 0`. Enforced in the benchmark wrapper rather than by editing the hash-pinned frozen import. Inert at 0 failures, which is every existing row. | `confbench_run_case.m` |
| F6 | — | Manifest now records **whether a cap was actually hit** (`manifest.cap_summary`: `any_cap_hit`, `n_cap_hit`, the offending rows, and what CAP_HIT means). | `performance_comparison.m` |

Not fixed, by design: F7 (external-repo provenance drift) is correct behaviour and non-blocking.
The output-overwrite guard was left as-is — `campaign_9mesh_r2` does not exist, so there is nothing
to overwrite, and broadening the guard risks blocking a legitimate re-run.

## 5. Files changed

| file | status |
|---|---|
| `examples/Performance/performance_comparison.m` | modified (F2, F3, F4, F6) |
| `examples/Performance/conference_bench/confbench_run_case.m` | modified (F1, F5) |
| `examples/Performance/conference_bench/confbench_classify.m` | **new** (F1) |
| `examples/Performance/conference_bench/confbench_selftest.m` | modified (T8–T12) |
| `examples/Performance/CONFERENCE_DRIVER_FINAL_AUDIT.md` | **new** (this report) |

**No scientific setting was changed anywhere.** No tolerance, filter, move limit, `p`, mass
interpolation, objective, optimizer, mesh sequence, Olhoff realization or evaluator definition was
touched. The only budget in play is the already-authorized Yuksel per-stage safety cap of 5000.

## 6. Scientific invariants verified unchanged

Preflight re-run after all edits: **85 checks, 85 PASS, 0 FAIL.** Profile ids bound:
`proposed_practical_move02_tol001`, `yuksel_practical_move01_tol001`,
`olhoff_m4_subspaceN2_R006_tolInner005_S2beta_settledmove_v1`. Per-mesh Olhoff verification passes
at all nine meshes (nested MMA, M4 subN, fixed physical R = 0.06, `tolInner` 0.05, outer RMS
semantics, S2 as frozen, single thread + diagnostics off).

## 7. Yuksel cap-propagation proof

`runYuksel` sets `mcfg.optimization.max_iters = 5000` (Stage-2 cap) and
`mcfg.optimization.yuksel.stage1_max_iters = 5000`. In `run_topopt_from_json`, `maxiter` is the
Stage-2 cap and `stage1MaxIter` defaults to it, overridden by the explicit field; the
`stage1BudgetIndependent` clamp is `min(stage1MaxIter, maxiter) = 5000`. The solver echoes
`telemetry.yuksel.stage1_max_iters`, which the classifier reads back. Measured end-to-end: with
`max_outer_override = 3` **and** `yuksel_max_iters = 5000` both supplied, the override correctly
dominates — `n1 = n2 = 3`, effective caps 3/3 — so a raised budget cannot lengthen a warm-up.

## 8. Cap-classification regression tests (mandatory)

| id | case | result |
|---|---|---|
| T8 | n1 = 1000 = cap, n2 = 966 < cap, overall stop reason `density_change_tolerance` | **CAP_HIT, ok=0** ✓ |
| T9 | n1 = 300 < cap, n2 = 5000 = cap | **CAP_HIT, ok=0** ✓ |
| T10 | both below cap, tolerance stop | **NATIVE_CONVERGED, ok=1** ✓ |
| T11 | failed subproblem **and** both stages at cap | **SOLVER_FAILURE** (outranks CAP_HIT) ✓ |
| T12 | scaling fit with 4 ok rows + 1 CAP_HIT row | fits n=4, names the meshes, excludes the capped one ✓ |

T8 is the historical `campaign_9mesh` 720×90 defect reproduced exactly; it previously classified as
`NATIVE_CONVERGED` and entered the fit. Full suite: **12/12 PASS** (T1–T7 pre-existing: RUN_ERROR
handling, fail-closed dispatch under deliberate shadowing, uniform record field set, timing identity
and its FAIL flag, no memory column, odd-nely rejection, long-campaign guard).

## 9–10. Dispatch proofs

Resolved live after the path scrub:

```
olhoffOpt : analysis/OlhoffM4Reconstruction/+frozen/algo/olhoffOpt.m
mmasub    : analysis/OlhoffM4Reconstruction/+frozen/mma_published/mmasub.m
```

Path scrub removed 0 entries from a clean session; T2 proves the gate **refuses** when
`Matlab/reproduction2007/algo` is deliberately prepended, and that it resolves correctly again once
removed. Import integrity, source attestation and declared-modification reconstruction all pass.

## 11–13. Timing, scaling, path/manifest proofs

Tiny end-to-end mechanics run (40×6, all three methods, from `cwd = /private/tmp`):

| method | status | Count1 | Count2 | residual | cross-check | overhead |
|---|---|---|---|---|---|---|
| Proposed | CAP_HIT | eigenanalysis_solves 1 | simp_iterations 4 | 0.00e+00 | 7.45e−04 | +0.1117 |
| Yuksel | CAP_HIT | stage1_iterations 4 | stage2_iterations 4 | 0.00e+00 | 7.67e−04 | +0.1619 |
| Du–Olhoff | CAP_HIT | outer_iterations 4 | inner_mma_iterations_total 126 | 0.00e+00 | 3.20e−03 | +0.0304 |

Zero accounting residual, cross-checks inside tolerance, **no negative overhead**. Scaling correctly
refused for a non-campaign run. All 7 artifacts written to the requested absolute directory from a
foreign cwd. Manifest records the effective config including `yukselMaxIters = 5000` alongside
`run_class.yuksel_max_iters_frozen = 1000`, MATLAB `25.2.0.2998904 (R2025b)`, thread count,
resolutions, methods, profile ids, resolved implementations, path scrub, and now `cap_summary`.

## 14–16. checkcode and remaining notices

All 17 active files parse; **no defects**. Remaining notices are 8 informational
"a suppression is no longer generated" lines on pre-existing `%#ok` pragmas
(`performance_comparison.m` ×4, `confbench_manifest.m`, `confbench_scaling_fit.m` ×2,
`confbench_selftest.m`). These are style pragmas, not defects, and were deliberately left untouched.

## 17. Final USER CONFIGURATION block (active)

```matlab
cfg.resolutions = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
cfg.confirmLongCampaign = true;
cfg.maxOuterOverride    = [];
cfg.yukselMaxIters      = 5000;
cfg.methods      = struct('proposed', true, 'yuksel', true, 'olhoff', true);
cfg.singleThread = true;  cfg.runWarmup = true;
cfg.runEvaluator = true;  cfg.fitScaling = true;
cfg.writeCSV = true; cfg.writeJSON = true; cfg.writeLaTeX = true;
cfg.outputDir = '';                 % auto
cfg.runLabel  = 'campaign_9mesh_r2';
cfg.timingTolAbs = 1e-6; cfg.timingTolRel = 1e-9; cfg.crosscheckTolRel = 0.05;
```

Semantics match the target exactly.

## 18–19. Output location and expected artifacts

`examples/Performance/conference_benchmark/campaign_9mesh_r2/` (does **not** yet exist — no
overwrite risk; the previous `campaign_9mesh` evidence now lives untracked at
`examples/conference_benchmark/` and is untouched).

Will contain: `conference_performance_table.csv` / `.tex`, `conference_performance_detailed.csv`,
`benchmark_results.json`, `benchmark_manifest.json`, `timing_schema.json`, `BENCHMARK_NOTES.md`,
`benchmark_records.mat`, the `table1_complexity_fit*` figures, a `topologies/` directory with 27
images, a `warmup/` subdirectory, and `preflight_FAILED.json` only on failure.

## 20. Remaining caveats

- The nine-mesh campaign is long: the previous run took ≈3.6 h for Olhoff alone; Yuksel will take
  longer than before because the raised cap lets it run to its own stopping rule.
- Fine-mesh Olhoff topologies are **diagnostic only**. The published reconstruction figure is
  160×20 (240×30 secondary, with its multiplicity caveat).
- The Olhoff empirical exponent must be labelled *empirical scaling of the frozen reconstruction*;
  `confbench_caveats().scaling` already carries that wording.
- `examples/conference_benchmark/` and the two zips are untracked; committing them would make the
  previous campaign's provenance durable.

## 21. Final hashes (changed files, post-audit)

```
23eccedba765416fac9ffb56fb836b947f744e28d8e9249de39f33bbeca46113  examples/Performance/performance_comparison.m
e6befc2736d7911df2005ec5ea10c55e8e613c4853fcc4a0c3175db8d0a7e672  examples/Performance/conference_bench/confbench_run_case.m
612967fb1fe57b98f6656d5a016e856bed53ee7874e68088d95c929d566a2c1f  examples/Performance/conference_bench/confbench_classify.m
4083e420c174e435a6eaa10d789f4e9e83ad6aeb7c9570b462a0965fa54195a7  examples/Performance/conference_bench/confbench_selftest.m
```

## Compact check table

| Check | Proposed | Yuksel | Du–Olhoff | PASS/FAIL |
|---|---|---|---|---|
| dispatch | ✓ | ✓ | ✓ frozen import | PASS |
| scientific config | ✓ frozen | ✓ frozen | ✓ frozen, 9/9 meshes | PASS |
| iteration caps | 2000 | **5000 both stages** | 400 outer | PASS |
| convergence semantics | ✓ | ✓ **numeric** | ✓ native | PASS |
| status classification | ✓ | ✓ **fixed** | ✓ **inner-fail guard** | PASS |
| timing | resid 0 | resid 0 | resid 0 | PASS |
| memory exclusion | ✓ | ✓ | ✓ | PASS |
| warm-up exclusion | ✓ | ✓ override dominates | ✓ | PASS |
| evaluator separation | ✓ | ✓ | ✓ | PASS |
| scaling eligibility | ✓ | ✓ censored excluded | ✓ + caveat | PASS |
| output path | ✓ absolute | ✓ | ✓ | PASS |
| manifest | ✓ | ✓ cap recorded | ✓ | PASS |
| path safety | ✓ | ✓ | ✓ fail-closed | PASS |
| manual usability | ✓ | ✓ | ✓ | PASS |
| final readiness | ✓ | ✓ | ✓ | PASS |

# CONFERENCE_NINE_MESH_RUN_READY
