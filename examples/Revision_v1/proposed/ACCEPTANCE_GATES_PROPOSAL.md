# Acceptance-gate safety patch — EXP2b

**Status: APPLIED 2026-07-14.** Patch: `proposed/acceptance_gates.patch`, applied second
(it depends on `stage_rewiring.patch` and does not apply to the pre-rewiring runner).
Verified in MATLAB R2025b: 18/18 acceptance-gate tests pass against the working tree, and
the capped-run rule is now enforced by `check_revision_run.m`. EXP1 is retired, so this
patch touches only the EXP2b gate; the title's former "EXP1 /" scope no longer applies.
This document is retained as the rationale record for the applied change.

> **REGENERATED 2026-07-13 — the EXP1 blocker is DISSOLVED, not worked around.**
> EXP1 is retired from the reviewer evidence chain
> ([SCIENTIFIC_DECISION_EXP1_EXP5.md](../SCIENTIFIC_DECISION_EXP1_EXP5.md)), so this patch
> **no longer touches `exp1_perf_table.m` and no longer adds a `localAccept_Exp1` gate**
> (4 files, not 5). It now tightens **EXP2b only**, plus `check_revision_run.m`, the tests,
> and the A4 `localRaiseNotImplemented` fix.
>
> **Consequence: comparator telemetry is NOT required and is NOT implemented.** The
> Part-4 blocker below — "EXP1 cannot pass its own convergence gate because the Olhoff and
> Yuksel comparators do not report a design change" — is void: there is no EXP1 gate. The
> comparators are not faithful reference implementations, so instrumenting them would have
> yielded a precise measurement of a construct-invalid comparison.

**Apply order is mandatory — the safety patch stacks on the rewiring patch:**

```
1. proposed/stage_rewiring.patch      (already reviewed)
2. proposed/acceptance_gates.patch    (this one)
```

Validated end-to-end in a disposable sandbox with **MATLAB R2025b**: 0 checkcode
errors, 18/18 gate tests in 0.01 s, and all runner modes exercised. No numerical
method, experiment algorithm, CR2 file, or A4 was touched. No result regenerated.

---

## Part 1 — Acceptance-gate audit

Both stages could incorrectly accept **all six** failure classes. Neither had *any*
cap check: the iteration-cap test existed only in `localAccept_Smoke`.

| Failure class | EXP1 (before) | EXP2b (before) | Root cause |
|---|---|---|---|
| **Iteration-capped run** | ❌ accepted | ❌ accepted | No cap comparison anywhere. EXP2b required `nIter` to *exist* but never compared it to `max_iters` — this is exactly how the capped α=1.00 and α=0.75 runs (2000/2000) were accepted. |
| **success = false** | ❌ accepted | ❌ accepted | A failed sample was swallowed by `catch`, left as `NaN`, then averaged away by `'omitnan'`. A partial failure silently shrank the sample count while `nSamples` still read 10. |
| **Missing termination metadata** | ❌ undetectable | ❌ undetectable | `run_topopt_from_json` was called with 5 outputs; the **6th output (`diagnostics`) was never requested**, so iterations/cap/termination data never reached the result. |
| **Missing convergence metadata** | ❌ undetectable | ❌ accepted | No design-change field at all. EXP2b's MAC loop used `continue` on empty `macData` — an alpha with *no* mode evidence passed **vacuously**. |
| **Unconverged design change** | ❌ accepted | ❌ accepted | Never checked. |
| **Invalid result schema** | ⚠️ partial | ⚠️ partial | Only a handful of fields were existence-checked. |

### The rule applied (nothing new invented)

The patch reuses the rule already implemented by
`exp2_authoritative_sweep/localClassify` and declared in `REVISION_EXECUTION_PLAN.md`:

> **not capped** (`iterations < max_iters`) **AND** `design_change <= declared tolerance`

where the declared tolerance is the run's own **`optimization.convergence_tol`**
(EXP2b: `0.002`) — read from the config, exactly as the authoritative
sweep reads it. A capped run is a failure, not a slow result.

### Changes

1. **`scripts/revision_v1/check_revision_run.m`** *(new, ~90 lines)* — the single,
   testable implementation of the rule. Rejection order: R1 schema → R2 `success=false`
   → R3 missing termination metadata → R4 cap → R5 missing convergence metadata →
   R6 design change > tolerance.
2. **`exp2b_building.m`** — requests the solver's existing 6th output and records
   `success`, `design_change`, `max_iters`, `design_change_tol` per alpha.
   **Metadata capture only; the computation is unchanged.**
   *(`exp1_perf_table.m` is NOT modified — EXP1 is retired.)*
3. **`localAccept_Exp2b`** — calls `check_revision_run` for every alpha, and now
   **requires** MAC evidence to be present rather than skipping when it is absent.
5. **Bug fix in the reviewed rewiring patch** (found by this validation, see below).

---

## Part 2 — Validation tests (executed)

`scripts/revision_v1/test_acceptance_gates.m` — pure struct fixtures, **no solver,
no file I/O**. Ran in **0.01 s**.

```
=== test_acceptance_gates ===
  [PASS] valid converged run
  [PASS] capped run (iterations == cap)
  [PASS] capped run (iterations > cap)
  [PASS] capped run with small design change     <- cap wins over a converged-looking dc
  [PASS] success = false
  [PASS] missing termination metadata (iterations NaN)
  [PASS] missing termination metadata (cap NaN)
  [PASS] missing termination metadata (cap = 0)
  [PASS] missing convergence metadata (design change NaN)
  [PASS] missing convergence metadata (design change empty)
  [PASS] invalid schema (not a struct)
  [PASS] invalid schema (field removed)
  [PASS] invalid schema (cap removed)
  [PASS] design change above declared tolerance
  [PASS] design change marginally above tolerance
  [PASS] design change exactly at tolerance      <- boundary: <= tol accepted, as declared
  [PASS] regression: historically accepted capped EXP2b run   (2000/2000, dc=0.155)
  [PASS] regression: capped Olhoff comparator                 (10000/10000, dc=NaN)
  passed: 18   failed: 0
```

The last two are the real historical mis-accepts. They are now rejected.

---

## Part 3 — Worktree validation (executed, with results)

Because the patches were authored against the **working tree** (which carries
pre-existing uncommitted edits), a `git worktree add HEAD` will *not* match. Seed the
sandbox from the working tree:

```bash
R=/Users/piotrek/Programming/topOpt4freqMax
S=/tmp/rv_sandbox && rm -rf $S && mkdir -p $S
rsync -a --exclude .git --exclude references --exclude paper \
      $R/examples $R/tools $R/scripts $R/analysis $S/
cd $S && git init -q . && git add -A && git -c user.email=a@b -c user.name=c commit -qm base

# 1-2. apply the stack, in order
git apply $R/examples/Revision_v1/proposed/stage_rewiring.patch
git apply $R/examples/Revision_v1/proposed/acceptance_gates.patch
```

```matlab
cd examples/Revision_v1
addpath(fullfile('..','..','scripts','revision_v1'));

% 3. checkcode
checkcode('run_all_revision_experiments.m')        % -> 0 errors
% 4. lightweight tests
test_acceptance_gates()                            % -> 18/18 in 0.01 s
% 5. dry run
run_all_revision_experiments('full','dry_run',true)
% 6. full campaign must abort
run_all_revision_experiments('full')
% 7. modes
run_all_revision_experiments('smoke')
run_all_revision_experiments('stage','EXP2','dry_run',true)
run_all_revision_experiments('full','resume',true)
```

### Observed results

| Step | Result |
|---|---|
| 3. `checkcode` | **0 parse/syntax errors** across all 5 files (`check_revision_run.m`: 0 messages at all). Remaining warnings are pre-existing style (preallocation/unused). |
| 4. tests | **18/18 passed, 0.01 s** |
| 5. `dry_run` | Stage table in correct order: **S1, EXP2, EXP2b, EXP3, A4, EXP1, EXP5**. Preflight reports `[A4] state: A4_NOT_IMPLEMENTED` and continues (dry-run is report-only, by design). |
| 6. `full` | **Aborts with `run_all:PreflightFailed` — "P3 mandatory stage not implemented: A4"** after ~1 s (MATLAB startup). Verified afterwards: `exp1/ exp2/ exp2b/ exp3/ exp5/ s1/` each contain **0 result files**, and the stage directories were never created. **No production computation started.** |
| 7. `smoke` | `Gate I1 PASSED` → `run_all:GateI1Confirmed`, condition correctly identified as *"reached iteration cap: 200/200 … design change = 5.00e-03"* (0.35 s). Unchanged. |
| 7. `stage` | `stage`/`EXP2` + `dry_run` works; preflight passes (a single implemented stage is allowed). |
| 7. `resume` | `full` + `resume=true` still aborts on A4 — **a placeholder can never be resume-skipped**. |
| 7. registry | `campaign_progress.json` written correctly (`campaign_id`, `status`, `mode`, `dry_run`, per-stage `output_directories`). |

### Defect found in the already-reviewed `stage_rewiring.patch`

Dispatching the A4 placeholder in **stage mode** raised
`MATLAB:maxlhs — Too many output arguments` instead of `run_all:A4NotImplemented`:
`localRunAndAccept` calls `res = stage.runFn()`, and the placeholder's handle wrapped a
bare `error()`, which returns no outputs. It still failed loud, but with a misleading
identifier and message.

Per "do not redesign the rewiring patch", the fix lives **in this safety patch**: the
handle now routes through a named 1-output `localRaiseNotImplemented`. Re-validated:

```
---- A4: Eigenpair-refresh study N={1,5,10,50,inf} (SS 400x50)
[A4] EXCEPTION after 0.0s
  Identifier: run_all:A4NotImplemented
  Message   : A4 is not implemented: no runner, configuration, or artifact exists.
              exp4_sensitivity_ablation is a pre-authoritative sensitivity ablation
              and is NOT an A4 implementation.
```

**Consequence: the two patches must be applied together.** Applying `stage_rewiring.patch`
alone leaves this defect in place.

---

## Part 4 — Safety review

### 1. Can the patch be safely applied?

**Yes**, with the stated apply order. It is mechanically verified (`git apply` clean on
a pristine sandbox), syntactically verified (0 checkcode errors), behaviourally verified
(18/18 tests; all six runner modes exercised), and it only *tightens* acceptance —
nothing that previously failed can now pass. It touches no numerical method, no CR2 file,
and does not implement A4.

### 2. Can any capped experiment still become reviewer evidence?

**Not through the master runner.** EXP1 and EXP2b are now gated by `check_revision_run`;
EXP2, EXP3 and S1 are gated by their own `classification` (rewiring patch); EXP5 inherits
EXP1. Verified against both historical mis-accepts.

**Yes, outside it — two residual routes:**
- **CR2** is *not a runner stage* (`run_cr2_production()` takes no `outDir` and writes to
  a hardcoded path). Its saved outputs are capped 400/400 under both OC and MMA. Nothing
  in code prevents those numbers being quoted; only the archive governance does.
- **Any script invoked manually** bypasses every gate. The gates protect the campaign,
  not the analyst.

### 3. Can any obsolete runner still be executed?

**Not from the runner.** Preflight **P2** denies `exp2_clamped_beam`,
`exp3_mesh_convergence`, `exp4_sensitivity_ablation` and any function resolving under
`archive/`, and the registry no longer references them.

**But the three files still sit in `examples/Revision_v1/` and can be called by hand.**
After the rewiring patch they are unreferenced. Recommendation: archive them to
`archive/superseded_formulation/` in a follow-up (pure `git mv`, no code change).

### 4. Can any historical timing dataset still be used accidentally?

**Not through the runner.** The rewiring patch's `localLoadExp1Result` hard-fails
(`run_all:Exp5MissingExp1`) instead of letting `exp5_scaling([])` fall back to its
embedded historical values.

**Yes if `exp5_scaling` is called directly** — `exp5_scaling()` with no argument still
silently uses the embedded table (it prints a "diagnostic fallback / not accepted
evidence" warning, but returns a fit anyway). The legacy `.mat` files are quarantined in
`archive/superseded_runs/pre_authoritative/` but remain readable. Residual, governance-only.

### 5. Remaining blocker before launching the production campaign

Two, and the second is new and hard:

**(a) A4 is not implemented.** Blocks `full`/`fast` by design. Individual stages still run.

**(b) EXP1 cannot pass its own convergence gate — the comparators do not report convergence.**
This is the important finding. `localFinalDesignChange` returns `NaN` for two of the three
EXP1 methods:
- **Olhoff** (`topFreqOptimization_MMA`) *computes* `change_x = norm(x-x_prev)/sqrt(nEl)`
  and keeps `dx_hist`, but its `diagnostics` struct exports only
  `initial/final/iterations/loop_time/t_iter` — **no design change, no converged flag**.
- **Yuksel**: the `run_topopt_from_json` Yuksel branch never assigns `diagnostics` at all,
  so it stays an empty struct.

So EXP1 will be **rejected at the first Olhoff sample** with *"missing convergence
metadata … convergence cannot be verified"*. That is the correct fail-loud outcome — you
cannot certify a comparator converged when it never reports convergence, and it is
precisely why Olhoff sat at its cap unnoticed. But it must be resolved deliberately:

- **Option 1 (recommended, ~2 lines per solver, metadata only):** export the
  already-computed values — `diagnostics.final_design_change = dx_hist(end);` and a
  `diagnostics.converged` flag — from `topFreqOptimization_MMA`, and assign `diagnostics`
  in the Yuksel branch. This changes no algorithm, but it touches `analysis/`, so it needs
  explicit approval and its own patch.
- **Option 2:** formally declare the comparators "run-to-cap by design", and drop every
  convergence, frequency-gap, speedup and scaling claim made against them.

Until (b) is decided, **EXP1 and therefore EXP5 cannot produce accepted evidence.**
