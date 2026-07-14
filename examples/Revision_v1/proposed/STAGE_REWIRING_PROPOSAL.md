# Stage rewiring — `run_all_revision_experiments.m`

**Status: APPLIED 2026-07-14.** The patch `proposed/stage_rewiring.patch` was applied
cleanly to the working tree (first of the two, before `acceptance_gates.patch`) and
verified in MATLAB R2025b: the registry is `S1 → EXP2 → EXP2b → EXP3 → A4`, dry-run
reports that graph, and `full` aborts at preflight P3 (A4 not implemented) before any
computation. This document is retained as the rationale record for the applied change.

> **REGENERATED 2026-07-13** — EXP1 and EXP5 are now **retired from the reviewer evidence
> chain** ([SCIENTIFIC_DECISION_EXP1_EXP5.md](../SCIENTIFIC_DECISION_EXP1_EXP5.md)). The
> patch no longer registers an EXP1 or EXP5 stage, no longer rebinds EXP5 to EXP1 at
> dispatch, and drops the `localLoadExp1Result` helper and the dead
> `localAccept_Exp1`/`localAccept_Exp5` gates. `exp1_perf_table` and `exp5_scaling` are
> added to the preflight P2 denylist. **No scaling stage remains active.** The active
> registry is **S1 → EXP2 → EXP2b → EXP3 → A4**. S1, EXP2, EXP2b, EXP3, A4 and CR2 are
> otherwise unchanged; smoke, dry_run, resume, force and progress tracking are preserved.
> Re-validated in MATLAB R2025b: runner parses, dry-run reports the five stages, `full`
> aborts with `run_all:PreflightFailed` (A4), `smoke` yields `run_all:GateI1Confirmed`.

The working-tree runner is byte-identical to `HEAD` (md5
`1369625edec3935837c866ad1df35a49`). Verified non-mutating:
`git apply --check proposed/stage_rewiring.patch` → **applies cleanly**.

No numerical algorithm, experiment implementation, or accepted
scientific conclusion was modified. Signature differences are resolved with thin
adapters in the registry, not by editing the experiments.

---

## 1. Audit of the current runner

| Area | Finding |
|---|---|
| Stage registry (`localBuildStages`, L429–475) | Six stages. EXP2→`exp2_clamped_beam`, EXP3→`exp3_mesh_convergence`, EXP4→`exp4_sensitivity_ablation` — all **pre-authoritative**. The authoritative scripts exist but are **not stages**. |
| Stage dispatch (L159–231) | Sequential; registry order **is** execution order. EXP5's `runFn` is rebound at dispatch (L188–190). |
| Acceptance gates | **The iteration-cap check exists only in `localAccept_Smoke`** (L953, `res.termination.capped`). `localAccept_Exp1/Exp2/Exp2b/Exp3/Exp4/Exp5` have **no cap check** — the header admits "partial check for legacy". This is why capped runs were marked *accepted*. |
| Dry run (`localDryRun`, L559) | Reports would-run/would-skip per stage. No path, registry, or placeholder validation. |
| Resume (L166–178) | Skips a stage when `localValidateStageArtifacts` passes. **The skip path never populates `allResults`.** |
| **EXP5 silent fallback (critical)** | If EXP1 is resume-skipped, `localGetField(allResults,'exp1')` → `[]`, and `exp5_scaling([])` falls back to **embedded historical timings** ("diagnostic fallback", `exp5_scaling.m:40`) yet still writes `exp5_scaling_results.mat` and would be **accepted**. A resumed campaign can therefore publish a scaling exponent fitted to hardcoded numbers. |
| Output conflict (`localPrepareDirFail`, L1296) | Fires **inside** the stage loop, so a conflict on EXP5 aborts *after* EXP1's 15.5 h. Only checks `.mat/.csv/.png`. This killed campaign `r1_full_20260706` in 0.15 s. |
| Path allowlist (`localAddActiveAnalysisPaths`, L1421) | Already correct: `{ourApproach, OlhoffApproach, YukselApproach, elastic2D, LabandaApproach}`. Not enforced as a *check*, only as a construction. |

Signatures (unchanged by this patch):

```
exp1_perf_table(nSamples, meshSizes, outDir)
exp2b_building(alphaVals, outDir)
exp5_scaling(perfResults, outDir)
exp2_authoritative_sweep(outDir)                 <- alphas fixed INTERNALLY
exp3_authoritative_mesh_convergence(outDir)
s1_mitigation_400x50_pilot(outDir)
run_cr2_production()                             <- no outDir; hardcoded output
```

---

## 2. Stage registry — before / after

### Before

| # | Tag | Runner | Signature | Out | Required artifact | Gate | Cap-aware? |
|---|---|---|---|---|---|---|---|
| 1 | EXP1 | `exp1_perf_table` | `(n, mesh, out)` | `output/exp1` | `exp1_perf_table_results.mat` | `localAccept_Exp1` | ❌ |
| 2 | EXP2 | `exp2_clamped_beam` ⛔ | `(alpha, out)` | `output/exp2` | `exp2_clamped_beam_results.mat` | `localAccept_Exp2` | ❌ |
| 3 | EXP2b | `exp2b_building` | `(alpha, out)` | `output/exp2b` | `exp2b_building_results.mat` | `localAccept_Exp2b` | ❌ |
| 4 | EXP3 | `exp3_mesh_convergence` ⛔ | `(alpha, out)` | `output/exp3` | `exp3_mesh_convergence_results.mat` | `localAccept_Exp3` | ❌ |
| 5 | EXP4 | `exp4_sensitivity_ablation` ⛔ | `(out)` | `output/exp4` | `..._results.mat`, `..._diary.txt` | `localAccept_Exp4` | ❌ |
| 6 | EXP5 | `exp5_scaling` | `(perf, out)` | `output/exp5` | `..._results.mat`, `..._loglog.png` | `localAccept_Exp5` | ❌ |

⛔ = pre-authoritative. A4 absent. S1 not a stage. No preflight.

### After

| # | Tag | Canonical runner | Invocation (adapter) | Out | Required artifacts | Acceptance gate | Depends on | Est. | Resumable when |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **S1** | `s1_mitigation_400x50_pilot` | `@() f(od.s1)` | `output/s1` | `s1_mitigation_400x50_result.mat`, `_manifest.json` | `localAccept_S1` — requires `classification == "accepted"` | — | 3 h | accepted + artifacts + hash |
| 2 | **EXP2** | `exp2_authoritative_sweep` | `@() f(od.exp2)` — drops `alphaVals` | `output/exp2` | `exp2_authoritative_sweep_result.mat`, `_manifest.json` | `localAccept_Exp2Authoritative` — **every** alpha `classification == "accepted"` **and** `all_accepted` | S1 | 3 h | as above |
| 3 | **EXP2b** | `exp2b_building` | `@() f(alphaVals, od.exp2b)` | `output/exp2b` | `exp2b_building_results.mat` | `localAccept_Exp2b` (unchanged) | S1 | 2 h | as above |
| 4 | **EXP3** | `exp3_authoritative_mesh_convergence` | `@() f(od.exp3)` — drops `alphaVals` | `output/exp3` | `exp3_authoritative_mesh_convergence_result.mat`, `_manifest.json` | `localAccept_Exp3Authoritative` — study `classification == "passed mesh convergence"` **and** every mesh accepted | S1, EXP2 | 6 h | as above |
| 5 | **A4** | *(none)* | `localMakePlaceholderStage` | `output/a4` | — | `localAccept_NotImplemented` — always fails | — | 16 h | **never** |

**EXP4 removed** (pre-authoritative; superseded by CR2; **not** relabelled as A4).
**EXP1 and EXP5 removed** as retired reviewer evidence. Their gates
(`localAccept_Exp1`, `localAccept_Exp5`), the EXP5→EXP1 dispatch rebinding and the
`localLoadExp1Result` helper are all deleted, and both runners are added to the preflight
P2 denylist. The now-dead `localAccept_Exp2`/`localAccept_Exp3`/`localAccept_Exp4` gates
are likewise removed. **No scaling stage remains active.**

S1 is registry position 1, so it executes before EXP2b and EXP3 (requirement 5).

---

## 3. Preflight (new — fails before any computation)

`localPreflight(stages, opts, mode)`, called at L157 immediately after the registry
is built and *before* dry-run and the stage loop:

| Check | Enforces |
|---|---|
| **P1** | No MATLAB path entry contains `OlhoffApproachExact`. |
| **P2** | No stage's `runFn` names a retired runner (`exp2_clamped_beam`, `exp3_mesh_convergence`, `exp4_sensitivity_ablation`, **`exp1_perf_table`, `exp5_scaling`**, `*_olhoff_exact*`), and no stage's function resolves via `which()` to a file under `archive/`. |
| **P3** | Mandatory placeholder stages (A4) block `full`/`fast`. `stage` mode still works for individual implemented stages, so EXP2 can be run alone. |
| **P4** | Output conflicts across **all** stage dirs are reported up-front, before any computation. |

In `dry_run` the findings print and the run continues (so the full table is still
reported); otherwise they raise `run_all:PreflightFailed`.

Preserved unchanged: smoke mode, stage mode, dry_run, resume, force, progress
tracking, campaign summary, manifest writing, stack-trace preservation.

---

## 4. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **A4 blocks every `full`/`fast` campaign** — by design. Until A4 exists, only `stage` mode runs. | High (intended) | This is the requested behaviour. Run stages individually, or implement A4. |
| R2 | **Cap gates are still missing on EXP1 and EXP2b.** The new EXP2/EXP3/S1 gates inherit cap-awareness from the scripts' own `classification`, but `localAccept_Exp1`/`localAccept_Exp2b` still cannot reject a capped run. **A capped EXP2b can still be accepted.** | **High** | Out of scope for this patch (not in the task list). Must be fixed in a follow-up before EXP2b is trusted. |
| R3 | `exp2_authoritative_sweep` **ignores** campaign `alphaVals` (hardcoded `[1 .75 .5 .25 0]` at L38). The stage config now records `alpha_source = 'script-internal fixed'` so the resume hash is honest, but `fast` mode's alpha list has no effect on EXP2. | Medium | Documented in the config. Do not add an `alphaVals` argument to the script — that would be a numerical-interface change. |
| R4 | **CR2 is not a stage.** Removing EXP4 leaves the omitted-sensitivity demand unregistered. `run_cr2_production()` takes no `outDir` and writes to a hardcoded directory, so registering it would require changing the experiment's signature — explicitly forbidden here. | **High** | Follow-up: either add an `outDir` parameter to `run_cr2_production` (a real code change, needs its own review) or keep CR2 as a governed standalone run and cite it manually. **Do not leave it forgotten.** |
| R5 | New artifact names must match what the scripts actually write. Verified: prefixes are built from the passed `outDir` (`exp3_..._result.mat` at L~392, `exp2_authoritative_sweep_*` at L418, `s1_mitigation_400x50_*` at L120), so writing into `output/exp2` etc. is correct. | Low | Covered by the validation commands below. |
| R6 | Untested in MATLAB. Struct-array field-order compatibility between `localMakeStage` and `localMakePlaceholderStage` is the main syntactic hazard (both build from `localEmptyStage`, so field names/order match). | Medium | Run the dry-run/smoke validation commands before any production launch. |
| R7 | `output/s1` and `output/a4` are new directories; the existing S1 results live in `output/s1_mitigation_400x50/`. The stage will not see them and will re-run S1. | Low | Intended — the saved S1 run classified itself `inconclusive` and must not be resumed as accepted. |

---

## 5. Validation commands

Run in order. The first three are non-mutating.

```bash
# 1. Confirm the patch still applies and the working tree is untouched
cd /Users/piotrek/Programming/topOpt4freqMax
git apply --check examples/Revision_v1/proposed/stage_rewiring.patch && echo OK
md5 -q examples/Revision_v1/run_all_revision_experiments.m   # 1369625edec3935837c866ad1df35a49

# 2. Review the diff
git apply --stat examples/Revision_v1/proposed/stage_rewiring.patch

# 3. Apply to a throwaway worktree, never to main
git worktree add /tmp/rw HEAD && cd /tmp/rw
git apply examples/Revision_v1/proposed/stage_rewiring.patch
```

```matlab
% 4. MATLAB syntax + static analysis (no experiment runs)
cd examples/Revision_v1
checkcode('run_all_revision_experiments.m')     % expect no errors

% 5. Registry + preflight, no computation
run_all_revision_experiments('full', 'dry_run', true)
%    EXPECT: preflight reports "A4_NOT_IMPLEMENTED"; stage table lists
%            S1, EXP2, EXP2b, EXP3, A4, EXP1, EXP5 in that order.

% 6. Preflight must BLOCK a real full campaign
run_all_revision_experiments('full')
%    EXPECT: error run_all:PreflightFailed -- P3 mandatory stage not implemented: A4
%            and NOTHING computed.

% 7. Fail-loud infrastructure still intact (Gate I1)
run_all_revision_experiments('smoke')
%    EXPECT: error run_all:GateI1Confirmed (unchanged behaviour)

% 8. Individual stage still runnable
run_all_revision_experiments('stage', 'S1', 'dry_run', true)
```

---

## 6. Recommendation

**Safe to apply — with two conditions.** The patch is mechanically verified
(`git apply --check` passes), touches only the registry / dispatch / gates / preflight,
and leaves every numerical implementation untouched. It strictly *tightens* acceptance:
nothing that previously failed can now pass.

It also fixes two live defects beyond the literal request, both of which threaten
published numbers:

- the **EXP5 silent fallback** to hardcoded historical timings on resume;
- the **late output-conflict abort** that wasted the 2026-07-06 campaign.

**Condition 1 — do not apply and launch in the same step.** Run validation steps 4–8 in
a worktree first. The patch has never been parsed by MATLAB.

**Condition 2 — schedule the two follow-ups (R2, R4) before any result is quoted.**
Cap gates are still absent on EXP1/EXP2b, so a capped EXP2b would still be accepted; and
CR2 is no longer represented in the registry at all. Neither is fixable inside a
"registry-only" patch, but leaving them implicit would recreate exactly the class of
error this audit exists to eliminate.

If you would rather keep the blast radius minimal, the patch can be split: hunks for the
registry/adapters/preflight (the requested change) and hunks for the EXP5 resume fix
(a bug fix). They are independent.
