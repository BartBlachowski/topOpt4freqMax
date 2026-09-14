# Restricted re-verification — B-1, B-2, B-3 only

Scope limited to the three blockers. No previously closed scientific question is reopened.

---

## B-1 — Yuksel timing replay work identity → **CLOSED**

**Does the replay now execute the same native work?** Yes. At 160×20 the replay executes Stage 1 = 121 against a recorded native Stage 1 of 121; at 240×30, 168 against 168. Before the fix the same replays executed 120 and 167.

**Is Stage 1 no longer clamped?** No longer clamped on the timing path. The effective Stage-1 budget reported by telemetry is 2000 (the recorded budget), not the Stage-2 horizon, and Stage 1 terminates on its own convergence test rather than the cap. The historical clamp remains the default for every other caller, so nothing else changed.

**Are Stage 2 and total accounting consistent?** Yes. Stage 2 equals the horizon exactly at both meshes (120, 167), and `total = Stage 1 + Stage 2` holds exactly (241 = 121 + 120; 335 = 168 + 167). Both are asserted at runtime, not merely observed: a mismatch raises `iefinal:TimingWorkMismatch` and fails closed.

**Is the timing firewall unchanged?** Yes. Candidate C, topology, persistence, rendering, figure export and trajectory disk I/O remain absent from the timed function (asserted against its source). Serial single-thread execution, one discarded warm-up plus three retained replays, and the deterministic fixed horizon are unchanged. Nested MMA inner work still counts inside native time.

Two further guarantees worth stating: the expected work is read from the **recorded native evidence** rather than any hard-coded constant, and the two meshes have genuinely different Stage-1 counts (121 vs 168), so no compensation constant could have produced this result.

---

## B-2 — common support → **CLOSED**

**Is common support explicitly computed?** Yes. `fit_scaling_table` determines the participants for each metric, intersects their eligible mesh sets into `S_common`, and fits the `common` family on that intersection only — constructed **before** fitting.

**What exact support is fitted in each validation case?**

| case | S_common | n | feasible |
|---|---|---:|:-:|
| all methods, all meshes | 3200, 7200, 12800, 20000 | 4 | yes |
| one method lacks one mesh | 3200, 7200, 12800 | 3 | yes |
| different methods lack different meshes | 7200, 12800 | 2 | no (fail-closed) |
| one common point | 7200 | 1 | no (fail-closed) |
| zero common points | *(empty)* | 0 | no (fail-closed) |
| RUN_ERROR cell present | 3200, 7200, 20000 | 3 | yes |
| end-to-end containment run | 7200 | 1 | no (fail-closed) |

Every set is asserted explicitly in the tests.

**Can a missing or RUN_ERROR cell leak into the fit?** No. Eligibility requires `P = 100`, a status in the frozen `fit_eligible` set, and a positive finite value. A `RUN_ERROR` cell fails all three. Verified directly: with Olhoff failing at 320×40, `12800` is absent from `S_common`, every common fit reports `n_valid = 3`, and no `included_meshes` string contains `12800`.

**Are C, p, R² and LOO computed on the identical support?** Yes, structurally — all four come from a single `ie2a.fit_power_law` call on one vector pair drawn from that support. They cannot diverge. Each row also carries `n_support` and the literal `support_meshes` list, and `scaling_common_support.csv` discloses the support per metric, so any published curve's support is auditable from the artifact.

A `common` fit can never silently fall back to method-specific support: the common rows are computed from `S_common` alone, and when the intersection is below the frozen minimum of 3 the fit is refused (`fitted = false`, `C = p = NaN`) with a stated reason.

---

## B-3 — cell-local failure containment → **CLOSED**

**Does a cell-local genuine failure produce RUN_ERROR?** Yes. Both injected failures produced a full 9-row q/P block with `status = 'RUN_ERROR'`, `censoring_reason = 'RUN_ERROR'`, and the exception identifier and message recorded.

**Does the campaign continue?** Yes. 6 of 6 cells executed with failures injected at an early mesh (160×20) and at the final mesh (320×40); later cells succeeded normally. Rows are checkpointed to disk after every cell, so no later failure can destroy completed work.

**Are unavailable quantities N/A?** Yes. `k_enter`, `k_cert`, `b_ref`, `B_meas`, `E1`, `E2`, `E3`, `Q`, the topology/volume/hard-gate flags and every accounting and timing field are `NaN` — never 0. `validate_results` now rejects a RUN_ERROR row carrying any non-`NaN` scientific quantity, and rejects one missing its error identifier. Method, variant, mesh, configuration and provenance identity are preserved.

**Is topology suppressed?** Yes. 4 of 12 rendered cells were skipped — the failed cells are drawn as labelled empty cells through the shared renderer, with no design substituted from another mesh or iteration.

**Are invalid cells excluded according to frozen scaling rules?** Yes. `RUN_ERROR` fails the frozen eligibility test, so failed cells enter neither the available nor the common fits, and their meshes drop out of `S_common`.

**Do integrity failures still abort rather than become RUN_ERROR?** Yes. The classifier is an allowlist, so anything unnamed is integrity by default. `iefinal:FingerprintMismatch`, `iefinal:ResultSchema`, `iefinal:PreflightFailed`, `iefinal:TrajectoryPrecision`, `iefinal:TrajectoryIdentity`, `iefinal:TimingWorkMismatch`, all `ie2a:*`, and ordinary programming errors (`MATLAB:badsubscript`, `MATLAB:undefinedFunction`, `MATLAB:nonExistentField`) are all confirmed campaign-fatal and rethrown.

One limitation, stated plainly: `localProduction` itself cannot be executed without unlocking production, which this task must not do. The verification drives the same components through the same control flow and additionally asserts `run.m`'s loop structure statically. That is the boundary of the evidence.

---

## New production-affecting defects

**None.** One latent defect was found while raising the verification floor to 160×20 — the shared renderer returns graphics handles that are invalid after it closes the figure, breaking `run_smoke`'s JSON summary. Production discards that return value and was never affected, so it is not a production blocker. It is fixed on the harness side; the frozen renderer is untouched.
