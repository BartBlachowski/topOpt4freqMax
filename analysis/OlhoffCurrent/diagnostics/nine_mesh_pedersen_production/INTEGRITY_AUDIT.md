# INTEGRITY_AUDIT (Parts 1, 2, 4, 8, 12, 17)

```
NINE_MESH_CAMPAIGN_IDENTITY_PASS
PERFORMANCE_RUNNER_CONFIG_IDENTITY_PASS
NINE_MESH_CAMPAIGN_INTEGRITY_PASS
```

Rules: PREREGISTRATION.md §10. Machine evaluation: `postprocess/nmp_report.py` → `EVIDENCE.json` `verdict_inputs`.

## 1. Before the first solve

| gate | evidence | result |
|---|---|---|
| Part 1 identity, run 1 | `evidence/IDENTITY_CHECK.run1_check_script_defect.json` | FAIL caused by a **check-script defect**. The script compared `cfg.provenance.preset`, which holds the *upstream* name `duOlhoffAdaptivePedersen`, with the OlhoffCurrent name. HEAD, tree, manifest and all nine hashes already matched. The record is kept. |
| Part 1 identity, run 2 | `evidence/IDENTITY_CHECK.json` | **PASS**. Verified: HEAD `b21483b1…`; branch; `+impl` tree `4ba9a3ae…` re-hashed from disk (79 files, no mismatch/missing/extra); SOURCE_MANIFEST `aee44aa9…`; production preset in PROVENANCE, benchmark selector and both frozen files. All nine meshes were resolved **independently** through three routes, and each hash equals **both** `CAMPAIGN_IDENTITY.json` and `NINE_MESH_CONFIGS.json`. The per-mesh formulation assertions pass. Only `nelx`, `nely`, `stop.tolerance` and `runtime.name` vary. The 87 rows of each resolved configuration are dumped and re-hash to the configuration hash. |
| Preregistration | `PREREGISTRATION.md`, SHA-256 `3cd56751…` | frozen (read-only) before any solve |
| Mechanics smoke (non-scientific) | `logs/smoke_*`, `examples/Performance/conference_benchmark/smoke_nine_mesh_pedersen_b21483b/` | Hooks fired at all three points for the warm-up and for 160×20. The precheck correctly flagged the cap-3 configuration hash, and only that. The tap was bit-consistent with the runner accounting. |
| Synthetic end path | `evidence/SYNTHETIC_ENDPATH.json` | PASS: scaling fit → manifest → export → `-v7.3` save → complexity plots → 9 topology images |
| Campaign lock | `CAMPAIGN_LOCK.json`, SHA-256 `202c1f27…` (read-only) | Pins HEAD, tree, manifest, preset, nine hashes, runner committed/edited/patch hashes, `olhoffcurrent_run.m` and hook lines, 13 script hashes, preregistration hash, output root |
| Part 4 runner configuration identity | `evidence/RUNNER_CONFIG_IDENTITY.json` | **PASS**. The edited runner was in place. Route R is `confbench_method_config` with the campaign `outputDir`; route S is the exact `olhoffcurrent_run` pre-solve resolution; route A is the API. All nine hashes equal both frozen files. Formulation assertions per mesh: preset; Pedersen; linear mass; adaptive box; no stage exhaustion; natural design-change stop; no projection; no p-continuation; nested MMA, no SOCP; R = 0.06·b; maxOuter 400. No mesh-specific scientific override. |
| Launcher refusal checks | `scripts/nmp_launch.sh`, `logs/campaign_LAUNCH.json` | passed (lock hash, HEAD, runner edit, tracked-dirty set, fresh output root, fresh log, empty cwd) |
| Hook arming | `logs/campaign_HOOKS_ARMED.json` | 01:29:54. Identity pass; line text verified; breakpoints at 124/128/249 confirmed by `dbstatus` |

## 2. Repository state lock (Part 2)

- **Before the campaign.** No tracked file differed from HEAD (`IDENTITY_CHECK.json` `git.tracked_changes = []`).
- **During the campaign.** Exactly one tracked file differed: `examples/Performance/performance_comparison.m`. It carried the locked two-literal run-selection edit (`evidence/campaign_runner_edit.patch`, `57e20f0c…`: `cfg.methods` Olhoff only, `cfg.runLabel`). All 20 identity snapshots confirm this: arm, 9 pre, 9 post, and after-run.
- **After the process exited.** The diff was verified byte-identical to the locked patch. The file was then restored to its committed bytes `5ed7ddc3…` (`evidence/RUNNER_RESTORE.json`), and the tracked tree is clean again.
- **Scientific source.** `+impl` and `analysis/OlhoffCurrent/*.m` were never dirty.
- **Untracked material.** The pre-existing untracked studies were not deleted, cleaned, stashed or modified.

## 3. Per-mesh PRECHECK and POSTCHECK (Parts 8 and 12)

All times are 2026-09-14, local.

| # | mesh | PRECHECK (before solver timer) | POSTCHECK | pre | post | runner hash = frozen | tap ↔ runner accounting |
|---|---|---|---|---|---|---|---|
| 1 | 160x20 | 01:30:03.022 | 01:33:47.067 | PASS | PASS | `b1a5744d…` ✓ | bitwise ✓ |
| 2 | 240x30 | 01:33:47.714 | 01:39:42.782 | PASS | PASS | `2fac1384…` ✓ | bitwise ✓ |
| 3 | 320x40 | 01:39:43.588 | 01:46:50.973 | PASS | PASS | `a1203d43…` ✓ | bitwise ✓ |
| 4 | 400x50 | 01:46:52.178 | 01:56:22.408 | PASS | PASS | `e40c6ba1…` ✓ | bitwise ✓ |
| 5 | 480x60 | 01:56:23.998 | 02:12:45.489 | PASS | PASS | `8e55d182…` ✓ | bitwise ✓ |
| 6 | 560x70 | 02:12:47.742 | 02:36:32.923 | PASS | PASS | `8b2533a3…` ✓ | bitwise ✓ |
| 7 | 640x80 | 02:36:35.838 | 03:13:31.795 | PASS | PASS | `cbe8dcf0…` ✓ | bitwise ✓ |
| 8 | 720x90 | 03:13:35.518 | 04:12:33.177 | PASS | PASS | `2893ad47…` ✓ | bitwise ✓ |
| 9 | 800x100 | 04:12:39.689 | 05:34:20.842 | PASS | PASS | `f9138743…` ✓ | bitwise ✓ |

**PRECHECK content** (`runs/<mesh>/PRECHECK.json`):

- identity snapshot: HEAD, branch, tracked-change set, runner = locked edit, `olhoffcurrent_run.m` committed, `+impl` tree re-hashed, SOURCE_MANIFEST bytes, lock unchanged;
- hash of the `cfg` object about to be passed to `olhoffSolve` = frozen hash = the hash `olhoffcurrent_run` recorded;
- preset stamp;
- formulation fields;
- run directory fresh;
- runner artifacts for that mesh not yet present;
- `maxNumCompThreads = 1`;
- `olhoffSolve` and `mmasub` resolve inside `+impl`;
- RNG state recorded (the solver uses no random numbers).

**POSTCHECK content** (`runs/<mesh>/POSTCHECK.json`):

- identity snapshot;
- `effective_config_hash` and a re-hash of `out.configuration` equal to the frozen hash;
- preset unchanged;
- tap cross-check: ρ SHA-256 = runner `x`; ω₁₋₃; outer; inner; Σ tOuter, Σ tInner, Σ tEig and callWall compared as IEEE hex.

Hook time was always outside `tCall`, per `runs/HOOK_EVENTS.log`: PRECHECK ≤ 0.154 s (the warm-up; ≤ 0.107 s on campaign meshes), TAP ≤ 0.025 s, POSTCHECK ≤ 0.121 s.

## 4. Final integrity rules (Part 17)

| rule (PREREGISTRATION §10.3) | result |
|---|---|
| (a) hooks armed; 9 PRECHECK and 9 POSTCHECK passing, in the order 160x20 → 800x100 (event log: warm-up, then the nine meshes) | ✓ |
| (b) runner records: exactly 9, all `method_key = olhoff`, meshes in preregistered order, `effective_config_hash` = frozen | ✓ |
| (c) HEAD, `+impl` tree, SOURCE_MANIFEST, runner and `olhoffcurrent_run.m` identical in all 20 snapshots | ✓ |
| (d) no repeated or replaced run: run directories are exactly the nine meshes plus the warm-up; no `_attempt*` root | ✓ |
| (e) exit code 0; `nmp_after_run` reached at 05:34:51; breakpoints still armed | ✓ |
| (f) tap bit-consistent with runner accounting for all nine meshes; runner `x` hash = tapped ρ hash | ✓ |
| (g) runner preflight PASS (148 checks) | ✓ |
| supplementary: no failure/error marker files; the 13 pinned scripts unchanged | ✓ |
| supplementary: only the Olhoff method executed (methods header, nine run lines, one warm-up line) | ✓, see §5 |

Other facts:

- **Exactly nine authorized scientific configurations were executed.** The runner's own discarded 48×6 warm-up is not campaign data. The runner derived `scientific_evidence = 1`, `performance_campaign = 1`, `cap_summary.any_cap_hit = false`.
- **No unrelated benchmark method ran.** `cfg.methods = {proposed: false, yuksel: false, olhoff: true}` in `benchmark_records.mat`.
- **Timing accounting.** Every record satisfies the runner identity T_total = T_outer\inner + T_inner + T_overhead with a residual of 0 s. The independent solver self-report cross-check is ≤ 6.1e-4 s.
- **No scientific parameter changed.** The hash is identical before and after each solve.
- **Retries.** None. No infrastructure failure occurred.

## 5. Post-processing defect disclosed

The first evaluation of the verdict script (`logs/POSTPROCESS_report.run1_check_defect.txt`, `evidence/EVIDENCE.run1_report_check_defect.json`) reported INTEGRITY_FAIL.

- **What failed.** A supplementary check, not one of (a)–(g). Its regex `^\s+(Proposed|Yuksel)\b` matched stdout line 13, `Yuksel stage budget : 5000 (frozen 1000)`. That line is the runner's header echo of the inert `cfg.yukselMaxIters` literal, and the runner prints it on every run. It is not an execution of Yuksel.
- **State of (a)–(g).** All preregistered criteria were already true in that evaluation.
- **Replacement check.** The check now tests execution directly: the `methods :` header line, the per-mesh `<method> ... <status>` run lines, and the warm-up lines.
- **Negative controls** (`evidence/ONLY_OLHOFF_CHECK_NEGATIVE_CONTROLS.json`). The replacement check fails on the real log with an added Yuksel entry in the methods header, an added Yuksel run line, or an added Yuksel warm-up line. It passes on the real log unchanged.
- **Scope.** No criterion of the preregistration was changed.
