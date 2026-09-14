# PREREGISTRATION — definitive Olhoff nine-mesh Pedersen production campaign

Frozen on 2026-09-14 at about 01:25 +02:00, before any solve of this study.
Its SHA-256 is recorded in `PREREGISTRATION.sha256`, in both lock files, and in `FINAL_SHA256.txt`.

This campaign is **observational**. From this document onward, no scientific setting is modified. That holds in particular after the first solve begins.
A failure, a CAP_HIT, an odd topology, a mode gap, a runtime anomaly or grayness is recorded as evidence. None of them is repaired.

## 0. What happened before this document (resolve-only, no solve)

| step | record | outcome |
|---|---|---|
| Part 1 identity check, run 1 | `evidence/IDENTITY_CHECK.run1_check_script_defect.json`, `logs/PART1_identity_check.run1_check_script_defect.log` | FAIL, caused by a **check-script defect**. The script compared `cfg.provenance.preset` with the OlhoffCurrent preset name, but that field holds the *upstream* preset name `duOlhoffAdaptivePedersen`. HEAD, the tree, the manifest and all nine hashes already matched. |
| Part 1 identity check, run 2 | `evidence/IDENTITY_CHECK.json`, `logs/PART1_identity_check.log` | **NINE_MESH_CAMPAIGN_IDENTITY_PASS**. The corrected assertion requires both names: `provenance.olhoffCurrentPreset` must be the production preset and `provenance.preset` must be `CAMPAIGN_IDENTITY.production.upstream_preset`. |
| mechanics probes (scratchpad only) | described in RUNNER_AUDIT.md | Conditional breakpoints work under `matlab -batch` and survive `clear` and path restore. A breakpoint in a caller file did not slow a JIT-sensitive callee (3.86 s in every condition). A condition that throws halts `-batch` in the debugger. |

## 1. Identity

| item | value |
|---|---|
| branch | `benchmark-methodology-r2` |
| HEAD | `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| git status | no tracked file differs from HEAD. Untracked: the nine pre-existing `analysis/OlhoffCurrent/diagnostics/*` studies listed in the task, plus this study's folder. Untracked material is not touched. |
| upstream shared implementation | `253069262407885a8b759a9e721c4f0a7d3a397d` |
| `+impl` tree SHA-256 (79 files) | `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` |
| `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` SHA-256 | `aee44aa9172a8105314bbb9475e28bed8a53d4d154824a8748abf64a208b1836` |
| `CAMPAIGN_IDENTITY.json` SHA-256 | `deb43641bc42d9db1b3154115478486403dc22ac4701cbde6de34c5d33c44e4f` |
| `NINE_MESH_CONFIGS.json` SHA-256 | `ea2f4751a5bb7c73fa472b57f67a64eaffbfe931fd6f9403ff4a0a45112acf9f` |
| production preset | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (upstream `duOlhoffAdaptivePedersen`) |
| config schema | 87 rows. The hash is `olhoffcurrent_config_hash`, with `runtime.name` excluded. |
| authoritative runner | `examples/Performance/performance_comparison.m`, SHA-256 at HEAD `5ed7ddc3e474095a006ad0024ff83d2e3bea4ba724ec246b945887ec5e496f0f` (blob `1a4dd08b`) |

## 2. The nine meshes and their expected configuration hashes (exactly these, in this order)

| # | mesh | NE | stop ε = 0.05·√(NE/3200) | r_min in elements = R/(b/nely) | expected config hash |
|---|---|---|---|---|---|
| 1 | 160x20 | 3200 | 0.05 | 1.2 | `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4` |
| 2 | 240x30 | 7200 | 0.075 | 1.8 | `2fac1384239527b1a0108e1d34725021cdc7c5a26f1939b602239415b34847e1` |
| 3 | 320x40 | 12800 | 0.1 | 2.4 | `a1203d43efb240308cfcb520c92a6d9bc37a0c703c728fc2f3d298f779bc5f6e` |
| 4 | 400x50 | 20000 | 0.125 | 3.0 | `e40c6ba16c9addbb587d08ecab29669f7db560d09b980ca62c2096856b849280` |
| 5 | 480x60 | 28800 | 0.15 | 3.6 | `8e55d18251152ec2d26836593489c14f4842390d912d7c66f43f3b18730c83de` |
| 6 | 560x70 | 39200 | 0.175 | 4.2 | `8b2533a3caf8432d40a54dbcbd8958900e333c410527d5547c53699b9e373ccf` |
| 7 | 640x80 | 51200 | 0.2 | 4.8 | `cbe8dcf070d3e318a889867b5168fdbb09f1b6babfd46f981447d96e9a073aaf` |
| 8 | 720x90 | 64800 | 0.225 | 5.4 | `2893ad47136fc9a7775f1f4d9ff1dcba38e7fe14271dd12b5727ff4bd8bb54f3` |
| 9 | 800x100 | 80000 | 0.25 | 6.0 | `f9138743067afd0edce255ad26a6d7ae5c0f632a537516a4793eafecbacbe4cd` |

No other mesh belongs to the campaign.

## 3. Frozen scientific configuration

The complete rows are in `evidence/resolved_configs/<mesh>.txt`. They rehash to the values above. Only `domain.mesh.nelx`, `domain.mesh.nely` and `stop.tolerance` vary across meshes; `runtime.name` also varies but is excluded from the hash.

- **Domain.** 8 × 1 beam, thickness 1. Simply supported at mid-height, both ends axially restrained. Q4 elements, consistent mass. E = 1e7, ν = 0.3, solid density 1.
- **Material.** Pedersen (2000) low-density stiffness: `material.stiffness.model = pedersen`, p = 3, linear below ρ = 0.1. Mass is linear, eq. (2), with q = 1. There is no p-continuation and no mass continuation. Eq. (4b) is not used.
- **Design.** Initial ρ = 0.5, ρ_min = 1e-3, volume fraction 0.5.
- **Filter.** Sensitivity filtering applied to all sensitivities. The radius is physical, R = 0.06 with b = 1, so **R = 0.06·b at every mesh**. The element radius is derived from it and is never fixed.
- **Projection.** Disabled.
- **Eigenproblem.** `eigs` with J = n + N_max = 1 + 4 = 5 modes, tolerance 1e-12, maxit 5000, Krylov factor 4, fixed start vector. The solver uses no random numbers.
- **Multiplicity.** `subspace` method, subspace size 2, diagonal offsets and off-diagonal terms retained. The `multiplicity.tolerance = 0.05` row is present in the configuration.
- **Inner solver.** Nested MMA, published Svanberg variant, increment variable, relative tolerance 0.05, 5 to 500 iterations. There is no LP, no SOCP and no stage exhaustion.
- **Controller.** Adaptive per-element move box: initial value and ceiling 0.1, floor 0.002, growth ×1.2, shrink ×0.7.
- **Stop.** `designChange` rule: ‖Δρ‖₂ < ε, with ε scaled by mesh as in §2. All guards are off (settledMove, ladderExhausted and maxDesignChange false; boxInactiveFraction 0).
- **Budget.** `runtime.maxOuter = 400`. Reaching it is CAP_HIT, never convergence.
- **Runtime.** Single thread, diagnostics recorder off, verbose off.

## 4. Environment

| item | value |
|---|---|
| MATLAB | R2025b, `25.2.0.2998904`, binary `/Applications/MATLAB_R2025b.app/bin/matlab` |
| OS | macOS 26.6.2 (25G83) |
| machine | Mac Studio, Apple M1 Max, 10 cores (8 performance + 2 efficiency) |
| RAM | 64 GB |
| power | AC. The launcher wraps MATLAB in `caffeinate -i -s` to prevent sleep. |
| threads | `maxNumCompThreads(1)`, pinned by the runner (`cfg.singleThread = true`) |
| disk free | 139 GiB |
| other processes | The user's own interactive MATLAB desktop session is open and idle; it is not touched. The agent runs no other MATLAB computation while the campaign process is alive. |
| campaign start timestamp | written by the launcher to `logs/campaign_LAUNCH.json` and copied to RUN_INDEX.json. It cannot be known at freeze time. |

## 5. Runner and invocation

- **Entry point.** Every production solve goes through `examples/Performance/performance_comparison.m` and its own callees: `confbench_method_config`, then `confbench_run_case`, then `olhoffcurrent_run`, then `olhoffSolve`. Nothing else starts a campaign solve.
- **Method and label selection.** The runner has no function, environment or argument interface. It starts with `clear` and its documented control surface is the literal `USER CONFIGURATION` block. The campaign therefore changes **exactly two literal lines** there and nothing else:
  - `cfg.methods = struct('proposed', false, 'yuksel', false, 'olhoff', true);` selects only the frozen Olhoff method through the runner's existing selector.
  - `cfg.runLabel = 'nine_mesh_pedersen_b21483b';` routes output to the fresh canonical root `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/`. The committed label `campaign_9mesh_r2` would overwrite tracked historical results.
- **Everything else stays as committed:**
  - the nine-row `cfg.resolutions` (ascending, exactly §2) and `confirmLongCampaign = true`;
  - `maxOuterOverride = []`, so each method's frozen budget applies;
  - `singleThread = true`;
  - `runWarmup = true`: one discarded 48×6, 5-outer solve outside the campaign meshes;
  - `runEvaluator = true`: the common E1/E2/E3 evaluator, run outside every timer;
  - `fitScaling = true`, CSV, JSON and LaTeX outputs;
  - the timing tolerances;
  - `yukselMaxIters = 5000`, which is inert because Yuksel is disabled.
- **Edit status.** The edit is a change to a tracked **harness selection** file. It is not scientific source: `confbench_method_config` does not read `cfg.methods` or `cfg.runLabel` for Olhoff, and the Olhoff configuration does not depend on `outputDir`, which Part 4 verifies.
  - The edit is pinned in `CAMPAIGN_LOCK.json` by patch and file SHA-256.
  - It is re-verified before and after every mesh.
  - No other tracked file may differ from HEAD.
  - After the campaign process exits, the runner is restored to its committed bytes and the restore is verified by SHA-256.
- **Process model.** One non-interactive MATLAB process runs all nine meshes, in the runner's order, which is ascending. This is the runner's designed campaign form: `performance_campaign = 1` is derived only when the full nine-row matrix runs in one invocation. The launcher is `scripts/nmp_launch.sh campaign`, and the exact command is recorded in `logs/campaign_LAUNCH.json` and RUNNER_INVOCATION.md.

## 6. Observation instrumentation (no source file is changed)

`olhoffcurrent_run` keeps only aggregates of `res.hist`. To retain per-iteration histories of the production solves themselves, three **conditional breakpoints** are armed in the committed file `analysis/OlhoffCurrent/olhoffcurrent_run.m` by `scripts/nmp_arm_hooks.m`. They are set in the same MATLAB process, immediately before `run(performance_comparison.m)`. Each condition calls a function that reads its arguments, writes files and returns `false`, so execution never stops.

| line | statement | hook | when | writes |
|---|---|---|---|---|
| 124 | `tCall = tic;` | `nmp_hook_pre` | configuration resolved and hashed; solver timer not started | `PRECHECK.json` |
| 128 | `out.x = double(res.rho(:));` | `nmp_hook_tap` | right after `callWall = toc(tCall)` | `SOLVER_RESULT.mat` (the solver's `res`, including `hist`, `aux`, `lambda`, `modeTable` and `log`, without the FE model `mdl`), `TAP.json` |
| 249 | `if out.is_warmup` | `nmp_hook_post` | result struct complete | `RUN_OUT.mat`, `POSTCHECK.json` |

- **Timing.** All hook work happens outside `tCall`. It is therefore excluded from `total_wall_time_s` and from every hist timer.
- **Fail-closed precheck.** If a campaign precheck fails, MATLAB exits with code 3 before the solve, so that mesh and every later mesh are not solved.
- **Fail-closed postcheck.** If a postcheck finds a changed or unverifiable identity, MATLAB exits with code 4 and nothing later runs.
- **Never stopping.** Because a throwing condition would halt `-batch`, every hook body is inside try/catch.
- **Per-mesh directories.** Each campaign mesh gets `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/runs/<mesh>/`. The warm-up gets `runs/warmup_48x6/`.

## 7. Pre-campaign mechanics checks (non-scientific, never evidence)

1. **Smoke run.** `performance_comparison.m` runs with the hooks armed in `smoke` mode, using the runner's own non-scientific knobs:
   - Olhoff only;
   - one added line `cfg.resolutions = [160 20];`;
   - `cfg.maxOuterOverride = 3`;
   - label `smoke_nine_mesh_pedersen_b21483b`.

   The runner itself marks this run non-scientific. It validates the hooks at the real call site and the Olhoff-only export path. Its outputs are kept and labelled as smoke.
2. **Synthetic end-path test.** Nine synthetic Olhoff-only records, with no solve, are passed through `confbench_scaling_fit`, `confbench_manifest`, `confbench_export`, the `-v7.3` save, the complexity plots and the topology images, in a temporary directory. This proves that the end of a full run cannot fail after hours of solving.

If either check exposes an **instrumentation** defect, the instrumentation may be corrected and the check repeated, with every attempt kept. Scientific settings cannot be changed by this route.

## 8. Failure and retry policy

- **Scientific outcomes.** CONVERGED, CAP_HIT, SOLVER_FAILURE, odd topology, near-multiplicity, grayness and slowness are reported unchanged and never retried.
- **Infrastructure failures.** These are a MATLAB crash unrelated to solver state, reboot, filesystem or disk error, a license problem, or an external kill. In that case:
  - the failed attempt's output root and log are renamed with the suffix `_attempt1_<reason>` and kept;
  - identity is re-proved;
  - the full nine-mesh runner invocation is repeated unchanged, except that the label gains `_attempt2`, because the runner cannot resume single meshes;
  - both attempts are documented.
- **Precheck or postcheck failures.** These are campaign integrity failures. They are reported, not retried.

## 9. Analysis plan (post-processing only; nothing is re-solved to obtain a metric)

**Per-mesh metrics.** Sources are the tapped `res`, the runner record and the final ρ.

- **Size.** NE; DOFs as `mdl.ndof` total and free.
- **Iterations.** Outer = `numel(hist.N)`. Inner total = `Σ hist.nInner`, together with the mean and maximum per outer iteration.
- **Frequencies.** Native ω₁, ω₂, ω₃ from `res.omega`, and λ₁, λ₂ from `res.lambda`.
- **Gap.** Terminal gap (ω₂−ω₁)/ω₁, and the minimum over the history of `hist.gap12`.
- **Volume and grayness.** Volume = mean(ρ) and volume error = mean(ρ) − 0.5. M_nd = 4·mean(ρ(1−ρ)). Gray fraction = mean(0.1 < ρ < 0.9), the upstream `greyFraction` and the evaluator's `gray_fraction_01_09`. The runner's evaluator topology metrics are reported as they are.
- **Termination.**
  - runner status and status note;
  - whether the `converged at outer iteration` log line is present;
  - final ‖Δρ‖₂ against ε, and final max|Δρ|;
  - final largest per-element box `hist.move(end)`, and the mean box `aux.moveMean(end)`;
  - the count of nested MMA solves that did not converge.
- **Timing.**
  - total wall time `callWall`;
  - Σ tOuter, and the mean and median of tOuter;
  - first-window and final-window tOuter statistics, each over **20** outer iterations;
  - Σ tEig and eigen time per outer (tEig spans FE assembly plus `eigs`);
  - inner time per outer and per inner iteration;
  - outer time excluding inner time.
- **Eigensolves.** Count = outer + 1: one per outer iterate plus the final analysis. This is derived from the code.
- **Hashes.** ρ SHA-256 over little-endian IEEE bytes in column order, the same convention as the upstream sweep audit; SHA-256 of the result file, the log and the configuration.
- **Eq. (4) re-evaluation of the final design.** SIMP p = 3 stiffness with eq. (4) mass (cut-off 0.1, exponent 6) and 3 `eigs` modes, using the arithmetic of `sd_verify_sweeps.m`. It serves **only** the like-for-like comparison with the upstream reported table, which used that re-evaluation.

**Spectral audit.** Only existing thresholds are used:

- **Near-multiplicity.** gap < `multiplicity.tolerance` = **0.05**, the configuration row.
- **Coalescence.** gap < **0.02**, the upstream `repro/run_repro.m` `coalescenceIter` test.
- **Sudden ω₁ drop (spike).** ω₁(k) < **0.7**·ω₁(k−1), the upstream sweep audit's `spike_events`.
- **Effectively bimodal terminal state.** Terminal native gap < 0.05.
- **Crossing or touching.** An iteration with gap < 0.02 later followed by gap > 0.05. Reported as a count.
- **Localized low-density mode indicators.**
  - spike events, and whether ω₁ recovers to ≥ 0.99 of its pre-spike value;
  - the runner evaluator's E1/E2/E3 `status` and selected ordinal;
  - a descriptive number with no threshold: the share of first-mode kinetic energy in elements with ρ ≤ 0.1, from one eigen-analysis of the final design in the native model.

**Topology.** The runner's shared renderer images are used, together with one nine-panel figure using the same [0,1] gray scale and 8 × 1 extent. Neighbour-mesh L1 and IoU at 0.5 are computed on a common grid by nearest-neighbour upsampling. They are descriptive only.

**Performance.** Reported as A, total wall time per mesh, and B, per-outer cost: wall per outer, mean and median tOuter, eigen per outer and inner per outer. Log-log OLS of each quantity against NE uses all nine points. The leave-one-out exponent range is reported and no point is excluded. The runner's own `confbench_scaling_fit` is reported as it stands. DOFs scale as ≈ 2·NE, so exponents against DOFs equal those against NE.

## 10. Verdict rules (fixed now)

The thresholds below were chosen by the analyst, who knew the historical sweep. They are generic engineering bounds and are **not fitted to that sweep**. The sweep is never an acceptance criterion.

1. **NINE_MESH_CAMPAIGN_IDENTITY_PASS / FAIL.** PASS iff every check in `nmp_identity_check('identity')` passes. Already issued: PASS, §0.
2. **PERFORMANCE_RUNNER_CONFIG_IDENTITY_PASS / FAIL.** PASS iff every check in `nmp_identity_check('runner')` passes with the edited runner in place. That covers routes R (runner method configs with the campaign `outputDir`), S (the exact `olhoffcurrent_run` pre-solve resolution) and A (API), all nine hashes against both frozen files, the per-mesh formulation assertions, and the lock. It is run after the lock and before launch. A FAIL stops before the first campaign solve.
3. **NINE_MESH_CAMPAIGN_INTEGRITY_PASS / FAIL.** PASS iff all of the following hold:
   - (a) the hooks were armed and nine campaign PRECHECKs and nine POSTCHECKs exist, all `pass = true`, in the order of §2;
   - (b) the runner records are exactly nine, all `method_key = olhoff`, with meshes equal to §2 in order and `effective_config_hash` equal to the frozen hash;
   - (c) every identity snapshot (arm, 9 pre, 9 post, after-run) shows the same HEAD, `+impl` tree and SOURCE_MANIFEST;
   - (d) no run was repeated, or repeats exist only under §8 and are retained;
   - (e) the process exit code is 0, `nmp_after_run` was reached, and the breakpoints were still armed;
   - (f) the tap is bit-consistent with the runner accounting for all nine meshes (ρ hash, ω₁–₃, outer, inner, Σ tOuter, Σ tInner, Σ tEig, callWall);
   - (g) the runner preflight passed.
4. **PEDERSEN_ADAPTIVE_CROSS_MESH_TERMINATION_PASS / FAIL.** PASS iff at **all nine** meshes:
   - status is `NATIVE_CONVERGED`, with solver status `CONVERGED` and the log line present;
   - final ‖Δρ‖₂ < ε;
   - outer < 400;
   - the count of nested MMA solves that did not converge is 0;
   - the final largest per-element box is above the floor 0.002. A stop with every element box at the floor is a box-driven, pathological termination.

   Any CAP_HIT, SOLVER_FAILURE, RUN_ERROR or UNRECOGNIZED_STOP gives FAIL.
5. **PEDERSEN_ADAPTIVE_MESH_TREND_ACCEPTABLE / SUSPICIOUS.** SUSPICIOUS iff any flag fires:
   - S1: any mesh is not `NATIVE_CONVERGED`;
   - S2: any adjacent-mesh relative change |Δω₁|/ω₁ is above 2 %;
   - S3: the coarse-to-fine relative change of ω₁ from 160x20 to 800x100 is above 5 %;
   - S4: the largest adjacent relative change among the four finest pairs (480→560 … 720→800) exceeds both 0.5 % and the largest among the four coarsest pairs (160→240 … 400→480);
   - S5: any |volume error| is above 1e-3;
   - S6: M_nd(800x100) > 2·M_nd(160x20), or gray fraction(800x100) > 2·gray fraction(160x20);
   - S7: a localized-mode indicator fires at a terminal state, meaning a spike event in the last 10 outer iterations or an evaluator status other than PASS.

   Otherwise ACCEPTABLE.
6. **OLHOFF_PRODUCTION_BASELINE_VALIDATED / NOT_VALIDATED.** VALIDATED iff verdicts 1 and 2 are PASS, 3 and 4 are PASS, 5 is ACCEPTABLE, and the runner's timing-accounting identity passes for all nine records. VALIDATED means only that the formulation is a defensible, reproducible, computationally characterized **observational** production baseline for the benchmark comparison. It does **not** establish KKT stationarity. It does not establish equivalence to Du & Olhoff (2007) either.

## 11. Comparison with the upstream fixed-R sweep (only after this campaign is frozen)

**Source.** `analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration/evaluations/sweep_verification.json` (sweep `S`) and the snapshot `res.mat` files, read-only. The values quoted in the task (169.7 / 170.8 …) are the **eq. (4) re-evaluation** of those designs. The native frequencies are separate: 169.21 / 170.40 … . The comparison is therefore like-for-like: native with native, and eq. (4) with eq. (4).

**Classification of each difference:**

- **Negligible or numerical.** Identical outer and inner counts, and |Δω|/ω ≤ 1e-12 or an identical ρ hash.
- **Plausibly environmental or reporting.**
  - wall-clock differences;
  - differences explained only by the reporting model (native vs. eq. (4)) or by rounding in the quoted table;
  - |Δω|/ω ≤ 1e-6 with identical counts.
- **Scientifically material.** Any difference in outer or inner count or in terminal status, or |Δω|/ω > 1e-6.

Nothing is rerun to reduce a discrepancy.

## 12. Historical gray/KKT context

`gray_kkt_forensic_audit` keeps `PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE`. That verdict concerns the historical formulation and states studied there. It is not changed, and it is not reinterpreted as a Pedersen result. This campaign neither proves nor disproves KKT stationarity of the Pedersen/adaptive formulation.
