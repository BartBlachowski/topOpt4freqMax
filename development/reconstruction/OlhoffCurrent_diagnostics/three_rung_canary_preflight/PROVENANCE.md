# PROVENANCE — three_rung_canary_preflight

## 1. Repository state

| | |
|---|---|
| branch | `benchmark-methodology-r2` |
| HEAD at start and at finish | `013cc48451d33bed61c5c4eea174bbd898d548a2` |
| working tree at start | clean except the untracked `diagnostics/nine_mesh_campaign_audit/` |
| created by this study | `diagnostics/three_rung_canary_preflight/` only |
| modified by this study | nothing outside that directory |
| implementation under test | `analysis/OlhoffCurrent/+impl`, tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`, 75 files, verified 75/75 |

The `+impl` tree hash is byte-identical to the `implTree` recorded *inside* the
validated three-rung C320 run (`three_rung_promotion_validation_retry1/METRICS.json`).
The code path under test is the code path that was validated.

## 2. Host

| | |
|---|---|
| machine | Apple M1 Max, 10 cores, 64 GiB RAM |
| OS | macOS 26.6.2 (build 25G83), Darwin 25.6.0, arm64 |
| MATLAB | `/Applications/MATLAB_R2025b.app`, version **25.2.0.2998904 (R2025b)**, `computer` = MACA64 |
| licence | network manager `zm8pc.ippt.pan.pl:27000`, `USE_SERVER`; no local licence file |
| swap | 0 B in use throughout |
| BLAS thread env | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` all unset |
| thread policy | `runtime.singleThread = true`; the solver calls `maxNumCompThreads(1)` |

Full record: `evidence/host_environment.json`.

### 2a. The licence interruption, and why it is on the record

The study's first preflight attempt **failed**: the network licence manager was
unreachable from the network the host was then on, and MATLAB would not start
(`evidence/preflight_execution_attempt.log`, 2026-09-12T09:17:43Z, MathWorks
Licensing Error 15, code −15.2; TCP 27000–27002 timing out against
148.81.54.174). `cp_preflight` could not resolve a configuration and the gate
aborted with **zero optimization executed**.

Network access was subsequently restored by the owner and the same gate,
unchanged, passed. The failed attempt is retained rather than deleted because
it is the only direct demonstration that the fail-closed mechanism refuses
before compute rather than reporting after it.

### 2b. MATLAB build differs from the validated run's

This host: `25.2.0.2998904 (R2025b)`. The validated three-rung C320 run
recorded `25.2.0.3042426 (R2025b) Update 1` — a later build. No toolchain
version is asserted by the preflight (none had been observed when the manifest
was frozen), so this is a disclosure, not a blocker. It would matter to a
floating-point-exact comparison against the validated C320 endpoint; this study
makes no such comparison, because the canaries sit at meshes the validated runs
never visited.

## 4. What was reused, by reference

| artefact | used for |
|---|---|
| `three_rung_promotion_validation_retry1/scripts/tr_config.m` | **called** by `cp_config.m` to build the canary configuration — the controller is not re-typed here |
| `two_branch_controller_validation/scripts/cv_config.m` | called transitively by `tr_config` |
| `two_branch_controller_validation/scripts/cv_telemetry.m`, `cv_export.m` | called by `cp_run.m`, so the canary CSV is column-for-column the validated C320's |
| `dynamical_regime/scripts/dr_telemetry.m` | called transitively by `cv_telemetry` |
| `nine_mesh_campaign_audit/MASTER_TABLE.csv`, `effective_configs.json`, `HISTORICAL_*.csv`, `LEGACY_VS_THREE_RUNG.csv` | legacy and historical comparators, read only |
| `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` | implementation integrity check |

Nothing in any reused study was modified.

## 5. What this study authored

MATLAB drivers: `scripts/cp_config.m`, `cp_preflight.m`, `cp_run.m`,
`cp_supplement.m`, `cp_dumpcfg.m`, `cp_fixedwork.m`, `cp_hostprobe.m`.

Python analysis: `scripts/cp_confighash.py`, `cp_config.py`, `cp_validate.py`,
`cp_predict.py`, `cp_integrity.py`, `cp_freeze.py`, `cp_host.py`,
`cp_expect.py`, `cp_perf.py`, `cp_figures.py`, `cp_emit_config.py`,
`cp_analyze.py`, `cp_canary_figures.py`, `cp_finalize.py`.

Plus the documents in this directory and the figures in `figures/`.

`cv_export`'s 55-column CSV schema is the validated C320 study's own and was
**not** modified; `cp_supplement.m` emits the additional per-iteration columns
Part B requires (ω₃…ω₅, gap23, `tEig`, `tGrad`, `tInner`, `tOther`) from the
retained trajectory instead, so the canary CSVs stay column-for-column
comparable to that oracle.

## 6. Offline reconstruction — scope and validation

`olhoffcurrent_config_hash` was reimplemented in Python so that the frozen
pre-run manifest could name an expected hash *before* any run. Validation:

* the nine recorded legacy campaign config hashes reproduce **9/9**
  (`scripts/cp_validate.py`);
* the recorded hash of the **validated** three-rung C320 configuration,
  `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`,
  reproduces exactly (`scripts/cp_predict.py`);
* the four-rung variant at the same mesh produces a *different* hash, so the
  reconstruction is sensitive to the single factor under test;
* derived quantities cross-check against independent recorded values: `eps` =
  0.15 / 0.25 and `rminEl` = 3.6 / 6.0 at 480 / 800 match the legacy campaign's
  recorded values at those meshes exactly.

**Scope limit.** This reconstruction is a *prediction of what the driver would
resolve*. It is not a runtime resolution and does not discharge Part A. It is
used for exactly one thing: giving `cp_preflight.m` a frozen value to fail
closed against.

## 7. Timing telemetry class

`tOuter`, `tEig`, `tGrad`, `tInner` are nondeterministic performance telemetry:
written by the solver, never read back, excluded from `olhoffcurrent_config_hash`
and from every scientific-state comparison. No wall-time ratio in this study is
presented as a same-machine causal saving, and every legacy timing quoted
carries its `runtime.diagnostics = false` provenance.

## 8. Epistemic class of the implementation

Unchanged from the parent implementation: a **reconstruction (class C)** of Du &
Olhoff (2007) — internally coherent, not a claimed historical implementation,
and not to be labelled "Olhoff 2007". Nothing in this study strengthens or
weakens that classification.
