# EVIDENCE INVENTORY

Exactly **one** scientific optimization run was executed. Everything else is
pre-existing evidence, declared by immutable path and measured SHA-256.

---

## 1. Evidence produced by this task

| artifact | bytes | role |
|---|---|---|
| `analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat` | 131 231 * 10³ approx. | **the new raw evidence**: `RHO` (7200 × 1358), `DRHO`, `move`, `hist`, `cfg`, `meta`, `exh`, `log`; MAT v7.3 |
| `runs/C240x30_iterations.csv` | — | 55-column per-iteration telemetry, written by the same `cv_export` as the prior arms |
| `runs/C240x30_record.json` | — | scalar record: status, stage starts, descents, exhaustion struct, `ρ` hash, config hash, preregistration hash |

The trajectory rebuild was proved exact inside the driver: the design
reconstructed from `res.diag.drho` ends bit-identically at `res.rho`, with
maximum clamp displacement `5.55e-17`.

**Telemetry retained per outer iteration** (Phase 9): `ω₁`, `ω₂`, `gap12`,
volume, `volErr`, `M_nd`, gray, mid, move, stage, `moveChanged`, `descent`,
`beta`, `betaStallRel`, `betaStallFires`, production-shadow stage/move/tolerance,
native-stop predicates (`prodStopRaw`, `prodSettled`, `prodStopAdmit`),
`exA`, `exB`, `exE`, `exNA`, `exNB`, `exDecl`, `exAmp`, `exCos`, `exNet`,
`exMedcos`, `exMednet`, `exTol`, `exStageStart`, `cosT`, `net_ratio`, unsaturated
variants, `boundFrac`, `revFrac`, `maxAbs`, `ratio`, `l2`, `rms`, `stepNorm`,
`path_W`, `net_W`, `nInner`, `cumInner`, `innerConv`, `multN`, `multJ`, `degen`,
`tOuter` — plus the **full raw density trajectory**.

## 2. Pre-existing evidence this study depends on

| artifact | role |
|---|---|
| `diagnostics/two_branch_controller_validation/scripts/cv_config.m` | **the controller**, called unchanged |
| `.../scripts/cv_telemetry.m`, `cv_export.m` | telemetry, called unchanged |
| `.../scripts/cv_run.m` | reference call sequence; **read, never edited** |
| `diagnostics/three_rung_architecture/METRICS.json` | the 160/320/400 rung-4 results this run is compared against |
| `.../evidence/{event_verification,analysis}.json` | prior events and rung decomposition |
| `.../FINAL_SHA256.txt`, `DATA_MANIFEST.json` | the sealed prior study |
| `evidence/two_branch_controller_validation/C{160x20,320x40,400x50}_trajectory.mat` | the three prior causal trajectories |
| `diagnostics/two_branch_maturity_240/METRICS.json` | the **fixed-move** 240×30 arm, used for the S1 cross-check only |
| `+impl/architecture/+olh/+move/{exhaustion,limit}.m`, `olhoffSolve.m` | the frozen rule, read and hashed |

All 27 required artifacts were present and hash-valid at the Phase-0 gate.

## 3. Gaps — declared, not worked around

### 3.1 No 240×30 production baseline exists

`two_branch_controller_validation/evidence/baselines.json` covers 160×20, 320×40
and 400×50 only. Production-relative `ω₁` and `M_nd` gates at 240×30 are reported
**`UNAVAILABLE`** and are **not** imputed, interpolated, or proxied from another
mesh. Every absolute and `S3`-relative quantity is unaffected.

### 3.2 The prior 240×30 raw trajectory is lost

`two_branch_maturity_240` has no `runs/` directory; its raw `.mat` artifacts are
among the losses that motivated the finalization gate, and that study's gate
correctly reports **FAIL** today. Only its hash-valid `METRICS.json` scalars were
used, and only for the S1 cross-check in `COUNTERFACTUAL_VALIDITY.md` §4. That
arm was **fixed-move** and could not have supplied `S2`, `S3` or `F` in any case.

### 3.3 Wall-clock time is unreliable

5.63× drift in seconds-per-inner-MMA-iteration within this run. Reported,
down-weighted, decisive for nothing.

## 4. This study's own outputs

| artifact | kind |
|---|---|
| `PREREGISTRATION.md` + `evidence/PREREGISTRATION.{frozen,sha256}` | frozen `f86d022e…`, **before** the run |
| `evidence/provenance_{start,final}.json` | Phase-0 gate, start and end |
| `evidence/single_factor.json` | single-factor + pre-run ladder-dependence audit |
| `evidence/event_verification.json` | replay, ten validity checks, declaration timing |
| `evidence/analysis.json` | S1/S2/S3/F extraction, rung decomposition, tail analysis |
| `evidence/figures.json` | figure digests |
| `METRICS.json` | frozen verdict mapping applied mechanically |
| `figures/F01…F16` | the sixteen required figures |
| `scripts/r240_{provenance,configaudit,singlefactor,run,finalize}.m` | MATLAB gates, audit, driver |
| `scripts/r240_{frozen,verify,analyze,metrics,figures}.py` | offline analysis |
| `EVIDENCE.json`, `DATA_MANIFEST.json`, `FINAL_SHA256.txt` | fail-closed retention |

Nothing under `+impl/` was written; `git status` reports 0 modifications there.
