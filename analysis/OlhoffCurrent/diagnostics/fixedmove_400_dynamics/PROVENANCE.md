# PROVENANCE — 400×50 fixed-move dynamics

## 1. State at task start

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| HEAD subject / date | `move transitions` · 2026-09-07 18:03:24 +0200 |
| `git status --porcelain` | two untracked directories — `diagnostics/topology_maturity_transition/` and `diagnostics/dynamical_regime/` (the two preceding studies, complete and hash-valid). **No tracked file modified.** |
| MATLAB | **25.2.0.3042426 (R2025b) Update 1** |
| Threads | `maxNumCompThreads(1)`, asserted inside `fm_run` before the solve |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → upstream `duOlhoffFrozenM4` |

HEAD was read at task start, not inherited from any brief. It equals the two
preceding studies' HEAD because neither committed anything.

## 2. Phase 0 gates — all PASS

| Gate | Result |
|---|---|
| currentness | **CURRENT** — promoted source intact, matches upstream `architecture/canonical-config` @ `695f03bdac20c423a4e1d389cf9db9187597bcc3`, 0 commits ahead |
| source integrity | **PASS** — 74/74 files, `+impl/` tree `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` |
| canonical manifest | **PASS** — recomputed tree hash equals `SOURCE_MANIFEST.json` |
| dispatch / path resolution | **PASS** — `ok=1`, 32 owned symbols, **0 blockers, 0 warnings** |
| **published MMA wins** | **PASS** — `mmasub`, `subsolv` → `+impl/mma_published/` |
| **sensitivity filter wins** | **PASS** — `applyFilter`, `prepFilter` → `+impl/filter/` |
| `olhoffSolve` resolution | **PASS** — `+impl/architecture/olhoffSolve.m` |
| forbidden Olhoff paths absent | **PASS** — 13 declared, none on the live path |
| **prior dynamical-regime evidence valid** | **PASS** — 33/33 hash-valid |
| **400×50 production trajectory valid** | **PASS** — `dynamical_regime/runs/runA_400x50.mat`, SHA-256 `465bb7475342a2fe…`, manifested |
| **320×40 extended fixed-move trajectory valid** | **PASS** — `dynamical_regime/runs/runB_320x40.mat`, SHA-256 `196ce8c317648dd4…`, manifested |
| all retained evidence | **PASS — 129/129 files** across `move_stop` (24), `admission_rule` (21), `move_transition` (36), `topology_maturity_transition` (15), `dynamical_regime` (33) |
| full test suite | **PASS — 4/4, 0 failures** (`test_currentness`, `test_path_isolation`, `test_preset_equivalence` bitwise, `test_source_integrity`) |

Not `FIXEDMOVE400_PROVENANCE_FAIL`.

The `+impl/` tree hash is identical to that recorded by all five preceding
studies, so this task and they share one implementation.

## 3. Preregistration

[`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256
`f6e84d8a65a3d8180508693f307a07ec9ae8ada51cf852a4668c087b1aeafcbc`, frozen
**2026-09-08T10:02:30Z — before the optimization run was launched**. Freeze
record: `evidence/PREREGISTRATION.frozen`.

## 4. What was executed

**Exactly one scientific optimization run** — RUN C, 400×50 fixed move 0.04.
No 160×20, no 320×40, no repeat of 400×50 production, no other mesh, no
nine-mesh campaign.

**No solver copy.** RUN C calls `+impl/architecture/olhoffSolve.m` unmodified
through the canonical configuration route. Nothing under `+impl/` was written.

### Single-factor gate, run before the solve

The resolved config was diffed field-for-field against
`olhoffcurrent_config(400, 50, 'MaxOuter', 1200, 'Diagnostics', true)` over the
entire schema:

| declared overrides | unexpected drift |
|---|---|
| `runtime.name`, `move.policy`, `move.initial`, `stop.tolerance`, `stop.toleranceRule` | **NONE** |

RUN C config hash `9a78ba9c9a577fcd69f9314472def7ca3a4b282edf958e66f6ced4eb684189ad`.
Production `stop.tolerance` at 400×50 resolves to **0.125**
(`meshScaled` = 0.05·√(NE/3200) = 0.05·2.5).

### Scope lock, asserted in code

`p = 3` · `eq4b` · `q = 1` · sensitivity filter · `R = 0.06·b` · projection off ·
published MMA · subspace multiplicity size 2, off-diagonals on ·
`move.levels = [0.04 0.02 0.01 0.005]`. Any mismatch aborts.

### Dynamical definitions reused by reference, not copied

`dr_telemetry.m`, `dr_dyn.m`, `dr_classify.m`, `dr_spatial.m` are **called
directly** from `diagnostics/dynamical_regime/scripts/` rather than re-typed, so
the definitions are provably the frozen ones. Their SHA-256 values are recorded
as input dependencies in `DATA_MANIFEST.json`.

### Native-stop replay validated before the run

The inherited native predicate — `‖Δρ‖₂ < tol` with the settled-move guard —
was validated against two independent known answers **before** RUN C was
launched:

| archive | replay predicts | archive records | match |
|---|---|---|---|
| `move_stop/runs/fixedmove_320x40.mat` (tol 0.1) | 216 | `NATIVE_CONVERGED`, 216 | ✔ |
| `dynamical_regime/runs/runA_400x50.mat` (tol 0.125) | 139 | `CONVERGED`, 139 | ✔ |

### Density reconstruction

Reconstructed from `res.diag.drho{k}` exactly as `olhoffSolve` forms it,
validated against `hist.vol` to `< 1e-12`, final column asserted identical to
`res.rho`. Failure aborts.

## 5. Integrity after the work

`+impl/` re-verified after the run — see `REPORT.md` §gates. No file under any
prior diagnostic directory was written, moved or deleted; all 129 prior evidence
files re-verified.
