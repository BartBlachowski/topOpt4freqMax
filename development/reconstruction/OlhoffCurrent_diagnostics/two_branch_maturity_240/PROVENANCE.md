# PROVENANCE — withheld 240×30 two-branch mechanism test

## 1. State at task start

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| HEAD subject / date | `move transitions` · 2026-09-07 18:03:24 +0200 |
| `git status --porcelain` | three untracked directories — `topology_maturity_transition/`, `dynamical_regime/`, `fixedmove_400_dynamics/` (the three preceding studies, complete and hash-valid). **No tracked file modified.** |
| MATLAB | **25.2.0.3042426 (R2025b) Update 1** |
| Threads | `maxNumCompThreads(1)`, asserted inside `tb_run` before the solve |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → upstream `duOlhoffFrozenM4` |

HEAD was read at task start, not inherited from any brief.

## 2. Phase 0 gates — all PASS

| Gate | Result |
|---|---|
| currentness | **CURRENT** — matches upstream `architecture/canonical-config` @ `695f03bdac20c423a4e1d389cf9db9187597bcc3`, 0 commits ahead |
| source integrity | **PASS** — 74/74 files, `+impl/` tree `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` |
| canonical manifest | **PASS** |
| path / dispatch | **PASS** — `ok=1`, 32 owned symbols, **0 blockers, 0 warnings** |
| **published MMA wins** | **PASS** — `mmasub`, `subsolv` → `+impl/mma_published/` |
| **sensitivity filter wins** | **PASS** — `applyFilter`, `prepFilter` → `+impl/filter/` |
| `olhoffSolve` resolution | **PASS** — `+impl/architecture/olhoffSolve.m` |
| forbidden Olhoff paths absent | **PASS** — 13 declared, none on the live path |
| **prior 160/320/400 evidence hash-valid** | **PASS — 157/157 files** across `move_stop` (24), `admission_rule` (21), `move_transition` (36), `topology_maturity_transition` (15), `dynamical_regime` (33), `fixedmove_400_dynamics` (28) |
| test suite | **PASS — 4/4, 0 failures** |

Not `WITHHELD240_PROVENANCE_FAIL`.

## 3. Preregistration and the withholding discipline

[`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256
`62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd`, frozen
**2026-09-08T12:04:53Z**, before the 240×30 run and **before any 240×30 numeric
content was opened**. Freeze record: `evidence/PREREGISTRATION.frozen`.

**Exactly what was known about 240×30 before the freeze**, disclosed in full:

* the *existence of filenames* returned by a repository-wide `find` for
  `*240x30*` — no file was opened;
* the resolved production `stop.tolerance` for the mesh, **0.075**, which is a
  deterministic function of `NE` under the existing `meshScaled` rule
  (`0.05·√(7200/3200)`) and carries no information about the trajectory.

Every threshold, window, persistence value and predicate in the frozen rule was
derived **only** from 160×20, 320×40 and 400×50, and the rule's training-set
behaviour — including its known weakness at 160×20 — is tabulated in
`PREREGISTRATION.md` §11 *before* the withheld run.

The 240×30 production scalar inventory (§Phase 8, in `DATA_INVENTORY.md`) was
read **after** the freeze and **after** the run had been launched.

## 4. What was executed

**Exactly one scientific optimization run** — 240×30, fixed move 0.04. No
160×20, no 320×40, no 400×50, no other mesh, no second arm, no campaign.

**No solver copy.** `+impl/architecture/olhoffSolve.m` is called unmodified
through the canonical configuration route. Nothing under `+impl/` was written.

### Single-factor gate, before the solve

Resolved config diffed field-for-field over the whole schema against
`olhoffcurrent_config(240, 30, 'MaxOuter', 1200, 'Diagnostics', true)`:

| declared overrides | unexpected drift |
|---|---|
| `runtime.name`, `move.policy`, `move.initial`, `stop.tolerance`, `stop.toleranceRule` | **NONE** |

Config hash `3a4653bd2c874c7b014d0fc330ccb5aa0378a5238031ac9e946b451270fd2bb6`.
`NE = 7200`; production `stop.tolerance` = **0.075**.

### Scope lock, asserted in code

`p = 3` · `eq4b` · `q = 1` · sensitivity filter · `R = 0.06·b` · projection off ·
published MMA · subspace multiplicity size 2, off-diagonals on ·
`move.levels = [0.04 0.02 0.01 0.005]`. Any mismatch aborts.

### Frozen definitions reused by reference

`dr_telemetry.m`, `dr_dyn.m`, `dr_classify.m` are **called** from
`diagnostics/dynamical_regime/scripts/`, not re-typed, so the dynamical
definitions are provably the frozen ones. Their SHA-256 values are recorded as
input dependencies in `DATA_MANIFEST.json`. The two-branch predicates themselves
live in `scripts/tb_branches.m`, written before the run and hashed.

### Density reconstruction

Reconstructed from `res.diag.drho{k}` exactly as `olhoffSolve` forms it,
validated against `hist.vol` to `< 1e-12`, final column asserted identical to
`res.rho`. Failure aborts.

## 5. Integrity after the work

`+impl/` re-verified after the run; all 157 prior evidence files re-verified. See
`REPORT.md`.
