# PROVENANCE — causal controller validation

Machine-readable forms: `evidence/provenance_start.json`,
`evidence/provenance_final.json`, `evidence/single_factor.json`,
`evidence/baselines.json`, `evidence/software_tests.json`, `DATA_MANIFEST.json`.

---

## 1. Repository state

| | at task start | at task end |
|---|---|---|
| branch | `benchmark-methodology-r2` | `benchmark-methodology-r2` |
| HEAD | `b6014ba8bca41f85671d79ab4c8bdee7419880bb` | *see `FINAL_SHA256.txt`* |
| `git status --porcelain` | **empty (clean)** | *see REPORT.md* |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 files) | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| currentness | `CURRENT` | `CURRENT` |

The starting tree was clean. The one file added and six modified under `+impl/`
are itemized in `IMPLEMENTATION.md` §2; `SOURCE_MANIFEST.json` was rewritten
through the sanctioned route (`olhoffcurrent_source_manifest('Write',true)`)
after the edits, so local integrity is internally consistent and the change is
recorded rather than hidden.

## 2. Environment

| | |
|---|---|
| MATLAB | **25.2.0.2998904 (R2025b)** — the base build |
| threads | `runtime.singleThread = true` → `maxNumCompThreads(1)`, asserted in the driver |
| platform | macOS (Darwin 25.6.0), Apple silicon (`maca64`) |
| production preset | `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` |

**MATLAB build differences across the evidence, disclosed.** The 160×20 and
320×40 production baselines (`move_stop`) and the 240×30 withheld-mechanism study
were produced under **25.2.0.3042426 (R2025b) Update 1**. The 400×50 baseline and
fixed-move arm (`move_activity_400`) were produced under **25.2.0.2998904**, the
build in use here. So:

* the **400×50** production-vs-candidate comparison is *same-binary*, and the
  §6 prefix-equivalence check against `F400` is meaningful as a bitwise test;
* the **160×20** and **320×40** comparisons are *same-configuration* but not
  *same-binary*. They are sound scientific comparisons; they are not bitwise
  reproductions, and are never described as such.

## 3. Gates (Phase 0)

All required, all recorded in `evidence/provenance_start.json`:

| gate | result |
|---|---|
| currentness | `CURRENT` |
| source integrity | 74/74 at start, 0 mismatches / 0 missing / 0 extra |
| dispatch (`olhoffcurrent_assert_dispatch`) | `ok = 1`, 0 blockers, 0 warnings |
| published MMA wins | yes — `mmasub` resolves under `mma_published/` |
| sensitivity filter wins | yes — `filter.type = 'sensitivity'`, `applyTo = 'all'` |
| forbidden Olhoff paths on the MATLAB path | **none** |
| tolerance law identity | `cfg.stop.tolerance == 0.05*sqrt(NE/3200)` at 160×20, 240×30, 320×40, 400×50 |
| `move.levels` | `[0.04 0.02 0.01 0.005]` |
| repository test suite | 5/5 suites, **0 failures** |
| controller software tests | 17/17, **0 failures** |
| single-factor gate | `CONTROLLER_SINGLE_FACTOR_PASS` |

## 4. Recovery of the frozen rule

`CONTROLLER_DEFINITION_RECOVERY_PASS`. Recovered from
`diagnostics/two_branch_maturity_240/PREREGISTRATION.md`
(SHA-256 `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd`) and
its executable form `scripts/tb_branches.m`, then **verified numerically** rather
than merely read: re-running `tb_branches` against the one surviving raw
fixed-move trajectory reproduced nine recorded 400×50 quantities bit-exactly
(`PREREGISTRATION.md` §2.6).

## 5. Missing prior evidence — disclosed, not worked around

Five raw artefacts named by earlier frozen studies are **absent from this
machine**:

```
diagnostics/two_branch_maturity_240/runs/runD_240x30.mat
diagnostics/two_branch_maturity_240/evidence/tb_analysis.mat
diagnostics/dynamical_regime/runs/runB_320x40.mat
diagnostics/fixedmove_400_dynamics/runs/runC_400x50.mat
diagnostics/fixedmove_400_dynamics/evidence/fm_analysis.mat
```

This is the same `.mat` retention loss `EVIDENCE_POLICY.md` was written about,
recurring in studies completed after that policy. Consequences for *this* study,
stated plainly:

* the 160×20, 240×30 and 320×40 fixed-move events could not be **recomputed**;
  they are read from the tracked `METRICS.json` of the frozen studies. Only the
  400×50 event was recomputed from raw data;
* `F400` itself stops at 369, so Branch B's *persistence* beyond 369 cannot be
  re-verified from surviving data — only the window's first iteration;
* the 160×20 and 320×40 production **final density vectors** are gone, so
  density-field distance and topology images for those two meshes compare against
  nothing. Those fields are marked `UNAVAILABLE` in `BASELINES.md` and in the
  figures, never fabricated.

The `move_activity_400` evidence gate passes (2/2 required artefacts present and
hash-valid), which is why the 400×50 comparison is the strongest of the three.

**This study's own raw evidence is declared and gated**, in
`analysis/OlhoffCurrent/evidence/two_branch_controller_validation/`, through
`EVIDENCE.json` and `olhoffcurrent_evidence_gate`.

## 6. Scientific runs

Exactly **three**, all candidate-controller, all with the same controller source:

```
C160x20    160x20
C320x40    320x40
C400x50    400x50
```

No 240×30 candidate. No 480×60, 560×70, 640×80, 720×90, 800×100. No nine-mesh
campaign. No rerun of any fixed-move mechanism arm. No production rerun. No
solver copy — `olhoffSolve` is the one solver, called unmodified in the sense
that the same file serves production and candidate and selects between them on
configuration alone.
