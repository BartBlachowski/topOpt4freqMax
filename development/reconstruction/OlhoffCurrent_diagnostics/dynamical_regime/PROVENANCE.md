# PROVENANCE — dynamical-regime evidence generation

## 1. State at task start

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| HEAD subject / date | `move transitions` · 2026-09-07 18:03:24 +0200 |
| `git status --porcelain` | one untracked directory: `analysis/OlhoffCurrent/diagnostics/topology_maturity_transition/` (the preceding study, complete and hash-valid) — no tracked file modified |
| Canonical implementation | `analysis/OlhoffCurrent` |
| MATLAB | **25.2.0.3042426 (R2025b) Update 1** |
| Threads | `maxNumCompThreads(1)` asserted inside `dr_run` before every solve |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → upstream `duOlhoffFrozenM4` |

The HEAD was read at task start and is not inherited from any earlier brief.
It happens to equal the preceding study's HEAD because that study committed
nothing.

## 2. Phase 0 gates — all PASS

| Gate | Result |
|---|---|
| currentness | **CURRENT** — promoted source intact, matches upstream `architecture/canonical-config` @ `695f03bdac20c423a4e1d389cf9db9187597bcc3`, 0 commits ahead |
| source integrity | **PASS** — 74/74 files, `+impl/` tree `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` |
| source manifest | **PASS** — recomputed tree hash equals `SOURCE_MANIFEST.json` |
| path dispatch | **PASS** — `ok=1`, 32 owned symbols resolved, **0 blockers, 0 warnings** |
| **published MMA wins** | **PASS** — `mmasub` → `+impl/mma_published/mmasub.m`; `subsolv` → `+impl/mma_published/subsolv.m` |
| **sensitivity filter wins** | **PASS** — `applyFilter`/`prepFilter` → `+impl/filter/` |
| no forbidden Olhoff path | **PASS** — 13 forbidden paths declared, none on the live path |
| test suite | **PASS — 4/4, 0 failures** (`test_currentness`, `test_path_isolation`, `test_preset_equivalence` bitwise, `test_source_integrity`) |
| retained evidence hash-valid | **PASS — 96/96 files** across `move_stop` (24), `admission_rule` (21), `move_transition` (36), `topology_maturity_transition` (15); 0 mismatches, 0 missing |

Not `DYNAMICAL_AUDIT_PROVENANCE_FAIL`.

The `+impl/` tree hash is identical to that recorded by all four preceding
studies, so this task and they share one implementation.

## 3. Preregistration

[`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256
`2ac361e769fe87ce458a4668840174a53de01de508da6b802c257982cab974b3`, frozen
**2026-09-08T07:50:04Z — before either scientific run was launched**. The freeze
record is `evidence/PREREGISTRATION.frozen`.

## 4. What was executed

**Exactly the two authorized scientific runs.** No third optimization run of any
kind; in particular no 160×20 run, no 800×100, no nine-mesh campaign.

**No solver copy exists.** Unlike the `move_transition` study, neither run needed
a non-schema control path, so both call `+impl/architecture/olhoffSolve.m`
unmodified through the canonical configuration route. Nothing under `+impl/` was
written.

### Single-factor gate, run before each solve

Each resolved config was diffed field-for-field against
`olhoffcurrent_config(nelx, nely, 'MaxOuter', cap, 'Diagnostics', true)` over the
entire schema. Result:

| run | declared overrides | unexpected drift |
|---|---|---|
| **A** 400×50 | `runtime.name` only | **NONE** |
| **B** 320×40 | `runtime.name`, `move.policy`, `move.initial`, `stop.tolerance`, `stop.toleranceRule` | **NONE** |

RUN A is therefore *bit-for-bit the production configuration* apart from its run
name. RUN B differs by exactly the experimental factor (fixed move) and the
stopping policy, as preregistered.

Config hashes: RUN A `044d50a496cb64d4…`, RUN B `4d49b0abf33edf77…`.
Production `stop.tolerance` resolved to 0.125 at 400×50 and 0.1 at 320×40
(`meshScaled` = 0.05·√(NE/3200)).

### Scope lock, asserted in code

`p = 3` · `eq4b` · `q = 1` · sensitivity filter · `R = 0.06·b` · projection off ·
published MMA · subspace multiplicity, size 2, off-diagonals on ·
`move.levels = [0.04 0.02 0.01 0.005]`. Every one asserted before each solve;
any mismatch aborts.

### Density reconstruction

Per-iteration density is reconstructed from `res.diag.drho{k}` exactly as
`olhoffSolve` forms it, validated against `hist.vol` to `< 1e-12`, and the final
column asserted identical to `res.rho`. A failure aborts the run.

## 5. Integrity after the work

`+impl/` re-verified after both runs: unchanged,
`c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c`, 74 files.
No file under any prior diagnostic directory was written, moved or deleted; all
96 prior evidence files re-verified.

## 6. Note on the preceding brief's 400×50 figures

The preceding study established that the 400×50 quantities quoted in an earlier
brief (first-descent `M_nd ≈ 32.33 %`, mature `≈ 16.16 %`, a 2,740,000-value
bitwise prefix, Jaccard 0.960/0.386) had no evidentiary basis, and that the
studies they cited had never existed. **None of those numbers is used here.**
RUN A is the first 400×50 production trajectory this repository has ever held,
and every 400×50 statement in this study is measured from it.
