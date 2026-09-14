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

---

# ADDENDUM — resumed session, 2026-09-09

This study was resumed in a second session. Everything below was established
**after** the preregistration was frozen and after C160/C320 had run; nothing
here changes the controller, the rule, or any preregistered bound.

## A1. Repository state at resumption

| | |
|---|---|
| branch | `benchmark-methodology-r2` |
| **HEAD at resumption** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` ("A & B tests") |
| `git status --porcelain` | **empty (clean)** |
| currentness | `CURRENT` |
| source integrity | **PASS** — `sourceOk = 1`, 0 mismatches, 0 missing, 0 extra |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| dispatch | ok, 0 blockers, 0 warnings |
| published MMA wins | yes | 
| sensitivity filter wins | yes |
| forbidden Olhoff paths | absent |
| tolerance law identical at all four meshes | yes |

The resumption HEAD is **not** the `b6014ba` recorded at first-session task
start: the first session's work was committed as `1438aa3`. The `+impl/` tree
hash at resumption is **byte-identical** to the `implTree` recorded inside both
`C160x20_record.json` and `C320x40_record.json`, so the controller source that
produced those runs is exactly the source present now. Machine-readable:
`evidence/provenance_resume_20260909.json`.

## A2. The MATLAB build changed between the runs and the resumption

The only R2025b installation on this machine is `/Applications/MATLAB_R2025b.app`,
and it now reports **`25.2.0.3042426 (R2025b) Update 1`**. The frozen
preregistration, `evidence/provenance_start.json`, and both completed candidate
runs all record **`25.2.0.2998904`**, the base build. The build was updated
between the first session and this one; the base build is no longer available
here, so it cannot simply be reselected.

This matters because PREREGISTRATION §6 conditions the C400 prefix-equivalence
check on *the same MATLAB build*, and because a build change between C160/C320
and C400 would leave the three runs mutually incomparable.

**It was tested rather than assumed.** C160 was re-executed under Update 1
through the unchanged `cv_run(160,20)` and compared against the committed
base-build run at full double precision, column by column, iteration by
iteration (`scripts/cv_binary_equiv.py`,
`evidence/rerun_20260909/C160x20_binary_equiv.json`):

| | |
|---|---|
| rows | 219 vs 219 |
| columns compared | 55 |
| columns differing | **1** — `tOuter`, the per-iteration wall clock |
| numerical columns differing | **0**, over all 219 iterations |
| final ρ SHA-256 | `332c00a5181372bfd0fe82bbe8b7309496b1fb9f082d0a45945a61f0dcd2624a` — **identical** |
| status / outer / inner | `CONVERGED` / 219 / 5074 — identical |
| descents, branches | `103, 142, 181`; `A, A, B` — identical |
| ω₁, ω₂, gap, volume, M_nd, gray, mid | identical to every recorded digit |
| wall time | 425.3 s (base) vs 477.7 s (Update 1) |

**Conclusion: the build change is numerically inert for this solver.** The only
difference it produces is timing. Therefore the three candidate runs are
cross-comparable, §6 remains meaningful, and P14 (one controller, unchanged
across all three runs) is unaffected. The disclosure in §2 above — that the
160×20 and 320×40 comparisons are "not same-binary" — is superseded for the
*candidate* runs by this direct measurement.

Wall-clock figures remain build- and load-sensitive and are reported as such.
The run of record for C160 stays the committed base-build run; the rerun is
retained beside it as verification, in `evidence/rerun_20260909/`.

## A3. Raw evidence present at resumption — and what was lost

`analysis/OlhoffCurrent/evidence/` was **empty** at resumption (only `.gitignore`
and `README.md`). That directory is deliberately untracked, so this is the
"fresh clone" state `EVIDENCE_POLICY.md` anticipates. Lost with it:

| declared artifact | study | status at resumption |
|---|---|---|
| `move_activity_400/P400_400x50_trajectory.mat` | `move_activity_400` (required) | **missing** → recomputed, §A4 |
| `move_activity_400/F400_400x50_trajectory.mat` | `move_activity_400` (required) | **missing** → recomputed, §A4 |
| `two_branch_controller_validation/C160x20_trajectory.mat` | this study | **missing** → restored by the §A2 rerun (ρ hash-identical) |
| `two_branch_controller_validation/C320x40_trajectory.mat` | this study | **missing** — not restored; see REPORT |

`olhoffcurrent_evidence_gate` on `move_activity_400` correctly reported
`ok=0, required=2, missing=2` before the recomputation.

**What survived, contrary to what the preregistration recorded.** The
preregistration §2.6 and §9 state that the fixed-move mechanism arms and the
160×20/320×40 baseline `.mat` files were absent from the machine, and §9
consequently marks the 160×20 and 320×40 baseline final densities
"UNAVAILABLE". At resumption those files are **present** under
`diagnostics/*/runs/`:

| file | contents |
|---|---|
| `move_stop/runs/baseline_160x20.mat` | production 160×20, `RHO` (91 × 3200) |
| `move_stop/runs/baseline_320x40.mat` | production 320×40, `RHO` (131 × 12800) |
| `move_stop/runs/fixedmove_160x20.mat` | fixed move 0.04, 160×20, 400 outer |
| `move_stop/runs/fixedmove_320x40.mat` | fixed move 0.04, 320×40, 216 outer |
| `dynamical_regime/runs/runB_320x40.mat`, `runA_400x50.mat` | mechanism arms |
| `fixedmove_400_dynamics/runs/runC_400x50.mat` | mechanism arm |
| `two_branch_maturity_240/runs/runD_240x30.mat` | the withheld 240×30 arm |

These were verified to *be* the frozen baselines rather than assumed to be:
recomputing `M_nd`, `gray`, `mid` and `volume` from the final `RHO` column of
each baseline file reproduces `evidence/baselines.json` to ~1e-14 (floating-point
summation order), at the recorded `nOuter` of 91 and 131 respectively.

Consequence: the production final densities that §9 marks UNAVAILABLE **are
available**, so the topology comparison the preregistration expected to be
impossible at 160×20 and 320×40 can in fact be made. This is an improvement in
available evidence, not a change to any preregistered definition or bound.

## A4. Recomputation of the lost 400×50 evidence, and what it proves

`ma4_run('P',400,50)` and `ma4_run('F',400,50)` were re-executed under MATLAB
Update 1 through their own unchanged drivers, restoring the two declared
`move_activity_400` artifacts.

**ARM P400 reproduces the original bitwise.** The study's tracked per-iteration
telemetry `move_activity_400/runs/P400_400x50_iterations.csv` was overwritten by
the recomputation and then compared against the committed file: **all 35 columns
over all 139 rows are bitwise identical** — `omega1`, `omega2`, `beta`, `l2`,
`nInner`, the whole activity distribution. `git diff` reports the file unchanged.
Independently, the final density hash of the regenerated trajectory is

```
0d8b799c77b86f62c9113e88331052b0431c4917ad2740fd651834c040eedaaf
```

which is **exactly** the `rho_sha256` frozen in `evidence/baselines.json` before
the file was lost, and `M_nd` reproduces to 13 decimals.

**Container hashes cannot be restored, and are not claimed to be.** The `.mat`
file hash differs from the declared one (36 585 921 B vs 36 576 273 B) because a
v7.3 MAT-file is HDF5 and embeds creation metadata, and because `meta` now
carries the Update-1 version string. The *content* is what was declared; the
*container* is new. `EVIDENCE.json`'s file-level hash therefore no longer matches
and is re-declared through the sanctioned route rather than edited by hand, with
the original declaration preserved.

**Consequence for §A2.** The build-inertness conclusion now rests on two
independent bitwise reproductions at opposite ends of the mesh range:

| check | mesh | iterations | result |
|---|---|---|---|
| C160 candidate rerun vs committed run | 160×20 | 219 | 54/55 columns bitwise; only `tOuter` (wall clock) differs; final ρ hash identical |
| P400 production recompute vs committed run | 400×50 | 139 | **35/35 columns bitwise**; final ρ hash identical to the frozen value |

The MATLAB base → Update 1 change is numerically inert for this solver at both
3 200 and 20 000 elements. The candidate runs are cross-comparable and §6 remains
a meaningful bitwise test.

## A5. An ω₁ convention inconsistency in the frozen baselines, disclosed

Two different final-ω₁ conventions exist in the evidence and
`evidence/baselines.json` inherited both:

* **final re-solve** — `res.omega(1)`, the eigenproblem evaluated on the design
  after the last update. This is what `move_stop/METRICS.json` records, what
  `baselines.json` carries for **160×20 and 320×40**, and what `cv_run` records
  for **every candidate run**.
* **last history entry** — `hist.omega(end)`, ω at the design used for the final
  iteration's sensitivities. This is what `move_activity_400/METRICS.json`
  records and what `baselines.json` carries for **400×50**
  (162.882615630062 = the last CSV row).

So the 160×20 and 320×40 comparisons are like-for-like, while the 400×50
production baseline is stated in the *other* convention. The like-for-like
production value at 400×50, measured in this session's recomputation, is

```
res.omega(1) = 162.8887798      vs      frozen baseline 162.882615630062
```

a relative difference of **3.8e-5**. Both are reported. The discrepancy is three
orders of magnitude below the P8 gate (1 % one-sided) and cannot affect any
verdict; using the frozen value makes P8 marginally *easier*, so the gate is
evaluated against both and passes or fails identically.

## A6. F400 restored, the evidence gate returned to PASS, and the frozen rule re-verified independently

`ma4_run('F',400,50)` reproduced the fixed-move arm: **369 outer, `CONVERGED`**,
`M_nd = 16.1589 %` against the frozen 16.158892933214315. Its tracked telemetry
`move_activity_400/runs/F400_400x50_iterations.csv` was overwritten by the
recomputation and `git diff` reports it **unchanged** — bitwise identical, as P400's
was.

`ma4_declare` was then re-run, the sanctioned route, which re-hashes the files on
disk and refuses to declare a required artifact that is absent:

```
REQUIRED_PRESENT_MATCH     P400_400x50_trajectory.mat
REQUIRED_PRESENT_MATCH     F400_400x50_trajectory.mat
2 required (2 present/match, 0 missing, 0 mismatch)   RESULT: PASS
```

The original declaration is preserved verbatim at
`evidence/rerun_20260909/move_activity_400_EVIDENCE.ORIG.json` (SHA-256
`7c224d3aa9287dfd7c13b98f6b31dd2e63ff9f548726e7c5f067a2b0f58e514b`).

**The frozen rule was then re-verified against the restored arm by an independent
implementation.** Preregistration §2.6 recorded specific values obtained by running
the frozen MATLAB detector `tb_branches.m` on the *original* F400. Since that file had
to be recomputed, re-running the same code would only have shown the recomputation was
faithful. `scripts/cv_frozen_rule_check.py` instead re-derives the same quantities
from the frozen **definitions** in §2.1–2.4 in Python, so agreement is between two
independent implementations:

| quantity | frozen (MATLAB `tb_branches`) | re-derived (Python, from the definitions) |
|---|---|---|
| `tol` | 0.125 | 0.125 |
| Branch A ever true | never | never |
| first `B(k)` true | 369 | **369** |
| `med₂₀cosθ(369)` | 0.9937451892796663 | 0.9937451892796668 |
| `med₂₀net_path(369)` | 0.9728297014945854 | 0.972829701494584 |
| `amp(369)` | 0.1241004791601378 | 0.12410047916013768 |
| `M_nd(369)` | 16.158892933214315 | 16.158892933214318 |
| `ω₁(369)` | 166.3649498603138 | 166.3649498603138 |

Agreement to ~1e-15 on every quantity; residuals are floating-point summation order.
`ALL FROZEN VALUES REPRODUCED: True`
(`evidence/frozen_rule_recheck.json`). This is a *stronger* recovery verification than
the preregistration was able to perform, and it closes
`CONTROLLER_DEFINITION_RECOVERY_PASS` on evidence that exists on this machine rather
than on a lost file.

With F400 present again, the controller software gate returns to **17 of 17, nFail = 0**
(`evidence/software_tests_rerun_20260909_after_F400_restore.json`). Three test records
are retained deliberately, because together they document the evidence loss rather
than hide it:

| record | when | result |
|---|---|---|
| `software_tests.json` | 2026-09-08T19:23:06Z, **before any scientific run** | 17/17, nFail 0 — this is the record that satisfies Phase 7 |
| `software_tests_rerun_20260909.json` | at resumption, F400 missing | 16/17 — item 16 reports `F400:MISSING` |
| `software_tests_rerun_20260909_after_F400_restore.json` | after the recomputation | 17/17, nFail 0 |
