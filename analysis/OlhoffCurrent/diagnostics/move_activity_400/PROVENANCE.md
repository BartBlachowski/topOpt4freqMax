# PROVENANCE — 400x50 third-mesh activity measurement

## Repository state

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| HEAD reported at task start | `7154d82` |
| Prior-study landing commit | `5059a11` — offline move-activity study |
| **Phase-A retention repair** | `4ef315b` — evidence gate, policy, tests |
| **Phase-C scientific starting point** | `4ef315b` (clean tree) |
| Preregistration SHA-256 | `706b8865075f97cb0d5824658fa6e562636d464dba2611580d6c10fdff7716d4` |
| Preregistration frozen at | 2026-09-07T21:09:40Z, **before** the first 400x50 run |

**Scientific-run commit:** the single commit that introduces this directory
(`git log --oneline -- analysis/OlhoffCurrent/diagnostics/move_activity_400`).
It is not named by hash here because a commit cannot contain its own hash; it is
the sole commit touching this path, so the reference is unambiguous.

Commit boundaries are as required by brief sec. B3, and are three-way separated:

| commit | contents |
|---|---|
| `5059a11` | the preceding offline study's deliverables — no infrastructure, no new runs |
| `4ef315b` | the evidence-retention repair — **no scientific results** |
| *(this directory's commit)* | the 400x50 measurement — **no infrastructure changes** |

The 138.6 MB of raw trajectory is **not** in any of them: it is untracked by
design and gated by `EVIDENCE.json` instead.

## Phase B1 — the mixed commit `7154d82`, inspected

Parent: `a1f2c6c` (the "current accepted HEAD" named in the previous task).

`7154d82` contains **143 files, all pure additions** (`git diff-tree` reports
`A` for every one; no modifications, no deletions):

| area | files | relation to this work |
|---|---:|---|
| `analysis/OlhoffCurrent/diagnostics/move_transition/**` | 37 | the diagnostic this line of work builds on |
| `analysis/OlhoffApproachExact/**` | 53 | unrelated historical |
| `analysis/OlhoffRegularized/**` | 35 | unrelated historical |
| `analysis/OlhoffApproach/**` | 13 | unrelated historical |
| `analysis/OlhoffReproduced2007/**` | 5 | unrelated historical |

**No canonical OlhoffCurrent executable source changed.**
`git diff-tree 7154d82 -- analysis/OlhoffCurrent/+impl/` is **empty**, and within
`analysis/OlhoffCurrent/` only `diagnostics/` was touched.

**Does the mixed commit affect scientific reproducibility? No.** All four
unrelated trees it adds — `OlhoffApproach`, `OlhoffApproachExact`,
`OlhoffRegularized`, `OlhoffReproduced2007` — are already named in
`olhoffcurrent_forbidden_paths.m`, so the path guard actively refuses to let
production execute from any of them, and `olhoffcurrent_assert_dispatch`
verifies every owned symbol resolves inside `+impl/`. The defect is **log
hygiene, not executable ambiguity**.

Per the brief, history was **not** rewritten and no historical directory was
deleted. The recommendation stands that a frozen diagnostic should land in a
commit of its own.

## Phase B2 — scientific starting point, verified

Measured on this machine at the Phase-C starting point:

| check | result |
|---|---|
| `olhoffcurrent_source_manifest('Verify',true)` | **ok = 1, 74/74 files** |
| `+impl` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` — unchanged |
| `olhoffcurrent_currentness` | **CURRENT** (`localOk = 1`) |
| `olhoffcurrent_assert_dispatch` | **PASS** |
| `which olhoffSolve` | `+impl/architecture/olhoffSolve.m` |
| `which mmasub` | `+impl/mma_published/mmasub.m` — **published MMA wins** |
| `which applyFilter` / `prepFilter` | `+impl/filter/…` — **sensitivity filter wins** |
| external/historical Olhoff solver on path | **none** (forbidden-paths guard + dispatch gate) |

Phase A added `olhoffcurrent_evidence_gate.m`,
`olhoffcurrent_evidence_declare.m`, `EVIDENCE_POLICY.md`, `evidence/` and
`tests/test_evidence_retention.m` — **all outside `+impl/`**, so the canonical
tree hash is untouched, as the table above confirms.

Resolved scientific configuration at 400x50 (production entry point):

| field | value |
|---|---|
| `material.stiffness.p` | 3 |
| `material.mass.model` / `.q` | `eq4b` / 1 |
| `filter.type` / `.radiusPhysical` | `sensitivity` / 0.06 |
| `projection.enabled` | `false` |
| `multiplicity.method` / `.subspaceSize` / `.offDiagonal` | `subspace` / 2 / `true` |
| `optimizer.inner.variant` | `published` |
| `move.policy` / `move.levels` | `ladder` / `[0.04 0.02 0.01 0.005]` |
| `stop.tolerance` / `.toleranceRule` | 0.125 / `meshScaled` |
| production config hash (400x50) | `044d50a496cb64d43ed4bf75a6a7976a6726cca6feaf65e0b911a57e4593969c` |
| preset | `duOlhoffFixedPenaltySensitivityFiltered` (upstream `duOlhoffFrozenM4`) |

Because `stop.toleranceRule = 'meshScaled'` makes the tolerance proportional to
`sqrt(NE)`, `epsRMS = tolerance/sqrt(NE) = 8.83883476483184e-4` is **identical at
160x20, 320x40 and 400x50**. The diagnostic activity thresholds are therefore
directly comparable across all three meshes with no rescaling.

## MATLAB build discrepancy, and the control that resolves it

This machine runs MATLAB **25.2.0.2998904 (R2025b)**. Every prior study recorded
**25.2.0.3042426 (R2025b Update 1)** — a *different, later* build. Combining a
400x50 result produced here with 160x20/320x40 numbers produced there would be
unsound unless the builds are shown to do identical arithmetic. This was not
noticed by any previous provenance check.

Preregistration sec. 10 therefore declared one control in advance: rerun the
production 160x20 baseline and compare against the committed
`move_stop/runs/baseline_160x20_iterations.csv`. Result:

| check | result |
|---|---|
| outer iterations | 91 vs 91 — **match** |
| move descents | `[79, 90]` vs `[79, 90]` — **match** |
| `omega1` max relative error | **4.136e-15** |
| `volume` | 9.993e-16 |
| `l2` | 3.946e-15 |
| `maxAbs` | 2.436e-15 |
| `move` | **0.000e+00** |
| `beta` | 3.700e-15 |

Every discrete fact matches exactly and every continuous field agrees to
round-off. Stated precisely: this is agreement **to the precision the archived
CSV carries** (`writetable`, ~15 significant digits), so it demonstrates
equivalence at the 1e-15 level, **not** literal bitwise identity — the archived
per-element state that would allow a bitwise claim is exactly what was lost. That
is sufficient to combine the three meshes, and the residual is recorded rather
than rounded away.

This control is a provenance check, not evidence recovery: the 160x20 CSV it
compares against already existed.
