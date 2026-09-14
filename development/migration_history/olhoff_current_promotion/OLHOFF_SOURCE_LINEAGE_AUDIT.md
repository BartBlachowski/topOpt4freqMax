# OLHOFF_SOURCE_LINEAGE_AUDIT

Read-only provenance audit of every Du–Olhoff implementation reachable from
this machine, performed **before** any promotion, as required by §1 of the
"establish one production Olhoff" brief.

Nothing was modified while this inventory was taken.

| | |
|---|---|
| Audit date | 2026-09-07 |
| Main repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| HEAD at audit time | `9cd1c8eed109e1c1b02c67fc6d9bb81f87179d6a` |
| Working tree | clean except four **untracked** Olhoff directories (§4) |
| MATLAB | R2025b **Update 1** (25.2.0.3042426) |

> **MATLAB version note.** The external refactor report was written against
> R2025b 25.2.0.**2998904**. This machine now runs Update 1
> (25.2.0.**3042426**). Every verification below was re-run under Update 1, and
> every equivalence claim made in the promotion is between two implementations
> executed under **the same** Update 1 — so the release change is controlled,
> not assumed away.

---

## 1. The two production dispatch chains found

There is not one production Olhoff path today. There are two, and they execute
**different implementations**:

| Driver | Chain | Implementation actually executed |
|---|---|---|
| `examples/Performance/performance_comparison.m` (current conference driver) | → `confbench_run_case` → `runOlhoff` → `olhoffm4_run` | `analysis/OlhoffM4Reconstruction/+frozen/` |
| `examples/Performance/olhoff_preflight.m` + `run_topopt_from_json` approach key `OlhoffDu2007Repro` | → `repro2007_paths` → `run_repro2007` → `olhoffOpt` | `Matlab/reproduction2007/` |

The second chain is the one `olhoff_preflight.m`'s own header describes, and it
is a *different* solver from the one the conference numbers came from. This is
precisely the ambiguity the brief exists to end.

---

## 2. External repository — `/Users/piotrek/Programming/Matlab/Olhoff`

| | |
|---|---|
| Version control | **git**, initialized during the post-conference refactor |
| Branch | `architecture/canonical-config` |
| HEAD | `695f03bdac20c423a4e1d389cf9db9187597bcc3` |
| HEAD date | 2026-09-06 23:24:25 +0200 |
| HEAD subject | `Phase 24: final report -- 12/12 anchors bitwise, verdict VERIFIED` |
| Dirty state | **clean** (`git status --porcelain` empty) |
| Tags | `post-conference-baseline` → `2029baa` |
| Tracked files | 1531 |
| Role | **DEVELOPMENT / RESEARCH UPSTREAM** |

Commit graph (linear, 7 commits):

```
695f03b  Phase 24: final report -- 12/12 anchors bitwise, verdict VERIFIED   <- HEAD
298edd4  Phase 13/20: continuation tests, preset trajectory equivalence, README
159d60c  Phase 16/17/19/21: describe(), clean static analysis, file classification
11e0483  Phase 9 steps 4-11: canonical solver; experiment IDs out of the mathematics
e7ed04b  Phase 9 steps 1-3: canonical config schema, validation, adapters, presets
b19f08f  Phases 0-8: provenance, variant inventory, provenance classification, plan
2029baa  Baseline: post-conference state of the Du-Olhoff eigenfrequency solver  (tag)
```

There is **no ambiguity about which commit is the accepted refactored state**:
the branch has a single head, it is clean, and `695f03b` is the commit whose
message and report both carry the acceptance verdict
`OLHOFF_ARCHITECTURE_REFACTOR_VERIFIED`.

### 2.1 What the refactor actually did (verified, not assumed)

`git diff --stat 2029baa 695f03b -- algo/ fem/ setpaths.m` confirms the report's
central claim that exactly **three** pre-existing files changed:

```
 algo/olhoffOpt.m | 542 +++++--------------------------------------------------
 fem/massScale.m  |  54 +++---
 setpaths.m       |   4 +
 3 files changed, 83 insertions(+), 517 deletions(-)
```

The semantic configuration architecture is present and is real:

* `architecture/+olh/+config/` — `schema`, `defaults`, `validate`, `resolve`,
  `assign`, `fromLegacy`, `toLegacy`, `describe`, `epsilonForMesh`;
* `architecture/+olh/+presets/` — ten named presets;
* `architecture/olhoffSolve.m` — the canonical solver, which branches on **no**
  experiment identifier;
* `algo/olhoffOpt.m` — reduced to a compatibility shim containing no mathematics.

Historical experiment codes survive only as provenance: `duOlhoffFrozenM4.m`
documents its own historical labels ("TMA, B0, REG160, frozen M4") in its help
text, and the solver's `R1`/`R2` spellings survive only inside a log-formatting
helper so old and new logs stay comparable.

### 2.2 Behavioural anchors — verified, not trusted

§2 of the brief requires verification rather than trust. Two independent checks
were run, both read-only:

**(a) Digest recomputation over all twelve stored anchors** — `anchorReport()`
recomputes both digests from the stored reference and candidate records:

```
anchors failing the equality standard: 0 of 12
```

All twelve report `science = BITWISE IDENTICAL` and `logShape = identical`.

**(b) Live re-execution of the production-relevant anchor** under this
machine's MATLAB. `A1_frozen160` is the frozen conference realization — the one
that matters for promotion. It was re-executed from source and its digest
recomputed **in memory**, comparing against the stored reference without
writing anything into the external repository:

```
LIVE  A1_frozen160  outer=  91 inner=  2241 status=CONVERGED  omega1=169.495227021538
LIVE  science  = 1afe4b7e0cf86a860482e34adf08ad20af47d070494d2147e0eedfdf1cca9001
REF   science  = 1afe4b7e0cf86a860482e34adf08ad20af47d070494d2147e0eedfdf1cca9001
VERDICT A1_frozen160 : science=IDENTICAL logShape=IDENTICAL
```

The reported anchor behaviour therefore reproduces from source under R2025b
Update 1, not merely from a stored record.

**Conclusion for §2: the external tree at `695f03b` IS the latest refactored
canonical implementation, and it is a recoverable committed state.**

---

## 3. Lineage: frozen conference core vs. external tree

The frozen conference solver (`analysis/OlhoffM4Reconstruction/+frozen/`, 23
files) was imported from the external repository on 2026-09-04, *before* the
external tree was placed under git. Re-hashing all 23 files against the
external tree's current state:

| Result | Count | Files |
|---|---|---|
| **bit-identical** | 20 | all of `fem/` except `massScale.m`, all of `filter/`, all of `mma/` + `mma_published/`, and `algo/{defaultCfg, deltaLambda, genGrad, innerLoopLP, innerLoopRho, moveControl, multRule, useMMA}` |
| differs | 3 | `algo/olhoffOpt.m`, `fem/massScale.m`, `algo/innerLoop.m` |
| not in external tree | 1 | `run_pinned_pinned.m` (an M4-local runner) |

Two of the three differences are the *declared* refactor changes
(`olhoffOpt.m` → shim, `massScale.m` → dispatcher). `algo/olhoffOpt.m` differs
for a second, independent reason as well: the M4 import applied a declared
timing-instrumentation patch (`patches/olhoffOpt.timing-instrumentation.diff`).

### 3.1 The undeclared difference: `algo/innerLoop.m`

`innerLoop.m` was **not** listed among the three files the refactor changed, and
`git log -- algo/innerLoop.m` shows it has only ever had one commit (the
baseline `2029baa`). So it changed in the external working tree **between the
M4 import (2026-09-04 13:43) and the git baseline**, while the tree was still
unversioned.

| | sha256 (first 12) |
|---|---|
| M4 `IMPORT_MANIFEST.json` recorded, and the frozen copy on disk | `9ec33d7dcbab` |
| external tree at `2029baa` and at `695f03b` | `0f9f7fbcc89d` |

The diff is **purely additive and dormant**: it introduces an optional
`ctx.volFun` handle supplied *only* by the projection path of `olhoffOpt`, and
guards it with `useVolFun = isfield(ctx,'volFun') && ~isempty(ctx.volFun)`. When
absent, the original linear volume-constraint expression executes verbatim. The
production preset does not enable projection, so the branch is unreachable
there.

**This is an argument, not a proof.** It is exactly the kind of divergence that
the §10 equivalence gate exists to catch, and it is why that gate is run at two
meshes against the frozen conference implementation rather than waived.

---

## 4. Inventory of every Olhoff tree in the main repository

Git state is per-tree; the repository as a whole is clean apart from the
untracked directories noted.

| Tree | Tracked / on disk | Git state | Purpose | Class |
|---|---|---|---|---|
| `analysis/OlhoffM4Reconstruction` | 45 / 45 | tracked, clean | Frozen conference reconstruction (M4), imported from the external repo; core under `+frozen/` | FROZEN_EVIDENCE (production **today**) |
| `Matlab/reproduction2007` | 74 / 74 | tracked, clean | Clean-room Du–Olhoff 2007 reproduction (Eq. 22 LP + paper-literal MMA); dispatched by `run_topopt_from_json` key `OlhoffDu2007Repro` | HISTORICAL, executable |
| `analysis/OlhoffApproach` | **0 / 88** | **untracked** | Original bound-formulation MMA implementation + Python port | HISTORICAL, executable |
| `analysis/OlhoffApproachExact` | **0 / 836** | **untracked** | "Exact Olhoff 2014" line: own FE, multiplicity, generalized gradients | HISTORICAL, executable |
| `analysis/OlhoffRegularized` | **0 / 35** | **untracked** | Globalized variant built on `reproduction2007` primitives | HISTORICAL, executable |
| `analysis/OlhoffReproduced2007` | **0 / 5** | **untracked** | Thin runner exposing `reproduction2007` on Yuksel geometries | HISTORICAL, executable |
| `analysis/OlhoffApproachExactOpus` | 0 / 160 | ignored by content (only `.png`/`.csv`/`.mat`/`.pyc`) | Clean-room re-derivation; **no `.m` or `.py` remains on disk** | AUDIT_ONLY (not executable) |
| `analysis/olhoff_stabilization_audit` | tracked | clean | Superseded stabilization profile + `olhoffOptStabilized.m` | HISTORICAL, executable |
| `analysis/olhoff_native_convergence` | tracked | clean | `olhoffOptTelemetry.m`, `nativeConvergenceDetector.m` | AUDIT_ONLY, executable |
| `analysis/olhoff_fixed_budget_audit` | tracked | clean | Fixed-budget audit runners | AUDIT_ONLY, executable |
| `analysis/olhoff_practical_convergence_audit` | tracked | clean | One offline audit runner | AUDIT_ONLY, executable |
| `analysis/olhoff_nested_mma_route_audit` | tracked | clean | Python + reports only, **no `.m`** | AUDIT_ONLY (not MATLAB-executable) |

### 4.1 Technical debt discovered

Four executable Olhoff trees — `OlhoffApproach`, `OlhoffApproachExact`,
`OlhoffRegularized`, `OlhoffReproduced2007` — hold **959 files on disk and zero
in git**. They are not `.gitignore`d; they are simply untracked. Their content
is therefore unversioned and unrecoverable if lost, and no commit pins what
they contained when the historical results that cite them were produced.

This is recorded as debt. Fixing it means either committing or deliberately
archiving them, which is historical reorganization and is **deferred by §15**.

---

## 5. Symbol-collision matrix

Bare (non-package, non-`private`) `.m` function names were enumerated across
all eleven trees plus the external repository: **320 distinct symbols, 49
duplicated across trees.**

Every one of the 49 collisions is between the **external repository** and
**`Matlab/reproduction2007`**, and they include the entire numerical core:

```
olhoffOpt   innerLoop   innerLoopLP  genGrad    deltaLambda  massScale
model2D     assemble2D  elemMats2D   eigSolve   classifyModes
prepFilter  applyFilter mmasub       subsolv    defaultCfg   useMMA(-)
setpaths    top88       top88_reference  ... and 29 more
```

MATLAB resolves these by path order. With both trees on the path, `which`
reports one file while `which -all` reports two, and a run can silently execute
a helper from the wrong solver even when the top-level entry point resolves
correctly. **This is the concrete mechanism the brief refuses to keep
tolerating.**

`analysis/OlhoffM4Reconstruction` does **not** participate in these collisions,
because its core lives under `+frozen/` and it exposes only `olhoffm4_*` names.
That design is sound and is the pattern the promotion adopts.

---

## 6. Production path exposure today

| Question | Finding |
|---|---|
| Does any production script `addpath(genpath('analysis'))`? | **Not the conference driver.** `performance_comparison.m` adds only four explicit directories. |
| Does anything else? | **Yes** — six scripts under `examples/Revision_v1/` call `addpath(genpath(fullfile(repoRoot,'analysis')))`. |
| Can that expose competing Olhoff code? | `genpath` skips `+frozen/`, so it cannot reach the M4 core. It *does* put `analysis/OlhoffApproach`, `OlhoffApproachExact`, `OlhoffRegularized`, `olhoff_stabilization_audit` and `olhoff_native_convergence` on the path, and those entries **persist for the rest of the MATLAB session**. |
| Is that already known? | Yes — `performance_comparison.m` calls `olhoffm4_scrub_forbidden_paths(repoRoot)` precisely to remove such inherited entries. |
| Is the external repo ever added by production? | **No.** No production script references `/Users/piotrek/Programming/Matlab/Olhoff` as a path. |

### 6.1 Gap in the existing gate

`confbench_preflight.m` §6 checks helper resolution with `which(name)` — the
**first** hit only. A shadowed second copy of `innerLoop` or `mmasub` further
down the path is invisible to that test whenever the first hit is correct. The
existing gate therefore proves *"the winner is right"*, not *"there is only one
candidate"*. Closing that gap with `which -all` is a requirement of §7 of this
brief.

---

## 7. Audit conclusions

1. The external repository at `695f03b` is a clean, committed, unambiguous
   state, and its canonical-configuration refactor is real and verified — both
   by digest recomputation over twelve anchors and by live re-execution of the
   production-relevant anchor. **§2 passes; promotion may proceed.**
2. The frozen conference core is 20/23 bit-identical to the external tree. The
   three differences are accounted for; one of them (`innerLoop.m`) is an
   **undeclared** dormant addition and must be settled by behavioural
   equivalence, not by reading the diff.
3. Two different production chains currently execute two different Olhoff
   solvers. Exactly one must survive as production.
4. 49 bare symbols collide between the external tree and `Matlab/reproduction2007`.
   Any promoted implementation that exposes bare names becomes a third
   competitor unless it is shielded the way `+frozen/` is.
5. The existing preflight cannot see helper shadowing. It must be strengthened.
6. 959 files across four executable Olhoff trees are untracked. Recorded as
   debt; reorganization is deferred by §15.
