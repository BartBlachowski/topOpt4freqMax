# OLHOFF_CURRENT_PROMOTION_REPORT

Establishing `analysis/OlhoffCurrent` as the sole production Du–Olhoff
implementation in `topOpt4freqMax`.

| | |
|---|---|
| Date | 2026-09-07 |
| Repository | `/Users/piotrek/Programming/topOpt4freqMax`, branch `benchmark-methodology-r2` |
| HEAD before | `9cd1c8eed109e1c1b02c67fc6d9bb81f87179d6a` |
| MATLAB | R2025b **Update 1** (25.2.0.3042426) |
| Supporting documents | [`OLHOFF_SOURCE_LINEAGE_AUDIT.md`](OLHOFF_SOURCE_LINEAGE_AUDIT.md), [`OLHOFF_IMPLEMENTATION_MAP.md`](OLHOFF_IMPLEMENTATION_MAP.md), [`OlhoffCurrent/PROVENANCE.md`](OlhoffCurrent/PROVENANCE.md), [`OlhoffCurrent/README.md`](OlhoffCurrent/README.md) |

---

## The seventeen questions

### 1. What was the canonical source before promotion?

**There were two production chains, executing two different solvers.** That is
the ambiguity this task was called to end:

| Driver | Executed |
|---|---|
| `examples/Performance/performance_comparison.m` → `confbench_run_case` → `olhoffm4_run` | `analysis/OlhoffM4Reconstruction/+frozen/` |
| `examples/Performance/olhoff_preflight.m` + `run_topopt_from_json` key `OlhoffDu2007Repro` | `Matlab/reproduction2007/` |

The conference numbers came from the first. The second is a different solver
whose own preflight header describes it as the Olhoff column.

### 2. Which exact upstream commit was promoted?

```
repository  /Users/piotrek/Programming/Matlab/Olhoff
branch      architecture/canonical-config
commit      695f03bdac20c423a4e1d389cf9db9187597bcc3
date        2026-09-06 23:24:25 +0200
subject     Phase 24: final report -- 12/12 anchors bitwise, verdict VERIFIED
dirty       clean (git status --porcelain empty)
```

Unambiguous: one head, clean tree, and the commit carrying the acceptance
verdict `OLHOFF_ARCHITECTURE_REFACTOR_VERIFIED`.

**Verified, not trusted.** `anchorReport()` recomputed all twelve regression
anchors — **0 of 12 failed**. Anchor `A1_frozen160` (the production-relevant
one) was additionally **re-executed live from source**, digest recomputed in
memory, nothing written into the upstream repository:

```
LIVE  A1_frozen160  outer=91 inner=2241 CONVERGED omega1=169.495227021538
science  1afe4b7e0cf86a860482e34adf08ad20af47d070494d2147e0eedfdf1cca9001
ref      1afe4b7e0cf86a860482e34adf08ad20af47d070494d2147e0eedfdf1cca9001
VERDICT  science=IDENTICAL  logShape=IDENTICAL
```

The semantic configuration architecture is real and present: `olh.config`
(80-field schema, validation, resolve/assign, legacy adapters, `describe`),
`olh.presets` (ten named presets), and `olhoffSolve.m`, which branches on **no**
experiment identifier. `git diff --stat 2029baa 695f03b` confirms exactly three
pre-existing files changed, as the refactor report claims.

### 3. What is now the sole production implementation?

**`analysis/OlhoffCurrent`.** 74 promoted source files under `+impl/`, tree
SHA-256 `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c`.

Entry point `olhoffcurrent_run(nelx, nely)`, which installs the fail-closed path
gate before it solves.

### 4. Which historical/external trees remain?

All of them. Nothing was moved, deleted or reorganized. Fourteen trees are
classified in `OLHOFF_IMPLEMENTATION_MAP.md`, with exactly one `PRODUCTION`
entry.

### 5. Can any production script execute them?

**No.** `olhoffcurrent_forbidden_paths` blocks every repository-relative tree
*and* the external repository by absolute path. The gate additionally refuses
any **undeclared** second candidate for an owned symbol, so a tree nobody has
declared still fails closed.

### 6. Were 160×20 and 320×40 behaviour preserved?

**Yes, at both meshes.**

| | 160×20 | 320×40 |
|---|---|---|
| outer iterations | 91 / 91 | 131 / 131 |
| inner iterations | 2241 / 2241 | 2614 / 2614 |
| status | CONVERGED / CONVERGED | CONVERGED / CONVERGED |
| ω₁ | 169.49522702153845 | 165.95078925220545 |
| volume | 0.49999900877797954 | 0.49999913861781514 |

(`frozen M4` / `OlhoffCurrent`, identical in every column.)

### 7. Were they bitwise identical?

**Yes.**

```
--- 160x20 ---
  shared content (28 fields): BITWISE IDENTICAL
  canonical-only recorders: 8 of 8 proved inert for all 91 iterations
  vs SAVED conference record: rho=BITWISE omega1=BITWISE vol=BITWISE
                              outer=BITWISE inner=BITWISE status=BITWISE
--- 320x40 ---
  shared content (28 fields): BITWISE IDENTICAL
  canonical-only recorders: 8 of 8 proved inert for all 131 iterations
  vs SAVED conference record: rho=BITWISE omega1=BITWISE vol=BITWISE
                              outer=BITWISE inner=BITWISE status=BITWISE

VERDICT: PROMOTION_EQUIVALENCE_PASS
```

Compared bitwise: design variables, physical density, design variable `z`,
eigenfrequencies, eigenvalues, volume, **every non-timing history field**
(`omega, N, beta, nInner, dxOuter, vol, degen, multJ, innerConv, cumInner,
dxNorm2, move, gap12, volErr, dBeta, stage`), outer and inner counts, status,
log length and log shape. Timing fields excluded, and only timing fields.

Two comparisons were run per mesh, independently: against a **freshly executed**
frozen-M4 run, and against the **saved nine-mesh conference campaign record**
(`campaign_9mesh_r2/benchmark_records.mat`) — existing evidence reused rather
than regenerated. Both bitwise.

**No tolerance was invoked anywhere.** Two apparent discrepancies were run to
ground rather than waived:

* **`hist.dBeta` reported "differs".** Cause: `isequal(NaN,NaN)` is false, and
  `dBeta(1)` is legitimately NaN in both records. `isequaln` returns true.
  A defect **in the comparator**, fixed there; the data were always identical.
* **Eight history fields exist only in OlhoffCurrent** — `pPen, pStage, pEvent,
  massLow, projBeta, projStage, projEvent, dxPhys2`. They are recorders for the
  p-continuation, mass-continuation and projection controllers, which the
  pre-canonical frozen solver does not have at all. They cannot be compared
  against something that does not exist, so each is instead required to sit at
  its **inert** value for every iteration: `pPen ≡ 3`, `pStage ≡ 1`,
  `pEvent ≡ 0`, `massLow ≡ 0`, `projBeta ≡ 0`, `projStage ≡ 1`, `projEvent ≡ 0`,
  `dxPhys2` all-NaN. All eight pass at both meshes. **That is positive evidence
  the controllers were disabled, not a waiver.**

### 8. Which production scripts were repointed?

| Production script | Previous Olhoff source | New source | Path gate | Verified |
|---|---|---|---|---|
| `examples/Performance/performance_comparison.m` | `analysis/OlhoffM4Reconstruction` (addpath + `olhoffm4_scrub_forbidden_paths`) | `analysis/OlhoffCurrent` (addpath + `olhoffcurrent_scrub_forbidden_paths`) | yes | preflight 34/34 under a genpath-contaminated session |
| `conference_bench/confbench_method_config.m` | `olhoffm4_config` | `olhoffcurrent_paths` + `olhoffcurrent_config` + `olhoffcurrent_preset` | yes | profile id stable across meshes |
| `conference_bench/confbench_run_case.m` | `olhoffm4_run` | `olhoffcurrent_run` | yes | equivalence at 160×20, 320×40 |
| `conference_bench/confbench_preflight.m` | `olhoffm4_verify_import`, `olhoffm4_forbidden_paths`, `which()` | source manifest + currentness + `olhoffcurrent_assert_dispatch`, `which -all` | yes | 34/34 pass |
| `conference_bench/confbench_manifest.m` | M4 import hashes | `olhoff_implementation` provenance block | n/a | manifest fields populated |
| `conference_bench/confbench_caveats.m` | `olhoffm4_caveat` | `olhoffcurrent_caveat` | n/a | resolves |
| `conference_bench/confbench_timing_schema.m` | `olhoffm4_caveat` | `olhoffcurrent_caveat` | n/a | resolves |
| `conference_bench/confbench_selftest.m` | M4 gate T2/T6 | OlhoffCurrent gate T2/T6 | yes | resolves |
| `confbench_frozen_budget.m`, `confbench_display_name.m`, `confbench_topology_images.m` | doc references to `olhoffm4_*` | updated text | n/a | comment-only |

No scientific parameter, benchmark methodology, stopping rule, filter or timing
boundary was changed. The timed region remains the solver call alone: the path
gate and configuration resolve both happen **outside** `tCall`, exactly as
before.

### 9. Does path contamination fail closed?

**Yes** — `analysis/OlhoffCurrent/tests/test_path_isolation.m`, **6 of 6 pass**:

```
[PASS] A  clean path + OlhoffCurrent                            -> PASS
[PASS] B  + analysis/OlhoffM4Reconstruction                     -> BLOCK
[PASS] C  + external /Matlab/Olhoff                             -> BLOCK
[PASS] D  + Matlab/reproduction2007                             -> BLOCK
[PASS] E  helper shadowing only, olhoffSolve still ours         -> BLOCK
[PASS] F  a declared collision that WINS the resolution         -> BLOCK
```

Every BLOCK case asserts the gate raised
`olhoffcurrent_assert_dispatch:PathContaminated` and that **no optimization
started**.

### 10. Does helper-function shadowing fail closed?

**Yes — this is TEST E, and it is the one that matters.** With
`Matlab/reproduction2007/{algo,fem}` prepended, `olhoffSolve` still resolves to
OlhoffCurrent (it exists nowhere else) while `innerLoop` resolves to the
competing tree. The test verifies that premise explicitly before asserting the
refusal:

```
premise: olhoffSolve is ours = 1, innerLoop shadowed = 1, gate refuses = 1
```

The gate checks **all 32 owned symbols** — derived from the directory, not a
hand-written list — with `which(name, '-all')`, and treats a second candidate as
a blocker in its own right. The gate this replaces used `which(name)`, the first
hit only, which proves *"the winner is right"* but never *"there is only one
candidate"*.

**A real latent hazard this caught.** `tools/Matlab/mmasub.m` is byte-identical
to the **`asfound`** MMA variant (`54c1680036e6…`), **not** the `published` copy
the production preset requires (`4507b73e3e44…`); the two differ in default
`move` and `asyinit`, so resolving to it would change the nested
sub-optimization silently. `tools/Matlab` is on the production path by design —
the Proposed and Yuksel methods need it — so it cannot simply be removed. It is
now **declared** in `olhoffcurrent_known_collisions.m` with its hash and reason.
Declaring permits it to *exist*, never to *win*: if it ever becomes the
resolution, that is a hard blocker (TEST F). Its presence is recorded as a
warning with its current hash in the run manifest, so a change to shared tooling
is visible in the artifact.

Class methods (`@cls/f.m`) and package members (`+pkg/f.m`) are skipped, because
they cannot take part in bare-name resolution — this is what stops MATLAB's own
`@cvdata/applyFilter.p` from being a false blocker.

### 11. Is the external repo absent from production execution?

**Yes.** No production script references it as a path; it is blocked by absolute
path in `olhoffcurrent_forbidden_paths`; TEST C proves the refusal. Its state is
recorded in the preflight as a **note, never a gate**.

### 12. Is OlhoffM4Reconstruction absent from production execution?

**Yes.** It is on the forbidden list, TEST B proves the refusal, and no
production `.m` file references `olhoffm4_*` any more (verified by grep). It
remains untouched on disk as frozen evidence, still hash-pinned by its own
`IMPORT_MANIFEST.json`.

### 13. Were any scientific settings changed?

**No.** The production preset delegates to the promoted upstream
`olh.presets.duOlhoffFrozenM4`, so there is exactly one definition of the
mathematics and no opportunity for drift. Bitwise equivalence at two meshes is
the evidence.

**One adaptation was necessary, and only one.** `+impl/architecture/olhoffSolve.m`
gains `hist.tOuter` — a `tic` at the top of the outer loop and a `toc` recorded
after the convergence test and every guard. The conference benchmark's nested
cost accounting requires the wall time of a complete outer iteration; the frozen
reconstruction carries exactly this instrumentation
(`OlhoffM4Reconstruction/patches/olhoffOpt.timing-instrumentation.diff`) and
upstream's `olhoffSolve.m` does not. Repointing without it would silently break
the accounting.

`hist.tOuter` is written and **never read back**. That is proved, not asserted:
anchor `A1_frozen160` executed against **uninstrumented** upstream code gives
`omega1 = 169.495227021538`, `outer = 91`, `inner = 2241`; the **instrumented**
promoted code gives the same numbers and reproduces the frozen conference design
vector bitwise. The instrumentation is demonstrably inert **across an
instrumented/uninstrumented boundary**.

Upstream hash `9f80dd8ed504dc5cce5d8ab995cd4520e1e388ec6478f6446378cb8a464e2543`
→ promoted `5d4abd37c8b186a42d1a7ef5b7066c8bc2429d33ee91f20b0b844bace772324b`.
**Every other one of the 74 promoted files is byte-identical to upstream.**

### 14. Was the nine-mesh campaign rerun?

**No.** Four solves were run in total: frozen-M4 and OlhoffCurrent at 160×20 and
320×40 — the two meshes §10 requires. The 240×30 and the six meshes from 400×50
to 800×100 were not touched, and the saved campaign record was **read**, never
regenerated.

### 15. What is the exact production preset?

```
duOlhoffFixedPenaltySensitivityFiltered
```

* **FixedPenalty** — SIMP `p = 3` held constant, no `p` continuation. (§2.1 of
  the paper says `p` is "normally assigned values increasing from 1 to 3";
  fixing it is a reconstruction ruling, made because the reported initial
  eigenfrequencies fit `p = 3` and not `p = 1`.)
* **SensitivityFiltered** — Sigmund (1997) sensitivity filter applied to every
  `f_sk`, fixed physical radius `R = 0.06·b`; no density filter, no projection.

Also: mass model eq. (4b); fixed multiplicity subspace of size 2 with the (25d)
diagonal offsets retained; published Svanberg MMA on the increment,
`tolInner = 0.05`; move ladder `[0.04 0.02 0.01 0.005]` on the bound-variable
stall signal; `‖Δρ‖₂ < 0.05·√(NE/3200)` with the settled-move guard;
`maxOuter = 400`; single-threaded.

`olh.config.describe(olhoffcurrent_config(nelx,nely))` prints the whole
formulation in scientific terms with the provenance class (A/B/C/D) of every
choice, readable with no knowledge of this project's audit history.

**Historical codes are provenance aliases, not API.** M4, TMA, B0, REG160, S2,
R1, R2, P1, PD1, PM1, T800 and upstream's `duOlhoffFrozenM4` are recorded so old
evidence matches new runs. No production script uses them; a test asserts the
production preset name is not one of them.

### 16. How is "OlhoffCurrent is current" checked?

`olhoffcurrent_currentness()` reports exactly one of `CURRENT`,
`LOCAL_MODIFIED`, `UPSTREAM_AHEAD`, `PROVENANCE_MISMATCH`,
`UPSTREAM_UNREACHABLE`. It **never updates anything**. Present state:

```
state           : CURRENT
local integrity : PASS (74 files, tree c1455374d5f8e256)
promoted commit : 695f03bdac20c423a4e1d389cf9db9187597bcc3
upstream branch : architecture/canonical-config
commits ahead   : 0
```

**`UPSTREAM_AHEAD` is not obsolescence.** Upstream is a development tree;
experimental commits land there constantly and most will never be promoted.
Production currentness changes on exactly one event: a human explicitly
**accepts** an upstream state and promotes it. The preflight treats
`LOCAL_MODIFIED` and `PROVENANCE_MISMATCH` as blockers and `UPSTREAM_AHEAD` as
information.

`test_currentness` proves the state model is real rather than decorative: it
corrupts a **copy** of the manifest, requires the state to flip to
`LOCAL_MODIFIED`, restores it, and requires it to flip back. 8 of 8 checks pass.

### 17. What historical cleanup remains deliberately deferred?

Per §15 of the brief, none of this was attempted:

* moving `OlhoffApproach*`, `OlhoffRegularized`, `OlhoffReproduced2007`;
* consolidating the audit trees or creating an archive;
* deleting any duplicate implementation;
* rewriting historical references;
* creating or populating `analysis/OlhoffExperiments`;
* retiring the second production chain (`run_topopt_from_json` key
  `OlhoffDu2007Repro` → `Matlab/reproduction2007`), which remains available for
  historical work and is blocked from the Olhoff production column.

**Technical debt discovered and recorded, not fixed:**

1. Four executable historical Olhoff trees hold **959 files on disk and zero in
   git** — untracked, not `.gitignore`d. Their contents are unversioned, and no
   commit pins what they held when the results citing them were produced.
2. `analysis/OLHOFF_IMPLEMENTATION_STATUS.md` still names OlhoffM4Reconstruction
   "conference-active". It is accurate about the campaign that produced the
   frozen results and is referenced by the frozen audit trail, so it was left
   unedited; `OLHOFF_IMPLEMENTATION_MAP.md` supersedes it for "what may
   production execute?".
3. Six scripts under `examples/Revision_v1/` still call
   `addpath(genpath(analysis))`. They are **not** Olhoff production scripts, and
   rewriting their path setup risks unrelated breakage, so they were left alone
   — the deliberate reading of "do not unnecessarily alter unrelated MATLAB path
   behavior". The production driver defends against them: it scrubs what the
   session inherited (**164 entries removed** in the audit run) and its gate
   fails closed if the scrub missed anything.

---

## Tree | Role | Production executable? | Mutable? | Notes

| Tree | Role | Production executable? | Mutable? | Notes |
|---|---|---|---|---|
| `analysis/OlhoffCurrent` | **PRODUCTION** | **YES — only this** | via promotion only | 74 files, tree `c1455374…`; gate `olhoffcurrent_paths` |
| `/Users/piotrek/Programming/Matlab/Olhoff` | DEVELOPMENT_UPSTREAM | no (blocked, absolute path) | yes, freely | `architecture/canonical-config` @ `695f03b`, clean |
| `analysis/OlhoffM4Reconstruction` | FROZEN_EVIDENCE | no (blocked) | **no** — hash-pinned | production *before* this promotion |
| `Matlab/reproduction2007` | HISTORICAL | no (blocked) | no | 49 bare names shared with upstream |
| `analysis/OlhoffApproach` | HISTORICAL | no (blocked) | no | untracked, 88 files |
| `analysis/OlhoffApproachExact` | HISTORICAL | no (blocked) | no | untracked, 836 files |
| `analysis/OlhoffRegularized` | HISTORICAL | no (blocked) | no | untracked, 35 files |
| `analysis/OlhoffReproduced2007` | HISTORICAL | no (blocked) | no | untracked, 5 files |
| `analysis/OlhoffApproachExactOpus` | AUDIT_ONLY | no (blocked) | no | no `.m`/`.py` on disk |
| `analysis/olhoff_stabilization_audit` | HISTORICAL | no (blocked) | no | superseded S1 profile |
| `analysis/olhoff_native_convergence` | AUDIT_ONLY | no (blocked) | no | telemetry variants |
| `analysis/olhoff_fixed_budget_audit` | AUDIT_ONLY | no (blocked) | no | |
| `analysis/olhoff_practical_convergence_audit` | AUDIT_ONLY | no (blocked) | no | |
| `analysis/olhoff_nested_mma_route_audit` | AUDIT_ONLY | no (blocked) | no | no `.m` |
| `analysis/OlhoffExperiments` | EXPERIMENTAL | no (blocked pre-emptively) | n/a | does not exist yet |

## Static audit after repointing (§18)

| Check | Result |
|---|---|
| production references to old Olhoff trees | none, except the deliberate blacklist entry and the negative test's contaminating path |
| absolute references to `/Matlab/Olhoff` in production | one — the blacklist entry that forbids it |
| broad `genpath` in production `.m` | none executable |
| direct calls to historical Olhoff entry points | none |
| `which -all` under production path setup | all 16 probed symbols resolve into `+impl`; 3 have a second candidate — 2 declared (`tools/Matlab`), 1 a MATLAB class method |
| `checkcode` on all 33 changed/new files | **0 new messages.** The 5 remaining are pre-existing at HEAD (verified by running `checkcode` on the HEAD versions: same count, same class) |

## Evidence index

| Artifact | What it shows |
|---|---|
| `OlhoffCurrent/SOURCE_MANIFEST.json` | integrity manifest, 74 files, tree hash |
| `OlhoffCurrent/PROVENANCE.json` / `.md` | source repo, branch, commit, dirty state, the one adaptation, alias policy |
| `OlhoffCurrent/tests/test_path_isolation.m` | TEST A–F, 6/6 |
| `OlhoffCurrent/tests/test_currentness.m` | 8/8, including a proved-reachable `LOCAL_MODIFIED` |
| `OlhoffCurrent/tests/test_preset_equivalence.m` | repeatable regression against the saved conference record |
| `OLHOFF_SOURCE_LINEAGE_AUDIT.md` | the read-only §1 audit, incl. the 49-symbol collision matrix |
| `OLHOFF_IMPLEMENTATION_MAP.md` | all 14 trees classified; exactly one PRODUCTION |
