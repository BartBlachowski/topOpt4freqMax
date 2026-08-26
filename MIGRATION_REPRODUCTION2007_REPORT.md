# Migration report — Du–Olhoff 2007 clean-room reproduction

**Date:** 2026-08-26
**Scope:** import the successful clean-room Du & Olhoff (2007) reproduction into
this repository and integrate it into the existing runner/benchmark ecosystem,
without disturbing the two historical implementations.
**Type:** migration / integration. No algorithm development, no new campaign.

---

## 1. Verdict

The migration is complete and passes its acceptance gate.

- **39 of 39** regression checks are **bit-identical** to the frozen
  source-repository results, across all three baselines, including the full
  1600-iteration histories and the final 7200-element density fields.
- All **61 imported files** are SHA256-identical to the source. Zero source
  files were modified.
- Path isolation is **asserted, not assumed**: 4 of 4 checks pass, including
  the case that motivated the requirement.
- All **five** benchmark approaches — Olhoff, Yuksel, ourApproach (Proposed),
  OlhoffExact and the new OlhoffDu2007Repro — still resolve and run through
  `run_topopt_from_json`.

Three independently identifiable and executable implementations now exist.

---

## 2. Repository state

| | starting | final |
|---|---|---|
| branch | `benchmark-methodology-r2` | `benchmark-methodology-r2` |
| HEAD | `cf290fc7f9daf9da27bc8224f9585a0e1657bff1` | `cf290fc…` (unchanged — **not committed, not pushed**) |
| working tree | **clean** — no modified, no untracked, no stashes | 2 modified, 1 new file, 1 new directory (75 files) |

No unrelated working-tree changes existed at WP0, so none could be disturbed.

### Final working tree

```
 M .gitignore
 M tools/Matlab/run_topopt_from_json.m
?? MIGRATION_REPRODUCTION2007_REPORT.md
?? Matlab/                                  (75 files, 2.5 MB)
```

---

## 3. Source-repository provenance

| field | value |
|---|---|
| source path | `/Users/piotrek/Programming/Matlab/Olhoff` |
| Git branch | **none** |
| Git HEAD | **none** |
| dirty/clean | **not applicable** |
| working-tree status | not under version control |
| implementation timestamps | 2026-08-25, 10:47–11:57 |
| result artifacts | 2026-08-25, through 15:52 |

`git rev-parse --is-inside-work-tree` in the source directory returns
`fatal: not a git repository`. **There is no upstream commit to cite.** The
SHA256 manifest is therefore the only provenance anchor for the imported bytes;
it is stored at `Matlab/reproduction2007/SOURCE_SHA256.txt`, was computed before
the copy, and was re-verified after it and again at the end of the migration.

### Paper and erratum

- Du & Olhoff (2007), *Topological design of freely vibrating continuum
  structures…*, **Struct Multidisc Optim 34:91–110**, DOI `10.1007/s00158-007-0101-y`
- **Publisher's Erratum, Struct Multidisc Optim 34:545**, DOI
  `10.1007/s00158-007-0167-6` — restores the missing `Δ` in eqs. (25d), (26f),
  (26g) and corrects the Fig. 2 caption symbols. **Mandatory**; the printed form
  is not a well-posed problem. Erratum text was extracted and transcribed into
  `PROVENANCE.md`.
- Second source for the corrected form: Olhoff & Du (2014), eq. (19d)/(20f,g).

The 2007 and 2014 PDFs in the source repo were verified byte-identical to
`references/Du2007_Topological.pdf` and `references/Olhoff2014_Structural.pdf`.
The erratum PDF was **not** imported (`.gitignore` excludes `*.pdf`); its
content is transcribed and its source path recorded.

---

## 4. Files imported / moved / modified

### Imported (61 files, all SHA256-verified byte-identical)

| source | destination | n |
|---|---|---|
| `algo/*.m` | `Matlab/reproduction2007/algo/` | 12 |
| `fem/*.m` | `…/fem/` | 6 |
| `filter/*.m` | `…/filter/` | 3 |
| `mma/*.m` | `…/mma/` | 2 |
| `runs/*.m` | `…/runs/` | 22 |
| `setpaths.m`, `top88.m` | `…/` | 2 |
| `NOTES.md`, `OLHOFFEXACT_FAILURE_POSTMORTEM.md` | `…/` | 2 |
| `CLAUDE.md` | `…/SOURCE_CLAUDE.md` *(renamed)* | 1 |
| `docs/figs/*.png` | `…/docs/figs/` | 5 |
| `results/{FIG4_definitive,FINAL_lp_240x30}.{mat,log}`, `results/lp240_rmin1.3.mat` | `…/baseline/` *(relocated)* | 5 |

Two **content-identical layout changes** were applied:

1. `CLAUDE.md` → `SOURCE_CLAUDE.md`, so this repository's Claude Code tooling
   does not read the source repository's instruction file as its own.
2. `results/*` → `baseline/`, separating frozen regression targets from the
   run-time output directory.

### Moved

**Nothing.** No existing repository file was moved or deleted.

### Modified (2 files)

| file | change |
|---|---|
| `tools/Matlab/run_topopt_from_json.m` | added the `olhoffdu2007repro` dispatch case (+ two aliases); extended the unknown-approach error message |
| `.gitignore` | four scoped negations so the frozen baselines, paper figures and SHA manifest are tracked despite the blanket `*.mat` / `*.log` / `*.png` / `*.txt` rules |

No existing benchmark setting was changed. `performance_comparison.json`,
`performance_comparison.m` and the frozen protocol ledger are untouched.

### Created (new integration code and documentation, 15 files)

Documentation (7): `Matlab/README.md`, `Matlab/legacy/README.md`,
`Matlab/full_coupling/README.md`, `Matlab/reproduction2007/PROVENANCE.md`,
`Matlab/reproduction2007/SOURCE_SHA256.txt`,
`Matlab/reproduction2007/results/README.md`, and this report.

Integration code (8), all under `Matlab/reproduction2007/runner/` and excluded
from the SHA manifest by construction:

| file | role |
|---|---|
| `run_repro2007.m` | the standardized runner (WP3) |
| `repro2007_config.m` | named, artifact-anchored configurations |
| `repro2007_history.m` | the WP4 history schema |
| `repro2007_paths.m` | isolated path environment with an `onCleanup` guard |
| `repro2007_assert_identity.m` | fail-closed implementation-identity assertion |
| `repro2007_root.m` | single source of truth for the implementation root |
| `repro2007_regression.m` | the WP5 acceptance gate |
| `repro2007_verify_isolation.m` | the WP6 path-safety verifier |

All eight pass MATLAB Code Analyzer with zero warnings.

### Deliberately not imported

The remaining ~80 exploratory artifacts in the source `results/` tree (≈ 84 MB
of sweep `.mat`/`.log`/`.png`). They are evidence for `NOTES.md`, not inputs to
the implementation, and remain at the source path.

---

## 5. Resulting directory tree

```
topOpt4freqMax/
├── Matlab/                              ← NEW
│   ├── README.md                        three-generation map (WP7)
│   ├── legacy/README.md                 pointer → analysis/OlhoffApproachExact/Matlab/legacy/
│   ├── full_coupling/README.md          pointer → analysis/OlhoffApproachExact/Matlab/
│   └── reproduction2007/                the imported clean-room implementation
│       ├── PROVENANCE.md                source, erratum, manifest, scope limits
│       ├── SOURCE_SHA256.txt            61-file manifest
│       ├── NOTES.md                     evidence log (imported verbatim)
│       ├── SOURCE_CLAUDE.md             source spec (imported verbatim)
│       ├── OLHOFFEXACT_FAILURE_POSTMORTEM.md   forensic 3-way comparison
│       ├── setpaths.m, top88.m
│       ├── algo/    12  olhoffOpt, innerLoopLP, innerLoop, genGrad, deltaLambda, …
│       ├── fem/      6  model2D, assemble2D, eigSolve, elemMats2D, massScale, …
│       ├── filter/   3  prepFilter, applyFilter, top88_reference
│       ├── mma/      2  mmasub, subsolv
│       ├── runs/    22  original reproduction scripts, runnable as-is
│       ├── docs/figs/ 5 digitized paper figures
│       ├── baseline/ 5  FROZEN regression targets
│       ├── results/     run-time output (git-ignored)
│       └── runner/   8  NEW integration layer
│
├── analysis/OlhoffApproachExact/Matlab/       ← UNCHANGED (generation 2)
│   └── legacy/                                ← UNCHANGED (generation 1)
├── analysis/YukselApproach/Matlab/            ← UNCHANGED
├── analysis/ourApproach/Matlab/               ← UNCHANGED
└── tools/Matlab/run_topopt_from_json.m        ← dispatch case added
```

---

## 6. Implementation mapping

```
historical nested MMA      -> analysis/OlhoffApproachExact/Matlab/legacy/
                              entry: run_simply_simply_exact.m, run_clamped_clamped_exact.m,
                                     run_clamped_simply_exact.m, verify_*.m
                              signpost: Matlab/legacy/README.md

full-coupling OlhoffExact  -> analysis/OlhoffApproachExact/Matlab/   (top level)
                              entry: topopt_freq_exact.m, run_olhoff_case.m,
                                     run_all_olhoff_2014.m
                              benchmark: optimization.approach = "OlhoffExact"
                              signpost: Matlab/full_coupling/README.md

Eq.(22) LP reproduction    -> Matlab/reproduction2007/
                              native  : algo/olhoffOpt.m + algo/innerLoopLP.m
                              runner  : runner/run_repro2007.m
                              benchmark: optimization.approach = "OlhoffDu2007Repro"
                              original: setpaths(); runs/*.m   (unchanged, still runnable)

Yuksel runner              -> analysis/YukselApproach/Matlab/
                              entry: top99neo_inertial_freq.m, top99neo_dynamic_freq.m,
                                     run_simply_supported.m, run_cantilever.m,
                                     run_fixed_pinned.m
                              benchmark: optimization.approach = "Yuksel"

Proposed runner            -> analysis/ourApproach/Matlab/topopt_freq.m
                              benchmark: optimization.approach = "ourApproach"
                              (labelled "ProposedApproach" in performance_comparison.m)
```

---

## 7. Runner interfaces

### Native, repository convention

```matlab
[x, omega, tIter, nIter, info] = run_repro2007(runCfg)
```

matching `topopt_freq` (Proposed) and the Yuksel entry points: density vector,
frequencies, mean time per iteration, iteration count, info struct.

`runCfg` exposes every parameter the parametric study needs — each mapping onto
exactly one field of the imported configuration, with unset fields keeping the
named configuration's documented value:

| WP3 requirement | `runCfg` key | cfg field |
|---|---|---|
| mesh dimensions | `nelx`, `nely` | `nelx`, `nely` |
| volume fraction | `volfrac` | `volfrac` |
| target mode | `target_mode` | `n` |
| filter radius | `rmin_elem` / `rmin_phys` | `rminEl` / `rminPhys` |
| move limit | `move` | `move` |
| multiplicity tolerance | `tol_mult` | `tolMult` |
| maximum outer iterations | `max_outer` | `maxOuter` |
| outer convergence tolerance | `tol_outer` | `tolOuter` |
| mass interpolation | `mass_interp` | `massInterp` (`4`/`4a`/`4b`/`lin`) |
| generalized-gradient filter mode | `filter_mode` | `filterMode` (`diag`/`all`/`none`) |

Plus route and model options: `inner_solver` (`lp`/`mma`), `off_diag`,
`n_modes_max`, `penal`, `rho_min`, `rho0`, `E0`, `nu`, `rho_m`, `L`, `H`,
`thickness`, `support_type` (SS/CS/CC), `support`, `axial`, `elem_type`,
`mass_type`, `eig_solver`, `threads`.

### Named configurations

Defaults correspond to a **documented, artifact-anchored** reproduction
configuration — no value was invented during migration.

| name | mesh | r_min | move | outer | anchored to |
|---|---|---|---|---|---|
| **`fig3a_best`** (default) | 240×30 | 1.3 el | 0.005 | 1600 | `baseline/lp240_rmin1.3.mat` |
| `fig4_history` | 240×30 | 1.3 el | 0.02 | 400 | `baseline/FIG4_definitive.mat` |
| `rmin1p8` | 240×30 | 0.06 phys | 0.005 | 1600 | `baseline/FINAL_lp_240x30.mat` |
| `paper_mma` | 160×20 | 3.0 el | 0.05 | 200 | `algo/defaultCfg.m` verbatim |
| `migration_smoke` | 160×20 | 1.2 el | 0.005 | 12 | *not a result — migration check only* |

`repro2007_config` **rejects unknown override field names** rather than silently
ignoring them, and clears the frozen-artifact claim whenever an override is
applied.

### Benchmark convention

```matlab
data.optimization.approach = 'OlhoffDu2007Repro';
[x, omega, tIter, nIter, mem, nIterStage, telemetry] = run_topopt_from_json(data);
```

Aliases `Olhoff2007Repro` and `reproduction2007` are also accepted. Unstated
paper quantities are exposed under an optional `optimization.repro2007` block;
absent keys keep the reproduction defaults.

### Original scripts preserved

```matlab
cd Matlab/reproduction2007; setpaths(); res = run_case('label', overrides);
```

`setpaths.m` and all 22 `runs/*.m` scripts were imported unchanged and remain
runnable for provenance and regression.

---

## 8. Standardized history (WP4)

`info.history` records per outer iteration:

| requirement | column(s) |
|---|---|
| iteration number | `iter`, `stage`, `stage_iter` |
| ω₁, ω₂, ω₃ and further requested modes | `omega1/2/3`, plus `omega` — **all** J+1 = 5 computed modes, full matrix |
| objective | `objective` = √β [rad/s], `beta` raw |
| volume | `vol`, `rV` |
| **actual eigengap** | `gap_rel`, `gap_abs` |
| **multiplicity N** | `N`, `bimodal`, `multJ` |
| move / max density increment | `d_inf`, `move_saturated`, `meta.move_limit` |
| inner/subproblem solves | `n_inner`, `cum_inner`, `inner_converged`, `degen_hits` |
| LP solver status | `lp_flag` |
| runtime components | `t_eig`, `t_grad`, `t_inner`, `elapsed_s` |

**`N` and the actual eigengap are both recorded.** `N` is a thresholded view of
`gap_rel` at `cfg.tolMult`, so reporting `N` alone would make the reported
multiplicity a function of an unstated tolerance — precisely the ambiguity this
reproduction exists to expose.

`lp_flag` is recovered without touching the solver: `innerLoopLP` sets
`conv = (flag == 1)`, which `olhoffOpt` stores per iteration, and failures are
written to `res.log` with the numeric flag, which the wrapper parses.

`info.timing` and `info.stopping` follow the shapes
`run_topopt_from_json` already consumes for the other approaches, so the
existing evaluator reads this method with no duplicate logic.
`info.native` carries the verbatim `olhoffOpt` result.

---

## 9. Compatibility decisions

### The historical implementations were not relocated

WP1 permits retaining the current physical location where moving would cause
disproportionate churn. `OlhoffApproachExact` is referenced from **18 places**,
including `tools/Matlab/run_topopt_from_json.m`, the frozen
`examples/Performance/ledger/protocol_ledger.json`, and three experiment trees
(`ablations`, `step_calibration`, `terminal_direction_audit`). Relocating it
mid-revision would churn the protocol ledger and every recorded experiment path
for no scientific gain.

The requested three-way structure is instead expressed as `Matlab/` with
`legacy/` and `full_coupling/` as **pointer READMEs** and `reproduction2007/`
holding real code.

### The reproduction was placed outside `analysis/` — a safety decision

This is the one substantive deviation from the literal sketch, and it exists
because of a hazard found during the WP0 audit.

`examples/Revision_v1/exp*.m` and `run_all_revision_experiments.m` do:

```matlab
addpath(toolsDir);
addpath(genpath(fullfile(repoRoot,'analysis')));   % prepends every subfolder
```

`genpath` prepends, so **anything under `analysis/` shadows `tools/Matlab/`**.
The reproduction ships its own `mmasub.m`, `subsolv.m` and `top88.m`, all of
which also exist in this repository. Importing it under `analysis/` would have
made those five revision experiments silently execute the reproduction's copies.

Placing it at `Matlab/reproduction2007/` puts it beyond the reach of every
`genpath(analysis)` sweep — at zero cost, with no edits to existing scripts, and
verified by check A below.

At migration time all three colliding pairs were **byte-identical** (SHA256
verified), so nothing was in fact mis-executed. That is a fact about today, not
a property that survives an edit, which is why the arrangement was avoided
rather than tolerated.

---

## 10. Path-isolation strategy

Four layers, all fail-closed:

1. **Location.** Outside `analysis/`; unreachable by `genpath(analysis)`.
2. **Narrow addpath.** `repro2007_paths()` adds exactly six directories,
   never with `genpath`.
3. **Scoped lifetime.** It returns an `onCleanup` guard restoring the previous
   path on return, *including on error*. Calling it without capturing the guard
   is itself an error.
4. **Asserted identity.** `repro2007_assert_identity()` verifies that **every**
   owned function resolves inside `repro2007_root()` and **errors** otherwise.
   A run that starts is a run whose implementation identity has been proved;
   the root is reported in `info.path_identity` and echoed to the console.

Benign collisions that resolve correctly are still reported, with a
byte-identity verdict, rather than hidden.

`tools/Matlab/run_topopt_from_json.m` adds **only** `reproduction2007/runner/`;
the algorithm itself stays behind the guard.

---

## 11. Reproduction regression results (WP5)

`repro2007_regression('full')` — the migrated code re-executed and compared
against the frozen source-repository artifacts at **zero tolerance**.

| baseline | iterations | checks | result | time |
|---|---|---|---|---|
| `fig4_history` — `FIG4_definitive.mat` | 400 | 13 | **13 pass / 0 fail** | 37.4 s |
| `fig3a_best` — `lp240_rmin1.3.mat` | 1600 | 13 | **13 pass / 0 fail** | 141.1 s |
| `rmin1p8` — `FINAL_lp_240x30.mat` | 1600 | 13 | **13 pass / 0 fail** | 144.1 s |

**Total: 39/39 bit-identical. No discrepancy to quantify.**

Per WP5's required items, for every baseline:

| # | item | result |
|---|---|---|
| 1 | initial spectrum | bit-identical (n=5) |
| 2 | early iteration history — ω, N, β, max\|Δρ\| | bit-identical (n=8000 / 1600 / 1600 / 1600) |
| 3 | multiplicity transition | bit-identical — iteration 26 (`fig4_history`), 95 (`fig3a_best`, `rmin1p8`) |
| 4 | representative later spectrum | bit-identical at the final iteration |
| 5 | volume | bit-identical (n=1600) |
| 6 | final density field | bit-identical (n=7200) |
| 7 | final frequencies | bit-identical (n=5) |

Plus: configuration transcription (34 fields), outer iteration count, and event
log length — all bit-identical.

Exact identity is the right test here because every source of run-to-run
variation is pinned: uniform ρ = 0.5 start; `eigSolve` forbids ARPACK's random
start vector and supplies a fixed deterministic `v0`; `linprog` runs
`dual-simplex-highs`; BLAS is pinned to one thread. Nothing was changed, so
anything but bit-identity would be a finding rather than a tolerance question.
The comparator quantifies and fails on any difference above `1e-12` relative
rather than absorbing it.

A cheap `repro2007_regression('prefix')` mode (40 iterations, ~15 s total) is
the routine gate; it also passed 25/25 on its comparable items.

### Headline reproduction, re-confirmed after migration

| quantity | paper | migrated code |
|---|---|---|
| ω₁ at the Fig. 3a optimum | 174.7 | 170.4709086 (−2.4 %) |
| ω₂ | 174.7 (bimodal) | 170.8658865 (gap 0.23 %) |
| ω₃ | 284.9 | 285.1939392 (+0.1 %) |
| ω₁⁰ initial, 160×20 | 68.7 | 68.3986 |

---

## 12. Historical-implementation regression (WP6)

### Path isolation — `repro2007_verify_isolation()`

| check | result |
|---|---|
| **A** — under the repository's own `addpath(genpath(analysis))` recipe, no reproduction function is reachable | **PASS** (54 owned names, 0 leaked) |
| **A2** — adding only `runner/` exposes entry points and nothing else | **PASS** (46 implementation names still unreachable) |
| **B** — `mmasub`, `subsolv` still resolve to `tools/Matlab/` | **PASS** |
| **C** — inside the guard, identity flips and is asserted | **PASS** (54 verified; 2 shadows reported, both byte-identical) |
| **D** — path restored exactly when the guard is released | **PASS** |

### Existing runners — all five through `run_topopt_from_json`

80×10, 8 iterations, from `examples/Performance/performance_comparison.json`:

| approach | result | ω₁ | iterations |
|---|---|---|---|
| `Olhoff` | **PASS** | 111.275 | 8 |
| `Yuksel` | **PASS** | 100.786 | 16 (2 stages) |
| `ourApproach` (Proposed) | **PASS** | 75.0945 | 8 |
| `OlhoffExact` | **PASS** | 135.783 | 8 |
| `OlhoffDu2007Repro` | **PASS** | 91.954 | 8 |

After **every** run, `which mmasub` and `which subsolv` still resolved to
`tools/Matlab/` — no leakage from the reproduction's path guard.

The historical implementations were neither moved nor edited, so their
behaviour cannot have changed; these runs confirm their entry points still
resolve in the presence of the new directory.

---

## 13. Deviations from the requested structure

| requested | delivered | why |
|---|---|---|
| `Matlab/{legacy,full_coupling,reproduction2007}` all holding code | `reproduction2007/` holds code; `legacy/` and `full_coupling/` are pointer READMEs | Moving `OlhoffApproachExact` would churn 18 references including the frozen protocol ledger. Explicitly permitted by WP1. |
| reproduction under `analysis/` alongside siblings | reproduction at `Matlab/reproduction2007/` | `addpath(genpath(analysis))` in five revision scripts would shadow `tools/Matlab/mmasub.m` and `subsolv.m`. §9. |
| per-iteration grayness in the history | **final iterate only** | See §14. |

---

## 14. Unresolved risks and limitations

**1. Per-iteration grayness and `d_rms` are not recorded.**
Both are functions of the full density field. The imported `olhoffOpt.m` records
only `mean(rho)` per iteration and keeps no density history. Producing them
requires adding a density recorder to `algo/olhoffOpt.m`, and that file is
deliberately held byte-identical to the clean-room source for the duration of
the paper revision — which is what makes the §11 result meaningful. The
final-iterate values *are* recorded, in `info.stopping.final_grayness`, and that
is what the performance comparison reads. Enabling per-iteration values later
is a one-line change plus a manifest note; it was not taken unilaterally.

**2. The benchmark's post-run eigensolve uses a different FE model.**
`run_topopt_from_json` recomputes topology modes with its own assembly
(`E_min_ratio` void stiffness) rather than the reproduction's SIMP + eq. (4)
mass model. In the WP6 smoke this showed as ω₁ = 91.9541 vs the solver's
91.9540 (≈1e-6 relative). Harmless at that size but **not** guaranteed to stay
small for a nearly-disconnected design, where the two void models diverge. When
citing frequencies for this method, use `omega` from the solver, not the
benchmark's post-hoc eigensolve.

**3. Nothing is committed.** Per the hard constraints, no commit and no push
were made. The migration lives in the working tree. Until it is committed, the
only provenance record of the imported bytes is `SOURCE_SHA256.txt` plus the
un-versioned source directory.

**4. The source repository is not under version control.** If it is edited or
deleted, the SHA manifest becomes unverifiable against any upstream. Committing
this migration is what makes the import durable.

**5. `mmasub`/`subsolv`/`top88` are duplicated.** All three pairs are currently
byte-identical, asserted at every run. If either copy is edited, check C in
`repro2007_verify_isolation` will report `*** DIFFERENT CONTENT ***` and fail —
but only when someone runs it. It is not wired into a CI gate, because this
repository has none for MATLAB.

**6. `support_type` cannot be inferred from `closest_point` BCs.** The
reproduction builds the paper's Fig. 2 supports itself, because which
idealization is used is one of its findings. A JSON specifying BCs as
`closest_point` entries yields `supportCode = 'NONE'`; the dispatch **errors**
rather than defaulting to simply supported, and requires an explicit
`optimization.repro2007.support_type`. This is intentional fail-closed
behaviour, and it means the reproduction cannot be dropped into an arbitrary
existing benchmark JSON without one added line.

**7. Only case (a) is reproduced.** Cases (b) and (c) optima, max-ω₂, the
gap problem (26), 3D plate and bimaterial are not started — see `NOTES.md` §10.
The migration changes none of that.

---

## 15. How to re-verify

```matlab
addpath('Matlab/reproduction2007/runner');

repro2007_verify_isolation();      % WP6 path safety,     ~10 s
repro2007_regression('prefix');    % WP5 fast gate,       ~15 s
repro2007_regression('full');      % WP5 full gate,       ~5.5 min
% which root am I running?  (needs the path installed first)
guard = repro2007_paths(); repro2007_assert_identity(true); clear guard
```

```bash
# WP2 byte-identity of the import against the source manifest
cd Matlab/reproduction2007
grep -v '^#' SOURCE_SHA256.txt | while read h f; do
  case "$f" in CLAUDE.md) t=SOURCE_CLAUDE.md;; results/*) t="baseline/$(basename $f)";; *) t="$f";; esac
  [ "$(shasum -a 256 "$t" | cut -d' ' -f1)" = "$h" ] || echo "MISMATCH: $f"
done
```

---

## 16. Addendum (2026-08-26) — benchmark solver swap for the Olhoff column

Requested after the migration: in `examples/Performance/performance_comparison.m`
the **name** `Olhoff` / `OlhoffApproach` must stay exactly as it appears in all
results, but the **solver** behind it must become `Matlab/reproduction2007`.
Inner and outer iteration counts must be shown in the result table.

### What changed

**Name and dispatch decoupled.** `approaches` keeps the method identity used
for every output path (console table, CSV `Method`, JSON `method`, LaTeX group
label). A new parallel array carries the dispatch key:

```matlab
approaches       = {'Olhoff',            'Yuksel', 'OurApproach'      };
solverApproaches = {'OlhoffDu2007Repro', 'Yuksel', 'OurApproach'      };
methodLabels     = {'OlhoffApproach',    'YukselApproach', 'ProposedApproach' };
```

Only `data.optimization.approach = solverApproaches{m}` changed. Nothing that
produces a name was touched, so `Olhoff`, `OlhoffApproach`, `Olhoff--Du` and
the `Olhoff` CSV/JSON keys all appear exactly as before. Verified in the smoke
run: CSV rows still start `Olhoff,`, JSON still records
`method: "Olhoff"`, `method_label: "OlhoffApproach"`.

**Outer/inner iterations are now reported.** Two new columns in Table 1
(`outer`, `inner`), two new CSV columns appended at the end
(`outer_iterations`, `inner_iterations` — appended so `regenerate_from_csv.m`
and `analyze_results.py`, which both read by column name, are unaffected), and
`record.iterations.{outer,inner,inner_solver}` in the JSON.

These are carried on a new `telemetry.iterations` field, **not** folded into
`nIterStage`: that struct means Yuksel's two native *stages*, and overloading
it would make `stage1` mean different things per method.

The two counts are genuinely independent, not a relabelling — measured at
160×20, 5 outer iterations:

| route | outer | inner | inner/outer |
|---|---|---|---|
| Eq. (22) LP | 5 | 5 | 1.0 |
| MMA + full (25d) coupling | 5 | **555** | **111.0** |

matching `NOTES.md` §7's "≈90–120 MMA sub-iterates per outer iteration".

### Two settings had to be scoped to this method

Both are set in one commented block in `performance_comparison.m` and affect
**no other method**. They are not preferences — the shared values produce
wrong or unrunnable results:

| setting | shared value | used for Olhoff | why |
|---|---|---|---|
| move limit | 0.2 | **0.005** | 0.2 is an MMA/OC move limit. Here it is an SLP trust region. **Measured** at 160×20, r_min = 2 el: the design collapses to a disconnected island and ω₁ ends at **2.9 rad/s** instead of ~160. `NOTES.md` §8c documents the same failure at 0.03. 0.005 is the documented `fig3a_best` value. |
| outer budget | `max_iters` = 10000 | **1600** | The LP always travels the full move limit, so `max\|Δρ\|` never falls below `convergence_tol` and the native stop test **cannot** fire (`move_saturated_frac = 1.000` in every frozen run). 10000 iterations × 9 meshes is a runtime bomb. 1600 is the documented `fig3a_best` budget. |

Filter radius was deliberately **left** at the benchmark's shared 2 elements.
The solver runs correctly there, and keeping the shared cross-resolution filter
is more defensible than deviating — but 2 elements is *not* the radius that
reproduces Fig. 3a (1.3 elements), so **the ω₁ reported for Olhoff in this
table is a valid operating point of the method, not the paper-reproduction
figure.**

### How to read the Olhoff column

Its `stop_reason` is **always** `max_outer_iterations` and its `iter_total` is
a fixed budget, not a convergence result. **`s/iter` and the fitted scaling
exponent are the meaningful entries for this method; `iter_total` and wall time
are not.** This is stated in the console legend and in the JSON metadata
(`olhoff_column_note`).

Provenance is recorded in the results: `benchmark_results.json` now carries
`metadata.method_dispatch = {Olhoff: "OlhoffDu2007Repro", …}`, so the file
itself says which solver produced each named column.

### Outstanding inconsistency — needs a decision

Only `performance_comparison.m` was switched, as asked. **Five other scripts
still dispatch `'Olhoff'` and therefore still run the *old*
`analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m`:**

- `examples/Performance/validate_determinism.m`
- `examples/Performance/validate_history_logging.m`
- `examples/Performance/validate_extension_invariance.m`
- `examples/Revision_v1/exp1_perf_table.m`
- `analysis/iteration_count_audit/verify_comparator_counts.m`

The name `Olhoff` now denotes **two different solvers** depending on which
script is run. That is a reporting hazard, not a code defect, and resolving it
is a scope decision: switch them too, or leave them pinned to the old solver
and rename one of the two. Not changed unilaterally.

### Verification

- `checkcode` clean on `performance_comparison.m` and `run_repro2007.m`;
  `run_topopt_from_json.m` retains only its two pre-existing warnings.
- End-to-end smoke at 80×10 and 160×20, all three methods, exit 0: table, CSV,
  JSON, LaTeX and complexity fits all produced.
- WP5 regression and WP6 path isolation re-run after the change: **both still
  pass**.
- No result artifact under `examples/Performance/` was overwritten; the smoke
  wrote into a scratch directory.
