# PROVENANCE — analysis/OlhoffCurrent

Where this implementation came from, exactly, and what was changed on the way in.

The machine-readable form of everything below is [`PROVENANCE.json`](PROVENANCE.json);
it is what `olhoffcurrent_provenance()` returns and what every production result
embeds.

---

## 1. Source

| | |
|---|---|
| Source repository | `/Users/piotrek/Programming/Matlab/Olhoff` |
| Its role | **DEVELOPMENT / RESEARCH UPSTREAM** |
| Source branch | `architecture/canonical-config` |
| Source commit | `695f03bdac20c423a4e1d389cf9db9187597bcc3` |
| Commit date | 2026-09-06 23:24:25 +0200 |
| Commit subject | `Phase 24: final report -- 12/12 anchors bitwise, verdict VERIFIED` |
| **Source dirty state at promotion** | **clean** — `git status --porcelain` was empty |
| Acceptance evidence | `architecture/docs/OLHOFF_ARCHITECTURE_REFACTOR_REPORT.md`, verdict `OLHOFF_ARCHITECTURE_REFACTOR_VERIFIED` |
| Promotion date | **2026-09-07** |

There was no ambiguity about which upstream commit to promote: the branch has a
single head, the tree was clean, and `695f03b` is the commit whose report
carries the acceptance verdict.

### Verified before promoting, not assumed

The brief requires the reported behaviour to be *verified* rather than trusted.
Two read-only checks were run against the upstream tree:

* `anchorReport()` recomputed both digests for all twelve stored regression
  anchors — **0 of 12 failed** the equality standard;
* anchor `A1_frozen160` — the frozen conference realization, the one that
  matters here — was **re-executed live from source** under this machine's
  MATLAB and its digest recomputed in memory:

  ```
  LIVE  A1_frozen160  outer=91 inner=2241 status=CONVERGED omega1=169.495227021538
  science digest  1afe4b7e0cf86a860482e34adf08ad20af47d070494d2147e0eedfdf1cca9001
  reference       1afe4b7e0cf86a860482e34adf08ad20af47d070494d2147e0eedfdf1cca9001
  VERDICT science=IDENTICAL  logShape=IDENTICAL
  ```

Nothing was written into the upstream repository to obtain this.

## 2. Main repository at promotion

| | |
|---|---|
| Path | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **Commit before promotion** | `9cd1c8eed109e1c1b02c67fc6d9bb81f87179d6a` |
| State | clean, except four untracked historical Olhoff directories (recorded as debt in `analysis/OLHOFF_SOURCE_LINEAGE_AUDIT.md` §4.1) |

## 3. What was promoted

```
analysis/OlhoffCurrent/+impl/
    algo/            fem/            filter/
    mma/             mma_published/  architecture/{+olh/, olhoffSolve.m, legacy/, docs/}
```

**74 files.** All are byte-identical to upstream `695f03b` **except the single
adaptation in §7**.

The sibling layout is preserved exactly as upstream, because it is load-bearing:
`algo/useMMA.m` locates the MMA variants as
`fileparts(fileparts(mfilename('fullpath')))/mma*`.

### Not promoted, and why

| Excluded | Why |
|---|---|
| `audit_*/`, `results/`, `runs/`, `NOTES.md`, `CLAUDE.md`, `EVIDENCE_MANIFEST.sha256` | upstream scientific evidence; it stays upstream and is not executable production code |
| `architecture/anchors/`, `architecture/tests/` | upstream's own regression harness for the refactor; it validates *upstream*, and its files hard-code the upstream absolute root |
| `setpaths.m` | replaced by `olhoffcurrent_paths.m`, which additionally *proves* the resolution instead of only setting it |
| `top88.m` (repo root) | a byte-identical duplicate of `filter/top88_reference.m`; upstream classifies it `ARCHIVE` |

## 4. Integrity manifest

| | |
|---|---|
| File | [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) |
| Files covered | 74 |
| **Source tree SHA-256** | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` |

The tree hash is the SHA-256 of the sorted `<relative path>  <sha256>` lines, so
it changes if any file changes, is added or is removed, and does not depend on
filesystem ordering. Verify with `olhoffcurrent_source_manifest()`.

## 5. Canonical production preset

```
duOlhoffFixedPenaltySensitivityFiltered
```

Delegates to the promoted upstream preset `olh.presets.duOlhoffFrozenM4`, so
there is exactly one definition of the mathematics and production cannot drift
from it. Resolved per mesh by `olhoffcurrent_config(nelx, nely)`; the mesh is
applied as an override *before* the derived rules run, so the mesh-scaled outer
tolerance `eps = 0.05·sqrt(NE/3200)` is re-derived rather than retyped.

### Historical audit IDs are provenance aliases, not canonical API terminology

**M4**, **TMA**, **B0**, **REG160**, **S2**, **R1**, **R2**, **P1**, **PD1**,
**PM1**, **T800** and the upstream preset name `duOlhoffFrozenM4` are
**provenance aliases and experiment identifiers**. They exist so historical
evidence can be matched to new runs. They are **not** user-facing API names, and
no production script uses them.

## 6. Relationships

| Tree | Relationship |
|---|---|
| `analysis/OlhoffM4Reconstruction` | **FROZEN_EVIDENCE.** The frozen conference reconstruction, imported from the *same* upstream tree on 2026-09-04. It is the realization this production preset must reproduce — and does, bitwise, at 160×20 and 320×40. It is **not** a production dependency and is hard-blocked from the production MATLAB path. |
| `/Users/piotrek/Programming/Matlab/Olhoff` | **DEVELOPMENT_UPSTREAM.** Never an executable dependency of a production run: it is where experiments happen, it is not pinned by this repository's history, and a run that reached it could not be reproduced from this repository alone. Blocked by absolute path. |
| `analysis/OlhoffExperiments` | **EXPERIMENTAL.** Not created by this task. Blocked pre-emptively so it cannot become a production dependency by accident. |
| all other Olhoff trees | **HISTORICAL / AUDIT_ONLY**, classified in `analysis/OLHOFF_IMPLEMENTATION_MAP.md` and blocked. |

## 7. The one adaptation

Everything promoted is byte-identical to upstream except **one file**.

| | |
|---|---|
| File | `+impl/architecture/olhoffSolve.m` |
| Kind | **BENCHMARK TIMING INSTRUMENTATION** |
| Upstream SHA-256 | `9f80dd8ed504dc5cce5d8ab995cd4520e1e388ec6478f6446378cb8a464e2543` |
| Promoted SHA-256 | `5d4abd37c8b186a42d1a7ef5b7066c8bc2429d33ee91f20b0b844bace772324b` |

**The change.** Adds `hist.tOuter`: a `tic` at the top of the outer loop, and a
`toc` recorded after the convergence test and every guard.

**Why it is necessary.** The conference benchmark's nested cost accounting
reports `outer_time_excluding_inner_s`, which requires the wall time of one
*complete* outer iteration. The frozen conference reconstruction carries exactly
this instrumentation — recorded upstream as
`analysis/OlhoffM4Reconstruction/patches/olhoffOpt.timing-instrumentation.diff` —
and upstream's `olhoffSolve.m` does not. Repointing production without it would
have silently broken the benchmark's accounting. It is the same modification,
already audited in its previous home, carried forward to the same effect.

**Effect on the trajectory: none.** `hist.tOuter` is written and never read back
by the solver.

**And that is proved, not asserted.** `A1_frozen160` was executed live against
**uninstrumented** upstream code and produced
`omega1 = 169.495227021538`, `outer = 91`, `inner = 2241`. The **instrumented**
promoted code at 160×20 produces `omega1 = 169.49522702153845`, `outer = 91`,
`inner = 2241`, and reproduces the frozen conference design vector bitwise. The
instrumentation is therefore demonstrably inert across an
instrumented/uninstrumented boundary, not merely argued to be.

## 8. Currentness — and why "upstream is ahead" is not "obsolete"

`olhoffcurrent_currentness()` reports exactly one of:

| State | Meaning |
|---|---|
| `CURRENT` | promoted source intact and matching the upstream commit it came from |
| `LOCAL_MODIFIED` | `+impl/` no longer hashes to `SOURCE_MANIFEST.json`. **The most serious state** — the recorded provenance no longer describes the code on disk |
| `UPSTREAM_AHEAD` | upstream has commits after the promoted one. **Informational** |
| `PROVENANCE_MISMATCH` | the promoted commit is not an ancestor of upstream HEAD — history rewritten, or the branch moved |
| `UPSTREAM_UNREACHABLE` | the development repository is absent; local integrity was still checked |

The upstream repository is a **development** tree. Experimental commits land
there constantly and most will never be promoted — they are audits, spikes and
abandoned branches. So an upstream commit that nobody has accepted does **not**
make production stale; it makes production *different from a draft*.

**Production currentness changes on exactly one event: a human explicitly
accepts an upstream state and promotes it.** Accordingly
`olhoffcurrent_currentness()` never updates anything. It reports; a person
decides.

## 9. The promotion workflow

```
external development
    -> experiment / audit
    -> accepted committed state
    -> explicit promotion into OlhoffCurrent
    -> regression verification  (path isolation, currentness, preset equivalence)
    -> production
```

Production scripts must **never** execute directly from the external repository.
Production source is never edited in place.
