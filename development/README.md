# development/ — archive. Not part of the supported code.

> **This directory contains historical reconstruction work, experiments,
> audits, evidence and superseded implementations.**
>
> **Nothing under `development/` is part of the supported runtime
> implementation. Production code, examples and current tests must not depend
> on files here.**
>
> For normal work use `analysis/`, `examples/`, `tools/` and `tests/` — see the
> repository [README](../README.md).

The material is kept on purpose, for **scientific provenance and
reproducibility**: recorded findings, frozen evidence and published numbers are
statements *about* this code and these runs, and they stay verifiable only while
the code and the runs still exist. Nothing was deleted when it was archived.

## What is where

| directory | contents |
|---|---|
| [`reconstruction/`](reconstruction/) | Du–Olhoff reconstruction history: the frozen conference realization `OlhoffM4Reconstruction`, the 2007 clean-room reproduction (`Matlab/reproduction2007`) and its runner `OlhoffReproduced2007`, and the diagnostics, raw evidence and study-governance gates produced on the production Olhoff tree while it was named `analysis/OlhoffCurrent` |
| [`legacy_implementations/`](legacy_implementations/) | superseded implementations: `OlhoffApproach`, `OlhoffApproachExact`, reference codes from `source_of_truth/`, a local zip backup |
| [`experiments/`](experiments/) | `OlhoffRegularized`, `LabandaApproach`, the iteration-efficiency study (`iteration_efficiency/`), the paper-revision experiments (`Revision_v1*`), the bimodality-gap study |
| [`audits/`](audits/) | convergence, stabilization, fixed-budget, nested-MMA-route, iteration-count and campaign forensic audits |
| [`benchmark_history/`](benchmark_history/) | earlier benchmark drivers, protocols, validation records and outputs (r3 / Table 1 era, `conference_benchmark_v1`), the profile-calibration study, targeted replays |
| [`legacy_examples/`](legacy_examples/) | ad-hoc example/test scripts with outdated path setup, old `results/` figures |
| [`migration_history/`](migration_history/) | the 2026-09-07 OlhoffCurrent promotion reports and implementation map |
| [`historical_documents/`](historical_documents/) | audit and methodology reports formerly at the repository root and in `docs/` |
| [`repository_cleanup/`](repository_cleanup/) | the 2026-09-14 migration that created this directory: baseline, source-of-truth analysis, manifest, verification |

## Reading archived material

**Paths inside archived files are historical.** A script here may say
`addpath(fullfile(repo,'analysis','OlhoffCurrent'))` or read
`analysis/iteration_efficiency_final/...`. Those strings are part of the
record and were deliberately not rewritten. The complete old → new mapping is in
[`repository_cleanup/MIGRATION_MANIFEST.tsv`](repository_cleanup/MIGRATION_MANIFEST.tsv)
(per directory) and
[`repository_cleanup/MIGRATION_FILEMAP.tsv`](repository_cleanup/MIGRATION_FILEMAP.tsv)
(per file, with SHA-1).

To **re-run** archived material exactly as it ran, check out the commit before
the migration — the parent of the migration commit, or `0591993` if the
migration was committed directly on it — where every path is still valid.
Running archived scripts from their new location is unsupported and will
generally fail on paths.

Several archived trees carry hash manifests (`IMPORT_MANIFEST.json`,
`FINAL_SHA256.txt`, `SHA256SUMS.txt`, `EVIDENCE.json`). The files were moved
byte-for-byte, so the hashes still hold; the recorded *paths* are the
pre-migration ones.

## Rules

1. Nothing in `analysis/`, `examples/`, `tools/` or `tests/` may reference this
   directory. `tests/test_development_firewall.py` enforces this statically. At
   run time, the Olhoff path gate (`olhoffcurrent_forbidden_paths`) refuses any
   MATLAB path entry below `development/`.
2. Material here may reference current code; it is not maintained against it.
3. To bring something back into use, **promote** it explicitly: copy it into
   the current tree, verify it there, and record the promotion. Do not add a
   `development/` path to a current script.
