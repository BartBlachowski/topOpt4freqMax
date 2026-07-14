# Revision_v1 experiment scope

The production entry point is `run_all_revision_experiments.m`. Its active
stages are `S1`, `EXP2`, `EXP2b`, `EXP3`, and `A4`, plus their declared
verification/acceptance gates.

Current state, remaining computation, and reviewer blockers are tracked in:

- `REVISION_CURRENT_STATE_REPORT.md` — inventory, classification, migration record
- `REVISION_EXECUTION_PLAN.md` — the stages still required, and in what order
- `SCIENTIFIC_DECISION_EXP1_EXP5.md` — why EXP1/EXP5 are retired as reviewer evidence

## Active stages

`S1 → EXP2 → EXP2b → EXP3 → A4` (plus the I1 smoke gate, and CR2 as a governed
standalone run). **EXP1 and EXP5 are retired**; no cross-code performance, timing, memory,
or scaling stage remains active, and comparator telemetry is deliberately not implemented.

## Comparator policy

- The only Olhoff comparison allowed in production is the local
  `analysis/OlhoffApproach` implementation, selected by the `Olhoff` approach
  key and labelled "local Olhoff-inspired implementation."
- `analysis/OlhoffApproachExact` and
  `analysis/OlhoffApproachExactOpus` are archived diagnostic reconstruction
  attempts. They are not canonical/reference implementations and are not
  reviewer evidence.
- No production configuration selects `OlhoffExact`.
- No production table, figure, convergence comparison, frequency-difference
  statement, runtime ratio, or scaling fit may consume an Exact artifact.

The production dependency chain is:

`run_all_revision_experiments` → `S1` → `EXP2` → `EXP2b` → `EXP3` → `A4`.

No active stage consumes a cross-code comparator: the stage that did
(`exp1_perf_table`, with `exp5_scaling` consuming its output) is retired and
archived. No link in this chain imports `OlhoffApproachExact`; the path
allowlist in `run_all_revision_experiments.m` (`localAddActiveAnalysisPaths`)
excludes it, and preflight P1/P2 fail the campaign if an archived tree or an
archived runner is reached.

## Directory layout

```
Revision_v1/
  run_all_revision_experiments.m   master runner (fail-loud; Gate I1 verified)
  exp*.m, cr2/, s1_*.m             active experiments (see execution plan)
  *.json                           active configurations
  output/                          results of ACTIVE experiments only
  archive/                         governed archive — NOT reviewer evidence
```

## Archive

`archive/` holds preserved provenance that is **not** reviewer evidence. See
`archive/README.md` for the governing rules. It contains:

- `archive/olhoff_exact_reconstruction/` — the six standalone `OlhoffApproachExact`
  reconstruction scripts (`scripts/`) and their seven output directories
  (`output/`), plus the `OLHOFF_EXACT_ARCHIVE.md` index. These are the only files
  in Revision_v1 that add `OlhoffApproachExact` to the MATLAB path, they are not
  stages of the master runner, and they must not be added to a production manifest.
- `archive/diagnostics/` — closed investigations (Eq. 4b hypothesis: refuted;
  S1 mode diagnostics; localized-mode onset study; alpha=1 discrepancy note).
- `archive/superseded_runs/` — results predating the authoritative load
  `F(x) = omega0^2 * M(x) * Phi0`, and the failed campaign
  `r1_full_20260701T141604990`.

Historical numeric files are intentionally unchanged; the archive classification
supersedes their former production, equivalence, or migration recommendations.
