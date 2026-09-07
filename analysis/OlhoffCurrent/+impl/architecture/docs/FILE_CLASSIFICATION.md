# FILE_CLASSIFICATION

Phase 19 requires that nothing be deleted until equivalence is established.
Nothing has been deleted. This is the prepared classification.

| Class | Meaning |
|---|---|
| `RETAIN_EVIDENCE` | immutable scientific record; never modify, never regenerate |
| `RETAIN_COMPATIBILITY` | superseded, but something still calls it |
| `ARCHIVE` | superseded and uncalled; move aside when convenient |
| `DELETE` | safe to remove |

## RETAIN_EVIDENCE

* every `audit_*/` directory, in full — reports, preregistrations, `runs/*.mat`,
  `*.npz`, figures, and the point-in-time solver snapshots (`baseline/tree/`,
  `audit_projection_800x100/solver/`) and `*_PRE_*.m` copies;
* `results/`, `runs/*.mat`, `docs/`;
* `NOTES.md`, `OLHOFFEXACT_FAILURE_POSTMORTEM.md`, `CLAUDE.md`;
* `EVIDENCE_MANIFEST.sha256` — the integrity record for all 1507 files;
* in the other repository, `analysis/OlhoffM4Reconstruction/` in full.

Verified after the refactor: 1504 of the 1507 recorded files are bit-identical
to the pre-refactor baseline; the three that changed are the three listed below
as deliberately modified. Zero files under `audit_*`, `results/`, `runs/` or
`docs/` changed.

## RETAIN_COMPATIBILITY

| File | Why it stays |
|---|---|
| `algo/olhoffOpt.m` | now a shim with no mathematics; every historical runner calls it |
| `algo/moveControl.m` | reached by the legacy path; also the reference the canonical `olh.move.limit` is tested against |
| `algo/multRule.m` | same, for `olh.multi.detect` |
| `algo/defaultCfg.m` | **superseded** — not any realization (eleven scientific differences from the frozen config). Retained only so historical scripts resolve. Marked in `MIGRATION_FROM_LEGACY.md` |
| `algo/innerLoop.m`, `innerLoopLP.m`, `innerLoopRho.m`, `genGrad.m`, `deltaLambda.m`, `useMMA.m` | live numerical kernels, unchanged |
| `fem/*`, `filter/*`, `mma/*`, `mma_published/*` | live numerical kernels |
| `architecture/legacy/olhoffOpt_PRE_CANONICAL.m` | the pre-refactor solver, preserved verbatim for diffing |
| all `audit_*/code/*.m` runners | untouched, still work |

## ARCHIVE

| File | Why |
|---|---|
| `top88.m` (repository root) | byte-identical duplicate of `filter/top88_reference.m` (`eb8613bd…`). Two copies of a reference implementation invite editing the wrong one. Not archived yet: it is on the path and something may call it by that name. |

## DELETE

Nothing, in this pass.

## Deliberately modified in this refactor

Three files, and only three:

| File | Change |
|---|---|
| `algo/olhoffOpt.m` | became a compatibility shim; the mathematics moved to `architecture/olhoffSolve.m` |
| `fem/massScale.m` | became a thin dispatcher onto `olh.material.massInterpolation`; the piecewise formulas now exist once |
| `setpaths.m` | one `addpath` so the `+olh` package resolves |

## Duplication recorded, not resolved

* `mma/subsolv.m` and `mma_published/subsolv.m` are byte-identical
  (`130033335ac5…`); only `mmasub.m` differs between the two "variants".
  Left alone: `useMMA` selects them by path order, and changing that is a
  trajectory risk for no scientific gain.
* Five complete solver snapshots exist under `audit_*/`. These are evidence,
  correctly so.
