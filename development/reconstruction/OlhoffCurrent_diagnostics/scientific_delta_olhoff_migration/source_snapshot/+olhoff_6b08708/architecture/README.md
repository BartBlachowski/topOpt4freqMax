# architecture/

The post-conference canonical layer for the Du–Olhoff eigenfrequency solver.

Everything here is additive. The numerical kernels in `../algo`, `../fem`,
`../filter` and `../mma*` keep their arithmetic; this layer replaces how a run
is *configured* and *identified*, not what it computes.

## Quick start

```matlab
setpaths

% what realizations exist
olh.presets.list()

% build one, and see exactly what it is
cfg = olh.config.resolve('duOlhoffFrozenM4', 'domain.mesh.nelx', 320, ...
                                             'domain.mesh.nely', 40);
olh.config.describe(cfg)

% run it
res = olhoffSolve(cfg);
res.status        % CONVERGED | CAP_HIT | SOLVER_FAILURE | STOPPED_OTHER
res.cfg           % the effective configuration, recorded with the result
```

Legacy code needs no change: `olhoffOpt(flatCfg)` still works and forwards to
the same solver.

## Layout

```
+olh/+config/     schema, defaults, validation, legacy adapters, resolve, describe
+olh/+presets/    ten named realizations; no mathematics
+olh/+material/   mass interpolation, value and derivative together
+olh/+multi/      multiplicity detection
+olh/+move/       move-limit policies and the stall controller
olhoffSolve.m     the main loop of Fig. 1
legacy/           the pre-canonical solver, preserved verbatim
anchors/          behavioural regression: configs, records, digests, references
tests/            the architecture-level test suite
docs/             see below
```

## Documents

| File | What it answers |
|---|---|
| `docs/ARCHITECTURE.md` | how the layers fit together and why |
| `docs/CONFIG_REFERENCE.md` | every field (generated from the schema) |
| `docs/PRESETS.md` | what each named realization is, and whose it is |
| `docs/SCIENTIFIC_CONFIG_PROVENANCE.md` | field-level A/B/C/D, with the quotations |
| `docs/TERMINOLOGY.md` | what "M4" means, the two betas, the density fields |
| `docs/MIGRATION_FROM_LEGACY.md` | the legacy field map |
| `docs/ARCHITECTURE_VARIANT_INVENTORY.md` | the pre-refactor audit |
| `docs/OLHOFF_ARCHITECTURE_REFACTOR_PLAN.md` | the plan this work followed |
| `docs/OLHOFF_ARCHITECTURE_REFACTOR_REPORT.md` | what was done and what it proved |
| `docs/FILE_CLASSIFICATION.md` | what is evidence, what is compatibility |
| `docs/PYMORPHOGEN_PORT_MAP.md` | readiness for a future Python port |

## Tests

```matlab
run_all_tests          % config, mass, modules, round-trip, presets
```

Behavioural regression, which needs MATLAB minutes rather than seconds:

```matlab
addpath architecture/anchors/code
runAnchor('A1_frozen160','candidate')   % compare against the stored reference
```

## Two rules

1. **Audit directories are immutable.** `audit_*/`, `results/`, `runs/*.mat` and
   `docs/` are scientific records. `EVIDENCE_MANIFEST.sha256` covers all 1507
   files in the tree and makes any accidental write detectable.
2. **A preset contains no mathematics.** If a realization needs a new branch in
   the solver, that branch is a new *configuration field* with a scientific name,
   not a test on the realization's name.
