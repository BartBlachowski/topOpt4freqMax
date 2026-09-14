# Import manifest — Du–Olhoff reconstruction (M4)

Human-readable rendering of `IMPORT_MANIFEST.json`. That file is the machine-readable
original; this one exists so the provenance can be read without a JSON viewer.

## Source

| | |
|---|---|
| source repository | `/Users/piotrek/Programming/Matlab/Olhoff` |
| version control | none (not a git repository) |
| state identification | SHA-256 of every file in algo/, fem/, filter/, mma/, mma_published/ plus NOTES.md, CLAUDE.md, setpaths.m |
| newest source file mtime | 2026-09-04T12:19:55.765722+02:00 |
| import date/time | 2026-09-04T13:43:04.285116+02:00 |
| destination | `analysis/OlhoffM4Reconstruction` |
| importing repo HEAD | `d0c78ca66f72b6041db908f808bd342621f5ced6` (branch `benchmark-methodology-r2`) |
| importing repo dirty at import | True |

The source is **not a git repository**, so its state is identified by the
SHA-256 of every file in its solver tree. The full listing is in
`SOURCE_SHA256.txt` and in the JSON under `source_repository.file_hashes`.

## Source repository status — observed 2026-09-04, **no longer in the imported state**

The external source directory has been **reverted to a pre-import generation**.
Of the 23 imported files, **7 are absent from it entirely**
(`multRule.m`, `moveControl.m`, `useMMA.m`, `innerLoopRho.m`, and all three
`mma_published/` files), **4 differ** (`defaultCfg.m`, `deltaLambda.m`,
`innerLoop.m`, `olhoffOpt.m`) and 12 are unchanged. The versions now present are
dated 2026-08-25 and predate the frozen realization itself — the `defaultCfg.m`
there has no `multRule`, `moveFamily`, `mmaVariant` or `outerGuard` fields at
all, and the `audit_*` directories cited under *Governing audits* are gone.

**Effect on this import: none.** Every one of the 23 files under `+frozen/`
still hashes exactly to its recorded `sha256_imported`. Nothing in
`analysis/OlhoffM4Reconstruction` changed. This was a rollback of an unversioned
scratch directory outside this repository.

Provenance is therefore proved from evidence held **inside this repository**,
which needs no access to that directory:

| | proof | checked by |
|---|---|---|
| **integrity** | every `+frozen/` file hashes to `sha256_imported` | `olhoffm4_verify_import.m` |
| **attestation** | for every file but the declared modification, the manifest records `sha256_imported == sha256_source` | `olhoffm4_verify_import.m` |
| **reconstruction** | `patches/olhoffOpt.m.source-verbatim` hashes to the recorded `sha256_source`, and applying `patches/olhoffOpt.timing-instrumentation.diff` to it reproduces `+frozen/algo/olhoffOpt.m` **byte for byte** | `olhoffm4_apply_unified_diff.m` |

The reconstruction check is *stronger* than the external-source comparison it
replaces: it **exhibits** the delta from the audited source instead of asserting
that one exists, so "declared modification" cannot become a loophole.

The external directory's state is still measured on every run and written into
every benchmark manifest — as **provenance, printed as a preflight note**, no
longer as a gate. A rollback in a directory this repository does not control
cannot change which code it runs, so it must not decide whether it may run it.
Full detail, including the observed hashes, is in `IMPORT_MANIFEST.json` under
`source_repository_status`.

What still fails closed: any `+frozen/` file whose hash moves; any imported file
neither attested byte-identical to the audited source nor covered by a declared
modification; a verbatim source copy that does not hash to `sha256_source`; a
declared diff that does not apply, or applies without reproducing
`sha256_imported`; any owned function resolving outside the import or inside a
superseded Olhoff tree.

## Governing audits

- **frozen_realization_spec** — audit_conference_admission/WP4_frozen_realization.md
- **filter_freeze** — audit_filter_mesh_admission/FROZEN_REALIZATION.json
- **termination_and_guard** — audit_termination_mesh_admission/REPORT.md (verdict S2_CONTINUATION_DEFECT)
- **latest_audit** — audit_s2_design_continuation/REPORT.md (verdict S2_LADDER_ITSELF_DEFECTIVE; the drms trigger was NOT adopted, s2Signal stays 'beta')
- **conference_active_realization** — the audit_termination_mesh_admission (TMA) realization: M4 subspace subN=2, physical filter R=0.06, tolInner=0.05, S2 beta ladder, outerGuard='settledmove', l2 outer norm with tolOuter = 0.05*sqrt(NE/3200)

## Epistemic status

- benchmark label: **Du-Olhoff reconstruction (M4)**
- must **not** be labelled: *Olhoff 2007*
- M4 is a RECONSTRUCTION of the published nested formulation, not a claimed exact historical implementation. Numerical continuation and inner convergence details are incompletely specified in the original publication.

## Declared modifications

### `+frozen/algo/olhoffOpt.m` — timing instrumentation only

Adds hist.tOuter, the wall time of each complete outer iteration (inner solve included). Three hunks: one hist field, one tic, one toc. No value is read back; no scientific quantity, tolerance, rule or ordering is touched.

- diff: `patches/olhoffOpt.timing-instrumentation.diff`
- verbatim source copy: `patches/olhoffOpt.m.source-verbatim`
- inertness evidence: `evidence/import_equivalence_160x20.json`

Every other imported file is **byte-identical** to its source.

## Not imported

development/plotting/analysis utilities not on the solver call graph:

- `algo/compareHistory.m`
- `algo/compareHistoryTo.m`
- `algo/compareTopology.m`
- `algo/imresizeNN.m`
- `algo/plotHistory.m`
- `algo/textStamp.m`
- `algo/topologyImage.m`
- `filter/top88_reference.m`
- `top88.m`
- `setpaths.m`
- `docs/`
- `runs/`
- `results/`
- `audit_*/`

## File map and hashes

| source | destination | SHA-256 (source) | SHA-256 (imported) | byte-identical |
|---|---|---|---|---|
| `algo/defaultCfg.m` | `+frozen/algo/defaultCfg.m` | `dd8f1cc7c19d41df…` | `dd8f1cc7c19d41df…` | yes |
| `algo/deltaLambda.m` | `+frozen/algo/deltaLambda.m` | `101519fcdcc2175d…` | `101519fcdcc2175d…` | yes |
| `algo/genGrad.m` | `+frozen/algo/genGrad.m` | `13f2ab8541b2f626…` | `13f2ab8541b2f626…` | yes |
| `algo/innerLoop.m` | `+frozen/algo/innerLoop.m` | `9ec33d7dcbabf4df…` | `9ec33d7dcbabf4df…` | yes |
| `algo/innerLoopLP.m` | `+frozen/algo/innerLoopLP.m` | `7724753c02f84d60…` | `7724753c02f84d60…` | yes |
| `algo/innerLoopRho.m` | `+frozen/algo/innerLoopRho.m` | `aded1849467e9320…` | `aded1849467e9320…` | yes |
| `algo/moveControl.m` | `+frozen/algo/moveControl.m` | `76ff15824f205db8…` | `76ff15824f205db8…` | yes |
| `algo/multRule.m` | `+frozen/algo/multRule.m` | `c8ee926c2cf985b7…` | `c8ee926c2cf985b7…` | yes |
| `algo/olhoffOpt.m` | `+frozen/algo/olhoffOpt.m` | `c6c4862440208c2e…` | `f83b468415cdeccf…` | **NO — see declared modifications** |
| `algo/useMMA.m` | `+frozen/algo/useMMA.m` | `b87ef5f65918c015…` | `b87ef5f65918c015…` | yes |
| `fem/assemble2D.m` | `+frozen/fem/assemble2D.m` | `7b2f1e10228d5723…` | `7b2f1e10228d5723…` | yes |
| `fem/classifyModes.m` | `+frozen/fem/classifyModes.m` | `0f93a2acce7aac69…` | `0f93a2acce7aac69…` | yes |
| `fem/eigSolve.m` | `+frozen/fem/eigSolve.m` | `b0784ceeb15fafe1…` | `b0784ceeb15fafe1…` | yes |
| `fem/elemMats2D.m` | `+frozen/fem/elemMats2D.m` | `b8ae7c424d42f023…` | `b8ae7c424d42f023…` | yes |
| `fem/massScale.m` | `+frozen/fem/massScale.m` | `90ba72c3df056ed0…` | `90ba72c3df056ed0…` | yes |
| `fem/model2D.m` | `+frozen/fem/model2D.m` | `3cc19e1c52023225…` | `3cc19e1c52023225…` | yes |
| `filter/applyFilter.m` | `+frozen/filter/applyFilter.m` | `461ec7c7950b4d08…` | `461ec7c7950b4d08…` | yes |
| `filter/prepFilter.m` | `+frozen/filter/prepFilter.m` | `18ceb629c567fafc…` | `18ceb629c567fafc…` | yes |
| `mma/mmasub.m` | `+frozen/mma/mmasub.m` | `54c1680036e6effd…` | `54c1680036e6effd…` | yes |
| `mma/subsolv.m` | `+frozen/mma/subsolv.m` | `130033335ac5a21f…` | `130033335ac5a21f…` | yes |
| `mma_published/README.md` | `+frozen/mma_published/README.md` | `14ac696e5b319a19…` | `14ac696e5b319a19…` | yes |
| `mma_published/mmasub.m` | `+frozen/mma_published/mmasub.m` | `4507b73e3e44cdd6…` | `4507b73e3e44cdd6…` | yes |
| `mma_published/subsolv.m` | `+frozen/mma_published/subsolv.m` | `130033335ac5a21f…` | `130033335ac5a21f…` | yes |

Full 64-character hashes are in `IMPORT_MANIFEST.json`. `olhoffm4_verify_import.m`
re-checks all of them, and re-checks the source too when it is reachable.
