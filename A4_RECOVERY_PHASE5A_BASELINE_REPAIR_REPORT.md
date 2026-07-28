# A4 Recovery Phase 5A — Immutable Baseline Repair Report

**Date:** 2026-07-27  
**Scope:** immutable V-P2-2 topology-baseline lifecycle only  
**Production campaign:** not executed  
**HEAD during repair:** `3542a2dc48201fda48dd05146152b2bcdccc8d30`

## 1. Root cause

The V-P2-2 reference topology was stored at
`examples/Revision_v1/output/a4/a4_topology_Ninf.csv`. That directory is the
mutable destination of a production campaign and is legitimately archived,
moved, emptied, or regenerated before a clean run. The Phase 4 regression
therefore depended on a stale production result being present in the output
directory. Phase 5 exposed the lifecycle defect correctly by emptying that
directory and stopping at the pre-run gate.

The immutable reference and current-run product now have separate roles and
locations:

- immutable repository fixture:
  `examples/Revision_v1/reference/a4/a4_topology_Ninf.csv`;
- current-run product:
  `examples/Revision_v1/output/a4/a4_topology_inf.csv`.

## 2. Byte-identical reference establishment

The source was the preserved Phase 3/Phase 5 pre-run copy:

`phase5_evidence/a4_phase5_prerun_output_20260724T084447Z/a4_topology_Ninf.csv`

It was copied byte-for-byte to:

`examples/Revision_v1/reference/a4/a4_topology_Ninf.csv`

The file was not regenerated, parsed, numerically transformed, or rewritten
through MATLAB. `cmp` confirmed byte identity. The installed fixture has 20,000
lines and 107,038 bytes.

SHA-256:

`9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`

The reference CSV is explicitly unignored and staged in the Git index;
`git ls-files --error-unmatch` confirms it is tracked.

## 3. Fail-loud pre-run validation

`a4_frozen_baseline_reference` resolves the one repository reference location
and the one produced topology path. It calls
`a4_validate_frozen_baseline` before configuration loading, fixture validators,
or the production arm loop.

The gate verifies:

1. the immutable reference exists;
2. its exact-byte SHA-256 equals the declared expected hash;
3. its canonical path is outside the mutable output directory;
4. its canonical path does not alias the produced topology path.

Precise failure identifiers are:

- `a4:FrozenBaselineMissing`;
- `a4:FrozenBaselineHashMismatch`;
- `a4:FrozenBaselineInsideMutableOutput`;
- `a4:FrozenBaselinePathAlias`.

The V-P2-2 validator additionally requires validated baseline metadata, rejects
an undeclared produced path, and compares the produced CSV SHA-256 with both
the immutable fixture SHA-256 and the declared expected hash. The in-memory
versus CSV-reload comparison remains non-blocking.

## 4. Modified and new files

Modified:

- `.gitignore`
- `examples/Revision_v1/a4_eigenpair_refresh.m`
- `scripts/revision_v1/a4_validate_frozen_identity.m`
- `scripts/revision_v1/test_a4_phase4.m`

New:

- `examples/Revision_v1/reference/a4/a4_topology_Ninf.csv`
- `scripts/revision_v1/a4_sha256_file.m`
- `scripts/revision_v1/a4_validate_frozen_baseline.m`
- `scripts/revision_v1/a4_frozen_baseline_reference.m`
- `scripts/revision_v1/test_a4_phase5a.m`
- `A4_RECOVERY_PHASE5A_BASELINE_REPAIR_REPORT.md`

The driver now records immutable-baseline path/hash metadata in the result
structure, result JSON, stage result, both manifests, and generated recovery
report.

## 5. Repository-reference audit

Live consumers found and updated:

- Phase 4 regression: now obtains the baseline through the immutable-reference
  locator and uses temporary produced files;
- V-P2-2 validator: now requires validated immutable-reference metadata;
- production driver: now runs the immutable-baseline gate before the arm loop
  and passes the validated reference to V-P2-2;
- report, stage-result, result, and manifest writers: now record the immutable
  reference path and hash;
- lifecycle regression: explicitly proves that no reference prerequisite
  exists in `output/a4`.

Remaining textual mentions of the old mutable path occur only in preserved
historical reports, archived artifacts/manifests, and the Phase 5 pre-run
failure record. They were intentionally not rewritten because they describe
the historical state that caused the valid stop.

No live code recreates `a4_topology_Ninf.csv` under `output/a4`.

## 6. Lifecycle regression evidence

`test_a4_phase5a` passed 12/12:

1. immutable reference exists with the declared SHA-256;
2. immutable reference is outside mutable output;
3. reference and produced paths are distinct;
4. production `output/a4` is empty at pre-run;
5. no immutable-baseline prerequisite exists in `output/a4`;
6. moving and deleting a temporary mutable output leaves the reference intact;
7. byte-identical produced topology passes V-P2-2;
8. modified produced topology fails V-P2-2;
9. a modified temporary reference copy fails with
   `a4:FrozenBaselineHashMismatch` before any checkpoint/arm artifact exists;
10. a missing reference fails with `a4:FrozenBaselineMissing`;
11. aliased reference/produced paths fail with
    `a4:FrozenBaselinePathAlias`;
12. a reference inside mutable output fails with
    `a4:FrozenBaselineInsideMutableOutput`.

All destructive lifecycle operations used temporary directories. The
authoritative fixture was read-only during negative tests.

## 7. Mandatory regression results

All required suites were run after the repair in one MATLAB R2025b session:

| Suite | Result |
|---|---:|
| `test_a4_phase5a` | **12/12 passed** |
| `test_a4_phase4` | **8/8 passed** |
| `test_a4_phase1` | **10/10 passed** |
| `test_a4_phase2` | **15/15 passed** |
| `test_a4_pipeline` | **28/28 passed** |
| `test_a4_refresh` | **22/22 passed** |
| `test_a4_classifier` | **17/17 passed** |

Combined log:
`phase5_evidence/a4_phase5a_mandatory_tests_20260727.log`

Log SHA-256:
`4567d4862d851e07348bb0da66898c5ec52032fe9501dc274285070f71da63f3`

The independent configuration/single-factor audit also passed 19/19.
`git diff --check` passed. MATLAB Code Analyzer reported no parse errors; its
messages were existing style/alignment and stale-suppression notices.

The real `examples/Revision_v1/output/a4` directory remained empty before,
during, and after the mandatory suites. Phase 4 therefore passes without any
preserved production result in mutable output.

## 8. Evidence preservation

The following historical evidence was not altered:

- `A4_RECOVERY_PHASE3_EXECUTION_REPORT.md`;
- `A4_RECOVERY_PHASE4_BLOCKER_RESOLUTION_REPORT.md`;
- `phase5_evidence/A4_PHASE5_PRE_RUN_FAILURE_20260727.md`;
- `phase5_evidence/a4_phase5_prerun_tests_20260727.log`;
- `phase5_evidence/a4_phase5_interrupted_20260727T075221Z.tar.gz`;
- the moved interrupted-output directory.

Verified SHA-256 values include:

- Phase 3 report:
  `99deb76d17e6875cd2d62310ea377f408a1373315afb092cdcba116fd11d2c99`;
- Phase 4 report:
  `06b285ccbc9a860e2287c08312974aa5001ea878d8246411035e0a112d4f9f32`;
- original failed MATLAB pre-run log:
  `2347ef6986b0e28aa173ccb62adf6b1fb9665b5e07eedc2a5e716d8726fd14d0`;
- interrupted-output archive:
  `5c02c0d6f49d6fdb0b0173f2c76e01192eca29662a4fc31baedb4a66a677cf14`.

The Phase 5 failure remains a valid historical failure and was not rewritten
as a pass.

## 9. Protocol audit

No scientific or protocol element changed. In particular, Phase 5A did not
change the scientific question, base configuration, factor levels, mode-search
ladder, ceiling, thresholds, diagnostic grid, diagnostic/operational
scheduling, deferral behavior, acceptance rules, event taxonomy, telemetry
schema, preserved scalar values, or expected topology hash.

No enforcement-disable or bypass mechanism was added. No production arm was
started.

## 10. Readiness decision

The immutable baseline is now repository-owned, tracked, exact-byte verified,
separate from mutable output, validated before any arm can start, and covered
by lifecycle and existing regressions.

**READY TO RESTART PHASE 5 PRE-RUN GATE**
