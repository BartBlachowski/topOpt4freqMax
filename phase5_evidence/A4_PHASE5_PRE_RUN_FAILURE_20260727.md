# A4 Recovery Phase 5 — Pre-Run Gate Failure

**Recorded UTC:** 2026-07-27  
**Required outcome:** `PHASE 5 PRE-RUN GATE FAILED`  
**Production campaign:** Not started.

## Provenance and preservation

- HEAD: `3542a2dc48201fda48dd05146152b2bcdccc8d30`
- The prior interrupted Phase 5 output was moved, without deletion, to
  `phase5_evidence/a4_phase5_interrupted_output_20260727T075221Z/`.
- The same output plus the authoritative Phase 2–4 reports, Phase 4 repair,
  and Phase 4 regressions was archived as
  `phase5_evidence/a4_phase5_interrupted_20260727T075221Z.tar.gz`.
- Archive SHA-256:
  `5c02c0d6f49d6fdb0b0173f2c76e01192eca29662a4fc31baedb4a66a677cf14`
- `A4_RECOVERY_PHASE3_EXECUTION_REPORT.md` had no working-tree diff and was
  not rewritten.
- Preserved `a4_topology_Ninf.csv` and the Phase 3 produced
  `a4_topology_inf.csv` both have SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`.

The working tree was dirty before this attempt. Its in-scope changes consisted
of the Phase 4 driver repair, new Phase 4 validator and regression, the Phase 4
report, and the partial July 24 Phase 5 output/evidence. The complete
`git status --short --branch` state was captured in the execution record. The
production output directory is empty after moving the interrupted output.

## Gate execution

The independent configuration/single-factor audit passed 19/19.

The mandatory MATLAB gate was invoked with MATLAB R2025b in this order:

1. `test_a4_phase4`
2. `test_a4_phase1`
3. `test_a4_phase2`
4. `test_a4_pipeline`
5. `test_a4_refresh`
6. `test_a4_classifier`

`test_a4_phase4` failed before completing:

```text
Error using readmatrix
Unable to find or open
examples/Revision_v1/output/a4/a4_topology_Ninf.csv.

Error in test_a4_phase4 (line 24)
csvValues = readmatrix(preservedPath);
```

The exact baseline remains present and hash-correct in the preserved archive,
but it is absent at the path the Phase 4 suite requires. The MATLAB gate log is
`phase5_evidence/a4_phase5_prerun_tests_20260727.log`, SHA-256
`2347ef6986b0e28aa173ccb62adf6b1fb9665b5e07eedc2a5e716d8726fd14d0`.

Because the first mandatory suite failed, the remaining MATLAB suites were not
executed in this attempt. No failure was repaired, bypassed, or suppressed.
No production arm was launched.

## Decision

**PHASE 5 PRE-RUN GATE FAILED**

Per the Phase 5 instructions, execution stops before the production campaign.
