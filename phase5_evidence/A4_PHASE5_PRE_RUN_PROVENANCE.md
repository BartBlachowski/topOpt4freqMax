# A4 Phase 5 Pre-Run Provenance

- Recorded UTC: `2026-07-24T08:44:47Z`
- HEAD: `3542a2dc48201fda48dd05146152b2bcdccc8d30`
- Authoritative base config:
  `examples/Revision_v1/a4_ss_400x50_base.json`
- Base config hash expected by the implementation: `fnv1a32_c141e407`
- Preserved `a4_topology_Ninf.csv` SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`
- Phase 3 `a4_topology_inf.csv` SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`
- Pre-run archive:
  `phase5_evidence/a4_phase5_prerun_20260724T084447Z.tar.gz`
- Archive SHA-256:
  `90c8042504f51a55a710f1a3b7f4570c46d9751d78fd8789173573db2cb75f02`

## Working-tree state before the pre-run gate

```text
 M examples/Revision_v1/a4_eigenpair_refresh.m
?? A4_RECOVERY_PHASE4_BLOCKER_RESOLUTION_REPORT.md
?? scripts/revision_v1/a4_validate_frozen_identity.m
?? scripts/revision_v1/test_a4_phase4.m
```

These are the Phase 4 blocker repairs and report. They are intentionally
preserved in the archive before Phase 5 execution. No Phase 3 report was
modified.
