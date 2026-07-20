# A4 Recovery Phase 1 Report

**Scope:** verified C-3 endpoint correction and C-4 provenance correction only.  
**Optimization rerun:** **No.**  
**Original optimization:** commit `2c945de`, created `2026-07-19T22:25:38Z`.  
**Post hoc regeneration:** `2026-07-20T09:42:16Z`.

## Implemented corrections

### C-3 — threshold endpoint

`a4_endpoint_eval` now passes the configured `endpoint.rho_min` to a shared
volume-preserving threshold helper. The undeclared `1e-3` floor was removed.

The completed-run topologies stored in `a4_eigenpair_refresh_results.mat` were
reused to recompute only `omega1_thresholded`. The original tracked endpoints,
classifications, decision, iterations, and topologies were retained.

| Arm | Tracked before | Tracked after | Threshold before | Threshold after |
|---|---:|---:|---:|---:|
| `N=inf` | 159.56562699328325 | 159.56562699328325 | 26.519321448943305 | 162.46769148252696 |
| `N=50` | 159.60117294919709 | 159.60117294919709 | 26.480232770798800 | 162.47879418740942 |

Both corrected threshold modes are index 1. Their mass-weighted MAC values to
the reconstructed solid reference are 0.9997152267004009 and
0.9997265495193155, respectively.

### C-4 — deterministic hashing

FNV-1a multiplication is now performed in `uint64` and masked explicitly to
32 bits after every byte. This implements modulo-`2^32` wrapping without
MATLAB `uint32` saturation.

| Provenance value | Before | After |
|---|---|---|
| A4 base-config file | `fnv1a32_ffffffff` | `fnv1a32_c141e407` |
| A4 stage config struct | `fnv1a32_ffffffff` | `fnv1a32_1f5c3fff` |

Post hoc recovery metadata in the result and manifests records both original
and corrected hashes and explicitly states that no optimization was rerun.

## Modified source files

- `scripts/revision_v1/a4_endpoint_eval.m` — uses configured `rho_min`.
- `scripts/revision_v1/a4_volume_preserving_threshold.m` — shared threshold
  operation with explicit configured-floor semantics.
- `scripts/revision_v1/fnv1a32_bytes.m` — correct wrapping FNV-1a core.
- `scripts/revision_v1/a4_hash_file.m` — binary file hashing.
- `scripts/revision_v1/fnv1a32_canonical_struct.m` — canonical struct hashing
  for campaign-stage provenance.
- `examples/Revision_v1/a4_eigenpair_refresh.m` — uses the corrected file hash.
- `examples/Revision_v1/run_all_revision_experiments.m` — uses the corrected
  canonical-struct hash.
- `scripts/revision_v1/a4_threshold_endpoint_from_topology.m` — reconstructs
  only the threshold eigensolve from an existing topology.
- `scripts/revision_v1/recover_a4_phase1.m` — guarded, no-optimization artifact
  regeneration; aborts if any unauthorized result field changes.
- `scripts/revision_v1/test_a4_phase1.m` — focused C-3/C-4 regressions.

## Regenerated artifacts

- `examples/Revision_v1/output/a4/a4_eigenpair_refresh_results.mat`
- `examples/Revision_v1/output/a4/a4_result.json`
- `examples/Revision_v1/output/a4/a4_table.md`
- `examples/Revision_v1/output/a4/a4_manifest.json`
- `examples/Revision_v1/output/a4/a4_stage_result.json`
- `examples/Revision_v1/output/a4/a4_stage_manifest.json`

No topology, figure, pre-screen result, optimization history, classification,
or scientific decision was regenerated.

## Validation

`test_a4_phase1` passes 10/10 checks:

1. Empty FNV-1a reference: `811c9dc5`.
2. Known vector `a`: `e40c292c`.
3. Known vector `foobar`: `bf9cf968`.
4. Identical files reproduce identical hashes.
5. Different files produce different hashes.
6. Canonical struct hashing is independent of field insertion order.
7. Canonical struct hashing changes when content changes.
8. Thresholding preserves the requested solid count.
9. Thresholded voids equal configured `rho_min` exactly.
10. No undeclared `1e-3` floor remains.

Three real repository files now produce distinct hashes:

- A4 base config: `c141e407`
- `README.md`: `808f8d74`
- `topopt_freq.m`: `c9e4dcd4`

The stored `.mat` result was compared before and after restoring only the
authorized fields in memory. `isequaln` returned true for the complete result.
Tracked-frequency doubles and topology arrays are bitwise equal.

The topology CSV SHA-256 values are unchanged:

- `N=inf`: `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`
- `N=50`: `5d9bdd2108dc83b77c120268ed1f50bda0f41fe4e6d54668cb36ea29cd690715`

## Scientific impact

No optimization result, trajectory, topology, refresh event, stopping outcome,
classification, or A4 decision changed. Only the threshold endpoint and
hash-derived provenance changed.

## Remaining Recovery Phase 2 work

Not implemented in Phase 1:

- adaptive candidate windows;
- common-grid screening diagnostics;
- B3/reference-unavailability changes;
- any finite-`N` optimization rerun;
- any A4 methodological or hypothesis change.
