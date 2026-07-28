# A4 Recovery Phase 4 — Blocker Resolution Report

**Date:** 2026-07-24  
**Scope:** Defects P3-1 and P3-2 only.  
**Production campaign:** Not executed in this phase.

## 1. Resolution summary

Both Phase 3 implementation blockers are repaired and covered by regressions.

- **P3-1:** V-P2-2 now makes its stop decision from two separate specified
  invariants: exact identity of all six §9.2 scalar values and SHA-256 identity
  of the produced topology CSV. The full-precision in-memory vector versus
  lossy CSV reload comparison is retained only as an explicitly non-blocking
  numerical diagnostic.
- **P3-2:** all result fields needed by normal and HALTED writers are initialized
  before the arm loop. Both HALTED branches persist accumulated artifacts and
  halt metadata through one helper before throwing the original halt. Reports,
  stage result, and matched manifests are refreshed on the partial-arm path.

The scientific protocol, adaptive search, diagnostic grid, thresholds, refresh
treatment, failure taxonomy, acceptance rules, telemetry schema, configuration,
and preserved reference values were not changed.

## 2. Exact root causes

### P3-1

`localFrozenIdentity` compared the full-precision `arm.topology` vector with
`readmatrix(a4_topology_Ninf.csv)`. The CSV was written with 15 significant
digits, which is insufficient for general IEEE-754 double round-trip identity.
That non-specified comparison could fail even when the produced topology CSV
was byte-for-byte identical to the preserved file.

The specified topology invariant is the CSV SHA-256:

`9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`

### P3-2

The HALTED branch invoked `localWriteArtifacts` before
`res.acceptance_checks` existed. Report generation called
`localSection11Lines`, which dereferenced that absent field and raised
`Unrecognized field name "acceptance_checks"`. This secondary writer exception
prevented report/manifest completion and replaced the true
`a4:FrozenBitIdentityFailed` reason.

## 3. Exact changes and defect-to-code mapping

| Defect | Code change |
|---|---|
| P3-1 | Added `a4_validate_frozen_identity`. It preserves `isequal` checks for `omega1_tracked`, `final_design_change`, `iterations`, `mode_index_jstar`, `mac_to_phi0`, and `omega1_omega2_gap`; computes SHA-256 from the produced topology CSV; and sets `pass = exact_scalar_identity && topology_file_identity`. |
| P3-1 | The driver writes `a4_topology_inf.csv` before V-P2-2 and passes that produced file to the validator. |
| P3-1 | The in-memory/CSV comparison now lives under `numerical_diagnostics`, carries `decision_role = "non-blocking"`, and cannot fail the gate. |
| P3-2 | The driver initializes `acceptance_checks`, `run_verdict`, `scientific_decision`, `decision`, and `halt` with writer-safe shapes before the arm loop. |
| P3-2 | `localPersistAndHalt` records the true identifier/reason, persists accumulated results, events, candidates, histories, reports, and manifests, then throws the original halt identifier. A writer failure is attached as a cause and cannot replace the original halt. |
| P3-2 | Artifact indexes are invalidated at the start of a rewrite; both manifests are then regenerated from the artifact files that actually exist and contain the identical file list. |
| P3-2 | Production report routing now uses the intended production level set, so a production run halted before all five arms finish still refreshes the repository-level production reports. |
| P3-2 | The stage result, result JSON/MAT, and recovery report serialize the halt record and original halt reason. |

## 4. Modified and new files

- Modified: `examples/Revision_v1/a4_eigenpair_refresh.m`
- New: `scripts/revision_v1/a4_validate_frozen_identity.m`
- New: `scripts/revision_v1/test_a4_phase4.m`
- New: `A4_RECOVERY_PHASE4_BLOCKER_RESOLUTION_REPORT.md`

No configuration, constants block, scientific specification, Phase 3 execution
report, production artifact, or preserved reference file was modified.

## 5. Executed tests and results

All executions used MATLAB R2025b. No complete five-arm production sweep or
production optimization probe was run.

| Test | Result |
|---|---:|
| `test_a4_phase4` | **8/8 passed** |
| `test_a4_phase1` | **10/10 passed** |
| `test_a4_phase2` | **15/15 passed** |
| `test_a4_pipeline` | **28/28 passed** |
| `git diff --check` | **passed** |
| MATLAB Code Analyzer on the three affected files | **No parse errors**; only style/stale-suppression notices |

The Phase 4 suite uses the real V-P2-2 helper and a real tiny-mesh invocation of
`a4_eigenpair_refresh`; the HALTED test is not an isolated artifact-writer mock.

## 6. V-P2-2 evidence

### Former false positive now passes

The positive regression:

1. loads the authoritative preserved topology values;
2. moves representable fractional values by four ulps only where their
   `%.15g` serialization remains unchanged;
3. writes that full-precision vector with `%.15g`;
4. confirms that the CSV reload differs from the in-memory vector;
5. confirms the produced CSV SHA-256 is the preserved expected value; and
6. confirms V-P2-2 passes.

Observed assertion: **PASS** —
`matching topology SHA passes despite several-ulp CSV reload loss`.

Independent file evidence:

- `a4_topology_Ninf.csv` SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`
- Phase 3 `a4_topology_inf.csv` SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`

### Real mismatches still fail

- A topology CSV with one altered value has a different SHA-256 and fails
  V-P2-2 while all exact scalar invariants remain true.
- An arm with `omega1_tracked` changed by one ulp fails V-P2-2 while topology
  file identity remains true.

Both negative regressions passed.

## 7. HALTED-path evidence

The Phase 4 test derives a 40×5, four-iteration config from the real base config,
runs the real driver with levels `[Inf, 2]`, and leaves V-P2-2 enforcement
enabled. The tiny frozen arm necessarily mismatches the production §9.2
reference and forces HALTED before completion of the arm loop.

The test confirmed:

- the propagated exception remains `a4:FrozenBitIdentityFailed`;
- neither `acceptance_checks` nor `Unrecognized field` appears in the
  propagated error;
- the recovery report and validation report are non-empty;
- `a4_manifest.json`, `a4_stage_manifest.json`, and `a4_stage_result.json` are
  non-empty and refreshed;
- pre-seeded stale manifests are replaced;
- both manifests contain identical file lists and `run_verdict = HALTED`;
- the MAT result contains the one accumulated arm and its screening events,
  candidate telemetry, and iteration histories;
- the result JSON and recovery report contain the original halt identifier and
  reason; and
- the event JSON, candidate CSV, and history CSV are non-empty.

This satisfies the repaired failure-path behavior required by §4.5 and
checklist item I-7.

## 8. Protocol and scientific-constant audit

No changes were made to:

- `m0`, `W`, `M_max`, any screen/stability/tie threshold, or grid `G`;
- diagnostic scheduling or read-only behavior;
- operational refresh or deferral behavior;
- the E-class taxonomy, warning conditions, or arm/run acceptance rules;
- candidate/event/history telemetry fields;
- the base configuration or factor levels;
- `delta`, the scientific question, or interpretation rules;
- any preserved §9.2 scalar; or
- the expected topology SHA-256.

M-1, M-2, M-3, M-7, and M-9 remain unimplemented and out of scope.

## 9. Remaining work

Phase 5 must restart and complete the full `{Inf, 50, 10, 5, 1}` production
campaign, including the production non-perturbation and finite-arm replay
validators, complete artifact reconstruction/tracking checks, and every
production-dependent Section 11 item. Phase 2 is not COMPLETE on the basis of
this repair phase.

## 10. Readiness decision

The two blocking implementation defects identified by Phase 3 are repaired,
their former failure modes are covered by executable regressions, all mandated
limited-validation suites pass, and no scientific protocol element changed.

**READY FOR PHASE 5 PRODUCTION RERUN**
