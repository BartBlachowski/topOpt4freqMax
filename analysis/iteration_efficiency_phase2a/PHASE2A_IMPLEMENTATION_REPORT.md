# Phase-2A implementation report

Date: 2026-08-30  
Classification: preregistered harness implementation; no production campaign

## Outcome

The frozen harness is implemented and its non-production tests pass, but production is
correctly blocked. The protected `olhoffOptStabilized.m` stores every intermediate density
snapshot as `single`, whereas the frozen implementation requirement calls for lossless
`double` fields for new trajectories unless a short/no-solve qualification proves identical
topology decisions, `k_enter`/`k_cert`, and bounded E1/E2/E3 values. No such qualification
exists. Adding a double observer inside that protected source would change a protected hash.
This matches the frozen stop condition “a method runner cannot expose required state without
algorithm modification,” so Phase 2A does not improvise an authorization.

## 1. Starting repository state

- Branch: `benchmark-methodology-r2`
- HEAD: `632e9b01811845709de33f93051fd853373ed5e1`
- Starting status is recorded verbatim in `implementation_provenance.json`. It already
  contained the WP0A visualization work, frozen/audit directories, campaign evidence, and
  two tracked modifications. Those artifacts were preserved.

## 2. Files created and modified

Created under this isolated area:

- `iteration_efficiency_contract.json`, `implementation_provenance.json`, and this report;
- fail-closed `iteration_efficiency_campaign.m`;
- package engines in `+ie2a/`: contract/hash/provenance validation, reference phase,
  authoritative `B_meas`, common evaluator, exact-count topology, persistence, status,
  method accounting, quality-effort, timing plan/execution/summary, scaling, tables,
  observer, trajectory analysis/fingerprints, method runners, production orchestration,
  output isolation, and production preflight;
- `run_phase2a_tests.m` and independent `offline_topology_replay.py`;
- isolated validation and locked production output roots.

Modified outside the isolated area:

- `tools/Matlab/topopt_history_record.m`: one opt-in, write-only post-update observer
  dispatch. With no root-appdata observer installed, the historical path only evaluates an
  `isappdata` guard. The callback cannot return optimizer state.

Phase 2A did not modify `performance_comparison.m`, either frozen numerical method source,
the common evaluator, any methodology/audit directory, or frozen campaign evidence.

## 3. Frozen contract and methodology mapping

Contract SHA-256:
`46318e6c7e74fdfe9cc643afeef93bbd6edfc5a894e89269be9cc04f0f27697c`.

| Frozen rule | Implementation |
|---|---|
| nine meshes/method/profile bindings/frozen hashes | `validate_contract`, `verify_provenance` |
| causal sustained-floor reference, no cap fallback | `reference_phase` |
| `min(max(B0,b_ref+P-1),B_ref)` and tail truncation | `measurement_budget` |
| unchanged E1/E2/E3, robust minimum ratio | `evaluate_common`, `analyze_trajectory` |
| stable exact-count projection and physical component gate | `exact_count_binary`, `topology_metrics` |
| retrospective `k_enter`, prospective `k_cert` | `scan_persistence` |
| Proposed/Yuksel/Olhoff native accounting | `account_iterations` |
| status precedence and censoring | `classify_status` |
| serial one-thread, warmup + 3 fixed-horizon replays | `timing_replay_plan`, `run_timing_replays`, `timing_summary` |
| log-OLS, common support, LOO diagnostics | `fit_power_law`, `generate_scaling` |
| all-row reporting and absolute E1/E2/E3 fields | `generate_tables`, `quality_effort` |
| accepted-state topology presentation | existing `render_iteration_efficiency_topology_grid.m`, which calls only shared `tools/Matlab/renderTopologyDensity.m` |

The contract explicitly records E1 as the verified Proposed interpolation identity and E2/E3
as sharing the piecewise `x^6` mass law. It also records `b_ref`, `B_meas`, tail truncation,
the corrected F8 anchors, and the exact Olhoff backend subclass/status wording. Frozen prior
absolute-quality context is preserved over eight complete triples only: Olhoff over Proposed
6.2–8.5%, and Olhoff over Yuksel 5.9–7.7%; Olhoff 800x100 is excluded as unavailable.

## 4. Tests and evidence

MATLAB R2025b no-production suite:

- 12/12 offline/unit tests passed;
- the authoritative budget function passed all three methods over every admissible block
  endpoint `b_ref = 600:100:3200`, including `b_ref=B_ref`, monotonicity, determinism,
  bounds, method-blind behavior, and the 99-update truncated tail;
- causal reference tests include a later post-freeze improvement that does not move the
  first-passage endpoint;
- frozen 160x20 Olhoff endpoint reproduces E1/E2/E3 values from
  `common_evaluators.csv` (maximum E1/E2/E3 difference below `1e-7`);
- topology tests cover stable ties, both support footprints, strict equality at
  `A_sig=0.01`, harmless multiple sub-threshold islands whose aggregate exceeds 0.01,
  and disconnected support failure;
- persistence tests cover first/last possible windows, interrupted runs, P=50/100/200,
  and no look-ahead;
- all three accounting adapters, status precedence, timing plans, and scaling fits passed;
- all eight contract negative controls were rejected: changed P, missing mesh, E1-only,
  aggregate-area veto, `B_meas=B0`, changed `B_ref`, changed q levels, and changed profile.

Tiny solver smoke (not production): 40x5, three updates, Proposed and Yuksel. Observer OFF
versus ON returned bit-identical densities, frequencies, and iteration counts. The shared
renderer smoke passed equivalent geometry/naming/status behavior for Proposed, Yuksel, and
Olhoff and confirmed failed/unavailable results are not rendered as final topologies.

Read-only frozen-evidence replay:

- all eight available Olhoff final fields pass the repaired topology gate;
- exhaustive 640x80 replay: 1,067 states, 1,014 passing; aggregate detached median 64,
  p95 147, maximum 674 elements / 2.633% of solid volume;
- 800x100 remains `RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT`; no topology was fabricated;
- output is labelled `IMPLEMENTATION_VALIDATION_ONLY_NOT_NEW_SCIENTIFIC_RESULTS`.

`verify_provenance` rechecked 27 protected audit, normative, profile, and numerical files.
All hashes match. The nine-mesh production experiment was not started.

## 5. Timing and visualization boundaries

Discovery trajectories retain fields for offline evaluation; their observer-inclusive time
is explicitly not publishable. Fixed-horizon timing replays run separately after endpoints
are frozen and do not invoke common evaluators, topology gates, persistence scanning,
rendering/export, or trajectory disk I/O inside the timed call. Rendering is a post-analysis
step and reuses the shared topology renderer; no method-specific drawing semantics exist.

## 6. Pre-production gate and blocker

All implemented preflight checks pass except `olhoff_lossless_trajectory`. Production
authorization is separately absent by design. The campaign script therefore fails closed
before creating production results. No methodology value is missing or contradictory; the
blocker is faithful instrumentation of one protected runner under the frozen lossless-storage
rule.

To launch later, only after the blocker is resolved and an explicit review authorizes the
campaign: open `iteration_efficiency_campaign.m`, set `authorizationToken` to the
review-issued literal `AUTHORIZE_FROZEN_NINE_MESH_PRODUCTION_AFTER_REVIEW`, and press Run
(equivalently run `cd('analysis/iteration_efficiency_phase2a'); iteration_efficiency_campaign`
from MATLAB). In the current state this action remains blocked by preflight.

The nine-resolution campaign is **not ready to run manually** until the Olhoff trajectory
precision/instrumentation qualification is resolved and reviewed.

PHASE 2A BLOCKED — METHODOLOGY IMPLEMENTATION ISSUE
