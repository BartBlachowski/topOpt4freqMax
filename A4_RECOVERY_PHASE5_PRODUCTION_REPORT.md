# A4 Recovery Phase 5 — Final Production Rerun Report

## Outcome

**PHASE 5 PRE-RUN GATE PASSED**

The clean default campaign executed all five required arms:

`N = {Inf, 50, 10, 5, 1}`

The run was not resumed from Phase 3 or from the interrupted Phase 5 output.
The repaired driver produced every arm, completed the frozen-arm gate before
starting finite arms, completed the diagnostics-off and finite-arm replays, and
wrote the full production artifact set.

Section 8.5 campaign verdict: **COMPLETE**.

No H0/H1 or other final scientific decision is emitted in Phase 2.

## Controlled pre-run gate

Before production:

- immutable reference:
  `examples/Revision_v1/reference/a4/a4_topology_Ninf.csv`;
- expected and actual SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`;
- mutable production directory: empty;
- produced path:
  `examples/Revision_v1/output/a4/a4_topology_inf.csv`;
- reference and produced paths: distinct;
- preserved Phase 3, Phase 4, Phase 5 failure, and Phase 5A evidence:
  unchanged.

The ordered gate results were:

| Order | Suite/audit | Passed | Failed |
|---:|---|---:|---:|
| 1 | `test_a4_phase5a` | 12 | 0 |
| 2 | `test_a4_phase4` | 8 | 0 |
| 3 | `test_a4_phase1` | 10 | 0 |
| 4 | `test_a4_phase2` | 15 | 0 |
| 5 | `test_a4_pipeline` | 28 | 0 |
| 6 | `test_a4_refresh` | 22 | 0 |
| 7 | `test_a4_classifier` | 17 | 0 |
| 8 | independent configuration/single-factor audit | 19 | 0 |
|  | **Total** | **131** | **0** |

Combined gate log:
`phase5_evidence/a4_phase5_prerun_restart_20260727.log`

Log SHA-256:
`4df04c43ba2da0d56926c7c6b318eabe682c48995093a4724c4932f370bf93b8`

The configuration audit confirmed that the five arms differed only in
`domain.load_cases[0].loads[0].update_after`.

## Production provenance

- production log:
  `phase5_evidence/a4_phase5_production_restart_20260727.log`;
- production-log SHA-256:
  `4707ec3f58efd25d3b879bdaa8359cef8fd8ddc6428498a7186c0075ba2d53b6`;
- commit recorded by the artifacts: `3542a2d`;
- base configuration hash: `fnv1a32_c141e407`;
- result creation timestamp: `2026-07-27T08:18:08Z`;
- process exit code: `0`;
- measured production runtime recorded by the acceptance gate:
  `34992.798483125 s`;
- predeclared runtime estimate: `43200 s`;
- runtime-order acceptance check: PASS.

No configuration, factor level, threshold, mode-search constant, diagnostic
grid, telemetry field, acceptance rule, or expected reference value was
changed for this rerun.

## Per-arm results

Arm status below uses only the Phase 2 acceptance vocabulary.

| N | Terminal state | Iterations | omega1 tracked | omega1 min | omega1 threshold | MAC to Phi0 | j* | omega1/omega2 gap | Scheduled/effective/deferred | Deferral fraction | Max window | Max selected index | Warnings | Degenerate | Arm status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|---|
| Inf | iteration cap | 2000 | 159.5656269933 | 159.5656269933 | 162.4676914825 | 0.9996284251 | 1 | 67.3726750257 | 0/0/0 | 0.000000 | 320 | 49 | W-2, W-5 | no | **ACCEPTED WITH WARNING** |
| 50 | converged | 540 | 159.6012966989 | 159.6012966989 | 162.4787941874 | 0.9996330849 | 1 | 65.9739559816 | 10/10/0 | 0.000000 | 320 | 49 | W-2, W-5 | no | **ACCEPTED WITH WARNING** |
| 10 | converged | 536 | 159.1229067497 | 98.5534933196 | 162.9133888903 | 0.9947725157 | 7 | 0.0002540607 | 53/3/50 | 0.943396 | 320 | 40 | W-1, W-2, W-5 | no | **ACCEPTED WITH WARNING** |
| 5 | converged | 1173 | 158.6727361063 | 158.6727361063 | 162.7405940415 | 0.9993522011 | 1 | 60.5852733544 | 234/1/233 | 0.995726 | 320 | 20 | W-1, W-2, W-5 | no | **ACCEPTED WITH WARNING** |
| 1 | converged | 1040 | 157.6328844741 | 157.6328844741 | 161.4558282387 | 0.9997471670 | 1 | 107.1096269987 | 1040/2/1038 | 0.998077 | 320 | 19 | W-1, W-2, W-5 | no | **ACCEPTED WITH WARNING** |

The longest consecutive deferral runs were 0, 0, 49, 233, and 1035 for
`N = Inf, 50, 10, 5, 1`, respectively. `REFERENCE_UNAVAILABLE` deferred and
continued; it did not terminate any finite arm.

The legacy scientific-classification evidence remains separate from arm
acceptance. In particular, the preserved artifacts record the open B4 issue
for `N = Inf` and tracked index 7 for `N = 10`; neither changes the Phase 2
measurement-integrity status above.

## Screening and event evidence

| N | Unique events | Diagnostic events | Operational attempts | Coincident/shared searches | Selected | Reference unavailable | E-0 | E-1 | E-2a | E-2b | E-3 | E-4 | E-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Inf | 25 | 25 | 0 | 0 | 22 | 3 | 19 | 3 | 3 | 0 | 0 | 3 | 0 |
| 50 | 23 | 20 | 10 | 7 | 20 | 3 | 17 | 3 | 3 | 0 | 0 | 3 | 0 |
| 10 | 62 | 20 | 53 | 11 | 7 | 55 | 4 | 3 | 3 | 52 | 0 | 3 | 0 |
| 5 | 239 | 23 | 234 | 18 | 1 | 238 | 1 | 0 | 3 | 235 | 0 | 3 | 0 |
| 1 | 1040 | 23 | 1040 | 23 | 2 | 1038 | 2 | 0 | 3 | 1035 | 0 | 3 | 0 |

Every operational attempt and diagnostic event is preserved in the arm
checkpoint JSONL files and in `a4_screening_events.json`. Coincident events
used one shared search and are marked `event_kind = "both"`.

No production event carries a B3 label. The sole output text occurrence of
the string `B3` is the fixture-suite assertion that E-1 is never E-2/B3.

## Frozen-arm and replay gates

- V-P2-2 passed before any finite arm ran.
- All six declared frozen scalars were exactly identical.
- Produced N=Inf topology SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`.
- Immutable reference SHA-256:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`.
- The non-blocking in-memory/CSV diagnostic reported a maximum absolute
  difference of `5.5511151231257827e-16`; this did not affect the byte-level
  fixture gate.
- Diagnostics-off N=Inf replay passed exact iteration, trajectory, endpoint,
  and topology identity.
- Finite N=50 replay completed 540 iterations and passed exact iteration,
  trajectory, endpoint, topology, and screening identity.

## Artifact audit

The production directory was empty before the run. Therefore every file now
present was created by this clean Phase 5 execution; no interrupted or Phase 3
output was reused.

Audit results:

- five arms exist in `a4_result.json`;
- five non-empty MATLAB v7.3 arm checkpoint files exist;
- aggregate MATLAB result exists and is non-empty;
- history rows: `5289`, equal to `2000 + 540 + 536 + 1173 + 1040`;
- unique screening events: `1389`, equal to `25 + 23 + 62 + 239 + 1040`;
- candidate rows: `430600`, reconciled by arm:
  `2120 + 2040 + 18000 + 76200 + 332240`;
- replay histories/events: `540/23`, matching the original N=50 arm;
- both manifests list the same 28 required artifacts in the same order;
- every manifest-listed file exists and is non-empty;
- output directory contains 46 current files including checkpoint and JSONL
  evidence not duplicated in the Section 10 manifest list;
- all five current-run topology CSVs exist;
- the mutable output directory contains no hidden
  `a4_topology_Ninf.csv` baseline prerequisite;
- immutable baseline path, expected hash, actual hash, mutable output path, and
  produced topology path are recorded in the result and stage-result JSON;
- commit SHA and base-config hash are recorded;
- all nine required PNG figures are valid, non-empty PNG files;
- both required Markdown tables exist and are non-empty;
- production and validation evidence reports are complete.

Current-run topology SHA-256 values:

| N | SHA-256 |
|---|---|
| Inf | `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806` |
| 50 | `35443db180187fa31cc24d4568dd71f574a15b99bf6bf4ceb3339d77f0062800` |
| 10 | `3eba65a2d0593f8ecfe69fe57b8ad7f8ca39a538dd68ac6f983927cdf23a8223` |
| 5 | `18115deb773d3c5abbad33c34099338ea0bd58851a0a1734bfd41febf722a794` |
| 1 | `11c6e132dce23a282d03c1b6a00291421ccbb9970c92bc093e94e67a5d24fe8a` |

## Evidence preservation

The historical evidence remains unchanged:

- `A4_RECOVERY_PHASE3_EXECUTION_REPORT.md`:
  `99deb76d17e6875cd2d62310ea377f408a1373315afb092cdcba116fd11d2c99`;
- `A4_RECOVERY_PHASE4_BLOCKER_RESOLUTION_REPORT.md`:
  `06b285ccbc9a860e2287c08312974aa5001ea878d8246411035e0a112d4f9f32`;
- `phase5_evidence/A4_PHASE5_PRE_RUN_FAILURE_20260727.md`:
  `c0f0928bd0a2c40590f12179936b548b38e435c9dd8597999fd756ed80bcdf30`;
- `A4_RECOVERY_PHASE5A_BASELINE_REPAIR_REPORT.md`:
  `d2bbdc37d05bd6a9bceae8981f21b1540b7f9136c2a4b129f2d9804e9d780fa4`;
- interrupted-output archive:
  `5c02c0d6f49d6fdb0b0173f2c76e01192eca29662a4fc31baedb4a66a677cf14`;
- original failed MATLAB pre-run log:
  `2347ef6986b0e28aa173ccb62adf6b1fb9665b5e07eedc2a5e716d8726fd14d0`.

The earlier Phase 5 pre-run failure remains valid historical evidence and was
not rewritten as a pass.

## Readiness and scope decision

Phase 2 now satisfies its production specification: every Section 11 item and
V-P2-1 through V-P2-9 passed, all five arms have valid measurement-integrity
statuses, and the Section 8.5 verdict is **COMPLETE**.

The following remain deferred to later scientific-classification work exactly
as specified: M-1, M-2, M-3, M-7, and M-9, including the B4 attribution issue,
limit-cycle measurement/classification, the baseline scientific reference
required by M-9, and any final H0/H1 or manuscript-level conclusion.

