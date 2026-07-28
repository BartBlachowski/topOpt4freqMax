# A4 Recovery Phase 5 — Production Validation

## Validation verdict

All production validators V-P2-1 through V-P2-9 passed on the clean Phase 5
rerun. The Section 11 acceptance checklist passed without failure, and the
campaign verdict is **COMPLETE**.

## Validator evidence

| Validator | Result | Production evidence |
|---|---|---|
| V-P2-1 | **PASS** | Clean diagnostics-off N=Inf replay: iterations, trajectory, endpoint, and topology all bit-identical. Recorded replay time `231.005090375 s`. |
| V-P2-2 | **PASS** | Exact identity of all six frozen scalars and SHA-256 identity of produced N=Inf topology against the immutable fixture. |
| V-P2-3 | **PASS** | Production window recovery includes N=Inf iteration 25, selected index 49 with MAC `0.9775288450`, and iteration 30, selected index 37 with MAC `0.9663501395`; both confirmed. |
| V-P2-4 | **PASS** | Screening symmetry fixture suite passed on the frozen implementation. |
| V-P2-5 | **PASS** | Adaptive ladder determinism fixture passed; production rungs and mandatory expansions are recorded per event. |
| V-P2-6 | **PASS** | Clean full N=50 replay: 540 iterations; iterations, trajectory, endpoint, topology, and screening replay all exactly identical. Recorded replay time `175.799207625 s`. |
| V-P2-7 | **PASS** | Event-class fixture suite passed; production recorded E-0, E-1, E-2a, E-2b, and E-4, with zero E-3/E-5 and no production B3 label. |
| V-P2-8 | **PASS** | Base hash `fnv1a32_c141e407` matched, and the negative-hash fixture passed. |
| V-P2-9 | **PASS** | Phase 1 regression remained 10/10 during the ordered pre-run gate; the embedded production fixture suite also passed 14/14 and included the V-P2-9 check. |

### V-P2-2 exact frozen identity

| Invariant | Expected | Produced | Exact |
|---|---:|---:|:---:|
| `omega1_tracked` | 159.56562699328325 | 159.56562699328325 | yes |
| `final_design_change` | 0.0030349036393301221 | 0.0030349036393301221 | yes |
| `iterations` | 2000 | 2000 | yes |
| `mode_index_jstar` | 1 | 1 | yes |
| `mac_to_phi0` | 0.99962842513639028 | 0.99962842513639028 | yes |
| `omega1_omega2_gap` | 67.372675025734623 | 67.372675025734623 | yes |

Topology hashes:

- immutable reference:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`;
- produced N=Inf:
  `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`.

The reference and output paths are distinct, and the reference path is outside
the mutable output directory.

## Count reconciliation

| N | Iterations/history rows | Events | Candidate rows | Scheduled operational attempts | Effective | Deferred |
|---|---:|---:|---:|---:|---:|---:|
| Inf | 2000 | 25 | 2120 | 0 | 0 | 0 |
| 50 | 540 | 23 | 2040 | 10 | 10 | 0 |
| 10 | 536 | 62 | 18000 | 53 | 3 | 50 |
| 5 | 1173 | 239 | 76200 | 234 | 1 | 233 |
| 1 | 1040 | 1040 | 332240 | 1040 | 2 | 1038 |
| **Total** | **5289** | **1389** | **430600** | **1337** | **16** | **1321** |

The aggregate CSV/JSON counts agree with the arm checkpoint JSONL files.
Every arm history length equals its true iteration count. The N=50 replay adds
540 histories and 23 events and is kept separate from the production-arm
totals.

## Artifact-only event reconstruction

These reconstructions read the production JSONL/candidate artifacts only.
MATLAB was not rerun to reproduce or reinterpret any event.

The four admissibility conditions encoded in every candidate row are:

1. `support_kinetic_fraction >= 0.5`;
2. `support_connectivity == true`;
3. `low_density_strain_fraction <= 0.5`;
4. `mac_prev >= 0.8`.

For each reconstruction below, exact per-candidate measured values, all four
boolean condition results, and the complete row-level rejection strings are
preserved in the cited JSONL and in `a4_candidate_telemetry.csv`. The
signature counts below partition the entire final candidate set without
omitting a candidate.

### E-1 reconstruction

Source:
`examples/Revision_v1/output/a4/a4_checkpoint_inf.mat.events.jsonl`,
N=Inf, iteration 20, event 8.

- kind: diagnostic;
- ladder rungs: `[20, 40, 80, 160]`;
- final window: 160;
- final candidate set: mode indices 1 through 160, 160 candidates;
- admissible candidates: 1;
- selected candidate: index 43;
- selected omega: `158.43937043126866`;
- selected MAC to previous reference: `0.99741169835800292`;
- selected support kinetic fraction: `0.91426057354508272` — pass;
- selected support connectivity: `true` — pass;
- selected low-density strain fraction: `0.0030194003354349261` — pass;
- selected MAC condition: pass;
- selected rejection reason: empty;
- stability: confirmed;
- stability MAC: `0.99999999999999933`;
- reference changed: no;
- event class: E-1;
- class reason: selected index 43 exceeds the old window 20.

Complete rejection partition:

- 158 candidates failed kinetic + strain + MAC;
- candidate 86 failed MAC only;
- candidate 43 passed all four conditions and was selected.

The partition is `158 + 1 + 1 = 160`.

### E-2 reconstruction

Source:
`examples/Revision_v1/output/a4/a4_checkpoint_inf.mat.events.jsonl`,
N=Inf, iteration 1, event 1.

- kind: diagnostic;
- ladder rungs: `[20, 40, 80, 160, 320]`;
- final window: 320;
- final candidate set: mode indices 1 through 320, 320 candidates;
- admissible candidates: 0;
- selected candidate: none;
- search outcome: `REFERENCE_UNAVAILABLE`;
- stability: not applicable;
- reference changed: no;
- deferral: no, because this is a diagnostic-only event;
- event classes: E-2a and E-4;
- class reason: no candidate passed the three physical-mode conditions;
- best-MAC candidate: index 1, but its support kinetic fraction was 0 and
  support connectivity was false.

Complete rejection partition:

- candidate 1 failed kinetic + support connectivity;
- candidates 2 through 320 failed kinetic + support connectivity + MAC;
- all 320 passed the strain condition;
- no candidate passed all four conditions.

The partition is `1 + 319 = 320`.

### Operational refresh reconstruction

Source:
`examples/Revision_v1/output/a4/a4_checkpoint_50.mat.events.jsonl`,
N=50, iteration 50, event 12.

This iteration was both a scheduled operational refresh and a diagnostic
grid point. The artifact records `event_kind = "both"` and one shared search.

- ladder rungs: `[20, 40]`;
- final window: 40;
- final candidate set: mode indices 1 through 40, 40 candidates;
- admissible candidates: 1;
- selected candidate: index 5;
- selected omega: `159.87120562344157`;
- selected MAC to previous reference: `0.99958051457424357`;
- selected support kinetic fraction: `0.96967366199212734` — pass;
- selected support connectivity: `true` — pass;
- selected low-density strain fraction:
  `0.000057734713717576887` — pass;
- selected MAC condition: pass;
- selected rejection reason: empty;
- stability: confirmed;
- stability MAC: `0.99999999999999989`;
- reference changed: yes;
- deferred: no;
- event class: E-0.

Complete rejection partition:

- 38 candidates failed kinetic + strain + MAC;
- candidate 18 failed MAC only;
- candidate 5 passed all four conditions and was selected.

The partition is `38 + 1 + 1 = 40`.

All three reconstructed events also record read-only proofs for design,
reference, matrices, and RNG state as bit-identical.

## Manifest and lifecycle validation

- `a4_manifest.json` and `a4_stage_manifest.json` contain identical 28-entry
  artifact lists in the same order.
- Every listed artifact exists and is non-empty.
- Nine required figures and two required tables exist.
- Five current-run topology files exist.
- Five production-arm checkpoint MAT files and the finite replay checkpoint
  exist and are valid non-empty MATLAB v7.3 files.
- The output directory was empty before production; no stale production file
  survived into this run.
- Moving or deleting mutable output cannot remove the immutable reference.
- The output directory contains no authoritative
  `a4_topology_Ninf.csv` prerequisite.
- Commit SHA, base-config hash, and immutable-baseline metadata are present in
  both result and stage metadata.

## Final validation decision

Phase 2 satisfies its production specification and its Section 8.5 campaign
verdict is **COMPLETE**.

This is a measurement-integrity conclusion only. M-1, M-2, M-3, M-7, and M-9
remain deferred, and no final H0/H1 or manuscript-level scientific conclusion
is emitted.
