# WP15 / WP21 — Precision requalification obligation
READ_ONLY_INDEPENDENT_DELTA_AUDIT — SPECIFICATION ONLY, NOT EXECUTED

    NEW_PRECISION_QUALIFICATION_REQUIRED = YES

## Why

The Phase-2B negative result was produced by one mechanism — a factor-1e5 value jump when a
density crosses ρ = 0.1 under Eq. (4). Any amendment that removes that jump removes the
mechanism, so the Phase-2B artifact can neither reject nor authorise float32 storage under an
amended evaluator. It tested a different instrument. This holds for Eq. (4a) and for any
continuous replacement.

There is nothing stale to invalidate: `production_preflight.m` requires
`analysis/iteration_efficiency_phase2a/validation_outputs/olhoff_new_trajectory_precision_qualification.json`,
and that file **does not exist** (directory verified: `README.md`, `observer_proposed.mat`,
`observer_unit.mat`, `observer_yuksel.mat`, `offline_topology_replay.json`). The gate fails
closed today, correctly.

## Minimum required scope

The qualification must compare **double against single storage of the same trajectory**, and
must prove identity — not similarity — for every item below, at every frozen q level:

| # | invariant | required outcome |
|---|---|---|
| 1 | pointwise `Q_E1`, `Q_E2`, `Q_E3` at every state | bounded, with the bound reported per evaluator |
| 2 | `volume_pass`, `topology_pass`, `hard_gate_pass` at every state | **identical** |
| 3 | `b_ref` | **identical** |
| 4 | `Q_ref_E1/E2/E3` | bounded, bound reported |
| 5 | `B_meas` and `certification_tail_truncated` | **identical** |
| 6 | acceptance `A_q(k)` at every state, q ∈ {0.98, 0.99, 0.995} | **identical**; report the count of differing states |
| 7 | `k_enter` at 0.98 / 0.99 / 0.995 | **identical** |
| 8 | `k_cert` at 0.98 / 0.99 / 0.995 | **identical** |
| 9 | final status under the frozen precedence | **identical** |
| 10 | OAT persistence `P = 50` and `P = 200` | **identical** `k_enter` |
| 11 | minimum observed decision margin, and its ratio to the measured storage perturbation | reported, not asserted |

Item 11 is added by this audit. Phase 2B reported margins
(`QUALITY_DECISION_EQUIVALENCE.csv`) but the amendment discussion never compared them to the
perturbation. That comparison is the only thing that converts "the error is small" into
"the decisions cannot move", and it must appear in the artifact.

## Is a new optimizer run required? — WP21, two separate answers

**(a) To JUSTIFY the mass-law amendment: NO.** A post-hoc evaluator change is fully
assessable from stored densities. This audit reproduced every Phase-2D stability claim, and
found the defect that governs the verdict, without running any optimizer.

**(b) To QUALIFY float32 storage and the downstream endpoint machinery: YES.** No density
trajectory of reference length exists anywhere in the repository (longest: 1601 snapshots;
required: 3200), and the four reference-length artifacts retain quality arrays only. The
requalification cannot be done offline.

## Minimum adequate regeneration experiment (specified, not run)

Only if and when a mass law has been settled. Do not run it under Eq. (4a) as currently
amended — finding D1 makes any such result uninterpretable.

| parameter | value | reason |
|---|---|---|
| mesh | 96×12 | even `nely`, 8:1 aspect, matches the Phase-2B artifact so results are directly comparable; `a_sig` = 1.44 elements makes the topology gate STRICTER_THAN_PRODUCTION |
| method | Olhoff, unmodified protected `olhoffOptStabilized.m` | the only method whose storage precision is in question |
| horizon | **3200 updates** (= `B_ref`), single run | reference must be establishable; `b_ref` was 2200 on the existing artifact |
| trajectory role | a **reference** run, separate from any measurement run | `reference.trajectory_separate_from_measurement: true` |
| precision captures | `rho_snapshots` in **float64** *and* the float32 image of the same states, at every one of the 3200 updates | the double trajectory is what is missing; do not reconstruct it by bracketing |
| fields to save | per update: full density vector (double), `hard_gate` inputs, native `omega`, `nInner`, `vol` | enough to recompute every item 1–11 offline under any mass law |
| frozen constants | `P = 100`, `L_ref = 500`, `ε_ref = 1e-3`, `B_ref = 3200`, `B0 = 3200`, `q ∈ {0.98, 0.99, 0.995}`, `volfrac = 0.5`, `A_sig = 0.01` | unchanged from the contract |
| outputs | the eleven invariants above, per q, double vs single; plus the decision-margin table | |
| additional requirement added by this audit | at every state, the **modal-participation diagnostic**: the fraction of mode-1 kinetic energy carried by elements with ρ ≤ 0.1, for each of E1/E2/E3 | this is the only diagnostic that would have caught D1, and it must not be omitted again |

Retaining double densities at 3200×1152 costs ≈ 29 MB. There is no reason to store single
only.

**This audit does not run it.**
