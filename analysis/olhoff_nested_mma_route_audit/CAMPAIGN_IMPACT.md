# WP18 — What would change if nested MMA became the principal Olhoff route
NESTED-MMA ROUTE AUDIT — IMPACT ASSESSMENT ONLY, NOTHING IMPLEMENTED

## The blocking structural fact

The iteration-efficiency study's Olhoff comparator is
`analysis/olhoff_stabilization_audit/olhoffOptStabilized.m`, bound to profile
`olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1`. That file **hard-codes the
LP route**:

    [drho,st]=innerLoopLP(ctx); tInner=toc(ti);

There is no `innerSolver` switch. Selecting MMA is not a configuration change; it requires
editing the source.

That file is a `protected_numerical_sources` entry in
`analysis/iteration_efficiency_phase2a/implementation_provenance.json`
(sha256 `95240cf60f82b40f8e5e892b9eea9b20a8fd3744b5eca6fdfc8dde2698d82aec`), verified by
`ie2a.verify_provenance`. Every prior audit phase — 2B, 2C, 2D, 2E, 2F — re-verified that
hash as unchanged. Changing it invalidates that chain.

## Artifacts and semantics that would have to change

| # | artifact | change required |
|---|---|---|
| 1 | `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` | add an inner-solver selector, or create a second protected runner | 
| 2 | `implementation_provenance.json` → `protected_numerical_sources` | new/changed sha256 entry |
| 3 | `iteration_efficiency_contract.json` → `methods[].profile_id` for Olhoff | new profile identity; the current id encodes `move005_0025_fixed1600` |
| 4 | profile manifest / `profile_sources` | new hash entry |
| 5 | `+ie2a/production_preflight.m` `profile_bindings` check | hard-codes the three profile id strings; must be updated |
| 6 | `+ie2a/frozen_contract_sha256.m` | contract digest changes |
| 7 | **iteration accounting** (`ITERATION_ACCOUNTING_SPEC.md`, `+ie2a/account_iterations.m`) | Olhoff currently contributes *outer* updates. MMA adds a second, non-commensurable inner axis (mean 142.8 sub-iterates/outer). The spec's per-method unit definition and the `k_native` convention both need revision. |
| 8 | status taxonomy | `GENERIC_LP_ITERATION_LIMIT_ONLY` is LP-specific and would be replaced by an MMA inner-cap class; `contract.statuses.known_olhoff_backend_subclass` changes |
| 9 | reference budget `B_ref = 3200` | the MMA route's outer trajectory has different maturation dynamics; `b_ref` would have to be re-established |
| 10 | measurement budget `B0` / `B_meas` | the Olhoff `B0 = B_ref = 3200` saturation currently relies on the fixed-1600 LP profile |
| 11 | trajectory storage | `olhoffOptStabilized` stores `snapshots` as **single**; an MMA runner would need the same or better, and the entire Phase-2B/2E precision qualification would have to be redone for the new trajectory |
| 12 | timing methodology (`TIMING_SPEC.md`) | at ~400× per outer iteration the timing decomposition and the replay plan change character completely |
| 13 | scaling figures (`SCALING_AND_FIGURE_SPEC.md`) | fits over `k_enter`/`k_cert` would be over a different trajectory family |
| 14 | paper tables (`PROPOSED_TABLE_LAYOUTS.md`) | a work *vector* replaces a scalar count for Olhoff, breaking the current one-number-per-method layout |
| 15 | frozen prior evidence | `contract.frozen_prior_evidence` and the 8 stored Olhoff production trajectories were all produced by the LP runner and would become non-comparable |

## Cost of switching, plainly

Items 1–6 are mechanical. Items 7, 8, 11, 14 are **methodological**: they change what an
"iteration" means for one of three methods, which is the study's central measured quantity.
Item 11 additionally re-opens the entire precision-qualification chain that Phases 2B–2F have
been working through.

And the campaign would have to be **re-run from scratch for Olhoff at all nine meshes**,
because the eight stored trajectories are LP-route artifacts. On the measured MMA cost that
is on the order of hundreds of CPU-hours (see `ITERATION_ACCOUNTING_OPTIONS.md`).

## The asymmetry worth stating

Adopting MMA as a **secondary reported variant** at one or two meshes costs items 1–2 only,
requires no contract change, no accounting change, and no campaign re-run — because a
secondary variant is reported beside the principal result rather than replacing it. It would
deliver the one thing the LP route structurally cannot: a measured inner/outer work
decomposition and the multiplicity-cost result (`N=1` 93.4 vs `N=2` 147.8 sub-iterates,
p = 1.1e-10).

The scientific value of the MMA evidence is therefore almost entirely obtainable without
paying any of the switching cost.
