# WP18 — Collateral-drift audit
READ_ONLY_INDEPENDENT_DELTA_AUDIT

## Method

Three independent checks, not one.

1. **Filesystem.** `find . -newermt "2026-08-30 13:40" ! -newermt "2026-08-30 14:20" -type f`
   excluding the Phase-2D directory and `.git`.
2. **Hashes.** All six `protected_numerical_sources`, all `profile_sources`, all
   `audit_records` from `analysis/iteration_efficiency_phase2a/implementation_provenance.json`,
   plus all twelve contract-listed `normative_documents`, re-hashed against their recorded
   digests (`scripts/wp0_hashes.py`, `WP0_INTEGRITY_pre.json`,
   `WP18_NORMATIVE_DOC_HASHES.csv`).
3. **Semantics.** Every item in the WP18 list traced to the source that defines it.

## Result of check 1 — exactly two files touched outside the Phase-2D directory

    analysis/iteration_efficiency_study_design/ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md
    analysis/iteration_efficiency_study_design/QUALITY_EFFORT_SPEC.md

Both declared in `AMENDMENT_SCOPE_LEDGER.csv` and `NORMATIVE_CHANGE_LEDGER.csv`. Both
pre-amendment copies hash-match the declared `pre` digests; both live files hash-match the
declared `post` digests. Both diffs read in full: they touch only the native-identity
paragraph and the 0.429% agreement figure.

## Result of check 2 — no protected source altered

| group | entries | mismatches |
|---|---|---|
| `protected_numerical_sources` | 6 | **0** |
| `profile_sources` | 3 | **0** |
| `audit_records` | 6 | **0** |
| Phase-2D's own declared-unchanged list | 9 | **0** |
| Phase-2D's own `SHA256SUMS.txt` self-check | 28 files | **0** |
| contract-listed `normative_documents` | 12 | **2** (the two amended documents — see N1) |

`analysis/three_method_parametric_study/study_evaluate_design.m` is byte-identical to its
`preamendment_copies/` snapshot and still matches the contract's
`quality.source_sha256 = 22a1b974…`. The contract file itself still matches the hard-coded
`ie2a.frozen_contract_sha256() = 46318e6c…`.

## Result of check 3 — semantics

| WP18 item | unchanged? | where verified |
|---|---|---|
| topology definition | yes | `+ie2a/topology_metrics.m` unmodified; independently re-implemented and reproduced 1600/1600 |
| `A_sig` = 0.01, `a_sig_by_mesh` | yes | `contract.topology`, contract hash unchanged |
| aggregate-island semantics (`DIAGNOSTIC_ONLY`) | yes | same |
| volume gate, 1e-3 relative tolerance | yes | same |
| exact-count projection, index tie-break | yes | `+ie2a/exact_count_binary.m` unmodified; reproduced independently |
| `P` = 100, OAT 50/200 | yes | `contract.persistence` |
| `q` levels 0.98/0.99/0.995 | yes | `contract.quality.levels` |
| `B_ref` = 3200, `L_ref` = 500, `ε_ref` = 1e-3 | yes | `contract.reference` |
| `B_meas` formula | yes | `+ie2a/measurement_budget.m` unmodified; reproduced |
| reference semantics | yes | `+ie2a/reference_phase.m` unmodified; reproduced exactly (b_ref 2200/2100) |
| `k_enter`, `k_cert` definitions | yes | `+ie2a/scan_persistence.m` unmodified; reproduced exactly |
| status precedence, `fit_eligible` | yes | `contract.statuses` |
| iteration accounting | yes | `+ie2a/account_iterations.m` unmodified |
| timing methodology | yes | `contract.timing`; threads 1, serial, 3 repetitions |
| scaling methodology | yes | `contract.scaling`; `fit_power_law.m` unmodified |
| mesh sequence | yes | `contract.production_meshes`, nine meshes |
| method profiles | yes | all three `profile_id` bindings unchanged, `profile_sources` hashes match |

## Ruling

    COLLATERAL_DRIFT = NONE

No unrelated scientific change was made. The only non-drift consequence of Phase-2D's two
edits is that two contract-pinned `normative_documents` digests are now stale; that is
finding **N1**, a bookkeeping obligation for refreeze, not scientific drift.
