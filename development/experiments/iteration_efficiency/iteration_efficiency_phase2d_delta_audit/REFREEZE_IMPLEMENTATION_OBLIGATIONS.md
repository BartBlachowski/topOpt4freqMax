# WP19 — Implementation and refreeze obligations
READ_ONLY_INDEPENDENT_DELTA_AUDIT — AUDITED SPECIFICATION, NOTHING EXECUTED

This audit does **not** approve refreeze (see `PHASE2D_DELTA_AUDIT.md`, finding D1). The
obligations below are audited and recorded so that they are complete and correct whenever a
mass law is finally settled — whether Eq. (4a), a modally-filtered variant, or another
source-defined law. Items marked **[added by this audit]** are missing from Phase-2D's
`PHASE2A_IMPLEMENTATION_IMPACT.md`.

## Was it appropriate to leave the contract and evaluator pre-amendment?

**Yes.** `analysis/three_method_parametric_study/study_evaluate_design.m` is a
`protected_numerical_sources` entry and its digest is pinned as `quality.source_sha256`;
the contract's own digest is pinned in `ie2a.frozen_contract_sha256()`. Editing either
before independent approval would have broken Phase-2A/2B/2C provenance verification and
would have constituted the self-refreeze Phase 2D was forbidden. Verified: the frozen
evaluator is byte-identical to its `preamendment_copies/` snapshot, and the contract still
matches `46318e6c…`.

## Obligations, in dependency order

| # | obligation | file | gate that enforces it |
|---|---|---|---|
| 1 | apply the settled mass law to the normative evaluator | `analysis/three_method_parametric_study/study_evaluate_design.m` | `protected_numerical_sources` |
| 2 | update the evaluator digest | `iteration_efficiency_contract.json` → `quality.source_sha256` | `production_preflight` `evaluator_hash` |
| 3 | update `quality.evaluators[E2].mass` and `[E3].mass` | contract | contract text |
| 4 | update `quality.evaluators[E2].identity` and `[E3].identity` — the native-identity claim is retired | contract | contract text |
| 5 | **[added]** update `quality.E2_E3_shared_mass_law`, currently the literal string `"piecewise_x6_below_or_at_0.1"` | contract | not in Phase-2D's ledger |
| 6 | **[added]** update `normative_documents[].sha256` for `QUALITY_EFFORT_SPEC.md` and `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` — **both are stale now**, broken by Phase-2D's own edits | contract | nothing enforces this; it is silent drift |
| 7 | update the frozen contract digest literal | `+ie2a/frozen_contract_sha256.m` | `production_preflight` `contract_hash` |
| 8 | **[added]** amend `FAIRNESS_RISK_REGISTER.md` row F01, a contract-listed normative document that still asserts E2/E3 native identity | study_design | normative-document list |
| 9 | **[added]** amend `PHASE2_FINAL_READINESS.md` item 4, which mandates quoting the now-false `x^6` shared-law description | final_recheck | freeze package |
| 10 | **[added]** replace the qualitative "closer to two-way" wording with the measured E2–E3 degeneracy (median 5.2e-09) | `QUALITY_EFFORT_SPEC.md` §2, `FAIRNESS_RISK_REGISTER.md` F01 | disclosure requirement |
| 11 | add the amendment entry to the freeze record | `METHODOLOGY_FREEZE_RECORD.md` | refreeze action, never an author action |
| 12 | update provenance | `implementation_provenance.json` protected digests | `+ie2a/verify_provenance.m` |
| 13 | preserve every native optimizer digest unchanged | six `protected_numerical_sources`, three `profile_sources` | `verify_provenance` |
| 14 | invalidate any precision-qualification artifact created under the old evaluator | `validation_outputs/olhoff_new_trajectory_precision_qualification.json` | see #15 |
| 15 | **[added]** bind the qualification artifact to the evaluator it was produced under | `+ie2a/production_preflight.m` `localQualificationPass` | **currently absent — see below** |
| 16 | produce a fresh qualification artifact | new run | `production_preflight` `olhoff_lossless_trajectory` |
| 17 | rerun `production_preflight` | — | fail-closed |

**Not required** (verified by reading the code, contrary to Phase-2D's cautious listing):
`+ie2a/run_negative_controls.m` asserts no mass string — its only evaluator line selects
`evaluators(1)` for the "E1 only" control. `+ie2a/validate_contract.m` checks evaluator
**ids** only.

## Obligation 15 — a latent gate weakness this audit found

`production_preflight.m` accepts the qualification artifact on two conditions only:

    ok = isfield(q,'pass') && q.pass && isfield(q,'scope') && strcmp(q.scope,'new_olhoff_trajectory')

Nothing binds it to the evaluator or contract it was produced under. Today this is harmless —
the file does not exist. The moment the evaluator changes, it stops being harmless: a
qualification produced under one mass law would silently satisfy the gate under another. The
artifact should carry, and `localQualificationPass` should check, the `quality.source_sha256`
and contract digest in force when it was produced.

This is not caused by the amendment. It becomes material *because* of it.

## Preflight consequence, stated completely

Phase 2D states that the moment the contract and evaluator are amended, preflight fails on
`contract_hash` and `evaluator_hash` until #7 and #2 are done — correct, and that is the gate
working as designed. Two additions:

- `olhoff_lossless_trajectory` **already fails today**, and will keep failing until #16.
- The two stale `normative_documents` digests (#6) are **not** checked by preflight. Nothing
  will catch them. They must be fixed by hand at refreeze.
