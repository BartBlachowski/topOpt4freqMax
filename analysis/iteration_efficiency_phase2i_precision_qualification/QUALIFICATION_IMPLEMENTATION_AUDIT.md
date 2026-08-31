# Phase 2I qualification implementation audit

Status before numerical execution: **PASS**. Machine-readable evidence is in
`raw/static_audit.json`.

## Pairing and indexing

- `rho_snapshots` is two-dimensional, `NE x (nDone+1)`.
- Column 1 is the initial state; the state after update `k` is column `k+1`.
- The protected runner leaves `res.rho` as double and writes
  `rho_snapshots(:,outer+1)=single(rho)` only after an accepted update.
- Existing Phase-2B pair capture correctly uses same-run `res.rho` and
  `rho_snapshots(:,end)`; cross-run comparisons are used only for prefix identity.
- Phase 2I adds an isolated qualification mirror whose source diff is limited to its
  function name/comments and the two observational snapshot writes. It captures the
  exact evolving double state and is required to reproduce the untouched protected
  runner's final density, cast snapshots, triggers, native status, and every non-timing
  history field.

## Frozen evaluator path

- `ie2a.evaluate_common` invokes the hash-bound `study_evaluate_design` Candidate-C
  implementation.
- E1 uses linear mass. E2/E3 use continuous Eq. (4a), including the `1e5*rho^6`
  low-density branch.
- Adaptive requests begin at 3 and double without a scientific ceiling.
- Validity is the unanimous strict rule: `voidKE<0.5`, `voidSE<0.5`, and
  `densityParticipation>0.5`; IPR remains diagnostic.
- No old discontinuous Eq. (4) branch exists in the active Candidate-C evaluator.

## Static integrity result

All static checks pass. Contract, evaluator, freeze, and normative identities match the
Phase-2H freeze. The historical independent Candidate-C pair audit contains 708
evaluator/state records and is retained as supporting, not endpoint, evidence.

No frozen evaluator, contract, normative methodology, native optimizer, or historical
audit directory was modified.
