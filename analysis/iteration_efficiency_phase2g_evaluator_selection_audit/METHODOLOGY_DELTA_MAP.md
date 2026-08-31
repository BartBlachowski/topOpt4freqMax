# Collateral methodology delta map

| artifact | category | required consequence |
|---|---|---|
| `analysis/three_method_parametric_study/study_evaluate_design.m` | evaluator-definition | implement Eq.(4a), eigenvectors, three-condition intersection, adaptive selection, fail-closed status |
| `QUALITY_EFFORT_SPEC.md` | evaluator-definition/reporting | replace algebraic mode 1 by C; state loss of E2/E3 native identity and retained family |
| `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` | evaluator-definition/reference | define C, unresolved semantics, and unchanged persistence formulas |
| `iteration_efficiency_contract.json` | evaluator-definition | update evaluator definitions, source hash, status, adaptive/failure contract |
| `+ie2a/evaluate_common.m`, `analyze_trajectory.m`, reference/persistence integration | reference/persistence consequence | consume C-selected values and propagate missing status fail-closed |
| `+ie2a/production_preflight.m` | precision consequence | bind qualification to evaluator/contract hashes and require new qualification |
| `+ie2a/frozen_contract_sha256.m`, implementation provenance | evaluator-definition | update hashes only in the controlled refreeze |
| `FAIRNESS_RISK_REGISTER.md` | reporting consequence | record modal-classification and missing cross-method evidence risks |
| `PHASE2_FINAL_READINESS.md`, `METHODOLOGY_FREEZE_RECORD.md` | reporting/refreeze | replace stale Eq.(4) language and record amendment |
| endpoint tables/figures/status taxonomy | reporting consequence | selected ordinal, diagnostics, escalation, unresolved status, evaluator cost |
| timing specification/replays | timing consequence | explicitly exclude C post-processing; report overhead separately |
| Olhoff/Yuksel/Proposed native optimizers and profiles | no change | remain byte-identical |
| `P`, `B_ref`, `B_meas` formula, q levels, topology gates | no change | preserve exactly |
| Olhoff route ruling | no change | LP principal; nested MMA secondary |

Historical audit and implementation reports are not rewritten; they remain records of the
instrument in force when produced.
