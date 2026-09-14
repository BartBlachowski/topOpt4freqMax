# WP21 — Independent evaluator-amendment delta-audit request
OFFLINE_AMENDMENT_VALIDATION — Phase 2D does not approve its own work (WP22)

The auditor is asked to verify the following twelve points. Evidence locations are given.

| # | Verify | Where |
|---|---|---|
| 1 | Phase-2C evidence justifies reopening the evaluator subsystem | `analysis/iteration_efficiency_evaluator_discontinuity_audit/E2_E3_DISCONTINUITY_AUDIT.md`; independently reproduced here in `OLD_DEFECT_REPRODUCTION.csv` (max old relative E2 = 2.6736e-02 over 236 stored paired states) |
| 2 | Scope remained narrow | `AMENDMENT_SCOPE_LEDGER.csv` — 1 file created, 2 normative files amended, 13 files explicitly not modified with reasons |
| 3 | Eq. (4a) is source-defined and faithfully implemented | `references/Du2007_Topological.pdf` §2.2 Eq. (4a), c0 = 1e5; `SOURCE_FIDELITY.csv` (Phase 2C); `EVALUATOR_UNIT_TESTS.md` |
| 4 | C0 continuity restored, C1 honestly not claimed | `EVALUATOR_UNIT_TESTS.md`: jump 1.0e-01 → 6.939e-17; one-sided derivatives 6 and 1 |
| 5 | Native optimizers byte-identical | `amendment_provenance.json` pre/post hashes for `massScale.m`, `defaultCfg.m`, `mass_interp.m`, Yuksel, Proposed, `olhoffOptStabilized.m`; all `protected_numerical_sources` re-verified against the Phase-2A record |
| 6 | Old native-identity claims removed or qualified | `NORMATIVE_CHANGE_LEDGER.csv`; pre-amendment copies in `preamendment_copies/` |
| 7 | Robust/common quality semantics internally consistent | acceptance rule, q levels, P, B_meas formula, k_enter/k_cert definitions all unchanged; `+ie2a` engines reused verbatim |
| 8 | Offline re-evaluation reproduces the expected stability improvement | `EQ4A_DOUBLE_ULP_STABILITY.csv` (4.0e-03 → 2.2e-13), `EQ4A_SINGLE_ROUNDING_STABILITY.csv` (2.67e-02 → 5.60e-08), `AMENDED_OLHOFF_TRAJECTORY_EVALUATION.csv` (2.65e-02 → 2.65e-10), `BINDING_EVALUATOR_ANALYSIS.csv` (150/1600 → 0/1600) |
| 9 | No topology/timing/accounting/scaling drift | `WP10` hard gate identical 1600/1600; no rule in those subsystems touched |
| 10 | Phase-2B correctly classified historically | `RETROSPECTIVE_ARTIFACT_CLASSIFICATION.csv`; Phase-2B files unmodified |
| 11 | A new precision qualification is required after refreeze | `PHASE2D_AMENDMENT_REPORT.md` WP19 |
| 12 | Production remains blocked | `production_preflight.m` unmodified; no qualification artifact; no token; no campaign |

## Two items the auditor should scrutinise hardest

**The C1 decision.** This amendment adopts Eq. (4a) (C0) and not Eq. (4b) (C1). The
justification is minimum correction: Phase 2C established a *finite jump in the function*,
and a post-hoc evaluator is never differentiated, so no evidence requires C1. An auditor who
believes the slope discontinuity is independently harmful should say so — Eq. (4b) is
equally source-defined and equally implemented in the repository.

**The unexercised reference/persistence path.** WP11/WP12 could not be exercised end-to-end
on stored artifacts. The frozen design requires a separate `B_ref = 3200` reference
trajectory (`reference.trajectory_separate_from_measurement: true`), and every stored
production file is a 1600-horizon measurement run on which reference does not establish
under either the old or the amended evaluator. Regenerating one requires an optimizer run,
which Phase 2D forbids. **The amendment's effect on `b_ref`, `k_enter` and `k_cert` is
therefore established by mechanism and by per-state evaluator stability, not by a direct
end-to-end offline re-run.** The auditor should decide whether that is sufficient for the
amendment, or whether a reference-length re-run must precede refreeze.
