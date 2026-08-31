# WP17/WP18/WP19 — Freeze impact, Phase-2B status, production status
READ_ONLY_AUDIT — NOT_NEW_OPTIMIZATION_EVIDENCE

## WP17 — Is this implementation-only or methodology-invalidating?

**Neither purely.** The implementation is correct; the *methodological choice* of a
discontinuous native interpolation as a co-primary common evaluator is what fails.

E2/E3 are source-faithful (WP14), so no implementation correction is owed on fidelity
grounds. But they are not robust at the frozen 0.5–2% quality scale (WP15), and the
instability propagates into the study's headline estimand. The finding is therefore
**methodology-implicating, confined to the evaluator / quality-reference subsystem.**

### Minimum scope of reopening

Requires reopening:

| Frozen item | Why |
|---|---|
| `quality.evaluators` E2/E3 definitions in `iteration_efficiency_contract.json` | the mass law is the defective object |
| `quality.co_primary` and the robust-minimum acceptance rule | WP13: 23.5% of states change binding evaluator; a min over a discontinuous member inherits the discontinuity |
| `IMPLEMENTATION_REQUIREMENTS.md` evaluator specification | must state which variant the common evaluator uses and why it differs from the native optimizer variant |
| `reference` subsystem values Q_ref and b_ref | derived from the evaluators; must be recomputed, not re-derived by argument |
| frozen prior absolute-quality context (Olhoff over Proposed 6.2–8.5%, over Yuksel 5.9–7.7%) | those margins are E1/E2/E3-derived and sit at the same scale as the 2.65e-2 instability |

Does **NOT** require reopening, on present evidence:

- topology gate — hard gate identical at 236/236 independently recomputed paired states and 45/45 Phase-2B pairs;
- iteration accounting, timing, scaling, method profiles, mesh sequence — no evidence implicates them;
- persistence *semantics* — `k_enter`/`k_cert` definitions are unaffected; only their computed *values* change because their input changes.

## WP18 — Phase-2B status

**Phase 2B remains valid and is not reinterpreted.** Its verdict stands as recorded:

    OLHOFF SINGLE-PRECISION TRAJECTORY NOT QUALIFIED
        under the E2/E3 evaluator definition frozen at the time of testing.

Phase 2B tested the frozen definition that existed. This audit explains *why* that
definition made single storage unqualifiable; it does not overturn the test. The
independent reproduction in WP9 confirms Phase-2B's numbers from stored paired density
fields (max relative E2/E3 error 2.674e-02, hard gate identical 236/236), and WP11
reproduces its estimand changes exactly (b_ref 2200 vs 2100; k_enter 233/315/609 vs
232/309/524; k_cert 332/414/708 vs 331/408/623).

**Would a corrected methodology require a new precision qualification? Yes.** A precision
qualification is a statement about a specific evaluator/decision pipeline. Changing E2/E3
changes every gate decision, `b_ref`, `k_enter` and `k_cert` the qualification compares, so
the Phase-2B result would not transfer and a fresh qualification would be required. On the
sensitivity estimate in `MINIMUM_CORRECTION_OPTIONS.md` that repeat qualification would
have a good prospect of passing, but that must be measured, not assumed.

## WP19 — Production status

Production remains blocked throughout and after this audit. Nothing was created, modified
or authorized:

- no passing precision artifact created;
- `production_preflight.m` not modified (SHA-256 unchanged);
- authorization token not set;
- nine-mesh campaign not started;
- no optimizer run, no methodology edit, no evaluator edit.

### Recommended next work package

**Narrow methodology revision plus delta audit, scoped to the evaluator / quality-reference
subsystem**, in this order:

1. A decision review on Option B (Du & Olhoff Eq. (4a), c0 = 1e5, in the common evaluator
   only) versus Options C/D, explicitly resolving whether the common evaluators must mirror
   native interpolations literally or may use the source's continuous variant.
2. If Option B is adopted: amend the contract's `quality.evaluators`, amend
   `IMPLEMENTATION_REQUIREMENTS.md`, and re-freeze — evaluator subsystem only.
3. Offline re-evaluation of existing trajectories under the corrected evaluator. No
   optimizer reruns are required. Note the Olhoff historical constraint: stored snapshots
   are float32, so the corrected-evaluator re-evaluation of historical data inherits the
   Mo9 scope limits already recorded in Phase 2B.
4. A delta audit of the amended subsystem.
5. A fresh Olhoff precision qualification against the corrected pipeline.
6. Only then, pre-production review.

Steps 1–2 are decisions for the methodology owner, not for an auditor.
