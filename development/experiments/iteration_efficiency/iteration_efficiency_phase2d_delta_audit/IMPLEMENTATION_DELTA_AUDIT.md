# WP4 / WP18 — Implementation and collateral-drift delta audit
READ_ONLY_INDEPENDENT_DELTA_AUDIT

## Textual delta

`diff analysis/three_method_parametric_study/study_evaluate_design.m` against
`analysis/iteration_efficiency_phase2d_evaluator_amendment/+ie2d/study_evaluate_design_eq4a.m`
yields exactly three hunks: the function name and an eight-line provenance docstring, and
two functional lines:

    line 47  E2:  g(low)  = z(low).^6;    ->  g(low)  = 1e5 * z(low).^6;
    line 54  E3:  g(low)  = z3(low).^6;   ->  g(low)  = 1e5 * z3(low).^6;

The reported effective change is therefore **exactly** what Phase 2D claims:
`z^6 -> 1e5*z^6` on the E2 low branch and `z3^6 -> 1e5*z3^6` on the E3 low branch.

## Semantic delta — call-path inspection, not textual diff alone

`study_evaluate_design.m` is a single self-contained file (131 lines) with four local
functions: `solve_modes`, `assembly_indices`, `q4_matrices`, `connectivity`, plus
`deterministic_v0`. It calls nothing outside itself. The two changed lines live inside
`solve_modes` in the `case 'E2'` and `case 'E3'` arms.

| component | reached by the change? | evidence |
|---|---|---|
| E1 stiffness and mass | **no** | `case 'E1'` arm untouched; verified bit-identical output on a field containing low-density elements |
| E2 stiffness `1e7*(1e-9+(1-1e-9)*z.^3)` | no | line unchanged; separate statement from the mass line |
| E3 stiffness `1e7*z3.^3` | no | unchanged |
| E2 additive mass floor `1e-9` | no | the floor is applied to `g` *after* the branch; the expression `rr = 1e-9+(1-1e-9)*g` is unchanged |
| E3 density clamp `z3 = max(z,1e-3)` | no | unchanged; the clamp is computed before the branch |
| exact-count binary projection | no | computed in the parent function from `x` alone, before `solve_modes` is called |
| topology / volume diagnostics | no | same — see WP10 below |
| eigensolver call, options, tolerance | no | `eigs(...,3,'smallestabs',opts)` unchanged |
| deterministic start vector | no | `RandStream('twister','Seed',42)` unchanged |
| mesh, Q4 matrices, assembly indices | no | `q4_matrices`, `assembly_indices` unchanged |
| midheight pinned supports | no | unchanged |
| eigenvalue sorting / omega extraction | no | unchanged |

Independently confirmed numerically: on the 160x20 trajectory over 1600 states the
Eq. (4) → Eq. (4a) level shift in **E1 is 0.0000e+00 at every state** (bit-identical), and
E2/E3 are bit-identical at every state where no element satisfies `z <= 0.1`.

## Subsystems outside the evaluator

| subsystem | file | consumes an evaluator value? | verdict |
|---|---|---|---|
| hard gate / topology | `+ie2a/topology_metrics.m` | **no** — signature is `(x, nelx, nely, opts)`; body reads only `x` | unreachable |
| exact-count projection | `+ie2a/exact_count_binary.m` | no | unreachable |
| reference phase | `+ie2a/reference_phase.m` | yes — takes `Q` and `H0`; contains no interpolation law | value-dependent, rule-unchanged |
| persistence scan | `+ie2a/scan_persistence.m` | takes a boolean pass matrix only | rule-unchanged |
| measurement budget | `+ie2a/measurement_budget.m` | takes `(B0,b_ref,P,B_ref)` scalars | rule-unchanged |
| iteration accounting | `+ie2a/account_iterations.m` | no | unaffected |
| timing replay | `+ie2a/run_timing_replays.m`, `timing_replay_plan.m` | no | unaffected |
| scaling fits | `+ie2a/fit_power_law.m`, `generate_scaling.m` | fitted **over k_enter/k_cert**, which are evaluator-dependent | endpoints must be recomputed |
| status precedence | `+ie2a/classify_status.m` | no | unaffected |
| contract validation | `+ie2a/validate_contract.m` | checks evaluator **ids** only, not mass strings | no change needed |
| negative controls | `+ie2a/run_negative_controls.m` | one line selects `evaluators(1)` for the "E1 only" control; asserts no mass string | no change needed (Phase 2D's flag here is over-cautious but harmless) |

## WP18 — collateral drift

Files modified anywhere in the repository during the Phase-2D window
(`find -newermt "2026-08-30 13:40" ! -newermt "2026-08-30 14:20"`, excluding the Phase-2D
directory itself): **exactly two**, both declared —
`analysis/iteration_efficiency_study_design/QUALITY_EFFORT_SPEC.md` and
`ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md`. Their pre-amendment copies in
`preamendment_copies/` hash-match the declared `pre` values, and the live files hash-match
the declared `post` values.

Both diffs were read in full. They touch only the native-identity paragraph and the 0.429%
agreement figure. No rule, threshold, definition or numeric constant elsewhere in either
document changed.

All twelve contract-listed normative documents were hashed against the contract's pins
(`WP18_NORMATIVE_DOC_HASHES.csv`): **ten match**; the two mismatches are the two amended
documents. See finding **N1**.

Checked and **unchanged**: topology definition, `A_sig` (0.01) and `a_sig_by_mesh`,
aggregate-island diagnostic-only semantics, volume gate and its 1e-3 relative tolerance,
exact-count projection with index tie-break, `P = 100` (and OAT 50/200), `q ∈
{0.98, 0.99, 0.995}`, `B_ref = 3200`, `L_ref = 500`, `epsilon_ref = 1e-3`, the `B_meas`
formula, reference semantics, `k_enter`/`k_cert` definitions, status precedence and
`fit_eligible`, iteration accounting, timing methodology (threads=1, serial, 3 repetitions),
scaling methodology, the nine-mesh sequence, and all three method profile bindings.

**WP4 ruling: IMPLEMENTATION_SCOPE = PASS.**
**WP18 ruling: COLLATERAL_DRIFT = NONE.**
