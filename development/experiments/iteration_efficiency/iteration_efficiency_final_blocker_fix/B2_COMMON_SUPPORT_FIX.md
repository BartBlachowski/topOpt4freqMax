# B-2 — true common support for scaling fits

## Root cause

`+iefinal/fit_scaling_table.m` fitted each series over **its own** eligible meshes:

```matlab
ix = string(T.method) == methods(i);
f  = ie2a.fit_power_law(T.element_count(ix), T.(metric)(ix), ...);
rec = struct(..., 'support', "common", ...);   % hardcoded label
```

No intersection was ever computed. `synthetic_scaling_validation.m` likewise reported `'common_support', true`. Because its synthetic input gave every method every mesh, the smoke could never expose the defect.

This contradicted the frozen scaling specification in `iteration_efficiency_contract.json`:

```json
"common_support_companion_required": true,
"cross_method_comparison_outside_common_support": false,
"minimum_valid_meshes": 3,
"fit_positive_finite_certified_only": true,
"fit_eligible": ["PASS","PASS_WITH_LATER_SOLVER_TERMINATION"]
```

## Definition used — taken from the frozen rules, nothing invented

A cell is **eligible** for a metric when all three frozen conditions hold:

1. `P == P_primary` (100);
2. `status ∈ {PASS, PASS_WITH_LATER_SOLVER_TERMINATION}` (the frozen `fit_eligible` set);
3. the metric value is positive and finite (frozen `fit_positive_finite_certified_only`).

**Participants** for a metric are the series with at least one eligible value for it. **S_common** is the intersection of the eligible mesh sets of all participants. No new censoring policy was introduced.

## Correction

`fit_scaling_table` now emits **two explicitly labelled families** per metric:

- `support = "available"` — each series over its own eligible meshes (what previously existed, now labelled honestly);
- `support = "common"` — every series restricted to S_common, intersected **before** fitting.

Each row carries `n_support`, `support_meshes` (the exact mesh list), `fitted`, `included_meshes`, `exclusions` and a `note`. `C`, `p`, `R2_log` and the leave-one-out bounds are all computed by a single `ie2a.fit_power_law` call on exactly the rows of that support — they cannot diverge, because they come from one fit of one vector pair.

**Fail-closed**: when `numel(S_common) < 3` (the frozen minimum) or a series does not participate, no fit is attempted — `fitted = false`, `C = p = NaN`, and the `note` says why. A common fit can never silently widen to method-specific support, because the common rows are computed from S_common alone.

`generate_scaling_outputs` writes the disclosure alongside the fits (`scaling_common_support.csv`: metric, participants, common mesh list, `n_support`, feasibility) and now filters status by the exact frozen `fit_eligible` set rather than a `startsWith(...,'PASS')` prefix test. The synthetic smoke figure titles the support it used (`common support n=…`), so a plotted series cannot visually imply a wider common fit than was performed; its report no longer asserts `common_support: true` but reports `common_support_enforced` and the actual `n`.

## Validation — expected support asserted explicitly

Meshes as element counts: 3200 = 160×20, 7200 = 240×30, 12800 = 320×40, 20000 = 400×50.

| # | case | expected S_common | `n_support` | feasible | result |
|---|---|---|---:|:-:|:-:|
| 1 | all methods have all meshes | `3200,7200,12800,20000` | 4 | yes | ✅ |
| 2 | one method lacks one mesh | `3200,7200,12800` | 3 | yes | ✅ |
| 3 | different methods lack different meshes | `7200,12800` | 2 | **no** | ✅ fail-closed |
| 4 | only one common point remains | `7200` | 1 | **no** | ✅ fail-closed |
| 5 | zero common points remain | *(empty)* | 0 | **no** | ✅ fail-closed |
| 6 | RUN_ERROR cell present | `3200,7200,20000` | 3 | yes | ✅ excluded |

In case 2 the `available` fit for the complete method still legitimately reports `n_valid = 4`, while its `common` fit reports 3 — the two families stay distinct and honest.

In case 6, Olhoff's 320×40 cell is `RUN_ERROR`; 12800 is absent from S_common, every `common` fit reports `n_valid = 3`, and no `included_meshes` string contains `12800`.

The end-to-end containment run (`verify_b3_containment.m`) exercises the hardest case: Olhoff fails at 160×20 and Proposed fails at 320×40, leaving S_common = `{7200}` with `n_support = 1` — correctly reported infeasible, with both failed meshes excluded (`common_excludes_failed = 1`).

## Verdict: **CLOSED**
