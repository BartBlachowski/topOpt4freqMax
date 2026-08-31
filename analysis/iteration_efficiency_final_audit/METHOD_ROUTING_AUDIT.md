# Method routing and identity audit

Entry: `iteration_efficiency_final(runMode, olhoffVariant)` → `iefinal.run` → `iefinal.config` → `iefinal.method_plan` → `iefinal.run_trajectory`.

## Proposed

`run_trajectory.localOc('Proposed')` → `study_base_config('proposed',…)` → `run_topopt_from_json` → `analysis/ourApproach/Matlab/topopt_freq.m` (hash-pinned, verified). Parameters `move=0.2`, `tol=0.01`, `rmin_element=2` match the manifest and the frozen contract profile. Domain 8×1, supports `closest_point` at (0, 0.5) and (8, 0.5) — the same simply-supported mid-height beam the Candidate-C evaluator assembles. No fallback or substitution path exists. **Identity: correct.**

## Yuksel

`localOc('Yuksel')` → `study_base_config('yuksel',…)` → `top99neo_inertial_freq.m` (hash-pinned, verified). `move=0.1`, `stage1_tol=stage2_tol=0.01`, `rmin_element=2.5`, `stage1_max_iters=2000`.

Stage-1 → Stage-2 carry-over: the observer records every recorded state with its `stage` marker; `localOc` selects `eligible = (stage==2)`, so state 0 of the measured trajectory is the Stage-1/Stage-2 handoff design, recovered from `first_xPhysPrev(2)` when available. Verified live: observer states are `double`, the last observed state bit-equals the returned design, and `has_first_xPhysPrev` is set.

Accounting: `native.stage1_updates` from `nIterStage.stage1` (field confirmed present at `run_topopt_from_json.m:588`), `stage2_updates = numel(idx)`, `total = stage1 + stage2`. `build_rows` writes `yuksel_stage1_iterations = s1`, `yuksel_stage2_iterations = k`, `yuksel_total_iterations = s1 + k`, `native_iterations = s1 + k`. Stage 1 is neither omitted nor double-counted. **Identity and accounting: correct.**

**But the Yuksel *timing* path does not reproduce this computation — see finding F-01 in `ACCOUNTING_AND_TIMING_AUDIT.md`.**

## Olhoff-LP (principal)

`localOlhoff('lp')` → `repro2007_config('fig3a_best')` → `olhoffOptStabilized(cfg, policy)` with `policy.id='S1'`, `move_sequence=[0.005 0.0025]`, `gap_threshold=0.01`, `persistence=100`. `olhoffOptStabilized` calls `innerLoopLP` — the only `innerLoopLP.m` in the repository outside a clearly-named diagnostic copy. `olhoffOpt` is unreachable from this branch.

Verified live under `variant='lp'`: `hist.nInner` is 1 per outer (one `linprog` call), `hist.lpBackendIterations` is genuine and variable. No nested-MMA execution occurs.

`fig3a_best` sets `rminEl=1.3`, `rminPhys=[]`, so `olhoffOptStabilized`'s physical-radius rescale is inactive and the filter radius stays 1.3 **elements** across all nine meshes — consistent with Proposed (2 el) and Yuksel (2.5 el), all three declared `radius_units:'element'`.

Accounting (`build_rows.localAccounting`): `olhoff_outer_updates = k`, `olhoff_lp_calls = min(k, lp_calls)`, `olhoff_failed_lp_calls`, `olhoff_lp_backend_iterations = sum` over the first `k` finite entries. **`nInner` never appears in any result row**, so the forbidden `nInner=1`-as-solver-iterations representation cannot occur. **Identity and accounting: correct.**

## Olhoff-MMA (secondary)

`localOlhoff('mma')` → `olhoffOpt(cfg)` with `innerSolver='mma'`, `offDiag=true`, `move=0.01`, `rminPhys=0.06`, `maxInner=300`, `tolInner=0.01`, `minInner=5`. Verified live at 160×20, 4 outer updates:

| check | result |
|---|---|
| `r.cfg.innerSolver` | `mma` |
| `r.cfg.offDiag` | true (full Eq. 25d coupling) |
| effective `rminEl` taken **post-call** | 1.2000, equals `0.06/(b/nely)` to < 1e-12 — not a stale console header |
| LP calls (`hist.lpFlag`) | all NaN → **zero**, no LP fallback |
| `hist.nInner` | 108, 102, 108, 101 — genuine nested MMA iterations |
| `class(rho_snapshots)` | `double` |

`innerLoop.m` holds `F`, `fJJ`, `lam` fixed and calls `mmasub`/`subsolv` per inner iteration — genuine nested MMA, no eigen-resolve inside. Accounting derives total/mean/median/p95/max inner, cap-hit count and fraction, converged count and fraction from `inner(1:k)`. **Identity and accounting: correct.**

## Selector correctness

`method_plan` → `ie2a.olhoff_variant_plan`:

| selector | rows produced | LP executed | MMA executed |
|---|---|---|---|
| `lp` | Proposed, Yuksel, Olhoff-LP | yes | no |
| `mma` | Proposed, Yuksel, Olhoff-MMA | no | yes |
| `both` | Proposed, Yuksel, Olhoff-LP, Olhoff-MMA | yes | yes |

Under `both`, the two Olhoff routes are separate table rows driving two independent `run_trajectory` calls into separate `reference/<id>` and `measurement/<id>` directories keyed by `result_label`, so no state is reused or contaminated. Confirmed by the retained smoke evidence (`runs/smoke/{lp,mma,both}/…/validation/selector.json`) and by the `run_final_integration_tests` selector assertions.

**Selector: correct.**

## Principal vs secondary separation

- `PRODUCTION_MANIFEST.json`: `principal_methods = [Proposed, Yuksel, Olhoff-LP]`, `optional_secondary_method = Olhoff-MMA`.
- Rows carry `method_variant ∈ {proposed, yuksel, lp, mma}` and MMA rows carry `route_role = secondary_paper_native_uncontrolled_vs_lp`.
- `fit_scaling_table` groups by `method + "-" + method_variant`, so `Olhoff-lp` and `Olhoff-mma` are distinct fit series; MMA cannot silently occupy the LP series.
- `render_topologies` iterates the rows actually present, so MMA topologies appear only when selected.
- `validate_results` rejects an `lp` row carrying MMA accounting and an `mma` row carrying non-zero LP calls (both verified live).

**MMA cannot silently replace LP in tables, fits, figures, timing or ranking. Separation: correct.**

Residual issue (MINOR, F-09): `olhoff_variant_plan` declares the MMA runner as `run_repro2007.m` and role `secondary_paper_literal`, while the harness calls `olhoffOpt` directly and the manifest says `secondary_paper_native_uncontrolled_vs_lp`. Those two fields are not consumed by `method_plan`; cosmetic only.
