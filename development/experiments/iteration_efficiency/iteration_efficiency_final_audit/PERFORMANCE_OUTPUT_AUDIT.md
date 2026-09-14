# Performance / scaling output audit

The question this section exists to answer: **does the harness finally deliver the performance comparison the campaign set out to obtain?** Almost — one defect (F-02) blocks.

## 1. Both interpretations are supported

**A — optimization maturation efficiency (how many updates?)**

| output | source | available |
|---|---|---|
| `k_enter` vs problem size | `build_rows` → `results.csv` → `scaling_fits.csv` | yes |
| `k_cert` vs problem size | same | yes |
| Proposed native iterations | `native_iterations` | yes |
| Yuksel Stage 1 / Stage 2 / total | `yuksel_stage1_iterations`, `yuksel_stage2_iterations`, `yuksel_total_iterations` | yes |
| Olhoff-LP outer updates / LP calls / genuine backend work | `olhoff_outer_updates`, `olhoff_lp_calls`, `olhoff_lp_backend_iterations` | yes |
| Olhoff-MMA outer / total inner / mean / median / p95 / max / cap-hit / converged fraction | `olhoff_mma_*` | yes |

**B — computational performance (how much wall clock?)**

| output | source | available |
|---|---|---|
| `T_enter` | `native_total_time_to_enter` | yes — **but see F-01 for Yuksel** |
| `T_cert` | `native_total_time_to_cert` | yes — same caveat |
| mean native iteration cost | `mean_native_iteration_time` | yes — same caveat |
| total-time scaling | `generate_scaling_outputs` metric list | yes |
| per-iteration-time scaling | `mean_native_iteration_time` in the metric list | yes |
| method-specific inner-work scaling | `olhoff_mma_total/mean/p95_inner_iterations` in the metric list | yes |
| Yuksel stage curves | `stage1_seconds` / `stage2_seconds` in `timing_replay_samples.csv`; stage counts in the rows | yes |

`generate_scaling_outputs` runs **after** `run_timing_firewall`, so the timing columns are populated before fitting. The harness is not merely an endpoint-count study; the original performance objective is preserved.

## 2. Scaling fits

`ie2a.fit_power_law` implements the frozen specification: unit-weight OLS of `log y` on `log Ne`, `y = C·n^p`, requiring ≥ 3 valid points, reporting `C`, `p`, `R2_log`, `n_valid`, `Ne_min`, `Ne_max`, `included_meshes`, `p_LOO_min`, `p_LOO_max`, `exclusions`, and flagging `weakly_identified` when `R2_log < 0.8` or the LOO range spans zero or exceeds `|p|`. **Both C and p are reported.** Only `P=100` rows with a `PASS*` status enter the fit.

## 3. Finding F-02 (MAJOR) — "common support" is asserted, never enforced

`+iefinal/fit_scaling_table.m` fits each method over **its own** available meshes:

```matlab
ix = string(T.method) == methods(i);
f  = ie2a.fit_power_law(T.element_count(ix), T.(metric)(ix), ...);
rec = struct(..., 'support', "common", ...);   % hardcoded
```

There is no intersection step. Measured with Proposed present at all nine meshes and Olhoff-lp at seven:

| method | `support` label | `n_valid` |
|---|---|---|
| Proposed | `common` | 9 |
| Olhoff-lp | `common` | 7 |

Both are labelled `common` while resting on different mesh ranges. `synthetic_scaling_validation.m` likewise reports `'common_support', true`; because its synthetic input gives all methods all four meshes, the smoke test could never expose this.

**Contract conflict.** `iteration_efficiency_contract.json`:

```json
"common_support_companion_required": true,
"cross_method_comparison_outside_common_support": false
```

**Consequence.** `scaling_fits.csv` is the authoritative source for the paper's scaling table. It would present exponents `p` fitted over *different* mesh ranges under a label asserting they share support — inviting exactly the cross-method comparison the contract forbids. Unequal support is the **expected** case here, not a corner case: the contract records Olhoff 800×100 as `RUN_ERROR`, and a native LP iteration-limit failure is documented at 400×50.

**Minimum correction.** Compute the intersection of fit-eligible meshes across the compared methods and emit two labelled families — `support="available"` (per method, as now) and `support="common"` (restricted to the intersection). Leave `fit_power_law` untouched.

**Minimum re-verification.** Rerun `fit_scaling_table` on the unequal-support fixture used above and confirm two labelled families with correct `n_valid` per family.

## 4. Absolute quality

Every row carries `E1`, `E2`, `E3` and the robust common `Q = min(Q/Q_ref)` at the endpoint state, plus `topology_pass`, `volume_pass`, `hard_gate_pass`. All are in `RESULT_SCHEMA.json`'s `required` list. `write_results` emits a dedicated `absolute_quality_and_acceptance.csv` alongside `results.csv`, so quality is prominent rather than buried. The harness therefore cannot imply that fewer iterations means a better method while hiding a quality deficit.

## 5. Smoke output cannot be mistaken for science

Synthetic pipeline output is written to `runs/smoke/…` as `SMOKE_SYNTHETIC_SCALING_INPUT_NOT_SCIENTIFIC.csv`, `SMOKE_SYNTHETIC_SCALING_FITS_NOT_SCIENTIFIC.csv` and `SMOKE_SYNTHETIC_SCALING_NOT_SCIENTIFIC.png`, with in-figure titles reading `SMOKE / SYNTHETIC / NOT SCIENTIFIC`. Production writes to `runs/production/…` under plain names. Unambiguous.

## 6. Topology output

`+iefinal/render_topologies.m` calls only `render_iteration_efficiency_topology_grid`, which delegates every cell to the shared `tools/Matlab/renderTopologyDensity.m` (confirmed in the artifact record). For each method it renders the accepted `k_enter` state twice — actual gray, and its exact-count binary via `ie2a.exact_count_binary`.

Proposed, Yuksel and Olhoff-LP are rendered from the rows actually present, so the earlier omission of the Du–Olhoff topology from the performance workflow **cannot recur** — the grid is driven by the same authoritative rows as the tables. Olhoff-MMA is added when selected.

Unavailable cells: verified live by rendering a mixed grid containing one record with `density=[]`, `admissible=false`. The renderer drew a labelled empty cell and substituted nothing. Overlay geometry (`domain_extent=[0 8 0 1]`, supports at `[0 .5; 8 .5]`) matches `study_base_config` and the evaluator pencil for all nine 8:1 meshes.

## 7. Single authoritative source

`results.json` / `results.csv` (from `build_rows`) are the sole input to `generate_scaling_outputs`, `render_topologies` and `absolute_quality_and_acceptance.csv`. No manually maintained parallel values exist.
