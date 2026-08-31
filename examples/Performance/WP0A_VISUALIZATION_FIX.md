# WP0A — shared final-topology visualization fix

Date: 2026-08-30

Classification: **non-numerical presentation/infrastructure fix**.

## Scope

- `tools/Matlab/renderTopologyDensity.m` is the single status-aware renderer for final and
  accepted-state density fields.
- `tools/Matlab/run_topopt_from_json.m` delegates its existing Proposed/Yuksel final snapshot
  path to that renderer.
- `examples/Performance/performance_comparison.m` calls the renderer after all solver-side and
  caller-side timing boundaries, including for the separate stabilized Du–Olhoff runner.
- `analysis/iteration_efficiency_study_design/render_iteration_efficiency_topology_grid.m`
  selects and lays out frozen-methodology states but delegates every nonempty cell to the same
  renderer. Failed/unavailable cells are labelled empty; no earlier state is substituted.
- `examples/Performance/test_shared_topology_renderer.m` is the geometry/status/export smoke
  test.

No optimizer, numerical source, evaluator, status rule, campaign configuration, timing
boundary, or frozen result value was changed. Existing CSV/JSON/LaTeX campaign artifacts are
not rewritten by this fix. New or standardized PNG/FIG output is presentation-only.

## Frozen-source and result verification

The following SHA-256 values were recorded before and after the successful MATLAB smoke rerun.
None of these paths is in the source diff, and every digest remained identical.

| Frozen path | SHA-256 |
|---|---|
| `analysis/ourApproach/Matlab/topopt_freq.m` | `6d9ea66fcc27f63b7380708b5735552b5d9f2885d3e65714af572daccdae72b2` |
| `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m` | `5afc3d16b4ed6af05793df461b541ed3b2ea62a6da8836f38301a9a3917e6ba2` |
| `Matlab/reproduction2007/algo/olhoffOpt.m` | `4784ecf3f6b42d8af6f5a9695e2924bf1f4c924ebf9c76be9a858a2ef769e5de` |
| `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` | `95240cf60f82b40f8e5e892b9eea9b20a8fd3744b5eca6fdfc8dde2698d82aec` |
| `analysis/olhoff_stabilization_audit/final_campaign_profile.json` | `5d12bc0ae6a09d2f4df01fb38d7f483a7450b06cac67eceac30d3fab3618b610` |
| `analysis/olhoff_stabilization_audit/selected_profile.json` | `60fa944f4aecf34611de5413096d6a0de235eae05febbc4dd481bee5a26a67da` |
| `analysis/three_method_parametric_study/results/profile_freeze_manifest.json` | `b55e31d87d18e90e8c0b8d278bd4111d494610b62f253194ea16cfcf78252eca` |
| `examples/Performance/final_campaign/table1_performance.csv` | `c4b8103b72017758064cd43bab52a2b05fe0803bc15458fe10657840db833c4d` |
| `examples/Performance/final_campaign/common_evaluators.csv` | `4bdeab03e7bc43832c48915560e25153bb825e74c9f8cbbfc8033d163a28d66b` |
| `examples/Performance/final_campaign/benchmark_results.json` | `765fc433b9254d91d7631d551246df2b47f6ee415837b918d8ee3807dbafe663` |

## Verification commands

Static checks:

```text
git diff --check
```

MATLAB smoke test (rerun when the network license is available):

```matlab
addpath('examples/Performance');
test_shared_topology_renderer
```

The first 2026-08-30 attempt was blocked by MathWorks Licensing Error 15 (`-15.2`) before the
test body could run. After license service was restored, the same command passed. It created
and removed only temporary visualization files; it did not invoke an optimizer.
