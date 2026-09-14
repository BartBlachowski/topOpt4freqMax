# Source identity audit

## Repository state at audit time

| item | value |
|---|---|
| branch | `benchmark-methodology-r2` |
| HEAD | `d4df2137e68851f91cff5e75de1ee4a99a6a7625` |
| working tree | dirty — 22 tracked-modified, 25 untracked (47 entries) at audit start; unchanged by this audit except for the untracked audit directory |
| MATLAB | R2025b (`/Applications/MATLAB_R2025b.app`) |
| platform | darwin 25.5.0, arm64 |
| threads | `maxNumCompThreads(1)` set in `run_trajectory.m` and `run_timing_firewall.localOnce` |
| production campaign run during audit | **NO** |

## Change attribution

Integration work (this package):

- `analysis/iteration_efficiency_final/` (untracked, whole tree)
- `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` (modified)
- `Matlab/reproduction2007/algo/olhoffOpt.m` (modified)
- `Matlab/reproduction2007/algo/innerLoopLP.m` (modified)
- `analysis/iteration_efficiency_phase2a/+ie2a/install_observer.m` (modified)
- `analysis/iteration_efficiency_phase2a/+ie2a/observer_capture.m` (modified)

Pre-existing (declared in `initial_provenance.json`, confirmed):

- `.gitignore`
- `analysis/iteration_efficiency_phase2i_precision_qualification/` (Phase-2I evidence)
- `analysis/OlhoffReproduced2007/` — unrelated to the harness and on no production code path; the harness calls `olhoffOptStabilized.m` and `olhoffOpt.m`, never `topopt_olhoff_reproduced2007.m`.

No unexpected file can influence production: `preflight.m` hash-verifies every method source and every pinned component before a run starts, and fails closed on any mismatch (verified — see `PREFLIGHT_NEGATIVE_CONTROLS.md`).

## Independent checksum verification

`shasum -a 256 -c analysis/iteration_efficiency_final/SHA256SUMS.txt` → **36/36 OK**, zero mismatches. The only package file not covered is `SHA256SUMS.txt` itself (expected).

Every hash pinned in `PRODUCTION_MANIFEST.json` was recomputed independently and matched:

| file | manifest sha256 | recomputed | match |
|---|---|---|---|
| `iteration_efficiency_contract.json` | `cc900b4a…` | `cc900b4a…` | yes |
| `study_evaluate_design.m` | `e14a21ef…` | `e14a21ef…` | yes |
| `topopt_freq.m` | `6d9ea66f…` | `6d9ea66f…` | yes |
| `top99neo_inertial_freq.m` | `5afc3d16…` | `5afc3d16…` | yes |
| `olhoffOptStabilized.m` | `10132a39…` | `10132a39…` | yes |
| `olhoffOpt.m` | `d34f62b9…` | `d34f62b9…` | yes |
| `topology_metrics.m` | `ebe14f15…` | `ebe14f15…` | yes |
| all 10 `pinned_components` | — | — | yes (10/10) |

Integration manifest integrity: **PASS**.

## Contract-vs-manifest hash divergence (finding F-05)

The binding contract `iteration_efficiency_contract.json` still pins the **pre-instrumentation** Olhoff sources:

| file | contract pin | actual | status |
|---|---|---|---|
| `olhoffOptStabilized.m` | `95240cf6…` | `10132a39…` | stale |
| `innerLoopLP.m` | `7724753c…` | `bfd16bcd…` | stale |

`preflight.m` calls `ie2a.validate_contract(c, VerifyFiles=false)`, so this divergence is never surfaced. The executed sources *are* pinned — to `PRODUCTION_MANIFEST.json` — and the instrumentation is proved bit-neutral (see `NUMERICAL_BEHAVIOR_NEUTRALITY.md`), so no scientific result is corrupted. But the binding document now disagrees with what runs, and the disagreement is silenced rather than recorded. **MODERATE, non-blocking.**
