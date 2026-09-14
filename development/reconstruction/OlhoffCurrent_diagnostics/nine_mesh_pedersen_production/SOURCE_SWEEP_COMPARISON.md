# SOURCE_SWEEP_COMPARISON (Part 18)

This comparison was made **only after** the nine new runs were complete, their raw outputs had been hashed and made read-only (`evidence/RAW_OUTPUT_FREEZE_SHA256.txt`), and the runner had been restored. The historical sweep was **not** used as an acceptance threshold. Nothing was re-run to reduce a discrepancy.

## The two sweeps

| | earlier upstream fixed-R sweep "S" | this campaign |
|---|---|---|
| source | upstream Olhoff repository, commit `6b0870850d74…` (snapshot `scientific_delta_olhoff_migration/source_snapshot/+olhoff_6b08708/repro/results/S<mesh>/`) | OlhoffCurrent `+impl` promoted from upstream `253069262407…`, HEAD `b21483b` |
| preset | `duOlhoffAdaptivePedersen` (upstream name) | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (resolves upstream `duOlhoffAdaptivePedersen`) |
| entry path | `repro/run_repro.m` → `olh.config.resolve` → `olhoffSolve`, with `runtime.verbose = true` | `examples/Performance/performance_comparison.m` → `confbench_run_case` → `olhoffcurrent_run` → `olhoffSolve`, with `runtime.verbose = false` |
| read-only inputs | `res.mat` per mesh plus `evaluations/sweep_verification.json` | tapped `SOLVER_RESULT.mat`, runner `benchmark_records.mat` |

**Which values the task quoted.** The table in the task (ω₁/ω₂ 169.7/170.8, gap 0.7 %, …) lists the upstream **eq. (4) re-evaluation** of the final designs: SIMP p = 3 stiffness, eq. (4) mass with cut-off 0.1, 3 `eigs` modes (`sd_verify_sweeps.m`). It does not list the native Pedersen/linear-mass frequencies. Both are compared below, like with like. The M_nd column in the task is the native M_nd of the design.

## 1. Per-mesh comparison

Native ω and eq. (4) ω are given as old = new wherever they agree bit for bit, which is every case here.

| mesh | native ω₁ old = new | native ω₂ old = new | eq. (4) ω₁ / ω₂ old = new (task table) | eq. (4) gap old = new (task) | outer old / new | inner old / new | M_nd old = new (task) | ρ SHA-256 | ω and nInner histories | wall old → new [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 169.210576386275 | 170.399975755934 | 169.739 / 170.843 (169.7 / 170.8) | 0.651 % (0.7 %) | 121 / 121 | 2369 / 2369 | 0.114616 (0.115) | identical `2a1c0d0a…` | identical | 406 → 224 |
| 240x30 | 167.342446131309 | 187.160188383023 | 167.623 / 187.240 (167.6 / 187.2) | 11.703 % (11.7 %) | 111 / 111 | 2077 / 2077 | 0.122668 (0.123) | identical `bc8fa779…` | identical | 708 → 355 |
| 320x40 | 165.856795502549 | 195.140032893032 | 166.138 / 195.266 (166.1 / 195.3) | 17.533 % (17.5 %) | 101 / 101 | 2001 / 2001 | 0.140618 (0.141) | identical `f246a943…` | identical | 963 → 427 |
| 400x50 | 166.455175955396 | 198.131780709087 | 166.729 / 198.239 (166.7 / 198.2) | 18.899 % (18.9 %) | 93 / 93 | 1913 / 1913 | 0.121592 (0.122) | identical `34c618a3…` | identical | 1255 → 570 |
| 480x60 | 166.009257697589 | 203.441222082928 | 166.323 / 203.574 (166.3 / 203.6) | 22.397 % (22.4 %) | 112 / 112 | 2137 / 2137 | 0.130673 (0.131) | identical `acf85ed4…` | identical | 1958 → 981 |
| 560x70 | 165.810051163182 | 206.375980582232 | 166.137 / 206.516 (166.1 / 206.5) | 24.305 % (24.3 %) | 130 / 130 | 2371 / 2371 | 0.132904 (0.133) | identical `b69314df…` | identical | 2535 → 1425 |
| 640x80 | 165.649974556035 | 205.973049745131 | 165.961 / 206.079 (166.0 / 206.1) | 24.173 % (24.2 %) | 156 / 156 | 2868 / 2868 | 0.132972 (0.133) | identical `65eedc95…` | identical | 3476 → 2216 |
| 720x90 | 165.423366700756 | 202.462560354619 | 165.741 / 202.583 (165.7 / 202.6) | 22.229 % (22.2 %) | 204 / 204 | 3752 / 3752 | 0.161661 (0.162) | identical `857c95da…` | identical | 4805 → 3538 |
| 800x100 | 165.432215867589 | 195.739346342934 | 165.764 / 195.859 (165.8 / 195.9) | 18.156 % (18.2 %) | 246 / 246 | 4650 / 4650 | 0.164540 (0.165) | identical `b89af055…` | identical | 6169 → 4901 |

- **Design identity.** The final densities are **bit-identical** at all nine meshes: `isequal(ρ_old, ρ_new)`, mean |Δρ| = 0, and the SHA-256 recomputed from the old `res.mat` equals the recorded value.
- **Frequency differences.** The maximum relative difference over native ω₁₋₃ and eq. (4) ω₁₋₃ is **exactly 0** at every mesh.
- **Histories.** The full per-iteration ω history (J × outer) and the nInner history are identical.
- **The task table.** Every value in it matches this campaign's eq. (4) re-evaluation and native M_nd at the table's rounding.
- **Native gap.** The native terminal gap, from 0.70 % at 160x20 through 24.47 % to 18.32 % at 800x100, is 0.05–0.17 percentage points above the eq. (4) gap quoted in the task. The difference comes only from the reporting model, since the design is identical.

## 2. Classification (preregistered, PREREGISTRATION.md §11)

| difference | meshes | class |
|---|---|---|
| native ω₁, ω₂, ω₃; eq. (4) ω₁, ω₂, ω₃; outer count; inner count; terminal status; ρ; histories | all 9 | **negligible/numerical**: exact equality |
| task-table ω and gap (eq. (4) re-evaluation, one decimal) against the native values of this report | all 9 | **plausibly environmental/reporting**: a reporting-model and rounding difference only |
| wall time (new 0.44–0.79× old) | all 9 | **plausibly environmental/reporting**: identical arithmetic, different execution context |
| `runtime.verbose` (true in the old sweep, false in production) | all 9 | **plausibly environmental/reporting**: console-only row; bit-identical results confirm it has no effect |
| anything scientifically material | — | **none** |

## 3. Qualitative topology correspondence

The designs are identical, so the correspondence is exact, including the features listed in TOPOLOGY_AUDIT.md:

- left–right asymmetry at 320x40 and 800x100;
- extra gray end-bay braces at 720x90;
- the grayness step at the two finest meshes.

These features belong to the formulation and its deterministic trajectory, not to the migration or to the benchmark path.

## 4. What this establishes

1. **Migration and harness fidelity at all nine meshes.** The production path `performance_comparison.m` → `olhoffcurrent_run` runs exactly the computation of the upstream research sweep, bit for bit and not only within tolerance. It adds a path guard, preset resolution, accounting, and a pre-solve configuration hash. This extends the gate's single 160x20 anchor to the whole series.
2. **Reproducibility across process, entry point and time.** The old and new runs used different entry scripts, process sessions and `verbose` settings, with wall times differing by up to 2.3×, yet produced identical results. The single-threaded computation is deterministic under these conditions.
3. **Material differences between the sweeps: none.** Every property this campaign reports as notable was already present, unreported or unexamined, in the earlier sweep: the fine-mesh grayness step, the ω₂ decline, the asymmetric designs, the growing outer count. They are properties of the formulation.
