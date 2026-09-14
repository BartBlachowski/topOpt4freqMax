# BASELINE — repository state before the cleanup migration

Recorded 2026-09-14 12:33:33 CEST by the repository-cleanup task, before any file was moved.

| | |
|---|---|
| repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| branch | `benchmark-methodology-r2` |
| HEAD | `05919933f3f4af68b36c1981e66130709a263593` — Fable restauration migration |
| linked worktree | `/Users/piotrek/Programming/topOpt4freqMax-migration-253069` @ b21483b (`migration/olhoffcurrent-upstream-253069`) — not touched |
| tracked files (`git ls-files`) | 3159 |
| files on disk (excl. `.git`, `.venv`) | 6073 (9.41 GB) |
| empty directories on disk | 8 |

Companion files (machine-readable):

* `BASELINE_INVENTORY.tsv` — every file on disk: `path<TAB>size<TAB>sha1` (ignored and untracked files included; `.git/` and `.venv/` excluded).
* `BASELINE_EMPTY_DIRS.txt` — empty directories (git cannot track them).
* `BASELINE_GIT_INDEX.tsv` — `git ls-files -s` (mode, blob id, stage, path).

## `git status --short` at start

```
?? docs/bimodality_gap/
?? examples/Performance/compose_nine_mesh_comparison.m
?? examples/Performance/conference_benchmark/nine_mesh_comparison_pedersen_b21483b/
?? examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/preview_table.tex
```

## Live processes at start that touch this repository

A MATLAB R2025b `-batch` job (PID 65691, started 11:01) was running arm `budget400` at 800×100 from
`docs/bimodality_gap/scripts`, calling `analysis/OlhoffCurrent`. Several other Claude Code sessions were open
(two on this repository, both idle). See VERIFICATION.md §0 for how the migration was sequenced around this.

## Top level

| entry | kind | tracked files |
|---|---|---|
| `.DS_Store` | file | 0 |
| `.gitignore` | file | 1 |
| `.venv` | dir | 0 |
| `BENCHMARK_FAIRNESS_AUDIT.md` | file | 1 |
| `BENCHMARK_PROTOCOL_R3.md` | file | 1 |
| `DIAGNOSTIC_REPRO2007_BENCHMARK.md` | file | 1 |
| `MIGRATION_REPRODUCTION2007_REPORT.md` | file | 1 |
| `Matlab` | dir | 77 |
| `OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md` | file | 1 |
| `OLHOFF_NATIVE_CONVERGENCE_DETECTOR.md` | file | 1 |
| `OLHOFF_PRACTICAL_CONVERGENCE_AUDIT.md` | file | 1 |
| `OlhoffFable.txt` | file | 0 |
| `PROPOSED_NATIVE_PROFILE_AUDIT.md` | file | 1 |
| `README.md` | file | 1 |
| `THREE_METHOD_PARAMETRIC_STUDY.md` | file | 1 |
| `analysis` | dir | 2822 |
| `development` | dir | 0 |
| `docs` | dir | 7 |
| `examples` | dir | 213 |
| `main.log` | file | 0 |
| `paper` | dir | 0 |
| `phase5_evidence` | dir | 0 |
| `references` | dir | 0 |
| `results` | dir | 0 |
| `scripts` | dir | 0 |
| `source_of_truth` | dir | 3 |
| `study_preregistration.json` | file | 1 |
| `tests` | dir | 0 |
| `texput.log` | file | 0 |
| `tools` | dir | 25 |

## `analysis/` (one level)

| entry | kind | tracked files | files on disk |
|---|---|---|---|
| `.DS_Store` | file | 0 | 1 |
| `LabandaApproach` | dir | 5 | 6 |
| `OLHOFF_CURRENT_EFFECTIVE_CONFIG_AUDIT.md` | file | 1 | 1 |
| `OLHOFF_CURRENT_PROMOTION_REPORT.md` | file | 1 | 1 |
| `OLHOFF_IMPLEMENTATION_MAP.md` | file | 1 | 1 |
| `OLHOFF_IMPLEMENTATION_STATUS.md` | file | 1 | 1 |
| `OLHOFF_SOURCE_LINEAGE_AUDIT.md` | file | 1 | 1 |
| `OlhoffApproach` | dir | 13 | 13 |
| `OlhoffApproachExact` | dir | 53 | 53 |
| `OlhoffArchive.zip` | file | 0 | 1 |
| `OlhoffCurrent` | dir | 2114 | 2488 |
| `OlhoffM4Reconstruction` | dir | 45 | 45 |
| `OlhoffRegularized` | dir | 35 | 35 |
| `OlhoffReproduced2007` | dir | 5 | 5 |
| `YukselApproach` | dir | 14 | 16 |
| `elastic2D` | dir | 2 | 4 |
| `iteration_count_audit` | dir | 5 | 33 |
| `iteration_efficiency_evaluator_discontinuity_audit` | dir | 10 | 22 |
| `iteration_efficiency_final` | dir | 125 | 137 |
| `iteration_efficiency_final_audit` | dir | 13 | 13 |
| `iteration_efficiency_final_blocker_fix` | dir | 13 | 13 |
| `iteration_efficiency_methodology_audit` | dir | 9 | 11 |
| `iteration_efficiency_methodology_delta_audit` | dir | 7 | 9 |
| `iteration_efficiency_methodology_final_recheck` | dir | 3 | 4 |
| `iteration_efficiency_phase2a` | dir | 43 | 46 |
| `iteration_efficiency_phase2b_precision` | dir | 8 | 24 |
| `iteration_efficiency_phase2b_recheck` | dir | 22 | 46 |
| `iteration_efficiency_phase2d_delta_audit` | dir | 33 | 56 |
| `iteration_efficiency_phase2d_evaluator_amendment` | dir | 12 | 28 |
| `iteration_efficiency_phase2f_evaluator_redesign` | dir | 21 | 56 |
| `iteration_efficiency_phase2g_evaluator_selection_audit` | dir | 17 | 45 |
| `iteration_efficiency_phase2h_c_refreeze` | dir | 23 | 25 |
| `iteration_efficiency_phase2i_precision_qualification` | dir | 46 | 46 |
| `iteration_efficiency_study_design` | dir | 15 | 17 |
| `olhoff_fixed_budget_audit` | dir | 6 | 6 |
| `olhoff_native_convergence` | dir | 8 | 8 |
| `olhoff_nested_mma_route_audit` | dir | 16 | 30 |
| `olhoff_practical_convergence_audit` | dir | 2 | 2 |
| `olhoff_stabilization_audit` | dir | 15 | 15 |
| `ourApproach` | dir | 2 | 7 |
| `performance_campaign_forensic_audit` | dir | 5 | 55 |
| `performance_campaign_targeted_replays` | dir | 33 | 62 |
| `three_method_parametric_study` | dir | 19 | 19 |

## `examples/` (one level)

| entry | kind | tracked files | files on disk |
|---|---|---|---|
| `.DS_Store` | file | 0 | 1 |
| `Building` | dir | 4 | 228 |
| `ClampedBeam` | dir | 3 | 227 |
| `ClampedHingedBeam` | dir | 3 | 227 |
| `HingedBeam` | dir | 3 | 231 |
| `Performance` | dir | 144 | 325 |
| `Revision_v1` | dir | 23 | 675 |
| `check_edof_and_harmonic_sensitivity_optionB.m` | file | 1 | 1 |
| `conference_benchmark.zip` | file | 0 | 1 |
| `conference_benchmark_v1` | dir | 24 | 95 |
| `demo_final_visualization_quality.m` | file | 1 | 1 |
| `elastic2D` | dir | 3 | 5 |
| `test_multi_load_cases_ourApproach.m` | file | 1 | 1 |
| `test_new_bc_types.m` | file | 1 | 1 |
| `test_passive_regions_rect.m` | file | 1 | 1 |
| `topopt_config_correlation.csv` | file | 0 | 1 |
| `weightedTopologyResultsHelper.m` | file | 1 | 1 |

## `docs/` (one level)

| entry | kind | tracked files | files on disk |
|---|---|---|---|
| `(Kennedy2021)TopologyOptimizationBenchmarkProblemsforAssessingthePerformanceofOptimizationAlgorithms.pdf.pdf` | file | 0 | 0 |
| `(Labanda2014)Benchmarking_optimization_solvers for_structural_topology.pdf` | file | 0 | 0 |
| `(Nocedal)_Numerical_Optimisation.pdf` | file | 0 | 0 |
| `.DS_Store` | file | 0 | 1 |
| `ARPACK.pdf` | file | 0 | 1 |
| `Angelucci (2022) - Topology optimization of multi‑story buildings under fully non‑stationary stochastic seismic ground motion.pdf` | file | 0 | 0 |
| `Arora and Wang - 2005 - Review of formulations for structural and mechanic.pdf` | file | 0 | 1 |
| `Bau_Numerical_linear_Algebra_BOOK.pdf` | file | 0 | 1 |
| `Chung (2018) - Implementation of topology optimization using openMDAO.pdf` | file | 0 | 0 |
| `Chung (2018) - Topology optimization in OpenMDAO.pdf` | file | 0 | 0 |
| `DatasetDacl10kLink copy.txt` | file | 0 | 1 |
| `DatasetDacl10kLink.txt` | file | 0 | 1 |
| `Gao(2020)_Level-set_topology_optimization_review.pdf` | file | 0 | 0 |
| `Giga-voxel_computation_morphogenesis_for_structural_design.pdf` | file | 0 | 1 |
| `Gomez (2019) - Topology optimization framework for structures subjected to stationary stochastic dynamic loads.pdf` | file | 0 | 0 |
| `Gray (2019) - OpenMDAO-an open-source framework for multidisciplinary design, analysis, and optimization.pdf` | file | 0 | 0 |
| `Holmberg(2013)-Stress constrained topology optimization.pdf` | file | 0 | 0 |
| `Huang(2009)_Evolutionary_topological_optimisation_of_vibrationg_continuum_structures_for_natural_frequencies.pdf` | file | 0 | 0 |
| `Klarbring(2009) - An Introduction to Structural Optimisation.pdf` | file | 0 | 0 |
| `Li(2021)-Topology optimization of vibrating structures with frequency band constraints.pdf` | file | 0 | 0 |
| `Lin(2022)_A multi-step relay implementation of the successive iteration of analysis and design method for large-scale natural frequency-related topology optimization - s00466-023-02372-1.pdf` | file | 0 | 0 |
| `Nguyen (2015) - Isogeometric analysis; An overview and computer implementation aspects.pdf` | file | 0 | 0 |
| `Nlpql.f` | file | 1 | 1 |
| `Pozo (2023) - TopSTO; a 115-line code for topology optimization of structures under stationary stochastic dynamic loading.pdf` | file | 0 | 0 |
| `Rojas(2015)_DTU_Mathematical_programming_methods.pdf` | file | 0 | 0 |
| `SQP_(Labanda2016)An_efficient_second-order_SQP_method.pdf` | file | 0 | 0 |
| `SQP_(Morales2012)A_sequential_quadratic_programming_algorithm_with_an_additional_equality_constrained_phase_a-sequential-quadratic-programming-algorithm-with-an-5ckaoebdch.pdf` | file | 0 | 0 |
| `SQP_(Murray2005)_SNOPT_Large-scale.pdf` | file | 0 | 0 |
| `Shin(2023)_Topology optimization via machine learning.pdf` | file | 0 | 0 |
| `Teimouri (2109) - Multi-objective BESO topology optimization for stiffness and frequency of continuum structures.pdf` | file | 0 | 0 |
| `Wang (2021) - A comprehensive review of educational articles on structural and multidisciplinary optimization.pdf` | file | 0 | 0 |
| `Xu(2019)_Level-set_topology_optimization.pdf` | file | 0 | 0 |
| `Xue(2019)-Topology optimization under finite deformation via Moving Morphable Void (MMV) approach.pdf` | file | 0 | 0 |
| `bimodality_gap` | dir | 0 | 243 |
| `complexity` | dir | 1 | 1 |
| `complexity.tex` | file | 1 | 1 |
| `ec-03-2025-0278en.pdf` | file | 0 | 1 |
| `olhoff_audit.md` | file | 1 | 1 |
| `olhoff_implementation_analysis.tex` | file | 1 | 1 |
| `olhoff_penalty_continuation_experiment.md` | file | 1 | 1 |
| `s00158-007-0101-y.pdf` | file | 0 | 1 |
| `s00158-007-0167-6.pdf` | file | 0 | 1 |
| `s00158-010-0594-7.pdf` | file | 0 | 1 |
| `s00158-018-2159-0.pdf` | file | 0 | 1 |
| `s00158-025-04186-6.pdf` | file | 0 | 1 |
| `s001580050130.pdf` | file | 0 | 1 |
| `s11831-021-09544-3.pdf` | file | 0 | 1 |
| `sequential_approximation_rationale.aux` | file | 0 | 1 |
| `sequential_approximation_rationale.log` | file | 0 | 1 |
| `sequential_approximation_rationale.pdf` | file | 0 | 1 |
| `topcut.pdf` | file | 0 | 1 |
| `topopt_config.schema.json` | file | 1 | 1 |

## `examples/Performance/` (one level)

| entry | kind | tracked files |
|---|---|---|
| `.DS_Store` | file | 0 |
| `CONFERENCE_BENCHMARK_PREFLIGHT.md` | file | 1 |
| `CONFERENCE_DRIVER_FINAL_AUDIT.md` | file | 1 |
| `FINAL_CAMPAIGN_PREFLIGHT.md` | file | 1 |
| `PLAN_two_table_redesign.md` | file | 1 |
| `STOP_RULE_AUDIT.md` | file | 1 |
| `WP0A_VISUALIZATION_FIX.md` | file | 1 |
| `benchmark_protocol_r3.json` | file | 1 |
| `benchmark_results.json` | file | 1 |
| `compose_nine_mesh_comparison.m` | file | 0 |
| `conference_bench` | dir | 17 |
| `conference_benchmark` | dir | 52 |
| `conference_benchmark.zip` | file | 0 |
| `determinism_validation.json` | file | 1 |
| `diagnostic_yuksel_table1` | dir | 2 |
| `equivalence` | dir | 26 |
| `extension_invariance_validation.json` | file | 1 |
| `final_campaign` | dir | 3 |
| `final_campaign_config.m` | file | 1 |
| `final_campaign_preflight.m` | file | 1 |
| `final_campaign_run_case.m` | file | 1 |
| `fit_complexity_model.m` | file | 1 |
| `history_logging_validation.json` | file | 1 |
| `instrumentation_validation.json` | file | 1 |
| `ledger` | dir | 4 |
| `legacy_r3` | dir | 2 |
| `olhoff_benchmark_path_hash.m` | file | 1 |
| `olhoff_equivalence_gate.m` | file | 1 |
| `olhoff_equivalence_report.m` | file | 1 |
| `olhoff_preflight.m` | file | 1 |
| `performance_benchmark_profile.m` | file | 1 |
| `performance_comparison.json` | file | 1 |
| `performance_comparison.m` | file | 1 |
| `performance_log.txt` | file | 0 |
| `plot_table1_complexity.m` | file | 1 |
| `print_complexity_fit_table.m` | file | 1 |
| `print_table1_paper_style.m` | file | 1 |
| `regenerate_from_csv.m` | file | 1 |
| `repro2007_direct_cfg.m` | file | 1 |
| `repro2007_normalized_config.m` | file | 1 |
| `repro2007_tree_hash.m` | file | 1 |
| `sha256_hex.m` | file | 1 |
| `table1_complexity_fit.csv` | file | 0 |
| `table1_complexity_fit.png` | file | 0 |
| `table1_complexity_fit_fixedexp.csv` | file | 0 |
| `table1_complexity_fit_fixedexp.png` | file | 0 |
| `table1_complexity_fit_fixedexp_linear.png` | file | 0 |
| `table1_complexity_fit_linear.png` | file | 0 |
| `table1_paper_style.pdf` | file | 0 |
| `table1_paper_style.tex` | file | 1 |
| `table1_performance.csv` | file | 0 |
| `test_shared_topology_renderer.m` | file | 1 |
| `validate_determinism.m` | file | 1 |
| `validate_extension_invariance.m` | file | 1 |
| `validate_history_logging.m` | file | 1 |
| `validate_instrumentation_invariance.m` | file | 1 |
| `verify_repro2007_benchmark_equivalence.m` | file | 1 |
