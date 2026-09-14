#!/usr/bin/env python3
"""Single source of the repository-cleanup classification.

Every rule maps an OLD repository path (a file, or a directory prefix) to a NEW
path.  A file on disk is governed by the rule with the LONGEST matching old
prefix, so a specific rule (e.g. analysis/OlhoffCurrent/diagnostics) overrides
the rule of its parent (analysis/OlhoffCurrent -> analysis/Olhoff).

    python3 manifest_rules.py manifest     # write MIGRATION_MANIFEST.tsv (+ coverage check)
    python3 manifest_rules.py filemap      # write MIGRATION_FILEMAP.tsv  (every baseline file)
    python3 manifest_rules.py plan         # print the ordered move plan (no side effects)

Nothing here moves a file.  The move plan is executed by execute_migration.py,
which applies the same ordering and was dry-run before it was executed.
"""
import csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

CUR, HIST = 'current', 'historical'
DEV = 'development'

# (old_path, new_path, classification, current_or_historical, reason, dependencies_found)
R = []
def rule(old, new, cls, coh, reason, deps='none found'):
    R.append((old, new, cls, coh, reason, deps))

# ---------------------------------------------------------------- top level
rule('.gitignore', '.gitignore', 'current_documentation', CUR,
     'repository ignore rules; negation patterns for tracked evidence are re-pointed to the new locations',
     'negations name Matlab/reproduction2007, examples/Performance/equivalence, analysis/iteration_efficiency_phase2i_precision_qualification, examples/Revision_v1')
rule('.DS_Store', '.DS_Store', 'current_documentation', CUR, 'Finder metadata (ignored); stays with its directory')
rule('README.md', 'README.md', 'current_documentation', CUR,
     'main README; rewritten as navigation (current implementations, examples, runner, development/ warning)',
     'old text names examples/case1.json and examples/BeamTopOptFreq.json, neither exists')
rule('tools', 'tools', 'current_implementation', CUR,
     'shared current code: MATLAB/Python JSON dispatchers, MMA copy, plotting, BC/passive/load-case helpers. '
     'Kept at top level: run_topopt_from_json derives repoRoot from its own depth and every example/runner addpaths tools/Matlab',
     'run_topopt_from_json.m dispatches legacy keys Olhoff/OlhoffExact/OlhoffDu2007Repro into analysis/OlhoffApproach, analysis/OlhoffApproachExact, Matlab/reproduction2007 (retired in Phase 8); tools/Python/run_topopt_from_json.py -> analysis/ourApproach/Python (re-pointed)')
rule('paper', 'paper', 'current_documentation', CUR, 'active manuscript/review material (paper/reviews)')
rule('tests', 'tests', 'current_test', CUR,
     'repository-level tests; receives the development-firewall check. Method test suites stay beside their implementation (analysis/Olhoff/tests)')
rule('tests/__pycache__', f'{DEV}/experiments/Revision_v1_tests_pycache', 'experiment', HIST,
     'only residue of the removed tests/test_revision_v1_validator.py (compiled .pyc, ignored); belongs to the Revision_v1 experiment')
rule('scripts', f'{DEV}/experiments/Revision_v1_scripts', 'experiment', HIST,
     'scripts/revision_v1 holds only __pycache__ and cr2_smoke_results.mat (ignored) of the Revision_v1 experiment')
rule('phase5_evidence', f'{DEV}/experiments/Revision_v1_phase5_evidence', 'evidence', HIST,
     'A4 Phase 5 run logs/outputs of the paper-revision experiment (2026-07-24..27)')
rule('references', 'paper/references', 'current_documentation', CUR,
     'literature PDFs cited by the manuscript (Du2007, Olhoff2014, Yuksel2025, ...); untracked/ignored; grouped with the paper',
     'examples/Performance/performance_benchmark_profile.m (historical) names references/Yuksel2025_Efficient.pdf as metadata')
rule('results', f'{DEV}/legacy_examples/results', 'legacy_example', HIST,
     'June 2026 output figures of run_topopt_from_json (Olhoff/Yuksel/ourApproach/cantilever); ignored. run_topopt_from_json recreates results/ on demand (mkdir)')
rule('source_of_truth', f'{DEV}/legacy_implementations/source_of_truth', 'legacy_implementation', HIST,
     'reference codes top88.m, top88_model.m, topcut.m used to seed early implementations (tracked since 2026-02-27); no current caller',
     'analysis/YukselApproach/Python/README.md mentions source_of_truth/fromPaper/Yuksel/ (path no longer exists)')
rule('Matlab', f'{DEV}/reconstruction/Matlab', 'reconstruction', HIST,
     'Matlab/reproduction2007 = clean-room Du-Olhoff 2007 reproduction (HISTORICAL in OLHOFF_IMPLEMENTATION_MAP, blocked from production); README.md, full_coupling/, legacy/ are signposts to historical generations',
     'run_topopt_from_json key OlhoffDu2007Repro (retired); olhoffcurrent_forbidden_paths; test_path_isolation/test_source_integrity/confbench_selftest use it as a CONTAMINANT; .gitignore negations')
for f, why in [
    ('BENCHMARK_FAIRNESS_AUDIT.md', 'benchmark methodology audit (2026-08)'),
    ('BENCHMARK_PROTOCOL_R3.md', 'protocol of the superseded r3 benchmark'),
    ('DIAGNOSTIC_REPRO2007_BENCHMARK.md', 'diagnosis of the repro2007 benchmark path'),
    ('MIGRATION_REPRODUCTION2007_REPORT.md', 'migration report of Matlab/reproduction2007'),
    ('OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md', 'r3-era Olhoff benchmark-path equivalence report'),
    ('OLHOFF_NATIVE_CONVERGENCE_DETECTOR.md', 'native-convergence detector audit'),
    ('OLHOFF_PRACTICAL_CONVERGENCE_AUDIT.md', 'practical-convergence audit'),
    ('PROPOSED_NATIVE_PROFILE_AUDIT.md', 'Proposed native-profile audit'),
    ('THREE_METHOD_PARAMETRIC_STUDY.md', 'report of the profile-calibration study'),
    ('study_preregistration.json', 'preregistration of the profile-selection study (named as selection_rule_source inside the frozen profile manifest, never read by code)'),
    ('OlhoffFable.txt', 'pasted session notes (worktree explanation, sweep summaries), 2026-09-14; ignored'),
    ('main.log', 'LaTeX/tool log (ignored)'),
    ('texput.log', 'LaTeX log (ignored)')]:
    rule(f, f'{DEV}/historical_documents/repository_root/{f}', 'historical_document', HIST, why)

# ---------------------------------------------------------------- analysis/
rule('analysis', 'analysis', 'current_implementation', CUR, 'current method implementations only after migration')
rule('analysis/.DS_Store', 'analysis/.DS_Store', 'current_documentation', CUR, 'Finder metadata (ignored)')

# Olhoff: the promoted production tree takes the canonical name
rule('analysis/OlhoffCurrent', 'analysis/Olhoff', 'current_implementation', CUR,
     'SOLE production Du-Olhoff implementation (SOURCE_OF_TRUTH.md section 1). Renamed to the canonical name; function names keep the olhoffcurrent_ prefix',
     'performance_comparison.m, compose_nine_mesh_comparison.m, conference_bench/*, docs/bimodality_gap/scripts/bg_run_arm.m; string literals analysis/OlhoffCurrent in olhoffcurrent_config/run/provenance/source_manifest(write)/currentness(message) and confbench_manifest')
rule('analysis/OlhoffCurrent/diagnostics', f'{DEV}/reconstruction/OlhoffCurrent_diagnostics', 'audit', HIST,
     '30 diagnostic/audit/migration studies performed on the production tree (move-stop, three-rung, gray-KKT, nine-mesh campaign audits, upstream migrations, c480 SOCP causal run)',
     'olhoffcurrent_presets.m cites four study paths as provenance strings (not file access); PROVENANCE.json acceptance_evidence cites two (record, unchanged); test_finalization_gate.m reads two_branch_controller_validation (archived with it)')
rule('analysis/OlhoffCurrent/evidence', f'{DEV}/reconstruction/OlhoffCurrent_evidence', 'evidence', HIST,
     'git-ignored raw evidence of those studies (1.3 GB)',
     'EVIDENCE.json files declare evidenceRoot analysis/OlhoffCurrent/evidence/<study> (historical record); docs/bimodality_gap/scripts/bg_extract_prior_interventions.m reads move_activity_400')
for f in ['olhoffcurrent_finalization_gate.m', 'olhoffcurrent_evidence_gate.m',
          'olhoffcurrent_evidence_declare.m', 'EVIDENCE_POLICY.md',
          'tests/test_finalization_gate.m', 'tests/gate_provenance_probes.m',
          'tests/test_evidence_retention.m']:
    rule(f'analysis/OlhoffCurrent/{f}', f'{DEV}/reconstruction/OlhoffCurrent_study_governance/{f}', 'audit', HIST,
         'study-finalization tooling for diagnostics/ and evidence/: callers are only each other and scripts under diagnostics/ (no production script, benchmark file or other test); hard-bound to the pre-migration diagnostics/evidence layout and to git objects at analysis/OlhoffCurrent/... in historical commits; archived with the studies it governs',
         'uses diagnostics/two_branch_controller_validation, evidence/_gate_selftest, a git clone of the repository at analysis/OlhoffCurrent')

rule('analysis/OLHOFF_CURRENT_EFFECTIVE_CONFIG_AUDIT.md', f'{DEV}/migration_history/olhoff_current_promotion/OLHOFF_CURRENT_EFFECTIVE_CONFIG_AUDIT.md', 'migration_artifact', HIST, 'effective-config audit of the 2026-09-07 promotion')
rule('analysis/OLHOFF_CURRENT_PROMOTION_REPORT.md', f'{DEV}/migration_history/olhoff_current_promotion/OLHOFF_CURRENT_PROMOTION_REPORT.md', 'migration_artifact', HIST, 'promotion evidence; superseded as a navigation document by README.md and SOURCE_OF_TRUTH.md', 'linked from analysis/OlhoffCurrent/README.md (link updated)')
rule('analysis/OLHOFF_IMPLEMENTATION_MAP.md', f'{DEV}/migration_history/olhoff_current_promotion/OLHOFF_IMPLEMENTATION_MAP.md', 'migration_artifact', HIST, 'classification of 14 Olhoff trees at promotion; superseded by SOURCE_OF_TRUTH.md + development/README.md', 'linked from OlhoffCurrent/README.md, cited in comments of olhoffcurrent_forbidden_paths.m and performance_comparison.m')
rule('analysis/OLHOFF_IMPLEMENTATION_STATUS.md', f'{DEV}/migration_history/olhoff_current_promotion/OLHOFF_IMPLEMENTATION_STATUS.md', 'migration_artifact', HIST, 'pre-promotion status (names M4 conference-active; stale, superseded by the map)')
rule('analysis/OLHOFF_SOURCE_LINEAGE_AUDIT.md', f'{DEV}/migration_history/olhoff_current_promotion/OLHOFF_SOURCE_LINEAGE_AUDIT.md', 'migration_artifact', HIST, 'read-only lineage audit supporting the promotion')

rule('analysis/OlhoffApproach', f'{DEV}/legacy_implementations/OlhoffApproach', 'legacy_implementation', HIST,
     'original bound-formulation MMA implementation + Python port; HISTORICAL in the implementation map; blocked from production',
     'run_topopt_from_json key Olhoff (retired); analysis/ourApproach/Python/topopt_freq.py and analysis/elastic2D/Python/topopt_elastic2D.py try to import mma.py from OlhoffApproach/Python -- that file does not exist on disk or in git at baseline (pre-existing dangling optional import, behaviour unchanged)')
rule('analysis/OlhoffApproachExact', f'{DEV}/legacy_implementations/OlhoffApproachExact', 'legacy_implementation', HIST,
     'the "exact Olhoff 2014" line (own FE, multiplicity, generalized gradients); HISTORICAL',
     'run_topopt_from_json key OlhoffExact (retired)')
rule('analysis/OlhoffArchive.zip', f'{DEV}/legacy_implementations/OlhoffArchive.zip', 'legacy_implementation', HIST,
     '552 MB local zip backup of Olhoff trees (ignored, never tracked: exceeds GitHub 100 MB limit)')
rule('analysis/OlhoffM4Reconstruction', f'{DEV}/reconstruction/OlhoffM4Reconstruction', 'reconstruction', HIST,
     'frozen conference reconstruction (FROZEN_EVIDENCE, hash-pinned by IMPORT_MANIFEST.json); production before 2026-09-07',
     'olhoffcurrent_forbidden_paths; test_path_isolation TEST B contaminant; olhoffcurrent_presets provenance string')
rule('analysis/OlhoffReproduced2007', f'{DEV}/reconstruction/OlhoffReproduced2007', 'reconstruction', HIST,
     'thin runner exposing Matlab/reproduction2007 on Yuksel geometries; HISTORICAL', 'calls repro2007_paths (moves with it)')
rule('analysis/OlhoffRegularized', f'{DEV}/experiments/OlhoffRegularized', 'experiment', HIST,
     'globalized variant on reproduction2007 primitives with its own audit; "superseded", self-contained')
rule('analysis/LabandaApproach', f'{DEV}/experiments/LabandaApproach', 'experiment', HIST,
     'TopIQP/reciprocal-SQP compliance-minimization experiment (2026-05); not dispatched by any runner, no example, no test, not one of the three compared methods')
rule('analysis/YukselApproach', 'analysis/Yuksel', 'current_implementation', CUR,
     'sole Yuksel implementation, dispatched by run_topopt_from_json key Yuksel; named by the frozen profile (SOURCE_OF_TRUTH.md section 2); canonical name',
     'tools/Matlab/run_topopt_from_json.m addpath; confbench_manifest.m hash list; Python runners import package YukselApproach.Python.solver (updated)')
rule('analysis/ourApproach', 'analysis/Proposed', 'current_implementation', CUR,
     'sole Proposed implementation, dispatched by key ourApproach (MATLAB + Python); named by the frozen profile (SOURCE_OF_TRUTH.md section 3); canonical name',
     'tools/Matlab/run_topopt_from_json.m addpath; tools/Python/run_topopt_from_json.py solver_dir; confbench_manifest.m hash list')
rule('analysis/elastic2D', 'analysis/elastic2D', 'current_implementation', CUR,
     'auxiliary compliance-minimization solver, dispatched by run_topopt_from_json keys elastic2D/elastc2D; current example examples/elastic2D',
     'Python optional mma import from OlhoffApproach/Python (dangling at baseline, see OlhoffApproach row)')

for d, why in [
    ('iteration_count_audit', 'iteration-count audit'),
    ('olhoff_fixed_budget_audit', 'fixed-budget audit runners (AUDIT_ONLY)'),
    ('olhoff_native_convergence', 'olhoffOptTelemetry / nativeConvergenceDetector (AUDIT_ONLY)'),
    ('olhoff_nested_mma_route_audit', 'nested-MMA route audit, Python + reports (AUDIT_ONLY)'),
    ('olhoff_practical_convergence_audit', 'practical-convergence audit runner (AUDIT_ONLY)'),
    ('olhoff_stabilization_audit', 'superseded S1 stabilization profile, dispatched by the previous (r3) benchmark driver'),
    ('performance_campaign_forensic_audit', 'forensic audit of an earlier performance campaign')]:
    rule(f'analysis/{d}', f'{DEV}/audits/{d}', 'audit', HIST, why,
         'olhoffcurrent_forbidden_paths names the olhoff_* ones; historical Performance scripts (moving too) reference olhoff_stabilization_audit')
rule('analysis/performance_campaign_targeted_replays', f'{DEV}/benchmark_history/performance_campaign_targeted_replays', 'benchmark_history', HIST, 'targeted replays of an earlier performance campaign')
for d in ['iteration_efficiency_evaluator_discontinuity_audit', 'iteration_efficiency_final',
          'iteration_efficiency_final_audit', 'iteration_efficiency_final_blocker_fix',
          'iteration_efficiency_methodology_audit', 'iteration_efficiency_methodology_delta_audit',
          'iteration_efficiency_methodology_final_recheck', 'iteration_efficiency_phase2a',
          'iteration_efficiency_phase2b_precision', 'iteration_efficiency_phase2b_recheck',
          'iteration_efficiency_phase2d_delta_audit', 'iteration_efficiency_phase2d_evaluator_amendment',
          'iteration_efficiency_phase2f_evaluator_redesign', 'iteration_efficiency_phase2g_evaluator_selection_audit',
          'iteration_efficiency_phase2h_c_refreeze', 'iteration_efficiency_phase2i_precision_qualification',
          'iteration_efficiency_study_design']:
    rule(f'analysis/{d}', f'{DEV}/experiments/iteration_efficiency/{d}', 'experiment', HIST,
         'phase of the iteration-efficiency study and its audits (2026-08/09); no current caller',
         'examples/Performance/test_shared_topology_renderer.m (historical) uses iteration_efficiency_study_design; .gitignore negation for phase2i')
rule('analysis/three_method_parametric_study', f'{DEV}/benchmark_history/three_method_parametric_study', 'benchmark_history', HIST,
     'profile-calibration study (stage A, holdout, engineering gates, selection); only its three frozen-profile files are current (split out below)',
     'historical Performance scripts (final_campaign_*, legacy_r3) reference it')
for f in ['study_base_config.m', 'study_evaluate_design.m', 'results/profile_freeze_manifest.json']:
    rule(f'analysis/three_method_parametric_study/{f}', f'examples/Performance/benchmark_profile/{os.path.basename(f)}',
         'current_implementation', CUR,
         'frozen Proposed/Yuksel benchmark profile + base-config builder + common E1/E2/E3 evaluator, read by the current runner; moved beside it byte-for-byte',
         'performance_comparison.m addpath; confbench_method_config, confbench_frozen_budget, confbench_preflight checks 7-8, confbench_manifest hash list (all re-pointed); compose_nine_mesh_comparison.m uses the OLD paths only as keys into recorded campaign manifests (unchanged)')

# ---------------------------------------------------------------- examples/
rule('examples', 'examples', 'current_example', CUR, 'current runnable examples only after migration')
rule('examples/.DS_Store', 'examples/.DS_Store', 'current_example', CUR, 'Finder metadata (ignored)')
for d in ['Building', 'ClampedBeam', 'ClampedHingedBeam', 'HingedBeam']:
    rule(f'examples/{d}', f'examples/{d}', 'current_example', CUR,
         'JSON config (approach ourApproach) + MATLAB runner (via weightedTopologyResultsHelper) + Python runner (tools.Python.run_topopt_from_json); image/fig/csv outputs are ignored run output kept in place',
         'MATLAB runner needs examples/ on the path for weightedTopologyResultsHelper')
rule('examples/elastic2D', 'examples/elastic2D', 'current_example', CUR, 'elastic2D JSON configs + Python runner')
rule('examples/weightedTopologyResultsHelper.m', 'examples/weightedTopologyResultsHelper.m', 'current_example', CUR,
     'helper called by the four current beam/building MATLAB runners')
for f in ['check_edof_and_harmonic_sensitivity_optionB.m', 'demo_final_visualization_quality.m',
          'test_multi_load_cases_ourApproach.m', 'test_new_bc_types.m', 'test_passive_regions_rect.m']:
    rule(f'examples/{f}', f'{DEV}/legacy_examples/examples_root/{f}', 'legacy_example', HIST,
         'ad-hoc developer check/demo (2026-02..03); addpaths <repo>/tools, which holds no .m since tools/*.m moved to tools/Matlab on 2026-02-23; referenced by no runner or suite')
rule('examples/topopt_config_correlation.csv', f'{DEV}/legacy_examples/examples_root/topopt_config_correlation.csv', 'legacy_example', HIST, 'correlation output of an old run (ignored)')
rule('examples/Performance/conference_benchmark.zip', f'{DEV}/benchmark_history/examples_Performance/conference_benchmark.zip', 'benchmark_history', HIST,
     'local zip backup of conference_benchmark (ignored, never tracked); missed by the first execution pass and moved in a second pass (MIGRATION_LOG.tsv)')
rule('examples/conference_benchmark.zip', f'{DEV}/benchmark_history/conference_benchmark.zip', 'benchmark_history', HIST, 'local zip backup (ignored)')
rule('examples/conference_benchmark_v1', f'{DEV}/benchmark_history/conference_benchmark_v1', 'benchmark_history', HIST, 'first conference benchmark version (superseded)')
rule('examples/Revision_v1', f'{DEV}/experiments/Revision_v1', 'experiment', HIST,
     'reviewer-demanded revision experiments (A1-A4, exp1-5; 2026-07); scripts call addpath(genpath(analysis)); not the current benchmark',
     'genpath(analysis) in six scripts; .gitignore negation for output/a4/*.png')

# examples/Performance: the directory stays, history inside it moves
rule('examples/Performance', 'examples/Performance', 'current_example', CUR,
     'current performance-comparison area: performance_comparison.m + conference_bench/ + benchmark_profile/ + compose_nine_mesh_comparison.m + current recorded campaigns')
for f in ['performance_comparison.m', 'conference_bench', 'compose_nine_mesh_comparison.m', '.DS_Store']:
    rule(f'examples/Performance/{f}', f'examples/Performance/{f}', 'current_example', CUR,
         {'performance_comparison.m': 'THE current three-method performance runner',
          'conference_bench': 'implementation of the current runner (config, run_case, preflight, manifest, export, plots, selftest)',
          'compose_nine_mesh_comparison.m': 'current composition of the three-method table (Proposed/Yuksel from campaign_9mesh_r2, Olhoff from nine_mesh_pedersen_b21483b); untracked at baseline',
          '.DS_Store': 'Finder metadata (ignored)'}[f],
         'paths re-pointed to analysis/Olhoff and examples/Performance/benchmark_profile' if f != '.DS_Store' else 'none found')
rule('examples/Performance/conference_benchmark', 'examples/Performance/conference_benchmark', 'current_example', CUR, 'output root of the current runner')
for d, why in [('campaign_9mesh_r2', 'current Proposed/Yuksel rows of the three-method table; read by compose_nine_mesh_comparison.m and test_preset_equivalence.m'),
               ('nine_mesh_pedersen_b21483b', 'current production Olhoff campaign (2026-09-14)'),
               ('nine_mesh_comparison_pedersen_b21483b', 'current three-method table composed from the two campaigns (untracked at baseline)'),
               ('.DS_Store', 'Finder metadata (ignored)')]:
    rule(f'examples/Performance/conference_benchmark/{d}', f'examples/Performance/conference_benchmark/{d}', 'current_example', CUR, why)
rule('examples/Performance/conference_benchmark/campaign_9mesh', f'{DEV}/benchmark_history/examples_Performance/conference_benchmark/campaign_9mesh', 'benchmark_history', HIST, 'empty output directory of the first (censored) nine-mesh campaign label')
rule('examples/Performance/conference_benchmark/smoke_nine_mesh_pedersen_b21483b', f'{DEV}/benchmark_history/examples_Performance/conference_benchmark/smoke_nine_mesh_pedersen_b21483b', 'benchmark_history', HIST, 'mechanics-only smoke run preceding the 2026-09-14 campaign; referenced only by the archived nine_mesh_pedersen_production diagnostic')
for f, why in [
    ('CONFERENCE_BENCHMARK_PREFLIGHT.md', 'preflight report of an earlier driver state'),
    ('CONFERENCE_DRIVER_FINAL_AUDIT.md', 'final audit of the driver before the OlhoffCurrent repointing'),
    ('FINAL_CAMPAIGN_PREFLIGHT.md', 'preflight of the r3 final_campaign'),
    ('PLAN_two_table_redesign.md', 'redesign plan (implemented)'),
    ('STOP_RULE_AUDIT.md', 'stop-rule audit (WP3)'),
    ('WP0A_VISUALIZATION_FIX.md', 'visualization work-package note'),
    ('benchmark_protocol_r3.json', 'r3 protocol'),
    ('benchmark_results.json', 'r3 results (listed as legacy evidence by confbench_preflight output-isolation check)'),
    ('determinism_validation.json', 'r3 validation record'),
    ('diagnostic_yuksel_table1', 'Yuksel Table 1 diagnostic'),
    ('equivalence', 'r3 Olhoff benchmark-path equivalence proofs (.mat tracked via .gitignore negation)'),
    ('extension_invariance_validation.json', 'r3 validation record'),
    ('final_campaign', 'r3 final_campaign output'),
    ('final_campaign_config.m', 'r3 driver helper (reads olhoff_stabilization_audit profile)'),
    ('final_campaign_preflight.m', 'r3 driver helper (addpaths Matlab/reproduction2007)'),
    ('final_campaign_run_case.m', 'r3 driver helper'),
    ('fit_complexity_model.m', 'r3 complexity fit (superseded by conference_bench/confbench_scaling_fit)'),
    ('history_logging_validation.json', 'r3 validation record'),
    ('instrumentation_validation.json', 'r3 validation record'),
    ('ledger', 'r3 protocol ledger / freeze record'),
    ('legacy_r3', 'previous driver preserved verbatim'),
    ('olhoff_benchmark_path_hash.m', 'r3 repro2007 benchmark-path hash'),
    ('olhoff_equivalence_gate.m', 'r3 equivalence gate (addpaths Matlab/reproduction2007)'),
    ('olhoff_equivalence_report.m', 'r3 equivalence report'),
    ('olhoff_preflight.m', 'second (repro2007) Olhoff chain retired by the 2026-09-07 promotion'),
    ('performance_benchmark_profile.m', 'r3 benchmark profile'),
    ('performance_comparison.json', 'JSON config of the pre-r3 runner'),
    ('performance_log.txt', 'July run log (ignored)'),
    ('plot_table1_complexity.m', 'Table-1 era plotting'),
    ('print_complexity_fit_table.m', 'Table-1 era table printer'),
    ('print_table1_paper_style.m', 'Table-1 era table printer'),
    ('regenerate_from_csv.m', 'Table-1 era regeneration'),
    ('repro2007_direct_cfg.m', 'repro2007 benchmark config'),
    ('repro2007_normalized_config.m', 'repro2007 benchmark config'),
    ('repro2007_tree_hash.m', 'repro2007 tree hash'),
    ('sha256_hex.m', 'hash helper of the r3 scripts (no current caller)'),
    ('table1_complexity_fit.csv', 'Table-1 era output'), ('table1_complexity_fit.png', 'Table-1 era output'),
    ('table1_complexity_fit_fixedexp.csv', 'Table-1 era output'), ('table1_complexity_fit_fixedexp.png', 'Table-1 era output'),
    ('table1_complexity_fit_fixedexp_linear.png', 'Table-1 era output'), ('table1_complexity_fit_linear.png', 'Table-1 era output'),
    ('table1_paper_style.pdf', 'Table-1 era output'), ('table1_paper_style.tex', 'Table-1 era output'),
    ('table1_performance.csv', 'Table-1 era output (listed as legacy evidence by confbench_preflight)'),
    ('test_shared_topology_renderer.m', 'test of the iteration-efficiency renderer (addpaths analysis/iteration_efficiency_study_design)'),
    ('validate_determinism.m', 'r3 validation script'), ('validate_extension_invariance.m', 'r3 validation script'),
    ('validate_history_logging.m', 'r3 validation script'), ('validate_instrumentation_invariance.m', 'r3 validation script'),
    ('verify_repro2007_benchmark_equivalence.m', 'r3 repro2007 equivalence verification')]:
    rule(f'examples/Performance/{f}', f'{DEV}/benchmark_history/examples_Performance/{f}', 'benchmark_history', HIST,
         why + '; not called by performance_comparison.m or conference_bench/')

# ---------------------------------------------------------------- docs/
rule('docs', 'docs', 'current_documentation', CUR, 'current documentation only after migration')
rule('docs/.DS_Store', 'docs/.DS_Store', 'current_documentation', CUR, 'Finder metadata (ignored)')
rule('docs/topopt_config.schema.json', 'docs/topopt_config.schema.json', 'current_documentation', CUR, 'JSON schema of the run_topopt_from_json task files')
rule('docs/complexity.tex', 'docs/complexity.tex', 'current_documentation', CUR, 'technical note: eigenvalue vs linear solver complexity (background of the performance comparison)')
rule('docs/complexity', 'docs/complexity', 'current_documentation', CUR, 'companion of complexity.tex')
rule('docs/bimodality_gap', f'{DEV}/experiments/bimodality_gap', 'experiment', HIST,
     'single-factor study of the Olhoff density-field bimodality gap (2026-09-14, untracked); an experiment on the production tree, not user documentation. MOVED ONLY AFTER its live MATLAB job ended (VERIFICATION.md section 0)',
     'scripts addpath analysis/OlhoffCurrent and read analysis/OlhoffCurrent/{evidence,diagnostics} and examples/Performance/conference_benchmark (paths recorded as-is; study files not edited)')
for f, why in [('olhoff_audit.md', 'audit of OlhoffApproach/topFreqOptimization_MMA.m'),
               ('olhoff_implementation_analysis.tex', 'analysis of historical Olhoff implementations'),
               ('olhoff_penalty_continuation_experiment.md', 'SIMP penalty-continuation experiment (2026-06-18, refuted)'),
               ('sequential_approximation_rationale.aux', 'LaTeX build output (ignored)'),
               ('sequential_approximation_rationale.log', 'LaTeX build output (ignored)'),
               ('sequential_approximation_rationale.pdf', 'compiled note (ignored; source not in repository)'),
               ('Nlpql.f', 'NLPQL Fortran reference source added with the SQP/LabandaApproach experiment (c4f893a, 2026-05-12)'),
               ('DatasetDacl10kLink.txt', 'unrelated dataset-link note (ignored)'),
               ('DatasetDacl10kLink copy.txt', 'unrelated dataset-link note (ignored)')]:
    rule(f'docs/{f}', f'{DEV}/historical_documents/docs/{f}', 'historical_document', HIST, why)
DOC_PDFS = None  # filled from disk below: every other docs/*.pdf is literature

# ---------------------------------------------------------------- development/
rule('development', 'development', 'migration_artifact', HIST, 'archive root created by this migration')


def lit_rules():
    # from the BASELINE inventory, so the rules stay reproducible after the move
    names = set()
    with open(os.path.join(HERE, 'BASELINE_INVENTORY.tsv')) as fh:
        for line in fh:
            p = line.split('\t', 1)[0]
            if p.startswith('docs/') and p.count('/') == 1:
                names.add(p[len('docs/'):])
    for f in sorted(names):
        if f.lower().endswith('.pdf') and not any(r[0] == f'docs/{f}' for r in R):
            rule(f'docs/{f}', f'paper/references/library/{f}', 'current_documentation', CUR,
                 'literature PDF (ignored); moved out of docs/ into the paper reference library')


def govern(path, rules):
    best = None
    for r in rules:
        o = r[0]
        if path == o or path.startswith(o + '/'):
            if best is None or len(o) > len(best[0]):
                best = r
    return best


def new_path(path, r):
    return r[1] + path[len(r[0]):]


def baseline():
    rows = []
    with open(os.path.join(HERE, 'BASELINE_INVENTORY.tsv')) as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            rows.append((p[0], p[1], p[2]))
    return rows


def tracked():
    out = subprocess.run(['git', '-C', REPO, 'ls-files', '-z'], capture_output=True, check=True).stdout
    return set(x for x in out.decode().split('\0') if x)


def main(mode):
    lit_rules()
    base = baseline()
    empties = [l.strip() for l in open(os.path.join(HERE, 'BASELINE_EMPTY_DIRS.txt')) if l.strip()]
    git = set(x.split('\t', 1)[1] for x in open(os.path.join(HERE, 'BASELINE_GIT_INDEX.tsv')).read().splitlines())
    counts = {r[0]: [0, 0] for r in R}
    ungoverned = []
    for p, size, h in base:
        r = govern(p, R)
        if r is None:
            ungoverned.append(p); continue
        counts[r[0]][0] += 1
        counts[r[0]][1] += p in git
    for d in empties:
        if govern(d, R) is None: ungoverned.append(d + '/')
    if ungoverned:
        print('UNGOVERNED (would be uncertain):', *ungoverned[:40], sep='\n  ')
        sys.exit(2)
    if mode == 'manifest':
        with open(os.path.join(HERE, 'MIGRATION_MANIFEST.tsv'), 'w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t', lineterminator='\n')
            w.writerow(['old_path', 'classification', 'new_path', 'current_or_historical', 'reason',
                        'dependencies_found', 'action', 'files_on_disk_governed', 'tracked_files_governed'])
            for r in sorted(R, key=lambda r: r[0]):
                n, t = counts[r[0]]
                if r[0] == r[1]:
                    action = 'keep'
                elif t > 0:
                    action = 'git mv'
                else:
                    action = 'mv (no tracked files)'
                if r[3] == CUR and r[0] != r[1]:
                    action += ' + update current references'
                w.writerow([r[0], r[2], r[1], r[3], r[4], r[5], action, n, t])
        print(f'{len(R)} rules, {len(base)} baseline files and {len(empties)} empty dirs governed; 0 uncertain')
    elif mode == 'filemap':
        with open(os.path.join(HERE, 'MIGRATION_FILEMAP.tsv'), 'w') as fh:
            fh.write('old_path\tnew_path\tsize\tsha1\ttracked\n')
            for p, size, h in base:
                r = govern(p, R)
                fh.write(f'{p}\t{new_path(p, r)}\t{size}\t{h}\t{int(p in git)}\n')
        print('filemap written')
    elif mode == 'plan':
        # moves only; deepest old path first so children leave before their parent is renamed
        moves = [r for r in R if r[0] != r[1]]
        moves.sort(key=lambda r: (-r[0].count('/'), r[0]))
        for r in moves:
            n, t = counts[r[0]]
            if n == 0 and not any(e == r[0] or e.startswith(r[0] + '/') for e in empties) \
                    and not os.path.exists(os.path.join(REPO, r[0])):
                continue
            print(f"{'GIT' if t else 'FS '}\t{r[0]}\t{r[1]}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'manifest')
