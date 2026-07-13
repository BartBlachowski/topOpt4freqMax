# OlhoffApproachExact evidence-migration report

**Audit date:** 2026-07-13  
**Project scope:** `Revision_v1`, manuscript and reviewer-response planning,
comparison implementations, campaign launchers, result summaries, tables,
figures, and supporting documentation

## Scientific decision applied

The completed reconstruction campaign did not establish a paper-faithful
implementation of Du and Olhoff (2007). The diagnostics eliminated the tested
FE formulation, interpolation, sensitivity, generalized-gradient,
multiplicity, mode-tracking, optimizer-stabilization, persistent-MMA,
regularization, and support-interpretation explanations as sufficient causes
of the discrepancy. Benchmark under-specification or unpublished
implementation details remain the most plausible explanation.

The migration therefore applies four distinct dispositions:

1. **Remove** claims and tasks that require a canonical reconstruction,
   published-optimum gap, canonical speedup, or Exact-based scaling.
2. **Rewrite** implementation-specific comparisons as comparisons with the
   local `analysis/OlhoffApproach` implementation only.
3. **Replace** `OlhoffApproachExact` by `OlhoffApproach` only where an existing
   local production comparison remains scientifically meaningful. No new
   experiment was introduced.
4. **Archive** the reconstruction code, scripts, reports, and numerical
   artifacts as diagnostic history that is excluded from reviewer evidence.

Published Du--Olhoff algorithms and results remain valid literature context.
They are now explicitly separated from executable repository evidence.

## Audit coverage

The audit searched direct aliases (`OlhoffApproachExact`, `OlhoffExact`,
`olhoff_exact`, and `olhoffexact`) and semantic roles such as canonical,
paper-faithful, reference/benchmark implementation, optimum gap, speedup,
convergence comparison, and scaling comparison.

- A no-ignore scan of the final tree finds 114 text files with a direct Exact
  alias, including ignored historical files and this migration report. Every one is
  either inside a governed archive or uses the alias only to state exclusion,
  withdrawal, archive compatibility, or diagnostic provenance.
- `analysis/OlhoffApproachExact/**` contains 402 regular files (399 after
  excluding three `.DS_Store` metadata files). The new archive README governs
  every file in that tree, including code, logs, CSV, MAT, PNG, and Markdown
  artifacts.
- `analysis/OlhoffApproachExactOpus/**` contains 224 regular files (221 after
  excluding two Python bytecode files and one `.DS_Store` metadata file). Its
  root README governs every file in that tree under the same archive policy.
- Seven Exact result families under `examples/Revision_v1/output/` contain 131
  files. `output/OLHOFF_EXACT_ARCHIVE.md` governs all of them, and every
  reviewer-readable Markdown summary in those families has its own archive
  banner.
- The production master, every production stage, all JSON configurations, the
  general performance campaign, the MATLAB dispatcher, the manuscript, both
  comparison-planning documents, and the reviewer audit were inspected.
- Reviewer-supplied inputs `paper/reviews/review1.txt`,
  `paper/reviews/final_review_V1.tex`, and
  `paper/reviews/final_review_V2.tex` were deliberately left unchanged as
  immutable review records. Their requests are answered in the revised plan by
  withdrawal and local-only scoping, not by altering the review record.
- This report enumerates 74 intentional migration paths: 68 modified tracked
  files, four new files, and two edited but ignored historical text summaries.
  Validation also refreshed ignored compiler outputs and bytecode caches;
  those derived files are identified below and contain no independent revision
  content.

The 626 files in the two analysis archive trees and the 131 saved Exact-output
artifacts are classified by the exhaustive directory rules below rather than
repeating hundreds of individual filenames. No direct-alias file exists
outside those rules, the explicitly archived standalone scripts, the immutable
review inputs, or the exclusion/withdrawal statements listed in this report.

## Modified files, sections, and justifications

### Revision governance and scientific status

| File | Modified section | Justification |
|---|---|---|
| `OLHOFF_EXACT_MIGRATION_REPORT.md` | New complete file | Supplies the requested exhaustive file/section/disposition record, remaining TODOs, validation evidence, and final dependency verdict. |
| `README.md` | Project description | Defines `OlhoffApproach` as local and both Exact trees as archived non-evidence. |
| `NUMERICAL_BEHAVIOR_FREEZE.md` | Entire memo | Converts the former production freeze into a superseded archive record; historical settings are no longer production settings. |
| `REVISION_R1_STATUS.md` | Exact campaign gate, P1, MS, gate/artifact tables, conclusions, minimum work | Closes the reconstruction as non-evidence, makes P1 local-only, records the completed comparator migration, and removes the former frozen/passed production status. |
| `SCIENTIFIC_DECISION_MEMO.md` | Comparator verdict, performance decisions, unsafe evidence, response treatment | Makes the failed reconstruction verdict final, permanently withdraws canonical headlines, and separates possible future local metrics from canonical claims. |
| `examples/Revision_v1/revision_v1_update1.md` | Workstreams 5 and 7, response planning, execution tiers | Removes the canonical reproduction route and old headline-recovery tasks; retains only scientifically meaningful local-comparison work. |
| `examples/Revision_v1/revision_implementation_audit.md` | Comparator status, Exp1 interpretation, reviewer-demand matrix, minimum work | Reclassifies regenerated data as local, removes canonical-overhead planning, and resolves the canonical request by withdrawal. |
| `scripts/revision_v1/IMPLEMENTATION_MAP.md` | E2, P1, MS, Tier 2, execution order | Deletes obsolete canonical tasks, scopes retained comparisons to `OlhoffApproach`, and renumbers the execution sequence after deletion. |
| `scripts/revision_v1/authoritative_formulation_audit.md` | Comparator and preserved-history rows | Reclassifies Exact code as diagnostic history and prevents formulation requirements from being inferred from it. |
| `paper/reviews/REVISION_AUDIT.md` | Current-status override, demand matrix, Exp1 analysis, remaining work, phases 4/6/8, acceptance criteria | Replaces “recover 8.6%/7.1x” logic with definitive withdrawal and local-only evidence requirements. |

### Reviewer-facing manuscript, plans, tables, and figures

| File | Modified section | Justification |
|---|---|---|
| `paper/main.tex` | Abstract; introduction; numerical-example scope; simply-supported figures/captions and discussion; former performance table; frequency-history caption; implementation section; conclusions; data availability | Removes the published-optimum gap and canonical speedup, removes the unsupported performance table, labels saved Olhoff figures as local endpoints, preserves literature context, and prevents local code from being called the Du--Olhoff implementation. |
| `paper/reviews/revision_plan.tex` | Comparator policy; A2/A6; B1/B4/B5; timeline; code changes; Algorithm 1 framing; operation table; performance-evidence section | Preserves the published algorithm as conceptual background, makes empirical planning local-only, and deletes canonical performance/gap/scaling structures without adding experiments. |
| `paper/reviews/algorithms_comparison.tex` | Introduction; Algorithm 1 framing; complete complexity/performance section | Separates published pseudocode from executable evidence and replaces empirical canonical tables and rankings with a local-only comparison policy. |
| `paper/reviews/revision_plan.toc` | Regenerated Algorithm 1 entry | Keeps the compiled plan navigation consistent with the literature-background label. |
| `paper/reviews/algorithms_comparison.toc` | Legacy-auxiliary banner and Algorithm 1 entry | Marks the stale auxiliary non-authoritative and removes its unqualified implementation implication. |
| `paper/reviews/revision_plan copy.toc` | Legacy-copy banner and Algorithm 1 entry | Prevents the retained duplicate auxiliary from presenting an obsolete unqualified heading. |

No raster figure was modified or regenerated. Existing files named
`Olhoff_*.png` are outputs of the local `OlhoffApproach`, not Exact outputs;
their manuscript captions now state that provenance and no longer identify an
endpoint as the published optimum. Exact PNG artifacts remain numerically
unchanged inside the governed archives.

### Active local comparison implementation and documentation

| File | Modified section | Justification |
|---|---|---|
| `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m` | Header and default display label | Identifies the solver as a local Olhoff-inspired comparator without changing its mathematics. |
| `analysis/OlhoffApproach/Matlab/run_freq_benchmark.m` | Header, console report, plot title, summary table labels | Converts published values from validation targets to literature context and removes optimum-gap interpretation. |
| `analysis/OlhoffApproach/Matlab/run_clamped_clamped.m` | Header, console and figure labels | Labels the CC run as a local endpoint; published values are context only. |
| `analysis/OlhoffApproach/Matlab/run_clamped_simply.m` | Header, console and figure labels | Labels the CS run as a local endpoint; published values are context only. |
| `analysis/OlhoffApproach/Matlab/run_simply_simply.m` | Header, console and figure labels | Labels the SS run as a local endpoint; published values are context only. |
| `analysis/OlhoffApproach/Python/solver.py` | Module documentation | Defines all solver outputs as local-code results. |
| `analysis/OlhoffApproach/Python/cases.py` | Published-value annotation | Prevents the stored literature values from serving as a reconstruction gate. |
| `analysis/OlhoffApproach/Python/profile_ops.py` | Module description | Scopes profiling to the local solver. |
| `analysis/OlhoffApproach/Python/run_case.py` | Plot titles, summaries, CLI description | Labels generated results local and adds an explicit no-gap/no-validation scope. |
| `analysis/OlhoffApproach/Python/run_clamped_clamped.py` | CLI description | Labels the runner as local. |
| `analysis/OlhoffApproach/Python/run_clamped_simply.py` | CLI description | Labels the runner as local. |
| `analysis/OlhoffApproach/Python/run_simply_simply.py` | CLI description | Labels the runner as local. |
| `analysis/OlhoffApproach/Python/output/SS/summary.txt` | Complete saved summary | Reclassifies the existing saved value as a local endpoint and the paper value as context. No number was regenerated. |
| `docs/olhoff_audit.md` | Scope, inner-loop findings, feature/summary interpretation | Makes component matches local provenance only and removes task language that implied an active fidelity campaign. |
| `docs/olhoff_implementation_analysis.tex` | Entire comparison framing and all quantitative performance passages | Removes canonical gap/speedup/scaling conclusions, corrects unsupported solver attribution, and retains only local provenance and qualitative structure. |
| `docs/topopt_config.schema.json` | `optimization.approach` description | States that `olhoff` selects the local implementation, not a canonical reconstruction. |

### Production campaign and performance planning

| File | Modified section | Justification |
|---|---|---|
| `examples/Performance/performance_comparison.m` | Method list, labels, CSV display names | Removes the Exact method entirely and fixes the comparison to three named local implementations. |
| `examples/Revision_v1/exp1_perf_table.m` | Comparator policy, labels, path setup | Uses `OlhoffApproach` only and excludes both Exact trees from MATLAB's production path. |
| `examples/Revision_v1/exp2_clamped_beam.m` | Path setup | Replaces recursive `analysis/**` import with an active-implementation allowlist. |
| `examples/Revision_v1/exp2b_building.m` | Path setup | Replaces recursive `analysis/**` import with an active-implementation allowlist. |
| `examples/Revision_v1/exp3_mesh_convergence.m` | Path setup | Replaces recursive `analysis/**` import with an active-implementation allowlist. |
| `examples/Revision_v1/exp4_sensitivity_ablation.m` | Path setup | Replaces recursive `analysis/**` import with an active-implementation allowlist. |
| `examples/Revision_v1/exp5_scaling.m` | Scope, labels, historical fallback warning | Restricts any retained fit to named local implementations and rejects the fallback as canonical or accepted evidence. |
| `examples/Revision_v1/exp_smoke_fail.m` | Path setup | Excludes archived trees even from the infrastructure smoke path. |
| `examples/Revision_v1/run_all_revision_experiments.m` | Campaign policy, stage description, path setup | Documents the sole local comparator and makes Exact trees unreachable through the production path. |
| `tools/Matlab/run_topopt_from_json.m` | `olhoffexact` dispatch and unknown-approach message | Retains diagnostic archive compatibility with a warning while removing Exact from the production-choice list. |
| `examples/Revision_v1/README.md` | New complete file | Defines the production entry point, local comparator, six archived standalone scripts, and non-dependency rule. |

The active path allowlist is `ourApproach`, `OlhoffApproach`,
`YukselApproach`, `elastic2D`, and `LabandaApproach`. Neither Exact tree is on
the path of any production Revision_v1 stage.

### Provenance corrections outside the comparator campaign

| File | Modified section | Justification |
|---|---|---|
| `analysis/ourApproach/Matlab/our_mass_interpolation.m` | Eq. 4b provenance comment | Makes the published equation, not Exact code, the source of the local formula. |
| `examples/Revision_v1/eq4b_exp3_400x50_hypothesis_test.m` | Eq. 4b provenance comment | Removes the archived reconstruction as production authority. |
| `examples/Revision_v1/output/eq4b_exp3_400x50/eq4b_port_review.md` | Port provenance | Records independent equation-based provenance. |
| `examples/Revision_v1/s1_mitigation_400x50_pilot.m` | Eq. 4b provenance comment | Removes Exact as a legitimizing dependency. |
| `examples/Revision_v1/output/s1_mitigation_400x50/s1_mitigation_400x50_summary.md` | Provenance note | Attributes the tested formula to the published equation/local helper. |

### Archived reconstruction documentation and scripts

| File | Modified section | Justification |
|---|---|---|
| `analysis/OlhoffApproachExact/README.md` | New complete file | Governs all 402 archive files with the final negative verdict and evidence prohibition. |
| `analysis/OlhoffApproachExact/OlhoffApproachExact.txt` | Opening status and historical-scope note | Supersedes the former implementation plan; this ignored archive note remains for provenance. |
| `analysis/OlhoffApproachExactOpus/README.md` | Title, status, final verdict | Governs all 224 Opus archive files as failed diagnostic reconstruction work. |
| `analysis/OlhoffApproachExactOpus/comparisons/paper_vs_implementation.md` | Opening status and verdict framing | Prevents component comparison from being treated as validation. |
| `analysis/OlhoffApproachExactOpus/implementation/PHASE4_DECISION.md` | Opening status and historical recommendation | Closes former recommendations and removes production meaning. |
| `docs/olhoff_penalty_continuation_experiment.md` | Archive banner | Retains the experiment as diagnostic history only. |
| `examples/Revision_v1/pilot_olhoff_exact_cc_160x20.m` | Opening banner | Marks the standalone pilot archived and absent from production. |
| `examples/Revision_v1/pilot_olhoff_exact_cc_80x10_alpha05.m` | Opening banner | Marks the standalone pilot archived and absent from production. |
| `examples/Revision_v1/phase1_olhoff_exact_cc_80x10_inner300.m` | Opening banner | Marks Phase 1 diagnostic-only. |
| `examples/Revision_v1/phase2_olhoff_exact_cc_80x10_asymptote_persistence.m` | Opening banner | Marks Phase 2 diagnostic-only. |
| `examples/Revision_v1/phase3_olhoff_exact_cc_80x10_outermove005.m` | Opening banner | Marks Phase 3 diagnostic-only. |
| `examples/Revision_v1/phase4_olhoff_exact_cc_80x10_outermove002.m` | Opening banner | Marks Phase 4 diagnostic-only and prevents the former freeze interpretation. |
| `examples/Revision_v1/output/OLHOFF_EXACT_ARCHIVE.md` | New complete file | Governs all seven output families and all 131 artifacts; numerical payloads are unchanged, while 13 Markdown summaries received archive banners. |

### Reviewer-readable Exact result summaries

Each file below received an archive/supersession banner. Numerical tables and
values were not changed.

| File | Modified section | Justification |
|---|---|---|
| `examples/Revision_v1/output/pilot_olhoff_exact_cc_80x10/algorithm_equivalence_audit.md` | Opening | Component audit is not algorithm equivalence evidence. |
| `examples/Revision_v1/output/pilot_olhoff_exact_cc_80x10/comparison_vs_ourApproach.md` | Opening | Quantitative comparison is excluded from reviewer evidence. |
| `examples/Revision_v1/output/pilot_olhoff_exact_cc_80x10/pilot_report.md` | Opening | Production-migration and reproduction recommendations are withdrawn. |
| `examples/Revision_v1/output/pilot_olhoff_exact_cc_80x10_alpha05/stabilized_pilot_summary.md` | Opening | Stabilization does not validate paper fidelity. |
| `examples/Revision_v1/output/phase1_olhoff_exact_cc_80x10_inner300/phase1_inner300_comparison.md` | Opening | Phase comparison is archived diagnostic history. |
| `examples/Revision_v1/output/phase1_olhoff_exact_cc_80x10_inner300/phase1_inner300_summary.md` | Opening | Phase result is archived diagnostic history. |
| `examples/Revision_v1/output/phase2_olhoff_exact_cc_80x10_asymptote_persistence/phase2_asymptote_persistence_comparison.md` | Opening | Phase comparison is archived diagnostic history. |
| `examples/Revision_v1/output/phase2_olhoff_exact_cc_80x10_asymptote_persistence/phase2_asymptote_persistence_summary.md` | Opening | Phase result is archived diagnostic history. |
| `examples/Revision_v1/output/phase3_olhoff_exact_cc_80x10_outermove005/phase3_outermove005_comparison.md` | Opening | Phase comparison is archived diagnostic history. |
| `examples/Revision_v1/output/phase3_olhoff_exact_cc_80x10_outermove005/phase3_outermove005_summary.md` | Opening | Phase result is archived diagnostic history. |
| `examples/Revision_v1/output/phase4_olhoff_exact_cc_80x10_outermove002/phase4_outermove002_comparison.md` | Opening | Phase comparison is archived diagnostic history. |
| `examples/Revision_v1/output/phase4_olhoff_exact_cc_80x10_outermove002/phase4_outermove002_summary.md` | Opening | Withdraws the former production-freeze recommendation. |
| `examples/Revision_v1/output/phase4_olhoff_exact_cc_80x10_outermove002/phase4_outermove002_vs_phase3.md` | Opening | Phase comparison is archived diagnostic history. |

## Archived artifact families left numerically unchanged

The following directory rules are exhaustive. Every file beneath each path is
archived diagnostic material, whether or not its filename contains an Exact
alias:

- `analysis/OlhoffApproachExact/**`
- `analysis/OlhoffApproachExactOpus/**`
- `examples/Revision_v1/output/pilot_olhoff_exact_cc_160x20/**`
- `examples/Revision_v1/output/pilot_olhoff_exact_cc_80x10/**`
- `examples/Revision_v1/output/pilot_olhoff_exact_cc_80x10_alpha05/**`
- `examples/Revision_v1/output/phase1_olhoff_exact_cc_80x10_inner300/**`
- `examples/Revision_v1/output/phase2_olhoff_exact_cc_80x10_asymptote_persistence/**`
- `examples/Revision_v1/output/phase3_olhoff_exact_cc_80x10_outermove005/**`
- `examples/Revision_v1/output/phase4_olhoff_exact_cc_80x10_outermove002/**`

MAT, CSV, JSON, log, and PNG payloads in these trees were not edited because
they are historical scientific records. Their archive README/index supplies
the current interpretation.

## Removed obsolete tasks and evidence structures

The following are no longer present as active tasks or acceptance criteria:

- paper-faithful/canonical Du--Olhoff reconstruction;
- a production numerical-behaviour freeze for `OlhoffApproachExact`;
- P1-8 canonical benchmark;
- T2-1 canonical speedup;
- T2-3 frequency gap to the Du--Olhoff optimum;
- T2-4 recovery of the 8.6%/7.1x headlines;
- canonical-overhead estimation or bounding;
- Exact-based convergence, timing, memory, or scaling comparison;
- promotion or migration of an Exact phase into production;
- acceptance based on reproducing published optimum behaviour.

The large canonical empirical tables in `paper/main.tex`,
`paper/reviews/algorithms_comparison.tex`, and the corresponding performance
portion of `paper/reviews/revision_plan.tex` were removed or reconstructed.
The remaining operation-count table in the revision plan is explicitly about
the named local implementation, while published Algorithm 1 remains conceptual
literature background.

## Remaining TODO items

These are pre-existing revision tasks that remain meaningful after the
migration:

1. If a quantitative performance comparison is retained, complete the
   existing P1 controlled timing protocol for the named local implementations.
   Any Olhoff-related row must be `analysis/OlhoffApproach`; no result may be
   generalized to canonical Du--Olhoff.
2. Complete the remaining non-Olhoff manuscript corrections and write the
   response letter. The response must state that the reconstruction attempt was
   unsuccessful and that the canonical gap/speedup claims were withdrawn; it
   must not cite an Exact artifact as evidence.
3. Complete the already planned reproducibility package for accepted
   production artifacts.
4. Restore the manuscript figure assets expected under
   `paper/figures/results/` before a full `paper/main.tex` build. The current
   build stops at the pre-existing missing `Olhoff_400x50.png`; this audit did
   not regenerate figures.

There is no remaining TODO to reproduce, validate, tune, benchmark, or promote
`OlhoffApproachExact`.

## Production-dependency confirmation

No remaining production experiment depends on `OlhoffApproachExact`.

Evidence for that conclusion:

- `run_all_revision_experiments.m` calls no Exact script or approach.
- Exp1 selects `Olhoff`, which dispatches to `analysis/OlhoffApproach`.
- Exp2, Exp2b, Exp3, Exp4, Exp5, and the smoke stage contain no Exact selection.
- No production JSON configuration selects `OlhoffExact` or
  `OlhoffApproachExact`.
- Production launchers no longer add the complete `analysis/**` tree; their
  explicit allowlist excludes both Exact directories.
- `examples/Performance/performance_comparison.m` no longer contains an Exact
  method row.
- The six standalone Exact MATLAB scripts are documented archives and are not
  referenced by the master campaign.
- The dispatcher retains `olhoffexact` solely for archive compatibility, emits
  an archive warning, and does not list it as a production choice.
- Exact result directories and reports are excluded from manuscript tables,
  figures, scaling fits, and reviewer-response evidence.

## Validation performed

- `git diff --check` passes.
- Direct-alias and semantic-role scans confirm that every remaining Exact
  mention is an archive, withdrawal, exclusion, compatibility, or diagnostic
  provenance statement. The obsolete task identifiers P1-8, T2-1, T2-3, and
  T2-4 do not remain as active work.
- Production-path scans confirm that the master and all six production stages
  select no Exact approach, import no Exact directory, and use the explicit
  active-implementation allowlist. No production JSON configuration selects
  an Exact approach.
- The modified Python files compile, the JSON schema parses, and
  `python3.13 -m unittest tests/test_revision_v1_validator.py -v` passes all 22
  tests.
- `paper/reviews/algorithms_comparison.tex` and
  `paper/reviews/revision_plan.tex` compile successfully with `latexmk`.
  Incidental changes to tracked build metadata were restored. Compilation
  refreshed the ignored validation outputs
  `paper/reviews/{algorithms_comparison,revision_plan}.{aux,bbl,blg,log,out,pdf}`;
  these derived files are not scientific sources or migration deliverables.
- A full `paper/main.tex` build reaches the pre-existing missing figure
  `paper/figures/results/Olhoff_400x50.png`; no figure or numerical result was
  regenerated to bypass that repository-state issue.
- MATLAB/Octave is unavailable in the audit environment. No numerical campaign
  was executed; MATLAB validation was therefore static and limited to launcher,
  dispatch, path, and configuration inspection.
- Python validation refreshed ignored `__pycache__/*.pyc` files only. No MAT,
  CSV, JSON result, log, PNG, NPZ, or other experimental payload was generated
  or modified.

## Final consistency verdict

- **Internally consistent with the reconstruction verdict:** yes. Literature
  context, the local comparator, and the failed diagnostic reconstruction now
  have separate roles.
- **Reviewer-facing dependency on `OlhoffApproachExact`:** none. Remaining
  mentions state withdrawal or archive status only.
- **Obsolete canonical-reproduction task still active:** none.

This verdict concerns the Olhoff evidence migration. Other revision gates
listed in `REVISION_R1_STATUS.md` remain open on their own scientific merits.
