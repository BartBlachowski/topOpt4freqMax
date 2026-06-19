# Revision_v1 Implementation Audit

## Scope

This audit evaluates whether `examples/Revision_v1` completely fulfills the
reviewer demands recorded in:

- `paper/reviews/revision_plan.tex`
- `paper/reviews/review1.txt`
- `paper/reviews/SAMO-D-26-00346_Comments_Reviewer_3.pdf`
- `paper/reviews/final_review_V1.tex`
- `paper/reviews/final_review_V2.tex`

The audit covers the experiment scripts, JSON configurations, generated MAT/CSV,
diary and image artifacts, and the master experiment result. It does not assume
that a script's stated intent proves completion: fulfillment requires a completed,
converged, traceable output that addresses the requested formulation.

## Verdict

**NOT COMPLETE. `examples/Revision_v1` does not completely fulfill the reviewer
demands and is not sufficient evidence for a resubmission-ready response.**

The directory contains useful partial work, particularly the 10-run performance
dataset, a scaling fit, a building eigenspectrum/MAC export, and a controlled
semi-harmonic sensitivity experiment. However, two central experiment groups
failed, several completed runs did not converge, the load-definition mismatch
identified by the revision plan remains unresolved, and some scripts claim to
validate the manuscript equations while actually testing a different load
formulation.

## Status Vocabulary

- **FULFILLED**: implemented, completed, converged where required, and directly
  answers the reviewer request.
- **PARTIAL**: useful evidence exists, but coverage, convergence, formulation, or
  traceability is insufficient.
- **NOT FULFILLED**: required evidence or implementation is absent or failed.
- **CONTRADICTED**: generated evidence refutes the claim the revision was meant
  to support; the manuscript must be revised accordingly.
- **MANUSCRIPT-ONLY / NOT ASSESSABLE HERE**: cannot be fulfilled by
  `examples/Revision_v1`; requires inspection of the revised manuscript,
  bibliography, response letter, README, or release metadata.

## Critical Findings

### 1. Exp2 and Exp3 failed in the recorded full run

`all_revision_results.mat` records:

| Experiment | Recorded elapsed time | Stored result |
|---|---:|---|
| Exp1 | 56,449 s | present |
| Exp2 | NaN | empty |
| Exp2b | 6,542 s | present |
| Exp3 | NaN | empty |
| Exp4 | 37.45 s | present |
| Exp5 | 17.77 s | present |

There is no `exp2_clamped_beam_results.mat`, no
`exp3_mesh_convergence_results.mat`, no Exp2/Exp3 correlation export, and no
`exp2_alpha1_topo_mode1.png` or `exp2_alpha1_topo_mode2.png`.

Consequences:

- the clamped-beam `alpha=0.75` diagnostic was not completed;
- the Table 3 non-monotonicity demand was not resolved;
- the requested clamped-beam mode 1/2 visualizations were not produced;
- initial clamped-beam frequencies were not saved as evidence;
- the two-mesh multi-mode convergence study was not produced;
- clamped-beam discreteness and full eigenspectrum evidence are absent.

The master runner catches failures and continues, then prints a global
"REVISION EXPERIMENTS COMPLETE" banner. This permits a partially failed run to
look complete. Completion should instead fail when any required experiment has
`NaN` timing or an empty result.

### 2. The A0 load-definition and reference-design mismatch is unresolved

The revision plan requires one consistent definition across paper, code and
examples. The implementation still contains incompatible definitions:

- `ss_beam.json` uses `semi_harmonic_baseline: "initial"`;
- clamped-beam and sensitivity-ablation configurations use
  `semi_harmonic_baseline: "solid"`;
- semi-harmonic scripts define the load using
  `omega0 * M_solid * Phi_solid`;
- harmonic/Eq. 7 configurations use
  `omega0^2 * M(x) * Phi_solid`.

The scripts compare these definitions but do not select and propagate one
authoritative method. Therefore final tables from different experiments are not
guaranteed to describe the same mathematical method. Reviewer V2/C2 and revision
plan A0 remain **NOT FULFILLED**.

### 3. Exp4's clean validation applies to a different load formulation

The finite-difference check is internally useful: six sampled derivatives in
Variant A agree with finite differences to approximately `1e-7` to `1e-6`
relative error. But it validates the derivative of the implemented
semi-harmonic load

`rho_nodal(x) * omega0 * M_solid * Phi_solid`,

not the manuscript load

`omega0^2 * M(x) * Phi`.

The Variant A/B comparison likewise isolates sensitivity for the semi-harmonic
implementation, not the stated Eq. 5/Eq. 6 formulation. Exp4 does run the
documented harmonic load in variants C/D, so the manuscript formulation is not
ignored. However, the clean finite-difference validation and controlled A/B
comparison apply only to the implemented semi-harmonic load; the documented
harmonic formulation appears only in the adverse, unconverged C/D comparison and
was not finite-difference checked.

All four Exp4 runs also reached `max_iters=400` rather than the convergence
tolerance:

| Variant | omega1 (rad/s) | Iterations | Final behavior |
|---|---:|---:|---|
| A: semi-harmonic, omitted load sensitivity | 109.6796 | 400 | not converged; final changes about 0.005-0.030 |
| B: semi-harmonic, full load sensitivity | 110.9146 | 400 | strongly oscillatory; change = 0.200 |
| C: harmonic Eq. 7 frozen | 131.2429 | 400 | not converged; final changes up to about 0.095 |
| D: harmonic Eq. 7 periodic N=50 | 49.8371 | 400 | not converged; final changes up to about 0.110 |

Comparing endpoint frequencies from unconverged trajectories cannot establish
that omission is negligible or that the frozen eigenpair is reliable. The
periodic case changes the result by roughly -62% relative to the frozen harmonic
case (131.24 to 49.84 rad/s). This is the most damaging single result in the
package because it invalidates the intended reliability argument and demands a
mode-tracking/convergence diagnosis. It does not, by itself, prove that the
frozen approximation is inaccurate: both trajectories are unconverged, and the
periodic refresh may have switched to a different or polluted mode. Reviewer R1,
V1/C1, V1/C4 and V2/C1 remain **NOT FULFILLED**.

### 4. Exp1's timing breakdown is methodologically unsound

`run_topopt_from_json` defines `tIter` as average optimization-loop time, with
initialization excluded for the proposed and Olhoff implementations. Exp1 then:

1. computes `tTotal = tIter * nIter`, which is loop time, not wall-clock total;
2. subtracts an estimated setup time from this already setup-free value to
   produce `tPerIterAdj`;
3. measures proposed setup with a standalone eigensolve on a uniform
   `x=volume_fraction` model even though the active performance configuration
   uses a solid semi-harmonic baseline;
4. times only `eigs`, excluding matrix assembly and other initialization work;
5. estimates Yuksel/Olhoff setup as one-iteration wall time minus mean loop time.

The saved setup values become exactly zero for Olhoff at the 320x40 and 400x50
meshes, demonstrating failure of the estimator. The issue is best described as a
methodologically unsound decomposition rather than a purely algebraic error: its
quantities do not represent consistently measured setup, loop and wall-clock
components. Thus Reviewer 1's requested separation of initialization cost,
per-iteration cost, and total runtime is only **PARTIAL** and must not be
presented as a measured decomposition.

### 5. Exp1 contains non-converged comparator results

The saved 10-run dataset is valuable and includes frequencies, timing standard
deviations and memory standard deviations at four meshes. However, Olhoff
reports exactly 2,000 iterations at every mesh, indicating termination at the
configured cap rather than convergence. Frequencies and timings from a capped
comparator do not establish a fair converged-method comparison.

The Exp1 evidence therefore only **PARTIALLY** fulfills:

- frequency versus mesh for all methods;
- timing variance;
- speedup decomposition;
- frequency-gap and canonical-comparator claims.

The regenerated values also conflict directly with the manuscript's headline
numbers. Across the four meshes, the proposed-versus-Olhoff omega1 differences
are approximately -0.5%, +0.6%, +0.4% and +0.5%, rather than the stated 8.6%
gap; at some meshes the proposed result is slightly higher. The capped Olhoff
times would imply a speedup near 80x at 400x50 rather than 7.1x. Because Olhoff
did not converge, these regenerated values do not establish that the proposed
method is actually more accurate or 80x faster. They establish that the new
dataset is unreconciled with the manuscript and cannot support the existing
8.6%/7.1x claims.

### 6. The building outputs reveal suspect, uncorrelated low modes

Exp2b completed and exported 100 topology modes and a broad reference-mode MAC
matrix. It is useful diagnostic evidence. However, the first ten topology modes
frequently contain modes whose maximum MAC against the reference set is below
0.01. Examples include:

- alpha=1.00: topology modes 1, 2, 5, 6, 8, 9 and 10;
- alpha=0.75: topology modes 1, 2, 6, 7, 8, 9 and 10;
- alpha=0.50: topology modes 2, 3, 5, 8, 9 and 10.

This does not demonstrate the absence of spurious low-density modes and is
consistent with spectral pollution. Low MAC alone does not prove that a mode is
spurious, localized, or non-structural; that conclusion requires mode-shape
localization, strain/kinetic-energy distribution, or density-support evidence.
The current evidence therefore flags suspect modes and leaves the no-spurious-mode
claim unsupported. Reviewer M7 is **NOT FULFILLED**; the manuscript must retract
or sharply qualify the claim unless the additional localization analysis clears
these modes.

Moreover, Exp2b itself reached `max_iters=2000` for alpha=1.00 and alpha=0.75.
Those two topologies and spectra are not demonstrated converged solutions.

### 7. CR1 remains open and the manuscript still states the false monotonicity claim

The experiment intended to resolve the clamped-beam alpha=0.75 anomaly is Exp2,
which failed. The manuscript still states that the Phi2 gain increases
monotonically as alpha decreases while presenting the sequence from 1.99x to
2.45x without acknowledging the intervening alpha=0.75 value of 1.73x that
breaks monotonicity (reported around `main.tex` line 469 in the audited version).
Thus CR1 is not merely missing experimental support: the known counterexample
remains uncorrected in the paper's prose. CR1 is **OPEN / NOT FULFILLED**.

## Demand Traceability Matrix

| Reviewer demand | Evidence in Revision_v1 | Status |
|---|---|---|
| Reframe novelty relative to Yuksel-Yilmaz | No manuscript evidence in this directory | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |
| Separate setup, per-iteration, iteration-count and total runtime | Exp1 attempts a decomposition, but its timing semantics and setup estimates are methodologically unsound | PARTIAL |
| Explain Yuksel iteration discrepancy against published 180-200 iterations | No original-code comparison or resolved diagnosis; Exp1 only reruns in-house code | NOT FULFILLED |
| Quantify frozen-eigenvector error / refresh every N iterations | Only N=50 versus frozen, both unconverged; planned N={1,5,10,50,infinity} study absent | NOT FULFILLED |
| Analyze clustering/coalescence and a posteriori target-mode validity | Building MAC/eigenspectrum exists; clamped-beam Exp2 failed; manuscript discussion unavailable | PARTIAL |
| Clarify speedup source | New timing data exist, but total/setup decomposition is invalid and Olhoff is capped | PARTIAL |
| Validate omitted load sensitivity for Eq. 6 | FD check validates different semi-harmonic formula; all ablation endpoints unconverged | NOT FULFILLED |
| Unify frozen reference design | Initial and solid baselines coexist | NOT FULFILLED |
| Reconcile OC/MMA and scope contribution | Configurations document optimizer choice, but manuscript/algorithm correction unavailable | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |
| Correct/qualify Table 3 alpha=0.75 non-monotonicity | Intended Exp2 output absent; manuscript still omits the 1.73x counterexample from its monotonicity statement | NOT FULFILLED |
| Report full spectrum below tracked clamped mode | Exp2 absent; building spectra do not substitute for clamped Table 3 case | NOT FULFILLED |
| Bound speedup versus canonical Olhoff | No canonical-algorithm ablation or removed-overhead estimate | NOT FULFILLED |
| Add omega1 for every method and mesh | Exp1 contains values, but Olhoff is nonconverged at all meshes | PARTIAL |
| Reference comparator for a multi-mode example | Exp2/Exp3 compare semi-harmonic against harmonic Eq. 7, not Olhoff/Yuksel; outputs absent | NOT FULFILLED |
| Timing and memory standard deviations | Exp1 contains 10-run standard deviations | FULFILLED as data collection; manuscript integration unverified |
| Hardware/software specification | Master script prints data to console, but no run log or hardware artifact was saved | PARTIAL / UNVERIFIABLE |
| Clamped mode-shape plots for alpha=1 | Expected PNG files absent because Exp2 failed | NOT FULFILLED |
| Discreteness metric for all proposed topologies | Building and Exp4 metrics exist; clamped results absent; Exp4 unconverged | PARTIAL |
| Demonstrate no spurious low-density modes | Building eigenspectrum shows many suspect low-correlation modes; localization/energy evidence is absent | NOT FULFILLED; claim unsupported |
| Mesh convergence for a multi-mode example | Exp3 failed; no result MAT/CSV | NOT FULFILLED |
| Define and enforce MAC validity threshold | Scripts define 0.8 and flag low values; clamped execution absent | PARTIAL |
| State clamped initial eigenfrequencies | Computation exists in Exp2 source, but no completed result artifact | NOT FULFILLED |
| Correct scaling claim with log-log fit | Exp5 completed; measured exponents are about 0.904, 0.910 and 1.037 | FULFILLED as experiment; manuscript integration unverified |
| Diagnose grey regions / report grayness | Building grayness 0.093-0.138; Exp4 grayness 0.113-0.249; clamped absent; no Heaviside comparison | PARTIAL |
| Consider RAMP mitigation | No RAMP experiment; discussion is manuscript-only | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |
| Heaviside sharpening case | Planned as optional; no corresponding config/result artifact | NOT IMPLEMENTED (optional in revision plan) |
| Application/method/Rayleigh/MMA/filter references | Bibliography/manuscript edits not represented here | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |
| Explicit sensitivity equation and load-sensitivity explanation | Requires revised manuscript; code comments are not a response | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |
| Discuss design-dependent load difficulties and symmetric/asymmetric applicability | Requires revised manuscript | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |
| Clarify supplementary material | No manifest explaining supplementary contents | NOT FULFILLED |
| Permanent code DOI/tag/commit and license | No release evidence in Revision_v1 | NOT FULFILLED |
| Script-to-table/figure map | Master script lists experiments but no complete manuscript figure/table mapping | NOT FULFILLED |
| Correct figure references, notation, citations, gap claim vs Huang et al. | Requires revised manuscript | MANUSCRIPT-ONLY / NOT ASSESSABLE HERE |

## Positive Evidence That Can Be Retained

The following work is useful after the blockers above are corrected:

1. **Exp1 repeated-run framework.** It executes 10 samples over four meshes and
   stores mean/standard-deviation arrays for frequencies, loop timing and memory.
2. **Exp5 scaling fit.** It uses measured Exp1 timings and yields exponents near
   0.904 (Olhoff), 0.910 (Yuksel) and 1.037 (proposed), directly showing that the
   existing `O(n_e^1.3)` claim should be replaced.
3. **Semi-harmonic derivative implementation check.** The FD checks demonstrate
   that the implemented semi-harmonic load derivative is coded consistently for
   the sampled elements. This is software verification, not validation of the
   manuscript Eq. 6 approximation.
4. **Building spectral export.** Exp2b provides 100-mode spectra, reference-mode
   MAC data and grayness values across alpha. These data are suitable for a
   transparent discussion of spectral pollution and low-MAC modes, provided the
   capped alpha=1.00 and 0.75 runs are rerun to convergence.
5. **Explicit experiment configurations.** JSON files make optimizer, baseline,
   interpolation, filter and update schedules auditable.

## Minimum Work Required Before Claiming Complete Fulfillment

1. Resolve A0: select one load equation and one reference design, then update all
   JSONs and rerun every affected table/figure.
2. Make the master runner fail its overall status when any required experiment
   fails or returns capped/nonconverged runs. Save a persistent run log including
   hardware/software information and failure stack traces.
3. Fix Exp2 and produce its MAT, correlation CSVs, alpha=0.75 diagnostics and two
   requested mode-shape PNGs.
4. Fix Exp3 and produce a converged two-mesh multi-mode comparison.
5. Redesign Exp4 so the full and omitted gradients correspond to the authoritative
   manuscript load. Run to convergence and report convergence histories. Perform
   the planned refresh sweep N={1,5,10,50,infinity}, not only N=50.
6. Correct Exp1 timing: store actual wall-clock total, measured initialization and
   measured loop time from each solver. Do not subtract setup from an already
   setup-free loop time. Rerun capped Olhoff cases or explicitly report failure to
   converge rather than treating them as comparable optima.
7. Add an Olhoff/Yuksel reference run for at least one multi-mode case and a
   canonical-Olhoff overhead ablation or defensible bound.
8. Treat Exp2b's low-MAC modes as evidence requiring characterization; do not claim
   absence of spurious modes without mode localization/energy evidence.
9. Produce a repository mapping from every revised manuscript table/figure to its
   script, JSON, result file and exact commit/release.
10. Audit the revised manuscript and response letter separately for all prose,
    citation, notation, equation, scope, optimizer, supplementary-material and
    reproducibility demands.

## Final Assessment

The implementation is a **substantial but incomplete experimental scaffold**.
It does not yet justify telling the editor or reviewers that all demands were
fulfilled. The highest-risk errors are not cosmetic: they affect which method is
being tested, whether comparison points converged, whether timing columns mean what
they claim, and whether the sensitivity evidence applies to the manuscript equation.
