# Revision_v1 Implementation Map

Source document: `examples/Revision_v1/revision_v1_update1.md`

This map lists every mandatory workstream item, its dependencies, required
artifacts, acceptance gate, tier, and category. Nothing is implemented here;
this is a planning reference only.

---

## Legend

| Column | Values |
|---|---|
| Tier | 1 = mandatory blocker · 2 = mandatory if claim retained · 3 = useful supplement |
| Category | code · experiment · manuscript · reproducibility |
| Gate | label defined in `revision_v1_update1.md` |

---

## A0 — Authoritative Formulation

**Source:** §"Mandatory Scientific Decision: Authoritative Formulation"

No production experiment may start before Gate A0 passes.

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| A0-F1 | Choose one reference design (fully solid domain); propagate to every MATLAB/Python path and every JSON config | `analysis/ourApproach/Matlab/topopt_freq.m`, `analysis/ourApproach/Python/topopt_freq.py`, all `examples/Revision_v1/*.json` | Updated configs; written-down reference-design declaration | A0 | 1 | code |
| A0-F2 | Replace `omega0` scalar with `omega0^2` in load construction | Same solvers as A0-F1 | Load vector `f_j` uses `ω₀² M(x) Φ₀` everywhere | A0 | 1 | code |
| A0-F3 | Remove extra `rho_nodal(x)` scaling (design dependence already in `M(x)`) | Same solvers | No double-scaling in load path | A0 | 1 | code |
| A0-F4 | Adopt one documented mode normalization and mass-weighted MAC definition across MATLAB and Python | Same solvers; `analysis/YukselApproach/Python/solver.py`; `analysis/YukselApproach/Matlab/` | Normalization convention written in code comments; MAC formula consistent with manuscript Eq. 9 | A0 | 1 | code |
| A0-F5 | Add explicit config option to select omitted vs complete load sensitivity | Same solvers; `examples/Revision_v1/*.json` | Config key (e.g. `load_sensitivity: omitted|complete`) accepted by both MATLAB and Python solvers | A0 | 1 | code |
| A0-F6 | Update equations, pseudocode, inline comments, and example JSON docs to match authoritative formulation | All solver files; `paper/` sources; `examples/Revision_v1/revision_v1_update1.md` response draft | No stale `omega0` (non-squared) references remain in active code paths | A0 | 1 | code |
| A0-G | Gate A0 parity test: small deterministic mesh, MATLAB vs Python; relative disagreement ≤1e-8 for reference eigenfrequencies, load-vector entries, objectives, and sensitivities (absolute 1e-12 fallback for near-zero quantities); mass-normalised eigenvectors with consistent sign/phase; modal subspace comparison when eigenvalues cluster | New test script `scripts/revision_v1/test_a0_parity.m` + `test_a0_parity.py` | Printed pass/fail table; both scripts exit non-zero on failure | A0 | 1 | code |

---

## Workstream 1 — Fail-Loud Experiment Infrastructure

**Source:** §"Workstream 1: Fail-Loud Experiment Infrastructure"

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| I1-1 | Extend every solver result struct/dict with telemetry: success flag, termination reason, iteration count, iteration cap, final Δx, constraint residual, per-iteration histories of objective/frequency/feasibility/grayness, tracked-mode index and MAC history, independently measured init/loop/postprocessing/total wall-clock times, config hash, source commit SHA, random seed, software and hardware metadata | `analysis/ourApproach/Matlab/topopt_freq.m`, `analysis/ourApproach/Python/topopt_freq.py` | Every result `.mat` / `.json` contains all telemetry fields; missing fields are a hard error | I1 | 1 | code |
| I1-2 | Update `run_all_revision_experiments.m` to propagate failure: exception → fail; empty result or required `NaN` → fail; iteration-cap without convergence → fail; MAC criterion violated (mode lost) → fail; missing MAT/CSV/image/diary/manifest → fail; complete stack traces preserved | `examples/Revision_v1/run_all_revision_experiments.m` | Master runner exits with non-zero status and identifies the exact failed acceptance condition | I1 | 1 | code |
| I1-3 | Route each experiment to its own isolated output directory; prevent silent overwriting of correlation files and mode plots | `run_all_revision_experiments.m`; per-experiment scripts | Collision on existing output raises an error rather than overwriting | I1 | 1 | code |
| I1-4 | Intentional failing smoke experiment: write a trivially failing stub that exercises every failure-detection path in the master runner | New stub script `scripts/revision_v1/smoke_fail.m` | `run_all_revision_experiments.m` fails and prints the correct failure condition; manual inspection confirms correctness | I1 | 1 | code |

---

## Workstream 2 — Solver Verification

**Source:** §"Workstream 2: Solver Verification"

Gate V1 must pass before Exp1–Exp5 are regenerated.

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| V1-1 | Small-mesh regression: reference eigenpairs and normalization (mass-normalised; consistent sign) | New `scripts/revision_v1/verify_eigenpairs.m` + `.py` | Eigenvalue and eigenvector norms printed; test passes/fails | V1 | 1 | code |
| V1-2 | Small-mesh regression: load construction ω₀²M(x)Φ₀ | Same verify scripts | Load-vector entries match reference values to machine precision | V1 | 1 | code |
| V1-3 | Small-mesh regression: weighted multi-mode objective assembly | Same verify scripts | Objective value matches hand-computed reference | V1 | 1 | code |
| V1-4 | Small-mesh regression: omitted-gradient variant vs complete-gradient variant (two separate paths) | Same verify scripts | Both variants compute without error; sensitivities are distinct as expected | V1 | 1 | code |
| V1-5 | Central finite-difference sensitivity check for complete-gradient variant; declared tolerance ≤1e-5 relative error | Same verify scripts | Printed FD vs analytic table; test fails if any component exceeds declared tolerance | V1 | 1 | code |
| V1-6 | Small-mesh regression: mass-weighted MAC and mode reordering | Same verify scripts | MAC values match reference; reordering selects correct mode | V1 | 1 | code |
| V1-7 | MATLAB/Python parity on all V1 checks | `test_a0_parity.m`/`.py` extended, or a separate parity runner | All V1 quantities agree between MATLAB and Python to ≤1e-8 relative (1e-12 absolute fallback) | V1 | 1 | code |

---

## Workstream 3 — Recover the Clamped-Beam Experiments

**Source:** §"Workstream 3: Recover the Clamped-Beam Experiments"

### Exp2: Full Clamped-Beam Analysis

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| E2-1 | Reproduce current Exp2 failure and capture complete stack trace | `examples/Revision_v1/exp2_clamped_beam.m` | `exp2_failure_trace.txt` written to output dir | E23 | 1 | experiment |
| E2-2 | Fix failure; run α∈{1, 0.75, 0.5, 0.25, 0} under authoritative formulation (A0-F1…F5 applied) | `examples/Revision_v1/exp2_clamped_beam.m`; `analysis/ourApproach/Matlab/topopt_freq.m` | Five converged result `.mat` files, one per α | E23 | 1 | experiment |
| E2-3 | Declare and enforce predeclared convergence rule (design change, feasibility, mode validity); capped runs remain failures | `exp2_clamped_beam.m`; master runner | Declared rule written in experiment header; master runner rejects any capped run | E23 | 1 | experiment |
| E2-4 | Export per-α: initial frequencies, complete spectrum below tracked mode, MAC matrices, selected mode indices, convergence histories, grayness, topology plots, tracked mode shapes | `exp2_clamped_beam.m` | For each α: `.mat` (histories), mode-shape `.png`, topology `.png`, MAC `.mat`, grayness scalar in result struct | E23 | 1 | experiment |
| E2-5 | Diagnose α=0.75 result without assuming monotonicity; explain observed behaviour | `exp2_clamped_beam.m`; analysis in response letter | Written diagnosis in experiment diary or response draft; does not assert monotonicity unless convergence data supports it | E23 | 1 | experiment |
| E2-6 | A5 a-posteriori check for every α case: state whether tracked mode is the structure's lowest mode; explain that the method targets a reference mode shape, not necessarily the fundamental | `exp2_clamped_beam.m` | Per-case flag `is_lowest_mode` in result struct; narrative explanation in response letter | E23 | 1 | experiment |
| E2-7 | One fair external-method comparison (prefer α=1 single-mode case); accurate comparator label | New sub-script or extension of `exp2_clamped_beam.m`; `analysis/OlhoffApproach/` or `analysis/LabandaApproach/` | Comparison table: objective, frequency, topology quality, iterations; comparator labelled as "Olhoff-inspired" unless canonical Olhoff is faithfully reproduced | E23 | 1 | experiment |

### Exp3: Mesh Convergence

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| E3-1 | Declare numerical convergence criterion before inspecting any result | `examples/Revision_v1/exp3_mesh_convergence.m` | Criterion written in experiment header comment before run | E23 | 1 | experiment |
| E3-2 | Run authoritative formulation at 200×25 and 400×50; identical physical geometry, interpolation, filtering, stopping rules, and mode-tracking logic | `exp3_mesh_convergence.m`; `analysis/ourApproach/Matlab/topopt_freq.m`; `clamped_beam_200x25.json`, `clamped_beam_400x50.json` | Two converged `.mat` result files | E23 | 1 | experiment |
| E3-3 | Report frequency, gain, MAC, selected mode index, grayness, constraint residual, and topology differences between meshes; report lack of convergence if criterion is not met | `exp3_mesh_convergence.m` | Summary table in diary; topology comparison figure; grayness and constraint-residual values in result struct | E23 | 1 | experiment |

---

## Workstream 4 — Separate CR2 and A4 Studies

**Source:** §"Workstream 4: Separate CR2 and A4 Studies"

### CR2: Omitted Load Sensitivity

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| CR2-1 | Implement Variant A: evolving-mass load M(x)Φ₀ with load-sensitivity term omitted from gradient | `analysis/ourApproach/Matlab/topopt_freq.m`; config option A0-F5 | Variant A selectable via config; no code change needed if A0-F5 is done | E4 | 1 | code |
| CR2-2 | Implement Variant B: same load with complete (ω₀)²(∂M/∂xₑ)Φ₀ load-sensitivity contribution | Same; A0-F5 complete-sensitivity path | Variant B selectable via config; analytic sensitivity formula documented | E4 | 1 | code |
| CR2-3 | FD verification of Variant B sensitivity; relative error ≤1e-5 (as per V1-5 tolerance) | `scripts/revision_v1/verify_eigenpairs.m` or dedicated `verify_cr2_fd.m` | Printed FD vs analytic table; test fails above declared tolerance | E4 | 1 | experiment |
| CR2-4 | Run Variant A and Variant B from identical initial conditions to convergence; enforce predeclared stopping rule | New `scripts/revision_v1/run_cr2.m` | Two converged `.mat` files; identical ICs documented in result struct | E4 | 1 | experiment |
| CR2-5 | Compare: topology, objective, tracked frequencies, mode validity (MAC), grayness, feasibility, iteration history, per-element sensitivity differences; conclude negligibility only if evidence supports it | `run_cr2.m`; analysis script | Comparison table and topology figures; sensitivity-difference heatmap; written conclusion | E4 | 1 | experiment |

### A4: Eigenpair Refresh Approximation

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| A4-1 | Run simply supported 400×50 case with N∈{1, 5, 10, 50, ∞} (N=∞ is permanently frozen reference) | New `scripts/revision_v1/run_a4.m`; `analysis/ourApproach/Matlab/topopt_freq.m`; `ss_beam_harmonic_frozen.json` / `ss_beam_harmonic_periodic.json` | Five result `.mat` files; N=∞ uses single-reference frozen load from A0-F1 | E4 | 1 | experiment |
| A4-2 | MAC plus frequency-continuity mode tracking after each refresh; document tracking decisions | `run_a4.m`; MAC utility in solver | Per-iteration tracked-mode index, MAC value, and refresh events stored in result struct | E4 | 1 | experiment |
| A4-3 | Save per-N: refresh event log, tracked mode indices, MAC, tracked frequencies, objective, feasibility, convergence history | `run_a4.m` | Result struct fields present for all N values; diary file per run | E4 | 1 | experiment |
| A4-4 | Treat N=1 as a variant; claim equivalence to canonical Yuksel-Yilmaz only if complete algorithms are shown to match; report oscillation or non-convergence if it occurs | `run_a4.m`; response letter | N=1 result flagged with equivalence status; any oscillation reported in diary and manuscript | E4 | 1 | experiment |

---

## Workstream 5 — Rebuild Performance Evidence

**Source:** §"Workstream 5: Rebuild Performance Evidence"

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| P1-1 | Instrument solvers: independently record init time, optimization-loop time, postprocessing time, total wall-clock time, per-iteration time, and peak memory; do not derive init by subtraction | `analysis/ourApproach/Matlab/topopt_freq.m`, `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`, `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m` | Telemetry fields in every result struct; timing methodology documented in experiment header | P1 | 1 | code |
| P1-2 | Benchmark mode: disable live plots, saved images, correlation exports, frequency-history eigensolves, and other diagnostics | Same solvers; `examples/Revision_v1/exp1_perf_table.m` | Config flag `benchmark_mode: true` accepted; confirmed no diagnostic overhead in timing | P1 | 1 | code |
| P1-3 | Warm-up runs then ≥10 measured executions at each mesh; report standard deviations for timing and memory | `examples/Revision_v1/exp1_perf_table.m` | Table with mean ± std for each timing component and mesh; min 10 measurements per data point | P1 | 1 | experiment |
| P1-4 | Investigate Yuksel 180–200 vs 1,000+ iteration discrepancy using matched stopping criteria and reference implementation where available | `exp1_perf_table.m`; `analysis/YukselApproach/Matlab/` | Written explanation of discrepancy; matched-criteria comparison table | P1 | 1 | experiment |
| P1-5 | Correct comparator labels: use "Olhoff-inspired" for the local modified implementation; claim performance against canonical Olhoff only after faithful reproduction; bound removed overhead explicitly if canonical cannot be run | `exp1_perf_table.m`; manuscript tables | Updated labels in all tables and figures; no "canonical" language unless reproduced | P1 | 1 | experiment |
| P1-6 | Recompute scaling fit from corrected timing data | `examples/Revision_v1/exp5_scaling.m` | New `exp5_scaling_loglog.png` and `.mat` generated from corrected data | P1 | 1 | experiment |
| P1-7 | Replace or re-verify 8.6% frequency gap, 7.1× speedup, and complexity claims from converged measurements | `exp1_perf_table.m`; `exp5_scaling.m`; manuscript | Claims updated to match regenerated evidence; stale numbers removed unless reproduced | P1 | 2 | experiment |
| P1-8 | Canonical Olhoff benchmark (faithful reproduction) | `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m` or `analysis/OlhoffApproachExact/` | Verified canonical result enabling "canonical speedup" language; or explicit removal of that language | P1 | 2 | experiment |

---

## Workstream 6 — Low-Mode and Grayness Diagnosis

**Source:** §"Workstream 6: Low-Mode and Grayness Diagnosis"

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| S1-1 | Rerun capped building cases to convergence under authoritative formulation | `examples/Revision_v1/exp2b_building.m`; `analysis/ourApproach/Matlab/topopt_freq.m` | Converged building result `.mat`; no iteration-cap flag in result | S1 | 1 | experiment |
| S1-2 | For every suspicious low-MAC mode: export mode shape, elementwise kinetic energy, elementwise strain energy, energy fraction in low-density regions, localization metric, density support, and MAC value | `exp2b_building.m`; post-processing script `scripts/revision_v1/diagnose_modes.m` | Per-mode `.mat` and mode-shape `.png`; summary table of energy fractions and localization metrics | S1 | 1 | experiment |
| S1-3 | Classify every suspicious mode from localization and energy evidence (not MAC alone); determine physical vs spurious | `diagnose_modes.m`; manual review | Classification table in experiment diary; each mode labelled with evidence-based classification | S1 | 1 | experiment |
| S1-4 | Compare baseline SIMP interpolation with one documented mitigation (RAMP, increased void-mass penalization, or Heaviside continuation) | New `scripts/revision_v1/run_s1_mitigation.m`; modified solver config | Mitigation result `.mat`; topology and mode comparison figures | S1 | 2 | experiment |
| S1-5 | Report grayness for every final topology in the paper; explain gray regions in affected figures | `diagnose_modes.m`; manuscript | Grayness scalar in every result struct; gray regions explained in figure captions | S1 | 1 | experiment |

---

## Workstream 7 — Manuscript and Response Update

**Source:** §"Workstream 7: Manuscript and Response Update"

All items depend on accepted artifacts from Workstreams 2–6. Tables and figures must be regenerated, not manually edited.

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| MS-1 | Correct or qualify α=0.75 monotonicity claim based on Exp2 diagnosis (E2-5) | `paper/` sources; response letter | Monotonicity claim removed or narrowed to match evidence | DoC-8 | 1 | manuscript |
| MS-2 | Reframe novelty as frozen-reference simplification of Yuksel-Yilmaz | `paper/` sources; response letter | Novelty statement updated; Yuksel-Yilmaz relation explained | DoC-8 | 1 | manuscript |
| MS-3 | State explicitly that load sensitivity is nonzero but omitted; state effect is evaluated by CR2 | `paper/` sources | Clear statement in methodology section; no claim of mathematical zero | DoC-8 | 1 | manuscript |
| MS-4 | Distinguish sensitivity omission (CR2) from eigenpair freezing/refresh (A4) | `paper/` sources | Two distinct subsections or paragraphs; no conflation | DoC-8 | 1 | manuscript |
| MS-5 | Add A5 lowest-mode check; explain that method targets a reference mode shape, not necessarily the fundamental | `paper/` sources | A5 check described; mode-targeting explanation present | DoC-8 | 1 | manuscript |
| MS-6 | Correct load equation, baseline equation, MAC definition, filter description, and sensitivity equations | `paper/` sources | Equations consistent with authoritative formulation and V1 verification | DoC-8 | 1 | manuscript |
| MS-7 | Correct manuscript Eq. 9 to use mass-weighted inner product matching the implementation | `paper/` sources | Eq. 9 uses ⟨Φᵢ, MΦⱼ⟩ form; consistent with A0-F4 | DoC-8 | 1 | manuscript |
| MS-8 | Reconcile OC/MMA descriptions and the actual contribution scope | `paper/` sources | No description of OC where MMA is used, or vice versa | DoC-8 | 1 | manuscript |
| MS-9 | Distinguish canonical methods (Olhoff, Yuksel-Yilmaz) from local modified implementations throughout | `paper/` sources; all tables | Every comparator has an accurate label | DoC-8 | 1 | manuscript |
| MS-10 | Replace unsupported speedup, frequency gap, convergence, complexity, and no-spurious-mode claims; replace with evidence-bounded statements | `paper/` sources; tables from P1, S1 | No unsubstantiated headline numbers remain; adverse findings incorporated | DoC-8 | 1 | manuscript |
| MS-11 | Report convergence status, feasibility, grayness, MAC validity, and mesh-convergence evidence for every result | `paper/` sources | Every result table includes these columns or a written statement | DoC-8 | 1 | manuscript |
| MS-12 | Explain design-dependent-load difficulties; state symmetric/asymmetric-domain applicability | `paper/` sources | Limitations section or dedicated paragraph | DoC-8 | 1 | manuscript |
| MS-13 | Correct Φ vs φ notation throughout; review non-standard "quasi-static" terminology | `paper/` sources | Uniform notation; "quasi-static" either defined rigorously or replaced | DoC-8 | 1 | manuscript |
| MS-14 | Reconcile void-material values (1e-9 vs 1e-6) and stopping tolerances (1e-3 vs 2e-3) across text and code | `paper/` sources; all solver configs | One declared value for each parameter; consistent with running configs | DoC-8 | 1 | manuscript |
| MS-15 | Verify Huang comparison and all figure/table cross-references | `paper/` sources | All cross-references checked; Huang comparison traceable to a citable artifact | DoC-8 | 1 | manuscript |
| MS-16 | Complete all bibliography entries (topology-method, Rayleigh, MMA, filter, load-sensitivity, design-dependent-load references) | `paper/` sources; `.bib` file | No `[?]` or missing-reference placeholders | DoC-8 | 1 | manuscript |
| MS-17 | Enumerate supplementary material precisely | `paper/` sources; reproducibility package (WS8) | Supplementary section lists each file with a one-line description | DoC-8 | 1 | manuscript |
| MS-18 | Regenerate all affected tables and figures from accepted artifacts | Plotting scripts; result `.mat` files from Exp1–Exp5, CR2, A4, S1 | All paper figures and tables sourced from accepted results; no manually edited numbers | DoC-8 | 1 | manuscript |
| MS-19 | Response letter: address every reviewer item individually with exact manuscript location and supporting artifact; incorporate adverse results; retract or qualify 8.6%/7.1×/4.61× building-gain narratives if not reproduced | Response letter document | Every reviewer item marked FULFILLED with citation; no PARTIAL or NOT FULFILLED items remaining | DoC-8 | 1 | manuscript |

---

## Workstream 8 — Reproducibility Package

**Source:** §"Workstream 8: Reproducibility Package"

| ID | Item | Dependent code / modules | Required outputs / artifacts | Gate | Tier | Category |
|---|---|---|---|---|---|---|
| RP-1 | Script/config/result/figure/table manifest (machine-readable) | All experiment scripts; result `.mat`/`.csv`/`.png` files | `manifest.json` or `manifest.csv` listing every artifact with path and role | DoC-3 | 1 | reproducibility |
| RP-2 | Clean-run instructions and expected artifact checks | `README` or `REPRODUCE.md`; manifest (RP-1) | Step-by-step instructions; checklist of expected outputs a reader can verify | DoC-3 | 1 | reproducibility |
| RP-3 | Saved hardware/software report and complete run log | New `scripts/revision_v1/collect_sysinfo.m` + `.py` | `sysinfo.txt` (MATLAB/Python versions, OS, CPU, RAM); `run.log` with full console output | DoC-3 | 1 | reproducibility |
| RP-4 | Supplementary-material inventory consistent with MS-17 | Manifest (RP-1); paper sources | Inventory table with file name, description, and corresponding manuscript location | DoC-3 | 1 | reproducibility |
| RP-5 | Immutable release tag and source commit hash | Git repository | `git tag v1.0-revision` (or equivalent); commit SHA recorded in manifest | DoC-3 | 1 | reproducibility |
| RP-6 | License and permanent archive / DOI (e.g., Zenodo) | Repository; archive service | `LICENSE` file present; DOI registered; DOI cited in manuscript supplementary section | DoC-3 | 1 | reproducibility |
| RP-7 | SHA-256 checksums for all accepted result artifacts | Result files; `scripts/revision_v1/checksum_results.sh` | `checksums.sha256` file committed alongside results | DoC-3 | 1 | reproducibility |
| RP-8 | Machine-readable completion report | Master runner output; manifest | `completion_report.json` with per-experiment pass/fail, gate status, and artifact inventory | DoC-3 | 1 | reproducibility |

---

## Tier 2 — Mandatory Only If Associated Claim Is Retained

These items are not needed if the corresponding claim is removed from the manuscript.

| ID | Claim at risk | Item | Gate | Category |
|---|---|---|---|---|
| T2-1 | Canonical Olhoff speedup / "canonical" benchmark language | Faithful reproduction of canonical Olhoff (P1-8) | P1 | experiment |
| T2-2 | Absence of spurious low-density modes | RAMP / Heaviside / void-mass mitigation comparison (S1-4) | S1 | experiment |
| T2-3 | 8.6% frequency gap | Regenerated evidence reproducing that value from converged, matched runs | P1 | experiment |
| T2-4 | 7.1× speedup | Regenerated evidence reproducing that value under matched stopping rules | P1 | experiment |
| T2-5 | 4.61× building frequency gain | Converged building result supporting that gain | S1 / E23 | experiment |
| T2-6 | Strong monotonicity (α sweep) | Converged α-sweep evidence supporting monotonicity | E23 | experiment |
| T2-7 | Frozen-reference accuracy claims | A4 converged comparisons supporting accuracy | E4 | experiment |

---

## Tier 3 — Useful Supplementary Additions

These must not delay Tier 1 corrections.

| ID | Item | Category |
|---|---|---|
| T3-1 | Additional mesh sizes beyond the required 200×25 and 400×50 | experiment |
| T3-2 | Additional refresh intervals N beyond {1, 5, 10, 50, ∞} | experiment |
| T3-3 | Broader interpolation comparisons (e.g., RAMP vs SIMP in non-building cases) | experiment |
| T3-4 | Broader optimizer comparisons (OC vs MMA on additional cases) | experiment |
| T3-5 | Extra mode visualizations not required for reviewer traceability | experiment |

---

## Recommended Implementation Order

The order follows §"Compute Strategy" and the gate dependency graph.

```
Phase 0 — Foundation (code, no production runs)
  Step 1   A0-F1   Reference design declaration
  Step 2   A0-F2   omega0 → omega0^2 in load construction
  Step 3   A0-F3   Remove rho_nodal double-scaling
  Step 4   A0-F4   Unified mode normalization + mass-weighted MAC
  Step 5   A0-F5   Config option: omitted vs complete load sensitivity
  Step 6   A0-F6   Update all docs/comments to match
  Step 7   A0-G    Gate A0 parity test (MATLAB vs Python)
            ★ GATE A0 must pass before any further step ★

Phase 1 — Infrastructure (code)
  Step 8   I1-1    Telemetry fields in all solver result structs
  Step 9   I1-2    Fail-loud master runner
  Step 10  I1-3    Per-experiment isolated output dirs
  Step 11  I1-4    Smoke failing test
            ★ GATE I1 must pass ★

Phase 2 — Verification (code + small-mesh runs)
  Step 12  V1-1/2  Eigenpair and load regressions
  Step 13  V1-3/4  Objective and sensitivity-variant regressions
  Step 14  V1-5    FD sensitivity check (complete gradient, ≤1e-5)
  Step 15  V1-6    MAC and mode-reordering regression
  Step 16  V1-7    MATLAB/Python parity on all V1 checks
            ★ GATE V1 must pass ★

Phase 3 — Scientific Experiments (one production config first, then sweeps)
  Step 17  E2-1    Reproduce Exp2 failure with stack trace
  Step 18  CR2-1/2 Implement Variant A and Variant B load-sensitivity
  Step 19  CR2-3   FD verification of Variant B
  Step 20  E2-2/3  Fix Exp2 and run α sweep to convergence
  Step 21  E2-4/5/6/7  Export Exp2 artifacts; diagnose α=0.75; A5 check; comparison
  Step 22  E3-1/2/3   Exp3 mesh convergence (declare criterion first)
            ★ GATE E23 must pass ★
  Step 23  CR2-4/5    Run CR2 Variant A vs B to convergence; compare
  Step 24  A4-1…4     Run A4 N sweep; track modes; report N=1 faithfully
            ★ GATE E4 must pass ★
  Step 25  S1-1…3/5   Rerun building cases; diagnose modes; report grayness
            ★ GATE S1 (classification) must pass ★

Phase 4 — Performance Evidence (only after scientific gates)
  Step 26  P1-1/2  Instrument solvers; benchmark mode
  Step 27  P1-3    Warm-up + ≥10 timing runs per mesh
  Step 28  P1-4    Investigate Yuksel iteration discrepancy
  Step 29  P1-5    Correct comparator labels
  Step 30  P1-6    Recompute scaling fit
  Step 31  P1-7    Replace/verify 8.6% / 7.1× claims (→ Tier 2 decision)
            ★ GATE P1 must pass ★

Phase 4b — Tier 2 decisions (parallel with Phase 4 or after)
  Step 32  T2-1    Canonical Olhoff benchmark (if claim retained)
  Step 33  S1-4    RAMP/Heaviside mitigation (if spurious-mode claim retained)

Phase 5 — Manuscript and Response (only after all gates pass)
  Step 34  MS-18   Regenerate all tables and figures from accepted artifacts
  Step 35  MS-1…17 Corrections and qualifications in order listed
  Step 36  MS-19   Response letter

Phase 6 — Reproducibility Package
  Step 37  RP-3    Collect sysinfo and run log
  Step 38  RP-1/4  Manifest and supplementary inventory
  Step 39  RP-7    Checksums for accepted artifacts
  Step 40  RP-2    Clean-run instructions
  Step 41  RP-5/6  Release tag + archive / DOI
  Step 42  RP-8    Machine-readable completion report
```

### Hard ordering constraints

- Gate A0 before any production experiment.
- Gate I1 before any experiment is counted as accepted.
- Gate V1 before Exp1–Exp5 are regenerated.
- Gates E23 and E4 before the performance study (Exp1/Exp5).
- All scientific and performance gates before manuscript regeneration.
- Manuscript regeneration (MS-18) before all other MS items.
- Accepted artifacts before reproducibility package.

### Parallel-safe steps (within a phase)

- A0-F1 through A0-F6 can proceed in parallel once the formulation decision is fixed (they touch different code sections).
- V1-1 through V1-6 can run in parallel once A0-G passes.
- CR2 and A4 are independent; E3 is independent of E2 once the solver is fixed — all three can run in parallel after V1 passes and E2-1 is resolved.
- S1-1 can start as soon as the authoritative formulation is stable (does not require E23 gate to initiate, but must converge before S1 gate is checked).
- P1-1 and P1-2 (instrumentation) can be implemented during Phase 3 in the background, as long as no timing data enters claims before Gate P1.
