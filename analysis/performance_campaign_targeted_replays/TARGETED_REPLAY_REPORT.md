# Targeted Replay Report

## Outcome

All three authorized cases reproduce their original campaign observations under numerically identical frozen profiles. No replay exposes implementation corruption, invalidates an original observation, or changes an admissible-row timing/scaling fit. The campaign can be frozen with its five censored observations intact.

Replay wall times are diagnostic only: Olhoff 1010.84 s, Yuksel 910.735 s, Proposed 7.47762/7.54851 s. They do not replace or refit original timing measurements.

## Scope and hard constraints

This closure covers only the already completed Olhoff 640x80, Yuksel 800x100, and two identical Proposed 160x20 diagnostic executions. All work after those executions was offline. No cap, tolerance, move, filter, eigensolver, LP option, optimizer, stabilization rule, or material interpolation was changed; no broad campaign rerun, tuning run, cap extension, RAM repair, or timing substitution was performed.

## Provenance

- Repository: branch `benchmark-methodology-r2`, commit `632e9b01811845709de33f93051fd853373ed5e1`.
- Runtime: MATLAB `25.2.0.2998904 (R2025b)`, `MACA64`, one computation thread.
- Pre-replay worktree: only the already untracked forensic-audit and final-campaign directories.
- Immutable evidence: all 21 files under `examples/Performance/final_campaign/` match the pre-replay forensic SHA-256 inventory; every frozen numerical source in the replay preflight matches its original hash.
- Raw diagnostic MAT files and logs remain under `raw/` and `logs/`. `SHA256SUMS.txt` covers the final closure package and excludes itself.

## Replay implementation audit

**Observation:** the normalized original/replay configurations compare equal for all three cases, and each source-hash gate passed before execution. The Olhoff mirror preserves the production LP objective, matrices, bounds, `dual-simplex-highs` option, ordering, update decision, and S1 rule; it merely requests and stores solver output, then computes diagnostics only after failure. Every retained non-timing Olhoff history field, density, and successful-update snapshot is bit-identical to production evidence. Yuksel history/audit retention is observational, and its exact endpoint/fingerprints reproduce. Proposed history/spectral retention does not feed updates; both runs exactly reproduce the original endpoint. The determinism check compares every retained numerical history field and structural marker while excluding only `elapsed_s`; the exported 107-row histories also compare exactly field by field.

**Inference:** the instrumentation did not alter the numerical behavior of any target.

**Unresolved mechanism:** exact reproduction cannot expose the solver-internal reason that HiGHS stopped the failed Olhoff LP after 38 reported iterations.

## Exact replay configurations

| Method | Mesh | Frozen numerical profile |
|---|---:|---|
| Olhoff | 640x80 | LP; SIMP; sensitivity filter r=1.3 elements, diagonal mode; volume 0.5; rho_min=0.001; tol_mult=0.05; initial/stabilized move 0.005/0.0025; S1 trigger N=2 and gap12<=0.01 for 100 iterations; fixed cap 1600; dual-simplex-highs |
| Yuksel | 800x100 | OC; SIMP p=3; sensitivity filter r=2.5 elements, symmetric boundary; volume 0.5; move 0.1; Stage-1/Stage-2 cap 1000/1000; tolerance 0.01 in each stage; E_min/E=1e-9; rho_min=1e-9 |
| Proposed | 160x20 | OC; SIMP p=3; sensitivity filter r=2 elements, symmetric boundary; volume 0.5; move 0.2; cap 2000; tolerance 0.01; solid semi-harmonic baseline, no load sensitivity/normalization; E_min/E=1e-9; rho_min=1e-9 |

The canonical machine-readable profiles are retained in `configurations/`.

## Table A — Replay configuration identity

| Method | Mesh | Original profile ID | Replay profile ID | Numerical config identical? | Source hashes equal? | Diagnostic-only differences | PASS/FAIL |
|---|---:|---|---|---|---|---|---|
| Olhoff | 640x80 | `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1` | `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1` | 1 | 1 | external failed-attempt LP output/residual retention; numerical cfg unchanged | **PASS** |
| Yuksel | 800x100 | `yuksel_practical_move01_tol001` | `yuksel_practical_move01_tol001` | 1 | 1 | record_history=true; scalar audit retention; density snapshot stride=10 | **PASS** |
| Proposed | 160x20 | `proposed_practical_move02_tol001` | `proposed_practical_move02_tol001` | 1 | 1 | record_history=true; native spectral history=true; identical run repeated once | **PASS** |


## Replay identity results

| Method/comparison | Iteration/status identity | Spectrum/fingerprint identity | Density/stopping/common identity | Verdict |
|---|---|---|---|---|
| Olhoff original vs replay | 1066 successful; failure attempt 1067; trigger k=204; SOLVER_FAILURE | every retained non-timing history field bit-identical | density and all successful-update snapshots bit-identical | FAILURE_REPRODUCED |
| Yuksel original vs replay | Stage 1=1000; Stage 2=1000; total=2000; CAP_HIT | final 160.879109/325.605370/566.129320 rad/s and objective-history fingerprint exact | density fingerprint, final max dx 0.0727982 and RMS dx 0.0006855 exact | ENDPOINT_IDENTICAL |
| Proposed original vs each replay | 107; NATIVE_CONVERGED | native 109.050082/109.489208/112.916324 rad/s and objective/frequency fingerprints exact | density fingerprint and all E1/E2/E3 raw/binary results exact | BOTH_ENDPOINTS_IDENTICAL |
| Proposed replay 1 vs replay 2 | 107 vs 107 | all retained numerical histories and spectra exact; only elapsed timing excluded | density, topology, stopping and common evaluator results exact | DETERMINISTIC |

## Olhoff 640x80

**Observation:** the replay is bit-identical to the original over every retained non-timing history field, the complete density trajectory, and all successful-update snapshots. Stabilization triggers at k=204, the reduced move is 0.0025, 1066 updates succeed, and attempted k=1067 fails again.

The failed attempt directly evaluates to ω = 172.823470961 / 173.265349435 / 200.336593966 rad/s, N=2, gap12=0.0025568198, and λref=29867.9521148. `linprog` returns exitflag 0 after 38 reported iterations, algorithm `dual-simplex-highs`, message “Solver stopped prematurely,” and no point. MATLAB R2025b's local `linprog.m` defines flag 0 as maximum iterations reached. The configured default `MaxIterations` is 2.1475e9 and `MaxTime=Inf`, so the result should be described as MATLAB's iteration-limit exit class, not a user-imposed 38-iteration cap.

The LP matrices are finite. Five normalized constraint rows have rank 5; normalized Gram rcond is 0.0206033 and the inequality-row norm ratio is 237.826. Because no point is returned, feasibility residuals and active-bound statistics are unavailable.

**Inference:** these proxies do not demonstrate degeneracy or pathological scaling. The supported causal class is limited to MATLAB's generic LP iteration-limit exit class.

**Unresolved mechanism:** N=2 and a small gap co-occur with the failure, but modal-branch responsibility, degeneracy, scaling pathology, and the internal reason for the 38-iteration stop are not established.

### Table B — Olhoff failure replay

| Original attempt | Replay attempt | exitflag | LP iterations | Modal state | gap12 | Move | Trigger state | Residual/scaling diagnostics | Reproduction | Causal class |
|---:|---:|---:|---:|---|---:|---:|---|---|---|---|
| 1067 | 1067 | 0 | 38 | 172.823/173.265/200.337; N=2 | 0.00255682 | 0.0025 | S1 already active; trigger k=204 | no returned point; finite matrices; row rank 5/5; Gram rcond 0.0206 | **FAILURE_REPRODUCED** | **GENERIC_LP_ITERATION_LIMIT_ONLY** |

There is no evidence of implementation corruption.

## Yuksel 800x100

**Observation:** the replay exactly matches the original fingerprint and again stops `CAP_HIT` at 1000+1000 iterations. In the final 300 Stage-2 iterations, max dx ranges 0.0109979–0.1, has median 0.0731213, never falls below 0.01, and hits the full 0.1 move limit 61 times. RMS dx has median 0.000668097; median max/RMS is 97.1197. The raw max-dx slopes have 42 strict positive-to-negative/negative-to-positive reversals, 54 after zero-slope plateaus are removed, and 67 state changes when entry to/exit from a plateau is counted. This explicit definition replaces the preliminary “approximately 75” estimate.

Only 3.215–3.515% of variables move above 1e-12 in this window, and P95 is effectively zero, giving a conservative `<5%` / `<4,000 of 80,000` bound on variables that can dominate the maximum. Ten-iteration binary turnover is small but persistent, 0–0.07%. The moving-load objective changes by a median 1.7037e-05 per iteration and drifts 0.5655% across the window.

### Table C — Yuksel late dynamics

| Window | Stage | max dx start→end (median) | RMS median | max/RMS median | Objective trend | Dominant-variable fraction | Diagnosis |
|---|---:|---|---:|---:|---|---|---|
| S2 701-800 | 2 | 0.1→0.08887 (0.07809) | 0.000703 | 101.1 | stable/slight drift; median relative step 1.79e-05 | <5% (<4,000); localized every sampled iteration | **IRREGULAR_OSCILLATION / PERSISTENT_NONCONVERGENCE** |
| S2 801-900 | 2 | 0.09173→0.05707 (0.0751) | 0.000758 | 92.3 | stable/slight drift; median relative step 2.00e-05 | <5% (<4,000); localized every sampled iteration | **IRREGULAR_OSCILLATION / PERSISTENT_NONCONVERGENCE** |
| S2 901-1000 | 2 | 0.05018→0.0728 (0.06966) | 0.000609 | 101.4 | stable/slight drift; median relative step 1.33e-05 | <5% (<4,000); localized every sampled iteration | **IRREGULAR_OSCILLATION / PERSISTENT_NONCONVERGENCE** |
| S2 final 300 | 2 | 0.1→0.0728 (0.07312) | 0.000668 | 97.1 | stable/slight drift; median relative step 1.70e-05 | <5% (<4,000); localized every sampled iteration | **IRREGULAR_OSCILLATION / PERSISTENT_NONCONVERGENCE** |


**Inference:** the primary late behavior is `IRREGULAR_OSCILLATION`, with secondary `LOCALIZED_VARIABLE_MOTION`; the cap interpretation is `PERSISTENT_NONCONVERGENCE`, not `LIKELY_SIMPLE_CAP_LIMIT`. A modest cap extension is not scientifically justified as a convergence-completion experiment because the late trajectory is far above tolerance and irregular rather than a monotone tail.

**Unresolved mechanism:** the history establishes the behavior but not which local variables or update interactions cause it. A future dynamics study could investigate that question; it is not a performance-freeze requirement.

## Proposed 160x20

**Observation:** both identical diagnostic executions reproduce the original 107-iteration native endpoint and are numerically deterministic after excluding only elapsed-time fields. Native ω is 109.050082311 / 109.489207638 / 112.9163242 rad/s. The density, binary topology, objective history, stopping metric, per-iteration native spectrum, and all common evaluator outputs are identical.

### Table D — Proposed determinism

| Run | Iterations | Native ω1/ω2/ω3 | Final dx | Density checksum | Common raw E1 | Common binary E1 | Verdict |
|---:|---:|---|---:|---|---:|---:|---|
| 1 | 107 | 109.050082/109.489208/112.916324 | 0.00985439 | `n=3200;sum=1600.4307665906422;weighted=2561489.4419286405;l2=37.359899455718576` | 153.675210 | 162.759719 | **DETERMINISTIC** |
| 2 | 107 | 109.050082/109.489208/112.916324 | 0.00985439 | `n=3200;sum=1600.4307665906422;weighted=2561489.4419286405;l2=37.359899455718576` | 153.675210 | 162.759719 | **DETERMINISTIC** |


### Table E — Proposed native/common interpretation

| Quantity | Native | E1 raw | E2 raw | E3 raw | E1 binary | E2 binary | E3 binary | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| omega1 | 109.050082 | 153.675210 | 154.335191 | 154.335190 | 162.759719 | 162.759702 | 162.759702 | low native first mode is absent under every common evaluator |
| omega2 | 109.489208 | 212.561561 | 213.414722 | 213.414670 | 231.724368 | 231.710714 | 231.710714 | common model shifts/reorders the clustered native spectrum |
| omega3 | 112.916324 | 274.967465 | 365.548881 | 365.548090 | 397.039268 | 396.819873 | 396.819873 | higher spectrum is strongly interpolation-sensitive |


Grayness is 0.25583585. The native mode recomputation agrees within 1.421e-14 rad/s. For modes 1–3, displacement fraction in rho≤0.1 material exceeds 0.999999996, kinetic fraction exceeds 0.99999858, displacement-weighted density is 0.00236–0.00246, and only 105–136 elements carry 90% of displacement magnitude. The shapes are localized along the nominally void interior band. The raw and exact-count binary load-carrying topologies are both connected and have strong common spectra.

**Inference:** the low native triplet consists of weak-material/local modes produced by the native floors/interpolation on this coarse gray design; the load-carrying skeleton itself is not shown to be qualitatively poor. Primary classification: **MODEL / INTERPOLATION DEPENDENCE**, directly confirmed and refined to weak-material modal localization at 160x20. The common evaluator is a comparison model, not a “true-frequency” model.

**Unresolved mechanism:** no KKT residual, basin/restart study, or every-resolution modal localization analysis was authorized, so stationarity and universal-resolution claims remain unsupported.

## Figures

1. [Olhoff pre-failure history](figures/01_olhoff_640_prefailure_history.png)
2. [Olhoff failed-LP diagnostics](figures/02_olhoff_640_lp_diagnostics.png)
3. [Yuksel late max/RMS change](figures/03_yuksel_800_late_density_change.png)
4. [Yuksel late objective](figures/04_yuksel_800_late_objective.png)
5. [Proposed native spectral history](figures/05_proposed_160_native_spectral_history.png)
6. [Proposed density and topology](figures/06_proposed_160_density_topology.png)
7. [Proposed native mode shapes](figures/07_proposed_160_native_mode_shapes.png)
8. [Proposed native/common spectra](figures/08_proposed_160_native_common_spectra.png)

## Direct answers

1. **Olhoff failure location:** yes, attempted k=1067 after 1066 successful updates.
2. **Exact linprog report:** exitflag 0, dual-simplex-highs, “Solver stopped prematurely,” no point.
3. **LP iterations:** 38 reported iterations.
4. **Feasibility/scaling/degeneracy:** residuals/activity unavailable without a point; finite, full-row-rank proxies do not show an abnormal signature sufficient for causation.
5. **Modal connection:** N=2/small gap and late branching co-occur; a causal link is not demonstrated.
6. **Implementation corruption:** no.
7. **Yuksel cap:** yes, exact `CAP_HIT` reproduction.
8. **Final 300:** persistent irregular excursions between 0.011 and 0.1 with small RMS and objective increments.
9. **Global or localized max:** localized; active fraction 3.2–3.5%, conservative dominant bound <5%.
10. **Objective stabilized:** practically stable with slight 0.57% late-window drift and median relative step 1.70e-5.
11. **Late behavior:** `IRREGULAR_OSCILLATION` plus `LOCALIZED_VARIABLE_MOTION`.
12. **Simple cap limitation:** no; `PERSISTENT_NONCONVERGENCE`.
13. **Proposed determinism:** yes, `DETERMINISTIC`.
14. **Producing topology:** connected truss-like raw/binary skeleton with substantial gray interfaces and a nominally void interior band.
15. **Low native modes:** localized along that low-density interior band.
16. **Gray/weak association:** overwhelmingly weak/low-density, not mainly the solid skeleton.
17. **Common evaluator effect:** yes, it again removes most of the native anomaly.
18. **Primary Proposed explanation:** confirmed `MODEL / INTERPOLATION DEPENDENCE`.
19. **Original observation invalidated:** none.
20. **Per-iteration scaling invalidated:** no; original fits/timing remain unchanged.
21. **Full rerun needed:** no.
22. **Freeze for paper:** yes, with the publication qualifications below.

## Limitations

The failed Olhoff solve returned no primal point, so residuals, feasibility, and active-bound statistics are unavailable. The deeper HiGHS stopping mechanism is unresolved. Yuksel 640x80 was not replayed, and the Yuksel 800 history does not isolate a variable-level cause. Proposed localization is demonstrated directly at 160x20, not every resolution; KKT stationarity and cross-method topology equivalence were not tested. Common evaluators are comparison models rather than ground truth. Original timing has one sample per mesh and method-specific endpoint semantics. RAM remains unreliable and excluded. A final attempt to rerun the offline MATLAB postprocessor was blocked by License Error 15 before execution; the objectively incorrect Olhoff residual panel was therefore regenerated deterministically from its retained CSV diagnostics without invoking an optimizer.

## Conclusions

No unresolved issue materially prevents the benchmark from supporting the qualified claims below. The five censored observations remain scientific results: Olhoff 480/560/640 solver failures and Yuksel 640/800 cap hits. Decision: `NO_FURTHER_RUNS_REQUIRED`.

## PUBLICATION-READY CLAIMS

- Final campaign technical integrity, status labels, and exactly five censored observations.
- Proposed 160x20 native endpoint determinism.
- Necessity of unchanged common evaluators for cross-method interpretation.
- Preservation of original timing tables and censoring masks.

## PUBLICATION-READY WITH QUALIFICATION

- Proposed coarse anomaly as native model/interpolation dependence, directly localized at 160x20.
- Proposed-versus-Yuksel common-raw quality from 320x40 onward, without topology-superiority claims.
- Olhoff 640 failure as a reproducible MATLAB/HiGHS iteration-limit exit class, without a deeper causal claim.
- Yuksel 800 cap behavior as persistent localized irregular motion, without generalizing to Yuksel 640.
- Per-iteration and practical endpoint/total-time scaling, with one-sample, stage, and terminal-semantics qualifications.
- Olhoff fixed-work timing on successful admissible rows and its nonmonotonic solver-failure island.
- p=1.5 only as a declared reference normalization, not a universal fitted law.

## EXCLUDED / NOT SUPPORTED CLAIMS

- Quantitative RAM scaling or replay timings as campaign performance data.
- Universal p=1.5, intrinsic total-time kernel complexity, or universal topology superiority/equivalence.
- KKT/stationarity claims for Proposed, a specific Olhoff degeneracy/scaling/modal cause, or that Yuksel merely needs more iterations.
- Claims that all 27 rows converged or that censored rows are timing/scaling successes.

PERFORMANCE CAMPAIGN FROZEN — READY FOR PAPER

FULL NINE-RESOLUTION RERUN: NOT REQUIRED

FURTHER TARGETED OPTIMIZATION RUNS: NOT REQUIRED
