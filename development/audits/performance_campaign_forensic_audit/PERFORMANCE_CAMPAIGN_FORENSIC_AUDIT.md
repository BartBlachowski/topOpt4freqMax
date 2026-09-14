# Performance Campaign Forensic Audit

## Executive verdict

The completed campaign is technically trustworthy **for the observations it actually made**: all 27 statuses revalidate, exactly five rows are censored, and no censored row entered an existing scaling fit. It is not yet sufficient for an unqualified paper freeze because Proposed/Yuksel raw designs and histories were not retained and the Olhoff failed LP attempt was only logged as a flag, leaving three causal claims dependent on narrow diagnostic follow-ups.

The most important corrections to the preliminary reading are:

- Proposed's apparent 109→159 native jump is primarily **MODEL / INTERPOLATION DEPENDENCE**, not a 50-rad/s common-quality jump. At 160x20, native/common-raw-E1/common-binary-E1 are 109.05/153.68/162.76 rad/s; at 240x30 they are 108.78/157.64/161.47; at 320x40 they are 158.76/158.76/162.12.
- Olhoff completed 357/399/1066 valid updates, but the first failed LP calls were attempted iterations **358/400/1067**. Each call to `linprog(..., Algorithm='dual-simplex-highs')` returned flag 0 (`maximum number of iterations reached`). The aggregate `final_lp_flag=1` and empty `lp_failure_iters` refer only to the last appended successful history row and must not be read as the failed call.
- Yuksel's capped rows cannot be assigned a late-history mechanism from the frozen evidence: `record_history=false`. Terminal max/RMS changes strongly suggest localized motion, but a checksum cannot distinguish monotone decay, stagnation, oscillation, or a topology event.

## WP0–WP2: freeze, inventory, and status semantics

Provenance is frozen in [provenance.json](provenance.json), including branch `benchmark-methodology-r2`, HEAD `632e9b01811845709de33f93051fd853373ed5e1`, MATLAB `25.2.0.2998904 (R2025b)`, the three profile IDs, hashes of configuration/execution sources, and SHA-256 hashes of every frozen result artifact. The frozen tree was read only.

The evidence inventory is [evidence_map.csv](evidence_map.csv). Olhoff retained full MAT histories, final densities, and every successful-update snapshot. Yuksel and Proposed retained terminal spectra, timing/stopping scalars, evaluator results, and checksums, but not density fields or histories. Consequently, checksums establish identity only; they cannot reconstruct topologies or trends.

Independent counts are:

| Method | Verified successful semantic | Count | Censored semantic | Count |
|---|---:|---:|---:|---:|
| Olhoff | VALID_STABILIZED_STATE_AT_FIXED_WORK | 6 | SOLVER_FAILURE | 3 |
| Yuksel | NATIVE_CONVERGED | 7 | CAP_HIT | 2 |
| Proposed | NATIVE_CONVERGED | 9 | — | 0 |

This is exactly five censored rows. Solver failure is never convergence; a cap hit is never convergence; and Olhoff's 1600 endpoint is fixed work, not native convergence. Row-level proof is in [campaign_integrity.csv](campaign_integrity.csv).

## WP3–WP4: Proposed anomaly and same-mesh Yuksel comparison

The native Proposed spectra at 160 and 240 are tightly clustered low triplets (109.05/109.49/112.92 and 108.78/109.92/117.83 rad/s), unlike the 320 spectrum (158.76/229.64/279.39). Yet the frozen common evaluators give Proposed raw omega1 values of 153.68/154.34/154.34 (E1/E2/E3) at 160 and 157.64/158.18/158.18 at 240; exact-count binary values are 162.76 and 161.47 under every evaluator family. Thus the discontinuity shrinks from about 50 rad/s natively to 5.1 rad/s in raw E1 and reverses slightly under binary E1.

The mechanism supported by code and results is low-density interpolation sensitivity. Proposed natively uses `Emin/E0=1e-9` with linear density-to-mass interpolation. Common E1 raises the void stiffness floor to `1e-6`; E2/E3 suppress mass strongly below density 0.1. Both changes eliminate the low native triplet. The effect is strongest where Proposed grayness is largest (0.256, 0.162, 0.122 at 160/240/320). A fixed two-element filter also changes physical radius with refinement and may influence grayness/basin selection, but the retained evidence cannot isolate that secondary contribution.

Compared with Yuksel on the same meshes, Proposed's common raw E1 omega1 differs by -3.49, -1.80, and -1.93 rad/s at 160/240/320, far smaller than the native -48.23, -50.71, and -1.98. Binary E1 places Proposed at +3.97, +1.07, and +0.61 rad/s relative to Yuksel. The primary anomaly classification is therefore **MODEL / INTERPOLATION DEPENDENCE**.

The 160/240 runs genuinely met their frozen native density-change stop and have small terminal relative objective changes. They are legitimate native terminal states, but “stationary/KKT local solutions” is not proved because histories, KKT metrics, densities, and restart tests were not retained. The requested topology comparison is consequently an explicit evidence-gap figure, [Figure 8](figures/08_proposed_topology_comparison_unavailable.png), not an invented reconstruction.

## WP5–WP7: Olhoff failure island and pre-failure quality

The first failing component is exact: `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` calls `Matlab/reproduction2007/algo/innerLoopLP.m`, which calls `linprog` with `dual-simplex-highs`. Flag 0 makes `st.conv=false`; the caller records `SOLVER_FAILURE` and breaks without appending that attempt to `hist`. MATLAB R2025b's `linprog.m` defines flag 0 as maximum iterations reached. It was neither an eigensolver exception, infeasibility flag, nor a nonfinite-state flag.

Although the failed attempt is absent from `hist`, its density is retained as `res.rho` and was passed to the campaign's common evaluators. Common E3 is native-equivalent for this Olhoff profile: across the six healthy endpoints, common-E3 and native omega1 differ by at most 4.5e-8 rad/s. It therefore reconstructs the failed-attempt spectra as 170.571/170.792/187.407, 171.715/171.998/251.226, and 172.823/173.265/200.337 rad/s for 480/560/640. Each is N=2 with gap12 0.00130/0.00165/0.00256; inferred lamref is omega1 squared. These are evaluator-based reconstructions, explicitly labelled as such, rather than directly logged failure-attempt telemetry.

All three failures occur after the causal S1 move reduction (triggers k=207/204/204; move 0.005→0.0025), not at a resource threshold. Larger 720/800 cases trigger at k=204/206 and complete 1600 healthy LP solves. The island is therefore non-monotonic and trajectory-dependent.

The retained trajectories identify a modal precursor but do not retain failed-attempt conditioning. At 480/560, N remains 2 while omega3 alternates sharply among modal branches in the last window (late-50 omega3 standard deviations 21.97 and 38.51 rad/s). At 640, the last 50 updates contain 5 N-switches and the last retained spectrum collapses to 144.62/173.01/173.09 before the next LP reaches its iteration limit. Healthy 720/800 end with N=2, sub-0.2% gap12, and late-50 omega3 standard deviations only 0.10/0.06. This supports a mode-branch/LP-degeneracy interaction, but the precise conditioning signature remains unrecorded.

The failed cases had reached mature, connected bimodal raw states. Their best valid native omega1 values were 170.64, 171.70, and 173.07 rad/s, with raw common E1 170.24, 171.26, and 172.76. They are not uniformly evaluator-robust: binary E2/E3 omega1 collapses to roughly 4–7 rad/s because threshold connectivity coexists with detached binary fragments (largest-component fractions below one). These pre-failure states remain diagnostic only and are not promoted into benchmark successes. See [olhoff_failure_forensics.csv](olhoff_failure_forensics.csv), [best-prior quality](olhoff_best_prior_quality.csv), [histories](olhoff_histories.csv), and [Figure 9](figures/09_olhoff_failure_neighborhood_histories.png).

## WP8–WP9: Yuksel cap hits

640x80 and 800x100 both execute Stage 1=1000 and Stage 2=1000, while 720x90 executes 1000+966 and meets the Stage-2 native criterion. This is not contradictory: Stage 1's 1000 is the prescribed handoff budget; the terminal success test is met in Stage 2.

At 640, final max/RMS dx are 0.03065/0.000366 with relative objective change 5.85e-6. At 800 they are 0.07280/0.000686 with relative objective change 2.64e-5. The max-to-RMS ratios show that a small subset of variables dominates the terminal maximum, while the global field and objective move little. Without the final 200 samples, however, this does not distinguish a localized oscillation from a late event or slow monotone tail.

- 640x80: **POSSIBLE_CAP_LIMIT**. A modest extension is plausible but not demonstrated.
- 800x100: **INDETERMINATE**. Its terminal max change is too far above tolerance to call a simple cap limit from one sample.

[Figure 10](figures/10_yuksel_late_history_unavailable_terminal_diagnostics.png) therefore shows only terminal diagnostics and labels the missing trajectory explicitly.

## WP10–WP14: computational scaling and timing

### Computational kernel / per-iteration scaling

| Method/stage | C_iter | p_iter | log R² | n | 95% CI for p |
|---|---:|---:|---:|---:|---:|
| Olhoff, all | 2.426e-06 | 1.194 | 0.9961 | 6 | [1.090, 1.298] |
| Yuksel, all | 5.884e-06 | 0.975 | 0.9705 | 7 | [0.779, 1.170] |
| Yuksel, Stage 1 | 4.148e-06 | 0.996 | 0.9815 | 7 | [0.838, 1.153] |
| Yuksel, Stage 2 | 7.679e-06 | 0.960 | 0.9583 | 7 | [0.730, 1.190] |
| Proposed, all | 8.381e-07 | 1.189 | 0.9606 | 9 | [0.973, 1.404] |

Stage 2 is measurably more expensive per iteration than Stage 1, so the combined Yuksel average should not replace the stage-specific fits. Full coefficients are in [per_iteration_scaling.csv](per_iteration_scaling.csv).

### End-to-end practical scaling

| Method | wall-time p | log R² | n | terminal semantics |
|---|---:|---:|---:|---|
| Olhoff | 1.193 | 0.9960 | 6 | time to fixed-work stabilized endpoint; not convergence |
| Yuksel | 1.706 | 0.9814 | 7 | time to native terminal state |
| Proposed | 1.418 | 0.9640 | 9 | time to native terminal state |

These reproduce the campaign's approximately 1.193/1.706/1.418 exponents. They are practical endpoint exponents, not intrinsic kernel complexity.

On identical admissible rows, log-slope decomposition for loop time is exact up to rounding: Yuksel p_loop = p_iter 0.975 + p_count 0.758; Proposed p_loop = p_iter 1.189 + p_count 0.279. Olhoff's count exponent is exactly zero by protocol, so its loop exponent is its per-iteration exponent. The growing Yuksel count steepens total scaling; Proposed's mildly decreasing/non-monotonic count offsets its steeper kernel scaling.

The fixed p=1.5 assessments are in [fixed_p15_assessment.csv](fixed_p15_assessment.csv). The declared audit rule is: EMPIRICALLY WELL SUPPORTED when the free-fit 95% interval contains 1.5 and fixed-fit MAPE is at most 15%; USEFUL NORMALIZATION ONLY when fixed-fit log R² is at least 0.90 and MAPE at most 35%; otherwise POOR MODEL. These are empirical goodness-of-reference tests, not theoretical complexity proofs. Whenever retained, C is a normalized empirical coefficient with units tied to the fitted Ne convention, not an intrinsic constant.

For Proposed and Yuksel, `wall ≈ init + loop + post` to within at most 0.0043 s of caller/measurement overhead. Yuksel Stage 1+Stage 2 equals loop time exactly in the stored precision. For Olhoff, init and post correctly remain null; wall−loop is positive unattributed time and is never converted to zero. See [timing_decomposition.csv](timing_decomposition.csv).

The most defensible main computational-complexity quantity is **per-iteration optimization-loop time**, with Yuksel stages separated. Total wall time should be a second, practical endpoint table with explicit terminal semantics. Loop total is useful for exact cost×count decomposition but is not end-to-end.

## WP15: memory

`MaxRAM_MB` is not absolute RAM or allocator peak. The code samples MATLAB process RSS every 0.25 s and reports peak RSS minus RSS at case start. Sequential allocator retention changes the baseline; short-lived peaks may be missed; method order can contaminate deltas; and there is one sample per case. The non-monotonic values are therefore not reproducible evidence of memory scaling.

**Classification: UNRELIABLE.** Do not publish quantitative memory comparisons from this campaign. Raw values and rationale are in [memory_assessment.csv](memory_assessment.csv).

## WP16–WP17: common quality

Olhoff's higher native first frequency survives all common **raw** evaluators on its six admissible rows: common raw E1 is about 166.7–172.9 versus roughly 157–161 for Yuksel/Proposed. It does not survive every representation: binary E2/E3 collapses to single-digit frequencies at valid 720/800 because disconnected binary fragments introduce low modes despite a left-to-right connected main component. The publication-safe claim is therefore “higher common-raw first-mode frequency at the fixed-work endpoint,” not universal topology superiority.

From 320 onward Proposed and Yuksel approach comparable common-raw first-mode quality (generally within about 1.2%), but their higher spectra and grayness differ. Binary E1 is also generally close after 480; Yuksel 400 is a notable binary pathology. No method can be declared topology-equivalent from scalar evaluators, and Proposed/Yuksel topology fields were not retained.

The consolidated 27-row table is [common_quality_comparison.csv](common_quality_comparison.csv). [publication_readiness.csv](publication_readiness.csv) separates supported, qualified, and unsupported claims.

## Direct answers to the 20 required questions

1. **Technically trustworthy?** Yes for terminal/status, timing, and common-evaluator observations; incomplete for three mechanistic diagnoses.
2. **Why Proposed 109→159?** Primarily native low-density material/mass interpolation sensitivity on grayer coarse fields; topology/basin details are not retained.
3. **Still present under common evaluation?** No. It shrinks strongly in raw E1 and disappears/reverses under binary E1/E2/E3.
4. **Why Olhoff 480/560/640 only?** Deterministic trajectory-dependent modal/LP behavior after stabilization, not monotonic size exhaustion.
5. **Exact solver mechanism?** `linprog` dual-simplex-highs exit flag 0 (maximum LP iterations) at attempted k=358/400/1067; native-equivalent E3 reconstruction shows N=2 on all three attempts.
6. **High quality before failure?** Mature connected bimodal native/raw states, yes; not robust under binary E2/E3.
7. **Why Yuksel caps at 640/800 but 720 converges?** Stage-2 trajectories differ; only terminal samples remain, so the causal late behavior cannot be identified.
8. **Would more work help?** 640 POSSIBLE_CAP_LIMIT; 800 INDETERMINATE.
9. **Per-iteration exponents?** Olhoff 1.194, Yuksel combined 0.975, Proposed 1.189.
10. **Yuksel stages?** Stage 1 0.996; Stage 2 0.960 per iteration. Count fits are separate.
11. **Cost vs count?** Yuksel count growth adds about 0.758 to loop scaling; Proposed count behavior adds 0.279; Olhoff adds zero.
12. **p=1.5 defensible?** Only as classified empirically per quantity in the fixed-p table; never as a theoretical conclusion from R² alone.
13. **Main complexity quantity?** Per-iteration loop time; separate Yuksel stages. Wall time is the secondary practical endpoint measure.
14. **RAM publication-ready?** No—UNRELIABLE.
15. **Proposed/Yuksel comparable from 320?** Comparable in common-raw omega1 with qualifications; not proved topology/spectrum equivalent.
16. **Olhoff advantage survives?** In common raw evaluation, yes; not under every binary model.
17. **Publication-ready now?** Status/censoring, per-iteration fits with one-sample qualification, endpoint timing semantics, common-raw comparison, and negative findings.
18. **Requires qualification?** Total-time fits, Olhoff quality, Proposed/Yuksel comparability, and all single-sample timing constants.
19. **Not supported?** Time-to-convergence for Olhoff, universal p=1.5 complexity, quantitative RAM scaling, simple-cap claims, topology equivalence, and KKT stationarity of coarse Proposed states.
20. **Targeted reruns?** Three diagnostic cases before an unqualified paper freeze; no full campaign rerun.

## Required figures

1. [Total wall time vs Ne](figures/01_total_wall_time_vs_Ne.png)
2. [Per-iteration loop time vs Ne](figures/02_per_iteration_loop_time_vs_Ne.png)
3. [Iteration count vs Ne](figures/03_iteration_count_vs_Ne.png)
4. [Yuksel stage counts](figures/04_yuksel_stage_iteration_counts.png)
5. [Native omega1](figures/05_native_omega1_vs_mesh.png)
6. [Common-evaluator quality](figures/06_common_evaluator_quality_vs_mesh.png)
7. [Grayness](figures/07_grayness_vs_mesh.png)
8. [Proposed topology evidence gap](figures/08_proposed_topology_comparison_unavailable.png)
9. [Olhoff failure-neighborhood histories](figures/09_olhoff_failure_neighborhood_histories.png)
10. [Yuksel terminal diagnostics / missing late histories](figures/10_yuksel_late_history_unavailable_terminal_diagnostics.png)
11. [Olhoff best-prior topologies](figures/11_olhoff_best_prior_topologies.png)

## Final decision

CAMPAIGN VALID — TARGETED FOLLOW-UP REQUIRED BEFORE FREEZE

FULL NINE-RESOLUTION RERUN: NOT REQUIRED

Minimal targeted follow-ups: Olhoff 640x80 diagnostics-only replay; Yuksel 800x100 same-cap history-retaining replay; Proposed 160x20 deterministic history/topology/mode-shape replay. Exact hypotheses and confirmation/refutation criteria are in [rerun_recommendations.md](rerun_recommendations.md).
