BOTTOM LINE

The frozen production formulation `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` was run once, at HEAD `b21483b`, through the authoritative runner `examples/Performance/performance_comparison.m`, on exactly the nine preregistered meshes from 160x20 to 800x100. The runner executed only the Olhoff method.

- **Termination.** All nine meshes ended **NATIVE_CONVERGED**, with no CAP_HIT, no nested-MMA failure and no retry.
- **Identity.** Every per-mesh configuration hash matched the frozen identity before and after its solve.
- **Reproduction.** The final designs, frequencies, iteration counts and full histories are **bit-identical** to the earlier upstream fixed-R sweep at all nine meshes.
- **Frequencies.** ω₁ follows a stable empirical mesh trend: 169.21 → 165.43 rad/s, with a 0.35 % range over the five finest meshes. ω₂ and the terminal gap vary strongly and non-monotonically. 160x20 alone is bimodal, with a gap of 0.70 %. No localized low-density mode appears.
- **Fine-mesh changes.** 720x90 and 800x100 are about 22 % grayer, need 204 and 246 outer iterations, and differ in layout: gray braces at 720x90, left–right asymmetry at 800x100. 320x40 is also asymmetric.
- **How the stop fires.** At every mesh it is the first crossing of ‖Δρ‖₂ just under ε (0.95–0.996 ε), while the design is still slowly clearing gray.
- **Verdicts.** Under the preregistered rules, the formulation is a **validated observational production baseline**. That verdict makes no claim of KKT stationarity, of topological mesh convergence, or of insensitivity to ε.

```
NINE_MESH_CAMPAIGN_IDENTITY_PASS
PERFORMANCE_RUNNER_CONFIG_IDENTITY_PASS
NINE_MESH_CAMPAIGN_INTEGRITY_PASS
PEDERSEN_ADAPTIVE_CROSS_MESH_TERMINATION_PASS
PEDERSEN_ADAPTIVE_MESH_TREND_ACCEPTABLE
OLHOFF_PRODUCTION_BASELINE_VALIDATED
```

The rules are in PREREGISTRATION.md §10 (frozen before any solve, SHA-256 `3cd56751…`). The machine evaluation is in EVIDENCE.json `verdict_inputs`. Two check-script defects were found, one before and one after the campaign. Both are preserved and disclosed (INTEGRITY_AUDIT.md §1 and §5), and neither changed a preregistered criterion.

---

## Answers

**1. Did the campaign start from exactly HEAD `b21483b158f58e05e7b56957f2fbe8e1d2891395`?**
Yes. HEAD was verified by the Part 1 check, the Part 4 check, the launcher, hook arming, all 9 PRECHECKs, all 9 POSTCHECKs, and after the run: 20 snapshots, all identical. The branch was `benchmark-methodology-r2`. The `+impl` tree was `4ba9a3ae…` and SOURCE_MANIFEST `aee44aa9…`.

**2. Was `performance_comparison.m` the actual production execution path?**
Yes. MATLAB executed `run('examples/Performance/performance_comparison.m')`, which calls `confbench_method_config` → `confbench_run_case` → `olhoffcurrent_run` → `olhoffSolve`. Nothing else started a campaign solve. The runner's own preflight passed (148 checks), and it produced the full artifact set in `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/`. It derived `scientific_evidence = 1` and `performance_campaign = 1`.

**3. What exact MATLAB command invoked it?**
The launcher was `scripts/nmp_launch.sh campaign`, which ran:
`cd …/logs/matlab_cwd_campaign && NMP_LOCK=…/CAMPAIGN_LOCK.json NMP_LOCK_SHA256=202c1f27… /usr/bin/caffeinate -i -s /Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath('…/nine_mesh_pedersen_production/scripts'); addpath('…/analysis/OlhoffCurrent'); nmp_arm_hooks(); run('…/examples/Performance/performance_comparison.m'); nmp_after_run();"`
The full verbatim command is in RUNNER_INVOCATION.md and `logs/campaign_LAUNCH.json`. The campaign ran from 2026-09-14 01:29:50 to 05:34:54 +02:00 and exited with code 0.

**4. Was only the authorized Olhoff production method executed?**
Yes. The runner's existing `cfg.methods` selector was set to `{proposed: false, yuksel: false, olhoff: true}`. That was one of the two locked literal edits in its USER CONFIGURATION block; the other was `cfg.runLabel`. The runner records contain exactly nine `olhoff` rows. The stdout log shows only Du-Olhoff in the methods header, the warm-up and the nine run lines. The runner's own 48×6 warm-up was discarded and is not campaign data. The edit was verified byte-identical to the locked patch and reverted to the committed bytes afterwards.

**5. Did all nine configuration hashes match the frozen identities?**
Yes. Before the campaign, all nine were resolved independently through the API, the runner's `confbench_method_config` route and the exact `olhoffcurrent_run` pre-solve route. Each equals both `CAMPAIGN_IDENTITY.json` and `NINE_MESH_CONFIGS.json`. During the campaign, the hash of the `cfg` handed to `olhoffSolve` equalled the frozen hash at every PRECHECK. At every POSTCHECK, `effective_config_hash` and a re-hash of the configuration still matched, and the runner records carry the same hashes.

**6. Was `+impl` unchanged throughout?**
Yes. The tree was re-hashed from disk to `4ba9a3ae…` (79 files, manifest ok) in every snapshot, from the pre-campaign checks through the after-run snapshot. `olhoffcurrent_run.m` stayed at its committed bytes in every snapshot.

**7. Were exactly the nine authorized meshes run?**
Yes: 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90 and 800x100, in that order, according to the hook event log timestamps. No other campaign mesh was solved. The only other solves were non-scientific mechanics checks: the runner's 48×6 warm-up, and a pre-campaign cap-3 smoke at 160x20 under a separate label that is never used as evidence.

**8. Was any run repeated?**
No.

**9. If repeated, why?**
Not applicable. No infrastructure failure occurred.

**10. What terminal status did each mesh reach?**
`NATIVE_CONVERGED` at all nine meshes (solver status `CONVERGED`). The log reads `converged at outer iteration N` with N = 121, 111, 101, 93, 112, 130, 156, 204, 246.

**11. Did any mesh hit the cap?**
No. The maximum was 246 of 400.

**12. What are ω₁ and ω₂ for all meshes?** (native, rad/s)

| mesh | ω₁ | ω₂ |
|---|---|---|
| 160x20 | 169.2106 | 170.4000 |
| 240x30 | 167.3424 | 187.1602 |
| 320x40 | 165.8568 | 195.1400 |
| 400x50 | 166.4552 | 198.1318 |
| 480x60 | 166.0093 | 203.4412 |
| 560x70 | 165.8101 | 206.3760 |
| 640x80 | 165.6500 | 205.9730 |
| 720x90 | 165.4234 | 202.4626 |
| 800x100 | 165.4322 | 195.7393 |

The eq. (4) re-evaluation, which is what the task's historical table quotes, is in SOURCE_SWEEP_COMPARISON.md.

**13. What are the terminal gaps?**
Native gaps are 0.70, 11.84, 17.66, 19.03, 22.55, 24.47, 24.34, 22.39 and 18.32 %. The eq. (4) re-evaluation gaps are 0.65, 11.70, 17.53, 18.90, 22.40, 24.30, 24.17, 22.23 and 18.16 %.

**14. How does ω₁ vary with mesh?**
- It falls 2.23 % from coarse to fine (−3.78 rad/s).
- The steep part is 160x20 → 320x40, with adjacent changes of −1.10 % and −0.89 %.
- 320x40 (the asymmetric design) lies 0.36 % below 400x50.
- From 400x50 onward ω₁ decreases monotonically in steps of ≤ 0.27 % and is flat at the last step (+0.005 %).
- The range over the five finest meshes is 0.35 %.

**15. How does ω₂ vary with mesh?**
It is not stable. It rises 21 % from 170.40 to a maximum of 206.38 at 560x70, then falls 5.2 % to 195.74 at 800x100.

**16. How does M_nd vary?**
0.115, 0.123, 0.141, 0.122, 0.131, 0.133, 0.133, then **0.162, 0.165**: a plateau up to 640x80 and a step of about 22 % at the two finest meshes. The 800x100 value is 1.44× the 160x20 value.

**17. How does the gray fraction vary?**
0.142, 0.145, 0.166, 0.140, 0.151, 0.153, 0.153, then **0.186, 0.188**. The pattern matches M_nd; the 800x100 value is 1.32× the 160x20 value. At every mesh the design was still becoming less gray at the stop.

**18. How does the outer count vary?**
It is non-monotone: 121, 111, 101, 93 (minimum at 400x50), 112, 130, 156, 204, 246. The fit against NE is p = 0.21 with R² = 0.45, which is not a useful law.

**19. How does total wall time scale?**
223.9, 354.9, 427.2, 570.1, 981.4, 1425.0, 2215.8, 3537.5 and 4901.0 s. A log-log OLS over all nine points gives T ∝ NE^0.96 (R² 0.92), but the leave-one-out exponent range is 0.89–1.14 and the residuals are systematically U-shaped because the outer count is non-monotone. Total time is not well described by a single power law (PERFORMANCE_AUDIT.md).

**20. How does wall time per outer iteration scale?**
Smoothly: 1.85 → 19.92 s, with wall/outer ∝ NE^0.755 (R² 0.992, leave-one-out 0.745–0.797).
- **Nested MMA:** 94–98 % of wall time; inner time per outer scales as NE^0.74.
- **Assembly plus `eigs`:** NE^1.10, 1.8–5.6 % of wall time.
- **Within a run:** the cost per outer iteration rises 2.1–4.3× between the first and last 20 iterations, while MMA iterations per step fall slightly.

**21. Were localized-mode collapses observed?**
No. There were no ω₁ drops by the existing 0.7 criterion; the largest one-step ω₁ decrease is 0.71 %. The terminal mode 1 is the fundamental bending mode at every mesh, with 0.3–0.7 % of its kinetic energy in the ρ ≤ 0.1 region. The common evaluator selects ordinal 1 under E1, E2 and E3 at all meshes.

**22. Were suspicious frequency spikes observed?**
No. The only large spectral transient is the same early ω₂ overshoot to about 300, within the first five iterations of the uniform start at every mesh. It is followed by a near-coalescence with ω₁ at outer 8–12.

**23. Did any fine mesh terminate near multiplicity?**
No. Every mesh from 240x30 upward ends with gap ≥ 11.8 %, and gap12 last falls below 0.05 at outer 31–86. The near-multiplicity phase is longer at finer meshes (24 iterations at 240x30, 76 at 800x100), but it always ends well before termination.

**24. Does 160x20 remain exceptional?**
Yes. It is the only mesh with a persistently near-multiple pair (gap < 0.05 in 114 of 121 iterations) and an effectively bimodal terminal state (gap 0.70 %). It is also the only fragmented design, with 16 components at a 0.5 threshold, and it has the coarsest filter resolution (R = 1.2 elements).

**25. Does the new campaign reproduce the qualitative upstream sweep behaviour?**
It reproduces it exactly, not merely qualitatively. At all nine meshes the final ρ is bit-identical, native and eq. (4) ω are identical, outer and inner counts are identical, and the ω and nInner histories are identical. Every value in the task's historical table matches at its rounding.

**26. Where do the old and new sweeps materially differ?**
Nowhere. The only differences are:
- **wall time:** the new runs take 0.44–0.79× the old wall time, which is environmental because the arithmetic is identical;
- **reporting model:** the quoted table uses the eq. (4) re-evaluation, while this report uses native frequencies;
- **the console-only `verbose` row.**

**27. Is adaptive-box natural termination cross-mesh robust?**
Formally, yes. All nine meshes stop naturally on the same per-element RMS criterion, with no cap, no inner failure, and no box collapse to its floor. The largest box stays at its 0.1 ceiling throughout, so the floor criterion caught nothing. Robustness beyond that is limited:
- every stop is a first crossing just under ε (0.95–0.996 ε);
- grayness is still decreasing at every stop;
- at 800x100, ω₁ is still rising slowly (+0.13 % over the last 20 iterations).

**28. Are fine-mesh endpoints free from obvious termination pathology?**
No preregistered pathology occurred: no CAP_HIT, no inner failure, no box-floor stop, and no terminal spike or evaluator failure. The fine meshes do show a different regime:
- many more outer iterations;
- higher grayness;
- falling ω₂ and gap;
- asymmetric or extra-member layouts;
- an objective not yet fully flat at 800x100.

This is documented as a regime change, not as pathology.

**29. Does the campaign establish KKT stationarity?**
No. The stop is a heuristic design-change test, not a KKT certificate. This campaign measured no optimality residual. The historical `gray_kkt_forensic_audit` verdict `PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE` concerns the historical SIMP + eq. (4b) formulation and states. It remains valid for them and is unchanged, and it does not transfer to the Pedersen designs in either direction (HISTORICAL_KKT_CONTEXT.md).

**30. Is this now a defensible production Olhoff benchmark baseline?**
Yes, under the preregistered rules (`OLHOFF_PRODUCTION_BASELINE_VALIDATED`), and **as an observational baseline**. It rests on:
- frozen and continuously verified identity;
- execution through the real benchmark runner;
- natural termination at all nine meshes;
- an empirically stable ω₁;
- no localized-mode collapse;
- bitwise reproducibility against an independent earlier sweep;
- fully decomposed timing.

It must be cited with its qualifications:
- no KKT claim;
- topology, grayness and ω₂ are not mesh-converged;
- the stop fires just under ε;
- 160x20 is bimodal;
- the regime changes at 720x90 and 800x100;
- the frequency model must be named.

**31. What should be compared next against Yuksel and Proposed?**
1. **Run the same runner and the same nine meshes** with all three methods in one session. Use the committed `cfg.methods` (all true) under a fresh label, so that wall times share one environment. Environment alone changed Olhoff wall time by up to 2.3× between sweeps. The Olhoff rows must again reproduce this campaign's ρ hashes bit for bit, which is a free integrity check.
2. **Compare frequencies with the common E1/E2/E3 evaluator** (`study_evaluate_design`, already run outside the timers), never native Olhoff ω against the other methods' native ω. Report the evaluator's selected ordinal and structural status for every method.
3. **Report cost two ways:**
   - total wall time;
   - method-native cost decomposition: Olhoff outer and nested MMA; Yuksel stage 1 and stage 2; Proposed reference eigenanalysis and SIMP.

   Show per-iteration scaling separately, because the Olhoff outer count is non-monotone. Do not fit Olhoff total time with one power law without the leave-one-out caveat.
4. **Compare discreteness and topology on one convention:** M_nd, gray fraction, connectivity, symmetry, and the 800×100-grid IoU, all per mesh. Flag the 720x90/800x100 grayness step and the Olhoff asymmetric designs as known properties.
5. **KKT audit.** A separate, preregistered KKT-residual audit of these nine Pedersen designs, using the `gray_kkt_forensic_audit` method, before any optimality claim is made for the Olhoff column.

---

## Files

| file | content |
|---|---|
| PREREGISTRATION.md / .sha256 | frozen plan and verdict rules |
| CAMPAIGN_LOCK.json / .sha256, SMOKE_LOCK.json / .sha256 | pinned identity for the hooks and launcher |
| RUNNER_AUDIT.md, RUNNER_INVOCATION.md | runner mechanics, selection, observation hooks, exact command |
| RUN_INDEX.json | order, timestamps, per-run file hashes, attempts (1) |
| INTEGRITY_AUDIT.md | identity, lock, per-mesh PRECHECK/POSTCHECK, Part 17, disclosed defects |
| TERMINATION_AUDIT.md | Part 13 |
| RESULTS_TABLE.csv / .md | the authoritative nine-row table |
| MESH_TREND.md, SPECTRAL_AUDIT.md, TOPOLOGY_AUDIT.md, PERFORMANCE_AUDIT.md | Parts 19, 14, 15, 16 |
| SOURCE_SWEEP_COMPARISON.md | Part 18 |
| SCIENTIFIC_INTERPRETATION.md, HISTORICAL_KKT_CONTEXT.md | Parts 20, 21 |
| METRICS.json, EVIDENCE.json, DATA_MANIFEST.json, FINAL_SHA256.txt | machine-readable metrics, verdict inputs, manifests, hashes |
| figures/ | the nine required figures plus histories and design-change convergence |
| evidence/ | identity checks, resolved configurations, runner patch and restore record, extract, histories, grids, symmetry, tail metrics, negative controls |
| scripts/, postprocess/ | hooks, launcher, lock builder, identity check; extraction, report, finalization |
| logs/ | every MATLAB log, launch and end records, hook-arming and after-run records, smoke logs |

Raw production outputs stay in `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/` (read-only): the runner artifacts, plus `runs/<mesh>/{PRECHECK.json, SOLVER_RESULT.mat, TAP.json, RUN_OUT.mat, POSTCHECK.json}` and `runs/HOOK_EVENTS.log`.
