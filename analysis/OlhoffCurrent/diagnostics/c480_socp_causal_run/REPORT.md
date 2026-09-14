BOTTOM LINE

**The one C480 exact-SOCP treatment run stopped fail-closed at outer iteration 15,
so the causal grayness question is unanswered.**

- **Before it stopped.** The run executed 14 fully certified, near-fully
  bound-saturated SOCP steps. They were well predicted by the local model and pushed
  the design much faster than repeated MMA: at matched iteration 14, ω₁ was 148.0 vs
  133.4 and M_nd 48 % vs 72 %.
- **Why it stopped.** The two lowest eigenvalues had been driven nearly together
  (gap12 2.68 → 0.014). The exact optimum of the 15th sub-problem then sat at the apex
  of the second-order cone, i.e. at a double predicted eigenvalue.
- **What failed there.** Both solver backends returned the same feasible point with
  exit flag 1, but the preregistered certificate could not find a dual witness within
  1e-8 (best 3.6e-4). The run terminated with `SOCP_CERTIFICATE_FAILURE`: no update,
  no fallback, no rerun.
- **Post-hoc, excluded from verdicts.** An exact dual solve proves the rejected point
  optimal to within 8.6e-6, not to 1e-8.
- **What this means.** The obstacle is certificate unattainability at the
  multiple-eigenvalue apex, the very regime Du–Olhoff problem (25) is built to reach.
  It is not a demonstrated solver error and not an observed outer-algorithm instability.
- **Also found before launch.** Problem (25) has a numerically flat optimal face:
  different exact backends select different points on it. The protocol was therefore
  amended to the certified-oracle backend before any treatment state existed.

```
C480_CONTROL_EVIDENCE_PASS
C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS
C480_FULL_RUN_SOCP_COVERAGE_FAIL
C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE
OUTER_MODEL_REALIZATION_INCONCLUSIVE
SOCP_INNER_SOLVER_CANDIDATE_REJECTED
FILTER_FORMULATION_STUDY_STILL_DEFERRED
PERFORMANCE_CAMPAIGN_STILL_BLOCKED
```

**Final scientific question, one sentence.** When Du–Olhoff problem (25) is solved
exactly and certified at every outer iteration, the 480×60 run neither loses nor keeps
its broad gray endpoint and shows no outer-algorithm failure. It never reaches an
endpoint: after 14 well-realized, fully move-saturated steps the exact sub-problem
optimum lands on the double-eigenvalue cone apex, where the preregistered certificate
is unattainable, and the run stops fail-closed at iteration 15.

---

| | control (retained canary) | control at outer 14 | **treatment (exact SOCP)** |
|---|---|---|---|
| status | CONVERGED, outer 386 | — | **SOCP_CERTIFICATE_FAILURE at outer 15**; 14 accepted |
| ω₁ / ω₂ | 163.932 / 185.210 | 133.433 / 174.616 | **148.029 / 150.148** |
| gap12 | 0.1298 | 0.3086 | **0.0143** |
| M_nd | 26.34 % | 71.84 % | **48.13 %** |
| gray / mid fraction | 0.287 / 0.119 | 0.799 / 0.350 | **0.545 / 0.307** |
| broad-core fraction / area | 0.130 / 1.041 | 0.702 / 5.613 | **0.363 / 2.900** |
| increments on a bound | 0 % (all 386) | 0 % | **median 99.98 %** |
| median r (realized/predicted λ₁ gain) | 0.711 (386) | 1.058 (first 14) | **1.033 (14)** |
| inner cost per outer iteration | 6.96 s (run mean) | 1.40 s | **≈ 37.1 s** |

## Answers

1. **Was the authoritative existing C480 control identified exactly?** Yes. The
   trajectory hash, ρ₀/ρ₃₈₆ hashes, config hash, `+impl` tree, 386 iterations, S1/S2/S3 =
   308/347/386 (all branch B) and levels [0.04 0.02 0.01] with `stageExhaustion`
   signal and stop all match. Geometry, per-iteration grayness, spectra and KKT
   reproduce the prior audits exactly. It is the three-rung canary, not legacy
   four-rung, beta, or old stopping (`CONTROL_IDENTITY.md`).
2. **Was the treatment a true single-factor change?** Yes. The treatment received the
   control's own `cfg` struct: `isequal`, same hash, 0 of 81 rows different. Production
   `olhoffSolve.m` reconstructs exactly from the tagged study copy. A 3-iteration replay
   of the control through the study driver was bitwise identical (`SINGLE_FACTOR_DIFF.md`).
3. **What exactly changed?** Only the inner solver of problem (25): repeated MMA
   (`innerLoop` → `mmasub`) became exact SOCP (`fp_problem` cone form, `coneprog` schur,
   then augmented), accepted only with an independent certificate.
4. **What remained frozen?** Everything else. Mesh, ρ₀, ρ_min, volume, p = 3, eq. 4b
   mass (q = 1), FE, eigs, normalization, sensitivities, the Sigmund filter at R = 0.06
   on all f_sk, N = 2 subspace with offsets and off-diagonals, next-mode row, move ladder,
   A-OR-B controller, stop rule, caps, single thread, and production code (tree hash
   verified).
5. **Did every accepted treatment inner problem satisfy the proved SOCP eligibility
   conditions?** Yes. E1–E8 passed at all 15 states, and E9–E12 at all 14 accepted
   points (worst cone error 1.5e-14, worst gradient error 2.3e-17).
6. **Were any N>2 or otherwise unsupported cases encountered?** No. N = 2 at 15 of 15
   iterations; zero unsupported-case hits. The stop was a certificate failure, not an
   eligibility failure.
7. **Did every accepted SOCP solution satisfy the certification thresholds?** Yes, all
   14 (worst gap 6.2e-9, worst primal residual 2.5e-11, worst normalized box
   complementarity 4.8e-8). One (outer 5) needed the preregistered second backend. The
   15th solution satisfied none of the certificate candidates (C3 gap, C6 box
   complementarity) and was rejected (`SOCP_CERTIFICATION.md`).
8. **How many outer iterations were executed?** 15: 14 accepted updates plus 1 rejected
   (no update). Wall time 667 s.
9. **At what iterations did S1, S2 and S3/controller events occur?** None in the
   treatment: it stopped in stage 1, and the earliest possible declaration is outer 39.
   Control: 308 / 347 / 386, all branch B (`CONTROLLER_EVENTS.md`).
10. **How often was the SOCP solution bound saturated?** At every step. Median 99.98 %
    of entries on a bound; ≥ 99 % at 13 of 14 steps; ≥ 99.9 % at 12. Minimum 94.2 %
    (outer 3, with 1 676 interior entries).
11. **How often were gray elements assigned full ±move?** Median 99.97 % of gray
    elements, at every accepted step (minimum 94.2 % at outer 3). All gray elements
    took the full move at 4 steps.
12. **Were there persistent sign reversals or two-cycle behavior?** No two-cycle:
    cos(Δρₖ, Δρₖ₋₁) stayed 0.22–0.84, recurrence 1.6–1.9, and no step had cos < −0.5.
    Element sign reversal did rise from 8 % to 38 %, up to 57 % in the gray core, with
    coherence falling monotonically. That is dithering onset, not cycling
    (`MOVE_REVERSAL_ANALYSIS.md`).
13. **What was the distribution of predicted-vs-realized gain ratios?** Treatment, 14
    steps: median 1.033, quartiles 1.014 / 1.050, range 0.897–1.312, Σact/Σpred 1.041.
    Control, all 386: median 0.711, quartiles 0.428 / 0.911.
14. **Did positive predicted gains normally produce positive realized gains?** Yes,
    14 of 14. λ₁ never decreased.
15. **Did the outer local model remain trustworthy?** For the 14 observed steps, yes
    (median |relative error| 4.8 %). The preregistered verdict is INCONCLUSIVE because
    fewer than 20 steps exist, and the late-stage and small-move regimes, where
    instability would be expected, were never reached (`MODEL_REALIZATION.md`).
16. **What is final ω₁ for control and treatment?** Control 163.93226 (outer 386).
    Treatment 148.02874 at its last accepted design (outer 14), versus 133.43279 for
    the control at outer 14.
17. **What is final M_nd?** Control 26.342 %. Treatment 48.127 % at outer 14, versus
    71.844 % for the control at outer 14.
18. **What are final gray fractions?** Control 0.2873. Treatment 0.5451 at outer 14,
    versus 0.7990 for the control at outer 14.
19. **What are final mid-density fractions?** Control 0.1186. Treatment 0.3073 at
    outer 14, versus 0.3497 for the control at outer 14.
20. **What are final broad-gray-core fractions/areas?** Control 0.1301 / 1.041.
    Treatment 0.3625 / 2.900 at outer 14, versus 0.7017 / 5.613 for the control at
    outer 14.
21. **Did broad physical gray patches disappear, shrink, stay similar or grow?** This
    cannot be determined: the treatment has no endpoint. At matched iteration 14 its
    broad core was about half the control's (0.36 vs 0.70), and max depth/R was
    6.7 vs 11.2.
22. **How different are the final density fields quantitatively?** Against the
    control endpoint: ‖Δρ‖₂ = 38.9, correlation 0.845, threshold agreement 83.2 %,
    relocation 14.4 %. At the matched iteration: ‖Δρ‖₂ = 30.9, correlation 0.874,
    threshold agreement 83.5 %, relocation 13.3 %. Same global layout; the treatment
    had denser flanges and web and a lighter center and support ends
    (`TOPOLOGY_COMPARISON.md`).
23. **Did physical KKT residual improve?** Not determinable. The mechanical ratio is
    "worsened" (gray-fit RMS 0.926 vs 0.334 at the control endpoint, 0.453 at matched
    14). But the treatment design is unconverged and at a near-double eigenvalue, where
    the simple-branch test is not valid (`STATIONARITY_COMPARISON.md`).
24. **Did filtered/subproblem residual improve?** Not determinable, for the same
    reasons. Mechanically: 0.838 vs 0.049 at the control endpoint (worsened); 0.838 vs
    0.455 at matched 14 (similar).
25. **Did exact solution of problem (25) materially change the trajectory?** Yes, over
    the 14 observed steps. Increments were bang-bang instead of never on a bound;
    ω₁ at iteration 14 was 10.9 % higher; M_nd fell 51 points instead of 28. The two lowest
    modes coalesced by outer 14; the control's gap12 was still 0.43 then and reached its
    minimum 0.0034 only later (first < 0.05 at outer 20).
26. **Was the repeated-MMA solver a major cause of C480 grayness?** Unknown. Not
    established, not refuted.
27. **Was it only a partial cause?** Unknown.
28. **Is remaining grayness now more plausibly attributable to formulation/filter
    behavior?** No new evidence either way. The filter study stays deferred.
29. **Did exact inner solution expose an outer-globalization problem?** Not within 14
    steps: realization was healthy, λ₁ was monotone, and there was no cycling. It
    exposed a different obstacle, certification at the double-eigenvalue cone apex.
    Whether a globalization problem would follow cannot be observed.
30. **Is SOCP still a scientifically plausible inner solver?** The preregistered
    certified-SOCP candidate is REJECTED: coverage failed because its apex
    certificate is incomplete. The primal evidence does not indict SOCP itself. Both
    backends agreed to 8e-13 in objective, exit flag 1, and a post-hoc dual bound gives
    ≤ 8.6e-6 suboptimality. A revised candidate is plausible only if an apex
    certificate is validated first.
31. **Is a filter-formulation study now justified?** No. `FILTER_FORMULATION_STUDY_STILL_DEFERRED`.
32. **Is the nine-mesh campaign still blocked?** Yes. `PERFORMANCE_CAMPAIGN_STILL_BLOCKED`.
33. **What is the single highest-information next experiment?** A zero-density-update
    frozen apex-certificate study at the rejected outer-15 state, which is reproducible
    bitwise from the declared evidence. Build and validate a structured conic dual
    certificate, or an apex-constrained LP re-solve, for the double-eigenvalue optimum.
    Only then preregister one new C480 treatment (`NEXT_ACTION.md`).
34. **Were any non-authorized scientific parameters changed?** No. Amendment 1 changed
    only the order of two backends for the same conic problem, and added non-gating
    telemetry. It was made before launch with no threshold changed. After launch
    nothing changed (`PROVENANCE.md`).

## Documents

[AUDIT_PREREGISTRATION](AUDIT_PREREGISTRATION.md) ·
[PREREGISTRATION_AMENDMENT_1](PREREGISTRATION_AMENDMENT_1.md) ·
[PROVENANCE](PROVENANCE.md) · [CONTROL_IDENTITY](CONTROL_IDENTITY.md) ·
[TREATMENT_CONFIG](TREATMENT_CONFIG.md) · [SINGLE_FACTOR_DIFF](SINGLE_FACTOR_DIFF.md) ·
[SOCP_ELIGIBILITY](SOCP_ELIGIBILITY.md) · [SOCP_CERTIFICATION](SOCP_CERTIFICATION.md) ·
[CONTROLLER_EVENTS](CONTROLLER_EVENTS.md) · [MODEL_REALIZATION](MODEL_REALIZATION.md) ·
[MOVE_REVERSAL_ANALYSIS](MOVE_REVERSAL_ANALYSIS.md) · [ENDPOINT_METRICS](ENDPOINT_METRICS.md) ·
[STATIONARITY_COMPARISON](STATIONARITY_COMPARISON.md) · [TOPOLOGY_COMPARISON](TOPOLOGY_COMPARISON.md) ·
[COST_ANALYSIS](COST_ANALYSIS.md) · [CAUSAL_VERDICT](CAUSAL_VERDICT.md) ·
[NEXT_ACTION](NEXT_ACTION.md) · [PERFORMANCE_STATUS](PERFORMANCE_STATUS.md) ·
[MASTER_METRICS.csv](MASTER_METRICS.csv) · [METRICS.json](METRICS.json)

## Figures

`figures/`: FIG_01 control ρ (final and outer 14) · FIG_02 treatment ρ · FIG_03 differences ·
FIG_04 histograms · FIG_05 gray/mid/broad · FIG_06 M_nd · FIG_07 gray · FIG_08 mid ·
FIG_09 ω₁/λ₁ · FIG_10 predicted vs realized · FIG_11 r_k · FIG_12 bound saturation ·
FIG_13 reversal/cosine · FIG_14 controller timeline · FIG_15 physical KKT · FIG_16
filtered residual · FIG_17 gray components · FIG_18 model error by stage · FIG_19 cost ·
FIG_20 causal summary.
