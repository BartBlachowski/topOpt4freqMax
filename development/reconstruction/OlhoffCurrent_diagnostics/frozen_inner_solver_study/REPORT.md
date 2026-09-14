BOTTOM LINE

The frozen solver failure is multicausal. The strongest isolated mechanism is the coarse absolute accuracy of each MMA approximation; its effect depends strongly on the locally restricted asymptote cap. Production already preserves MMA history, and enabling conservative acceptance makes no difference on the tested paths. The relative-step rule stops at 19 while recovering only 0.277% of the certified gain; the authenticated 5000-call replay still recovers only 53.70%. Correcting cap and approximate-solve accuracy together reaches 99.935% gain recovery at 500 calls, but d2=0.19949 and exact KKT=3.524e-4 still fail the frozen fidelity bars.

Select exact SOCP as the sole candidate for one future, otherwise identical fixed-N=2 C480 causal experiment, with per-subproblem certification and rejection of unsupported cases. It alone passes all bars in two deterministic fresh solves. No topology run or density update was performed, and production/reference files remain unchanged. The filter study stays deferred and the performance campaign stays blocked.

| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0_CURRENT_REPEATED_MMA | 500 | 39.198% | 0.921062 | 1.00529 | 81.725% | 0.000% | 6.345e-02 | FAIL |
| S2_ASYINIT_001 | 500 | 14.481% | 0.945952 | 1.00711 | 75.989% | 0.000% | 6.220e-02 | FAIL |
| S3_CANONICAL_CLAMP | 500 | 22.840% | 0.728821 | 1.02982 | 97.286% | 0.000% | 1.419e-02 | FAIL |
| S4_SUBSOLV_ACCURACY | 455 | 77.965% | 0.824808 | 1.00324 | 99.219% | 22.589% | 1.224e-02 | FAIL |
| S5_UNIT_BOX | 500 | 52.797% | 0.816843 | 1.00866 | 95.981% | 0.000% | 1.378e-02 | FAIL |
| G0_UNSAFE | 500 | 19.886% | 0.733473 | 1.014 | 98.808% | 0.000% | 1.372e-02 | FAIL |
| G1_GCMMA | 500 | 19.886% | 0.733473 | 1.014 | 98.808% | 0.000% | 1.372e-02 | FAIL |
| G2_GCMMA_ACCURATE | 500 | 99.911% | 0.216586 | 1.07377 | 99.722% | 22.596% | 4.679e-04 | FAIL |
| S34_CLAMP_ACCURACY | 500 | 99.935% | 0.199492 | 1.10236 | 99.784% | 22.589% | 3.524e-04 | FAIL |

Certified SOCP: beta=26921.772248325, all fidelity bars PASS, both repetitions bitwise-identical.

## Required questions

1. **Is the SOCP oracle identity intact?** Yes; all reference hashes and fresh objective/constraint, weak-duality, KKT and equivalence checks pass.

2. **Was rho updated anywhere?** No. Zero accepted or outer rho updates; only the unchanged production volume expression is evaluated read-only.

3. **Was any topology optimization run executed?** No. Zero topology runs. Toy mathematical programs validate GCMMA only.

4. **Was production iteration 19 reproduced exactly?** Yes: drho, beta and nInner=19 reproduce bit-for-bit, and fresh call 500 matches the earlier replay.

5. **Does original repeated MMA approach the oracle in objective?** It makes partial, nonmonotone progress: 0.277% recovery at 19 and 53.702% at 5000; it does not reach fidelity.

6. **Does it approach in design space?** Partially: d2 decreases from 0.994774 to 0.829748, far above .01. No convergence is established.

7. **Does it approach the oracle active set?** Not to the declared tolerance: same-bound agreement remains zero at 5000; sign agreement alone is insufficient.

8. **Does relative step correlate with oracle distance?** There is trend correlation (fresh-500 Spearman 0.7670), but every B0 stop hit fails fidelity. It is not a valid optimality proxy.

9. **Can lowering tolInner alone solve the problem?** Not demonstrated; the 5000 replay still fails. A fixed approximate-solve accuracy floor remains. No tolerance sweep was run.

10. **Which state resets between MMA calls?** The internal subsolv Newton primal/dual/slack workspace is freshly initialized. MMA approximation history is not reset between inner calls.

11. **Which state is preserved?** xold1, xold2, returned low/upp, the current increment/beta iterate and the increasing inner counter. These reset only upon a new outer problem.

12. **Are these choices specified by Du & Olhoff?** The paper specifies the frozen increment subproblem and use of MMA, but not numerical state policies, constants, scaling or stopping thresholds.

13. **Does persistent history materially improve convergence?** The frozen sequence already has it. S1 is a bitwise identity; cross-outer warm-start effects remain untested.

14. **Does asymptote initialization materially matter?** Historical .01 changes the path but ends at 14.481% recovery. Active production already uses canonical .5; it is not a missing correction.

15. **Does asymptote update materially matter?** Yes, conditionally. The active cap is .2 widths versus canonical 10. Cap restoration alone improves design distance but worsens gain; at tight approximate-solve accuracy it materially improves both at call 100 and common call 455. This is MODERATE evidence because S4 lacks the registered call-500 confirmation.

16. **Does GCMMA reach the oracle?** Neither native G1 nor accurate G2 meets all registered fidelity bars.

17. **How close does it get?** G2: recovery 99.911%, d2=0.216586, dinf=1.073769, exact KKT=4.679e-04 at 500 calls.

18. **Cheapest method reaching oracle fidelity?** Direct SOCP is the only measured method that reaches every bar: about 45.5s solve plus separately measured assembly/certification.

19. **Where does current MMA disagree?** Across void, gray shell, gray core and solid. Gray carries 82.183% of squared increment error at 5000.

20. **Is disagreement void dominated?** No in oracle design distance. Void/solid contribute substantial reduced-cost loss, but the density-space discrepancy is broader.

21. **Does void amplification correlate with disagreement?** Void RMS amplification is 30.69x; within-void amplification-versus-distance Spearman is negative (~-0.594). This is association, not filter causality.

22. **What accounts for 99.94% saturation?** 9,404 lower-density + 4,906 lower-move + 4,788 upper-move + 9,686 upper-density bounds; 16 interior, zero coincident.

23. **Why only ~33.7% move dominated?** That statistic counts only the 9,694 ±move-bound variables, excluding 19,090 density-limited active bounds.

24. **Is the oracle approximately a threshold rule?** Exactly a coupled reduced-cost KKT rule; approximately filtered F11 versus volume threshold. The first-mode-only rule loses ~0.000600 of gain but still misses strict design fidelity.

25. **Can the N=2 SOCP generalize?** Yes conditionally: the consistent-offset cluster constraint is an affine PSD inequality; N=1 is LP and diagonal equality paths are LP.

26. **What about N>2?** General full coupling gives an SDP, not a universal SOCP. Inconsistent offsets/nonlinear volume need separate proof; no production SDP is claimed.

27. **Is direct SOCP a legitimate candidate?** SOCP_INNER_SOLVER_CANDIDATE_JUSTIFIED; scope and fallback are explicit in SOCP_PROMOTION_GATE.md.

28. **Which single choice has strongest causal evidence?** The absolute approximate-subproblem target epsimin=1e-7. Changing only it to the preregistered 1e-12 meets STRONG causal bars in S3-to-S34 and G1-to-G2 at both 100 and 500 calls. Its effect is conditional: B0-to-S4 alone fails the joint bar, and no approximate variant reaches full oracle fidelity. It is not established as a sufficient single correction.

29. **Which solver should be carried forward?** C. Exact SOCP, restricted to the proven N=2 reconstruction and independently certified before any future outer update.

30. **Is one future corrected C480 run justified?** ONE_C480_CORRECTED_INNER_SOLVER_RUN_JUSTIFIED. None was executed.

31. **Is the filter study deferred?** FILTER_FORMULATION_STUDY_STILL_DEFERRED

32. **Is the nine-mesh campaign blocked?** PERFORMANCE_CAMPAIGN_STILL_BLOCKED

33. **Were production files untouched?** Yes; protected-file and implementation-tree hashes are rechecked at finalization.

## Verdicts

- FROZEN_SOLVER_ORACLE_PASS
- PERSISTENT_STATE_NO_MATERIAL_EFFECT
- STOPPING_RULE_NOT_A_VALID_OPTIMALITY_PROXY
- GCMMA_APPROACHES_BUT_DOES_NOT_REACH_ORACLE
- SOCP_INNER_SOLVER_CANDIDATE_JUSTIFIED
- MMA_FAILURE_MULTICAUSAL
- ONE_C480_CORRECTED_INNER_SOLVER_RUN_JUSTIFIED
- FILTER_FORMULATION_STUDY_STILL_DEFERRED
- PERFORMANCE_CAMPAIGN_STILL_BLOCKED

## Evidence and limitations

The final MATLAB audit re-evaluates 202 saved checkpoints with zero scalar-metric discrepancy and maximum conic/production constraint discrepancy 1.008e-14. G0/G1 increments and nonlinear duals agree bit-for-bit. All recorded fresh approximate iterates are primal feasible, but none passes joint fidelity. S34 has a transient gain-recovery drop at call 211; its favorable terminal objective does not establish reliable convergence.

See [PROVENANCE.md](PROVENANCE.md) for the preregistered reuse of the 5000 replay, logger/validation fixes, numerical warnings and cost limits. [CAUSAL_ATTRIBUTION.md](CAUSAL_ATTRIBUTION.md) gives the isolated comparisons and evidence grades. [MASTER_METRICS.csv](MASTER_METRICS.csv) and [METRICS.json](METRICS.json) contain common metrics; evaluations retains checkpoints and every conservative trial. No experimental variants were added after preregistration. The supplied Pedersen (2000) paper is useful background for localized modes and filtering, but supplies no missing MMA accuracy or state prescription; see [PEDERSEN_2000_CONTEXT.md](PEDERSEN_2000_CONTEXT.md).

## Figures

- [FIG_01_oracle_drho](figures/FIG_01_oracle_drho.png)
- [FIG_02_B0_19_drho](figures/FIG_02_B0_19_drho.png)
- [FIG_03_B0_5000_drho](figures/FIG_03_B0_5000_drho.png)
- [FIG_04_oracle_minus_19](figures/FIG_04_oracle_minus_19.png)
- [FIG_05_oracle_minus_5000](figures/FIG_05_oracle_minus_5000.png)
- [FIG_06_gain_vs_iteration](figures/FIG_06_gain_vs_iteration.png)
- [FIG_07_d2_vs_iteration](figures/FIG_07_d2_vs_iteration.png)
- [FIG_08_dinf_vs_iteration](figures/FIG_08_dinf_vs_iteration.png)
- [FIG_09_kkt_vs_iteration](figures/FIG_09_kkt_vs_iteration.png)
- [FIG_10_step_vs_distance](figures/FIG_10_step_vs_distance.png)
- [FIG_11_sign_vs_iteration](figures/FIG_11_sign_vs_iteration.png)
- [FIG_12_bounds_vs_iteration](figures/FIG_12_bounds_vs_iteration.png)
- [FIG_13_persistence_vs_B0](figures/FIG_13_persistence_vs_B0.png)
- [FIG_14_gcmma_vs_B0](figures/FIG_14_gcmma_vs_B0.png)
- [FIG_15_gain_vs_work](figures/FIG_15_gain_vs_work.png)
- [FIG_16_kkt_vs_work](figures/FIG_16_kkt_vs_work.png)
- [FIG_17_density_class_disagreement](figures/FIG_17_density_class_disagreement.png)
- [FIG_18_oracle_bound_categories](figures/FIG_18_oracle_bound_categories.png)
- [FIG_19_threshold_structure](figures/FIG_19_threshold_structure.png)
- [FIG_20_causal_comparison](figures/FIG_20_causal_comparison.png)
- [FIG_21_d2_vs_calls](figures/FIG_21_d2_vs_calls.png)
