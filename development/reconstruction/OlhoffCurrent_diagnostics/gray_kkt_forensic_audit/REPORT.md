BOTTOM LINE

The authoritative saved designs develop **broad physical gray patches**, but they are **not KKT-stationary optima of the evaluated relaxed FE problem**. Even the volume multiplier fitted to minimize gray-region residual leaves normalized RMS 0.355 / 0.334 / 1.163 at 400 / 480 / 800. Native finite differences validate the raw sensitivities at a much finer scale. The result is robust to bound classification and is not evidence for changing the validated controller.

The strongest demonstrated issue is the mismatch between the physical eigenvalue derivative and the sensitivity-filtered local spectral model supplied to MMA. At 400/480 the latter is nearly flat within gray regions while the former is not. At 800, a close two-mode pair makes approximate subspace optimality important: an optimistic relaxed dual fit reduces raw/filtered residuals to .127/.050, but is not an exact KKT certificate. No claim of a proved MMA coding defect or an isolated causal explanation of the entire mesh transition is made.

Projection is premature; p-continuation is not newly justified; no nine-mesh campaign should start. Resolve filtered-subproblem versus physical optimality first. **Zero optimization runs and zero rho updates were executed.**

| mesh | Mnd % | gray % / area | mid % / area | broad core % / area | max depth / R |
| --- | --- | --- | --- | --- | --- |
| 400 | 15.3732 | 17.960% / 1.43680 | 3.920% / 0.31360 | 0.000% / 0.00000 | 0.05657 / 0.943 |
| 480 | 26.3416 | 28.729% / 2.29833 | 11.861% / 0.94889 | 13.014% / 1.04111 | 0.36667 / 6.111 |
| 800 | 34.4123 | 37.762% / 3.02100 | 16.090% / 1.28720 | 20.870% / 1.66960 | 0.38000 / 6.333 |

| mesh | gray RMS, all-interior dual | gray RMS, best gray dual | gray p95, best gray dual | broad RMS, best gray dual | filtered gray RMS, best gray dual |
| --- | --- | --- | --- | --- | --- |
| 400 | 0.555345 | 0.354769 | 0.781865 | empty | 0.0222383 |
| 480 | 0.444567 | 0.334039 | 0.754716 | 0.171565 | 0.0489931 |
| 800 | 1.16312 | 1.16254 | 1.69269 | 0.311465 | 1.15835 |

## Final verdicts

```
GRAY_FORENSICS_EVIDENCE_PASS
FINAL_STATE_SENSITIVITY_VALIDATED
GRAY_REGIONS_NOT_KKT_STATIONARY
MULTIPLICITY_SECONDARY_OR_TRANSIENT
PROJECTION_CANARY_EXPERIMENT_PREMATURE
P_CONTINUATION_REOPENING_NOT_JUSTIFIED
PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE
```

## Required answers

1. **Are the final artifacts authoritative?** Yes. All three exact density hashes and all five source MAT containers match; 400 is the proven exact S3 counterfactual at 466, not the four-rung last column.

2. **What exact mathematical problem is solved?** Maximize the smallest generalized FE eigenvalue with p=3 stiffness, eq4b mass, mean rho<=.5 and .001<=rho<=1, approached by filtered nonlinear spectral increment subproblems. See FORMULATION_RECOVERY.md for the crucial surrogate distinction.

3. **What does MMA update?** MMA updates [Delta rho; beta/lambda1_ref] within each inner subproblem; the outer solver adds Delta rho to the unfiltered FE/design density.

4. **What variable is filtered?** Only generalized sensitivity vectors are filtered on this path; physical/design rho is not filtered.

5. **What derivative reaches MMA?** The objective derivative is zero in density coordinates and -1 in scaled beta; spectral constraint rows use filtered subspace eigenvalue derivatives /lambda_ref, while volume uses 1/(.5NE).

6. **Thin interfaces or broad gray regions?** 480/800 contain broad physical end-region patches, not merely increasingly resolved thin interfaces.

7. **How much area is gray?** Gray fractions are 17.960%,28.729%,37.7625%; physical areas 1.4368,2.29833,3.0210 out of area 8.

8. **How much is strongly mid-density?** Mid fractions are 3.920%,11.861%,16.090%; physical areas .3136,.94889,1.2872.

9. **Does physical thickness grow?** Maximum gray depth grows .05657→.36667→.38 (R=.06). The large jump is 400→480; further refinement primarily expands area.

10. **What are stiffness and mass contributions?** gK is nonnegative and gM nonpositive for diagonal modes; full signed tensors are retained. Gray median |gK|/|gM| is 1.586/1.423,1.792/.782,.458/.0487 in unitful eigenvalue derivatives.

11. **Do they strongly cancel?** Some do, but not the dominant mesh-growing mechanism: median gray C=.398,.490,.960, with only 11.14%,6.43%,7.39% at C<.1.

12. **Does cancellation strengthen with mesh?** No. The exact-active cancellation becomes weaker. The exploratory two-mode-weighted 800 C≈.469 also does not establish systematic strengthening.

13. **Does filtering suppress raw gradients?** At 400/480 it suppresses spatial variation dramatically (std ratios .0627/.1467), with many sign reversals, while RMS magnitude ratios are about .906. The exact first-mode 800 comparison is different; near-cluster dual treatment matters.

14. **Are gray elements KKT stationary?** No for the physical relaxed problem in its actual rho variable. Filtered gray-subproblem residuals can be much smaller, which is not a physical KKT certificate.

15. **What are normalized residuals?** All-interior-dual normalized gray RMS=.5553,.4446,1.1631. Best possible nonnegative gray-only dual gives .3548,.3340,1.1625. Normalizer is raw interior objective-gradient RMS; full quantiles are retained.

16. **Stationary or merely stuck?** Flat observables coexist with unresolved physical first-order stationarity. The data do not establish whether the last inner MMA solve itself was inaccurate; it may have solved a filtered surrogate adequately.

17. **Did finite differences validate sensitivities?** Yes at the scale needed here: every accepted raw derivative error is <7.85e-5 of raw gradient RMS. Tiny void derivative relative signs remain roundoff-limited; full tensor directional checks also pass at relevant scale.

18. **Does multiplicity explain grayness?** It materially affects terminal 800 stationarity through the close first pair, but does not explain the 400→480 broad-patch transition as a common primary mechanism.

19. **Are 800 warnings aligned with final grayness?** No terminal alignment: 81 flags within 26–113, then 355 warning-free iterations. Broad final-core elements are already gray then and continue evolving afterward.

20. **When is most grayness generated?** Everything starts gray at rho=.5. The persistent broad end patches are already present by stage-1 end; later stages slightly reduce grayness.

21. **What changes between 400 and 480?** Broad end patches appear: broad core 0→13.0% of the domain, mid area triples. The first eigenpair remains well separated at 480; neither raw nonstationarity nor filter flattening begins uniquely there.

22. **Is the controller implicated?** No direct controller inconsistency was found. Its frozen convergence rule does not claim to be a KKT certificate.

23. **Is MMA implicated?** Unresolved local optimality is implicated; a published-MMA implementation defect or inaccurate final inner solve is not established without its dual/iterate state.

24. **Is stiffness/mass balance implicated?** Not as a mesh-strengthening cancellation mechanism. Opposing terms exist, but cancellation-to-zero is not the volume-constrained stationarity condition.

25. **Is the filter implicated?** Yes, strongly in the physical-gradient versus optimizer-subproblem mismatch; direct causality for the whole topology transition remains unisolated.

26. **Is projection justified?** Premature: broadness is proved but stationary relaxed grayness is not. Projection would additionally change filtering/variable representation.

27. **Should p-continuation reopen?** No new evidence justifies reopening p-continuation; the prior failed connectivity experiment is respected within its own scope.

28. **Is a nine-mesh campaign justified?** No. The final performance campaign remains blocked by the optimality issue.

29. **Single highest-information next action?** Resolve the physical-objective versus filtered-subproblem optimality mismatch before any new scientific optimization. The single highest-information follow-up is a separately authorized frozen-state 480 subproblem certification that retains its complete inner primal/dual state and evaluates KKT/complementarity, without applying the proposed density increment. That isolates inadequate inner solution from accurate solution of a surrogate that is inconsistent with physical KKT. It is specified only; no inner optimization was executed in this audit.

30. **Were zero optimization runs executed?** Yes: zero optimization runs, zero optimizer calls, zero density updates. Only retained-data analysis, FE/sensitivity evaluation, derivative perturbations and numerical dual reconstruction. Earlier FD recorder errors caused repeated analysis evaluations, fully disclosed.

## Ranked causes

| cause | evidence level | supporting observations | contradicting observations | remaining uncertainty |
| --- | --- | --- | --- | --- |
| 1. sensitivity filter | STRONG EVIDENCE | Direct physical-gradient/subproblem-gradient mismatch; gray variation std reduced to 6.27% and 14.67% at 400/480; raw sign flips 43.26%/20.28%. Filtered gray-fit residual 0.022/0.049 versus raw 0.355/0.334. | At 800 the single active mode is not flattened; near-cluster dual treatment matters. No filter A/B was run. | Strong for the first-order mismatch; not a complete causal proof of gray-patch generation. |
| 2. MMA / local optimality failure | MODERATE EVIDENCE | Substantial physical KKT residuals remain after global convergence; inner stopping is relative step change, no retained KKT certificate. | Filtered local residual at 400/480 is much smaller; nothing proves a bug in published MMA or failure of the last actual inner solve. | Cannot separate inner accuracy from solving a filtered surrogate without final inner dual/iterate evidence. |
| 3. multiplicity / subspace treatment | MODERATE EVIDENCE | 800 exact simple-mode gray RMS 1.163 reduces to optimistic two-mode bound 0.127 raw, 0.050 filtered; terminal gap12=3.49e-5. | 480 has broad patches and a well-separated first mode. J warnings end at 16/113; not terminal. | Near-cluster local optimality is material at 800; warning-induced path causality is unidentifiable. |
| 4. topology bifurcation / alternate basin | WEAK EVIDENCE | 400-to-480 changes from interface bands to two broad end patches; 800 has merged connected gray network. | Different meshes alone do not demonstrate alternative basins or a bifurcation. | No same-formulation basin comparison or matched-field FE test. |
| 5. relaxed SIMP formulation (stationary gray optimum) | EVIDENCE AGAINST | Interior gray reduced gradients fail a necessary stationarity condition with any nonnegative volume multiplier. | Relaxed formulations can in principle admit gray stationary states; this audit does not exclude other stationary designs. | No global-optimum claim or diagnosis of every individual element. |
| 6. stiffness/mass interpolation balance (strengthening cancellation) | EVIDENCE AGAINST | Median C rises 0.398→0.490→0.960, meaning weaker cancellation for the exact active branch. C<0.1 in only 11.14%,6.43%,7.39% of gray. | Opposing terms do exist; the exploratory 800 dual-weighted C is 0.469, so single-mode comparisons are basis/weight sensitive. | No counterfactual mass law; balance against the volume multiplier is distinct from gK+gM≈0. |
| 7. stopping / controller | EVIDENCE AGAINST | Broad patches already present by stage-1 end; stages 2/3 slightly reduce grayness; terminal global windows are flat; exact frozen policy events verified. | Flat global observables do not establish KKT stationarity. | No more-iterations counterfactual is authorized; controller is frozen, not a KKT certificate. |
| 8. FE discretization | NOT TESTED | Meshes differ and gray support changes qualitatively. | Native eigen residuals and derivative checks do not identify an FE assembly/derivative defect. | No common physical density re-evaluated on different meshes; derivative validation is not mesh convergence. |
| 9. other: telemetry/state indexing | STRONG EVIDENCE | 400 hist.omega(466) is pre-update; reevaluated rho466 gives 166.452298433, consistent with later table. | This indexing difference cannot create the saved gray density field. | Explains a reported scalar discrepancy only, not grayness. |

## Evidence and practical limits

Identity/provenance/config are verified; the exact filter/sensitivity chain is recovered; physical-problem KKT and frozen-subproblem residuals are mathematically distinguished; no unexplained material derivative disagreement remains. Therefore the requested stop conditions were not triggered. The approximate near-cluster dual fit is explicitly not used as an exact physical multiplier. The audit convention .1 is not a standard optimization tolerance, and FD validation does not certify tiny void derivative signs.

All elementwise fields, tensors, samples, trajectories, metrics, source declarations and figures are retained. Read KKT_STATIONARITY.md and FINITE_DIFFERENCE_VALIDATION.md for the key qualifications. Repeated evaluations due to an audit recorder type error are disclosed in evaluations/EXECUTION_NOTES.md. Scientific generation of the 400→480 topology regime is not uniquely identified by the nonstationarity result; a valid stationary-relaxed-gray conclusion is nevertheless ruled out for these exact endpoints at the stated diagnostic scale.


### Local-core qualification

A separate core-only multiplier fit gives raw broad-core normalized RMS **.06063 at 480** and **.18259 at 800**. At 480 the fitted raw derivative threshold 1.35315 closely matches the filtered gray threshold 1.35847; using that filtered-derived dual gives core RMS .06064. This is positive evidence of **approximate local balance inside much of the 480 broad patch**. It does not eliminate raw residuals in the surrounding gray field: all-gray RMS is .37060 with the same core-fitted multiplier. At 800 even the core-only fit remains above .1; using its filtered gray dual gives core RMS .37089.

Thus the primary nonstationarity category applies to the **complete constrained design and its substantial unresolved gray regions**, not every gray element. Some gray locations are locally balanced, and the 480 core could participate in a nearby stationary gray design. This audit cannot rule that out. The mixed-stationarity category is not issued as a certification of those patches because no common admissible physical multiplier makes the surrounding free design stationary. Local fits cannot be assigned independently to different parts of one volume-constrained problem. These results narrow the conclusion: “the whole endpoint is a genuine relaxed optimum” is unsupported; “every broad patch is itself necessarily nonstationary” is also unsupported.
