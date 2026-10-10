# Common stopping criterion for the three-method comparison

Prepared 10 October 2026. Scope: review of the current implementations, saved benchmark results, supplied plots, and relevant benchmarking literature. No solver, configuration, or plotting code was changed, and no new optimization runs were performed. Numerical tolerances proposed below require validation.

**Recommendation:** use the same maximum change in physical element density for every method, require it to remain small over several iterations, and combine it with feasibility and stability of the method's actual objective. Freeze one set of tolerances across all meshes after a documented accuracy study. Report grayness separately. A stopping rule measures whether an algorithm has settled; it cannot guarantee that the formulation produces a binary design.

The common rule can be effective and broadly applicable without producing the same iteration count, the same grayness, or visually similar frequency histories. Those are different outcomes. Choosing a tolerance to shorten a flat part of a plot is difficult to defend scientifically; choosing it to bound the change in the reported results is defensible.

**What the existing evidence establishes.** The saved campaign named `campaign_mac_convergence_relative_l2_change` does not use relative L2 for all three methods. Its manifest selects relative L2 with tolerance 0.001 for Olhoff, while Proposed and Yuksel retain the default maximum-change criterion with tolerance 0.04. Yuksel uses 0.04 for both stages. This campaign therefore does not yet test a common criterion. The generated notes also repeat parts of the frozen Olhoff configuration; the effective manifest and recorded stopping reason should take precedence when describing a particular run.

The following values are from the saved detailed benchmark tables. Frequencies use the common E1 evaluator, in rad/s. Grayness is the dimensionless measure G = mean[4 rho(1-rho)] and is not the fraction of gray elements.

| Olhoff campaign | Mesh | Outer iterations | E1 frequency | G |
|---|---:|---:|---:|---:|
| `campaign_mac_convergence_corrected` | 240×30 | 52 | 167.4044 | 0.11998 |
| `campaign_mac_convergence_corrected` | 800×100 | 37 | 153.1844 | 0.48702 |
| `campaign_mac_convergence_corrected_olhoff_08` | 240×30 | 75 | 167.3539 | 0.12148 |
| `campaign_mac_convergence_corrected_olhoff_08` | 800×100 | 235 | 165.3908 | 0.16823 |
| `campaign_mac_convergence_relative_l2_change` | 240×30 | 114 | 167.3368 | 0.12266 |
| `campaign_mac_convergence_relative_l2_change` | 800×100 | 256 | 165.4159 | 0.16261 |

At 240×30, the difference between the 52- and 114-iteration results is approximately 0.040% in E1 frequency. At 800×100, the 37-iteration result is approximately 7.39% below the 256-iteration result. Meanwhile, going from 235 to 256 iterations changes the fine-grid frequency by only about 0.015%, with appreciable grayness remaining. These comparisons support concern about an early stop on the fine mesh and show why additional iterations need not eliminate grayness. They do not establish the limiting design or prove that each tolerance change leaves the complete trajectory identical.

The supplied gray topology is labelled iteration 125; it should not be treated as the 256-iteration final design in this table. Without its complete history, that image alone cannot establish whether its gray regions are transient or persistent.

**Why neither norm solves the entire problem.** The original rule

\[
\|\Delta x\|_2 < 0.05\sqrt{n_e/3200}
\]

is exactly equivalent on equal-volume elements to

\[
\sqrt{\frac{1}{n_e}\sum_e(\Delta x_e)^2}
 < \frac{0.05}{\sqrt{3200}}
 \approx 8.84\times10^{-4}.
\]

Thus, square-root scaling is a standard RMS normalization. The number 3200 only expresses how its tolerance was calibrated. Rewriting it as an RMS criterion removes that reference mesh from the definition, although the tolerance still needs justification. On unequal elements, use the volume-weighted RMS, sqrt[sum(v_e Delta rho_e²)/sum(v_e)].

Relative L2 also removes the trivial dependence on element count when comparable fields are refined uniformly. It nevertheless averages local changes and depends on the current density distribution. At volume fraction 0.5, the RMS density increases from 0.5 for a uniform field to about 0.707 for a binary field; the same relative threshold therefore admits a larger absolute RMS step as the design becomes binary.

Local changes can be diluted as the mesh grows. For illustration, changing one element by 0.01 in an otherwise uniform density-0.5 field gives relative L2 approximately 2.36×10^-4 on 240×30 and 7.07×10^-5 on 800×100. Maximum change remains 0.01 in both cases. This illustrates averaging; it does not imply that a fixed physical feature always changes only one element under refinement.

The maximum norm is a sensible primary choice because it bounds every density update. Its definition applies to all meshes, but it can be dominated by a few oscillating boundary elements. It also cannot distinguish stationarity from a small step imposed by a move limit. A common mathematical definition does not by itself guarantee equal optimization accuracy.

**Suggested practical rule.** Let rho_e^k be the physical density used to assemble the material matrices after accepted update k. Use the same field convention, element set, update timing, and strict/non-strict inequality in all implementations. Normalize density to its admissible range if the methods use materially different ranges. Internal design-variable changes can still be recorded as diagnostics.

Define

\[
d_\infty^k=\max_e|\rho_e^k-\rho_e^{k-1}|,
\qquad
s_f^k=\frac{\max_{j\in W_k}f_j-\min_{j\in W_k}f_j}
 {\max_{j\in W_k}|f_j|},
\]

where W_k contains the latest ten accepted updates and their associated objective evaluations. For these nonzero objectives, the denominator is well-defined. For broader applications involving objectives near zero, define a fixed characteristic objective scale before running the benchmark.

Declare practical convergence only when all of the following hold:

1. d_inf is below epsilon_x for all ten updates in the window. Also inspect net density displacement over the window during validation: small steps can accumulate into significant drift.
2. The relative range s_f of the method's actual optimized objective is below epsilon_f over the window. Use the range, since equal endpoints can hide oscillation.
3. All final-problem constraints meet common normalized feasibility tolerances. For an upper volume bound, use max(0, V/V_max - 1); for an equality volume constraint, use abs(V/V_target - 1). Include density bounds and any additional constraints actually imposed by the method.
4. The method has reached its final stage and completed any prescribed continuation. Restart the observation window after a stage or formulation change. Iteration-budget exhaustion is reported separately from convergence.

A reasonable starting experiment is epsilon_x = 10^-3, epsilon_f = 10^-4, and ten updates of persistence. A normalized volume tolerance of 10^-4 is an initial feasibility target, subject to the numerical precision of the constraint solve. These are proposed trial settings, not literature-mandated or already validated universal constants. A single compound criterion still provides a common stopping policy: all methods satisfy the same tests and thresholds.

The objective test needs care here. Proposed optimizes a compliance-based surrogate; its plotted structural frequency is a performance output. Monitoring only that frequency could miss continued surrogate optimization. Apply the same relative-stability definition to each method's own objective, and verify the common E1 frequencies as additional benchmark outputs. This gives a comparison of methods solving their respective formulations; it is not a comparison of three optimizers solving an identical mathematical problem.

For Yuksel, terminal convergence refers to the final stage. Preserve and disclose the stage-transition policy separately: changing its Stage-1 stop can change the starting design for Stage 2. For Olhoff, distinguish accepted density displacement from the proposed increment before clipping; the current convergence metric is formed from `drho`. Keep objective evaluations and density snapshots aligned, since the Olhoff history records frequencies of the design analyzed before its update.

Persistence prevents an isolated dip below a threshold from terminating the run. It does not certify optimality. Record how often updates hit their move bounds and whether move limits recently contracted. Persistent small updates with active tiny move bounds can indicate controller-induced stagnation. Checking only the largest move box is insufficient when boxes are element-specific.

**How to choose and justify the tolerances.** First define acceptable errors in the reported quantities, independently of plot appearance. For example, a provisional requirement could be that tighter convergence changes the common structural frequency by less than 0.1%, with separate tolerances for grayness and topology displacement. The appropriate values depend on the precision of the paper's claims.

Use pilot meshes 160×20, 240×30, an intermediate mesh, and 800×100, for all methods. Compare epsilon_x = 10^-2, 3×10^-3, and 10^-3, with the same objective and feasibility tests. From candidate stopping points, continue with a tighter criterion and a sufficiently long observation period. Check both the optimized objective and common E1 frequencies, volume, grayness, cumulative density displacement, and topology. If even 10^-3 fails the accuracy target, tighten further for every method. Select the loosest common setting that meets the declared targets, freeze it, then confirm it on the remaining meshes.

Where diagnostic histories are already available, evaluate candidate rules retrospectively before commissioning expensive runs. Histories that end at an early native stop cannot validate what happens afterward. Exact restart state matters for adaptive move limits, OC/MMA state, and multi-stage methods. A continuation test gives empirical support, not a guarantee against later escape from a plateau.

Report a tolerance-sensitivity table alongside the main result. The reviewer should see whether the method ranking, frequency, and topology remain stable when the criterion is tightened. A long flat frequency tail may then be legitimate evidence that topology or the native objective converges more slowly than frequency. Use complete histories, optionally with a transient zoom and convergence-metric panels; do not choose convergence tolerances to make the tail disappear.

**Grayness and mesh refinement require a separate diagnosis.** Report both

\[
G=\frac{\sum_e v_e\,4\rho_e(1-\rho_e)}{\sum_e v_e},
\qquad
q_g=\frac{\sum_e v_e\,\mathbf{1}_{0.05<\rho_e<0.95}}{\sum_e v_e}.
\]

The first measures intermediate-density intensity; the second measures the volume fraction occupied by intermediate densities under explicitly stated cutoffs. Track their histories. Continuing reduction suggests unfinished evolution; stable broad gray regions suggest a property of the formulation, filtering, or algorithmic stagnation. Neither conclusion follows from one image.

Olhoff's production preset uses a fixed physical sensitivity-filter radius R = 0.06. Proposed and Yuksel use radii of 2 and 2.5 elements. On the 8×1 domain, these correspond to physical radii 0.0667 and 0.0833 at 240×30, but 0.02 and 0.025 at 800×100. Consequently, the physical regularization scales evolve differently under refinement. The material laws also differ: Olhoff uses Pedersen low-density stiffness with linear mass, while the other methods use their own interpolation models. A common stopping rule cannot remove these differences.

For a controlled mesh study, hold each prescribed physical length scale fixed and disclose any differences between methods. If the intended experiment instead compares the native implementations, retain their settings but make that scope explicit. Comparing at a common binary-quality target is a separate experiment: a grayness bound can be used as an acceptance requirement, but a method that never attains it must be reported as failing that target. Simply waiting indefinitely for G to become small is not a universal convergence policy. Adding projection, changing penalization, or thresholding a final design changes the experiment and requires fresh feasibility and frequency evaluation.

**Stronger optimality evidence.** A commonly scaled KKT residual, including stationarity, primal and dual feasibility, and complementarity, is preferable when comparing optimizers for the same mathematical problem. Rojas-Labanda and Stolpe explicitly recompute common KKT errors despite different native solver stopping definitions; they also discuss the OC maximum-change exception. Their numerical tolerances should not be transferred without matching their scaling and formulation. [Primary publication record](https://orbit.dtu.dk/en/publications/benchmarking-optimization-solvers-for-structural-topology-optimiz/).

For the present methods, such a certificate requires additional mathematical work. The compliance surrogate and eigenfrequency formulation have different stationarity equations. Sensitivity filtering need not supply the exact derivative of a common explicitly filtered objective. Repeated eigenvalues require treatment of the active eigenspace or an appropriate spectral formulation, rather than an arbitrary individual eigenvector derivative. KKT residuals of an inner MMA approximation alone do not certify the outer problem. These limitations support describing the recommended rule as practical convergence and supporting it with a tolerance study.

Kennedy and Fu discuss the difficulty of matching convergence tolerances and use fixed iteration budgets for their benchmark, while acknowledging that steps can have different computational costs. Their work supports adding a budget-based comparison, not assuming equal iterations are universally fair. Here, where Olhoff has substantial nested optimization cost, common wall-time budgets and quality-versus-time curves would be a useful secondary evaluation. [Author's publication listing](https://gkennedy.gatech.edu/publications.html).

Suggested manuscript wording, after validation: “All methods were terminated using a common practical convergence policy based on the maximum change in physical element density, relative stability of the optimized objective, and normalized constraint feasibility. The change and objective conditions were required to hold over a fixed observation window after the final algorithmic stage. The thresholds were fixed across meshes and methods, and a tighter-tolerance study verified the stability of the reported frequencies and designs. Grayness was reported separately. Runs exhausting their computational budget were identified separately.” Insert the validated numerical values; do not claim this study has already been completed.

**Local evidence reviewed:** `examples/bimodality/run_pinned_pinned_{olhoff,ourApproach,yuksel}_freq_history.m`; `examples/Performance/conference_bench/confbench_method_config.m`; `examples/Performance/benchmark_profile/study_base_config.m`; `analysis/Olhoff/+impl/architecture/olhoffSolve.m`; `tools/Matlab/topopt_history_record.m`; manifests and detailed tables of the three campaigns listed above. Literature sections reviewed: Kennedy and Fu (2021), introduction; Rojas-Labanda and Stolpe (2015), Section 3.3 and Table 1, using the local PDFs under `docs/`.
