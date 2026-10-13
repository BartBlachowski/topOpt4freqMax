# Audit of `stop_criterion_paper.tex`

Reviewed 13 October 2026. Line numbers refer to the manuscript as reviewed. The manuscript has not been edited.

The English is generally grammatical and understandable. The principal weaknesses are repetition, conversational or adversarial wording, and claims that exceed the evidence. Several numerical and methodological inconsistencies also need correction before language polishing. The argument will be stronger if presented as an empirical study of this benchmark, with a promising common stopping rule, rather than as a proof that design-change criteria cannot work universally.

Scope: full LaTeX source, its tables, the local replay and recording scripts, the recorded CSV summaries, and selected configuration files. External papers and quotations were not independently verified; rendered page layout was not reviewed.

## 1. Corrections that affect the scientific interpretation

### 1.1 Limit the claim of universality

**Locations:** lines 40, 55, 139, 185, 223–227.

“No universal tolerance exists,” “norms of any kind,” and “the conjunction is what makes the rule universal” are substantially stronger than this experiment supports. The study tests selected tolerances, three implementations, one problem, and one initial design. It also calibrates R1 on the records used to assess it. Table A4 includes tighter versions of R1 that terminate on every record, so R1 is not literally the only successful candidate.

Suggested replacements:

- “None of the tested design-change tolerances provided a satisfactory compromise between delayed termination and premature stopping across the recorded runs.”
- “The combined criterion terminated all 23 recorded runs with a maximum observed frequency deficit of 0.91% relative to their recorded end states.”
- Replace the subsection title with “Limitations of the tested design-change criteria.”
- Describe R1 as **evaluated** or **calibrated and assessed** on this benchmark, rather than independently validated.

Define what counts as a satisfactory compromise. Without a stated accuracy and iteration-cost target, a claim that a tolerance “fails” is partly subjective.

### 1.2 Replace the stochastic explanation with an appropriately qualified observation

**Locations:** lines 40, 144–152, 223; Table A2 caption, line 292.

The manuscript repeatedly calls threshold crossings “random dips,” describes a “lottery,” and states that waiting times grow with mesh size. Recorded trajectories demonstrate irregular crossings, but the paper does not establish a stochastic model or a statistical waiting-time law. The reported stopping iterations are not monotonic in mesh size.

“Noise floor” also needs qualification. Table A2 contains many end-of-record norms below the native thresholds: for example, the quasi-static maximum change at 320×40 is 0.002. Thus the statement that all norms settle above the thresholds throughout all 23 records is not supported by the table. Boundary motion may explain persistent changes, but scalar histories alone do not establish where the moving elements are located.

Suggested wording:

> Persistent oscillations in the design-change measures can delay termination after the frequency and grayness have stabilized. Threshold crossings are irregular and sensitive to both mesh resolution and tolerance.

Provide element-level evidence if retaining the stronger causal attribution to boundary oscillations. Change the Table A2 caption: a median is neither a necessary threshold for termination nor a lower bound on individual observations. “These medians characterize typical design-change magnitudes near the reference stagnation point and near the end of each record” is accurate.

### 1.3 Specify the monitored objective unambiguously

**Locations:** lines 89, 177–185.

The definition “ω₁ or the eigenvalue λ₁ = ω₁²” is not interchangeable for a fixed relative tolerance. Squaring the monitored quantity changes its relative range. The replay explicitly assigns `T['obj'] = T['w1']` for Du–Olhoff (`scripts/replay.py`, line 27).

State:

> For the Du–Olhoff implementation, the monitored quantity is the fundamental angular frequency, fₖ = ω₁,ₖ. For the two static methods, fₖ is the recorded compliance-type objective.

Distinguish the optimizer's objective from the signal used for stopping. Also replace the claim that R1 directly monitors the benchmark frequency for all methods: the static methods monitor surrogate objectives. Its relationship to frequency accuracy is empirical.

### 1.4 Resolve iteration indexing before finalizing numerical results

**Locations:** lines 131, 187; Tables A1–A4; `scripts/replay.py`, lines 61–65 and 96–108.

`stagnation()` returns a zero-based array index. `evaluate()` exports that value directly as `kstag`, but exports stop iterations as `kk + 1`. It then divides the one-based stop iteration by the zero-based stagnation index. These conventions are inconsistent.

Keep the zero-based index for array access, but report `kstag = index + 1` if the paper uses one-based iteration numbers. Regenerate the tables, figures, and ratios after checking the pre-update/post-update alignment for each method. This is a small numerical correction, but it affects a principal performance measure.

Also explain why 300 recorded quasi-static iterations produce 299 replayable updates; the alignment code deliberately drops one row.

### 1.5 Define the window as ten updates and eleven states

**Locations:** lines 177–185 and the table headings.

The interval k−W ≤ j ≤ k contains W+1 observations. Equation (3) and `win_range()` agree, but “the last ten iterations” can be read as ten observations. R2, in contrast, checks exactly ten consecutive change values.

To preserve the implemented rule, write:

> The ranges are evaluated over W+1 consecutive states, spanning W updates; W = 10.

Changing the summation interval to ten observations would change the criterion and require recalculation.

### 1.6 Qualify the exactness of replay and disclose frequency sampling

**Locations:** lines 53, 89, 104, 116–129, 131.

The text says every design has an evaluated frequency, but Table 1 specifies every-fifth-iteration evaluation for several runs. `replay.py` fills missing frequencies with the most recent observation. Consequently, deficits and the frequency component of the reference stagnation point depend on sampled, forward-filled values.

Suggested wording:

> Candidate stopping decisions can be reconstructed from the recorded signals without rerunning the optimization, provided the stopping test does not alter the trajectory. Frequency deficits and reference stagnation points use the sampled frequency histories described in Table 1.

Explicitly describe forward filling. Evaluate frequency at each selected stop if claiming precise stop-state deficits. The frequency-only candidate also needs a sampling qualification. Explain when each signal becomes available relative to a design update before claiming that a replayed iteration is exactly the iteration at which an online implementation would terminate.

### 1.7 Correct the treatment of the two stages

**Locations:** lines 187, 209–211; Tables A1 and A3.

Line 209 says both Yuksel–Yilmaz stages terminate under R1. The experiment retains the native stage-1 rule and replays candidates only on stage 2. Applying R1 to stage 1 changes the trajectory and requires new runs; it cannot be supported by this replay.

Suggested replacement:

> R1 is applied to the quasi-static and Du–Olhoff methods and to stage 2 of the Yuksel–Yilmaz method. The native stage-1 handoff criterion is retained.

The Yuksel–Yilmaz iteration numbers and ratios include stage 1. They are not stage-2-only counts. For example, the 480×60 record has 586 stage-1 iterations: the reported R1 stop of 718 is approximately 132 stage-2 iterations, whereas the reported reference point of 884 corresponds to approximately 298. The stage-2 ratio is therefore about 0.44, rather than the reported total-run ratio of 0.81, before the indexing correction above.

Label total iteration counts explicitly, or present stage-2 counts and ratios separately. Including the unchanged first stage makes total-run ratios appear closer to one.

### 1.8 Correct the comparison horizon

**Locations:** lines 209 and 225; related wording in the abstract.

“After 300 or 600 more iterations” is incorrect. These are run or stage caps, not additional iterations following the selected stop. For example, the Du–Olhoff 800×100 R1 stop is 241 in a 300-iteration record.

Use:

> relative to the end of the extended record, with a 300-iteration cap for the quasi-static and Du–Olhoff methods and a 600-iteration cap for stage 2 of the Yuksel–Yilmaz method.

### 1.9 Do not infer discreteness or optimality from stable grayness

**Locations:** lines 185, 215, 217.

A small range of M_nd does not imply a small value of M_nd, an unchanged topology, or satisfaction of optimality conditions. The statement that residual gray material must be a sensitivity-filter band is not demonstrated by stable scalar grayness. Moreover, “M_nd changed by less than 0.01 after k_stag” follows from the definition of k_stag; it is not independent validation of that interpretation.

Suggested wording:

> The grayness condition limits changes in an aggregate measure of intermediate densities; it does not impose an absolute discreteness target or establish optimality.

Replace the defensive final sentence of line 217 with:

> For applications requiring smaller frequency deficits, tighter stopping tolerances or additional validation runs are needed.

## 2. Specific factual and numerical corrections

| Location | Finding and proposed correction |
|---|---|
| Line 71 | Introduces “two different ways” but describes three. Change to “three approaches.” |
| Line 89 | Says both static methods use move limit 0.2. The configuration selected by the recording runner uses 0.2 for the quasi-static method and 0.1 for Yuksel–Yilmaz. Correct the latter value; it also resolves the conflict with lines 161 and 189. |
| Line 89 | “Fixed inertial load” needs checking. The frozen quasi-static profile specifies F(x) = ω₀² M(x) Φ₀, with a fixed reference mode but design-dependent mass. Distinguish a fixed reference mode from a fixed load vector. |
| Line 104 | “Every method was run on all meshes” conflicts with the five Du–Olhoff records. State nine meshes for each static method and five for Du–Olhoff, giving 23 runs. |
| Line 144 | Du–Olhoff ranges omit two cases. Table A2 gives maximum-change medians of 0.011–0.073 and relative-L₂ medians of 0.0015–0.0064. |
| Line 146 | Changing 0.019 to 0.020 is an increase of 0.001, about 5.3%, not a “two-digit change.” |
| Line 148 | Groups R1 with rules producing hundred-iteration tails on the coarse meshes. Table A3 gives R1 stops of 66 and 57 on the two coarsest Du–Olhoff meshes. Remove R1 from that claim. Also distinguish the RMS threshold 3.5×10⁻³ from the relative-L₂ threshold 5×10⁻³. |
| Line 161 | “Pinned at ... 0.1 throughout stage 2” on all five finest meshes conflicts with the Table A2 end medians, including 0.016, 0.049, and 0.078. Use a qualified description of recurrent move-limit activation, supported by the traces. |
| Line 191 | The objective-only plateau example is 800×100, with 1.09% deficit, rather than 240×30 with 0.5%. |
| Line 191 | The grayness-only worst deficit is 2.71%; flip criteria reach 2.82% and 3.29%, rather than “up to 2.6%.” The 0.025 maximum-change rule also fails to terminate three records, so “tied with R1” is misleading. |
| Line 203 | Loosening the grayness tolerance gives the same worst deficit as the objective-only rule, but not identical stopping behavior: Table A4 gives different median ratios, 1.25 and 1.13. |
| Line 203 | “10 to 50% more iterations” is not a valid bound across runs. The recorded stricter W=20 rule increases the quasi-static 160×20 stop from 44 to 133, approximately 202%. State the actual summary statistic, or report per-run overheads. |
| Line 213 | Coarse Du–Olhoff native Euclidean-rule stops range from 93 to 121, not 93 to 122. The value 122 belongs to the maximum-change rule. |
| Line 215 | The claim that both static methods become more discrete in their late tails conflicts with Table A2: Yuksel–Yilmaz grayness increases between the reference point and the endpoint on every listed mesh. Specify method and reference interval. |
| Line 223 | “Three different rules” conflicts with line 91, where both static methods use the same maximum-change rule. Use “different native stopping criteria.” |
| Table A1, line 251 | The displayed RMS threshold is not equal to 0.05√(n_e/3200). That expression is the corresponding Euclidean-norm threshold. Write the full equivalent inequality or identify c = 0.05. |
| Tables A3–A4 | Headings such as M_nd < 0.005 suggest an absolute grayness constraint. Write “range(M_nd) < 0.005” and “relative range(f) < 0.001.” |

## 3. Language and presentation edits

The prose often sounds like an internal discussion of benchmark results rather than a journal article. Retain its directness, but remove rhetorical assertions and informal metaphors.

| Current wording | Suggested wording |
|---|---|
| “the rule that decides when a run stops decides the result” (40) | “the stopping criterion can substantially affect the reported comparison” |
| “Methods usually arrive with their own rules” (40) | “Implementations commonly use different stopping criteria” |
| “This study asked whether ... can serve three methods” (40) | “This study examines whether a single design-change tolerance can be used across three methods” |
| “is as good as possible” (47) | “is maximized” |
| “figures that end up in the benchmark table” (47) | “reported performance measures” |
| “The last two depend entirely on the rule” (47) | “The stopping criterion directly affects the iteration count and total runtime” |
| “the motion of the optimizer” (63, 91) | “changes in the design variables between iterations” |
| “the stop caught the design before this drift” (133) | “termination occurred before the subsequent frequency decrease” |
| “the objection ... is cosmetic” (150) | “the same criterion can be expressed as a mesh-normalized RMS threshold” |
| “fired on all 23 of 23 records” (187) | “triggered termination in all 23 recorded runs” |
| “left ... on the table” (225) | “produced a frequency deficit of ... relative to the recorded endpoint” |
| “the honest statement is ... not that the rule is wrong” (217) | “the remaining slow improvement limits the accuracy attainable with the selected tolerances” |

Additional presentation changes:

- Use “stopping criterion” consistently for the mathematical test; “stopping rule” is also acceptable, but avoid alternating with “gate,” “test,” and “condition” when the same object is meant.
- Use “recorded run” or “iteration history” consistently instead of alternating among “record,” “history,” and “trajectory” without a distinction.
- Define “native stopping criterion” once as the criterion used in the reference implementation.
- Choose a consistent spelling convention: the text combines “behavior/behaviour,” “analyzing/analysing”-type conventions, and “grayness.” Follow the target journal's preference.
- Shorten the Introduction's literature survey because Section 2 repeats it. Reserve detailed tolerances and quotations for Section 2.
- Split the long paragraph at line 148 into observations on premature stopping, the two-phase fine-mesh history, and the comparison of candidate rules.
- Define the signed deficit explicitly as 100[ω₁,end − ω₁(k*)]/ω₁,end. State that negative values indicate a frequency higher than the endpoint value.
- State whether each change norm uses design or physical densities and whether the relative norm denominator is the previous or current state. The metric script uses the previous state.
- Clarify whether filter radius is fixed in physical units or element widths. A radius fixed in element widths changes its physical size under mesh refinement, which matters when interpreting mesh effects.
- Add a repository URL and immutable version or archive identifier to the data-availability statement; a local folder path alone does not identify the public research artifact.

## 4. Suggested title and abstract

Suggested title:

**A common stagnation-based stopping criterion for comparing density-based topology optimization methods across mesh resolutions**

Suggested abstract, retaining the currently reported numerical result pending the indexing and sampling checks:

> Comparisons of topology optimization methods depend on the criteria used to terminate each run. This study examines whether a common stopping criterion can be used for three methods for fundamental-frequency maximization across multiple mesh resolutions. We analyze 23 extended iteration histories for a simply supported beam: nine meshes for each of two static surrogate methods and five meshes for a direct eigenvalue method. Candidate stopping criteria are evaluated retrospectively, while the native first-stage handoff criterion of the two-stage method is retained. The tested design-change criteria exhibit a trade-off between delayed termination caused by persistent density changes and premature termination on intermediate plateaus. We therefore assess a criterion that combines the relative range of a method-specific objective signal with the absolute range of a grayness measure over a window spanning ten updates. Using the same tolerances for all methods, this criterion terminates all 23 recorded runs, with a maximum observed fundamental-frequency deficit of 0.91% relative to the recorded end states. The criterion is dimensionless and contains no explicit mesh-size parameter. Its tolerances were calibrated on this benchmark; further studies are needed to assess their transferability to other problems.

The main revision priority is to align the claims, mathematical definitions, scripts, and tables. A language edit can then make the argument substantially shorter and clearer without overstating its scope.
