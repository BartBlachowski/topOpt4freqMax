# Independent Scientific Audit — Phase B

## Executive assessment

The report establishes a strong but narrower result than its title and final
verdict imply. For the reconstruction actually tested, a full-box Eq. (25)
update from the uniform clamped–clamped design is catastrophically too long:
the exact LP solution and the converged MMA solution both predict improvement
but destroy the nonlinear structure when applied in full. This finding is
supported by the inner-budget sweep, the exact-LP comparison, the step profile,
and replication on two meshes (Report §§2.3–2.5, 3.3, and 7.5). The report also
convincingly establishes that none of its non-collapsed trajectories satisfies
its declared design-convergence test (Report §§7.1–7.5).

The report does **not** establish several stronger causal claims:

- that fail-closed acceptance is scientifically *necessary for validity*;
- that a move limit, rather than the combination of inner move bounds, outer
  bounds, and damping used in Regime B, is the uniquely missing historical
  procedure;
- that the move limit must contract;
- that failure to retain bimodality is primarily a step-length problem and not
  also a cluster/model/tracking problem;
- that optimizer non-convergence is the primary cause of the residual frequency
  gap;
- that the observed lag-2 behavior is an *exact* period-2 limit cycle; or
- that the tested reconstruction proves the Du–Olhoff procedure itself,
  including any undocumented implementation details, numerically nonviable.

The appropriate scientific conclusion is:

> The tested full-box, no-step-restriction reconstruction is numerically
> nonviable from the uniform clamped–clamped starting design under the reported
> meshes and reconstruction choices; the tested bounded variants remain
> non-converged and do not reproduce a persistent bimodal optimum.

This is a publishable negative result after substantial revision. The current
manuscript is not ready because its strongest mechanistic claims exceed its own
evidence.

## Strengths

1. **Unusually clear provenance.** The report distinguishes paper-explicit
   statements, observations, inferences, and reconstruction decisions, and
   candidly lists 18 unresolved assumptions (Report §§1.3, 3.2, and 12).

2. **A strong first-step forensic analysis.** The budget sweep shows that the
   collapse persists when MMA meets the report's declared inner stopping test:
   the paper-literal solve requires 312 iterations and gives
   \(\omega_1=0.0953\), while the exact LP step gives \(0.0638\) (Report
   §§2.3–2.4). This effectively excludes “30 inner iterations were merely too
   few” as the explanation for the first-step collapse.

3. **Useful separation of formulation and solver at the initial state.** For
   the simple-eigenvalue, inactive-J-constraint subproblem, the report writes the
   reduced LP explicitly, solves it independently, and shows that both the exact
   LP and converged MMA steps collapse the structure (Report §2.4). That is
   substantially stronger than diagnosing MMA from warnings alone.

4. **Transparent continuation reconstruction.** The report correctly says that
   continuation exists in the paper but its schedule does not, labels its
   schedules as decisions, and reports three schedules (Report §§3.1–3.2 and
   5.4).

5. **Good negative convergence evidence.** Large terminal RMS design changes,
   saturated component changes, lag diagnostics, increment correlations, and
   objective-tail statistics make it clear that the long runs have not met
   design convergence (Report §§7.2–7.5).

6. **Multiplicity is not inferred from the solver flag alone.** Independent
   eigengaps at several tolerances and a forced-\(N=2\) probe provide useful
   checks (Report §6).

7. **Adverse results are reported.** The report does not hide disconnected
   topologies, void-localized modes, non-converged inner solves, or continuation
   schedules that perform poorly (Report §§5.3–5.4, 7.5, and 8.1).

These strengths establish the behavior of the tested code. The bit-identical
tests in Report §4.2 establish equivalence to the current production code, not
fidelity to every undocumented detail of the 2007 implementation.

## Critical findings

### 1. The final verdict exceeds the identifiable reconstruction

The report itself documents low-confidence choices for the inner convergence
test and tolerance, MMA constants, beta bound, asymptote reinitialization,
multiplicity tolerance, continuation schedule, damping, and several other
details (Report §§1.3 and 12). It also observes that the beta bound is not
numerically inert (Report §2.6). Therefore the evidence can condemn the
**specified reconstruction**, but cannot exclude an undocumented
contemporaneous implementation choice in the original computation.

The wording in Report §11 partly narrows “paper-literal” to Eq. (25), full box,
and full update. With that explicit definition the first-step verdict is
justified. The headline and C6 nevertheless read as a verdict on the historical
Du–Olhoff procedure. That is too broad. “Mesh-independent” is likewise too
strong when only two meshes were tested; “consistent on both tested meshes” is
supported.

### 2. The contraction claim is not demonstrated and contains a logical error

Report §§6.3 and 9.1 state that “the smallest step the procedure can take” is
`move_lim × alpha = 0.1`. A move limit is an **upper bound**, not a lower bound.
The observed increments happen to saturate that upper bound, but the procedure
is capable in principle of returning smaller components. Consequently the
report has not measured a basin width or shown that \(0.1\) is larger than it.

The fixed Regime-B restriction keeps runs alive, while those runs continue to
saturate and fail to converge (Report §§2.5, 7.2–7.4). This supports the
hypothesis that the chosen fixed step controls are too permissive late in these
trajectories. It does not prove that contraction would yield convergence or
retain a cluster. A smaller fixed restriction, damping alone, a different
cluster model, or another unresolved reconstruction choice is not excluded.
The claimed “order 10” contraction factor in Report §11 has no presented basin
measurement.

For the requested A/B/C classification:

- a finite step restriction as a necessary ingredient of this tested
  reconstruction: **B — strongly supported but still a hypothesis**;
- the claim that it is the missing detail in the historical implementation:
  **B**, because Fig. 4 and standard practice are indirect evidence, not
  implementation evidence;
- the claim that the restriction must later contract: **C — speculative**;
- the composite executive-summary claim (“the missing ingredient is a move
  limit which then must contract”): **C**, because its decisive second half is
  untested.

The Regime-B comparison also changes `move_lim`, `outer_move`, and `alpha`
together (Report §§1.1 and 5.1). It therefore does not isolate which member or
combination of that bundle averts the first-step collapse.

### 3. “Fail-closed is necessary for validity” is a declared policy, not a
scientific theorem

The report shows that budget-30 inner solves do not meet the reconstruction's
successive-iterate test (Report §§2.3 and 5.2). It does not show that this test
is a sufficient or necessary certificate of subproblem optimality: no KKT
residual, dual residual, objective-gap criterion, or independent inner-solution
certificate is reported. The tolerance and its RMS scaling are themselves
reconstruction choices (Report §§1.3, C2, and 12/A3).

An inexact inner step can also be a legitimate component of an outer sequential
method if that is the intended algorithm. Because the paper does not specify
the inner stopping semantics, “did not meet our declared test” does not entail
“scientifically invalid.” Fail-closed is a defensible campaign rule that makes
the reconstruction auditable; its universal necessity is not established.

Likewise, V4 versus VR shows that changing from 30 to converged inner solves
changes the trajectory and terminal value (300.90 versus 328.55; Report §5.3).
It does not establish that the 9% difference is “because of invalidity” rather
than because truncation acts as a materially different effective update.

### 4. The multiplicity alternatives are not convincingly excluded

The forced-\(N=2\) experiment establishes only that, from one near-coalescent
160×20 design, the implemented \(N=2\) step with the existing step controls
does not retain the cluster (Report §6.3). It does not independently validate
the generalized-gradient implementation against derivatives, demonstrate
basis invariance, validate cluster construction, or test the effect of the
reported \(\bar\lambda\) substitution (Report §§1.3 and 10). The number of
computed modes is untested (Report §12/A6).

The tolerance diagnostic begins from the same single design and supplies eight
descendant steps. It does not show what a different tolerance would do to the
earlier trajectory. The report itself admits an effective sample size of one
(Report C4).

Individual eigenvector MAC is also fragile at or near a multiple eigenvalue:
the basis can rotate without destruction of the modal subspace. The reported
MAC11 values of 0.016 and 0.000 therefore do not, by themselves, prove that
physical modal identity is destroyed. No subspace MAC or equivalent
cluster-invariant diagnostic is presented.

The evidence supports: “multiplicity detection alone did not retain the one
tested near-cluster.” It does not support: “this is not a detection or
\(N=2\)-subproblem defect; it is a step-length failure” (Report §§6.3 and C4).
Generalized-gradient formulation, cluster construction, modal labeling,
eigensolver behavior, and tolerance-path effects remain live alternative
explanations.

### 5. The causal ranking of the residual frequency gap is unsupported

Report C5 correctly shows that the current bounded runs are non-converged. That
fact makes their terminal frequencies inadmissible as optima. It does not show
where a converged version of this reconstruction would end, and therefore
cannot quantify how much of the 456.4-to-343.04 gap is caused by
non-convergence.

Agreement of the *initial* frequency within 0.4% and M-orthonormality
(Report C5) validate one forward state and basic eigensolver normalization.
They do not exclude discrepancies in sensitivities, filtering, cluster
handling, interpolation derivatives, continuation details, or other
optimization-path choices listed in Report §12. “Remaining modelling
discrepancy — not supported” should be replaced by the narrower statement that
there is no large discrepancy in the reported initial scalar eigenfrequency.

### 6. “Exact period-2 limit cycle” overstates the diagnostics

For V4 at 160×20, the lag-2/lag-1 ratio is 0.1895, consecutive increment
correlation is -0.982, two-step correlation is +0.966, and the median
two-step infinity change is 0.136 (Report §§7.2–7.3). These are strong
period-2-like signatures, but they are not exact equality of alternating
states. The report's own classification threshold is 0.25, not zero.

Detailed increment correlations are presented only for V4 at 160×20, while the
executive summary applies -0.982 and “exact” to four runs (Report §§0 and 7.3).
The other three runs have strong lag-2 ratios (0.1901, 0.1420, and 0.0843;
Report §7.5), but exact two-state repetition is not shown. “Maximum-amplitude
random walk” in Report §7.4 is also conceptually inconsistent with an exact
periodic orbit.

The fair classification is “persistent, move-saturated, period-2-like
oscillation over the observed tail.” The evidence is sufficient to reject
design convergence, but not to assert a mathematically exact or asymptotic
limit cycle.

## Internal consistency findings

1. **Executive-summary frequencies are wrong for two of four oscillatory
   runs.** Report §0 says four runs cycle at approximately 301 (160×20) and 343
   (240×30). Those are V4 values. VR ends at 328.55 and 371.54 (Report §§5.2 and
   7.5).

2. **The -0.982 correlation is generalized beyond its evidence.** It is
   calculated for V4 at 160×20 only (Report §7.3), not all four classified
   oscillations.

3. **“End in a mechanism” is inconsistent with the classifier.** Mechanism is
   defined from the *minimum over the run* (Report §7.1). V5a is labeled
   `MECHANISM_COLLAPSE` because it dips to 12.88, but ends at 354.49
   (Report §5.4 and the classification table). Thus the statements in Report
   §§0 and 11 that seven runs “end in” a near-zero mechanism are false for at
   least V5a. The label also conflates transient entry with terminal collapse.

4. **“At every budget from 20 to 2000” is too literal.** Report §11 tests
   selected budgets in that interval, not every integer budget; the displayed
   sweep in §2.3 ends at 1000, with 2000 used in the matrix.

5. **“Every qualitative behavior transfers” is broader than the tested
   transfer.** Only progression-gate variants plus V0 were run at 240×30
   (Report §7.5). The statement is valid for those paired variants, not the full
   19-run design.

6. **The singular-warning wording needs qualification.** Report §5.3 says
   warnings are “exactly zero for V4, V5 and VR,” while the aggregate inner
   table records seven warnings for V4 at 240×30. If §5.3 refers only to the
   160×20 comparison, it should say so.

7. **“All progress the production solver has ever made” exceeds the campaign.**
   Report §4.3 establishes 0/300 convergence for the tested recorded
   configurations, not every historical production execution.

8. **Gate totals are not a scientific scale.** Runs with G7 `n/a` are still
   reported out of eight, and vacuous passes are acknowledged (Report §8.2).
   The “best score 6/8” rhetoric gives equal weight to heterogeneous,
   partly non-applicable criteria and should not carry inferential weight.

## Reconstruction methodology

### Continuation

The paper-explicit versus reconstructed distinction is handled well in
Report §§3.1–3.2. The report is justified in saying that the three reconstructed
schedules do not rescue the tested configurations. It is not justified in
saying the result is therefore “not an artefact of the invented schedule”
(Report §5.4): three fixed schedules do not span update triggers, stage
convergence, state transfer choices, or other unspecified continuation
semantics. The single-step probes at each \(p\) begin independently from the
uniform field and are not continuation trajectories (Report §3.3).

Supported conclusion: continuation is neither necessary nor sufficient **in
the tested reconstructions**. “Continuation is harmful” must retain the
qualification “as reconstructed.”

### Fail-closed semantics

The implementation is transparent and the gate behavior is tested
(Report §4). It is a valid conservative audit convention. The word “validity”
must be defined operationally as “meeting this campaign's declared inner
stopping rule,” not as mathematical or scientific validity. Alternative
inexact-inner interpretations remain possible because the paper is silent.

### LP diagnosis

For the initial \(N=1\) state, the LP diagnosis is the report's strongest
mechanistic result. The inactive J constraint, exact greedy solution, nonlinear
evaluation of both exact-LP and MMA steps, and close objective/direction
agreement separate the full-box formulation failure from MMA truncation
(Report §2.4).

Two generalizations should be removed:

- A general LP optimum need not put *every* design variable at a box bound; with
  a global volume constraint, degeneracy or a marginal variable can occur. The
  report observes 0.9997 at a bound for this LP, which is enough for this case.
- The initial \(N=1\), inactive-J result does not prove that every later,
  clustered Eq. (25) subproblem has the same greedy-box structure. The claim in
  Report §9.1 that every outer iteration necessarily produces a 0/1 design is
  not demonstrated.

Thus, “the collapse at the initial state is a property of the tested full-box
linearized formulation, not an MMA convergence artefact” is rigorous. A global
statement about every Eq. (25) state is not.

### Move-limit argument

The step profile directly shows that fractions of the destructive direction
improve the objective while the full step fails (Report §2.5). This strongly
supports overshoot at iteration 1. However, Regime B bundles three controls and
the later contraction claim is untested. The requested classification is
therefore **B for a finite bound**, **C for contraction**, and **C for the
combined claim as written**.

## Multiplicity conclusions and alternative interpretations

The campaign demonstrates that persistent bimodality is not reproduced. It
also demonstrates that forcing the implemented \(N=2\) model at the one
160×20 near-cluster is not sufficient to retain that cluster under the same
step controls.

It does not demonstrate that step length is the primary cause. The following
alternatives remain open on the report's own evidence:

- the \(N=1\) model is expressly acknowledged to be wrong at the near-cluster
  because the off-diagonal generalized gradient is large (Report §6.3);
- the \(N=2\) reconstruction is tested behaviorally but not independently
  verified;
- the cluster-mean substitution differs from the paper for \(N\ge2\)
  (Report §§1.3 and 10);
- the multiplicity tolerance and number of modes are reconstruction choices,
  with the latter untested (Report §12);
- individual MAC can report basis rotation rather than loss of a clustered
  invariant subspace;
- the one-state tolerance probe cannot exclude trajectory-level tolerance
  effects; and
- the only actual \(N=2\) state at 240×30 occurs on a badly degraded design, so
  it is not evidence about retention near the claimed optimum (Report §6.5).

The report should describe step length as one plausible and well-motivated
contributor, not as an excluded-alternatives diagnosis.

## Convergence analysis

The diagnostics are sufficient to conclude that none of the reported long
trajectories met the declared design stopping criterion within the allocated
budgets. Terminal RMS changes of \(4.2\times10^{-2}\) to
\(8.7\times10^{-2}\), versus \(10^{-6}\), make this conclusion insensitive to
reasonable adjustment of the tolerance (Report §§7.2 and 7.5). V5's tail
frequency CV of 0.22–0.26 supports “non-stationary over the observed tail.”

The criteria are fair for **rejecting convergence**. They are less secure for
assigning asymptotic dynamical classes:

- tail length 40 and the 0.25 lag-ratio threshold are declared but not
  justified;
- final-step design change is not by itself a robust convergence test, although
  the full tail evidence removes concern here;
- a minimum-frequency mechanism criterion labels a run forever even if it
  recovers, as V5a shows;
- “limit cycle” should be “period-2-like oscillation over the observed tail”;
  and
- “wanders without settling” should be “did not settle within the observed
  budget.”

Other asymptotic interpretations remain possible because finite trajectories
cannot prove non-convergence for unlimited iteration count. This does not
rescue any reported terminal value as a converged optimum.

## Assessment of principal conclusions

| Principal conclusion | Confidence | Evidence sufficient as written? | Additional experiments required? |
|---|---|---:|---:|
| The tested full-box first step collapses the CC structure | High | Yes | No |
| At the initial \(N=1\) state, the subproblem is an LP and exact LP/MMA both collapse | High | Yes | No |
| The initial collapse is not caused by a 30-iteration truncation | High | Yes | No |
| Every Eq. (25) iteration necessarily ends at a full 0/1 box vertex | Low | No | Yes, or narrow the claim |
| Tested continuation schedules do not avert collapse | High | Yes | No |
| Continuation is generally harmful | Low | No | Yes, or retain “as reconstructed” |
| Fail-closed enforces the campaign's declared acceptance semantics | High | Yes | No |
| Fail-closed is necessary for mathematical/scientific validity | Low | No | Yes, or redefine “validity” operationally |
| A finite step restriction is the missing ingredient in this reconstruction | Medium | Partly | Yes to isolate it; otherwise call it a hypothesis |
| A move restriction existed in the original implementation | Medium | No direct evidence | Documentary evidence would be required |
| The move restriction must contract | Low | No | Yes, if retained |
| Persistent bimodality was not reproduced | High | Yes | No |
| Detection alone does not retain the one tested near-cluster | High | Yes | No |
| Step length is the primary reason bimodality is not retained | Low | No | Yes, if retained |
| None of the long runs met declared design convergence | High | Yes | No |
| Four runs show strong period-2-like tail behavior | High | Yes | No |
| Four runs are exact asymptotic period-2 limit cycles | Low | No | Yes, or weaken |
| Optimizer non-convergence primarily explains the residual frequency gap | Low | No | Yes, or remove the causal ranking |
| The tested literal full-box reconstruction remains nonviable | High | Yes | No |
| The historical Du–Olhoff procedure is numerically nonviable | Medium–Low | No | Undocumented procedural evidence would be required |
| Prior campaign terminal values must not be called converged optima | High, if configurations are as reported | Yes | No |

## Unsupported or overstated conclusions

The following text should be withdrawn or rewritten before publication:

- Report §0: “the missing procedural ingredient is therefore ... a rule that
  bounds and then contracts”;
- Report §§6.3 and C4: “the smallest step the procedure can take” and the
  asserted basin comparison;
- Report C4: categorical exclusion of detection and \(N=2\)-subproblem defects;
- Report C5: the ranked claim that optimizer non-convergence primarily causes
  the published-frequency gap;
- Report §7.3 and the executive/final summaries: “exact period-2 limit cycle”;
- Report §9.1: “every element moves to one bound” and “every outer iteration
  produces a 0/1 design” as general LP properties;
- Report §5.4: schedule sensitivity proves the conclusion is not an artefact of
  the invented continuation;
- Report C2: “necessary for validity” without an operational definition;
- Report §11: “mesh-independent” and “at every budget from 20 to 2000”; and
- Report C7: the policy statement that the entire analysis directory must not
  be used for any reviewer-facing purpose. The evidence supports forbidding
  convergence and optimality claims from these runs, not every possible use.

## Reviewer #2 criticisms

### Critical

1. The final verdict conflates failure of a particular reconstruction with
   failure of the incompletely specified historical implementation.
2. The principal proposed missing mechanism—late contraction—is untested, and
   the argument incorrectly treats a move cap as a minimum possible step.
3. The report declares fail-closed necessary for validity without validating
   its inner stopping test as a certificate of subproblem solution.
4. The categorical step-length explanation for multiplicity failure is based
   on one near-cluster and does not exclude the alternatives the report is asked
   to evaluate.
5. The causal ranking of the residual frequency gap is not identifiable from
   non-converged runs.

### Major

1. “Exact period-2” is inconsistent with nonzero lag-2 distance and is not
   demonstrated for all four classified runs.
2. Regime B changes three step-control quantities together, preventing causal
   attribution to a move limit alone.
3. The exact-LP result is generalized from the initial simple-eigenvalue,
   inactive-J case to every outer iteration.
4. Individual-mode MAC is used near multiplicity without a cluster-invariant
   subspace diagnostic.
5. The forced-\(N=2\) implementation is not independently verified; behavioral
   failure does not prove correctness.
6. Three continuation schedules do not eliminate continuation-schedule
   ambiguity, especially because stage triggers and state-transfer semantics are
   unspecified.
7. The convergence classifier conflates transient mechanism entry with terminal
   mechanism collapse; V5a exposes the problem.
8. Gate G5 contains the subjective alternative “strong evidence bimodality is
   unreachable,” while gate totals mix failures, vacuous passes, and `n/a`.
9. Spectral validity is primarily a final-state gate; calling whole trajectories
   spectrally valid requires care.
10. Matching the initial frequency does not exclude optimization-path modeling
    discrepancies.
11. Claims about all historical production progress exceed the executions
    documented by the campaign.
12. The report must reconcile the seven singular warnings for V4 at 240×30
    with the “exactly zero for V4, V5 and VR” statement.

### Minor

1. Correct the executive-summary cycle frequencies to include the VR values.
2. Qualify every use of “continuation harmful” with “under the tested
   reconstructed schedules.”
3. Replace “at every budget from 20 to 2000” with “at every tested budget.”
4. Replace “mesh-independent” with “consistent on both tested meshes.”
5. Distinguish a near-cluster above `mult_tol` at 160×20 from an actual
   solver-detected \(N=2\) state.
6. Explain “structural component” versus raw connected-component counts so the
   topology prose can be reconciled with the aggregate table.
7. Define how prediction ratios are handled when predicted improvement is near
   zero.
8. State explicitly whether “rejected outer steps” for ungated runs means
   “would have been rejected”; otherwise the table conflicts with unconditional
   acceptance.
9. Do not use a summed gate score as if it were a calibrated measure of
   scientific quality.
10. Preserve the useful OBS/INF/DEC convention, but relabel conclusion previews
    if the statement that conclusions occur only in §§9–11 is retained.

## Required corrections before publication

1. Narrow the title, executive summary, C6, and final verdict to the tested
   full-box reconstruction and the tested meshes, budgets, and schedules.
2. Recast finite step restriction as strongly supported and contraction as an
   untested hypothesis. Delete the “smallest step” and basin-width claims.
3. Define inner “validity” operationally, or remove the necessity claim. Do not
   equate one successive-iterate tolerance with mathematical validity without
   evidence.
4. Replace the primary step-length diagnosis of multiplicity loss with a
   bounded conclusion: the implemented \(N=2\) model plus existing step controls
   failed at the one tested near-cluster, while listed alternatives remain open.
5. Replace “exact limit cycle” with “strong period-2-like tail oscillation” and
   distinguish observed finite-budget behavior from asymptotic convergence.
6. Remove the causal ranking of the residual frequency gap, or explicitly state
   that the data only show current terminal values are non-converged and cannot
   be interpreted as optima.
7. Repair the internal numerical and counting inconsistencies identified above,
   especially the VR frequencies, V5a mechanism wording, selected-budget
   language, and singular-warning count.
8. Present acceptance gates individually. Do not use the aggregate 6/8 score as
   substantive evidence.
9. Limit C7 withdrawals to claims actually invalidated by the campaign:
   convergence, optimality, and admissibility of terminal frequency comparisons.

No new optimizer or globalization scheme is required to make these corrections.
Additional experiments are necessary only if the authors choose to retain the
causal claims about contraction, unique move-limit causation, generalized
multiplicity behavior, exact asymptotic cycling, or the source of the residual
frequency gap.

## Overall recommendation

**Major revision required**

The negative first-step result and the rejection of convergence are valuable
and well evidenced. The manuscript's current headline mechanism, multiplicity
diagnosis, and historical verdict are stronger than the data permit, but they
can be corrected by narrowing claims and separating observations from
untested causal hypotheses.
