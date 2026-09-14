# Iteration accounting specification

Status: design draft. Counts below describe the frozen practical profiles used by the
completed performance campaign; that campaign and its artifacts remain untouched.

Phase-1C delta: references are generated on separate trajectories and their work is
reported but not charged to measurement endpoints. Primary endpoints are evaluated at
q=98%, 99%, and 99.5% under the all-E1/E2/E3 gate.

## Common rule

An iteration is counted only at a method-native optimization-loop level. Offline common
evaluation, topology analysis, image production, checkpoint serialization, and timing
replays do not add optimization iterations. Failed attempted solver calls are recorded as
work attempts but are not silently converted into successful state updates.

The acceptance scan uses the return-equivalent state convention in
`ACCEPTANCE_GATE_SPEC.md`. Phase 2 must checkpoint-test off-by-one alignment.

## Proposed

Profile: `proposed_practical_move02_tol001`, source
`analysis/ourApproach/Matlab/topopt_freq.m`, optimizer OC.

One Proposed iteration, `N_OC += 1`, is one execution of the single optimization loop:

1. assemble current mass and stiffness;
2. form the frozen-reference semi-harmonic load and solve equilibrium;
3. compute/filter sensitivities;
4. perform one OC update and form the new physical field;
5. compute the native change diagnostic.

The current `info.iterations` / loop counter is the count. The post-loop eigensolve is
not an OC iteration. It is separately timed as native finalization. Proposed has no
stage count and no method-specific multiplicity requirement.

Under the frozen `semi_harmonic_baseline='solid'` and
`reference_refresh_interval=0`, the reference eigenpair is computed once before the loop
and never refreshed: a Proposed OC iteration contains **no eigensolve**. In the frozen
800x100 evidence the Olhoff eigensolve alone was about 75% of its outer-update cost. This
qualitative difference must accompany adjacent count comparisons.

Main report: the paired `k_enter = N_OC_to_enter` and `k_cert = N_OC_to_cert`. Supplement:
native stop iteration, if it occurred, and both differences from `k_native` (which may be
negative).

## Yuksel

Profile: `yuksel_practical_move01_tol001`, source
`analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`, optimizer OC.

### Stage 1

One `N_stage1` iteration is one compliance-stage loop pass: current physical field,
stiffness solve under the point load, compliance sensitivity, and one OC update.

Stage 1 is not merely the first part of a homogeneous objective. The implementation does
carry it forward: `x=xPhys` makes the Stage-1 filtered physical field the Stage-2 design
variable, and Stage-1 displacement becomes the initial mode estimate. The design state is
continuous, while raw/physical identification has a one-time re-filtering shift. Stage 1
remains separately reported because it optimizes point-load compliance under a distinct
update regime and timing structure—not because its design is discarded.

### Stage 2

One `N_stage2` iteration is one inertial-stage loop pass: build `M(x)`, form the moving
inertial load from the current mode estimate, solve `K(x)U=F`, update the normalized mode
estimate, form the partial compliance sensitivity, and perform one OC update.

Only Stage-2 states are eligible for external acceptance.

The completed campaign capped Stage 1 at 1000 updates at 640x80, 720x90, and 800x100.
The Phase-2 reference/measurement budget of up to 2000 permits a different Stage-1 handoff
state at those meshes. New `N_stage1`, Stage-2 trajectory, and chronological total are not
comparable to the frozen campaign values there; the old rows are budget evidence only.

### Headline and retained decomposition

\[
N_{\rm total}=N_{\rm stage1}+N_{\rm stage2}.
\]

The sum is justified as chronological method-level update work required before a Stage-2
result can be returned. It does **not** imply equal work, equal objective, continuous
density evolution, or equal seconds per iteration. Every table must retain all three
counts. For each endpoint `e in {enter,cert}` reached at Stage-2 local iteration `s_e`,
report `N_stage1`, `N_stage2_to_e=s_e`, and `N_total_to_e=N_stage1+s_e`.

## Olhoff / Du–Olhoff selected S1 profile

Profile: `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1`, source mirror
`analysis/olhoff_stabilization_audit/olhoffOptStabilized.m`, LP subproblem
`Matlab/reproduction2007/algo/innerLoopLP.m`.

One successful outer iteration consists of:

1. one current-design generalized eigensolve and multiplicity detection;
2. generalized-gradient construction and filtering;
3. one LP subproblem call;
4. one accepted density update.

`N_outer` counts successful outer updates. Also retain:

- `N_outer_attempted`;
- `N_LP_calls`;
- `N_LP_success` and `N_LP_failure`;
- the failed attempt index and exit flag, if any.

### What the implementation calls an inner iteration

There are two different code paths:

- `innerLoop.m` is an MMA reconstruction. Its `nInner` is the number of MMA
  sub-iterations taken while the outer state is fixed.
- The selected production profile uses `innerLoopLP.m`. That routine makes exactly one
  `linprog(...,'dual-simplex-highs')` call and hard-codes `st.nInner=1`, irrespective of
  the number of HiGHS/simplex iterations. Therefore `sum(nInner)` is just the number of
  successful LP calls and is normally identical to `N_outer`.

These are not the same as:

- a successful LP solver call;
- a MATLAB `linprog` output iteration;
- a HiGHS pivot/simplex iteration.

The targeted 640x80 failure replay demonstrates the distinction: the failed outer attempt
made one `linprog` call, returned exit flag 0, and reported 38 LP solver iterations, while
the frozen production `nInner` convention would still be one call.

### Reporting recommendation

Main table: paired `k_enter` and `k_cert` in successful outer iterations. Add LP calls to
each endpoint only if a failure/retry policy ever makes them differ; otherwise state in a
footnote that one LP call occurs per successful outer iteration.

Supplement: total/mean/median/max MATLAB-reported LP solver iterations to `k_enter` and
`k_cert`, LP exit flags, eigensolve/gradient/LP time totals, and attempt counts. The code-level
`N_inner_total` may be archived for compatibility but should not occupy a main-table
column because it is redundant and invites a false work-equivalence claim.

## Failed and post-certification work

- A failed attempt before `k_cert` prevents certification and is reported separately from
  the last successful update count.
- A failure after a completed persistence window does not change the earlier minimum-work
  result; report `PASS_WITH_LATER_FAILURE` and the later attempt.
- Observer continuation after `k_cert` is validation work, not minimum optimization work.
- Offline evaluator calls never enter any method's iteration count.

## Headline hierarchy and persistence arithmetic

The main scientific statement uses `k_enter`: it locates when the trajectory first
entered the accepted persistent regime. `k_cert` appears adjacent with equal visual
prominence and states the conservative evidence-acquisition cost. For any finite baseline
pair, `k_cert=k_enter+99`; this constant does not change absolute count differences among
successful methods, but it changes ratios and power-law slopes. Count scaling must
therefore fit both endpoints independently, exclude censored cells, and interpret
`k_cert` as certification scaling rather than a second estimate of maturation scaling.

The same arithmetic applies to the global Yuksel total because Stage 1 is fully charged
before either eligible Stage-2 endpoint. It does not imply that the 99 added updates have
equal time cost across methods; `T_cert-T_enter` is measured, not assumed.

## Reference work and method-gate floors

Archive `N_reference` separately for each method/stage and never add it to measurement
`k_enter/k_cert`. Report `k_gate` beside every q-level endpoint: Proposed has no added
method condition, Yuksel's is Stage-2 entry, and Olhoff's is the first simultaneous policy
stage-2 / N=2 / gap pass. Frozen Olhoff 160x20 evidence has policy trigger 245, which is a
visible gate floor rather than hidden maturation work.

## Quantities that look comparable but are not

`N_OC`, Yuksel Stage-1 updates, Yuksel Stage-2 updates, Olhoff outer updates, LP solver
iterations, and seconds are different units of algorithmic work. The primary comparison
is deliberately method-level iteration effort with transparent decompositions—not a
claim that one count unit costs or accomplishes the same operation across methods.

## Phase 2H Olhoff route amendment

Olhoff-LP is principal and reports outer updates, LP calls, and genuine solver iterations
only when returned by the backend. Olhoff-MMA is secondary and reports outer updates,
total/mean/median/p95 nested MMA iterations, cap hits, and converged-inner fraction. With
selector `both`, the routes are separate rows and trajectories; their counts are never
summed, averaged, substituted, or relabelled as a single Olhoff result.
